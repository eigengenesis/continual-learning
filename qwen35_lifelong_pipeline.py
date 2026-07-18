#!/usr/bin/env python3
"""Unified Stage-1 geometric continual-learning pipeline for Qwen3.5.

The pipeline is intentionally stage-aware.  It reuses the proven adapter
teacher, tomography, and no-proxy projection machinery while adding frozen
manifests, a persistent profile registry, on-policy context acquisition,
label-free geometric consolidation, selective release, and exact stage resume.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import alien_ladder_cl_audit as al
import qwen35_five_skill_cl_audit as five
import qwen_continual_proof as qp
import qwen_tomography as qt
from standalone_latent_lora_qwen import (
    LatentLoRAConfig,
    attach_latent_lora,
    choose_dtype,
    load_causal_lm,
    load_tokenizer,
)


SCHEMA_VERSION = 1
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
STAGES = (
    "00_bootstrap",
    "10_acquire_a",
    "11_consolidate_a",
    "20_acquire_b",
    "21_consolidate_b",
    "30_zero_shot_composition",
    "40_acquire_composition",
    "41_consolidate_composition",
    "50_policy_change",
    "51_selective_release_update",
    "60_final_audit",
    "70_finalize",
)
UPDATE_TASKS_BY_STAGE = {
    "10_acquire_a": ("skill_a",),
    "11_consolidate_a": ("skill_a",),
    "20_acquire_b": ("skill_b_v1",),
    "21_consolidate_b": ("skill_b_v1",),
    "40_acquire_composition": ("composition_direct_v1",),
    "41_consolidate_composition": ("composition_direct_v1",),
    "50_policy_change": ("skill_b_v2_changed",),
    "51_selective_release_update": ("skill_b_v2_changed",),
}
FROZEN_HPARAMETERS = (
    "dtype",
    "teacher_min_steps",
    "teacher_block_steps",
    "teacher_max_steps",
    "teacher_lr",
    "teacher_rank",
    "teacher_alpha",
    "teacher_gate_init",
    "target_suffixes",
    "min_layers",
    "consolidation_steps",
    "consolidation_lr",
    "composition_reward_steps",
    "composition_hybrid_steps",
    "composition_lr",
    "rollouts_per_prompt",
    "context_kl_weight",
    "old_kl_weight",
    "old_hidden_weight",
    "new_kl_weight",
    "new_hidden_weight",
    "projection_strength",
    "batch_size",
    "micro_batch_size",
    "eval_batch_size",
    "max_seq_len",
    "grad_clip",
    "gradient_checkpointing",
    "wikitext_eval_samples",
    "eval_interval",
    "log_interval",
    "checkpoint_interval",
)

COLORS = tuple(f"C{idx}" for idx in range(8))
ACTIONS = tuple(f"A{idx}" for idx in range(8))
GLYPH_ALPHABET = tuple(chr(0x03B1 + idx) for idx in range(25)) + tuple(
    chr(0x0410 + idx) for idx in range(16)
)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def directory_checksums(root: Path, *, exclude: Sequence[str] = ()) -> Dict[str, str]:
    excluded = set(exclude)
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in excluded
    }


def verify_directory_checksums(root: Path, expected: Mapping[str, str]) -> None:
    failures = []
    for relative, digest in expected.items():
        path = root / relative
        if not path.exists() or sha256_file(path) != digest:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"resume artifact checksum mismatch: {failures[:5]}")


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def frozen_hyperparameters(args: argparse.Namespace) -> Dict[str, Any]:
    return {name: getattr(args, name) for name in FROZEN_HPARAMETERS}


def validate_frozen_hyperparameters(args: argparse.Namespace, manifest: Mapping[str, Any]) -> None:
    frozen = manifest.get("hyperparameters", {})
    mismatches = []
    for name in FROZEN_HPARAMETERS:
        if name not in frozen:
            mismatches.append(f"{name}=missing")
            continue
        runtime = getattr(args, name)
        if runtime != frozen[name]:
            mismatches.append(f"{name}:manifest={frozen[name]!r},runtime={runtime!r}")
    if mismatches:
        raise ValueError("runtime arguments differ from frozen manifest: " + "; ".join(mismatches))


def assert_finite_model(model, label: str) -> None:
    for name, param in model.named_parameters():
        if not torch.isfinite(param.detach()).all():
            raise FloatingPointError(f"{label}: non-finite parameter {name}")


def assert_finite_gradients(params: Sequence[torch.nn.Parameter], label: str) -> None:
    for idx, param in enumerate(params):
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise FloatingPointError(f"{label}: non-finite gradient at trainable parameter {idx}")


def glyphs(count: int) -> List[str]:
    if count <= len(GLYPH_ALPHABET):
        return list(GLYPH_ALPHABET[:count])
    return [f"G{idx:02d}" for idx in range(count)]


def glyph_color_prompt(glyph: str, variant: int, *, held_out: bool = False) -> str:
    prefixes = (
        (
            "HELD-OUT SYMBOL QUERY. Respond with the learned color code only.",
            "NOVEL LEXICON PHRASING. Give only the canonical color label.",
        )
        if held_out
        else (
            "AURORA LEXICON. Return only the canonical color code.",
            "ALIEN GLYPH TABLE. Output only the color code.",
            "LEXICON LOOKUP. No explanation. Color code only.",
        )
    )
    prefix = prefixes[variant % len(prefixes)]
    return f"{prefix}\nGlyph: {glyph}\nColor code:"


def color_action_prompt(color: str, variant: int, *, held_out: bool = False) -> str:
    prefixes = (
        (
            "HELD-OUT CONTROL QUERY. Respond with the learned action code only.",
            "NOVEL POLICY PHRASING. Give only the canonical action label.",
        )
        if held_out
        else (
            "AURORA POLICY. Return only the canonical action code.",
            "CONTROL POLICY LOOKUP. Output only the action code.",
            "ACTION TABLE. No explanation. Action code only.",
        )
    )
    prefix = prefixes[variant % len(prefixes)]
    return f"{prefix}\nColor code: {color}\nAction code:"


def direct_prompt(glyph: str, variant: int) -> str:
    prefix = (
        "AURORA COMPOSITION. Compose the learned glyph lexicon and action policy. Final action code only.",
        "COMPOSE THE TWO TABLES. Glyph to color to action. Output only the final action code.",
        "TWO STEP LOOKUP. Use the learned lexicon and policy. Final action code only.",
    )[variant % 3]
    return f"{prefix}\nGlyph: {glyph}\nAction code:"


def prompted_prompt(glyph: str, variant: int) -> str:
    return (
        "AURORA COMPOSITION CHECK. Map the glyph to color and then color to action.\n"
        f"Glyph: {glyph}\n"
        "Return exactly: COLOR=<color>; ACTION=<action>\nAnswer:"
    )


def compressed_prompt(glyph: str, variant: int) -> str:
    return (
        "AURORA COMPACT ROUTE. Compose both learned tables.\n"
        f"Glyph: {glyph}\n"
        "Return exactly: C=<color>; A=<action>\nAnswer:"
    )


def row(row_id: str, prompt: str, target: str, **metadata: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "row_id": row_id,
        "prompt": prompt.strip(),
        "target": target.strip(),
        "source": prompt.strip(),
        "raw_target": target.strip(),
    }
    payload.update(metadata)
    return payload


def task_payload(
    name: str,
    display_name: str,
    train_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
    *,
    max_new_tokens: int = 12,
    exact_gate: float = 0.0,
) -> Dict[str, Any]:
    return {
        "recipe": {
            "name": name,
            "display_name": display_name,
            "dataset_id": "synthetic:gpsp_lifelong_stage1",
            "config": None,
            "train_splits": ["manifest"],
            "eval_splits": ["manifest"],
            "source_keys": ["source"],
            "target_keys": ["target"],
            "max_new_tokens": max_new_tokens,
            "token_gate": exact_gate,
            "exact_gate": exact_gate,
        },
        "source": "synthetic:gpsp_lifelong_stage1",
        "train": list(train_rows),
        "eval": list(eval_rows),
    }


def build_stage1_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    glyph_count = int(args.glyph_count)
    glyph_values = glyphs(glyph_count)
    train_glyph_count = int(args.composition_train_glyphs)
    if not 1 <= train_glyph_count < glyph_count:
        raise ValueError("--composition-train-glyphs must be between 1 and glyph_count-1")

    color_for = {glyph: COLORS[idx % len(COLORS)] for idx, glyph in enumerate(glyph_values)}
    action_v1 = {color: ACTIONS[idx] for idx, color in enumerate(COLORS)}
    ranked_colors = sorted(
        COLORS,
        key=lambda color: sha256_bytes(f"{args.seed}:policy-change:{color}".encode("utf-8")),
    )
    changed_count = max(1, int(round(len(COLORS) * float(args.changed_fraction))))
    changed_colors = ranked_colors[:changed_count]
    action_v2 = dict(action_v1)
    for color in changed_colors:
        old_idx = ACTIONS.index(action_v1[color])
        action_v2[color] = ACTIONS[(old_idx + 3) % len(ACTIONS)]

    a_train = [
        row(
            f"a:train:{idx}",
            glyph_color_prompt(glyph_values[idx % glyph_count], idx),
            color_for[glyph_values[idx % glyph_count]],
            glyph=glyph_values[idx % glyph_count],
        )
        for idx in range(int(args.train_samples))
    ]
    a_eval = [
        row(
            f"a:eval:{idx}",
            glyph_color_prompt(glyph_values[(idx * 7 + 5) % glyph_count], idx + 1000, held_out=True),
            color_for[glyph_values[(idx * 7 + 5) % glyph_count]],
            glyph=glyph_values[(idx * 7 + 5) % glyph_count],
        )
        for idx in range(int(args.eval_samples))
    ]
    b_train = [
        row(
            f"b:train:{idx}",
            color_action_prompt(COLORS[idx % len(COLORS)], idx),
            action_v1[COLORS[idx % len(COLORS)]],
            color=COLORS[idx % len(COLORS)],
            policy_version="v1",
        )
        for idx in range(int(args.train_samples))
    ]
    b_eval = [
        row(
            f"b:eval:{idx}",
            color_action_prompt(COLORS[(idx * 5 + 1) % len(COLORS)], idx + 1000, held_out=True),
            action_v1[COLORS[(idx * 5 + 1) % len(COLORS)]],
            color=COLORS[(idx * 5 + 1) % len(COLORS)],
            policy_version="v1",
        )
        for idx in range(int(args.eval_samples))
    ]

    train_glyphs = glyph_values[:train_glyph_count]
    eval_glyphs = glyph_values[train_glyph_count:]

    def composition_rows(version: str, selected: Sequence[str], split: str) -> Dict[str, List[Dict[str, Any]]]:
        mapping = action_v1 if version == "v1" else action_v2
        direct_rows: List[Dict[str, Any]] = []
        prompted_rows: List[Dict[str, Any]] = []
        compressed_rows: List[Dict[str, Any]] = []
        count = int(args.composition_eval_samples if split == "eval" else args.composition_train_samples)
        for idx in range(count):
            glyph = selected[idx % len(selected)]
            color = color_for[glyph]
            action = mapping[color]
            prefix = f"composition:{version}:{split}:{idx}"
            direct_rows.append(row(prefix + ":direct", direct_prompt(glyph, idx), action, glyph=glyph, color=color, action=action, policy_version=version))
            prompted_rows.append(row(prefix + ":prompted", prompted_prompt(glyph, idx), f"COLOR={color}; ACTION={action}", glyph=glyph, color=color, action=action, policy_version=version))
            compressed_rows.append(row(prefix + ":compressed", compressed_prompt(glyph, idx), f"C={color}; A={action}", glyph=glyph, color=color, action=action, policy_version=version))
        return {"direct": direct_rows, "prompted": prompted_rows, "compressed": compressed_rows}

    comp_v1_train = composition_rows("v1", train_glyphs, "train")
    comp_v1_eval = composition_rows("v1", eval_glyphs, "eval")
    comp_v2_train = composition_rows("v2", train_glyphs, "train")
    comp_v2_eval = composition_rows("v2", eval_glyphs, "eval")

    changed_train: List[Dict[str, Any]] = []
    changed_eval: List[Dict[str, Any]] = []
    stable_eval: List[Dict[str, Any]] = []
    for idx in range(int(args.train_samples)):
        color = changed_colors[idx % len(changed_colors)]
        changed_train.append(row(f"b:v2:train:{idx}", color_action_prompt(color, idx + 3000), action_v2[color], color=color, old_action=action_v1[color], policy_version="v2"))
    for idx in range(int(args.eval_samples)):
        color = COLORS[(idx * 5 + 1) % len(COLORS)]
        target_row = row(f"b:v2:eval:{idx}", color_action_prompt(color, idx + 4000, held_out=True), action_v2[color], color=color, old_action=action_v1[color], policy_version="v2")
        (changed_eval if color in changed_colors else stable_eval).append(target_row)

    tasks = {
        "skill_a": task_payload("glyph_color", "Glyph to Color", a_train, a_eval, exact_gate=0.70),
        "skill_b_v1": task_payload("color_action", "Color to Action V1", b_train, b_eval, exact_gate=0.70),
        "composition_direct_v1": task_payload("glyph_action_direct", "Direct Composition V1", comp_v1_train["direct"], comp_v1_eval["direct"], max_new_tokens=10),
        "composition_prompted_v1": task_payload("glyph_action_prompted", "Prompted Composition V1", [], comp_v1_eval["prompted"], max_new_tokens=24),
        "composition_compressed_v1": task_payload("glyph_action_compressed", "Compressed Composition V1", [], comp_v1_eval["compressed"], max_new_tokens=24),
        "skill_b_v2_changed": task_payload("color_action_v2_changed", "Changed Color Policy", changed_train, changed_eval, exact_gate=0.75),
        "skill_b_v2_stable": task_payload("color_action_v2_stable", "Stable Color Policy", [], stable_eval),
        "composition_direct_v2": task_payload("glyph_action_direct_v2", "Direct Composition V2", comp_v2_train["direct"], comp_v2_eval["direct"], max_new_tokens=10),
        "composition_prompted_v2": task_payload("glyph_action_prompted_v2", "Prompted Composition V2", [], comp_v2_eval["prompted"], max_new_tokens=24),
        "composition_compressed_v2": task_payload("glyph_action_compressed_v2", "Compressed Composition V2", [], comp_v2_eval["compressed"], max_new_tokens=24),
    }
    history_path = Path(args.history_manifest).expanduser() if args.history_manifest else None
    history_sha256 = sha256_file(history_path) if history_path is not None and history_path.exists() else None
    task_hashes = {
        key: {
            "recipe_sha256": sha256_bytes(canonical_json(payload["recipe"]).encode("utf-8")),
            "train_sha256": sha256_bytes(canonical_json(payload.get("train", [])).encode("utf-8")),
            "eval_sha256": sha256_bytes(canonical_json(payload.get("eval", [])).encode("utf-8")),
        }
        for key, payload in tasks.items()
    }
    profile_dependencies = {
        "base_language": [],
        "history:*": [],
        "skill_a": [],
        **{f"skill_b:v1:{color}": [] for color in COLORS},
        **{f"composition:v1:{color}": [f"skill_b:v1:{color}"] for color in COLORS},
        **{f"skill_b:v2:{color}": [] for color in changed_colors},
        **{f"composition:v2:{color}": [f"skill_b:v2:{color}"] for color in changed_colors},
    }

    manifest: Dict[str, Any] = {
        "schema": "qwen35_lifelong_stage1",
        "version": SCHEMA_VERSION,
        "model_id": args.model_id,
        "model_identity": checkpoint_identity(args.model_id),
        "history_manifest": args.history_manifest or None,
        "history_manifest_sha256": history_sha256,
        "seed": int(args.seed),
        "stage_order": list(STAGES),
        "mappings": {
            "glyph_to_color": color_for,
            "color_to_action_v1": action_v1,
            "color_to_action_v2": action_v2,
            "changed_colors": changed_colors,
        },
        "splits": {
            "composition_train_glyphs": train_glyphs,
            "composition_eval_glyphs": eval_glyphs,
        },
        "tasks": tasks,
        "task_hashes": task_hashes,
        "profile_dependencies": profile_dependencies,
        "hyperparameters": {
            **frozen_hyperparameters(args),
            "context_mixture": {"full": 0.4, "compressed": 0.3, "none": 0.3},
        },
        "verifiers": {
            "action": {
                "correct": 1.0,
                "valid_wrong": 0.1,
                "invalid": -0.25,
                "trajectory_retention": "all_on_policy",
                "separate_gold_target_field_in_trajectory_artifact": False,
                "composition_teacher_context": "model_generated_only",
                "policy_change_teacher_context": "explicit_update_notice",
            },
        },
        "gates": {
            "primitive_exact": 0.70,
            "protected_max_drop": 0.10,
            "positive_context_rate": 0.50,
            "composition_direct_exact": 0.50,
            "composition_direct_gain": 0.30,
            "composition_prompted_exact": 0.70,
            "changed_policy_exact": 0.75,
            "stale_rate": 0.10,
        },
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    return manifest


def write_manifest(manifest: Mapping[str, Any], path: Path) -> Path:
    payload = dict(manifest)
    payload["manifest_sha256"] = manifest_digest(payload)
    validate_manifest_structure(payload)
    atomic_json(path, payload)
    return path


def validate_manifest_structure(payload: Mapping[str, Any]) -> None:
    if list(payload.get("stage_order", [])) != list(STAGES):
        raise ValueError("manifest stage order does not match the Stage-1 lifecycle")
    tasks = payload.get("tasks", {})
    hashes = payload.get("task_hashes", {})
    if set(tasks) != set(hashes):
        raise ValueError("manifest task hashes do not cover every task")
    seen_row_ids: set[str] = set()
    for task_key, task in tasks.items():
        expected = hashes[task_key]
        for split in ("train", "eval"):
            rows = list(task.get(split, []))
            actual_hash = sha256_bytes(canonical_json(rows).encode("utf-8"))
            if expected.get(f"{split}_sha256") != actual_hash:
                raise ValueError(f"manifest task hash mismatch: task={task_key} split={split}")
            for item in rows:
                row_id = str(item.get("row_id", ""))
                if not row_id or row_id in seen_row_ids:
                    raise ValueError(f"manifest row IDs must be non-empty and globally unique: {row_id!r}")
                seen_row_ids.add(row_id)
        recipe_hash = sha256_bytes(canonical_json(task.get("recipe", {})).encode("utf-8"))
        if expected.get("recipe_sha256") != recipe_hash:
            raise ValueError(f"manifest recipe hash mismatch: task={task_key}")
    train_glyphs = set(payload.get("splits", {}).get("composition_train_glyphs", []))
    eval_glyphs = set(payload.get("splits", {}).get("composition_eval_glyphs", []))
    if not train_glyphs or not eval_glyphs or train_glyphs & eval_glyphs:
        raise ValueError("composition train/eval glyph sets must be non-empty and disjoint")
    mappings = payload.get("mappings", {})
    for color in mappings.get("changed_colors", []):
        if mappings["color_to_action_v1"].get(color) == mappings["color_to_action_v2"].get(color):
            raise ValueError(f"changed policy target did not change for color={color}")


def load_manifest(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "qwen35_lifelong_stage1" or int(payload.get("version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported lifelong manifest: {path}")
    expected = payload.get("manifest_sha256")
    actual = manifest_digest(payload)
    if expected != actual:
        raise ValueError(f"manifest hash mismatch: expected={expected} actual={actual}")
    validate_manifest_structure(payload)
    return payload


@dataclass
class PipelineState:
    schema_version: int
    manifest_sha256: str
    source_model_id: str
    current_checkpoint: str
    current_stage: str = ""
    current_step: int = 0
    status: str = "initialized"
    completed_stages: List[str] = field(default_factory=list)
    stage_records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    best_metrics: Dict[str, float] = field(default_factory=dict)
    profile_registry_path: str = "profiles/registry.json"
    resume_artifact: str = ""
    resume_checksum: str = ""
    resume_components: Dict[str, Optional[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PipelineState":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})


@dataclass
class ProfileEntry:
    profile_id: str
    task_name: str
    scope: Dict[str, str]
    dependencies: List[str]
    selected_layers: List[int]
    checkpoint_sha256: str
    creation_stage: str
    status: str
    tensor_path: str
    layer_metadata: Dict[str, Dict[str, Any]]


class ProfileRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "registry.json"
        self.entries: Dict[str, ProfileEntry] = {}
        self._transaction_depth = 0
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = {key: ProfileEntry(**value) for key, value in payload.get("profiles", {}).items()}

    def flush(self) -> None:
        atomic_json(
            self.path,
            {
                "version": 1,
                "profiles": {key: asdict(value) for key, value in sorted(self.entries.items())},
            },
        )

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {key: asdict(value) for key, value in self.entries.items()}

    def _flush_if_ready(self) -> None:
        if self._transaction_depth == 0:
            self.flush()

    @contextmanager
    def transaction(self):
        snapshot = self.snapshot()
        self._transaction_depth += 1
        try:
            yield self
        except Exception:
            self._transaction_depth -= 1
            self.restore(snapshot)
            raise
        else:
            self._transaction_depth -= 1
            self._flush_if_ready()

    def restore(self, snapshot: Mapping[str, Mapping[str, Any]]) -> None:
        restored = {key: ProfileEntry(**dict(value)) for key, value in snapshot.items()}
        retained_paths = {entry.tensor_path for entry in restored.values()}
        stale_paths = {
            entry.tensor_path
            for key, entry in self.entries.items()
            if key not in restored or restored[key].tensor_path != entry.tensor_path
        }
        self.entries = restored
        self.flush()
        for relative_path in stale_paths - retained_paths:
            path = self.root / relative_path
            if path.exists():
                path.unlink()

    def add_metadata(self, entry: ProfileEntry) -> None:
        if entry.profile_id in self.entries:
            raise ValueError(f"duplicate profile_id={entry.profile_id}")
        self.entries[entry.profile_id] = entry
        self._flush_if_ready()

    def add_profile(
        self,
        profile_id: str,
        profile: qt.TaskProfile,
        *,
        scope: Mapping[str, str],
        dependencies: Sequence[str],
        selected_layers: Sequence[int],
        checkpoint_sha256: str,
        creation_stage: str,
    ) -> ProfileEntry:
        try:
            from safetensors.torch import save_file
        except Exception as exc:  # pragma: no cover - GPU environment dependency
            raise RuntimeError("safetensors>=0.8.0 is required to persist profiles") from exc
        tensors: Dict[str, torch.Tensor] = {}
        layer_metadata: Dict[str, Dict[str, Any]] = {}
        for layer_idx, layer in sorted(profile.layer_profiles.items()):
            if not torch.isfinite(layer.activation_basis).all() or not torch.isfinite(layer.gradient_basis).all():
                raise FloatingPointError(f"profile_id={profile_id} contains non-finite basis tensors at layer={layer_idx}")
            activation_key = f"layer_{layer_idx}_activation"
            gradient_key = f"layer_{layer_idx}_gradient"
            tensors[activation_key] = layer.activation_basis.detach().to(device="cpu", dtype=torch.float32).contiguous()
            tensors[gradient_key] = layer.gradient_basis.detach().to(device="cpu", dtype=torch.float32).contiguous()
            layer_metadata[str(layer_idx)] = {
                "layer_name": layer.layer_name,
                "activation_key": activation_key,
                "gradient_key": gradient_key,
                "effective_act_rank": layer.effective_act_rank,
                "effective_grad_rank": layer.effective_grad_rank,
                "explained_variance": layer.explained_variance,
            }
        tensor_path = self.root / f"{profile_id}.safetensors"
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tensor_path.with_name(tensor_path.name + ".tmp")
        save_file(tensors, str(tmp))
        os.replace(tmp, tensor_path)
        entry = ProfileEntry(
            profile_id=profile_id,
            task_name=profile.task_name,
            scope=dict(scope),
            dependencies=list(dependencies),
            selected_layers=[int(value) for value in selected_layers],
            checkpoint_sha256=checkpoint_sha256,
            creation_stage=creation_stage,
            status="protected",
            tensor_path=str(tensor_path.relative_to(self.root)),
            layer_metadata=layer_metadata,
        )
        self.add_metadata(entry)
        return entry

    def load_profile(self, profile_id: str) -> qt.TaskProfile:
        try:
            from safetensors.torch import load_file
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("safetensors>=0.8.0 is required to load profiles") from exc
        entry = self.entries[profile_id]
        tensors = load_file(str(self.root / entry.tensor_path), device="cpu")
        layers: Dict[int, qt.LayerProfile] = {}
        for raw_idx, meta in entry.layer_metadata.items():
            idx = int(raw_idx)
            layers[idx] = qt.LayerProfile(
                layer_index=idx,
                layer_name=meta["layer_name"],
                activation_basis=tensors[meta["activation_key"]],
                gradient_basis=tensors[meta["gradient_key"]],
                effective_act_rank=int(meta["effective_act_rank"]),
                effective_grad_rank=int(meta["effective_grad_rank"]),
                explained_variance=float(meta["explained_variance"]),
            )
        return qt.TaskProfile(task_name=entry.task_name, stage_label=entry.creation_stage, layer_profiles=layers)

    def protected_ids(
        self,
        *,
        exclude_ids: Sequence[str] = (),
        exclude_creation_stage: str = "",
    ) -> List[str]:
        excluded = set(exclude_ids)
        return [
            key
            for key, entry in self.entries.items()
            if entry.status == "protected"
            and key not in excluded
            and (not exclude_creation_stage or entry.creation_stage != exclude_creation_stage)
        ]

    def protected_profiles(
        self,
        *,
        exclude_ids: Sequence[str] = (),
        exclude_creation_stage: str = "",
    ) -> List[qt.TaskProfile]:
        return [
            self.load_profile(key)
            for key in self.protected_ids(
                exclude_ids=exclude_ids,
                exclude_creation_stage=exclude_creation_stage,
            )
        ]

    def dependency_closure(self, initial: Sequence[str]) -> List[str]:
        released = set(initial)
        changed = True
        while changed:
            changed = False
            for key, entry in self.entries.items():
                if entry.status not in {"protected", "released"} or key in released:
                    continue
                if any(dependency in released for dependency in entry.dependencies):
                    released.add(key)
                    changed = True
        return sorted(released)

    def release_closure(self, initial: Sequence[str]) -> List[str]:
        released = self.dependency_closure(initial)
        for key in released:
            if key in self.entries and self.entries[key].status == "protected":
                self.entries[key].status = "released"
        self._flush_if_ready()
        return released

    def retire(self, profile_ids: Sequence[str]) -> None:
        for key in profile_ids:
            if key in self.entries:
                self.entries[key].status = "retired"
        self._flush_if_ready()


class DataAccessAudit:
    def __init__(self, path: Path, manifest: Optional[Mapping[str, Any]] = None) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.allowed_update_rows: Dict[str, Dict[str, set[str]]] = {}
        if manifest is not None:
            for stage, task_keys in UPDATE_TASKS_BY_STAGE.items():
                self.allowed_update_rows[stage] = {
                    task_key: {
                        str(item["row_id"])
                        for item in task_rows(manifest, task_key, "train")
                    }
                    for task_key in task_keys
                }

    def violation(self, payload: Mapping[str, Any]) -> str:
        if payload.get("purpose") != "update" or payload.get("split") != "train":
            return ""
        stage = str(payload.get("stage", ""))
        task = str(payload.get("task", ""))
        row_ids = {str(value) for value in payload.get("row_ids", [])}
        if not row_ids:
            return f"empty update row IDs at stage={stage} task={task}"
        if self.allowed_update_rows:
            allowed_tasks = self.allowed_update_rows.get(stage, {})
            if task not in allowed_tasks:
                return f"prohibited update task at stage={stage}: task={task} allowed={sorted(allowed_tasks)}"
            unknown = sorted(row_ids - allowed_tasks[task])
            if unknown:
                return f"prohibited update rows at stage={stage} task={task}: {unknown[:5]}"
        return ""

    def validate_existing(self) -> List[str]:
        if not self.path.exists():
            return []
        violations = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            message = self.violation(json.loads(line))
            if message:
                violations.append(f"line={line_number} {message}")
        return violations

    def rewind_stage(self, stage: str, step: int) -> None:
        if not self.path.exists():
            return
        retained = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item_step = item.get("step")
            discard = (
                item.get("stage") == stage
                and item.get("purpose") == "update"
                and item_step is not None
                and int(item_step) > int(step)
            )
            if not discard:
                retained.append(item)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for item in retained:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        os.replace(tmp, self.path)

    def log(
        self,
        *,
        stage: str,
        task: str,
        split: str,
        row_ids: Sequence[str],
        purpose: str,
        allowed_train_tasks: Sequence[str] = (),
        step: Optional[int] = None,
    ) -> None:
        if purpose == "update" and split == "train" and task not in set(allowed_train_tasks):
            raise RuntimeError(f"prohibited training access: stage={stage} task={task} allowed={list(allowed_train_tasks)}")
        payload = {
            "time": time.time(),
            "stage": stage,
            "task": task,
            "split": split,
            "purpose": purpose,
            "row_ids": list(row_ids),
            "step": int(step) if step is not None else None,
        }
        message = self.violation(payload)
        if message:
            raise RuntimeError(message)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def payload_to_task(payload: Mapping[str, Any]) -> al.TaskData:
    return five.task_from_payload(dict(payload))


def task_rows(manifest: Mapping[str, Any], key: str, split: str) -> List[Dict[str, Any]]:
    return list(manifest["tasks"][key].get(split, []))


def parse_code(text: str, valid: Sequence[str]) -> str:
    normalized = al.truncate_completion(str(text)).upper().replace("`", " ")
    for code in valid:
        if code in normalized.replace("=", " ").replace(";", " ").replace(",", " ").split():
            return code
    stripped = normalized.strip().strip(".;,: ")
    return stripped if stripped in valid else ""


def action_reward(completion: str, expected: str) -> Tuple[float, Dict[str, Any]]:
    prediction = parse_code(completion, ACTIONS)
    if prediction == expected:
        return 1.0, {"prediction": prediction, "correct": True, "valid": True}
    if prediction:
        return 0.1, {"prediction": prediction, "correct": False, "valid": True}
    return -0.25, {"prediction": "", "correct": False, "valid": False}


def make_on_policy_trajectory(
    *,
    stage: str,
    step: int,
    context_mode: str,
    rollout_index: int,
    task_key: str,
    item: Mapping[str, Any],
    prompt: str,
    teacher_prompt: str,
    completion: str,
    reward: float,
    verifier: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "stage": str(stage),
        "step": int(step),
        "context_mode": str(context_mode),
        "rollout_index": int(rollout_index),
        "row_id": str(item["row_id"]),
        "task_key": str(task_key),
        "prompt": str(prompt),
        "teacher_prompt": str(teacher_prompt),
        "completion": str(completion),
        "reward": float(reward),
        "prediction": str(verifier.get("prediction", "")),
        "correct": bool(verifier.get("correct", False)),
        "valid": bool(verifier.get("valid", False)),
        "color": str(item.get("color", "")),
        "glyph": str(item.get("glyph", "")),
    }


def deterministic_modes(total: int, mixture: Mapping[str, float], salt: str) -> List[str]:
    names = tuple(mixture)
    raw = {name: max(0.0, float(mixture[name])) * int(total) for name in names}
    counts = {name: int(math.floor(raw[name])) for name in names}
    for name in sorted(names, key=lambda item: (raw[item] - counts[item], item), reverse=True)[: int(total) - sum(counts.values())]:
        counts[name] += 1
    values = [(name, idx) for name in names for idx in range(counts[name])]
    values.sort(key=lambda item: sha256_bytes(f"{salt}:{item[0]}:{item[1]}".encode("utf-8")))
    return [name for name, _ in values]


def checkpoint_identity(path_or_id: str) -> str:
    path = Path(path_or_id).expanduser()
    if path.exists():
        if path.is_file():
            return sha256_file(path)
        candidates = sorted(path.glob("*.safetensors"))
        candidates.extend(
            candidate
            for candidate in (path / "model.safetensors.index.json", path / "config.json")
            if candidate.exists()
        )
        if not candidates:
            raise FileNotFoundError(f"checkpoint has no safetensors/config artifacts: {path}")
        digest = hashlib.sha256()
        for candidate in candidates:
            digest.update(candidate.name.encode("utf-8"))
            digest.update(sha256_file(candidate).encode("utf-8"))
        return digest.hexdigest()
    return sha256_bytes(path_or_id.encode("utf-8"))


def trainable_state(model) -> Dict[str, torch.Tensor]:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    return {
        name: value.detach().to("cpu").clone()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def restore_named_state(model, state: Mapping[str, torch.Tensor]) -> None:
    current = model.state_dict()
    missing = [name for name in state if name not in current]
    if missing:
        raise KeyError(f"state contains unknown model keys: {missing[:5]}")
    for name, tensor in state.items():
        current[name].copy_(tensor.to(device=current[name].device, dtype=current[name].dtype))


def optimizer_to_cpu(state: Mapping[str, Any]) -> Dict[str, Any]:
    def move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().to("cpu")
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    return move(dict(state))


def save_pretrained_atomic(model, tokenizer, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tokenizer.save_pretrained(tmp)
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(tmp, destination)
    return destination


def prepare_completion_policy_loss(
    model,
    tokenizer,
    prompts: Sequence[str],
    completions: Sequence[str],
    advantages: torch.Tensor,
    device: str,
    max_seq_len: int,
) -> torch.Tensor:
    eos = tokenizer.eos_token or ""
    batch = qp._prepare_supervised_batch(
        tokenizer,
        list(prompts),
        [str(value).strip() + eos for value in completions],
        device,
        max_seq_len,
    )
    outputs = model(**batch, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    safe_labels = labels.masked_fill(~mask, 0)
    logps = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    sequence_logp = (logps * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
    return -(advantages.to(device=device, dtype=sequence_logp.dtype).detach() * sequence_logp).mean()


def task_metric(metrics: Mapping[str, Any], task_name: str, suffix: str = "exact") -> float:
    value = metrics.get(f"{task_name}_{suffix}", float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class StageStore:
    def __init__(self, output_dir: Path, manifest: Mapping[str, Any], source_model_id: str, resume: str) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = output_dir / "pipeline_state.json"
        self.resume_root = output_dir / "resume"
        self.checkpoint_root = output_dir / "checkpoints"
        self.manifest_hash = str(manifest["manifest_sha256"])
        if self.state_path.exists():
            if resume == "never":
                raise FileExistsError(f"state already exists: {self.state_path}; use --resume auto|required")
            self.state = PipelineState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))
            if self.state.manifest_sha256 != self.manifest_hash:
                raise ValueError("existing pipeline state belongs to a different manifest")
        else:
            if resume == "required":
                raise FileNotFoundError(f"--resume required but no state exists at {self.state_path}")
            self.state = PipelineState(
                schema_version=SCHEMA_VERSION,
                manifest_sha256=self.manifest_hash,
                source_model_id=source_model_id,
                current_checkpoint=source_model_id,
            )
            self.flush()

    def flush(self) -> None:
        atomic_json(self.state_path, asdict(self.state))

    def completed(self, stage: str) -> bool:
        return stage in self.state.completed_stages

    def begin(self, stage: str) -> None:
        self.state.current_stage = stage
        self.state.current_step = 0
        self.state.status = "running"
        self.flush()

    def update_step(
        self,
        stage: str,
        step: int,
        resume_artifact: str = "",
        *,
        resume_checksum: str = "",
        resume_components: Optional[Mapping[str, Optional[str]]] = None,
    ) -> None:
        self.state.current_stage = stage
        self.state.current_step = int(step)
        self.state.resume_artifact = resume_artifact
        self.state.resume_checksum = resume_checksum
        self.state.resume_components = dict(resume_components or {})
        self.flush()
        atomic_json(
            self.resume_root / "latest.json",
            {
                "stage": stage,
                "step": int(step),
                "artifact": resume_artifact,
                "artifact_sha256": resume_checksum,
                "components": self.state.resume_components,
                "manifest_sha256": self.manifest_hash,
            },
        )

    def stage_resume_dir(self, stage: str) -> Path:
        return self.resume_root / stage

    def resume_pointer(self, stage: str) -> Optional[Dict[str, Any]]:
        latest = self.resume_root / "latest.json"
        if not latest.exists():
            return None
        payload = json.loads(latest.read_text(encoding="utf-8"))
        if payload.get("manifest_sha256") != self.manifest_hash:
            raise ValueError("latest resume pointer belongs to a different manifest")
        if payload.get("stage") != stage:
            return None
        return payload

    def prune_resume_versions(self, stage: str, keep: Path) -> None:
        root = self.stage_resume_dir(stage)
        if not root.exists():
            return
        for path in root.iterdir():
            if path == keep:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    def clear_resume(self, stage: str) -> None:
        path = self.stage_resume_dir(stage)
        if path.exists():
            shutil.rmtree(path)
        self.state.resume_artifact = ""
        self.state.resume_checksum = ""
        self.state.resume_components = {}
        self.state.current_step = 0
        self.flush()
        latest = self.resume_root / "latest.json"
        if latest.exists():
            latest.unlink()

    def save_adapter_resume(
        self,
        stage: str,
        *,
        step: int,
        model,
        optimizer,
        best_state: Mapping[str, torch.Tensor],
        best_score: Any,
        extra: Mapping[str, Any],
    ) -> Path:
        path = self.stage_resume_dir(stage) / f"adapter_step_{int(step):06d}.pt"
        atomic_torch_save(
            path,
            {
                "step": int(step),
                "trainable_state": trainable_state(model),
                "optimizer": optimizer_to_cpu(optimizer.state_dict()),
                "best_state": {key: value.detach().to("cpu") for key, value in best_state.items()},
                "best_score": best_score,
                "rng": capture_rng_state(),
                "extra": dict(extra),
            },
        )
        checksum = sha256_file(path)
        self.update_step(
            stage,
            step,
            str(path),
            resume_checksum=checksum,
            resume_components={"model": str(path), "optimizer": str(path), "rng": str(path), "scheduler": None},
        )
        self.prune_resume_versions(stage, path)
        return path

    def load_adapter_resume(self, stage: str) -> Optional[Dict[str, Any]]:
        pointer = self.resume_pointer(stage)
        if not pointer:
            return None
        path = Path(str(pointer.get("artifact", "")))
        if path.suffix != ".pt":
            return None
        if not path.exists():
            raise FileNotFoundError(f"latest adapter resume artifact is missing: {path}")
        expected = self.store_resume_checksum(stage, path)
        if expected and sha256_file(path) != expected:
            raise RuntimeError(f"adapter resume checksum mismatch: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def store_resume_checksum(self, stage: str, path: Path) -> str:
        payload = self.resume_pointer(stage)
        if not payload:
            return ""
        if Path(str(payload.get("artifact", ""))) == path:
            return str(payload.get("artifact_sha256", ""))
        return ""

    def save_full_resume(self, stage: str, step: int, model, tokenizer, optimizer, extra: Mapping[str, Any]) -> Path:
        destination = self.stage_resume_dir(stage) / f"full_step_{int(step):06d}"
        tmp = destination.with_name(destination.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        (tmp / "model").mkdir(parents=True)
        model.save_pretrained(tmp / "model", safe_serialization=True)
        tokenizer.save_pretrained(tmp / "model")
        torch.save(optimizer_to_cpu(optimizer.state_dict()), tmp / "optimizer.pt")
        torch.save(
            {"step": int(step), "rng": capture_rng_state(), "scheduler": None, "extra": dict(extra)},
            tmp / "resume.pt",
        )
        atomic_json(tmp / "checksums.json", directory_checksums(tmp, exclude=("checksums.json",)))
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(tmp, destination)
        checksum = sha256_file(destination / "checksums.json")
        self.update_step(
            stage,
            step,
            str(destination),
            resume_checksum=checksum,
            resume_components={
                "model": str(destination / "model"),
                "optimizer": str(destination / "optimizer.pt"),
                "rng": str(destination / "resume.pt"),
                "scheduler": None,
            },
        )
        self.prune_resume_versions(stage, destination)
        return destination

    def load_full_resume(self, stage: str) -> Optional[Dict[str, Any]]:
        pointer = self.resume_pointer(stage)
        if not pointer:
            return None
        path = Path(str(pointer.get("artifact", "")))
        if path.suffix == ".pt":
            return None
        if not (path / "resume.pt").exists():
            raise FileNotFoundError(f"latest full resume artifact is missing: {path}")
        checksums_path = path / "checksums.json"
        if not checksums_path.exists():
            raise FileNotFoundError(f"resume artifact is missing checksums: {checksums_path}")
        expected_manifest_hash = self.store_resume_checksum(stage, path)
        if expected_manifest_hash and sha256_file(checksums_path) != expected_manifest_hash:
            raise RuntimeError(f"resume checksum manifest mismatch: {checksums_path}")
        verify_directory_checksums(path, json.loads(checksums_path.read_text(encoding="utf-8")))
        payload = torch.load(path / "resume.pt", map_location="cpu", weights_only=False)
        payload["model_path"] = str(path / "model")
        payload["optimizer"] = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False)
        return payload

    def commit(
        self,
        stage: str,
        metrics: Mapping[str, Any],
        *,
        model=None,
        tokenizer=None,
        save_model: bool = False,
        existing_checkpoint: Optional[Path] = None,
    ) -> Optional[Path]:
        checkpoint: Optional[Path] = existing_checkpoint
        if save_model:
            if model is None or tokenizer is None:
                raise ValueError("save_model requires model and tokenizer")
            checkpoint = save_pretrained_atomic(model, tokenizer, self.checkpoint_root / stage)
        if checkpoint is not None:
            self.state.current_checkpoint = str(checkpoint)
        if stage not in self.state.completed_stages:
            self.state.completed_stages.append(stage)
        self.state.stage_records[stage] = dict(metrics)
        for key, value in metrics.items():
            if not finite_number(value):
                continue
            numeric = float(value)
            previous = self.state.best_metrics.get(key)
            lower_is_better = key.endswith("_loss") or key.endswith("_ppl") or key == "stale_action_rate"
            if previous is None or (lower_is_better and numeric < previous) or (not lower_is_better and numeric > previous):
                self.state.best_metrics[key] = numeric
        self.state.current_stage = ""
        self.state.current_step = 0
        self.state.resume_artifact = ""
        self.state.status = "complete" if stage == STAGES[-1] else "ready"
        self.clear_resume(stage)
        self.flush()
        return checkpoint

    def prune_stage_checkpoints(self, keep: int = 2) -> None:
        if not self.checkpoint_root.exists():
            return
        paths = [path for path in self.checkpoint_root.iterdir() if path.is_dir() and path.name != "final"]
        paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        current = Path(self.state.current_checkpoint).resolve() if Path(self.state.current_checkpoint).exists() else None
        for path in paths[int(keep):]:
            if current is not None and path.resolve() == current:
                continue
            shutil.rmtree(path)


class PipelineRunner:
    def __init__(self, args: argparse.Namespace, manifest: Mapping[str, Any]) -> None:
        self.args = args
        self.manifest = dict(manifest)
        self.output_dir = Path(args.output_dir).expanduser()
        self.store = StageStore(self.output_dir, manifest, args.model_id, args.resume)
        self.registry = ProfileRegistry(self.output_dir / "profiles")
        self.audit = DataAccessAudit(self.output_dir / "data_access.jsonl", manifest)
        self.logger = al.ArtifactLogger(self.output_dir / "metrics")
        self.aux_device = al.resolve_aux_device(args.teacher_device, args.device)
        self.tokenizer = None
        self.model = None
        self.cfg: Optional[qp.RuntimeConfig] = None
        self.history_tasks: List[al.TaskData] = []
        self.history_payloads: List[Dict[str, Any]] = []
        self.tasks = {key: payload_to_task(payload) for key, payload in self.manifest["tasks"].items()}
        frozen_path = self.output_dir / "frozen_manifest.json"
        if frozen_path.exists():
            frozen = load_manifest(frozen_path)
            if frozen["manifest_sha256"] != self.manifest["manifest_sha256"]:
                raise ValueError("output directory contains a different frozen manifest")
        else:
            atomic_json(frozen_path, self.manifest)

    def make_config(self) -> qp.RuntimeConfig:
        return qp.RuntimeConfig(
            model_id=self.store.state.current_checkpoint,
            device=self.args.device,
            dtype=choose_dtype(self.args.dtype),
            local_files_only=self.args.local_files_only,
            resume=self.args.resume != "never",
            smoke=self.args.smoke,
            output_dir=self.output_dir,
            backup_dir=None,
            seed=self.args.seed,
            phase_scope="qwen35_lifelong_stage1",
            task_suite="synthetic_lifelong",
            batch_size=self.args.batch_size,
            eval_batch_size=self.args.eval_batch_size,
            consolidation_micro_batch_size=self.args.micro_batch_size,
            max_seq_len=self.args.max_seq_len,
            gradient_checkpointing=self.args.gradient_checkpointing,
            wikitext_eval_samples=self.args.wikitext_eval_samples,
            eval_interval=self.args.eval_interval,
            log_interval=self.args.log_interval,
            grad_clip=self.args.grad_clip,
        )

    def load_runtime(self, model_path: Optional[str] = None) -> None:
        path = model_path or self.store.state.current_checkpoint
        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(path, local_files_only=self.args.local_files_only)
        if self.model is not None:
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.model = load_causal_lm(
            path,
            device=self.args.device,
            dtype=choose_dtype(self.args.dtype),
            local_files_only=self.args.local_files_only,
        )
        self.cfg = self.make_config()
        self.cfg.model_id = path
        if self.args.gradient_checkpointing:
            qp._configure_gradient_checkpointing(self.model, True)

    def load_history(self) -> None:
        history = self.args.history_manifest or self.manifest.get("history_manifest")
        if not history:
            return
        path = Path(history).expanduser()
        expected = self.manifest.get("history_manifest_sha256")
        if expected and sha256_file(path) != expected:
            raise ValueError(f"history manifest hash mismatch: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.history_payloads = list(payload.get("skills", []))
        self.history_tasks = [five.task_from_payload(item) for item in self.history_payloads]

    def ensure_wikitext(self) -> None:
        assert self.cfg is not None and self.tokenizer is not None
        try:
            chunks = qp.load_wikitext_texts(
                self.tokenizer,
                split="validation",
                max_seq_len=self.cfg.max_seq_len,
                max_samples=self.cfg.wikitext_eval_samples,
                local_files_only=self.args.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(f"WikiText validation data is required for Stage 1: {exc}") from exc
        if not chunks:
            raise RuntimeError("WikiText validation data is required for Stage 1 but zero chunks were loaded")
        self.cfg._wikitext_val = list(chunks)

    def evaluate(self, stage: str, task_keys: Sequence[str], *, include_history: bool = True) -> Dict[str, Any]:
        assert self.model is not None and self.tokenizer is not None and self.cfg is not None
        tasks = ([*self.history_tasks] if include_history else []) + [self.tasks[key] for key in task_keys]
        for task in tasks:
            rows = task.manifest.get("eval_payload", [])
            self.audit.log(
                stage=stage,
                task=task.spec.name,
                split="eval",
                row_ids=[str(item.get("row_id", f"eval:{idx}")) for idx, item in enumerate(rows)],
                purpose="evaluation",
            )
        metrics = five.evaluate_suite_progress(
            self.model,
            self.tokenizer,
            tasks,
            self.cfg,
            do_generation=True,
            include_wikitext=True,
            wikitext_val=getattr(self.cfg, "_wikitext_val", []),
        )
        self.logger.log_stage_summary(stage, metrics, old_task_examples=0, proxy_batches=0)
        if "wikitext_ppl" in metrics and not finite_number(metrics["wikitext_ppl"]):
            raise FloatingPointError(f"{stage}: non-finite WikiText PPL")
        return metrics

    def rows_for(self, task_key: str, split: str) -> List[Dict[str, Any]]:
        return task_rows(self.manifest, task_key, split)

    def logged_supervised_batch(
        self,
        task_key: str,
        stage: str,
        step: int,
        *,
        batch_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        assert self.tokenizer is not None and self.cfg is not None
        rows = self.rows_for(task_key, "train")
        size = int(batch_size or self.cfg.batch_size)
        rng = np.random.default_rng(self.args.seed + al.stable_seed(stage, 7000) + 7919 * int(step))
        indices = rng.integers(0, len(rows), size=size)
        selected = [rows[int(index)] for index in indices]
        self.audit.log(
            stage=stage,
            task=task_key,
            split="train",
            row_ids=[str(item["row_id"]) for item in selected],
            purpose="update",
            allowed_train_tasks=[task_key],
            step=step,
        )
        eos = self.tokenizer.eos_token or ""
        return qp._prepare_supervised_batch(
            self.tokenizer,
            [item["prompt"] for item in selected],
            [item["target"] + eos for item in selected],
            self.cfg.device,
            self.cfg.max_seq_len,
        )

    def stop_requested(self, stage: str) -> bool:
        return bool(self.args.stop_after and self.args.stop_after == stage)

    def artifact_path(self, stage: str, name: str) -> Path:
        path = self.output_dir / "artifacts" / stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def protected_profiles(
        self,
        *,
        exclude_ids: Sequence[str] = (),
        exclude_creation_stage: str = "",
    ) -> List[qt.TaskProfile]:
        return self.registry.protected_profiles(
            exclude_ids=exclude_ids,
            exclude_creation_stage=exclude_creation_stage,
        )

    def rewind_stage_outputs(self, stage: str, step: int) -> None:
        self.audit.rewind_stage(stage, step)
        path = self.logger.curves_path
        if not path.exists():
            return
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        retained = []
        for item in rows:
            try:
                item_step = int(item.get("step", 0))
            except (TypeError, ValueError):
                item_step = 0
            if item.get("stage") == stage and item_step > int(step):
                continue
            retained.append(item)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.logger.curve_fields)
            writer.writeheader()
            writer.writerows(retained)
        os.replace(tmp, path)

    def select_layers(self, stage: str, task_key: str) -> List[int]:
        assert self.model is not None and self.tokenizer is not None and self.cfg is not None
        rows = self.rows_for(task_key, "train")
        self.audit.log(
            stage=stage,
            task=task_key,
            split="train",
            row_ids=[str(item["row_id"]) for item in rows],
            purpose="profile",
        )
        layers, _ = al.select_layers_generic(
            model=self.model,
            tokenizer=self.tokenizer,
            task=self.tasks[task_key],
            protected_profiles=self.protected_profiles(),
            cfg=self.cfg,
            min_layers=self.args.min_layers,
            stage=stage,
            logger=self.logger,
        )
        return [int(value) for value in layers]

    def attach_adapters(self, model, selected_layers: Sequence[int]) -> None:
        qp._freeze_model(model)
        config = LatentLoRAConfig(
            rank=int(self.args.teacher_rank),
            alpha=float(self.args.teacher_alpha),
            dropout=0.0,
            projection_strength=1.0,
            gate_init=float(self.args.teacher_gate_init),
            freeze_base=True,
        )
        attached = attach_latent_lora(
            model,
            suffixes=tuple(self.args.target_suffixes.split(",")),
            layer_indices=set(int(value) for value in selected_layers),
            config=config,
        )
        if not attached:
            raise RuntimeError("no teacher adapters were attached")

    def acquire_supervised_teacher(self, stage: str, task_key: str) -> Dict[str, Any]:
        assert self.model is not None and self.tokenizer is not None and self.cfg is not None
        resume = self.store.load_adapter_resume(stage)
        layers = (
            [int(value) for value in resume.get("extra", {}).get("selected_layers", [])]
            if resume
            else self.select_layers(stage, task_key)
        )
        if not layers:
            raise RuntimeError(f"{stage}: no selected layers in fresh or resumed acquisition")
        teacher = qp._clone_model(self.model, self.args.device)
        self.attach_adapters(teacher, layers)
        params = qp._trainable_params(teacher)
        optimizer = torch.optim.AdamW(params, lr=float(self.args.teacher_lr), foreach=False)
        best_state: Dict[str, torch.Tensor] = {}
        best_score: Tuple[float, float, float, float] = (-math.inf, -math.inf, -math.inf, -math.inf)
        start_step = 0
        if resume:
            restore_named_state(teacher, resume["trainable_state"])
            optimizer.load_state_dict(resume["optimizer"])
            best_state = dict(resume.get("best_state", {}))
            best_score = tuple(resume.get("best_score", best_score))  # type: ignore[assignment]
            start_step = int(resume["step"])
            restore_rng_state(resume["rng"])
            self.rewind_stage_outputs(stage, start_step)

        teacher.train()
        completed_step = start_step
        gate = float(self.manifest["gates"]["primitive_exact"])
        final_metrics: Dict[str, Any] = {}
        for step in range(start_step + 1, int(self.args.teacher_max_steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            batch = self.logged_supervised_batch(task_key, stage, step)
            split = qp._split_tensor_batch(batch, self.cfg.consolidation_micro_batch_size)
            loss_value = 0.0
            for micro in split:
                outputs = teacher(**micro, use_cache=False)
                if not torch.isfinite(outputs.loss):
                    raise FloatingPointError(f"{stage}: non-finite teacher loss at step {step}")
                loss = outputs.loss / max(1, len(split))
                loss.backward()
                loss_value += float(loss.item())
            assert_finite_gradients(params, stage)
            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            optimizer.step()
            completed_step = step

            should_eval = step % int(self.args.teacher_block_steps) == 0 or step == int(self.args.teacher_max_steps)
            if step % int(self.args.log_interval) == 0 or should_eval:
                self.logger.log_curve(stage, step, loss=loss_value, method="adapter_teacher")
                print(f"[{stage}] step={step:04d}/{self.args.teacher_max_steps} loss={loss_value:.4f}", flush=True)
            gate_reached = False
            if should_eval:
                teacher.eval()
                final_metrics = al.evaluate_task(teacher, self.tokenizer, self.tasks[task_key], self.cfg, do_generation=True)
                score = al.score_focus(final_metrics, self.tasks[task_key])
                if score > best_score:
                    best_score = score
                    best_state = trainable_state(teacher)
                exact = task_metric(final_metrics, self.tasks[task_key].spec.name)
                print(f"[{stage}] eval exact={al.fmt(exact)} score={tuple(round(x, 4) for x in score)}", flush=True)
                if step >= int(self.args.teacher_min_steps) and finite_number(exact) and exact >= gate:
                    gate_reached = True
                teacher.train()
            if step % int(self.args.checkpoint_interval) == 0:
                self.store.save_adapter_resume(
                    stage,
                    step=step,
                    model=teacher,
                    optimizer=optimizer,
                    best_state=best_state,
                    best_score=best_score,
                    extra={"selected_layers": layers, "task_key": task_key},
                )
            if gate_reached:
                break

        if best_state:
            restore_named_state(teacher, best_state)
            final_metrics = al.evaluate_task(teacher, self.tokenizer, self.tasks[task_key], self.cfg, do_generation=True)
        exact = task_metric(final_metrics, self.tasks[task_key].spec.name)
        if not finite_number(exact) or exact < gate:
            raise RuntimeError(f"{stage}: teacher acquisition gate failed exact={exact:.4f} need>={gate:.4f}")

        artifact = self.artifact_path(stage, "adapter_teacher.pt")
        atomic_torch_save(
            artifact,
            {
                "stage": stage,
                "task_key": task_key,
                "selected_layers": layers,
                "teacher_rank": int(self.args.teacher_rank),
                "teacher_alpha": float(self.args.teacher_alpha),
                "teacher_gate_init": float(self.args.teacher_gate_init),
                "target_suffixes": self.args.target_suffixes,
                "trainable_state": trainable_state(teacher),
                "metrics": final_metrics,
                "completed_step": completed_step,
            },
        )
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.store.commit(stage, {"task_key": task_key, "adapter_artifact": str(artifact), **final_metrics})
        return torch.load(artifact, map_location="cpu", weights_only=False)

    def load_adapter_teacher(self, artifact: Mapping[str, Any], device: str, *, base_checkpoint: Optional[str] = None):
        assert self.model is not None
        teacher = (
            load_causal_lm(
                base_checkpoint,
                device=device,
                dtype=choose_dtype(self.args.dtype),
                local_files_only=self.args.local_files_only,
            )
            if base_checkpoint
            else qp._clone_model(self.model, device)
        )
        self.attach_adapters(teacher, artifact["selected_layers"])
        restore_named_state(teacher, artifact["trainable_state"])
        qp._freeze_model(teacher)
        return teacher

    def full_resume_or_current(self, stage: str) -> Tuple[int, Optional[Dict[str, Any]]]:
        resume = self.store.load_full_resume(stage)
        if not resume:
            return 0, None
        self.load_runtime(resume["model_path"])
        restore_rng_state(resume["rng"])
        self.rewind_stage_outputs(stage, int(resume["step"]))
        return int(resume["step"]), resume

    def consolidate_supervised(self, stage: str, task_key: str, artifact_stage: str) -> Dict[str, Any]:
        assert self.model is not None and self.tokenizer is not None and self.cfg is not None
        base_checkpoint = self.store.state.current_checkpoint
        artifact = torch.load(self.artifact_path(artifact_stage, "adapter_teacher.pt"), map_location="cpu", weights_only=False)
        start_step, resume = self.full_resume_or_current(stage)
        assert self.model is not None and self.cfg is not None
        old_teacher = load_causal_lm(
            base_checkpoint,
            device=self.aux_device,
            dtype=choose_dtype(self.args.dtype),
            local_files_only=self.args.local_files_only,
        )
        qp._freeze_model(old_teacher)
        new_teacher = self.load_adapter_teacher(artifact, self.aux_device, base_checkpoint=base_checkpoint)
        params = al._trainable_full_student(self.model, self.cfg)
        optimizer = torch.optim.AdamW(params, lr=float(self.args.consolidation_lr), foreach=False)
        if resume:
            optimizer.load_state_dict(resume["optimizer"])
        protected = self.protected_profiles(exclude_creation_stage=stage)
        layers = [int(value) for value in artifact["selected_layers"]]

        for step in range(start_step + 1, int(self.args.consolidation_steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            batch = self.logged_supervised_batch(task_key, stage, step)
            split = qp._split_tensor_batch(batch, self.cfg.consolidation_micro_batch_size)
            loss_value = 0.0
            for micro in split:
                student_outputs = self.model(**micro, output_hidden_states=True, use_cache=False)
                teacher_micro = al.move_batch(micro, self.aux_device)
                with torch.no_grad():
                    old_outputs = old_teacher(**teacher_micro, output_hidden_states=True, use_cache=False)
                    new_outputs = new_teacher(**teacher_micro, output_hidden_states=True, use_cache=False)
                new_kl = al.kl_divergence_to_student_device(student_outputs.logits, new_outputs.logits)
                old_kl = al.kl_divergence_to_student_device(student_outputs.logits, old_outputs.logits)
                new_hidden = al.hidden_alignment_to_student_device(student_outputs, new_outputs, layers, self.cfg.device)
                old_hidden = al.hidden_alignment_to_student_device(student_outputs, old_outputs, layers, self.cfg.device)
                loss = (
                    student_outputs.loss
                    + float(self.args.new_kl_weight) * new_kl
                    + float(self.args.new_hidden_weight) * new_hidden
                    + float(self.args.old_kl_weight) * old_kl
                    + float(self.args.old_hidden_weight) * old_hidden
                ) / max(1, len(split))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{stage}: non-finite consolidation loss at step {step}")
                loss.backward()
                loss_value += float(loss.item())
            assert_finite_gradients(params, stage)
            projected = al.project_old_occupied_gradients(
                self.model,
                protected,
                layers,
                strength=float(self.args.projection_strength),
            )
            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            optimizer.step()
            if step % int(self.args.log_interval) == 0 or step == int(self.args.consolidation_steps):
                print(f"[{stage}] step={step:04d}/{self.args.consolidation_steps} loss={loss_value:.4f} projected={projected}", flush=True)
                self.logger.log_curve(stage, step, loss=loss_value, projected_modules=projected, method="amoeba_no_proxy")
            if step % int(self.args.checkpoint_interval) == 0 and step < int(self.args.consolidation_steps):
                self.store.save_full_resume(stage, step, self.model, self.tokenizer, optimizer, {"task_key": task_key, "layers": layers})

        assert_finite_model(self.model, stage)
        task_keys = ["skill_a"] if task_key == "skill_a" else ["skill_a", "skill_b_v1"]
        metrics = self.evaluate(stage, task_keys)
        gate = float(self.manifest["gates"]["primitive_exact"])
        exact = task_metric(metrics, self.tasks[task_key].spec.name)
        if not finite_number(exact) or exact < gate:
            raise RuntimeError(f"{stage}: consolidation acquisition gate failed exact={exact:.4f} need>={gate:.4f}")
        self.protected_drop_check(stage, metrics, "00_bootstrap" if task_key == "skill_a" else "11_consolidate_a")

        checkpoint = save_pretrained_atomic(self.model, self.tokenizer, self.store.checkpoint_root / stage)
        checkpoint_hash = checkpoint_identity(str(checkpoint))
        with self.registry.transaction():
            self.register_task_profile(
                profile_id="skill_a",
                task_key="skill_a",
                scope={"skill": "glyph_color"},
                dependencies=[],
                selected_layers=layers,
                checkpoint_hash=checkpoint_hash,
                creation_stage=stage,
            ) if task_key == "skill_a" else self.register_skill_b_profiles(layers, checkpoint_hash, stage, version="v1")
        self.store.commit(stage, metrics, existing_checkpoint=checkpoint)
        self.store.prune_stage_checkpoints(self.args.keep_stage_checkpoints)
        del old_teacher, new_teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics

    def task_from_selected_rows(self, task_key: str, rows: Sequence[Dict[str, Any]], name: str) -> al.TaskData:
        base = self.manifest["tasks"][task_key]
        payload = task_payload(name, name.replace("_", " ").title(), rows, rows, max_new_tokens=int(base["recipe"]["max_new_tokens"]))
        return payload_to_task(payload)

    def register_task_profile(
        self,
        *,
        profile_id: str,
        task_key: str,
        scope: Mapping[str, str],
        dependencies: Sequence[str],
        selected_layers: Sequence[int],
        checkpoint_hash: str,
        creation_stage: str,
        selected_rows: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        assert self.model is not None and self.tokenizer is not None and self.cfg is not None
        if profile_id in self.registry.entries:
            existing = self.registry.entries[profile_id]
            if existing.creation_stage == creation_stage:
                print(f"[profile-resume] reuse profile_id={profile_id}", flush=True)
                return
            raise ValueError(f"profile_id={profile_id} already belongs to stage={existing.creation_stage}")
        rows = list(selected_rows or self.rows_for(task_key, "train"))
        task = self.task_from_selected_rows(task_key, rows, profile_id)
        self.audit.log(
            stage=creation_stage,
            task=task_key,
            split="train",
            row_ids=[str(item["row_id"]) for item in rows],
            purpose="profile",
        )
        profile = al.collect_profile(self.model, self.tokenizer, task, self.cfg, profile_id)
        self.registry.add_profile(
            profile_id,
            profile,
            scope=scope,
            dependencies=dependencies,
            selected_layers=selected_layers,
            checkpoint_sha256=checkpoint_hash,
            creation_stage=creation_stage,
        )

    def register_skill_b_profiles(self, layers: Sequence[int], checkpoint_hash: str, stage: str, *, version: str) -> None:
        task_key = "skill_b_v1" if version == "v1" else "skill_b_v2_changed"
        rows = self.rows_for(task_key, "train")
        for color in COLORS:
            selected = [item for item in rows if item.get("color") == color]
            if not selected:
                continue
            self.register_task_profile(
                profile_id=f"skill_b:{version}:{color}",
                task_key=task_key,
                scope={"skill": "color_action", "color": color, "version": version},
                dependencies=[],
                selected_layers=layers,
                checkpoint_hash=checkpoint_hash,
                creation_stage=stage,
                selected_rows=selected,
            )

    def selected_layers_from_profiles(self, profile_ids: Optional[Sequence[str]] = None) -> List[int]:
        ids = list(profile_ids or self.registry.protected_ids())
        layers = sorted({layer for key in ids if key in self.registry.entries for layer in self.registry.entries[key].selected_layers})
        if len(layers) < int(self.args.min_layers):
            total = int(getattr(getattr(self.model, "config", None), "num_hidden_layers", 24) or 24)
            layers.extend(idx for idx in range(total) if idx not in layers)
        return layers[: max(int(self.args.min_layers), len(layers))]

    def sample_manifest_rows(self, task_key: str, stage: str, step: int) -> List[Dict[str, Any]]:
        rows = self.rows_for(task_key, "train")
        rng = np.random.default_rng(self.args.seed + al.stable_seed(stage, 8100) + 7919 * int(step))
        indices = rng.integers(0, len(rows), size=int(self.args.batch_size))
        selected = [rows[int(index)] for index in indices]
        self.audit.log(
            stage=stage,
            task=task_key,
            split="train",
            row_ids=[str(item["row_id"]) for item in selected],
            purpose="update",
            allowed_train_tasks=[task_key],
            step=step,
        )
        return selected

    def generate(self, model, prompts: Sequence[str], *, sample: bool, max_new_tokens: int) -> List[str]:
        assert self.tokenizer is not None
        return al.generate_on_policy_completions(
            model,
            self.tokenizer,
            prompts,
            self.args.device,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=1.0 if sample else 0.0,
            top_p=0.95 if sample else 1.0,
        )

    def composition_contexts(self, model, rows: Sequence[Dict[str, Any]], modes: Sequence[str]) -> Tuple[List[str], List[bool]]:
        glyph_values = [str(item["glyph"]) for item in rows]
        color_outputs = self.generate(
            model,
            [glyph_color_prompt(glyph, 9000 + idx) for idx, glyph in enumerate(glyph_values)],
            sample=False,
            max_new_tokens=8,
        )
        predicted_colors = [parse_code(value, COLORS) for value in color_outputs]
        safe_colors = [value if value in COLORS else COLORS[0] for value in predicted_colors]
        action_outputs = self.generate(
            model,
            [color_action_prompt(color, 9100 + idx) for idx, color in enumerate(safe_colors)],
            sample=False,
            max_new_tokens=8,
        )
        predicted_actions = [parse_code(value, ACTIONS) for value in action_outputs]
        prompts: List[str] = []
        correct: List[bool] = []
        for item, mode, color, action in zip(rows, modes, predicted_colors, predicted_actions):
            direct = str(item["prompt"])
            if mode == "full":
                prompts.append(
                    f"{direct}\n\nPrivileged on-policy route from the current model:\n"
                    f"COLOR={color or 'UNKNOWN'}; ACTION={action or 'UNKNOWN'}\n"
                    "Use this route and return only the final action code."
                )
            elif mode == "compressed":
                prompts.append(f"{direct}\n\nOn-policy route: C={color or 'UNKNOWN'}; A={action or 'UNKNOWN'}")
            else:
                prompts.append(direct)
            correct.append(bool(color and color == item["color"] and action and action == item["target"]))
        return prompts, correct

    def policy_change_contexts(self, rows: Sequence[Dict[str, Any]], modes: Sequence[str]) -> Tuple[List[str], List[bool]]:
        prompts: List[str] = []
        correct: List[bool] = []
        for item, mode in zip(rows, modes):
            direct = str(item["prompt"])
            color = str(item["color"])
            target = str(item["target"])
            old_action = str(item["old_action"])
            if mode == "full":
                prompts.append(
                    f"{direct}\n\nPrivileged policy update:\n"
                    f"The former mapping {color}->{old_action} is stale. The current mapping is {color}->{target}.\n"
                    "Apply the current mapping and return only the action code."
                )
            elif mode == "compressed":
                prompts.append(f"{direct}\n\nUPDATE {color}:{old_action}->{target}")
            else:
                prompts.append(direct)
            correct.append(mode != "none")
        return prompts, correct

    def acquire_on_policy_teacher(
        self,
        stage: str,
        task_key: str,
        *,
        kind: str,
        selected_profile_ids: Optional[Sequence[str]] = None,
        release_plan: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        assert self.model is not None and self.tokenizer is not None and self.cfg is not None
        resume = self.store.load_adapter_resume(stage)
        layers = (
            [int(value) for value in resume.get("extra", {}).get("selected_layers", [])]
            if resume
            else self.selected_layers_from_profiles(selected_profile_ids)
        )
        if not layers:
            raise RuntimeError(f"{stage}: no selected layers in fresh or resumed acquisition")
        teacher = qp._clone_model(self.model, self.args.device)
        self.attach_adapters(teacher, layers)
        params = qp._trainable_params(teacher)
        optimizer = torch.optim.AdamW(params, lr=float(self.args.composition_lr), foreach=False)
        start_step = 0
        best_state: Dict[str, torch.Tensor] = {}
        best_score = -math.inf
        trajectories: List[Dict[str, Any]] = []
        context_window: List[float] = []
        if resume:
            restore_named_state(teacher, resume["trainable_state"])
            optimizer.load_state_dict(resume["optimizer"])
            best_state = dict(resume.get("best_state", {}))
            best_score = float(resume.get("best_score", -math.inf))
            start_step = int(resume["step"])
            restore_rng_state(resume["rng"])
            resume_extra = resume.get("extra", {})
            if "exact_rollouts" in resume_extra and "trajectories" not in resume_extra:
                raise RuntimeError(
                    f"{stage}: legacy reward-filtered resume is incompatible; remove this stage's resume artifact"
                )
            trajectories = list(resume_extra.get("trajectories", []))
            context_window = list(resume.get("extra", {}).get("context_window", []))
            self.rewind_stage_outputs(stage, start_step)

        reward_steps = int(self.args.composition_reward_steps)
        hybrid_steps = int(self.args.composition_hybrid_steps)
        total_steps = reward_steps + hybrid_steps
        mixture = self.manifest["hyperparameters"]["context_mixture"]
        modes = deterministic_modes(max(1, hybrid_steps * self.args.batch_size), mixture, f"{self.args.seed}:{stage}")
        mode_cursor = max(0, (start_step - reward_steps) * int(self.args.batch_size))
        max_new = int(self.manifest["tasks"][task_key]["recipe"]["max_new_tokens"])

        for step in range(start_step + 1, total_steps + 1):
            rows = self.sample_manifest_rows(task_key, stage, step)
            direct_prompts = [str(item["prompt"]) for item in rows]
            if step <= reward_steps:
                batch_modes = ["none"] * len(rows)
            else:
                batch_modes = [modes[(mode_cursor + idx) % len(modes)] for idx in range(len(rows))]
                mode_cursor += len(rows)
            if kind == "composition":
                teacher_prompts, context_correct = self.composition_contexts(teacher, rows, batch_modes)
            elif kind == "policy_change":
                teacher_prompts, context_correct = self.policy_change_contexts(rows, batch_modes)
            else:
                raise ValueError(f"unknown on-policy acquisition kind={kind}")
            context_window.extend(float(value) for value in context_correct)
            context_window = context_window[-30 * max(1, int(self.args.batch_size)) :]

            rollout_prompts: List[str] = []
            rollout_teacher_prompts: List[str] = []
            rollout_rows: List[Dict[str, Any]] = []
            rollout_modes: List[str] = []
            for prompt, teacher_prompt, item, mode in zip(direct_prompts, teacher_prompts, rows, batch_modes):
                for _ in range(int(self.args.rollouts_per_prompt)):
                    rollout_prompts.append(prompt)
                    rollout_teacher_prompts.append(teacher_prompt)
                    rollout_rows.append(item)
                    rollout_modes.append(mode)
            teacher.eval()
            with torch.no_grad():
                completions = self.generate(teacher, rollout_prompts, sample=True, max_new_tokens=max_new)
            rewards: List[float] = []
            infos: List[Dict[str, Any]] = []
            for rollout_index, (prompt, teacher_prompt, completion, item, mode) in enumerate(
                zip(rollout_prompts, rollout_teacher_prompts, completions, rollout_rows, rollout_modes)
            ):
                reward, info = action_reward(completion, str(item["target"]))
                rewards.append(reward)
                infos.append(info)
                trajectories.append(
                    make_on_policy_trajectory(
                        stage=stage,
                        step=step,
                        context_mode=mode,
                        rollout_index=rollout_index,
                        task_key=task_key,
                        item=item,
                        prompt=prompt,
                        teacher_prompt=teacher_prompt,
                        completion=completion,
                        reward=reward,
                        verifier=info,
                    )
                )

            advantages: List[float] = []
            group = int(self.args.rollouts_per_prompt)
            for start in range(0, len(rewards), group):
                values = rewards[start : start + group]
                mean = float(sum(values) / max(1, len(values)))
                advantages.extend(value - mean for value in values)
            advantage_tensor = torch.tensor(advantages, dtype=torch.float32)
            teacher.train()
            optimizer.zero_grad(set_to_none=True)
            policy_loss = prepare_completion_policy_loss(
                teacher,
                self.tokenizer,
                rollout_prompts,
                completions,
                advantage_tensor,
                self.args.device,
                self.cfg.max_seq_len,
            )
            loss = policy_loss
            kd_value = 0.0
            if step > reward_steps and any(mode != "none" for mode in batch_modes):
                student_batch = al._prepare_completion_kl_batch(
                    self.tokenizer, rollout_prompts, completions, self.args.device, self.cfg.max_seq_len
                )
                teacher_batch = al._prepare_completion_kl_batch(
                    self.tokenizer, rollout_teacher_prompts, completions, self.args.device, self.cfg.max_seq_len
                )
                student_outputs = teacher(
                    input_ids=student_batch["input_ids"],
                    attention_mask=student_batch["attention_mask"],
                    use_cache=False,
                )
                with torch.no_grad():
                    context_outputs = teacher(
                        input_ids=teacher_batch["input_ids"],
                        attention_mask=teacher_batch["attention_mask"],
                        use_cache=False,
                    )
                kd_loss = al._completion_forward_kl_loss(student_outputs, context_outputs, student_batch["completion_mask"])
                loss = loss + float(self.args.context_kl_weight) * kd_loss
                kd_value = float(kd_loss.item())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{stage}: non-finite on-policy loss at step {step}")
            loss.backward()
            assert_finite_gradients(params, stage)
            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            optimizer.step()
            mean_reward = float(sum(rewards) / max(1, len(rewards)))
            exact_rate = float(sum(1 for info in infos if info["correct"]) / max(1, len(infos)))
            if mean_reward > best_score:
                best_score = mean_reward
                best_state = trainable_state(teacher)
            if step % int(self.args.log_interval) == 0 or step == total_steps:
                context_rate = float(sum(context_window) / max(1, len(context_window)))
                print(
                    f"[{stage}] step={step:04d}/{total_steps} loss={float(loss.item()):.4f} "
                    f"reward={mean_reward:.3f} exact={exact_rate:.3f} context={context_rate:.3f} kd={kd_value:.4f}",
                    flush=True,
                )
                self.logger.log_curve(
                    stage,
                    step,
                    loss=float(loss.item()),
                    mean_reward=mean_reward,
                    exact=exact_rate,
                    positive_context_rate=context_rate,
                    kd_loss=kd_value,
                    method=f"on_policy_{kind}",
                )
            if step % int(self.args.checkpoint_interval) == 0 and step < total_steps:
                self.store.save_adapter_resume(
                    stage,
                    step=step,
                    model=teacher,
                    optimizer=optimizer,
                    best_state=best_state,
                    best_score=best_score,
                    extra={
                        "selected_layers": layers,
                        "task_key": task_key,
                        "kind": kind,
                        "trajectories": trajectories,
                        "context_window": context_window,
                        "release_plan": list(release_plan or []),
                    },
                )

        if best_state:
            restore_named_state(teacher, best_state)
        context_rate = float(sum(context_window) / max(1, len(context_window)))
        gate = float(self.manifest["gates"]["positive_context_rate"])
        if context_rate < gate:
            raise RuntimeError(f"{stage}: positive context rate={context_rate:.4f} need>={gate:.4f}")
        if not trajectories:
            raise RuntimeError(f"{stage}: no on-policy trajectories were collected")

        artifact = self.artifact_path(stage, "adapter_teacher.pt")
        atomic_torch_save(
            artifact,
            {
                "stage": stage,
                "task_key": task_key,
                "kind": kind,
                "selected_layers": layers,
                "teacher_rank": int(self.args.teacher_rank),
                "teacher_alpha": float(self.args.teacher_alpha),
                "teacher_gate_init": float(self.args.teacher_gate_init),
                "target_suffixes": self.args.target_suffixes,
                "trainable_state": trainable_state(teacher),
                "positive_context_rate": context_rate,
                "best_mean_reward": best_score,
                "release_plan": list(release_plan or []),
            },
        )
        rollout_path = self.artifact_path(stage, "on_policy_trajectories.jsonl")
        with rollout_path.open("w", encoding="utf-8") as handle:
            for item in trajectories:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        if release_plan:
            with self.registry.transaction():
                actual = self.registry.release_closure(release_plan)
                if sorted(actual) != sorted(set(release_plan)):
                    raise RuntimeError("release dependency closure changed after acquisition")
        metrics = {
            "task_key": task_key,
            "kind": kind,
            "positive_context_rate": context_rate,
            "best_mean_reward": best_score,
            "on_policy_trajectories": len(trajectories),
            "correct_trajectories": sum(int(item["correct"]) for item in trajectories),
            "adapter_artifact": str(artifact),
            "trajectory_artifact": str(rollout_path),
            "release_plan": list(release_plan or []),
        }
        self.store.commit(stage, metrics)
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics

    def read_on_policy_trajectories(self, artifact_stage: str) -> List[Dict[str, Any]]:
        path = self.artifact_path(artifact_stage, "on_policy_trajectories.jsonl")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise RuntimeError(f"no on-policy trajectories in {path}")
        required = {"row_id", "prompt", "teacher_prompt", "completion", "reward", "prediction", "correct", "valid"}
        for index, item in enumerate(rows):
            missing = required - set(item)
            if missing:
                raise ValueError(f"trajectory index={index} missing fields={sorted(missing)}")
            if "expected" in item or "target" in item:
                raise ValueError(f"trajectory index={index} contains prohibited gold target")
        return rows

    def consolidate_on_policy_no_proxy(
        self,
        stage: str,
        artifact_stage: str,
        *,
        eval_task_keys: Sequence[str],
    ) -> Dict[str, Any]:
        assert self.model is not None and self.tokenizer is not None and self.cfg is not None
        base_checkpoint = self.store.state.current_checkpoint
        artifact = torch.load(self.artifact_path(artifact_stage, "adapter_teacher.pt"), map_location="cpu", weights_only=False)
        trajectories = self.read_on_policy_trajectories(artifact_stage)
        start_step, resume = self.full_resume_or_current(stage)
        assert self.model is not None and self.cfg is not None
        old_teacher = load_causal_lm(
            base_checkpoint,
            device=self.aux_device,
            dtype=choose_dtype(self.args.dtype),
            local_files_only=self.args.local_files_only,
        )
        qp._freeze_model(old_teacher)
        new_teacher = self.load_adapter_teacher(artifact, self.aux_device, base_checkpoint=base_checkpoint)
        params = al._trainable_full_student(self.model, self.cfg)
        optimizer = torch.optim.AdamW(params, lr=float(self.args.composition_lr), foreach=False)
        if resume:
            optimizer.load_state_dict(resume["optimizer"])
        layers = [int(value) for value in artifact["selected_layers"]]
        release_plan = list(artifact.get("release_plan", [])) if artifact["kind"] == "policy_change" else []
        protected = self.protected_profiles(
            exclude_ids=release_plan,
            exclude_creation_stage=stage,
        )
        task_key = str(artifact["task_key"])

        for step in range(start_step + 1, int(self.args.consolidation_steps) + 1):
            rng = np.random.default_rng(self.args.seed + al.stable_seed(stage, 9200) + 7919 * step)
            indices = rng.integers(0, len(trajectories), size=int(self.args.batch_size))
            selected = [trajectories[int(index)] for index in indices]
            self.audit.log(
                stage=stage,
                task=task_key,
                split="train",
                row_ids=[str(item["row_id"]) for item in selected],
                purpose="update",
                allowed_train_tasks=[task_key],
                step=step,
            )
            prompts = [str(item["prompt"]) for item in selected]
            teacher_prompts = [str(item["teacher_prompt"]) for item in selected]
            completions = [str(item["completion"]) for item in selected]
            student_batch = al._prepare_completion_kl_batch(
                self.tokenizer, prompts, completions, self.args.device, self.cfg.max_seq_len
            )
            old_batch = al._prepare_completion_kl_batch(
                self.tokenizer, prompts, completions, self.aux_device, self.cfg.max_seq_len
            )
            new_batch = al._prepare_completion_kl_batch(
                self.tokenizer, teacher_prompts, completions, self.aux_device, self.cfg.max_seq_len
            )
            optimizer.zero_grad(set_to_none=True)
            student_outputs = self.model(
                input_ids=student_batch["input_ids"],
                attention_mask=student_batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            with torch.no_grad():
                old_outputs = old_teacher(
                    input_ids=old_batch["input_ids"],
                    attention_mask=old_batch["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
                new_outputs = new_teacher(
                    input_ids=new_batch["input_ids"],
                    attention_mask=new_batch["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
            new_kl = al._completion_forward_kl_loss(student_outputs, new_outputs, student_batch["completion_mask"])
            old_kl = al._completion_forward_kl_loss(student_outputs, old_outputs, student_batch["completion_mask"])
            new_hidden = al.hidden_alignment_to_student_device(student_outputs, new_outputs, layers, self.cfg.device)
            old_hidden = al.hidden_alignment_to_student_device(student_outputs, old_outputs, layers, self.cfg.device)
            loss = (
                float(self.args.new_kl_weight) * new_kl
                + float(self.args.new_hidden_weight) * new_hidden
                + float(self.args.old_kl_weight) * old_kl
                + float(self.args.old_hidden_weight) * old_hidden
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{stage}: non-finite label-free consolidation loss at step {step}")
            loss.backward()
            assert_finite_gradients(params, stage)
            projected = al.project_old_occupied_gradients(
                self.model,
                protected,
                layers,
                strength=float(self.args.projection_strength),
            )
            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            optimizer.step()
            if step % int(self.args.log_interval) == 0 or step == int(self.args.consolidation_steps):
                print(f"[{stage}] step={step:04d}/{self.args.consolidation_steps} loss={float(loss.item()):.4f} projected={projected}", flush=True)
                self.logger.log_curve(stage, step, loss=float(loss.item()), projected_modules=projected, method="on_policy_no_proxy")
            if step % int(self.args.checkpoint_interval) == 0 and step < int(self.args.consolidation_steps):
                self.store.save_full_resume(stage, step, self.model, self.tokenizer, optimizer, {"artifact_stage": artifact_stage, "layers": layers})

        assert_finite_model(self.model, stage)
        metrics = self.evaluate(stage, eval_task_keys)
        if artifact["kind"] == "composition":
            self.validate_composition_metrics(stage, metrics)
        else:
            self.validate_policy_update_metrics(stage, metrics)
        checkpoint = save_pretrained_atomic(self.model, self.tokenizer, self.store.checkpoint_root / stage)
        checkpoint_hash = checkpoint_identity(str(checkpoint))
        with self.registry.transaction():
            if artifact["kind"] == "composition":
                self.register_composition_profiles(layers, checkpoint_hash, stage, version="v1")
            else:
                released = list(artifact.get("release_plan", []))
                missing = [key for key in released if key not in self.registry.entries]
                if missing:
                    raise RuntimeError(f"cannot retire missing released profiles: {missing}")
                self.registry.retire(released)
                self.register_skill_b_profiles(layers, checkpoint_hash, stage, version="v2")
                self.register_composition_profiles(
                    layers,
                    checkpoint_hash,
                    stage,
                    version="v2",
                    colors=self.manifest["mappings"]["changed_colors"],
                )
        self.store.commit(stage, metrics, existing_checkpoint=checkpoint)
        self.store.prune_stage_checkpoints(self.args.keep_stage_checkpoints)
        del old_teacher, new_teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics

    def register_composition_profiles(
        self,
        layers: Sequence[int],
        checkpoint_hash: str,
        stage: str,
        *,
        version: str,
        colors: Sequence[str] = COLORS,
    ) -> None:
        task_key = "composition_direct_v1" if version == "v1" else "composition_direct_v2"
        rows = self.rows_for(task_key, "train")
        for color in colors:
            selected = [item for item in rows if item.get("color") == color]
            if not selected:
                continue
            dependency = f"skill_b:{version}:{color}"
            self.register_task_profile(
                profile_id=f"composition:{version}:{color}",
                task_key=task_key,
                scope={"skill": "glyph_action", "color": color, "version": version},
                dependencies=[dependency],
                selected_layers=layers,
                checkpoint_hash=checkpoint_hash,
                creation_stage=stage,
                selected_rows=selected,
            )

    def protected_drop_check(self, stage: str, metrics: Mapping[str, Any], reference_stage: str) -> None:
        reference = self.store.state.stage_records.get(reference_stage, {})
        max_drop = float(self.manifest["gates"]["protected_max_drop"])
        failures: List[str] = []
        for key, before in reference.items():
            if not key.endswith("_exact") or key not in metrics:
                continue
            after = metrics[key]
            if finite_number(before) and finite_number(after) and float(before) - float(after) > max_drop:
                failures.append(f"{key}:{float(before):.3f}->{float(after):.3f}")
        if failures:
            raise RuntimeError(f"{stage}: protected retention gate failed: {', '.join(failures)}")

    def bootstrap(self) -> Dict[str, Any]:
        assert self.model is not None and self.tokenizer is not None and self.cfg is not None
        assert_finite_model(self.model, "00_bootstrap")
        metrics = self.evaluate(
            "00_bootstrap",
            [
                "skill_a",
                "skill_b_v1",
                "composition_direct_v1",
                "composition_prompted_v1",
                "composition_compressed_v1",
            ],
        )
        checkpoint_hash = checkpoint_identity(self.store.state.current_checkpoint)
        chunks = list(getattr(self.cfg, "_wikitext_val", []))
        if chunks and "base_language" not in self.registry.entries:
            batch_fn = qp.make_wikitext_batch_fn(self.tokenizer, chunks, self.cfg.device, self.cfg, self.args.seed + 9900)
            profile = qp._collect_profiles(self.model, "base_language", batch_fn)
            self.registry.add_profile(
                "base_language",
                profile,
                scope={"skill": "base_language"},
                dependencies=[],
                selected_layers=sorted(profile.layer_profiles),
                checkpoint_sha256=checkpoint_hash,
                creation_stage="00_bootstrap",
            )
        for payload, task in zip(self.history_payloads, self.history_tasks):
            profile_id = f"history:{task.spec.name}"
            if profile_id in self.registry.entries or not task.train:
                continue
            raw_rows = payload.get("train", [])
            self.audit.log(
                stage="00_bootstrap",
                task=task.spec.name,
                split="train",
                row_ids=[str(item.get("row_id", f"history:{task.spec.name}:{idx}")) for idx, item in enumerate(raw_rows)],
                purpose="profile",
            )
            profile = al.collect_profile(self.model, self.tokenizer, task, self.cfg, profile_id)
            self.registry.add_profile(
                profile_id,
                profile,
                scope={"skill": task.spec.name},
                dependencies=[],
                selected_layers=sorted(profile.layer_profiles),
                checkpoint_sha256=checkpoint_hash,
                creation_stage="00_bootstrap",
            )
        metrics["checkpoint_identity"] = checkpoint_hash
        metrics["history_profiles"] = len(self.history_tasks)
        self.store.commit("00_bootstrap", metrics)
        return metrics

    def zero_shot_composition(self) -> Dict[str, Any]:
        metrics = self.evaluate(
            "30_zero_shot_composition",
            [
                "skill_a",
                "skill_b_v1",
                "composition_direct_v1",
                "composition_prompted_v1",
                "composition_compressed_v1",
            ],
        )
        metrics["immutable_zero_shot"] = True
        self.store.commit("30_zero_shot_composition", metrics)
        return metrics

    def validate_composition_metrics(self, stage: str, metrics: Mapping[str, Any]) -> None:
        gates = self.manifest["gates"]
        direct = task_metric(metrics, self.tasks["composition_direct_v1"].spec.name)
        prompted = task_metric(metrics, self.tasks["composition_prompted_v1"].spec.name)
        a_exact = task_metric(metrics, self.tasks["skill_a"].spec.name)
        b_exact = task_metric(metrics, self.tasks["skill_b_v1"].spec.name)
        zero = self.store.state.stage_records["30_zero_shot_composition"]
        zero_direct = task_metric(zero, self.tasks["composition_direct_v1"].spec.name)
        failures = []
        if not finite_number(direct) or direct < float(gates["composition_direct_exact"]):
            failures.append(f"direct={direct:.3f}")
        if not finite_number(direct - zero_direct) or direct - zero_direct < float(gates["composition_direct_gain"]):
            failures.append(f"gain={direct - zero_direct:.3f}")
        if not finite_number(prompted) or prompted < float(gates["composition_prompted_exact"]):
            failures.append(f"prompted={prompted:.3f}")
        if min(a_exact, b_exact) < float(gates["primitive_exact"]):
            failures.append(f"primitive_min={min(a_exact, b_exact):.3f}")
        if failures:
            raise RuntimeError(f"{stage}: composition gate failed: {', '.join(failures)}")

    def validate_policy_update_metrics(self, stage: str, metrics: Mapping[str, Any]) -> None:
        gates = self.manifest["gates"]
        changed = task_metric(metrics, self.tasks["skill_b_v2_changed"].spec.name)
        stable = task_metric(metrics, self.tasks["skill_b_v2_stable"].spec.name)
        a_exact = task_metric(metrics, self.tasks["skill_a"].spec.name)
        direct = task_metric(metrics, self.tasks["composition_direct_v2"].spec.name)
        reference = self.store.state.stage_records["41_consolidate_composition"]
        a_before = task_metric(reference, self.tasks["skill_a"].spec.name)
        b_before = task_metric(reference, self.tasks["skill_b_v1"].spec.name)
        stale_rate = self.stale_action_rate()
        failures = []
        if changed < float(gates["changed_policy_exact"]):
            failures.append(f"changed={changed:.3f}")
        if stale_rate > float(gates["stale_rate"]):
            failures.append(f"stale_rate={stale_rate:.3f}")
        if finite_number(stable) and finite_number(b_before) and b_before - stable > float(gates["protected_max_drop"]):
            failures.append(f"stable_drop={b_before - stable:.3f}")
        if finite_number(a_exact) and finite_number(a_before) and a_before - a_exact > float(gates["protected_max_drop"]):
            failures.append(f"skill_a_drop={a_before - a_exact:.3f}")
        if direct < float(gates["composition_direct_exact"]):
            failures.append(f"v2_direct={direct:.3f}")
        if failures:
            raise RuntimeError(f"{stage}: selective update gate failed: {', '.join(failures)}")
        if isinstance(metrics, MutableMapping):
            metrics["stale_action_rate"] = stale_rate

    def stale_action_rate(self) -> float:
        assert self.model is not None and self.tokenizer is not None
        rows = self.rows_for("skill_b_v2_changed", "eval")
        if not rows:
            return float("nan")
        outputs = self.generate(self.model, [str(item["prompt"]) for item in rows], sample=False, max_new_tokens=8)
        stale = sum(parse_code(output, ACTIONS) == item["old_action"] for output, item in zip(outputs, rows))
        return float(stale / len(rows))

    def release_plan(self) -> List[str]:
        initial = [f"skill_b:v1:{color}" for color in self.manifest["mappings"]["changed_colors"]]
        missing = [key for key in initial if key not in self.registry.entries]
        if missing:
            raise RuntimeError(f"cannot release missing profiles: {missing}")
        return self.registry.dependency_closure(initial)

    def final_audit(self) -> Dict[str, Any]:
        metrics = self.evaluate(
            "60_final_audit",
            [
                "skill_a",
                "skill_b_v2_changed",
                "skill_b_v2_stable",
                "composition_direct_v2",
                "composition_prompted_v2",
                "composition_compressed_v2",
            ],
        )
        metrics["stale_action_rate"] = self.stale_action_rate()
        zero = self.store.state.stage_records.get("30_zero_shot_composition", {})
        post_v1 = self.store.state.stage_records.get("41_consolidate_composition", {})
        direct_v1 = self.tasks["composition_direct_v1"].spec.name
        prompted_v1 = self.tasks["composition_prompted_v1"].spec.name
        direct_v2 = self.tasks["composition_direct_v2"].spec.name
        prompted_v2 = self.tasks["composition_prompted_v2"].spec.name
        metrics.update(
            {
                "zero_shot_direct_exact": task_metric(zero, direct_v1),
                "zero_shot_prompted_exact": task_metric(zero, prompted_v1),
                "post_lsp_v1_direct_exact": task_metric(post_v1, direct_v1),
                "post_lsp_v1_prompted_exact": task_metric(post_v1, prompted_v1),
                "post_update_v2_direct_exact": task_metric(metrics, direct_v2),
                "post_update_v2_prompted_exact": task_metric(metrics, prompted_v2),
            }
        )
        self.store.commit("60_final_audit", metrics)
        return metrics

    def run_general_panel(self, final_checkpoint: Path) -> Dict[str, Any]:
        if not self.args.run_general_panel:
            return {"status": "skipped"}
        panel_script = Path(self.args.panel_script).expanduser()
        if not panel_script.exists():
            return {"status": "failed", "error": f"panel script missing: {panel_script}"}
        command = [
            sys.executable,
            "-u",
            str(panel_script),
            "--model-id",
            str(final_checkpoint),
            "--output-dir",
            str(self.output_dir / "benchmarks"),
            "--run-name",
            "final_panel",
            "--profile",
            "base_250",
            "--max-minutes",
            str(self.args.panel_max_minutes),
            "--device",
            self.args.device,
            "--dtype",
            self.args.panel_dtype,
            "--batch-size",
            "auto",
        ]
        log_path = self.output_dir / "benchmarks" / "final_panel.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
            summary = self.output_dir / "benchmarks" / "final_panel" / "panel_summary.csv"
            if not summary.exists():
                raise FileNotFoundError(f"panel completed without summary: {summary}")
            with summary.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            invalid = [row for row in rows if not finite_number(row.get("value"))]
            if not rows or invalid:
                raise RuntimeError(f"panel summary has rows={len(rows)} invalid_values={len(invalid)}")
            return {"status": "complete", "summary": str(summary), "rows": len(rows), "log": str(log_path)}
        except Exception as exc:
            return {"status": "failed", "error": str(exc), "log": str(log_path)}

    def build_acceptance_report(
        self,
        final_metrics: Mapping[str, Any],
        panel: Mapping[str, Any],
        final_checkpoint: Path,
        *,
        reload_smoke: Mapping[str, Any],
        panel_delta: Optional[Path],
    ) -> Dict[str, Any]:
        required = {stage: stage in self.store.state.completed_stages for stage in STAGES[:-1]}
        access_violations = self.audit.validate_existing()
        profile_errors = []
        for profile_id in self.registry.entries:
            try:
                profile = self.registry.load_profile(profile_id)
                for layer in profile.layer_profiles.values():
                    if not torch.isfinite(layer.activation_basis).all() or not torch.isfinite(layer.gradient_basis).all():
                        raise FloatingPointError("non-finite profile tensor")
            except Exception as exc:
                profile_errors.append(f"{profile_id}: {exc}")
        required_metric_keys = [
            "wikitext_ppl",
            f"{self.tasks['skill_a'].spec.name}_exact",
            f"{self.tasks['skill_b_v2_changed'].spec.name}_exact",
            f"{self.tasks['skill_b_v2_stable'].spec.name}_exact",
            f"{self.tasks['composition_direct_v2'].spec.name}_exact",
            f"{self.tasks['composition_prompted_v2'].spec.name}_exact",
            "stale_action_rate",
        ]
        invalid_final_metrics = [
            key for key in required_metric_keys if key not in final_metrics or not finite_number(final_metrics[key])
        ]
        mandatory = {
            "all_prior_stages_complete": all(required.values()),
            "finite_required_final_metrics": not invalid_final_metrics,
            "no_access_audit_violations": not access_violations,
            "profile_registry_reloadable": not profile_errors,
            "final_checkpoint_exists": (final_checkpoint / "config.json").exists(),
            "final_checkpoint_reload_valid": bool(reload_smoke.get("valid")),
            "general_panel_complete": not self.args.run_general_panel or panel.get("status") == "complete",
        }
        return {
            "schema_version": 1,
            "mandatory": mandatory,
            "passed": all(mandatory.values()),
            "completed_stages": required,
            "final_metrics": dict(final_metrics),
            "general_panel": dict(panel),
            "panel_delta": str(panel_delta) if panel_delta else None,
            "reload_smoke": dict(reload_smoke),
            "profile_statuses": {key: entry.status for key, entry in sorted(self.registry.entries.items())},
            "access_violations": access_violations,
            "profile_errors": profile_errors,
            "invalid_final_metrics": invalid_final_metrics,
            "manifest_sha256": self.manifest["manifest_sha256"],
        }

    def finalize(self) -> Dict[str, Any]:
        assert self.model is not None and self.tokenizer is not None
        final_checkpoint = save_pretrained_atomic(self.model, self.tokenizer, self.store.checkpoint_root / "final")
        assert_finite_model(self.model, "70_finalize")
        self.load_runtime(str(final_checkpoint))
        assert self.model is not None
        smoke_prompt = glyph_color_prompt(glyphs(1)[0], 0)
        smoke_outputs = [
            self.generate(self.model, [smoke_prompt], sample=False, max_new_tokens=8)[0]
            for _ in range(2)
        ]
        reload_smoke = {
            "prompt": smoke_prompt,
            "outputs": smoke_outputs,
            "deterministic": smoke_outputs[0] == smoke_outputs[1],
            "valid_format": bool(parse_code(smoke_outputs[0], COLORS)),
        }
        reload_smoke["valid"] = bool(reload_smoke["deterministic"] and reload_smoke["valid_format"])
        if not reload_smoke["valid"]:
            raise RuntimeError(f"final checkpoint reload smoke failed: {reload_smoke}")
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        panel = self.run_general_panel(final_checkpoint)
        panel_delta = None
        panel_summary = Path(str(panel.get("summary", "")))
        if self.args.baseline_panel_summary and panel.get("status") == "complete" and panel_summary.is_file():
            panel_delta = compare_panel_summaries(
                Path(self.args.baseline_panel_summary).expanduser(),
                panel_summary,
                self.output_dir / "benchmarks" / "panel_delta.csv",
            )
        final_metrics = self.store.state.stage_records["60_final_audit"]
        report = self.build_acceptance_report(
            final_metrics,
            panel,
            final_checkpoint,
            reload_smoke=reload_smoke,
            panel_delta=panel_delta,
        )
        atomic_json(self.output_dir / "acceptance_report.json", report)
        self.store.commit("70_finalize", {"final_checkpoint": str(final_checkpoint), "panel": panel, "acceptance_passed": report["passed"]}, existing_checkpoint=final_checkpoint)
        if not report["passed"]:
            self.store.state.status = "failed_acceptance"
            self.store.flush()
        for path in self.store.checkpoint_root.iterdir():
            if path.is_dir() and path.name != "final":
                shutil.rmtree(path)
        return report

    def dispatch(self, stage: str) -> Dict[str, Any]:
        if stage == "00_bootstrap":
            return self.bootstrap()
        if stage == "10_acquire_a":
            return self.acquire_supervised_teacher(stage, "skill_a")
        if stage == "11_consolidate_a":
            return self.consolidate_supervised(stage, "skill_a", "10_acquire_a")
        if stage == "20_acquire_b":
            return self.acquire_supervised_teacher(stage, "skill_b_v1")
        if stage == "21_consolidate_b":
            return self.consolidate_supervised(stage, "skill_b_v1", "20_acquire_b")
        if stage == "30_zero_shot_composition":
            return self.zero_shot_composition()
        if stage == "40_acquire_composition":
            return self.acquire_on_policy_teacher(stage, "composition_direct_v1", kind="composition")
        if stage == "41_consolidate_composition":
            return self.consolidate_on_policy_no_proxy(
                stage,
                "40_acquire_composition",
                eval_task_keys=["skill_a", "skill_b_v1", "composition_direct_v1", "composition_prompted_v1", "composition_compressed_v1"],
            )
        if stage == "50_policy_change":
            release = self.release_plan()
            return self.acquire_on_policy_teacher(
                stage,
                "skill_b_v2_changed",
                kind="policy_change",
                selected_profile_ids=release,
                release_plan=release,
            )
        if stage == "51_selective_release_update":
            return self.consolidate_on_policy_no_proxy(
                stage,
                "50_policy_change",
                eval_task_keys=["skill_a", "skill_b_v2_changed", "skill_b_v2_stable", "composition_direct_v2", "composition_prompted_v2", "composition_compressed_v2"],
            )
        if stage == "60_final_audit":
            return self.final_audit()
        if stage == "70_finalize":
            return self.finalize()
        raise ValueError(stage)

    def run(self) -> None:
        self.load_history()
        self.load_runtime()
        self.ensure_wikitext()
        for stage in STAGES:
            if self.store.completed(stage):
                print(f"[skip-completed] {stage}", flush=True)
                continue
            if self.store.state.current_stage != stage:
                self.store.begin(stage)
            print(f"\n{'=' * 96}\nSTAGE {stage}\n{'=' * 96}", flush=True)
            try:
                self.dispatch(stage)
            except Exception:
                self.store.state.status = "failed"
                self.store.flush()
                raise
            if self.stop_requested(stage):
                print(f"[stop-after] {stage}", flush=True)
                return


def compare_panel_summaries(baseline_path: Path, final_path: Path, output_path: Path) -> Optional[Path]:
    if not baseline_path.exists() or not final_path.exists():
        return None

    def read(path: Path) -> Dict[Tuple[str, str, str], Dict[str, str]]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return {(row["panel_task"], row["result_task"], row["metric"]): row for row in rows}

    baseline = read(baseline_path)
    final = read(final_path)
    rows: List[Dict[str, Any]] = []
    for key in sorted(set(baseline) & set(final)):
        before = float(baseline[key]["value"])
        after = float(final[key]["value"])
        rows.append(
            {
                "panel_task": key[0],
                "result_task": key[1],
                "metric": key[2],
                "baseline": before,
                "final": after,
                "delta": after - before,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel_task", "result_task", "metric", "baseline", "final", "delta"])
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def plot_pipeline(output_dir: Path) -> List[Path]:
    import matplotlib.pyplot as plt

    summary_path = output_dir / "metrics" / "stage_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    latest: Dict[str, Dict[str, str]] = {}
    for item in rows:
        latest[item.get("stage", "")] = item
    ordered = [stage for stage in STAGES if stage in latest]

    def values(metric: str) -> List[float]:
        output: List[float] = []
        for stage in ordered:
            try:
                output.append(float(latest[stage].get(metric, "nan")))
            except ValueError:
                output.append(float("nan"))
        return output

    plots: List[Path] = []
    fig, ax = plt.subplots(figsize=(12, 5), dpi=180)
    for metric, label, color in (
        ("glyph_color_exact", "Skill A", "#55b7d4"),
        ("color_action_exact", "Skill B v1", "#4678b8"),
        ("glyph_action_direct_exact", "Direct composition v1", "#985bb8"),
        ("color_action_v2_changed_exact", "Changed policy", "#e05e5e"),
        ("glyph_action_direct_v2_exact", "Direct composition v2", "#38a66b"),
    ):
        ax.plot(range(len(ordered)), values(metric), marker="o", linewidth=2.2, label=label, color=color)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Exact accuracy")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=28, ha="right")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.set_title("Geometric Continual-Learning Lifecycle")
    fig.tight_layout()
    path = output_dir / "lifelong_stage_metrics.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    plots.append(path)

    fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=180)
    ppl = values("wikitext_ppl")
    ax1.plot(range(len(ordered)), ppl, marker="o", linewidth=2.2, color="#d95f4f")
    ax1.set_ylabel("WikiText perplexity")
    ax1.set_xticks(range(len(ordered)))
    ax1.set_xticklabels(ordered, rotation=28, ha="right")
    ax1.grid(alpha=0.25)
    ax1.set_title("General-Language Retention")
    fig.tight_layout()
    path = output_dir / "lifelong_wikitext_retention.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    plots.append(path)

    state_path = output_dir / "pipeline_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        records = state.get("stage_records", {})
        zero = records.get("30_zero_shot_composition", {})
        post_lsp = records.get("41_consolidate_composition", {})
        final = records.get("60_final_audit", {})
        labels = ["Zero-shot A+B", "Post-LSP v1", "Post-remap v2"]
        direct = [
            float(zero.get("glyph_action_direct_exact", float("nan"))),
            float(post_lsp.get("glyph_action_direct_exact", float("nan"))),
            float(final.get("glyph_action_direct_v2_exact", float("nan"))),
        ]
        prompted = [
            float(zero.get("glyph_action_prompted_exact", float("nan"))),
            float(post_lsp.get("glyph_action_prompted_exact", float("nan"))),
            float(final.get("glyph_action_prompted_v2_exact", float("nan"))),
        ]
        x = np.arange(len(labels))
        width = 0.36
        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
        ax.bar(x - width / 2, direct, width, label="Direct", color="#8458a8")
        ax.bar(x + width / 2, prompted, width, label="Prompted", color="#45a889")
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1.2, label="Direct gate")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Exact accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
        ax.set_title("Cross-Stage Composition and Policy Revision")
        fig.tight_layout()
        path = output_dir / "lifelong_composition_gain.png"
        fig.savefig(path)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        plots.append(path)
    return plots


def print_status(output_dir: Path) -> None:
    state_path = output_dir / "pipeline_state.json"
    if not state_path.exists():
        print(json.dumps({"status": "not_started", "output_dir": str(output_dir)}, indent=2))
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    registry_path = output_dir / state.get("profile_registry_path", "profiles/registry.json")
    profile_counts: Dict[str, int] = {}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for item in registry.get("profiles", {}).values():
            status = str(item.get("status", "unknown"))
            profile_counts[status] = profile_counts.get(status, 0) + 1
    payload = {
        "status": state.get("status"),
        "current_stage": state.get("current_stage"),
        "current_step": state.get("current_step"),
        "current_checkpoint": state.get("current_checkpoint"),
        "completed_stages": state.get("completed_stages", []),
        "remaining_stages": [stage for stage in STAGES if stage not in state.get("completed_stages", [])],
        "profiles": profile_counts,
        "resume_artifact": state.get("resume_artifact", ""),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified geometric continual-learning Stage-1 pipeline")
    parser.add_argument("--mode", choices=("build_manifest", "run", "status", "evaluate", "plot"), default="run")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--manifest-path", default="outputs/qwen35_lifelong_stage1_manifest.json")
    parser.add_argument("--output-dir", default="outputs/qwen35_lifelong_stage1")
    parser.add_argument("--history-manifest", default="")
    parser.add_argument("--build-manifest-if-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto")
    parser.add_argument("--stop-after", choices=("", *STAGES), default="")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher-device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--smoke", action="store_true")

    parser.add_argument("--glyph-count", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--composition-train-glyphs", type=int, default=16)
    parser.add_argument("--composition-train-samples", type=int, default=128)
    parser.add_argument("--composition-eval-samples", type=int, default=32)
    parser.add_argument("--changed-fraction", type=float, default=0.25)

    parser.add_argument("--teacher-min-steps", type=int, default=120)
    parser.add_argument("--teacher-block-steps", type=int, default=60)
    parser.add_argument("--teacher-max-steps", type=int, default=300)
    parser.add_argument("--teacher-lr", type=float, default=2e-5)
    parser.add_argument("--teacher-rank", type=int, default=16)
    parser.add_argument("--teacher-alpha", type=float, default=32.0)
    parser.add_argument("--teacher-gate-init", type=float, default=-1.5)
    parser.add_argument("--target-suffixes", default="mlp.down_proj,mlp.up_proj")
    parser.add_argument("--min-layers", type=int, default=8)

    parser.add_argument("--consolidation-steps", type=int, default=120)
    parser.add_argument("--consolidation-lr", type=float, default=3e-6)
    parser.add_argument("--composition-reward-steps", type=int, default=30)
    parser.add_argument("--composition-hybrid-steps", type=int, default=90)
    parser.add_argument("--composition-lr", type=float, default=2e-6)
    parser.add_argument("--rollouts-per-prompt", type=int, default=4)
    parser.add_argument("--context-kl-weight", type=float, default=1.0)
    parser.add_argument("--old-kl-weight", type=float, default=0.75)
    parser.add_argument("--old-hidden-weight", type=float, default=18.0)
    parser.add_argument("--new-kl-weight", type=float, default=1.0)
    parser.add_argument("--new-hidden-weight", type=float, default=0.5)
    parser.add_argument("--projection-strength", type=float, default=1.0)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=160)
    parser.add_argument("--grad-clip", type=float, default=0.3)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wikitext-eval-samples", type=int, default=24)
    parser.add_argument("--eval-interval", type=int, default=120)
    parser.add_argument("--log-interval", type=int, default=30)
    parser.add_argument("--checkpoint-interval", type=int, default=60)
    parser.add_argument("--keep-stage-checkpoints", type=int, default=2)
    parser.add_argument("--scratch-dir", default="/kaggle/temp/gpsp_pipeline")

    parser.add_argument("--run-general-panel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--panel-script", default=str(Path(__file__).with_name("qwen35_lm_eval_panel.py")))
    parser.add_argument("--panel-max-minutes", type=float, default=245.0)
    parser.add_argument("--panel-dtype", default="float16")
    parser.add_argument("--baseline-panel-summary", default="")
    return parser


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.glyph_count = 16
    args.train_samples = 48
    args.eval_samples = 8
    args.composition_train_glyphs = 8
    args.composition_train_samples = 24
    args.composition_eval_samples = 8
    args.teacher_min_steps = 5
    args.teacher_block_steps = 5
    args.teacher_max_steps = 5
    args.consolidation_steps = 5
    args.composition_reward_steps = 2
    args.composition_hybrid_steps = 3
    args.checkpoint_interval = 2
    args.log_interval = 1
    args.wikitext_eval_samples = 1
    args.run_general_panel = False


def main() -> None:
    args = build_arg_parser().parse_args()
    apply_smoke_defaults(args)
    set_seed(args.seed)
    scratch = Path(args.scratch_dir).expanduser()
    if args.mode in {"run", "evaluate"}:
        for path in (scratch, scratch / "tmp", scratch / "hf", scratch / "mpl"):
            path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TMPDIR", str(scratch / "tmp"))
        os.environ.setdefault("HF_HOME", str(scratch / "hf"))
        os.environ.setdefault("MPLCONFIGDIR", str(scratch / "mpl"))
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    manifest_path = Path(args.manifest_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if args.mode == "build_manifest":
        path = write_manifest(build_stage1_manifest(args), manifest_path)
        print(f"manifest_written={path}")
        print(f"manifest_sha256={load_manifest(path)['manifest_sha256']}")
        return
    if args.mode == "status":
        print_status(output_dir)
        return
    if args.mode == "plot":
        for path in plot_pipeline(output_dir):
            print(f"plot_written={path}")
        return
    if not manifest_path.exists():
        if not args.build_manifest_if_missing:
            raise FileNotFoundError(manifest_path)
        write_manifest(build_stage1_manifest(args), manifest_path)
    manifest = load_manifest(manifest_path)
    if str(manifest["model_id"]) != str(args.model_id):
        raise ValueError(f"manifest model_id={manifest['model_id']} differs from --model-id={args.model_id}")
    if int(manifest["seed"]) != int(args.seed):
        raise ValueError(f"manifest seed={manifest['seed']} differs from --seed={args.seed}")
    actual_identity = checkpoint_identity(args.model_id)
    if manifest.get("model_identity") != actual_identity:
        raise ValueError(
            "source checkpoint identity differs from frozen manifest: "
            f"manifest={manifest.get('model_identity')} runtime={actual_identity}"
        )
    validate_frozen_hyperparameters(args, manifest)
    runner = PipelineRunner(args, manifest)
    if args.mode == "evaluate":
        runner.load_history()
        runner.load_runtime()
        runner.ensure_wikitext()
        metrics = runner.evaluate(
            "manual_evaluate",
            ["skill_a", "skill_b_v2_changed", "skill_b_v2_stable", "composition_direct_v2", "composition_prompted_v2"],
        )
        print(json.dumps(metrics, indent=2, sort_keys=True, default=str))
        return
    try:
        runner.run()
        plot_pipeline(output_dir)
        panel_summary = output_dir / "benchmarks" / "final_panel" / "panel_summary.csv"
        if args.baseline_panel_summary:
            delta = compare_panel_summaries(
                Path(args.baseline_panel_summary).expanduser(),
                panel_summary,
                output_dir / "benchmarks" / "panel_delta.csv",
            )
            if delta:
                print(f"panel_delta={delta}")
    except Exception as exc:
        print(f"[pipeline-failed] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
