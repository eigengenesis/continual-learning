#!/usr/bin/env python3
"""Three alien benchmark continual-learning ladder for Qwen.

This script is intentionally separate from the Qwen/Gemma audit paths. It reuses
their proven geometry utilities, but owns a generic TaskSpec interface,
structured logging, stage gates, checkpoints, baselines, and an expansion branch.

Stdout is still useful, but the paper artifacts are written under:
  outputs/alien_ladder_seed{seed}/
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from datasets import get_dataset_config_names, load_dataset
except Exception:  # pragma: no cover - Kaggle/Colab normally has datasets.
    get_dataset_config_names = None
    load_dataset = None

import qwen_continual_proof as qp
from qwen_cl_desiderata_audit import project_old_occupied_gradients, select_layers
from standalone_latent_lora_qwen import (
    LatentLoRAConfig,
    attach_latent_lora,
    choose_dtype,
    default_model_id,
    load_causal_lm,
    load_tokenizer,
)


TARGET_SUFFIXES = ("mlp.down_proj", "mlp.up_proj")
MAX_GENERATION_PROMPT_LEN = 256
ALL_D_VARIANTS = ("naive_sft", "sdft_baseline", "fixed_no_proxy", "expanded_no_proxy")


def line(char: str = "=") -> None:
    print(char * 96, flush=True)


def section(title: str) -> None:
    line("=")
    print(title, flush=True)
    line("=")


def subsection(title: str) -> None:
    line("-")
    print(title, flush=True)
    line("-")


def fmt(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def now_sec() -> float:
    return time.time()


def stable_seed(label: str, offset: int = 0) -> int:
    total = int(offset)
    for idx, char in enumerate(str(label)):
        total += (idx + 1) * ord(char)
    return total


def release(*models: Optional[nn.Module]) -> None:
    for model in models:
        if model is not None:
            del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_aux_device(requested: str, primary: str) -> str:
    if requested == "same":
        return primary
    if requested == "auto":
        if primary.startswith("cuda") and torch.cuda.device_count() > 1:
            return "cuda:1"
        return primary
    return requested


def move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def stage_weight(args: argparse.Namespace, name: str, fallback: str) -> float:
    value = getattr(args, name)
    if value is None:
        value = getattr(args, fallback)
    return float(value)


def parse_d_variants(value: str) -> List[str]:
    aliases = {
        "naive": "naive_sft",
        "sft": "naive_sft",
        "naive_sft": "naive_sft",
        "sdft": "sdft_baseline",
        "sdft_baseline": "sdft_baseline",
        "fixed": "fixed_no_proxy",
        "no_proxy": "fixed_no_proxy",
        "fixed_no_proxy": "fixed_no_proxy",
        "expanded": "expanded_no_proxy",
        "expansion": "expanded_no_proxy",
        "expanded_no_proxy": "expanded_no_proxy",
    }
    raw = str(value or "all").strip().lower()
    if raw in {"all", "default", "*"}:
        return list(ALL_D_VARIANTS)
    selected: List[str] = []
    for part in raw.split(","):
        key = part.strip().lower().replace("-", "_")
        if not key:
            continue
        if key not in aliases:
            allowed = ", ".join(["all", *ALL_D_VARIANTS])
            raise ValueError(f"unknown D variant '{part.strip()}'; allowed: {allowed}")
        variant = aliases[key]
        if variant not in selected:
            selected.append(variant)
    if not selected:
        raise ValueError("--d-variants selected no variants")
    return selected


def kl_divergence_to_student_device(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    teacher_logits = teacher_logits.to(device=student_logits.device, dtype=student_logits.dtype)
    return qp.kl_divergence(student_logits, teacher_logits, temperature=temperature)


def hidden_alignment_to_student_device(student_outputs, teacher_outputs, layer_indices: Sequence[int], device: str) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for layer_index in layer_indices:
        idx = int(layer_index) + 1
        teacher_hidden = teacher_outputs.hidden_states[idx].to(
            device=student_outputs.hidden_states[idx].device,
            dtype=student_outputs.hidden_states[idx].dtype,
        )
        losses.append(F.mse_loss(student_outputs.hidden_states[idx], teacher_hidden))
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def truncate_completion(text: str) -> str:
    text = str(text)
    for stop in ("\n\n", "<|endoftext|>"):
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


def bpe_token_acc(tokenizer, prediction: str, target: str) -> float:
    pred_ids = tokenizer(prediction, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return 1.0
    pred_prefix = pred_ids[: len(target_ids)]
    correct = sum(int(a == b) for a, b in zip(pred_prefix, target_ids))
    return float(correct / len(target_ids))


def prefix_exact(prediction: str, target: str) -> float:
    pred_norm = normalize_text(truncate_completion(prediction))
    target_norm = normalize_text(target)
    if not target_norm:
        return 1.0
    return float(pred_norm == target_norm or pred_norm.startswith(target_norm))


def _capture_trainable_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _restore_trainable_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    if not state:
        return
    param_map = dict(model.named_parameters())
    for name, value in state.items():
        param = param_map.get(name)
        if param is not None:
            param.data.copy_(value.to(device=param.device, dtype=param.dtype))


def unfreeze_gated_layers(model: nn.Module) -> int:
    count = 0
    gated_cls = getattr(qp, "GatedQwenLayer", None)
    if gated_cls is None:
        return 0
    for module in model.modules():
        if isinstance(module, gated_cls):
            for param in module.parameters():
                param.requires_grad = True
                count += param.numel()
    return count


@dataclass
class AlienExample:
    prompt: str
    target: str
    source: str
    raw_target: str


@dataclass
class TaskSpec:
    name: str
    display_name: str
    prompt_template: str
    dataset_id: str = ""
    config: Optional[str] = None
    candidate_configs: Tuple[str, ...] = ()
    train_split_names: Tuple[str, ...] = ("train",)
    eval_split_names: Tuple[str, ...] = ("validation", "dev", "test")
    source_keys: Tuple[str, ...] = ("source", "question", "utterance", "command", "commands", "input", "sentence", "text", "src")
    target_keys: Tuple[str, ...] = ("target", "actions", "output", "query", "funql", "logical_form", "semantic_parse", "semantics", "parse", "program", "label")
    max_target_tokens: int = 80
    max_source_tokens: int = 128
    max_new_tokens: int = 96
    train_samples: int = 512
    eval_samples: int = 96
    exact_gate: float = 0.0
    token_gate: float = 0.0
    loss_improvement_gate: float = 0.30
    is_synthetic: bool = False
    citation: str = ""

    def format_prompt(self, source: str) -> str:
        return self.prompt_template.format(source=str(source).strip())


@dataclass
class TaskData:
    spec: TaskSpec
    train: List[AlienExample]
    eval: List[AlienExample]
    manifest: Dict[str, Any] = field(default_factory=dict)

    def make_batch(
        self,
        tokenizer,
        device: str,
        batch_size: int,
        max_seq_len: int,
        seed: int,
        include_eos: bool = True,
    ) -> Callable[[int], Dict[str, torch.Tensor]]:
        eos = tokenizer.eos_token or ""

        def _batch(step: int) -> Dict[str, torch.Tensor]:
            rng = np.random.default_rng(seed + 7919 * int(step))
            idxs = rng.integers(0, len(self.train), size=int(batch_size))
            prompts = [self.train[int(i)].prompt for i in idxs]
            targets = [
                f"{self.train[int(i)].target}{eos if include_eos else ''}"
                for i in idxs
            ]
            return qp._prepare_supervised_batch(tokenizer, prompts, targets, device, max_seq_len)

        return _batch


class ArtifactLogger:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = out_dir / "metrics.jsonl"
        self.curves_path = out_dir / "curves.csv"
        self.summary_path = out_dir / "stage_summary.csv"
        self.z_path = out_dir / "z_tomography.csv"
        self.curve_fields = [
            "time_sec",
            "stage",
            "step",
            "loss",
            "exact",
            "token_acc",
            "tf_token_acc",
            "tf_loss",
            "wikitext_ppl",
            "old_task_examples",
            "proxy_batches",
            "extra_json",
        ]
        self.summary_rows: List[Dict[str, Any]] = []
        if self.summary_path.exists():
            try:
                with self.summary_path.open("r", newline="", encoding="utf-8") as handle:
                    self.summary_rows = list(csv.DictReader(handle))
            except Exception:
                self.summary_rows = []
        self._init_csv(self.curves_path, self.curve_fields)
        self._init_csv(
            self.z_path,
            [
                "time_sec",
                "stage",
                "task",
                "layer_index",
                "layer_name",
                "gradient_norm",
                "activation_shift",
                "occupied_overlap",
                "free_overlap",
                "free_rank_estimate",
                "activation_overlap",
                "rank_pressure",
                "saturation_score",
                "learning_pressure",
                "selected",
                "total_pressure",
                "reason",
            ],
        )

    @staticmethod
    def _init_csv(path: Path, fields: Sequence[str]) -> None:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields))
            writer.writeheader()

    def write_json(self, name: str, payload: Dict[str, Any]) -> None:
        with (self.out_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)

    def log_event(self, event: str, **payload: Any) -> None:
        row = {"time_sec": now_sec(), "event": event, **payload}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def log_curve(self, stage: str, step: int, **metrics: Any) -> None:
        row = {
            "time_sec": now_sec(),
            "stage": stage,
            "step": int(step),
            "loss": metrics.get("loss", metrics.get("tf_loss", "")),
            "exact": metrics.get("exact", ""),
            "token_acc": metrics.get("token_acc", ""),
            "tf_token_acc": metrics.get("tf_token_acc", ""),
            "tf_loss": metrics.get("tf_loss", ""),
            "wikitext_ppl": metrics.get("wikitext_ppl", ""),
            "old_task_examples": metrics.get("old_task_examples", ""),
            "proxy_batches": metrics.get("proxy_batches", ""),
            "extra_json": json.dumps(
                {k: v for k, v in metrics.items() if k not in self.curve_fields},
                sort_keys=True,
                default=str,
            ),
        }
        with self.curves_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.curve_fields)
            writer.writerow(row)

    def log_stage_summary(self, stage: str, metrics: Dict[str, Any], **extras: Any) -> None:
        row = {"stage": stage, **metrics, **extras}
        self.summary_rows.append(row)
        self.log_event("stage_summary", **row)
        self.flush_stage_summary()

    def flush_stage_summary(self) -> None:
        fields: List[str] = ["stage"]
        for row in self.summary_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with self.summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in self.summary_rows:
                writer.writerow(row)

    def log_tomography(self, stage: str, task: str, tomography: Any) -> None:
        selected = set(int(v) for v in getattr(tomography, "selected_layer_indices", []))
        total_pressure = getattr(tomography, "total_pressure", "")
        reason = getattr(tomography, "selection_reason", "")
        fields = [
            "time_sec",
            "stage",
            "task",
            "layer_index",
            "layer_name",
            "gradient_norm",
            "activation_shift",
            "occupied_overlap",
            "free_overlap",
            "free_rank_estimate",
            "activation_overlap",
            "rank_pressure",
            "saturation_score",
            "learning_pressure",
            "selected",
            "total_pressure",
            "reason",
        ]
        with self.z_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            for layer in getattr(tomography, "layer_saturations", []):
                writer.writerow(
                    {
                        "time_sec": now_sec(),
                        "stage": stage,
                        "task": task,
                        "layer_index": getattr(layer, "layer_index", ""),
                        "layer_name": getattr(layer, "layer_name", ""),
                        "gradient_norm": getattr(layer, "gradient_norm", ""),
                        "activation_shift": getattr(layer, "activation_shift", ""),
                        "occupied_overlap": getattr(layer, "occupied_overlap", ""),
                        "free_overlap": getattr(layer, "free_overlap", ""),
                        "free_rank_estimate": getattr(layer, "free_rank_estimate", ""),
                        "activation_overlap": getattr(layer, "activation_overlap", ""),
                        "rank_pressure": getattr(layer, "rank_pressure", ""),
                        "saturation_score": getattr(layer, "saturation_score", ""),
                        "learning_pressure": getattr(layer, "learning_pressure", ""),
                        "selected": int(getattr(layer, "layer_index", -1) in selected),
                        "total_pressure": total_pressure,
                        "reason": reason,
                    }
                )


def _first_existing_key(row: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    lower_map = {str(key).lower(): key for key in row.keys()}
    for key in keys:
        if key in row:
            return key
        real_key = lower_map.get(key.lower())
        if real_key is not None:
            return real_key
    return None


def _pick_split(dataset: Any, candidates: Sequence[str], *, avoid_gen: bool = False) -> Any:
    if not isinstance(dataset, dict):
        return dataset
    keys = list(dataset.keys())
    lowered = {key.lower(): key for key in keys}
    for wanted in candidates:
        if wanted.lower() in lowered:
            return dataset[lowered[wanted.lower()]]
    for key in keys:
        low = key.lower()
        if avoid_gen and ("gen" in low or "generalization" in low):
            continue
        if any(name in low for name in candidates):
            return dataset[key]
    return dataset[keys[0]]


def _available_configs(dataset_id: str, preferred: Optional[str], candidates: Sequence[str]) -> List[Optional[str]]:
    configs: List[Optional[str]] = []
    if preferred:
        configs.append(preferred)
    configs.extend([cfg for cfg in candidates if cfg and cfg not in configs])
    if get_dataset_config_names is not None:
        try:
            for cfg in get_dataset_config_names(dataset_id):
                if cfg not in configs:
                    configs.append(cfg)
        except Exception:
            pass
    if not configs:
        configs.append(None)
    return configs


def _row_to_example(row: Dict[str, Any], spec: TaskSpec) -> Optional[Tuple[str, str]]:
    source_key = _first_existing_key(row, spec.source_keys)
    target_key = _first_existing_key(row, spec.target_keys)
    if source_key is None or target_key is None:
        return None
    source = row[source_key]
    target = row[target_key]
    if isinstance(source, (list, tuple)):
        source = " ".join(str(x) for x in source)
    if isinstance(target, (list, tuple)):
        target = " ".join(str(x) for x in target)
    source = str(source).strip()
    target = str(target).strip()
    if not source or not target:
        return None
    return source, target


def _filter_and_format(
    rows: Iterable[Dict[str, Any]],
    spec: TaskSpec,
    tokenizer,
    limit: int,
) -> List[AlienExample]:
    examples: List[AlienExample] = []
    for raw in rows:
        item = _row_to_example(dict(raw), spec)
        if item is None:
            continue
        source, target = item
        source_ids = tokenizer(source, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        if len(source_ids) > spec.max_source_tokens or len(target_ids) > spec.max_target_tokens:
            continue
        examples.append(
            AlienExample(
                prompt=spec.format_prompt(source),
                target=target,
                source=source,
                raw_target=target,
            )
        )
        if len(examples) >= limit:
            break
    return examples


def load_hf_task(spec: TaskSpec, tokenizer, seed: int) -> TaskData:
    if load_dataset is None:
        raise RuntimeError("datasets is not installed; use --smoke or install datasets.")
    errors: List[str] = []
    configs = _available_configs(spec.dataset_id, spec.config, spec.candidate_configs)
    for config in configs:
        try:
            dataset = load_dataset(spec.dataset_id, config) if config else load_dataset(spec.dataset_id)
            avoid_gen = spec.name == "cogs"
            train_split = _pick_split(dataset, spec.train_split_names, avoid_gen=avoid_gen)
            eval_split = _pick_split(dataset, spec.eval_split_names, avoid_gen=avoid_gen)
            rng = np.random.default_rng(seed)
            train_rows = list(train_split)
            eval_rows = list(eval_split)
            rng.shuffle(train_rows)
            rng.shuffle(eval_rows)
            train = _filter_and_format(train_rows, spec, tokenizer, spec.train_samples)
            eval_examples = _filter_and_format(eval_rows, spec, tokenizer, spec.eval_samples)
            if train and eval_examples:
                manifest = {
                    "name": spec.name,
                    "dataset_id": spec.dataset_id,
                    "config": config,
                    "train_examples": len(train),
                    "eval_examples": len(eval_examples),
                    "citation": spec.citation,
                    "max_target_tokens": spec.max_target_tokens,
                }
                return TaskData(spec=spec, train=train, eval=eval_examples, manifest=manifest)
            errors.append(f"{config}: empty after filtering")
        except Exception as exc:
            errors.append(f"{config}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"could not load {spec.name} from {spec.dataset_id}: {' | '.join(errors[:6])}")


def load_proof_v2_record_task(tokenizer, seed: int, train_samples: int, eval_samples: int) -> TaskData:
    spec = TaskSpec(
        name="proof_v2_record",
        display_name="proof_v2 record routing",
        prompt_template="{source}",
        max_target_tokens=64,
        max_source_tokens=128,
        max_new_tokens=80,
        train_samples=train_samples,
        eval_samples=eval_samples,
        exact_gate=0.50,
        token_gate=0.60,
        is_synthetic=True,
        citation="Internal proof_v2 routing fallback.",
    )
    rng = np.random.default_rng(seed)

    def build(count: int, heldout: bool) -> List[AlienExample]:
        out: List[AlienExample] = []
        for _ in range(count):
            prompt, target = qp._proof_v2_record_example(rng, heldout=heldout)
            out.append(AlienExample(prompt=prompt, target=target, source=prompt, raw_target=target))
        return out

    return TaskData(
        spec=spec,
        train=build(train_samples, heldout=False),
        eval=build(eval_samples, heldout=True),
        manifest={
            "name": spec.name,
            "dataset_id": "synthetic:proof_v2_record",
            "train_examples": train_samples,
            "eval_examples": eval_samples,
            "citation": spec.citation,
        },
    )


def _eval_listops(expr: Any) -> int:
    if isinstance(expr, int):
        return expr
    if isinstance(expr, str):
        expr = expr.replace("(", " ( ").replace(")", " ) ").split()
    token = expr.pop(0)
    if token == "(":
        op = expr.pop(0)
        values: List[int] = []
        while expr[0] != ")":
            values.append(_eval_listops(expr))
        expr.pop(0)
        if op == "MAX":
            return max(values)
        if op == "MIN":
            return min(values)
        if op == "SM":
            return int(sum(values) % 10)
        if op == "MED":
            values = sorted(values)
            return values[len(values) // 2]
        raise ValueError(op)
    return int(token)


def load_synthetic_listops_task(seed: int, train_samples: int, eval_samples: int) -> TaskData:
    spec = TaskSpec(
        name="listops64",
        display_name="ListOps-64 fallback",
        prompt_template="LISTOPS TASK\nExpression: {source}\nValue: ",
        max_target_tokens=8,
        max_source_tokens=96,
        max_new_tokens=16,
        train_samples=train_samples,
        eval_samples=eval_samples,
        token_gate=0.75,
        exact_gate=0.50,
        is_synthetic=True,
        citation="ListOps-style bounded synthetic fallback.",
    )
    rng = np.random.default_rng(seed)
    ops = ("MAX", "MIN", "SM", "MED")

    def expr(depth: int) -> str:
        if depth <= 0 or rng.random() < 0.35:
            return str(int(rng.integers(0, 10)))
        arity = int(rng.integers(2, 5))
        children = " ".join(expr(depth - 1) for _ in range(arity))
        return f"( {ops[int(rng.integers(0, len(ops)))]} {children} )"

    def build(count: int) -> List[AlienExample]:
        out: List[AlienExample] = []
        while len(out) < count:
            source = expr(3)
            target = str(_eval_listops(source))
            out.append(AlienExample(prompt=spec.format_prompt(source), target=target, source=source, raw_target=target))
        return out

    return TaskData(
        spec=spec,
        train=build(train_samples),
        eval=build(eval_samples),
        manifest={
            "name": spec.name,
            "dataset_id": "synthetic:listops64",
            "train_examples": train_samples,
            "eval_examples": eval_samples,
            "citation": spec.citation,
        },
    )


def build_task_specs(args: argparse.Namespace) -> Tuple[TaskSpec, TaskSpec, TaskSpec]:
    scan = TaskSpec(
        name="scan",
        display_name="SCAN",
        dataset_id="Punchwe/SCAN_MCDSplits",
        config=args.scan_config,
        # MCD splits are compositional generalization stress tests. For the
        # flagship CL ladder, use a safer standard/simple SCAN skill unless the
        # caller explicitly requests MCD with --scan-config.
        candidate_configs=("simple", "length", "mcd1", "mcd2", "mcd3"),
        prompt_template="SCAN TASK\nCommand: {source}\nActions: ",
        source_keys=("commands", "command", "source", "input", "text"),
        target_keys=("actions", "action", "target", "output"),
        max_target_tokens=args.scan_max_target_tokens,
        max_source_tokens=96,
        max_new_tokens=96,
        train_samples=args.task_train_samples,
        eval_samples=args.task_eval_samples,
        token_gate=0.90,
        exact_gate=0.50,
        citation="Lake and Baroni, 2018, SCAN.",
    )
    cogs = TaskSpec(
        name="cogs",
        display_name="COGS filtered",
        dataset_id="GWHed/cogs",
        config=args.cogs_config,
        candidate_configs=("default",),
        train_split_names=("train",),
        eval_split_names=("dev", "validation", "test"),
        prompt_template="COGS TASK\nSentence: {source}\nLogical form: ",
        source_keys=("sentence", "source", "input", "text"),
        target_keys=("logical_form", "semantic_parse", "semantics", "target", "output", "parse"),
        max_target_tokens=args.cogs_max_target_tokens,
        max_source_tokens=96,
        max_new_tokens=128,
        train_samples=args.task_train_samples,
        eval_samples=args.task_eval_samples,
        token_gate=0.60,
        exact_gate=0.20,
        loss_improvement_gate=0.30,
        citation="Kim and Linzen, 2020, COGS.",
    )
    geo = TaskSpec(
        name="geoquery",
        display_name="GeoQuery",
        dataset_id="GWHed/geoquery",
        config=args.geoquery_config,
        candidate_configs=("standard", "funql", "default"),
        prompt_template="GEOQUERY TASK\nQuestion: {source}\nFunQL: ",
        source_keys=("question", "utterance", "source", "input", "sentence", "text"),
        target_keys=("funql", "query", "logical_form", "program", "target", "output"),
        max_target_tokens=args.geoquery_max_target_tokens,
        max_source_tokens=128,
        max_new_tokens=128,
        train_samples=args.task_train_samples,
        eval_samples=args.task_eval_samples,
        token_gate=0.65,
        exact_gate=0.20,
        loss_improvement_gate=0.30,
        citation="GeoQuery semantic parsing benchmark.",
    )
    return scan, cogs, geo


def load_c_task_with_fallback(args: argparse.Namespace, tokenizer, logger: ArtifactLogger) -> TaskData:
    _, cogs_spec, _ = build_task_specs(args)
    if args.force_c_task == "listops":
        task = load_synthetic_listops_task(args.seed + 202, args.task_train_samples, args.task_eval_samples)
        logger.log_event("c_task_loaded", selected=task.spec.name, forced=True, synthetic=True, manifest=task.manifest)
        return task
    if args.force_c_task == "proof_v2_record":
        task = load_proof_v2_record_task(tokenizer, args.seed + 203, args.task_train_samples, args.task_eval_samples)
        logger.log_event("c_task_loaded", selected=task.spec.name, forced=True, synthetic=True, manifest=task.manifest)
        return task
    if args.force_c_task != "auto" and args.force_c_task != "cogs":
        raise ValueError(f"unknown --force-c-task {args.force_c_task!r}")

    attempts: List[str] = []
    try:
        task = load_hf_task(cogs_spec, tokenizer, args.seed + 200)
        logger.log_event("c_task_loaded", selected=task.spec.name, fallback=False, manifest=task.manifest)
        return task
    except Exception as exc:
        attempts.append(f"COGS failed: {exc}")
        logger.log_event("c_task_load_failed", task="cogs", error=str(exc))

    listops_spec = TaskSpec(
        name="listops64",
        display_name="ListOps-64",
        dataset_id="fengyang0317/listops-64",
        prompt_template="LISTOPS TASK\nExpression: {source}\nValue: ",
        source_keys=("source", "input", "text", "expression"),
        target_keys=("target", "label", "output", "value"),
        max_target_tokens=8,
        max_source_tokens=96,
        max_new_tokens=16,
        train_samples=args.task_train_samples,
        eval_samples=args.task_eval_samples,
        token_gate=0.75,
        exact_gate=0.50,
        citation="ListOps fallback.",
    )
    try:
        task = load_hf_task(listops_spec, tokenizer, args.seed + 201)
        logger.log_event("c_task_loaded", selected=task.spec.name, fallback=True, manifest=task.manifest)
        return task
    except Exception as exc:
        attempts.append(f"ListOps HF failed: {exc}")
        logger.log_event("c_task_load_failed", task="listops64_hf", error=str(exc))

    if args.allow_synthetic_c_fallback:
        if args.synthetic_c_fallback == "listops":
            task = load_synthetic_listops_task(args.seed + 202, args.task_train_samples, args.task_eval_samples)
        else:
            task = load_proof_v2_record_task(tokenizer, args.seed + 203, args.task_train_samples, args.task_eval_samples)
        logger.log_event(
            "c_task_loaded",
            selected=task.spec.name,
            fallback=True,
            synthetic=True,
            attempts=attempts,
            manifest=task.manifest,
        )
        return task
    raise RuntimeError("C task failed and synthetic fallback is disabled: " + " | ".join(attempts))


@torch.no_grad()
def teacher_forced_metrics(model, tokenizer, task: TaskData, device: str, max_seq_len: int, eval_batch_size: int) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    correct = 0
    total = 0
    examples = task.eval
    eos = tokenizer.eos_token or ""
    for start in range(0, len(examples), eval_batch_size):
        chunk = examples[start : start + eval_batch_size]
        batch = qp._prepare_supervised_batch(
            tokenizer,
            [ex.prompt for ex in chunk],
            [ex.target + eos for ex in chunk],
            device,
            max_seq_len,
        )
        outputs = model(**batch, use_cache=False)
        losses.append(float(outputs.loss.item()))
        logits = outputs.logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        mask = labels != -100
        if mask.any():
            preds = logits.argmax(dim=-1)
            correct += int((preds[mask] == labels[mask]).sum().item())
            total += int(mask.sum().item())
    mean_loss = float(sum(losses) / max(len(losses), 1))
    return {
        f"{task.spec.name}_tf_loss": mean_loss,
        f"{task.spec.name}_tf_ppl": float(math.exp(min(mean_loss, 20.0))),
        f"{task.spec.name}_tf_token_acc": float(correct / max(total, 1)),
    }


@torch.no_grad()
def generation_metrics(model, tokenizer, task: TaskData, device: str, eval_batch_size: int) -> Dict[str, float]:
    model.eval()
    exacts: List[float] = []
    token_accs: List[float] = []
    for start in range(0, len(task.eval), eval_batch_size):
        chunk = task.eval[start : start + eval_batch_size]
        prompts = [ex.prompt for ex in chunk]
        completions = qp._generate_batch_tokens(
            model,
            tokenizer,
            prompts,
            device,
            max_new_tokens=task.spec.max_new_tokens,
        )
        for completion, ex in zip(completions, chunk):
            exacts.append(prefix_exact(completion, ex.target))
            token_accs.append(bpe_token_acc(tokenizer, truncate_completion(completion), ex.target))
    return {
        f"{task.spec.name}_exact": float(sum(exacts) / max(len(exacts), 1)),
        f"{task.spec.name}_token_acc": float(sum(token_accs) / max(len(token_accs), 1)),
    }


def evaluate_task(
    model,
    tokenizer,
    task: TaskData,
    cfg: qp.RuntimeConfig,
    *,
    do_generation: bool = True,
) -> Dict[str, float]:
    metrics = teacher_forced_metrics(model, tokenizer, task, cfg.device, cfg.max_seq_len, cfg.eval_batch_size)
    if do_generation:
        metrics.update(generation_metrics(model, tokenizer, task, cfg.device, cfg.eval_batch_size))
    return metrics


def evaluate_suite(
    model,
    tokenizer,
    tasks: Sequence[TaskData],
    cfg: qp.RuntimeConfig,
    *,
    do_generation: bool = True,
    include_wikitext: bool = True,
    wikitext_val: Optional[List[torch.Tensor]] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    _wikitext = wikitext_val or getattr(cfg, '_wikitext_val', None)
    if include_wikitext and _wikitext is not None:
        try:
            metrics.update(qp.evaluate_retention(model, tokenizer, _wikitext, cfg.device, cfg.eval_batch_size))
        except Exception as exc:
            metrics["wikitext_error"] = str(exc)
    for task in tasks:
        metrics.update(evaluate_task(model, tokenizer, task, cfg, do_generation=do_generation))
    return metrics


def focus_metrics(metrics: Dict[str, float], task: TaskData) -> Dict[str, float]:
    name = task.spec.name
    return {
        "exact": float(metrics.get(f"{name}_exact", float("nan"))),
        "token_acc": float(metrics.get(f"{name}_token_acc", float("nan"))),
        "tf_token_acc": float(metrics.get(f"{name}_tf_token_acc", float("nan"))),
        "tf_loss": float(metrics.get(f"{name}_tf_loss", float("nan"))),
    }


def score_focus(metrics: Dict[str, float], task: TaskData) -> Tuple[float, float, float, float]:
    f = focus_metrics(metrics, task)
    return (
        -math.inf if math.isnan(f["token_acc"]) else f["token_acc"],
        -math.inf if math.isnan(f["exact"]) else f["exact"],
        -math.inf if math.isnan(f["tf_token_acc"]) else f["tf_token_acc"],
        math.inf if math.isnan(f["tf_loss"]) else -f["tf_loss"],
    )


def require_generated_acquisition(stage: str, metrics: Dict[str, float], task: TaskData) -> None:
    """Hard gate expensive stages on hybrid acquisition, not just generated token acc.

    Teacher-forced scores can look excellent when the model has learned local
    next-token logits but still cannot autoregressively emit the task output.
    However, strict free-generation gating can be too brittle for symbolic datasets.
    We now use a hybrid approach that allows TF token acc to satisfy the gate.
    """
    f = focus_metrics(metrics, task)
    token_acc = 0.0 if math.isnan(f["token_acc"]) else f["token_acc"]
    exact = 0.0 if math.isnan(f["exact"]) else f["exact"]
    tf_token_acc = 0.0 if math.isnan(f["tf_token_acc"]) else f["tf_token_acc"]
    hybrid_acc = max(token_acc, tf_token_acc)
    
    ok = hybrid_acc >= float(task.spec.token_gate) or exact >= float(task.spec.exact_gate)
    if ok:
        print(
            f"[gate:{stage}] PASS hybrid_acquisition "
            f"{task.spec.name}_hybrid={fmt(hybrid_acc)} (gen={fmt(token_acc)} tf={fmt(tf_token_acc)}) "
            f"need hybrid>={task.spec.token_gate:.3f} or exact>={task.spec.exact_gate:.3f}",
            flush=True,
        )
        return
    raise RuntimeError(
        f"{stage}: hybrid acquisition gate failed for {task.spec.name}: "
        f"hybrid_acc={hybrid_acc:.4f} (gen={token_acc:.4f}, tf={tf_token_acc:.4f}), exact={exact:.4f}, "
        f"need hybrid>={task.spec.token_gate:.3f} or exact>={task.spec.exact_gate:.3f}. "
        "Stop and tune the task interface/teacher before running consolidation."
    )


def _trainable_full_student(student, cfg: qp.RuntimeConfig) -> List[nn.Parameter]:
    qp._unfreeze_model(student)
    if cfg.device.startswith("cuda"):
        qp._configure_gradient_checkpointing(student, cfg.gradient_checkpointing)
    params = qp._trainable_params(student)
    if not params:
        raise RuntimeError("student has no trainable parameters")
    return params


def select_layers_generic(
    *,
    model,
    tokenizer,
    task: TaskData,
    protected_profiles: Sequence[Any],
    cfg: qp.RuntimeConfig,
    min_layers: int,
    stage: str,
    logger: ArtifactLogger,
) -> Tuple[List[int], Any]:
    batch_fn = task.make_batch(tokenizer, cfg.device, cfg.batch_size, cfg.max_seq_len, cfg.seed + 1700)
    # Use the proven select_layers from the Qwen audit which handles tomography internally
    layers, tomography = select_layers(
        model=model,
        tokenizer=tokenizer,
        task_name=task.spec.name,
        task_batch_fn=batch_fn,
        protected_profiles=list(protected_profiles),
        cfg=cfg,
    )
    # Ensure minimum layer count
    if len(layers) < min_layers:
        seen = set(layers)
        all_layer_count = len(list(model.model.layers)) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 24
        for idx in range(all_layer_count):
            if idx not in seen:
                layers.append(idx)
                seen.add(idx)
            if len(layers) >= min_layers:
                break
    print(
        f"[z_tomography:{stage}:{task.spec.name}] selected_layers={layers}",
        flush=True,
    )
    try:
        logger.log_tomography(stage, task.spec.name, tomography)
    except Exception:
        pass
    return layers, tomography


def collect_profile(model, tokenizer, task: TaskData, cfg: qp.RuntimeConfig, label: str) -> Any:
    batch_fn = task.make_batch(tokenizer, cfg.device, cfg.batch_size, cfg.max_seq_len, cfg.seed + 2300)
    del tokenizer
    return qp._collect_profiles(model, label, batch_fn)


def train_adapter_teacher(
    *,
    model,
    tokenizer,
    task: TaskData,
    active_eval_tasks: Sequence[TaskData],
    cfg: qp.RuntimeConfig,
    logger: ArtifactLogger,
    stage: str,
    selected_layers: Sequence[int],
    steps: int,
    lr: float,
    rank: int,
    alpha: float,
    gate_init: float,
    eval_interval: int,
    train_gated_layers: bool = False,
) -> Dict[str, float]:
    qp._freeze_model(model)
    lora_cfg = LatentLoRAConfig(
        rank=int(rank),
        alpha=float(alpha),
        dropout=0.0,
        projection_strength=1.0,
        gate_init=float(gate_init),
        freeze_base=True,
    )
    attached = attach_latent_lora(
        model,
        suffixes=TARGET_SUFFIXES,
        layer_indices=set(int(v) for v in selected_layers),
        config=lora_cfg,
    )
    gated_params = unfreeze_gated_layers(model) if train_gated_layers else 0
    params = qp._trainable_params(model)
    if not params:
        raise RuntimeError(f"{stage}: no trainable params after attaching LoRA")
    optimizer = torch.optim.AdamW(params, lr=float(lr), foreach=False)
    if cfg.device.startswith("cuda"):
        qp._configure_gradient_checkpointing(model, cfg.gradient_checkpointing)
    batch_fn = task.make_batch(tokenizer, cfg.device, cfg.batch_size, cfg.max_seq_len, cfg.seed + stable_seed(stage, 3100))
    best_state: Dict[str, torch.Tensor] = {}
    best_score: Tuple[float, float, float, float] = (-math.inf, -math.inf, -math.inf, -math.inf)
    best_step = 0
    start = time.time()
    print(f"[{stage}] attached_modules={len(attached)} train_gated_params={gated_params}", flush=True)
    model.train()
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = batch_fn(step)
        losses: List[float] = []
        split = qp._split_tensor_batch(batch, cfg.consolidation_micro_batch_size)
        for micro in split:
            outputs = model(**micro, use_cache=False)
            loss = outputs.loss / max(1, len(split))
            loss.backward()
            losses.append(float(outputs.loss.item()))
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        mean_loss = float(sum(losses) / max(len(losses), 1))
        if step % cfg.log_interval == 0 or step == steps:
            print(f"[{stage}] step={step:04d}/{steps} loss={mean_loss:.4f}", flush=True)
            logger.log_curve(stage, step, loss=mean_loss)
        if step % int(eval_interval) == 0 or step == steps:
            eval_metrics = evaluate_task(model, tokenizer, task, cfg, do_generation=True)
            focus = focus_metrics(eval_metrics, task)
            logger.log_curve(stage, step, **focus)
            score = score_focus(eval_metrics, task)
            if score > best_score:
                best_score = score
                best_state = _capture_trainable_state(model)
                best_step = step
                print(
                    f"[{stage}] best_update step={step:04d}/{steps} "
                    f"score={tuple(round(x, 4) for x in score)} "
                    f"token_acc={fmt(focus['token_acc'])} exact={fmt(focus['exact'])} "
                    f"tf={fmt(focus['tf_token_acc'])}",
                    flush=True,
                )
    if best_state:
        _restore_trainable_state(model, best_state)
        print(f"[{stage}] restored_best_step={best_step:04d}/{steps}", flush=True)
    final_metrics = evaluate_suite(model, tokenizer, active_eval_tasks, cfg, do_generation=True, include_wikitext=True)
    print_metrics(stage, final_metrics, [t.spec.name for t in active_eval_tasks])
    print(f"[{stage}] wall_time_sec={time.time() - start:.1f}", flush=True)
    logger.log_stage_summary(stage, final_metrics, wall_time_sec=time.time() - start, attached_modules=len(attached))
    return final_metrics


def consolidate_no_proxy(
    *,
    student,
    teacher_old,
    teacher_new,
    tokenizer,
    task: TaskData,
    active_eval_tasks: Sequence[TaskData],
    selected_layers: Sequence[int],
    old_profiles: Sequence[Any],
    cfg: qp.RuntimeConfig,
    logger: ArtifactLogger,
    stage: str,
    steps: int,
    lr: float,
    old_kl_weight: float,
    old_hidden_weight: float,
    new_kl_weight: float,
    new_hidden_weight: float,
    project_gradients: bool,
    projection_strength: float,
    teacher_device: str,
) -> Dict[str, float]:
    qp._freeze_model(teacher_old)
    qp._freeze_model(teacher_new)
    teacher_old.to(teacher_device)
    teacher_new.to(teacher_device)
    params = _trainable_full_student(student, cfg)
    optimizer = torch.optim.AdamW(params, lr=float(lr), foreach=False)
    batch_fn = task.make_batch(tokenizer, cfg.device, cfg.batch_size, cfg.max_seq_len, cfg.seed + stable_seed(stage, 4100))
    start = time.time()
    student.train()
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = batch_fn(step)
        split = qp._split_tensor_batch(batch, cfg.consolidation_micro_batch_size)
        loss_value = 0.0
        for micro in split:
            student_outputs = student(**micro, output_hidden_states=True, use_cache=False)
            teacher_micro = move_batch(micro, teacher_device)
            with torch.no_grad():
                old_outputs = teacher_old(**teacher_micro, output_hidden_states=True, use_cache=False)
                new_outputs = teacher_new(**teacher_micro, output_hidden_states=True, use_cache=False)
            new_kl = kl_divergence_to_student_device(student_outputs.logits, new_outputs.logits)
            old_kl = kl_divergence_to_student_device(student_outputs.logits, old_outputs.logits)
            new_hidden = hidden_alignment_to_student_device(
                student_outputs,
                new_outputs,
                list(selected_layers),
                micro["input_ids"].device,
            )
            old_hidden = hidden_alignment_to_student_device(
                student_outputs,
                old_outputs,
                list(selected_layers),
                micro["input_ids"].device,
            )
            loss = (
                student_outputs.loss
                + float(new_kl_weight) * new_kl
                + float(new_hidden_weight) * new_hidden
                + float(old_kl_weight) * old_kl
                + float(old_hidden_weight) * old_hidden
            ) / max(1, len(split))
            loss.backward()
            loss_value += float(loss.item())
        projected_modules = 0
        if project_gradients:
            projected_modules = project_old_occupied_gradients(
                student,
                list(old_profiles),
                list(selected_layers),
                strength=float(projection_strength),
            )
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        if step % cfg.log_interval == 0 or step == steps:
            print(
                f"[{stage}] step={step:04d}/{steps} old_task_examples=0 proxy_batches=0 "
                f"same_batch_old_kl={old_kl_weight:.3f} same_batch_old_hidden={old_hidden_weight:.3f} "
                f"new_kl={new_kl_weight:.3f} new_hidden={new_hidden_weight:.3f} "
                f"teacher_device={teacher_device} projected_modules={projected_modules}",
                flush=True,
            )
            logger.log_curve(
                stage,
                step,
                loss=loss_value,
                old_task_examples=0,
                proxy_batches=0,
                projected_modules=projected_modules,
            )
    metrics = evaluate_suite(student, tokenizer, active_eval_tasks, cfg, do_generation=True, include_wikitext=True)
    print_metrics(stage, metrics, [t.spec.name for t in active_eval_tasks])
    print(f"[{stage}] wall_time_sec={time.time() - start:.1f}", flush=True)
    logger.log_stage_summary(
        stage,
        metrics,
        wall_time_sec=time.time() - start,
        old_task_examples=0,
        proxy_batches=0,
        method="no_proxy",
    )
    return metrics


def consolidate_naive(
    *,
    student,
    tokenizer,
    task: TaskData,
    active_eval_tasks: Sequence[TaskData],
    cfg: qp.RuntimeConfig,
    logger: ArtifactLogger,
    stage: str,
    steps: int,
    lr: float,
) -> Dict[str, float]:
    params = _trainable_full_student(student, cfg)
    optimizer = torch.optim.AdamW(params, lr=float(lr), foreach=False)
    batch_fn = task.make_batch(tokenizer, cfg.device, cfg.batch_size, cfg.max_seq_len, cfg.seed + stable_seed(stage, 5100))
    start = time.time()
    student.train()
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = batch_fn(step)
        split = qp._split_tensor_batch(batch, cfg.consolidation_micro_batch_size)
        loss_value = 0.0
        for micro in split:
            outputs = student(**micro, use_cache=False)
            loss = outputs.loss / max(1, len(split))
            loss.backward()
            loss_value += float(loss.item())
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        if step % cfg.log_interval == 0 or step == steps:
            print(f"[{stage}] step={step:04d}/{steps} loss={loss_value:.4f} method=naive_sft protection=none", flush=True)
            logger.log_curve(stage, step, loss=loss_value, old_task_examples=0, proxy_batches=0)
    metrics = evaluate_suite(student, tokenizer, active_eval_tasks, cfg, do_generation=True, include_wikitext=True)
    print_metrics(stage, metrics, [t.spec.name for t in active_eval_tasks])
    logger.log_stage_summary(stage, metrics, wall_time_sec=time.time() - start, old_task_examples=0, proxy_batches=0, method="naive_sft")
    print(f"[{stage}] wall_time_sec={time.time() - start:.1f}", flush=True)
    return metrics


def _prompts_from_supervised_batch(tokenizer, batch: Dict[str, torch.Tensor]) -> List[str]:
    prompts: List[str] = []
    labels = batch["labels"]
    input_ids = batch["input_ids"]
    for row_idx in range(input_ids.shape[0]):
        mask = labels[row_idx] == -100
        prompt_len = int(mask.sum().item())
        if prompt_len <= 0:
            # Very long targets can consume the whole supervised sequence. Keep
            # on-policy generation from handing an empty prompt to decoder-only
            # models, which can crash some Qwen3.5 linear-attention paths.
            attention_mask = batch.get("attention_mask")
            valid_len = int(attention_mask[row_idx].sum().item()) if attention_mask is not None else int(input_ids.shape[1])
            prompt_ids = input_ids[row_idx, : max(1, min(valid_len, 1))]
        else:
            prompt_ids = input_ids[row_idx, :prompt_len]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False).strip()
        prompts.append(prompt_text if prompt_text else (tokenizer.bos_token or tokenizer.eos_token or " "))
    return prompts


@torch.no_grad()
def generate_on_policy_completions(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: str,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> List[str]:
    if not prompts:
        return []
    prompts = [prompt if str(prompt).strip() else (tokenizer.bos_token or tokenizer.eos_token or " ") for prompt in prompts]
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_GENERATION_PROMPT_LEN,
        ).to(device)
    finally:
        tokenizer.padding_side = original_padding_side
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": False,
    }
    if do_sample:
        kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
    with torch.backends.cudnn.flags(enabled=False):
        outputs = model.generate(**inputs, **kwargs)
    prompt_width = int(inputs["input_ids"].shape[1])
    return [
        tokenizer.decode(outputs[row_idx, prompt_width:], skip_special_tokens=True).strip()
        for row_idx in range(len(prompts))
    ]


def _prepare_completion_kl_batch(
    tokenizer,
    prompts: Sequence[str],
    completions: Sequence[str],
    device: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """Prepare fixed-width prompt+completion inputs with completion masks.

    Student and teacher prompts can have different lengths, but the same
    completion is right-aligned in both batches. This lets us compare their
    per-token distributions over completion positions only.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    eos_id = tokenizer.eos_token_id
    rows: List[List[int]] = []
    masks: List[List[int]] = []
    comp_masks: List[List[int]] = []
    fixed_len = max(2, int(max_length))
    for prompt, completion in zip(prompts, completions):
        prompt_ids = tokenizer(str(prompt), add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(str(completion), add_special_tokens=False)["input_ids"]
        if not completion_ids:
            completion_ids = [int(eos_id if eos_id is not None else pad_id)]
        if len(completion_ids) >= fixed_len:
            completion_ids = completion_ids[:fixed_len]
            prompt_tail: List[int] = []
        else:
            budget = fixed_len - len(completion_ids)
            prompt_tail = prompt_ids[-budget:] if budget else []
        ids = prompt_tail + completion_ids
        token_mask = [1] * len(ids)
        completion_mask = [0] * len(prompt_tail) + [1] * len(completion_ids)
        pad_len = fixed_len - len(ids)
        rows.append([int(pad_id)] * pad_len + ids)
        masks.append([0] * pad_len + token_mask)
        comp_masks.append([0] * pad_len + completion_mask)
    return {
        "input_ids": torch.tensor(rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
        "completion_mask": torch.tensor(comp_masks, dtype=torch.long, device=device),
    }


def _completion_forward_kl_loss(student_outputs, teacher_outputs, student_completion_mask: torch.Tensor) -> torch.Tensor:
    student_logits = student_outputs.logits[:, :-1, :]
    teacher_logits = teacher_outputs.logits[:, :-1, :].to(device=student_logits.device, dtype=student_logits.dtype)
    mask = student_completion_mask[:, 1:].to(device=student_logits.device, dtype=student_logits.dtype)
    if not torch.any(mask > 0):
        return student_logits.sum() * 0.0
    student_logps = F.log_softmax(student_logits, dim=-1)
    teacher_logps = F.log_softmax(teacher_logits, dim=-1)
    per_vocab = F.kl_div(student_logps, teacher_logps, reduction="none", log_target=True)
    per_token = per_vocab.sum(dim=-1)
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def _demo_conditioned_teacher_prompt(prompt: str, target: str) -> str:
    return (
        f"{str(prompt).strip()}\n\n"
        "This is an example for a response to the question:\n"
        f"{str(target).strip()}\n\n"
        "Now answer with a response of your own, including the thinking process.\n"
    )


def _sample_task_texts(task: TaskData, seed: int, step: int, batch_size: int) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed + 7919 * int(step))
    idxs = rng.integers(0, len(task.train), size=int(batch_size))
    prompts = [task.train[int(i)].prompt for i in idxs]
    targets = [task.train[int(i)].target for i in idxs]
    return prompts, targets


def _sync_ref_model(student, ref_model, alpha: float) -> None:
    with torch.no_grad():
        for student_param, ref_param in zip(student.parameters(), ref_model.parameters()):
            ref_param.data.mul_(1.0 - float(alpha)).add_(student_param.data.to(ref_param.device), alpha=float(alpha))


def sdft_forward_kl_loss(student_outputs, teacher_outputs, labels: torch.Tensor) -> torch.Tensor:
    student_logits = student_outputs.logits[:, :-1, :]
    teacher_logits = teacher_outputs.logits[:, :-1, :].to(
        device=student_logits.device,
        dtype=student_logits.dtype,
    )
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    if not mask.any():
        return student_logits.sum() * 0.0
    student_logps = F.log_softmax(student_logits, dim=-1)
    teacher_logps = F.log_softmax(teacher_logits, dim=-1)
    per_vocab = F.kl_div(student_logps, teacher_logps, reduction="none", log_target=True)
    per_token = per_vocab.sum(dim=-1)
    return (per_token * mask).sum() / mask.sum().clamp(min=1)


def consolidate_sdft(
    *,
    student,
    teacher_new,
    tokenizer,
    task: TaskData,
    active_eval_tasks: Sequence[TaskData],
    cfg: qp.RuntimeConfig,
    logger: ArtifactLogger,
    stage: str,
    steps: int,
    lr: float,
    teacher_device: str,
    loss_type: str,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Dict[str, float]:
    qp._freeze_model(teacher_new)
    teacher_new.to(teacher_device)
    params = _trainable_full_student(student, cfg)
    optimizer = torch.optim.AdamW(params, lr=float(lr), foreach=False)
    batch_fn = task.make_batch(tokenizer, cfg.device, cfg.batch_size, cfg.max_seq_len, cfg.seed + stable_seed(stage, 6100))
    eos = tokenizer.eos_token or ""
    start = time.time()
    for step in range(1, int(steps) + 1):
        # Teacher-forced SDFT avoids brittle decoder generation paths while
        # still distilling the frozen task teacher on the active task tokens.
        sdft_batch = batch_fn(step)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        split = qp._split_tensor_batch(sdft_batch, cfg.consolidation_micro_batch_size)
        loss_value = 0.0
        for micro in split:
            outputs = student(**micro, use_cache=False)
            if loss_type == "forward_kl":
                teacher_micro = move_batch(micro, teacher_device)
                with torch.no_grad():
                    teacher_outputs = teacher_new(**teacher_micro, use_cache=False)
                loss = sdft_forward_kl_loss(outputs, teacher_outputs, micro["labels"]) / max(1, len(split))
            else:
                loss = outputs.loss / max(1, len(split))
            loss.backward()
            loss_value += float(loss.item())
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        if step % cfg.log_interval == 0 or step == steps:
            print(
                f"[{stage}] step={step:04d}/{steps} loss={loss_value:.4f} "
                f"method=teacher_forced_sdft loss_type={loss_type} protection=none "
                f"old_task_examples=0 proxy_batches=0",
                flush=True,
            )
            logger.log_curve(
                stage,
                step,
                loss=loss_value,
                old_task_examples=0,
                proxy_batches=0,
                method="teacher_forced_sdft",
                sdft_loss_type=loss_type,
            )
    metrics = evaluate_suite(student, tokenizer, active_eval_tasks, cfg, do_generation=True, include_wikitext=True)
    print_metrics(stage, metrics, [t.spec.name for t in active_eval_tasks])
    logger.log_stage_summary(
        stage,
        metrics,
        wall_time_sec=time.time() - start,
        old_task_examples=0,
        proxy_batches=0,
        method="teacher_forced_sdft",
    )
    print(f"[{stage}] wall_time_sec={time.time() - start:.1f}", flush=True)
    return metrics


def consolidate_on_policy_demo_sdft(
    *,
    student,
    ref_model,
    tokenizer,
    task: TaskData,
    active_eval_tasks: Sequence[TaskData],
    cfg: qp.RuntimeConfig,
    logger: ArtifactLogger,
    stage: str,
    steps: int,
    lr: float,
    teacher_device: str,
    do_sample: bool,
    temperature: float,
    top_p: float,
    ref_model_mixup_alpha: float,
    ref_model_sync_steps: int,
) -> Dict[str, float]:
    """Shen et al.-style on-policy demo-conditioned SDFT baseline.

    The student generates completions from the ordinary prompt. The frozen /
    slowly-synced reference model scores the same completion tokens under a
    demonstration-conditioned teacher prompt that contains the gold response.
    The student is trained with per-token forward KL on completion positions.
    """
    qp._freeze_model(ref_model)
    ref_model.to(teacher_device)
    params = _trainable_full_student(student, cfg)
    optimizer = torch.optim.AdamW(params, lr=float(lr), foreach=False)
    eos = tokenizer.eos_token or ""
    start = time.time()
    for step in range(1, int(steps) + 1):
        prompts, targets = _sample_task_texts(task, cfg.seed + stable_seed(stage, 6200), step, cfg.batch_size)
        student.eval()
        with torch.no_grad():
            completions = generate_on_policy_completions(
                student,
                tokenizer,
                prompts,
                cfg.device,
                max_new_tokens=task.spec.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
        completions = [truncate_completion(text) + eos for text in completions]
        teacher_prompts = [_demo_conditioned_teacher_prompt(prompt, target) for prompt, target in zip(prompts, targets)]
        student_batch = _prepare_completion_kl_batch(tokenizer, prompts, completions, cfg.device, cfg.max_seq_len)
        teacher_batch = _prepare_completion_kl_batch(tokenizer, teacher_prompts, completions, teacher_device, cfg.max_seq_len)

        student.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = student(
            input_ids=student_batch["input_ids"],
            attention_mask=student_batch["attention_mask"],
            use_cache=False,
        )
        with torch.no_grad():
            teacher_outputs = ref_model(
                input_ids=teacher_batch["input_ids"],
                attention_mask=teacher_batch["attention_mask"],
                use_cache=False,
            )
        loss = _completion_forward_kl_loss(outputs, teacher_outputs, student_batch["completion_mask"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        if int(ref_model_sync_steps) > 0 and step % int(ref_model_sync_steps) == 0:
            _sync_ref_model(student, ref_model, float(ref_model_mixup_alpha))
        if step % cfg.log_interval == 0 or step == steps:
            loss_value = float(loss.item())
            print(
                f"[{stage}] step={step:04d}/{steps} loss={loss_value:.4f} "
                "method=on_policy_demo_sdft loss_type=forward_kl protection=none "
                "old_task_examples=0 proxy_batches=0",
                flush=True,
            )
            logger.log_curve(
                stage,
                step,
                loss=loss_value,
                old_task_examples=0,
                proxy_batches=0,
                method="on_policy_demo_sdft",
                sdft_loss_type="forward_kl",
            )
    metrics = evaluate_suite(student, tokenizer, active_eval_tasks, cfg, do_generation=True, include_wikitext=True)
    print_metrics(stage, metrics, [t.spec.name for t in active_eval_tasks])
    logger.log_stage_summary(
        stage,
        metrics,
        wall_time_sec=time.time() - start,
        old_task_examples=0,
        proxy_batches=0,
        method="on_policy_demo_sdft",
    )
    print(f"[{stage}] wall_time_sec={time.time() - start:.1f}", flush=True)
    return metrics


def print_metrics(label: str, metrics: Dict[str, float], task_names: Sequence[str]) -> None:
    parts = [f"{label:<28}", f"ppl={fmt(metrics.get('wikitext_ppl')):>7}"]
    for name in task_names:
        parts.extend(
            [
                f"{name}_exact={fmt(metrics.get(f'{name}_exact')):>7}",
                f"{name}_tok={fmt(metrics.get(f'{name}_token_acc')):>7}",
                f"{name}_tf={fmt(metrics.get(f'{name}_tf_token_acc')):>7}",
                f"{name}_loss={fmt(metrics.get(f'{name}_tf_loss')):>7}",
            ]
        )
    print(" ".join(parts), flush=True)


def checkpoint_model(model, tokenizer, out_dir: Path, stage: str, policy: str, metadata: Dict[str, Any]) -> Optional[Path]:
    if policy == "none":
        return None
    if policy == "final_pretrained":
        if stage != "final":
            return None
        policy = "pretrained"
    if policy == "final_state":
        if stage != "final":
            return None
        policy = "state"
    if policy not in {"state", "pretrained"}:
        raise ValueError(f"unknown checkpoint policy: {policy}")

    stage_dir = out_dir / "checkpoints" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    with (stage_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=str)
    if policy == "pretrained":
        model.save_pretrained(stage_dir, safe_serialization=True)
        tokenizer.save_pretrained(stage_dir)
        return stage_dir
    torch.save(model.state_dict(), stage_dir / "state_dict.pt")
    try:
        tokenizer.save_pretrained(stage_dir / "tokenizer")
    except Exception:
        pass
    return stage_dir


def make_runtime_config(args: argparse.Namespace, out_dir: Path) -> qp.RuntimeConfig:
    return qp.RuntimeConfig(
        model_id=args.model_id,
        device=args.device,
        dtype=choose_dtype(args.dtype),
        local_files_only=args.local_files_only,
        resume=False,
        smoke=args.smoke,
        output_dir=out_dir,
        backup_dir=None,
        seed=args.seed,
        phase_scope="alien_ladder",
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        consolidation_micro_batch_size=args.micro_batch_size,
        max_seq_len=args.max_seq_len,
        gradient_checkpointing=args.gradient_checkpointing,
        wikitext_eval_samples=args.wikitext_eval_samples,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        grad_clip=args.grad_clip,
    )


def preflight_task(
    *,
    base_model,
    tokenizer,
    task: TaskData,
    cfg: qp.RuntimeConfig,
    args: argparse.Namespace,
    logger: ArtifactLogger,
) -> Tuple[bool, Dict[str, float]]:
    subsection(f"Preflight: {task.spec.display_name}")
    base_metrics = evaluate_task(base_model, tokenizer, task, cfg, do_generation=False)
    teacher = qp._clone_model(base_model, cfg.device)
    layers, _ = select_layers_generic(
        model=teacher,
        tokenizer=tokenizer,
        task=task,
        protected_profiles=[],
        cfg=cfg,
        min_layers=args.min_layers,
        stage=f"preflight_{task.spec.name}",
        logger=logger,
    )
    metrics = train_adapter_teacher(
        model=teacher,
        tokenizer=tokenizer,
        task=task,
        active_eval_tasks=[task],
        cfg=cfg,
        logger=logger,
        stage=f"preflight_teacher_{task.spec.name}",
        selected_layers=layers,
        steps=args.preflight_steps,
        lr=args.teacher_lr,
        rank=args.teacher_rank,
        alpha=args.teacher_alpha,
        gate_init=args.teacher_gate_init,
        eval_interval=max(args.preflight_steps, 1),
    )
    loss0 = float(base_metrics.get(f"{task.spec.name}_tf_loss", float("inf")))
    loss1 = float(metrics.get(f"{task.spec.name}_tf_loss", float("inf")))
    improvement = 0.0 if not math.isfinite(loss0) or loss0 <= 0 else (loss0 - loss1) / max(loss0, 1e-9)
    token_acc = float(metrics.get(f"{task.spec.name}_token_acc", 0.0))
    exact = float(metrics.get(f"{task.spec.name}_exact", 0.0))
    ok = (
        token_acc >= float(task.spec.token_gate)
        or exact >= float(task.spec.exact_gate)
        or improvement >= float(task.spec.loss_improvement_gate)
    )
    print(
        f"[preflight:{task.spec.name}] {'PASS' if ok else 'FAIL'} "
        f"token_acc={fmt(token_acc)} exact={fmt(exact)} loss_improvement={fmt(improvement)} "
        f"gates tok>={task.spec.token_gate:.3f} exact>={task.spec.exact_gate:.3f} "
        f"loss_improve>={task.spec.loss_improvement_gate:.3f}",
        flush=True,
    )
    logger.log_event(
        "preflight_gate",
        task=task.spec.name,
        passed=ok,
        token_acc=token_acc,
        exact=exact,
        loss_improvement=improvement,
        base_loss=loss0,
        teacher_loss=loss1,
    )
    release(teacher)
    return ok, metrics


def run_preflight(
    base_model,
    tokenizer,
    scan_task: TaskData,
    c_task: TaskData,
    geo_task: TaskData,
    cfg: qp.RuntimeConfig,
    args: argparse.Namespace,
    logger: ArtifactLogger,
) -> TaskData:
    section("PREFLIGHT SCOUT")
    scan_ok, _ = preflight_task(base_model=base_model, tokenizer=tokenizer, task=scan_task, cfg=cfg, args=args, logger=logger)
    c_ok, _ = preflight_task(base_model=base_model, tokenizer=tokenizer, task=c_task, cfg=cfg, args=args, logger=logger)
    if not c_ok and c_task.spec.name == "cogs":
        print("[preflight] COGS failed; trying fallback C task.", flush=True)
        c_task = load_synthetic_listops_task(args.seed + 303, args.task_train_samples, args.task_eval_samples)
        c_ok, _ = preflight_task(base_model=base_model, tokenizer=tokenizer, task=c_task, cfg=cfg, args=args, logger=logger)
        if not c_ok:
            c_task = load_proof_v2_record_task(tokenizer, args.seed + 304, args.task_train_samples, args.task_eval_samples)
            c_ok, _ = preflight_task(base_model=base_model, tokenizer=tokenizer, task=c_task, cfg=cfg, args=args, logger=logger)
    geo_ok, _ = preflight_task(base_model=base_model, tokenizer=tokenizer, task=geo_task, cfg=cfg, args=args, logger=logger)
    if not (scan_ok and c_ok and geo_ok):
        raise RuntimeError(f"preflight failed: scan={scan_ok} c={c_ok} geo={geo_ok}")
    return c_task


def run() -> None:
    args = build_arg_parser().parse_args()
    try:
        args.d_variants = parse_d_variants(args.d_variants)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    selected_d_variants = set(args.d_variants)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_id = args.model_id or default_model_id(args.local_files_only)
    args.model_id = model_id
    out_dir = Path(args.output_dir).expanduser() / f"alien_ladder_seed{args.seed}"
    logger = ArtifactLogger(out_dir)
    cfg = make_runtime_config(args, out_dir)
    aux_teacher_device = resolve_aux_device(args.teacher_device, args.device)

    section("ALIEN LADDER REPLAY-FREE CONTINUAL LEARNING AUDIT")
    print(f"model_id={model_id}", flush=True)
    print(f"device={args.device} dtype={args.dtype} seed={args.seed}", flush=True)
    print(f"teacher_device={aux_teacher_device} (for frozen teachers during consolidation/SDFT)", flush=True)
    print("tasks: B=SCAN -> C=COGS/ListOps/proof_v2 fallback -> D=GeoQuery", flush=True)
    print(f"variants selected: {', '.join(args.d_variants)}", flush=True)
    print("stdout is a convenience artifact; JSONL/CSV are the plot artifacts.", flush=True)

    logger.write_json("run_config.json", vars(args))
    tokenizer = load_tokenizer(model_id, local_files_only=args.local_files_only)
    base_model = load_causal_lm(
        model_id,
        device=args.device,
        dtype=choose_dtype(args.dtype),
        local_files_only=args.local_files_only,
    )

    if args.gradient_checkpointing:
        qp._configure_gradient_checkpointing(base_model, True)

    # Load wikitext for retention evaluation
    wikitext_val = qp.load_wikitext_texts(
        tokenizer, split="validation",
        max_seq_len=cfg.max_seq_len, max_samples=cfg.wikitext_eval_samples,
        local_files_only=args.local_files_only,
    )
    cfg._wikitext_val = wikitext_val  # attach so evaluate_suite can find it

    scan_spec, _, geo_spec = build_task_specs(args)
    if args.smoke:
        scan_task = load_synthetic_listops_task(args.seed + 11, 64, 16)
        scan_task.spec.name = "scan"
        scan_task.spec.display_name = "SCAN smoke"
        c_task = load_synthetic_listops_task(args.seed + 12, 64, 16)
        geo_task = load_synthetic_listops_task(args.seed + 13, 64, 16)
        geo_task.spec.name = "geoquery"
        geo_task.spec.display_name = "GeoQuery smoke"
    else:
        scan_task = load_hf_task(scan_spec, tokenizer, args.seed + 100)
        c_task = load_c_task_with_fallback(args, tokenizer, logger)
        geo_task = load_hf_task(geo_spec, tokenizer, args.seed + 300)

    logger.write_json(
        "dataset_manifest.json",
        {
            "scan": scan_task.manifest,
            "c": c_task.manifest,
            "geoquery": geo_task.manifest,
        },
    )

    if not args.skip_preflight:
        c_task = run_preflight(base_model, tokenizer, scan_task, c_task, geo_task, cfg, args, logger)
        logger.write_json(
            "dataset_manifest.json",
            {
                "scan": scan_task.manifest,
                "c": c_task.manifest,
                "geoquery": geo_task.manifest,
            },
        )
        if args.preflight_only:
            print("[preflight] completed; --preflight-only set.", flush=True)
            return

    protected_profiles: List[Any] = []
    if args.protect_base_language_profile:
        subsection("Base language geometry profile")
        wikitext_profile_chunks = qp.load_wikitext_texts(
            tokenizer,
            split=args.base_language_profile_split,
            max_seq_len=cfg.max_seq_len,
            max_samples=args.base_language_profile_samples,
            local_files_only=args.local_files_only,
        )
        base_language_batch_fn = qp.make_wikitext_batch_fn(
            tokenizer,
            wikitext_profile_chunks,
            cfg.device,
            cfg,
            cfg.seed + 9001,
        )
        base_language_profile = qp._collect_profiles(base_model, "base_language", base_language_batch_fn)
        protected_profiles.append(base_language_profile)
        logger.log_event(
            "base_language_profile_collected",
            split=args.base_language_profile_split,
            samples=len(wikitext_profile_chunks),
            protected_profile_count=len(protected_profiles),
            note="Profile is precomputed once for geometry protection; no WikiText/proxy batches are used during task updates.",
        )
        print(
            f"[base_language_profile] split={args.base_language_profile_split} "
            f"samples={len(wikitext_profile_chunks)} protected_profiles={len(protected_profiles)} "
            "update_proxy_batches=0",
            flush=True,
        )

    subsection("Stage A: base model evaluation")
    base_metrics = evaluate_suite(base_model, tokenizer, [scan_task, c_task, geo_task], cfg, do_generation=True, include_wikitext=True)
    print_metrics("base_A", base_metrics, [scan_task.spec.name, c_task.spec.name, geo_task.spec.name])
    logger.log_stage_summary("base_A", base_metrics)
    checkpoint_model(base_model, tokenizer, out_dir, "base_A", args.checkpoint_policy, {"stage": "base_A", "metrics": base_metrics})

    subsection("Stage B: train SCAN teacher")
    teacher_b = qp._clone_model(base_model, cfg.device)
    b_layers, _ = select_layers_generic(
        model=teacher_b,
        tokenizer=tokenizer,
        task=scan_task,
        protected_profiles=protected_profiles,
        cfg=cfg,
        min_layers=args.min_layers,
        stage="teacher_B_SCAN",
        logger=logger,
    )
    teacher_b_metrics = train_adapter_teacher(
        model=teacher_b,
        tokenizer=tokenizer,
        task=scan_task,
        active_eval_tasks=[scan_task],
        cfg=cfg,
        logger=logger,
        stage="teacher_B_SCAN",
        selected_layers=b_layers,
        steps=args.b_steps,
        lr=args.teacher_lr,
        rank=args.teacher_rank,
        alpha=args.teacher_alpha,
        gate_init=args.teacher_gate_init,
        eval_interval=args.eval_interval,
    )
    require_generated_acquisition("teacher_B_SCAN", teacher_b_metrics, scan_task)
    checkpoint_model(teacher_b, tokenizer, out_dir, "teacher_B_SCAN", args.checkpoint_policy, {"stage": "teacher_B_SCAN", "metrics": teacher_b_metrics})

    subsection("Stage AB: consolidate SCAN with no proxy")
    base_ab = qp._clone_model(base_model, cfg.device)
    base_ab_metrics = consolidate_no_proxy(
        student=base_ab,
        teacher_old=base_model,
        teacher_new=teacher_b,
        tokenizer=tokenizer,
        task=scan_task,
        active_eval_tasks=[scan_task],
        selected_layers=b_layers,
        old_profiles=protected_profiles,
        cfg=cfg,
        logger=logger,
        stage="base_AB_no_proxy",
        steps=args.b_consol_steps,
        lr=stage_weight(args, "ab_consolidation_lr", "consolidation_lr"),
        old_kl_weight=stage_weight(args, "ab_old_kl_weight", "no_proxy_old_kl_weight"),
        old_hidden_weight=stage_weight(args, "ab_old_hidden_weight", "no_proxy_old_hidden_weight"),
        new_kl_weight=stage_weight(args, "ab_new_kl_weight", "new_kl_weight"),
        new_hidden_weight=stage_weight(args, "ab_new_hidden_weight", "new_hidden_weight"),
        project_gradients=args.gradient_projection,
        projection_strength=args.projection_strength,
        teacher_device=aux_teacher_device,
    )
    require_generated_acquisition("base_AB_no_proxy", base_ab_metrics, scan_task)
    checkpoint_model(base_ab, tokenizer, out_dir, "base_AB_no_proxy", args.checkpoint_policy, {"stage": "base_AB_no_proxy", "metrics": base_ab_metrics})
    protected_profiles.append(collect_profile(base_ab, tokenizer, scan_task, cfg, "SCAN_after_AB"))
    del teacher_b
    del base_model
    release()

    subsection(f"Stage C: train {c_task.spec.display_name} teacher")
    teacher_c = qp._clone_model(base_ab, cfg.device)
    c_layers, _ = select_layers_generic(
        model=teacher_c,
        tokenizer=tokenizer,
        task=c_task,
        protected_profiles=protected_profiles,
        cfg=cfg,
        min_layers=args.min_layers,
        stage=f"teacher_C_{c_task.spec.name}",
        logger=logger,
    )
    teacher_c_metrics = train_adapter_teacher(
        model=teacher_c,
        tokenizer=tokenizer,
        task=c_task,
        active_eval_tasks=[scan_task, c_task],
        cfg=cfg,
        logger=logger,
        stage=f"teacher_C_{c_task.spec.name}",
        selected_layers=c_layers,
        steps=args.c_steps,
        lr=args.teacher_lr,
        rank=args.teacher_rank,
        alpha=args.teacher_alpha,
        gate_init=args.teacher_gate_init,
        eval_interval=args.eval_interval,
    )
    require_generated_acquisition(f"teacher_C_{c_task.spec.name}", teacher_c_metrics, c_task)
    checkpoint_model(teacher_c, tokenizer, out_dir, f"teacher_C_{c_task.spec.name}", args.checkpoint_policy, {"stage": "teacher_C", "metrics": teacher_c_metrics})

    subsection(f"Stage ABC: consolidate {c_task.spec.display_name} with no proxy")
    base_abc = qp._clone_model(base_ab, cfg.device)
    base_abc_metrics = consolidate_no_proxy(
        student=base_abc,
        teacher_old=base_ab,
        teacher_new=teacher_c,
        tokenizer=tokenizer,
        task=c_task,
        active_eval_tasks=[scan_task, c_task],
        selected_layers=c_layers,
        old_profiles=protected_profiles,
        cfg=cfg,
        logger=logger,
        stage="base_ABC_no_proxy",
        steps=args.c_consol_steps,
        lr=args.consolidation_lr,
        old_kl_weight=args.no_proxy_old_kl_weight,
        old_hidden_weight=args.no_proxy_old_hidden_weight,
        new_kl_weight=args.new_kl_weight,
        new_hidden_weight=args.new_hidden_weight,
        project_gradients=args.gradient_projection,
        projection_strength=args.projection_strength,
        teacher_device=aux_teacher_device,
    )
    require_generated_acquisition("base_ABC_no_proxy", base_abc_metrics, scan_task)
    require_generated_acquisition("base_ABC_no_proxy", base_abc_metrics, c_task)
    checkpoint_model(base_abc, tokenizer, out_dir, "base_ABC_no_proxy", args.checkpoint_policy, {"stage": "base_ABC_no_proxy", "metrics": base_abc_metrics})
    protected_profiles.append(collect_profile(base_abc, tokenizer, c_task, cfg, f"{c_task.spec.name}_after_ABC"))
    del teacher_c
    del base_ab
    release()

    subsection("Stage D: train GeoQuery teacher")
    teacher_d = qp._clone_model(base_abc, cfg.device)
    d_layers, _ = select_layers_generic(
        model=teacher_d,
        tokenizer=tokenizer,
        task=geo_task,
        protected_profiles=protected_profiles,
        cfg=cfg,
        min_layers=args.d_min_layers,
        stage="teacher_D_GeoQuery",
        logger=logger,
    )
    teacher_d_metrics = train_adapter_teacher(
        model=teacher_d,
        tokenizer=tokenizer,
        task=geo_task,
        active_eval_tasks=[scan_task, c_task, geo_task],
        cfg=cfg,
        logger=logger,
        stage="teacher_D_GeoQuery",
        selected_layers=d_layers,
        steps=args.d_steps,
        lr=args.teacher_lr,
        rank=args.d_rank,
        alpha=args.d_alpha,
        gate_init=args.teacher_gate_init,
        eval_interval=args.eval_interval,
    )
    require_generated_acquisition("teacher_D_GeoQuery", teacher_d_metrics, geo_task)
    checkpoint_model(teacher_d, tokenizer, out_dir, "teacher_D_GeoQuery", args.checkpoint_policy, {"stage": "teacher_D_GeoQuery", "metrics": teacher_d_metrics})

    active_all = [scan_task, c_task, geo_task]
    variant_metrics: Dict[str, Optional[Dict[str, float]]] = {label: None for label in ALL_D_VARIANTS}

    if "naive_sft" in selected_d_variants:
        subsection("Variant 1: D naive SFT")
        naive = qp._clone_model(base_abc, cfg.device)
        naive_metrics = consolidate_naive(
            student=naive,
            tokenizer=tokenizer,
            task=geo_task,
            active_eval_tasks=active_all,
            cfg=cfg,
            logger=logger,
            stage="D_naive_sft",
            steps=args.d_variant_steps,
            lr=args.consolidation_lr,
        )
        variant_metrics["naive_sft"] = naive_metrics
        checkpoint_model(naive, tokenizer, out_dir, "D_naive_sft", args.checkpoint_policy, {"stage": "D_naive_sft", "metrics": naive_metrics})
        del naive
        release()
    else:
        logger.log_event("variant_skipped", variant="naive_sft", reason="not selected by --d-variants")

    if "sdft_baseline" in selected_d_variants:
        subsection("Variant 2: D SDFT baseline")
        sdft = qp._clone_model(base_abc, cfg.device)
        sdft_metrics = consolidate_sdft(
            student=sdft,
            teacher_new=teacher_d,
            tokenizer=tokenizer,
            task=geo_task,
            active_eval_tasks=active_all,
            cfg=cfg,
            logger=logger,
            stage="D_sdft_baseline",
            steps=args.d_variant_steps,
            lr=args.consolidation_lr,
            teacher_device=aux_teacher_device,
            loss_type=args.sdft_loss,
            do_sample=args.sdft_do_sample,
            temperature=args.sdft_temperature,
            top_p=args.sdft_top_p,
        )
        variant_metrics["sdft_baseline"] = sdft_metrics
        checkpoint_model(sdft, tokenizer, out_dir, "D_sdft_baseline", args.checkpoint_policy, {"stage": "D_sdft_baseline", "metrics": sdft_metrics})
        del sdft
        release()
    else:
        logger.log_event("variant_skipped", variant="sdft_baseline", reason="not selected by --d-variants")

    if "fixed_no_proxy" in selected_d_variants:
        subsection("Variant 3: D fixed no-proxy")
        fixed = qp._clone_model(base_abc, cfg.device)
        fixed_metrics = consolidate_no_proxy(
            student=fixed,
            teacher_old=base_abc,
            teacher_new=teacher_d,
            tokenizer=tokenizer,
            task=geo_task,
            active_eval_tasks=active_all,
            selected_layers=d_layers,
            old_profiles=protected_profiles,
            cfg=cfg,
            logger=logger,
            stage="D_fixed_no_proxy",
            steps=args.d_variant_steps,
            lr=args.consolidation_lr,
            old_kl_weight=args.no_proxy_old_kl_weight,
            old_hidden_weight=args.no_proxy_old_hidden_weight,
            new_kl_weight=args.new_kl_weight,
            new_hidden_weight=args.new_hidden_weight,
            project_gradients=args.gradient_projection,
            projection_strength=args.projection_strength,
            teacher_device=aux_teacher_device,
        )
        variant_metrics["fixed_no_proxy"] = fixed_metrics
        checkpoint_model(fixed, tokenizer, out_dir, "D_fixed_no_proxy", args.checkpoint_policy, {"stage": "D_fixed_no_proxy", "metrics": fixed_metrics})
        del fixed
        release()
    else:
        logger.log_event("variant_skipped", variant="fixed_no_proxy", reason="not selected by --d-variants")

    if "expanded_no_proxy" in selected_d_variants:
        subsection("Variant 4: D expanded no-proxy")
        insert_after = int(d_layers[0]) if d_layers else 0
        expanded_teacher = qp._clone_model(base_abc, cfg.device)
        expanded_teacher, _ = qp.insert_expansion_layer(expanded_teacher, insert_after)
        expanded_layers = sorted(set([idx if idx <= insert_after else idx + 1 for idx in d_layers] + [insert_after + 1]))
        expanded_teacher_metrics = train_adapter_teacher(
            model=expanded_teacher,
            tokenizer=tokenizer,
            task=geo_task,
            active_eval_tasks=active_all,
            cfg=cfg,
            logger=logger,
            stage="teacher_D_GeoQuery_expanded",
            selected_layers=expanded_layers,
            steps=args.expansion_teacher_steps,
            lr=args.teacher_lr,
            rank=args.d_rank,
            alpha=args.d_alpha,
            gate_init=args.teacher_gate_init,
            eval_interval=args.eval_interval,
            train_gated_layers=True,
        )
        checkpoint_model(
            expanded_teacher,
            tokenizer,
            out_dir,
            "teacher_D_GeoQuery_expanded",
            args.checkpoint_policy,
            {"stage": "teacher_D_GeoQuery_expanded", "metrics": expanded_teacher_metrics, "insert_after": insert_after},
        )

        expanded_old = qp._clone_model(base_abc, cfg.device)
        expanded_old, _ = qp.insert_expansion_layer(expanded_old, insert_after)
        expanded_student = qp._clone_model(base_abc, cfg.device)
        expanded_student, _ = qp.insert_expansion_layer(expanded_student, insert_after)
        expanded_metrics = consolidate_no_proxy(
            student=expanded_student,
            teacher_old=expanded_old,
            teacher_new=expanded_teacher,
            tokenizer=tokenizer,
            task=geo_task,
            active_eval_tasks=active_all,
            selected_layers=expanded_layers,
            old_profiles=protected_profiles,
            cfg=cfg,
            logger=logger,
            stage="D_expanded_no_proxy",
            steps=args.expansion_steps,
            lr=args.consolidation_lr,
            old_kl_weight=args.no_proxy_old_kl_weight,
            old_hidden_weight=args.no_proxy_old_hidden_weight,
            new_kl_weight=args.new_kl_weight,
            new_hidden_weight=args.new_hidden_weight,
            project_gradients=args.gradient_projection,
            projection_strength=args.projection_strength,
            teacher_device=aux_teacher_device,
        )
        variant_metrics["expanded_no_proxy"] = expanded_metrics
        checkpoint_model(
            expanded_student,
            tokenizer,
            out_dir,
            "D_expanded_no_proxy",
            args.checkpoint_policy,
            {"stage": "D_expanded_no_proxy", "metrics": expanded_metrics, "insert_after": insert_after},
        )
        del expanded_old
        del expanded_teacher
        del expanded_student
    else:
        logger.log_event("variant_skipped", variant="expanded_no_proxy", reason="not selected by --d-variants")

    del teacher_d
    del base_abc
    release()

    section("FINAL VERDICT")
    summarize_verdict(
        logger=logger,
        scan=scan_task,
        c_task=c_task,
        geo=geo_task,
        base_abc=base_abc_metrics,
        naive=variant_metrics["naive_sft"],
        sdft=variant_metrics["sdft_baseline"],
        fixed=variant_metrics["fixed_no_proxy"],
        expanded=variant_metrics["expanded_no_proxy"],
        args=args,
    )
    print(f"artifacts={out_dir}", flush=True)


def summarize_verdict(
    *,
    logger: ArtifactLogger,
    scan: TaskData,
    c_task: TaskData,
    geo: TaskData,
    base_abc: Dict[str, float],
    naive: Optional[Dict[str, float]],
    sdft: Optional[Dict[str, float]],
    fixed: Optional[Dict[str, float]],
    expanded: Optional[Dict[str, float]],
    args: argparse.Namespace,
) -> None:
    def m(metrics: Dict[str, float], key: str) -> float:
        try:
            return float(metrics.get(key, float("nan")))
        except Exception:
            return float("nan")

    rows_all = {
        "naive_sft": naive,
        "sdft_baseline": sdft,
        "fixed_no_proxy": fixed,
        "expanded_no_proxy": expanded,
    }
    rows = {label: metrics for label, metrics in rows_all.items() if metrics is not None}
    row_stats: Dict[str, Dict[str, float]] = {}
    for label in ALL_D_VARIANTS:
        if rows_all.get(label) is None:
            print(f"{label:<20} SKIPPED by --d-variants", flush=True)
            logger.log_event("verdict_row_skipped", variant=label, reason="not selected by --d-variants")
    for label, metrics in rows.items():
        scan_ret = m(metrics, f"{scan.spec.name}_token_acc")
        scan_exact = m(metrics, f"{scan.spec.name}_exact")
        scan_tf_ret = m(metrics, f"{scan.spec.name}_tf_token_acc")
        scan_hybrid = max(scan_ret, scan_tf_ret)
        c_ret = m(metrics, f"{c_task.spec.name}_token_acc")
        c_exact = m(metrics, f"{c_task.spec.name}_exact")
        c_tf_ret = m(metrics, f"{c_task.spec.name}_tf_token_acc")
        c_hybrid = max(c_ret, c_tf_ret)
        geo_acc = m(metrics, f"{geo.spec.name}_token_acc")
        geo_exact = m(metrics, f"{geo.spec.name}_exact")
        geo_tf = m(metrics, f"{geo.spec.name}_tf_token_acc")
        geo_hybrid = max(geo_acc, geo_tf)
        geo_loss = m(metrics, f"{geo.spec.name}_tf_loss")
        ppl_ratio = m(metrics, "wikitext_ppl") / max(m(base_abc, "wikitext_ppl"), 1e-9)
        ppl_preservation = 1.0 / max(ppl_ratio, 1e-9)
        row_stats[label] = {
            "scan_ret": scan_ret,
            "scan_exact": scan_exact,
            "scan_tf_ret": scan_tf_ret,
            "scan_hybrid": scan_hybrid,
            "c_ret": c_ret,
            "c_exact": c_exact,
            "c_tf_ret": c_tf_ret,
            "c_hybrid": c_hybrid,
            "geo_acc": geo_acc,
            "geo_exact": geo_exact,
            "geo_tf": geo_tf,
            "geo_hybrid": geo_hybrid,
            "geo_loss": geo_loss,
            "ppl_ratio": ppl_ratio,
            "ppl_preservation": ppl_preservation,
        }
        print(
            f"{label:<20} scan_ret={fmt(scan_ret)} scan_tf={fmt(scan_tf_ret)} "
            f"scan_hybrid={fmt(scan_hybrid)} "
            f"c_ret={fmt(c_ret)} c_tf={fmt(c_tf_ret)} "
            f"c_hybrid={fmt(c_hybrid)} "
            f"geo_tok={fmt(geo_acc)} geo_tf={fmt(geo_tf)} geo_loss={fmt(geo_loss)} "
            f"ppl_ratio_vs_base_ABC={fmt(ppl_ratio)} ppl_preservation={fmt(ppl_preservation)} "
            f"old_task_examples=0 proxy_batches=0",
            flush=True,
        )
        logger.log_event(
            "verdict_row",
            variant=label,
            scan_retention=scan_ret,
            scan_exact=scan_exact,
            scan_tf_retention=scan_tf_ret,
            scan_hybrid_retention=scan_hybrid,
            c_retention=c_ret,
            c_exact=c_exact,
            c_tf_retention=c_tf_ret,
            c_hybrid_retention=c_hybrid,
            geo_token_acc=geo_acc,
            geo_exact=geo_exact,
            geo_tf_token_acc=geo_tf,
            geo_hybrid_acc=geo_hybrid,
            geo_loss=geo_loss,
            ppl_ratio_vs_base_ABC=ppl_ratio,
            ppl_preservation=ppl_preservation,
            old_task_examples=0,
            proxy_batches=0,
        )

    # Treat expansion as optional capacity, not a mandatory win condition. If
    # fixed no-proxy already has enough room, it should be allowed to win.
    no_proxy_labels = tuple(label for label in ("fixed_no_proxy", "expanded_no_proxy") if label in row_stats)
    baseline_labels = tuple(label for label in ("naive_sft", "sdft_baseline") if label in row_stats)
    best_no_proxy = ""
    if no_proxy_labels:
        best_no_proxy = max(
            no_proxy_labels,
            key=lambda label: (
                row_stats[label]["scan_hybrid"]
                + row_stats[label]["c_hybrid"]
                + row_stats[label]["geo_hybrid"]
                + 0.25 * row_stats[label]["ppl_preservation"]
            ),
        )
    expanded_beats_fixed = False
    if "expanded_no_proxy" in row_stats and "fixed_no_proxy" in row_stats:
        expanded_beats_fixed = row_stats["expanded_no_proxy"]["geo_hybrid"] >= row_stats["fixed_no_proxy"]["geo_hybrid"] and (
            row_stats["expanded_no_proxy"]["scan_hybrid"] >= row_stats["fixed_no_proxy"]["scan_hybrid"] - 0.02
            and row_stats["expanded_no_proxy"]["c_hybrid"] >= row_stats["fixed_no_proxy"]["c_hybrid"] - 0.02
        )
    expansion_needed = bool(expanded_beats_fixed)
    best_baseline_ppl = min((row_stats[label]["ppl_ratio"] for label in baseline_labels), default=float("nan"))
    best_no_proxy_ppl = min((row_stats[label]["ppl_ratio"] for label in no_proxy_labels), default=float("nan"))
    comparative_claim_available = bool(no_proxy_labels and baseline_labels)
    ppl_rescue_pass = bool(comparative_claim_available and best_no_proxy_ppl < best_baseline_ppl)

    def beats_retention(no_proxy_label: str, baseline_label: str) -> bool:
        return (
            row_stats[no_proxy_label]["scan_hybrid"] >= row_stats[baseline_label]["scan_hybrid"]
            and row_stats[no_proxy_label]["c_hybrid"] >= row_stats[baseline_label]["c_hybrid"]
        )

    no_proxy_retention_rescue = bool(
        comparative_claim_available
        and all(any(beats_retention(no_proxy_label, baseline_label) for no_proxy_label in no_proxy_labels) for baseline_label in baseline_labels)
    )
    no_proxy_new_task_competitive = bool(
        comparative_claim_available
        and best_no_proxy
        and row_stats[best_no_proxy]["geo_hybrid"] >= min(row_stats[label]["geo_hybrid"] for label in baseline_labels) - 0.02
    )
    ppl_ok = bool(no_proxy_labels and best_no_proxy_ppl <= float(args.max_ppl_ratio))

    def task_gate_pass(label: str, task: TaskData) -> bool:
        if task is scan:
            hybrid = row_stats[label]["scan_hybrid"]
            exact = row_stats[label]["scan_exact"]
        elif task is c_task:
            hybrid = row_stats[label]["c_hybrid"]
            exact = row_stats[label]["c_exact"]
        else:
            hybrid = row_stats[label]["geo_hybrid"]
            exact = row_stats[label]["geo_exact"]
        return bool(hybrid >= float(task.spec.token_gate) or exact >= float(task.spec.exact_gate))

    sequential_method_pass = bool(
        best_no_proxy
        and task_gate_pass(best_no_proxy, scan)
        and task_gate_pass(best_no_proxy, c_task)
        and task_gate_pass(best_no_proxy, geo)
        and row_stats[best_no_proxy]["ppl_ratio"] <= float(args.max_ppl_ratio)
    )
    stress_test_pass = bool(comparative_claim_available and no_proxy_retention_rescue and ppl_rescue_pass and no_proxy_new_task_competitive)
    paper_ready = bool(stress_test_pass and ppl_ok)
    print(
        f"stress_test={'PASS' if stress_test_pass else 'PARTIAL'} "
        f"paper_ready={'PASS' if paper_ready else 'PARTIAL'} "
        f"single_method_sequential={'PASS' if sequential_method_pass else 'PARTIAL'} "
        f"comparative_claim_available={comparative_claim_available} "
        f"best_no_proxy={best_no_proxy or 'none'} "
        f"expansion_needed={expansion_needed} "
        f"expanded_beats_fixed={expanded_beats_fixed} "
        f"no_proxy_retention_rescue={no_proxy_retention_rescue} "
        f"no_proxy_new_task_competitive={no_proxy_new_task_competitive} "
        f"ppl_rescue_pass={ppl_rescue_pass} strict_ppl_ok={ppl_ok}",
        flush=True,
    )
    logger.log_event(
        "final_verdict",
        stress_test_pass=stress_test_pass,
        paper_ready=paper_ready,
        single_method_sequential_pass=sequential_method_pass,
        comparative_claim_available=comparative_claim_available,
        best_no_proxy=best_no_proxy,
        expansion_needed=expansion_needed,
        expanded_beats_fixed=expanded_beats_fixed,
        no_proxy_retention_beats_sdft=bool("sdft_baseline" in baseline_labels and any(beats_retention(label, "sdft_baseline") for label in no_proxy_labels)),
        no_proxy_retention_beats_naive=bool("naive_sft" in baseline_labels and any(beats_retention(label, "naive_sft") for label in no_proxy_labels)),
        no_proxy_retention_rescue=no_proxy_retention_rescue,
        no_proxy_new_task_competitive=no_proxy_new_task_competitive,
        ppl_rescue_pass=ppl_rescue_pass,
        strict_ppl_ok=ppl_ok,
        best_no_proxy_ppl_ratio=best_no_proxy_ppl,
        best_baseline_ppl_ratio=best_baseline_ppl,
    )



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Three alien benchmark replay-free CL + expansion audit")
    parser.add_argument("--model-id", default="", help="HF model id; default is Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--teacher-device",
        default="auto",
        help="Device for frozen teachers during consolidation/SDFT. Use auto to place teachers on cuda:1 when available.",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use tiny synthetic tasks for plumbing only")
    parser.add_argument(
        "--d-variants",
        default="all",
        help=(
            "Comma-separated final D branches to run. Use all, or any of: "
            "naive_sft, sdft_baseline, fixed_no_proxy, expanded_no_proxy. "
            "Aliases: naive, sdft, fixed, expanded."
        ),
    )

    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-steps", type=int, default=300)
    parser.add_argument("--scan-config", default="simple")
    parser.add_argument("--cogs-config", default=None)
    parser.add_argument("--geoquery-config", default=None)
    parser.add_argument("--allow-synthetic-c-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synthetic-c-fallback", choices=("listops", "proof_v2_record"), default="listops")
    parser.add_argument("--force-c-task", choices=("auto", "cogs", "listops", "proof_v2_record"), default="auto")

    parser.add_argument("--task-train-samples", type=int, default=512)
    parser.add_argument("--task-eval-samples", type=int, default=96)
    parser.add_argument("--scan-max-target-tokens", type=int, default=80)
    parser.add_argument("--cogs-max-target-tokens", type=int, default=50)
    parser.add_argument("--geoquery-max-target-tokens", type=int, default=80)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--wikitext-eval-samples", type=int, default=128)
    parser.add_argument(
        "--protect-base-language-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Precompute a tiny WikiText geometry profile from the frozen base and add it to protected profiles. This uses no proxy batches during updates.",
    )
    parser.add_argument("--base-language-profile-samples", type=int, default=64)
    parser.add_argument("--base-language-profile-split", default="train")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--b-steps", type=int, default=1500)
    parser.add_argument("--b-consol-steps", type=int, default=600)
    parser.add_argument("--c-steps", type=int, default=2200)
    parser.add_argument("--c-consol-steps", type=int, default=600)
    parser.add_argument("--d-steps", type=int, default=2200)
    parser.add_argument("--d-variant-steps", type=int, default=2000)
    parser.add_argument("--expansion-teacher-steps", type=int, default=1600)
    parser.add_argument("--expansion-steps", type=int, default=2000)
    parser.add_argument("--sdft-loss", choices=("forward_kl", "ce"), default="forward_kl")
    parser.add_argument("--sdft-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sdft-temperature", type=float, default=0.7)
    parser.add_argument("--sdft-top-p", type=float, default=0.95)

    parser.add_argument("--teacher-rank", type=int, default=48)
    parser.add_argument("--teacher-alpha", type=float, default=96.0)
    parser.add_argument("--d-rank", type=int, default=64)
    parser.add_argument("--d-alpha", type=float, default=128.0)
    parser.add_argument("--teacher-lr", type=float, default=8e-5)
    parser.add_argument("--consolidation-lr", type=float, default=1e-5)
    parser.add_argument(
        "--ab-consolidation-lr",
        type=float,
        default=None,
        help="Optional Stage AB consolidation LR. Stage AB writes the first skill and often needs a stronger LR than later retention-heavy stages.",
    )
    parser.add_argument("--teacher-gate-init", type=float, default=-1.5)
    parser.add_argument("--min-layers", type=int, default=8)
    parser.add_argument("--d-min-layers", type=int, default=10)

    parser.add_argument("--no-proxy-old-kl-weight", type=float, default=0.75)
    parser.add_argument("--no-proxy-old-hidden-weight", type=float, default=18.0)
    parser.add_argument("--new-kl-weight", type=float, default=1.0)
    parser.add_argument("--new-hidden-weight", type=float, default=0.5)
    parser.add_argument(
        "--ab-old-kl-weight",
        type=float,
        default=None,
        help="Optional Stage AB old/base KL weight. Use lower protection here to acquire the first skill before later retention-heavy stages.",
    )
    parser.add_argument(
        "--ab-old-hidden-weight",
        type=float,
        default=None,
        help="Optional Stage AB old/base hidden weight.",
    )
    parser.add_argument(
        "--ab-new-kl-weight",
        type=float,
        default=None,
        help="Optional Stage AB new-teacher KL weight.",
    )
    parser.add_argument(
        "--ab-new-hidden-weight",
        type=float,
        default=None,
        help="Optional Stage AB new-teacher hidden weight.",
    )
    parser.add_argument("--gradient-projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projection-strength", type=float, default=1.0)

    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--checkpoint-policy",
        choices=("none", "state", "pretrained", "final_state", "final_pretrained"),
        default="state",
    )
    parser.add_argument("--max-ppl-ratio", type=float, default=1.12)
    return parser


if __name__ == "__main__":
    run()
