from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .contexts import ContextResult, register_context
from .events import (
    AcquisitionBudget,
    ContextSpec,
    DatasetRef,
    ExampleRecord,
    GateBundle,
    GateRule,
    LearningEvent,
    PublicExample,
    TargetRef,
    VerifierSpec,
)
from .trajectories import SamplingConfig
from ._io import sha256_file, sha256_json


def _records(
    manifest: Mapping[str, Any],
    task_key: str,
    split: str,
    *,
    composition_context: bool = False,
) -> Tuple[ExampleRecord, ...]:
    rows = manifest["tasks"][task_key][split]
    records: List[ExampleRecord] = []
    for row in rows:
        metadata: Dict[str, Any] = {"task_key": task_key, "policy_version": row.get("policy_version", "")}
        if composition_context:
            glyph = str(row["glyph"])
            metadata.update(
                {
                    "hop_a_prompt": (
                        "AURORA LEXICON. Return only the canonical color code.\n"
                        f"Glyph: {glyph}\nColor code:"
                    ),
                    "hop_b_prompt_template": (
                        "AURORA POLICY. Return only the canonical action code.\n"
                        "Color code: {intermediate}\nAction code:"
                    ),
                }
            )
        if task_key == "skill_b_v2_changed" and row.get("old_action"):
            metadata["stale_output"] = str(row["old_action"])
        records.append(
            ExampleRecord(
                example_id=str(row["row_id"]),
                prompt=str(row["prompt"]),
                target=str(row["target"]),
                metadata=metadata,
            )
        )
    return tuple(records)


def _dataset(
    manifest: Mapping[str, Any],
    task_key: str,
    split: str,
    *,
    composition_context: bool = False,
) -> DatasetRef:
    source = manifest["tasks"][task_key].get("source", {})
    source_uri = (
        str(source.get("dataset_id", "frozen_stage1_manifest"))
        if isinstance(source, Mapping)
        else str(source or "frozen_stage1_manifest")
    )
    return DatasetRef(
        dataset_id=f"stage1:{task_key}",
        split=split,
        records=_records(manifest, task_key, split, composition_context=composition_context),
        source_uri=source_uri,
        source_checksum=str(manifest.get("task_hashes", {}).get(task_key, "")),
    )


def stage1_events_from_manifest(
    manifest: Mapping[str, Any],
    *,
    acquisition_steps: int = 120,
    composition_steps: int = 120,
    group_size: int = 4,
) -> Tuple[LearningEvent, ...]:
    seed = int(manifest.get("seed", 1337))
    primitive_gate = float(manifest.get("gates", {}).get("primitive_exact", 0.70))
    composition_gate = float(manifest.get("gates", {}).get("composition_direct_exact", 0.50))
    direct_gain = float(manifest.get("gates", {}).get("composition_direct_gain", 0.30))
    budget = AcquisitionBudget(
        max_optimizer_steps=acquisition_steps,
        max_rollouts=max(4096, acquisition_steps * group_size),
        max_tokens=1_000_000,
        max_wall_seconds=43_200,
        group_size=group_size,
        batch_size=1,
    )
    composition_budget = AcquisitionBudget(
        max_optimizer_steps=composition_steps,
        max_rollouts=max(4096, composition_steps * group_size),
        max_tokens=1_000_000,
        max_wall_seconds=43_200,
        group_size=group_size,
        batch_size=1,
    )
    skill_a = LearningEvent(
        event_id="skill_a",
        revision=0,
        kind="demonstration",
        examples=_dataset(manifest, "skill_a", "train"),
        eval_examples=_dataset(manifest, "skill_a", "eval"),
        targets=TargetRef(visibility="optimizer"),
        gates=GateBundle(
            rules=(GateRule("skill_a", "capability", "capability", "ge", primitive_gate),)
        ),
        budget=budget,
        seed=seed,
        metadata={"stage1_task": "skill_a"},
    )
    skill_b = LearningEvent(
        event_id="skill_b_v1",
        revision=0,
        kind="demonstration",
        examples=_dataset(manifest, "skill_b_v1", "train"),
        eval_examples=_dataset(manifest, "skill_b_v1", "eval"),
        targets=TargetRef(visibility="optimizer"),
        gates=GateBundle(
            rules=(GateRule("skill_b", "capability", "capability", "ge", primitive_gate),)
        ),
        budget=budget,
        seed=seed + 1,
        metadata={"stage1_task": "skill_b_v1"},
    )
    composition_v1 = LearningEvent(
        event_id="composition_v1",
        revision=0,
        kind="reward",
        examples=_dataset(manifest, "composition_direct_v1", "train", composition_context=True),
        eval_examples=_dataset(manifest, "composition_direct_v1", "eval", composition_context=True),
        targets=TargetRef(visibility="verifier_only"),
        verifier=VerifierSpec("exact_match"),
        privileged_context=ContextSpec("two_hop_self"),
        dependencies=("capability:skill_a:r0", "capability:skill_b_v1:r0"),
        gates=GateBundle(
            rules=(
                GateRule("composition_absolute", "capability", "capability", "ge", composition_gate),
                GateRule("composition_gain", "capability", "capability", "delta_ge", direct_gain),
            )
        ),
        budget=composition_budget,
        seed=seed + 2,
        metadata={"stage1_task": "composition_direct_v1", "gold_rollout_rescue": False},
    )
    skill_b_v2 = LearningEvent(
        event_id="skill_b_v2",
        revision=1,
        kind="revision",
        examples=_dataset(manifest, "skill_b_v2_changed", "train"),
        eval_examples=_dataset(manifest, "skill_b_v2_changed", "eval"),
        targets=TargetRef(visibility="optimizer"),
        verifier=VerifierSpec("revision_exact"),
        supersedes=("capability:skill_b_v1:r0",),
        gates=GateBundle(
            rules=(GateRule("skill_b_v2", "staleness", "capability", "ge", 0.75),)
        ),
        budget=budget,
        seed=seed + 3,
        metadata={"stage1_task": "skill_b_v2_changed", "explicit_revision": True},
    )
    composition_v2 = LearningEvent(
        event_id="composition_v2",
        revision=1,
        kind="reward",
        examples=_dataset(manifest, "composition_direct_v2", "train", composition_context=True),
        eval_examples=_dataset(manifest, "composition_direct_v2", "eval", composition_context=True),
        targets=TargetRef(visibility="verifier_only"),
        verifier=VerifierSpec("exact_match"),
        privileged_context=ContextSpec("two_hop_self"),
        dependencies=("capability:skill_a:r0", "capability:skill_b_v2:r1"),
        gates=GateBundle(
            rules=(GateRule("composition_v2", "capability", "capability", "ge", composition_gate),)
        ),
        budget=composition_budget,
        seed=seed + 4,
        metadata={"stage1_task": "composition_direct_v2", "gold_rollout_rescue": False},
    )
    return skill_a, skill_b, composition_v1, skill_b_v2, composition_v2


def write_stage1_events(
    manifest_path: Path,
    output_dir: Path,
    *,
    acquisition_steps: int = 120,
    composition_steps: int = 120,
    group_size: int = 4,
) -> Tuple[Path, ...]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = stage1_events_from_manifest(
        manifest,
        acquisition_steps=acquisition_steps,
        composition_steps=composition_steps,
        group_size=group_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, event in enumerate(events):
        path = output_dir / f"{index:02d}_{event.event_id}_r{event.revision}.json"
        from ._io import atomic_write_json

        atomic_write_json(path, event.to_dict())
        paths.append(path)
    return tuple(paths)


def history_profile_events_from_manifest(manifest: Mapping[str, Any]) -> Tuple[LearningEvent, ...]:
    """Convert a frozen five-skill/shen3 history manifest into profile-only events."""
    events: List[LearningEvent] = []
    seed = int(manifest.get("seed", 1337))
    for index, payload in enumerate(manifest.get("skills", ())):
        recipe = payload.get("recipe", {})
        name = str(recipe.get("name") or payload.get("name") or f"history_{index}")
        train_rows = payload.get("train", ())
        eval_rows = payload.get("eval", ())
        if not train_rows or not eval_rows:
            continue

        def convert(rows: Sequence[Mapping[str, Any]], split: str) -> Tuple[ExampleRecord, ...]:
            converted = []
            for row_index, item in enumerate(rows):
                converted.append(
                    ExampleRecord(
                        example_id=str(item.get("row_id", f"history:{name}:{split}:{row_index}")),
                        prompt=str(item.get("prompt") or item.get("source") or ""),
                        target=str(item.get("target") or item.get("raw_target") or ""),
                        metadata={"history_skill": name},
                    )
                )
            return tuple(converted)

        source = str(payload.get("source", "frozen_history_manifest"))
        train = DatasetRef(
            dataset_id=f"history:{name}",
            split="train",
            records=convert(train_rows, "train"),
            source_uri=source,
            source_checksum=sha256_json(payload),
        )
        evaluation = DatasetRef(
            dataset_id=f"history:{name}",
            split="eval",
            records=convert(eval_rows, "eval"),
            source_uri=source,
            source_checksum=sha256_json(payload),
        )
        events.append(
            LearningEvent(
                event_id=f"history_{name}",
                revision=0,
                kind="evaluation",
                examples=train,
                eval_examples=evaluation,
                targets=TargetRef(visibility="verifier_only"),
                verifier=VerifierSpec("exact_match"),
                gates=GateBundle(
                    rules=(GateRule(f"history_{name}_finite", "capability", "capability", "finite"),),
                    require_staleness_on_revision=False,
                ),
                seed=seed + index,
                metadata={"profile_only": True, "history_skill": name},
            )
        )
    if not events:
        raise ValueError("history manifest contains no skills with both train and eval rows")
    return tuple(events)


def write_history_profile_events(manifest_path: Path, output_dir: Path) -> Tuple[Path, ...]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = history_profile_events_from_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    from ._io import atomic_write_json

    for index, event in enumerate(events):
        path = output_dir / f"{index:02d}_{event.event_id}.json"
        atomic_write_json(path, event.to_dict())
        paths.append(path)
    return tuple(paths)


def write_general_canary_event(canary_path: Path, output_path: Path, *, seed: int = 1337) -> Path:
    payload = json.loads(canary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("general canary must be a non-empty JSON list")
    records = tuple(
        ExampleRecord(
            example_id=f"base_general:{index:04d}",
            prompt=str(item["prompt"]),
            target=str(item["target"]),
            metadata={"canary_family": str(item.get("family", "general"))},
        )
        for index, item in enumerate(payload)
    )
    dataset = DatasetRef(
        dataset_id="base_general_canary",
        split="eval",
        records=records,
        source_uri=str(canary_path),
        source_checksum=sha256_file(canary_path),
    )
    event = LearningEvent(
        event_id="base_general",
        revision=0,
        kind="evaluation",
        examples=dataset,
        eval_examples=dataset,
        targets=TargetRef(visibility="verifier_only"),
        verifier=VerifierSpec("exact_match"),
        gates=GateBundle(
            rules=(GateRule("base_general_finite", "capability", "capability", "finite"),),
            require_staleness_on_revision=False,
        ),
        seed=int(seed),
        metadata={"profile_only": True, "base_general": True},
    )
    from ._io import atomic_write_json

    atomic_write_json(output_path, event.to_dict())
    return output_path


class TwoHopSelfContextProvider:
    """Builds privileged context only from the live policy's two primitive predictions."""

    def __init__(self, max_new_tokens: int = 10) -> None:
        self.max_new_tokens = int(max_new_tokens)

    def build(self, example: PublicExample, mode: str, policy: Any) -> ContextResult:
        if mode == "none":
            return ContextResult(example.prompt, mode)
        hop_a_prompt = str(example.metadata["hop_a_prompt"])
        hop_b_template = str(example.metadata["hop_b_prompt_template"])
        sampling = SamplingConfig(
            group_size=1,
            temperature=0.1,
            top_p=1.0,
            max_new_tokens=self.max_new_tokens,
            seed=0,
        )
        first = policy.generate(hop_a_prompt, sampling, seed=17).completion.strip()
        second_prompt = hop_b_template.format(intermediate=first)
        second = policy.generate(second_prompt, sampling, seed=23).completion.strip()
        if mode == "full":
            prompt = (
                f"{example.prompt}\n\nPrivileged route generated by the current model:\n"
                f"INTERMEDIATE={first}; FINAL={second}\nReturn only the final answer."
            )
        elif mode == "compressed":
            prompt = f"{example.prompt}\n\nSelf-route: I={first}; F={second}"
        else:
            raise ValueError(f"unsupported two-hop context mode={mode}")
        quality = 1.0 if first and second else 0.0
        return ContextResult(
            prompt,
            mode,
            quality=quality,
            metadata={"first_length": len(first), "second_length": len(second)},
        )


@register_context("two_hop_self")
def _two_hop_factory(spec: ContextSpec) -> TwoHopSelfContextProvider:
    return TwoHopSelfContextProvider(max_new_tokens=int(spec.config.get("max_new_tokens", 10)))
