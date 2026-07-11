from __future__ import annotations

import math
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from ._io import atomic_write_json, sha256_json
from .algorithms import DemonstrationAlgorithm, GroupRelativeAlgorithm, HybridAlgorithm
from .artifacts import AcquisitionArtifact
from .contexts import ContextProvider, NoContextProvider, deterministic_context_modes
from .events import LearningEvent
from .profiles import ProfileRecord
from .trajectories import (
    PolicyVersion,
    RolloutGroup,
    SamplingConfig,
    Trajectory,
    TrajectoryLedger,
    TrainingSample,
)
from .verifiers import RewardResult, Verifier, score_sync


@dataclass(frozen=True)
class PolicyOutput:
    completion: str
    sample: TrainingSample
    entropy: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyStepMetrics:
    loss: float
    policy_loss: float = 0.0
    ce_loss: float = 0.0
    reference_kl: float = 0.0
    anchor_kl: float = 0.0
    entropy: float = 0.0
    grad_norm: float = 0.0

    def __post_init__(self) -> None:
        values = asdict(self).values()
        if any(not math.isfinite(float(value)) for value in values):
            raise FloatingPointError("policy update produced non-finite metrics")


@dataclass(frozen=True)
class PolicyUpdateConfig:
    kl_coefficient: float = 0.02
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.0
    grad_clip: float = 0.3


class TemporaryPolicy(Protocol):
    @property
    def version(self) -> PolicyVersion:
        ...

    @property
    def base_policy_hash(self) -> str:
        ...

    @property
    def candidate_layers(self) -> Sequence[int]:
        ...

    def generate(self, prompt: str, sampling: SamplingConfig, *, seed: int) -> PolicyOutput:
        ...

    def supervised_sample(self, prompt: str, target: str) -> TrainingSample:
        ...

    def reference_logprobs(self, prompt: str, completion: str, *, token_ids: Sequence[int]) -> Sequence[float]:
        ...

    def update(self, trajectories: Sequence[Trajectory], config: PolicyUpdateConfig) -> PolicyStepMetrics:
        ...

    def save_temporary(self, path: Path) -> None:
        ...

    def save_resume(self, path: Path) -> None:
        ...

    def load_resume(self, path: Path) -> None:
        ...


@dataclass(frozen=True)
class AcquisitionConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    update: PolicyUpdateConfig = field(default_factory=PolicyUpdateConfig)
    context_mixture: Mapping[str, float] = field(
        default_factory=lambda: {"full": 0.4, "compressed": 0.3, "none": 0.3}
    )
    context_kl_weight: float = 1.0
    target_anchor_kl: float = 0.02
    adaptive_kl: bool = True
    save_interval: int = 25


class OnPolicyAcquirer:
    def __init__(self, config: Optional[AcquisitionConfig] = None) -> None:
        self.config = config or AcquisitionConfig()

    def acquire(
        self,
        *,
        policy: TemporaryPolicy,
        event: LearningEvent,
        verifier: Verifier,
        output_dir: Path,
        protected_profiles: Sequence[ProfileRecord] = (),
        context_provider: Optional[ContextProvider] = None,
        step_offset: int = 0,
        max_steps: Optional[int] = None,
    ) -> AcquisitionArtifact:
        del protected_profiles  # The backend owns anchor/profile regularization; kept in the stable public API.
        output_dir.mkdir(parents=True, exist_ok=True)
        final_artifact = output_dir / "acquisition.json"
        progress_path = output_dir / "progress.json"
        resume_policy = output_dir / "resume_policy"
        if final_artifact.exists() and progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("completed"):
                if not resume_policy.exists():
                    raise RuntimeError("completed acquisition is missing its resumable policy state")
                policy.load_resume(resume_policy)
                return AcquisitionArtifact.from_dict(json.loads(final_artifact.read_text(encoding="utf-8")))
        ledger = TrajectoryLedger(output_dir / "trajectories.jsonl")
        has_privileged_context = context_provider is not None
        context_provider = context_provider or NoContextProvider()
        budget = event.budget
        total_steps = min(max_steps or budget.max_optimizer_steps, budget.max_optimizer_steps)
        total_slots = total_steps * budget.batch_size
        modes = (
            deterministic_context_modes(total_slots, self.config.context_mixture, event.event_key)
            if has_privileged_context
            else ("none",) * total_slots
        )
        records = list(event.examples.records)
        started = time.monotonic()
        start_step = 0
        rollout_count = token_count = successes = valid = zero_advantage_groups = 0
        generation_errors = verifier_errors = 0
        context_quality: List[float] = []
        step_metrics: List[PolicyStepMetrics] = []
        kl_coefficient = float(self.config.update.kl_coefficient)
        used_ids: List[str] = []
        if progress_path.exists() != resume_policy.exists():
            raise RuntimeError("incomplete acquisition resume checkpoint")
        if progress_path.exists() and resume_policy.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            start_step = int(progress.get("step", 0))
            policy.load_resume(resume_policy)
            rollout_count = int(progress.get("rollout_count", 0))
            token_count = int(progress.get("token_count", 0))
            successes = int(progress.get("successes", 0))
            valid = int(progress.get("valid", 0))
            zero_advantage_groups = int(progress.get("zero_advantage_groups", 0))
            generation_errors = int(progress.get("generation_errors", 0))
            verifier_errors = int(progress.get("verifier_errors", 0))
            context_quality = [float(value) for value in progress.get("context_quality", ())]
            step_metrics = [PolicyStepMetrics(**item) for item in progress.get("step_metrics", ())]
            kl_coefficient = float(progress.get("kl_coefficient", kl_coefficient))
            used_ids = [str(value) for value in progress.get("used_ids", ())]
            _rewind_ledger(ledger.path, start_step)
        else:
            _rewind_ledger(ledger.path, 0)

        last_step = start_step
        for local_step in range(start_step + 1, total_steps + 1):
            if time.monotonic() - started > budget.max_wall_seconds:
                break
            step_rng = random.Random(event.seed + 104729 * (local_step + step_offset))
            selected = [records[step_rng.randrange(len(records))] for _ in range(budget.batch_size)]
            trainable: List[Trajectory] = []
            for batch_index, record in enumerate(selected):
                mode_index = (local_step - 1) * budget.batch_size + batch_index
                mode = modes[mode_index] if mode_index < len(modes) else "none"
                context = context_provider.build(record.public(), mode, policy)
                context_quality.append(float(context.quality))
                version = policy.version
                group_id = sha256_json(
                    {"event": event.event_key, "step": local_step + step_offset, "example": record.example_id}
                )[:24]
                group = RolloutGroup(event.event_key, record.example_id, group_id, version)
                for rollout_index in range(budget.group_size):
                    if rollout_count >= budget.max_rollouts or token_count >= budget.max_tokens:
                        break
                    seed = event.seed + 1_000_003 * (local_step + step_offset) + 997 * batch_index + rollout_index
                    try:
                        output = policy.generate(record.prompt, self.config.sampling, seed=seed)
                    except Exception as exc:
                        generation_errors += 1
                        rollout_count += 1
                        trajectory = Trajectory.new(
                            event_key=event.event_key,
                            example_id=record.example_id,
                            group_id=group_id,
                            rollout_id=sha256_json(
                                {"group": group_id, "rollout": rollout_index, "policy": str(version)}
                            )[:32],
                            policy_version=version,
                            prompt=record.prompt,
                            completion="",
                            sample=None,
                            context_mode=mode,
                            context_fingerprint=context.fingerprint if mode != "none" else "",
                            entropy=0.0,
                            sampling=asdict(self.config.sampling),
                            metadata={"acquisition_step": local_step, "rollout_error": True},
                            error=f"generation:{type(exc).__name__}",
                        )
                        try:
                            result = score_sync(verifier, record, trajectory.verifier_view())
                        except Exception as verifier_exc:
                            verifier_errors += 1
                            result = RewardResult(
                                -0.25,
                                False,
                                False,
                                components={"rollout_error": 1.0},
                                diagnostics={"error_type": type(verifier_exc).__name__},
                            )
                        trajectory.reward = result
                        successes += int(result.success)
                        valid += int(result.valid)
                        group.add(trajectory)
                        continue
                    sample = output.sample
                    if mode != "none":
                        reference = list(
                            policy.reference_logprobs(
                                context.prompt,
                                output.completion,
                                token_ids=sample.token_ids,
                            )
                        )
                        if len(reference) != len(sample.token_ids):
                            raise ValueError("context reference logprobs do not align with the rollout tokens")
                        sample.reference_logprobs = reference
                    trajectory = Trajectory.new(
                        event_key=event.event_key,
                        example_id=record.example_id,
                        group_id=group_id,
                        rollout_id=sha256_json(
                            {"group": group_id, "rollout": rollout_index, "policy": str(version)}
                        )[:32],
                        policy_version=version,
                        prompt=record.prompt,
                        completion=output.completion,
                        sample=sample,
                        context_mode=mode,
                        context_fingerprint=context.fingerprint if mode != "none" else "",
                        entropy=float(output.entropy),
                        sampling=asdict(self.config.sampling),
                        metadata={"output": dict(output.metadata), "acquisition_step": local_step},
                    )
                    try:
                        result = score_sync(verifier, record, trajectory.verifier_view())
                    except Exception as exc:
                        verifier_errors += 1
                        trajectory.error = f"verifier:{type(exc).__name__}"
                        result = RewardResult(
                            -0.25,
                            False,
                            False,
                            components={"verifier_error": 1.0},
                            diagnostics={"error_type": type(exc).__name__},
                        )
                    trajectory.reward = result
                    successes += int(result.success)
                    valid += int(result.valid)
                    rollout_count += 1
                    token_count += sum(bool(value) for value in sample.completion_mask)
                    group.add(trajectory)
                if policy.version != version:
                    raise RuntimeError("policy changed while a rollout group was in flight")
                if not group.trajectories:
                    continue
                if any(item.context_mode != "none" for item in group.trajectories):
                    algorithm = HybridAlgorithm(reference_weight=self.config.context_kl_weight)
                else:
                    algorithm = GroupRelativeAlgorithm()
                metrics = algorithm.finalize_group(group)
                zero_advantage_groups += int(metrics.zero_advantage)
                ledger.append_group(group)
                trainable.extend(item for item in group.trajectories if item.is_trainable)
                used_ids.append(record.example_id)
            last_step = local_step
            if trainable:
                before = policy.version
                update_config = replace(self.config.update, kl_coefficient=kl_coefficient)
                metrics = policy.update(trainable, update_config)
                if policy.version <= before:
                    raise RuntimeError("temporary policy backend did not increment its version after an update")
                step_metrics.append(metrics)
                if self.config.adaptive_kl and self.config.target_anchor_kl > 0:
                    if metrics.anchor_kl > self.config.target_anchor_kl * 1.5:
                        kl_coefficient = min(10.0, kl_coefficient * 1.5)
                    elif metrics.anchor_kl < self.config.target_anchor_kl / 1.5:
                        kl_coefficient = max(1e-6, kl_coefficient / 1.5)
            if local_step % self.config.save_interval == 0:
                policy.save_resume(resume_policy)
                _save_progress(
                    progress_path,
                    step=local_step,
                    rollout_count=rollout_count,
                    token_count=token_count,
                    successes=successes,
                    valid=valid,
                    zero_advantage_groups=zero_advantage_groups,
                    generation_errors=generation_errors,
                    verifier_errors=verifier_errors,
                    context_quality=context_quality,
                    step_metrics=step_metrics,
                    kl_coefficient=kl_coefficient,
                    used_ids=used_ids,
                )
            if rollout_count >= budget.max_rollouts or token_count >= budget.max_tokens:
                break

        policy_path = output_dir / "temporary_policy"
        policy.save_temporary(policy_path)
        elapsed = time.monotonic() - started
        metrics_payload: Dict[str, float] = {
            "optimizer_steps": float(len(step_metrics)),
            "rollouts": float(rollout_count),
            "completion_tokens": float(token_count),
            "success_rate": float(successes / rollout_count) if rollout_count else 0.0,
            "valid_rate": float(valid / rollout_count) if rollout_count else 0.0,
            "zero_advantage_groups": float(zero_advantage_groups),
            "generation_errors": float(generation_errors),
            "verifier_errors": float(verifier_errors),
            "mean_entropy": _mean([item.entropy for item in step_metrics]),
            "mean_anchor_kl": _mean([item.anchor_kl for item in step_metrics]),
            "mean_reference_kl": _mean([item.reference_kl for item in step_metrics]),
            "mean_context_quality": _mean(context_quality),
            "wall_seconds": float(elapsed),
            "final_kl_coefficient": float(kl_coefficient),
        }
        atomic_write_json(output_dir / "metrics.json", metrics_payload)
        access_hash = sha256_json(sorted(set(used_ids)))
        artifact = AcquisitionArtifact(
            event_key=event.event_key,
            route="reward",
            base_policy_version=policy.version.committed,
            base_policy_hash=policy.base_policy_hash,
            temporary_policy_version=str(policy.version),
            adapter_path=str(policy_path),
            sample_ledger_path=str(ledger.path),
            example_ids=tuple(sorted(set(used_ids))),
            candidate_layers=tuple(sorted(set(int(value) for value in policy.candidate_layers))),
            metrics=metrics_payload,
            access_log_hash=access_hash,
            contains_optimizer_targets=False,
            metadata={"strict_policy_lag": 0, "all_outcomes_retained": True},
        )
        artifact.save(output_dir / "acquisition.json")
        policy.save_resume(resume_policy)
        _save_progress(
            progress_path,
            step=last_step,
            rollout_count=rollout_count,
            token_count=token_count,
            successes=successes,
            valid=valid,
            zero_advantage_groups=zero_advantage_groups,
            generation_errors=generation_errors,
            verifier_errors=verifier_errors,
            context_quality=context_quality,
            step_metrics=step_metrics,
            kl_coefficient=kl_coefficient,
            used_ids=used_ids,
            completed=True,
        )
        return artifact


class DemonstrationAcquirer:
    def __init__(self, config: Optional[AcquisitionConfig] = None) -> None:
        self.config = config or AcquisitionConfig()

    def acquire(
        self,
        *,
        policy: TemporaryPolicy,
        event: LearningEvent,
        output_dir: Path,
        max_steps: Optional[int] = None,
    ) -> AcquisitionArtifact:
        if event.targets is None or event.targets.visibility != "optimizer":
            raise ValueError("demonstration acquisition requires optimizer-visible targets")
        output_dir.mkdir(parents=True, exist_ok=True)
        final_artifact = output_dir / "acquisition.json"
        progress_path = output_dir / "progress.json"
        resume_policy = output_dir / "resume_policy"
        if final_artifact.exists() and progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("completed"):
                if not resume_policy.exists():
                    raise RuntimeError("completed demonstration is missing its resumable policy state")
                policy.load_resume(resume_policy)
                return AcquisitionArtifact.from_dict(json.loads(final_artifact.read_text(encoding="utf-8")))
        ledger = TrajectoryLedger(output_dir / "demonstrations.jsonl")
        budget = event.budget
        steps = min(max_steps or budget.max_optimizer_steps, budget.max_optimizer_steps)
        records = list(event.examples.records)
        used_ids: List[str] = []
        step_metrics: List[PolicyStepMetrics] = []
        started = time.monotonic()
        token_count = 0
        start_step = 0
        if progress_path.exists() != resume_policy.exists():
            raise RuntimeError("incomplete demonstration resume checkpoint")
        if progress_path.exists() and resume_policy.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            start_step = int(progress.get("step", 0))
            policy.load_resume(resume_policy)
            used_ids = [str(value) for value in progress.get("used_ids", ())]
            token_count = int(progress.get("token_count", 0))
            step_metrics = [PolicyStepMetrics(**item) for item in progress.get("step_metrics", ())]
            _rewind_ledger(ledger.path, start_step)
        else:
            _rewind_ledger(ledger.path, 0)
        last_step = start_step
        for step in range(start_step + 1, steps + 1):
            if time.monotonic() - started > budget.max_wall_seconds:
                break
            step_rng = random.Random(event.seed + 104729 * step)
            trajectories: List[Trajectory] = []
            for batch_index in range(budget.batch_size):
                record = records[step_rng.randrange(len(records))]
                if record.target is None:
                    raise ValueError(f"demonstration example {record.example_id} has no target")
                sample = policy.supervised_sample(record.prompt, record.target)
                version = policy.version
                group_id = sha256_json(
                    {"event": event.event_key, "step": step, "example": record.example_id, "demo": True}
                )[:24]
                trajectory = Trajectory.new(
                    event_key=event.event_key,
                    example_id=record.example_id,
                    group_id=group_id,
                    rollout_id=sha256_json({"group": group_id, "batch": batch_index})[:32],
                    policy_version=version,
                    prompt=record.prompt,
                    completion=record.target,
                    sample=sample,
                    context_mode="demonstration",
                    entropy=0.0,
                    sampling={"teacher_forced": True},
                    metadata={"acquisition_step": step},
                )
                group = RolloutGroup(event.event_key, record.example_id, group_id, version, [trajectory])
                DemonstrationAlgorithm().finalize_group(group)
                ledger.append(trajectory)
                trajectories.append(trajectory)
                used_ids.append(record.example_id)
                token_count += sum(sample.completion_mask)
            before = policy.version
            metrics = policy.update(trajectories, self.config.update)
            if policy.version <= before:
                raise RuntimeError("temporary policy backend did not increment after demonstration update")
            step_metrics.append(metrics)
            last_step = step
            if step % self.config.save_interval == 0:
                policy.save_resume(resume_policy)
                _save_progress(
                    progress_path,
                    step=step,
                    token_count=token_count,
                    step_metrics=step_metrics,
                    used_ids=used_ids,
                )
        policy_path = output_dir / "temporary_policy"
        policy.save_temporary(policy_path)
        elapsed = time.monotonic() - started
        metrics_payload = {
            "optimizer_steps": float(len(step_metrics)),
            "demonstrations": float(len(used_ids)),
            "completion_tokens": float(token_count),
            "mean_loss": _mean([item.loss for item in step_metrics]),
            "mean_ce_loss": _mean([item.ce_loss for item in step_metrics]),
            "wall_seconds": float(elapsed),
        }
        atomic_write_json(output_dir / "metrics.json", metrics_payload)
        artifact = AcquisitionArtifact(
            event_key=event.event_key,
            route="demonstration",
            base_policy_version=policy.version.committed,
            base_policy_hash=policy.base_policy_hash,
            temporary_policy_version=str(policy.version),
            adapter_path=str(policy_path),
            sample_ledger_path=str(ledger.path),
            example_ids=tuple(sorted(set(used_ids))),
            candidate_layers=tuple(sorted(set(int(value) for value in policy.candidate_layers))),
            metrics=metrics_payload,
            access_log_hash=sha256_json(sorted(set(used_ids))),
            contains_optimizer_targets=True,
            metadata={"teacher_forced": True},
        )
        artifact.save(output_dir / "acquisition.json")
        policy.save_resume(resume_policy)
        _save_progress(
            progress_path,
            step=last_step,
            token_count=token_count,
            step_metrics=step_metrics,
            used_ids=used_ids,
            completed=True,
        )
        return artifact


def acquire_from_reward(
    model: TemporaryPolicy,
    event: LearningEvent,
    verifier: Verifier,
    rollout_policy: Optional[AcquisitionConfig],
    protected_profiles: Sequence[ProfileRecord],
    *,
    output_dir: Path,
    context_provider: Optional[ContextProvider] = None,
) -> AcquisitionArtifact:
    return OnPolicyAcquirer(rollout_policy).acquire(
        policy=model,
        event=event,
        verifier=verifier,
        output_dir=output_dir,
        protected_profiles=protected_profiles,
        context_provider=context_provider,
    )


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _save_progress(path: Path, *, completed: bool = False, **values: Any) -> None:
    payload = dict(values)
    if "step_metrics" in payload:
        payload["step_metrics"] = [asdict(item) for item in payload["step_metrics"]]
    payload["completed"] = bool(completed)
    atomic_write_json(path, payload)


def _rewind_ledger(path: Path, step: int) -> None:
    if not path.exists():
        return
    retained = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        observed = int(row.get("metadata", {}).get("acquisition_step", 0))
        if observed <= int(step):
            retained.append(row)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
