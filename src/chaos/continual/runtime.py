from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ._io import atomic_write_json, sha256_json
from .acquisition import PolicyOutput, PolicyStepMetrics, PolicyUpdateConfig, TemporaryPolicy
from .artifacts import AcquisitionArtifact, CandidateArtifact
from .commit_store import CurrentVersion, TransactionHandle
from .consolidation import ConsolidationResult, RuntimeEvaluation
from .events import ExampleRecord, LearningEvent
from .evaluator import SystemChecks
from .geometry import GeometryMeasurement, GeometryPlan, LayerMeasurement
from .profiles import ProfileRecord, ProfileRegistry
from .trajectories import PolicyVersion, SamplingConfig, Trajectory, TrainingSample
from .verifiers import build_verifier, score_sync


class TabularTemporaryPolicy(TemporaryPolicy):
    """Deterministic CPU reference backend used to test the full transaction engine.

    It is intentionally small, but it performs real grouped policy updates over a
    categorical policy. Production model backends implement the same interface.
    """

    def __init__(
        self,
        *,
        choices: Sequence[str],
        committed_version: int,
        base_policy_hash: str,
        seed: int = 0,
        learning_rate: float = 0.25,
        candidate_layers: Sequence[int] = (0, 1),
        state: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> None:
        if not choices:
            raise ValueError("tabular policy requires at least one output choice")
        self.choices = tuple(str(value) for value in choices)
        self._version = PolicyVersion(int(committed_version), 1, 0)
        self._base_policy_hash = str(base_policy_hash)
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self._candidate_layers = tuple(int(value) for value in candidate_layers)
        self.logits: Dict[str, Dict[str, float]] = {
            str(prompt): {str(choice): float(score) for choice, score in scores.items()}
            for prompt, scores in (state or {}).items()
        }

    @property
    def version(self) -> PolicyVersion:
        return self._version

    @property
    def base_policy_hash(self) -> str:
        return self._base_policy_hash

    @property
    def candidate_layers(self) -> Sequence[int]:
        return self._candidate_layers

    def _scores(self, prompt: str) -> Dict[str, float]:
        return self.logits.setdefault(prompt, {choice: 0.0 for choice in self.choices})

    def _probabilities(self, prompt: str) -> Dict[str, float]:
        scores = self._scores(prompt)
        maximum = max(scores.values())
        exp = {key: math.exp(value - maximum) for key, value in scores.items()}
        total = sum(exp.values())
        return {key: value / total for key, value in exp.items()}

    def generate(self, prompt: str, sampling: SamplingConfig, *, seed: int) -> PolicyOutput:
        probabilities = self._probabilities(prompt)
        rng = random.Random(self.seed + int(seed))
        threshold = rng.random()
        cumulative = 0.0
        choice = self.choices[-1]
        for candidate in self.choices:
            cumulative += probabilities[candidate]
            if threshold <= cumulative:
                choice = candidate
                break
        choice_id = self.choices.index(choice) + 1
        logprob = math.log(max(probabilities[choice], 1e-12))
        entropy = -sum(value * math.log(max(value, 1e-12)) for value in probabilities.values())
        sample = TrainingSample(
            token_ids=[choice_id],
            completion_mask=[True],
            rollout_logprobs=[logprob],
        )
        return PolicyOutput(choice, sample, entropy, {"backend": "tabular"})

    def supervised_sample(self, prompt: str, target: str) -> TrainingSample:
        if target not in self.choices:
            raise ValueError(f"target {target!r} is not in the tabular policy choices")
        probabilities = self._probabilities(prompt)
        return TrainingSample(
            token_ids=[self.choices.index(target) + 1],
            completion_mask=[True],
            rollout_logprobs=[math.log(max(probabilities[target], 1e-12))],
        )

    def reference_logprobs(self, prompt: str, completion: str, *, token_ids: Sequence[int]) -> Sequence[float]:
        probabilities = self._probabilities(prompt)
        value = math.log(max(probabilities.get(completion, 1e-12), 1e-12))
        return [value if index < len(token_ids) else 0.0 for index in range(len(token_ids))]

    def update(self, trajectories: Sequence[Trajectory], config: PolicyUpdateConfig) -> PolicyStepMetrics:
        if not trajectories:
            raise ValueError("tabular update requires trajectories")
        losses: List[float] = []
        entropies: List[float] = []
        for trajectory in trajectories:
            if trajectory.sample is None or trajectory.completion not in self.choices:
                continue
            scores = self._scores(trajectory.prompt)
            probabilities = self._probabilities(trajectory.prompt)
            chosen = trajectory.completion
            advantage = trajectory.sample.advantages[-1] if trajectory.sample.advantages else 0.0
            ce_weight = trajectory.sample.ce_weights[-1] if trajectory.sample.ce_weights else 0.0
            ref_weight = (
                trajectory.sample.reference_kl_weights[-1]
                if trajectory.sample.reference_kl_weights
                else 0.0
            )
            signal = advantage + ce_weight + ref_weight
            for choice in self.choices:
                gradient = (1.0 if choice == chosen else 0.0) - probabilities[choice]
                scores[choice] += self.learning_rate * signal * gradient
            losses.append(-signal * math.log(max(probabilities[chosen], 1e-12)))
            entropies.append(
                -sum(value * math.log(max(value, 1e-12)) for value in probabilities.values())
            )
        self._version = PolicyVersion(
            self._version.committed,
            self._version.attempt,
            self._version.update + 1,
        )
        loss = sum(losses) / len(losses) if losses else 0.0
        return PolicyStepMetrics(
            loss=loss,
            policy_loss=loss,
            ce_loss=loss,
            reference_kl=0.0,
            anchor_kl=min(abs(loss) * 0.001, 1.0),
            entropy=sum(entropies) / len(entropies) if entropies else 0.0,
            grad_norm=abs(loss),
        )

    def save_temporary(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            path / "tabular_policy.json",
            {
                "choices": self.choices,
                "version": asdict(self._version),
                "base_policy_hash": self._base_policy_hash,
                "seed": self.seed,
                "learning_rate": self.learning_rate,
                "candidate_layers": self._candidate_layers,
                "logits": self.logits,
            },
        )

    def save_resume(self, path: Path) -> None:
        self.save_temporary(path)

    def load_resume(self, path: Path) -> None:
        payload = json.loads((path / "tabular_policy.json").read_text(encoding="utf-8"))
        self.logits = {
            str(prompt): {str(choice): float(score) for choice, score in scores.items()}
            for prompt, scores in payload.get("logits", {}).items()
        }
        self._version = PolicyVersion(**payload["version"])

    @classmethod
    def load(cls, path: Path) -> "TabularTemporaryPolicy":
        payload = json.loads((path / "tabular_policy.json").read_text(encoding="utf-8"))
        version = payload.pop("version")
        return cls(
            choices=payload["choices"],
            committed_version=version["committed"],
            base_policy_hash=payload["base_policy_hash"],
            seed=payload["seed"],
            learning_rate=payload["learning_rate"],
            candidate_layers=payload["candidate_layers"],
            state=payload["logits"],
        )


class TabularContinualRuntime:
    """Complete reference runtime for transaction, stream, and controller tests."""

    def __init__(
        self,
        *,
        choices: Sequence[str],
        seed: int = 0,
        learning_rate: float = 0.4,
        dimensions_per_layer: int = 16,
    ) -> None:
        self.choices = tuple(choices)
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self.dimensions_per_layer = int(dimensions_per_layer)

    def initialize_model(self, path: Path) -> Path:
        policy = TabularTemporaryPolicy(
            choices=self.choices,
            committed_version=0,
            base_policy_hash=sha256_json({"choices": self.choices, "seed": self.seed}),
            seed=self.seed,
            learning_rate=self.learning_rate,
        )
        policy.save_temporary(path)
        return path

    def open_temporary_policy(
        self,
        current: CurrentVersion,
        event: LearningEvent,
        transaction: TransactionHandle,
    ) -> TabularTemporaryPolicy:
        del transaction
        state = self._state(Path(current.model_path))
        return TabularTemporaryPolicy(
            choices=state.get("choices", self.choices),
            committed_version=current.version,
            base_policy_hash=current.commit_hash,
            seed=event.seed,
            learning_rate=self.learning_rate,
            candidate_layers=state.get("candidate_layers", (0, 1)),
            state=state.get("logits", {}),
        )

    def profile_existing(
        self,
        current: CurrentVersion,
        event: LearningEvent,
        registry: ProfileRegistry,
        transaction: TransactionHandle,
    ) -> ConsolidationResult:
        policy = self._policy_for_path(Path(current.model_path), current.version, current.commit_hash)
        metrics = self._evaluate_policy(policy, event, (event.eval_examples or event.examples).records)
        candidate_registry = registry.clone()
        profile_id = f"capability:{event.event_id}:r{event.revision}"
        candidate_registry.register(
            ProfileRecord(
                profile_id=profile_id,
                capability=event.event_id,
                dependencies=tuple(event.dependencies),
                selected_layers=tuple(policy.candidate_layers),
                checkpoint_version=current.version + 1,
                checkpoint_hash=current.commit_hash,
                creation_event=event.event_key,
                scope={"profile_only": "true"},
                metrics={"baseline_capability": metrics["capability"], "effective_rank": 1.0},
                canary_dataset=(event.eval_examples or event.examples).dataset_id,
            )
        )
        registry_path = transaction.root / "candidate" / "registry.json"
        candidate_registry.save(registry_path)
        metrics_path = transaction.root / "candidate" / "profile_metrics.json"
        atomic_write_json(metrics_path, metrics)
        candidate = CandidateArtifact(
            event_key=event.event_key,
            model_path=current.model_path,
            registry_path=str(registry_path),
            metrics_path=str(metrics_path),
            source_policy_version=current.version,
            selected_layers=tuple(policy.candidate_layers),
            profile_ids=(profile_id,),
            numerical_stable=True,
            metadata={"runtime": "tabular_reference", "reuse_current_model": True},
        )
        return ConsolidationResult(candidate, candidate_registry)

    def measure_geometry(
        self,
        current: CurrentVersion,
        event: LearningEvent,
        acquisition: AcquisitionArtifact,
        registry: ProfileRegistry,
        transaction: TransactionHandle,
    ) -> GeometryMeasurement:
        del transaction
        before = self._state(Path(current.model_path)).get("logits", {})
        after = self._state(Path(acquisition.adapter_path)).get("logits", {})
        pressure = self._delta_pressure(before, after)
        layers = acquisition.candidate_layers or (0, 1)
        measurements: List[LayerMeasurement] = []
        active = registry.active()
        for layer in layers:
            occupying = [record for record in active if int(layer) in record.selected_layers]
            occupied_rank = min(self.dimensions_per_layer, len(occupying))
            free = 1.0 - occupied_rank / float(self.dimensions_per_layer)
            overlaps = {record.profile_id: 1.0 / max(1, len(occupying)) for record in occupying}
            conflicts = {
                record.profile_id: 1.0 if record.profile_id in event.supersedes else 0.0
                for record in occupying
            }
            measurements.append(
                LayerMeasurement(
                    layer=int(layer),
                    pressure=pressure / max(1, len(layers)),
                    residual_energy=max(0.0, free),
                    occupied_rank=occupied_rank,
                    dimension=self.dimensions_per_layer,
                    profile_overlaps=overlaps,
                    directional_conflicts=conflicts,
                )
            )
        return GeometryMeasurement(
            event_key=event.event_key,
            layers=tuple(measurements),
            source_policy_hash=current.commit_hash,
            acquisition_hash=acquisition.fingerprint,
        )

    def consolidate(
        self,
        current: CurrentVersion,
        event: LearningEvent,
        acquisition: AcquisitionArtifact,
        plan: GeometryPlan,
        registry: ProfileRegistry,
        transaction: TransactionHandle,
    ) -> ConsolidationResult:
        candidate_model = transaction.candidate_model_dir
        if candidate_model.exists():
            shutil.rmtree(candidate_model)
        shutil.copytree(Path(acquisition.adapter_path), candidate_model)
        candidate_registry = registry.clone()
        if plan.release_profile_ids:
            released = candidate_registry.release_closure(event.supersedes)
            candidate_registry.retire(released)
        profile_id = f"capability:{event.event_id}:r{event.revision}"
        candidate_registry.register(
            ProfileRecord(
                profile_id=profile_id,
                capability=event.event_id,
                dependencies=tuple(event.dependencies),
                selected_layers=tuple(plan.selected_layers),
                checkpoint_version=current.version + 1,
                checkpoint_hash=acquisition.fingerprint,
                creation_event=event.event_key,
                status="protected",
                scope={"event_kind": event.kind},
                metrics=dict(acquisition.metrics),
                canary_dataset=(event.eval_examples or event.examples).dataset_id,
            )
        )
        registry_path = transaction.root / "candidate" / "registry.json"
        candidate_registry.save(registry_path)
        metrics_path = transaction.root / "candidate" / "consolidation_metrics.json"
        atomic_write_json(metrics_path, {"parameter_copy": True, "selected_layers": plan.selected_layers})
        candidate = CandidateArtifact(
            event_key=event.event_key,
            model_path=str(candidate_model),
            registry_path=str(registry_path),
            metrics_path=str(metrics_path),
            source_policy_version=current.version,
            selected_layers=tuple(plan.selected_layers),
            profile_ids=(profile_id,),
            numerical_stable=True,
            metadata={"runtime": "tabular_reference"},
        )
        return ConsolidationResult(candidate, candidate_registry)

    def evaluate(
        self,
        current: CurrentVersion,
        event: LearningEvent,
        result: ConsolidationResult,
        transaction: TransactionHandle,
    ) -> RuntimeEvaluation:
        del transaction
        baseline_policy = self._policy_for_path(Path(current.model_path), current.version, current.commit_hash)
        candidate_policy = self._policy_for_path(
            Path(result.candidate.model_path), current.version + 1, result.candidate.fingerprint
        )
        evaluation_records = (event.eval_examples or event.examples).records
        baseline = self._evaluate_policy(baseline_policy, event, evaluation_records)
        candidate = self._evaluate_policy(candidate_policy, event, evaluation_records)
        candidate["general_score"] = 1.0
        baseline["general_score"] = 1.0
        return RuntimeEvaluation(
            candidate_metrics=candidate,
            baseline_metrics=baseline,
            checks=SystemChecks(
                numerical_stable=result.candidate.numerical_stable,
                access_audit_clean=True,
                within_budget=True,
                details={"runtime": "tabular_reference"},
                retention_stable=True,
                general_stable=True,
                staleness_clean=candidate.get("stale_rate", 0.0) <= 0.10,
            ),
        )

    def _policy_for_path(self, path: Path, version: int, identity: str) -> TabularTemporaryPolicy:
        state = self._state(path)
        return TabularTemporaryPolicy(
            choices=state.get("choices", self.choices),
            committed_version=version,
            base_policy_hash=identity,
            seed=self.seed,
            learning_rate=self.learning_rate,
            candidate_layers=state.get("candidate_layers", (0, 1)),
            state=state.get("logits", {}),
        )

    def _evaluate_policy(
        self,
        policy: TabularTemporaryPolicy,
        event: LearningEvent,
        records: Sequence[ExampleRecord],
    ) -> Dict[str, float]:
        verifier = build_verifier(event.verifier) if event.verifier else None
        correct = 0
        valid = 0
        stale = 0
        for record in records:
            probabilities = policy._probabilities(record.prompt)
            completion = max(policy.choices, key=lambda choice: (probabilities[choice], choice))
            if verifier is not None:
                result = score_sync(
                    verifier,
                    record,
                    Trajectory(
                        event_key=event.event_key,
                        example_id=record.example_id,
                        group_id="eval",
                        rollout_id="eval",
                        policy_version=policy.version,
                        prompt=record.prompt,
                        completion=completion,
                    ).verifier_view(),
                )
                correct += int(result.success)
                valid += int(result.valid)
                stale += int(result.stale)
            elif record.target is not None:
                correct += int(completion.strip() == record.target.strip())
                valid += int(bool(completion.strip()))
        count = max(1, len(records))
        return {
            "capability": correct / count,
            "success_rate": correct / count,
            "valid_rate": valid / count,
            "stale_rate": stale / count,
        }

    def _state(self, path: Path) -> Dict[str, Any]:
        file_path = path / "tabular_policy.json" if path.is_dir() else path
        if not file_path.exists():
            return {"choices": self.choices, "logits": {}, "candidate_layers": (0, 1)}
        return json.loads(file_path.read_text(encoding="utf-8"))

    def _delta_pressure(
        self,
        before: Mapping[str, Mapping[str, float]],
        after: Mapping[str, Mapping[str, float]],
    ) -> float:
        keys = set(before) | set(after)
        total = 0.0
        for prompt in keys:
            choices = set(before.get(prompt, {})) | set(after.get(prompt, {}))
            total += sum(
                abs(float(after.get(prompt, {}).get(choice, 0.0)) - float(before.get(prompt, {}).get(choice, 0.0)))
                for choice in choices
            )
        return max(total, 1e-9)
