from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ._io import append_jsonl
from .verifiers import RewardResult, TrajectoryView, prohibited_keys


@dataclass(frozen=True, order=True)
class PolicyVersion:
    committed: int
    attempt: int = 0
    update: int = 0

    def __post_init__(self) -> None:
        if min(self.committed, self.attempt, self.update) < 0:
            raise ValueError("policy version components must be non-negative")

    def __str__(self) -> str:
        return f"v{self.committed:06d}.a{self.attempt:03d}.u{self.update:06d}"


@dataclass(frozen=True)
class SamplingConfig:
    group_size: int = 4
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        if self.group_size <= 0 or self.max_new_tokens <= 0:
            raise ValueError("group_size and max_new_tokens must be positive")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("sampling temperature/top_p are invalid")


@dataclass
class TrainingSample:
    token_ids: List[int]
    completion_mask: List[bool]
    rollout_logprobs: List[float] = field(default_factory=list)
    reference_logprobs: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    ce_weights: List[float] = field(default_factory=list)
    reference_kl_weights: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        length = len(self.token_ids)
        if len(self.completion_mask) != length:
            raise ValueError("completion mask must align with token IDs")
        for name in (
            "rollout_logprobs",
            "reference_logprobs",
            "advantages",
            "ce_weights",
            "reference_kl_weights",
        ):
            values = getattr(self, name)
            if values and len(values) != length:
                raise ValueError(f"{name} must be empty or align with token IDs")
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} contains non-finite values")

    def broadcast_to_completion(self, field_name: str, value: float) -> None:
        setattr(
            self,
            field_name,
            [float(value) if trainable else 0.0 for trainable in self.completion_mask],
        )


@dataclass
class Trajectory:
    event_key: str
    example_id: str
    group_id: str
    rollout_id: str
    policy_version: PolicyVersion
    prompt: str
    completion: str
    reward: Optional[RewardResult] = None
    sample: Optional[TrainingSample] = None
    context_mode: str = "none"
    context_fingerprint: str = ""
    entropy: float = 0.0
    sampling: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if prohibited_keys(self.metadata):
            raise ValueError("trajectory metadata contains prohibited target-like fields")
        if not math.isfinite(float(self.entropy)):
            raise ValueError("trajectory entropy must be finite")

    @classmethod
    def new(
        cls,
        *,
        event_key: str,
        example_id: str,
        group_id: str,
        policy_version: PolicyVersion,
        prompt: str,
        completion: str,
        rollout_id: str = "",
        **kwargs: Any,
    ) -> "Trajectory":
        return cls(
            event_key=event_key,
            example_id=example_id,
            group_id=group_id,
            rollout_id=rollout_id or str(uuid.uuid4()),
            policy_version=policy_version,
            prompt=prompt,
            completion=completion,
            **kwargs,
        )

    @property
    def is_trainable(self) -> bool:
        if self.sample is None:
            return False
        streams = (self.sample.advantages, self.sample.ce_weights, self.sample.reference_kl_weights)
        return any(any(value != 0.0 for value in stream) for stream in streams if stream)

    def verifier_view(self) -> TrajectoryView:
        return TrajectoryView(self.prompt, self.completion, dict(self.metadata))

    def to_record(self, *, include_tokens: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "event_key": self.event_key,
            "example_id": self.example_id,
            "group_id": self.group_id,
            "rollout_id": self.rollout_id,
            "policy_version": asdict(self.policy_version),
            "prompt": self.prompt,
            "completion": self.completion,
            "reward": self.reward.public_dict() if self.reward else None,
            "context_mode": self.context_mode,
            "context_fingerprint": self.context_fingerprint,
            "entropy": self.entropy,
            "sampling": dict(self.sampling),
            "metadata": dict(self.metadata),
            "error": self.error,
            "created_at": self.created_at,
        }
        if include_tokens and self.sample is not None:
            payload["sample"] = asdict(self.sample)
        leaked = prohibited_keys(payload.get("metadata", {}))
        if leaked:
            raise ValueError(f"trajectory record leaks target-like metadata keys={sorted(leaked)}")
        return payload


@dataclass
class RolloutGroup:
    event_key: str
    example_id: str
    group_id: str
    policy_version: PolicyVersion
    trajectories: List[Trajectory] = field(default_factory=list)

    def add(self, trajectory: Trajectory) -> None:
        if trajectory.group_id != self.group_id:
            raise ValueError("trajectory belongs to a different rollout group")
        if trajectory.policy_version != self.policy_version:
            raise ValueError("rollout group cannot mix policy versions")
        if trajectory.example_id != self.example_id:
            raise ValueError("rollout group cannot mix examples")
        self.trajectories.append(trajectory)

    @property
    def successful(self) -> int:
        return sum(1 for item in self.trajectories if item.reward and item.reward.success)


class TrajectoryLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, trajectory: Trajectory) -> None:
        append_jsonl(self.path, trajectory.to_record())

    def append_group(self, group: RolloutGroup) -> None:
        for trajectory in group.trajectories:
            self.append(trajectory)

    def records(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
