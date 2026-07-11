from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .trajectories import RolloutGroup


@dataclass(frozen=True)
class AlgorithmMetrics:
    mean_reward: float
    success_rate: float
    valid_rate: float
    mean_advantage: float
    zero_advantage: bool
    rollout_count: int


class AcquisitionAlgorithm(Protocol):
    name: str

    def finalize_group(self, group: RolloutGroup) -> AlgorithmMetrics:
        ...


class GroupRelativeAlgorithm:
    name = "group_relative"

    def __init__(self, *, normalize_std: bool = False, minimum_std: float = 1e-6) -> None:
        self.normalize_std = bool(normalize_std)
        self.minimum_std = float(minimum_std)

    def finalize_group(self, group: RolloutGroup) -> AlgorithmMetrics:
        if not group.trajectories:
            raise ValueError("cannot finalize an empty rollout group")
        rewards = [float(item.reward.reward) if item.reward else 0.0 for item in group.trajectories]
        mean = sum(rewards) / len(rewards)
        centered = [value - mean for value in rewards]
        if self.normalize_std:
            variance = sum(value * value for value in centered) / len(centered)
            scale = max(math.sqrt(variance), self.minimum_std)
            centered = [value / scale for value in centered]
        for trajectory, advantage in zip(group.trajectories, centered):
            if trajectory.sample is not None:
                trajectory.sample.broadcast_to_completion("advantages", advantage)
        trainable = [value for value in centered if value != 0.0]
        return AlgorithmMetrics(
            mean_reward=mean,
            success_rate=sum(bool(item.reward and item.reward.success) for item in group.trajectories)
            / len(group.trajectories),
            valid_rate=sum(bool(item.reward and item.reward.valid) for item in group.trajectories)
            / len(group.trajectories),
            mean_advantage=sum(trainable) / len(trainable) if trainable else 0.0,
            zero_advantage=not trainable,
            rollout_count=len(group.trajectories),
        )


class DemonstrationAlgorithm:
    name = "demonstration_ce"

    def finalize_group(self, group: RolloutGroup) -> AlgorithmMetrics:
        if not group.trajectories:
            raise ValueError("cannot finalize an empty demonstration group")
        for trajectory in group.trajectories:
            if trajectory.sample is not None:
                trajectory.sample.broadcast_to_completion("ce_weights", 1.0)
        return AlgorithmMetrics(0.0, 0.0, 1.0, 0.0, False, len(group.trajectories))


class ReferenceDistillationAlgorithm:
    name = "reference_distillation"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = float(weight)

    def finalize_group(self, group: RolloutGroup) -> AlgorithmMetrics:
        if not group.trajectories:
            raise ValueError("cannot finalize an empty distillation group")
        for trajectory in group.trajectories:
            if trajectory.sample is not None:
                trajectory.sample.broadcast_to_completion("reference_kl_weights", self.weight)
        rewards = [float(item.reward.reward) for item in group.trajectories if item.reward]
        return AlgorithmMetrics(
            sum(rewards) / len(rewards) if rewards else 0.0,
            sum(bool(item.reward and item.reward.success) for item in group.trajectories) / len(group.trajectories),
            sum(bool(item.reward and item.reward.valid) for item in group.trajectories) / len(group.trajectories),
            0.0,
            False,
            len(group.trajectories),
        )


class HybridAlgorithm:
    name = "hybrid_group_relative_reference"

    def __init__(self, *, reference_weight: float = 1.0, normalize_std: bool = False) -> None:
        self.group_relative = GroupRelativeAlgorithm(normalize_std=normalize_std)
        self.reference = ReferenceDistillationAlgorithm(reference_weight)

    def finalize_group(self, group: RolloutGroup) -> AlgorithmMetrics:
        metrics = self.group_relative.finalize_group(group)
        self.reference.finalize_group(group)
        return metrics
