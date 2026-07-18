from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .artifacts import AcquisitionArtifact, CandidateArtifact
from .commit_store import CurrentVersion, TransactionHandle
from .events import LearningEvent
from .evaluator import SystemChecks
from .geometry import GeometryMeasurement, GeometryPlan
from .profiles import ProfileRegistry


@dataclass(frozen=True)
class ConsolidationResult:
    candidate: CandidateArtifact
    registry: ProfileRegistry


@dataclass(frozen=True)
class RuntimeEvaluation:
    candidate_metrics: Mapping[str, Any]
    baseline_metrics: Mapping[str, Any]
    checks: SystemChecks


class ContinualRuntime(Protocol):
    def profile_existing(
        self,
        current: CurrentVersion,
        event: LearningEvent,
        registry: ProfileRegistry,
        transaction: TransactionHandle,
    ) -> ConsolidationResult:
        ...

    def open_temporary_policy(
        self,
        current: CurrentVersion,
        event: LearningEvent,
        transaction: TransactionHandle,
    ) -> Any:
        ...

    def measure_geometry(
        self,
        current: CurrentVersion,
        event: LearningEvent,
        acquisition: AcquisitionArtifact,
        registry: ProfileRegistry,
        transaction: TransactionHandle,
    ) -> GeometryMeasurement:
        ...

    def consolidate(
        self,
        current: CurrentVersion,
        event: LearningEvent,
        acquisition: AcquisitionArtifact,
        plan: GeometryPlan,
        registry: ProfileRegistry,
        transaction: TransactionHandle,
    ) -> ConsolidationResult:
        ...

    def evaluate(
        self,
        current: CurrentVersion,
        event: LearningEvent,
        result: ConsolidationResult,
        transaction: TransactionHandle,
    ) -> RuntimeEvaluation:
        ...
