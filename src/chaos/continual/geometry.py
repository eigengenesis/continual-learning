from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Tuple

from ._io import sha256_json
from .events import LearningEvent
from .profiles import ProfileRegistry


class GeometryDecision(str, Enum):
    CONSOLIDATE = "consolidate"
    COMPRESS_AND_RETRY = "compress_and_retry"
    RELEASE_AND_CONSOLIDATE = "release_and_consolidate"
    REJECT = "reject"
    EXPAND = "expand"


@dataclass(frozen=True)
class LayerMeasurement:
    layer: int
    pressure: float
    residual_energy: float
    occupied_rank: int
    dimension: int
    profile_overlaps: Mapping[str, float] = field(default_factory=dict)
    directional_conflicts: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = [self.pressure, self.residual_energy]
        values.extend(self.profile_overlaps.values())
        values.extend(self.directional_conflicts.values())
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"layer {self.layer} geometry contains non-finite values")
        if self.layer < 0 or self.dimension <= 0 or not 0 <= self.occupied_rank <= self.dimension:
            raise ValueError(f"layer {self.layer} has invalid rank/dimension")
        if self.pressure < 0 or not 0 <= self.residual_energy <= 1.000001:
            raise ValueError(f"layer {self.layer} has invalid pressure/residual energy")

    @property
    def free_fraction(self) -> float:
        return max(0.0, 1.0 - float(self.occupied_rank) / float(self.dimension))


@dataclass(frozen=True)
class GeometryMeasurement:
    event_key: str
    layers: Tuple[LayerMeasurement, ...]
    source_policy_hash: str
    acquisition_hash: str

    def __post_init__(self) -> None:
        indices = [item.layer for item in self.layers]
        if not self.layers or len(indices) != len(set(indices)):
            raise ValueError("geometry measurement requires unique layer measurements")


@dataclass(frozen=True)
class CapacityReport:
    mean_free_fraction: float
    mean_residual_energy: float
    selected_pressure_coverage: float
    saturated_layers: Tuple[int, ...]
    expansion_recommended: bool


@dataclass(frozen=True)
class GeometryPlan:
    event_key: str
    decision: str
    selected_layers: Tuple[int, ...]
    protected_profile_ids: Tuple[str, ...]
    release_profile_ids: Tuple[str, ...]
    conflict_profile_ids: Tuple[str, ...]
    capacity: CapacityReport
    reasons: Tuple[str, ...]
    measurement_hash: str

    @property
    def fingerprint(self) -> str:
        return sha256_json(asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint
        return payload


class GeometryController:
    def __init__(
        self,
        *,
        pressure_coverage: float = 0.8,
        min_layers: int = 2,
        max_layers: int = 0,
        min_residual_energy: float = 0.15,
        saturation_free_fraction: float = 0.08,
        conflict_threshold: float = 0.65,
        enable_expansion: bool = False,
    ) -> None:
        if not 0 < pressure_coverage <= 1:
            raise ValueError("pressure_coverage must be in (0, 1]")
        self.pressure_coverage = float(pressure_coverage)
        self.min_layers = int(min_layers)
        self.max_layers = int(max_layers)
        self.min_residual_energy = float(min_residual_energy)
        self.saturation_free_fraction = float(saturation_free_fraction)
        self.conflict_threshold = float(conflict_threshold)
        self.enable_expansion = bool(enable_expansion)

    def plan(
        self,
        event: LearningEvent,
        measurement: GeometryMeasurement,
        registry: ProfileRegistry,
    ) -> GeometryPlan:
        if measurement.event_key != event.event_key:
            raise ValueError("geometry measurement belongs to a different event")
        release = registry.dependency_closure(event.supersedes) if event.supersedes else []
        protected = sorted(record.profile_id for record in registry.protected_excluding(release))
        ordered = sorted(measurement.layers, key=lambda item: (-item.pressure, item.layer))
        total_pressure = sum(item.pressure for item in ordered)
        selected: List[LayerMeasurement] = []
        covered = 0.0
        for item in ordered:
            selected.append(item)
            covered += item.pressure
            coverage = covered / total_pressure if total_pressure > 0 else 0.0
            enough = len(selected) >= max(1, self.min_layers) and coverage >= self.pressure_coverage
            if enough or (self.max_layers and len(selected) >= self.max_layers):
                break
        if self.max_layers:
            selected = selected[: self.max_layers]
        selected_coverage = covered / total_pressure if total_pressure > 0 else 0.0
        mean_free = sum(item.free_fraction for item in selected) / len(selected)
        mean_residual = sum(item.residual_energy for item in selected) / len(selected)
        saturated = tuple(item.layer for item in selected if item.free_fraction < self.saturation_free_fraction)
        conflicts = sorted(
            {
                profile_id
                for item in selected
                for profile_id, score in item.directional_conflicts.items()
                if score >= self.conflict_threshold and profile_id in protected
            }
        )
        expansion_recommended = bool(saturated) and mean_residual < self.min_residual_energy
        reasons: List[str] = [
            f"selected {len(selected)} layers covering {selected_coverage:.3f} of measured pressure",
            f"mean residual energy={mean_residual:.3f}, mean free fraction={mean_free:.3f}",
        ]
        if total_pressure <= 0 or not selected:
            decision = GeometryDecision.REJECT
            reasons.append("acquisition produced no measurable learning pressure")
        elif expansion_recommended and self.enable_expansion:
            decision = GeometryDecision.EXPAND
            reasons.append("selected geometry is saturated and expansion is enabled")
        elif mean_residual < self.min_residual_energy:
            decision = GeometryDecision.COMPRESS_AND_RETRY
            reasons.append("too little update energy remains after protecting active profiles")
        elif release:
            decision = GeometryDecision.RELEASE_AND_CONSOLIDATE
            reasons.append(f"explicit revision releases dependency closure={release}")
        else:
            decision = GeometryDecision.CONSOLIDATE
            if conflicts:
                reasons.append("profile conflicts are reported but not automatically released")
        capacity = CapacityReport(
            mean_free_fraction=mean_free,
            mean_residual_energy=mean_residual,
            selected_pressure_coverage=selected_coverage,
            saturated_layers=saturated,
            expansion_recommended=expansion_recommended,
        )
        return GeometryPlan(
            event_key=event.event_key,
            decision=decision.value,
            selected_layers=tuple(sorted(item.layer for item in selected)),
            protected_profile_ids=tuple(protected),
            release_profile_ids=tuple(release),
            conflict_profile_ids=tuple(conflicts),
            capacity=capacity,
            reasons=tuple(reasons),
            measurement_hash=sha256_json(asdict(measurement)),
        )
