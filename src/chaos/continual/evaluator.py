from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Tuple

from ._io import sha256_json
from .events import GateRule, LearningEvent


@dataclass(frozen=True)
class SystemChecks:
    numerical_stable: bool
    access_audit_clean: bool
    within_budget: bool
    details: Mapping[str, Any]
    retention_stable: bool = True
    general_stable: bool = True
    staleness_clean: bool = True


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    category: str
    passed: bool
    required: bool
    observed: Any
    threshold: Any
    reason: str


@dataclass(frozen=True)
class EvaluationReport:
    event_key: str
    passed: bool
    candidate_metrics: Mapping[str, Any]
    baseline_metrics: Mapping[str, Any]
    gates: Tuple[GateResult, ...]

    @property
    def fingerprint(self) -> str:
        return sha256_json(asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint
        return payload


class CommitEvaluator:
    def evaluate(
        self,
        event: LearningEvent,
        *,
        candidate_metrics: Mapping[str, Any],
        baseline_metrics: Mapping[str, Any],
        checks: SystemChecks,
    ) -> EvaluationReport:
        results = [self._metric_gate(rule, candidate_metrics, baseline_metrics) for rule in event.gates.rules]
        if event.gates.require_numerical_stability:
            results.append(
                GateResult(
                    "system:numerical",
                    "numerical",
                    checks.numerical_stable,
                    True,
                    checks.numerical_stable,
                    True,
                    "all candidate losses, gradients, parameters, profiles, and metrics must be finite",
                )
            )
        if event.gates.require_access_audit:
            results.append(
                GateResult(
                    "system:access",
                    "access",
                    checks.access_audit_clean,
                    True,
                    checks.access_audit_clean,
                    True,
                    "only current-event training rows may be used for updates",
                )
            )
        if event.gates.require_budget:
            results.append(
                GateResult(
                    "system:budget",
                    "budget",
                    checks.within_budget,
                    True,
                    checks.within_budget,
                    True,
                    "event acquisition and consolidation must remain inside the frozen budget",
                )
            )
        if event.gates.require_retention:
            results.append(
                GateResult(
                    "system:retention",
                    "retention",
                    checks.retention_stable,
                    True,
                    checks.retention_stable,
                    True,
                    "all active historical capability canaries must remain within their allowed drop",
                )
            )
        if event.gates.require_general_capability:
            results.append(
                GateResult(
                    "system:general",
                    "general",
                    checks.general_stable,
                    True,
                    checks.general_stable,
                    True,
                    "the frozen general-capability canary must remain within tolerance",
                )
            )
        if event.kind == "revision" and event.gates.require_staleness_on_revision:
            results.append(
                GateResult(
                    "system:staleness",
                    "staleness",
                    checks.staleness_clean,
                    True,
                    checks.staleness_clean,
                    True,
                    "a revision must suppress explicitly superseded behavior",
                )
            )
        passed = all(item.passed for item in results if item.required)
        return EvaluationReport(
            event_key=event.event_key,
            passed=passed,
            candidate_metrics=dict(candidate_metrics),
            baseline_metrics=dict(baseline_metrics),
            gates=tuple(results),
        )

    def _metric_gate(
        self,
        rule: GateRule,
        candidate: Mapping[str, Any],
        baseline: Mapping[str, Any],
    ) -> GateResult:
        observed = candidate.get(rule.metric)
        before = baseline.get(rule.metric)
        passed = False
        reason = ""
        if rule.operator == "finite":
            passed = finite(observed)
            reason = f"candidate metric {rule.metric} must be finite"
        elif rule.operator == "true":
            passed = bool(observed)
            reason = f"candidate check {rule.metric} must be true"
        elif not finite(observed):
            reason = f"candidate metric {rule.metric} is missing or non-finite"
        elif rule.operator == "ge":
            passed = float(observed) >= rule.threshold
            reason = f"{observed} >= {rule.threshold}"
        elif rule.operator == "le":
            passed = float(observed) <= rule.threshold
            reason = f"{observed} <= {rule.threshold}"
        elif not finite(before):
            reason = f"baseline metric {rule.metric} is missing or non-finite"
        elif rule.operator == "delta_ge":
            delta = float(observed) - float(before)
            passed = delta >= rule.threshold
            reason = f"delta={delta} >= {rule.threshold}"
        elif rule.operator == "drop_le":
            drop = float(before) - float(observed)
            passed = drop <= rule.threshold
            reason = f"drop={drop} <= {rule.threshold}"
        return GateResult(
            rule.gate_id,
            rule.category,
            passed,
            rule.required,
            observed,
            rule.threshold,
            reason,
        )


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
