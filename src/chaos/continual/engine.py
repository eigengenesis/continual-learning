from __future__ import annotations

import traceback
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Optional, Set

from .access import EventDataAccessAudit
from .acquisition import AcquisitionConfig, DemonstrationAcquirer, OnPolicyAcquirer
from .artifacts import AcquisitionArtifact, CandidateArtifact
from .commit_store import TransactionHandle, TransactionStore
from .consolidation import ConsolidationResult, ContinualRuntime
from .contexts import build_context_provider
from .events import LearningEvent
from .evaluator import CommitEvaluator, SystemChecks
from .geometry import GeometryController, GeometryDecision
from .profiles import ProfileRegistry
from .router import LearningSignalRouter, RoutingDecision
from .stream import DirectoryEventSource
from .verifiers import build_verifier


class ContinualLearningEngine:
    def __init__(
        self,
        *,
        store: TransactionStore,
        runtime: ContinualRuntime,
        router: Optional[LearningSignalRouter] = None,
        geometry: Optional[GeometryController] = None,
        evaluator: Optional[CommitEvaluator] = None,
        acquisition_config: Optional[AcquisitionConfig] = None,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.router = router or LearningSignalRouter()
        self.geometry = geometry or GeometryController()
        self.evaluator = evaluator or CommitEvaluator()
        self.acquisition_config = acquisition_config or AcquisitionConfig()
        self.access = EventDataAccessAudit(self.store.root / "data_access.jsonl")

    def process_event(self, event: LearningEvent) -> Dict[str, Any]:
        existing = self.store.event_status(event.event_key)
        if existing == "committed":
            current = self.store.current()
            return {"event_key": event.event_key, "status": "already_committed", "version": current.version}
        registry = self.store.registry()
        missing_dependencies = sorted(set(event.dependencies) - set(registry.records))
        if missing_dependencies:
            return {
                "event_key": event.event_key,
                "status": "blocked",
                "reason": f"missing capability dependencies={missing_dependencies}",
            }
        route = self.router.route(event)
        transaction = self.store.resumable(event) or self.store.begin(event, route)
        try:
            return self._process(event, route, transaction)
        except Exception as exc:
            state = transaction.state()
            if state.get("status") == "running":
                transaction.write_json(
                    "failure.json",
                    {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
                )
                self.store.reject(transaction, event=event, reason=f"{type(exc).__name__}: {exc}")
            raise

    def _process(
        self,
        event: LearningEvent,
        route: RoutingDecision,
        transaction: TransactionHandle,
    ) -> Dict[str, Any]:
        current = self.store.current()
        registry = self.store.registry()
        if route.acquisition == "evaluation":
            return self._profile_existing(event, route, transaction, current, registry)
        committed_event_keys = self._committed_event_keys()
        self.access.assert_update_allowed(event, committed_event_keys)
        transaction.set_phase("acquiring", policy_version=current.version)
        policy = self.runtime.open_temporary_policy(current, event, transaction)
        acquisition = self._acquire(policy, event, route, transaction, registry)
        self.access.log(
            event=event,
            row_ids=acquisition.example_ids,
            purpose="update",
            committed_event_keys=committed_event_keys,
        )
        violations = self.access.validate_event(event, committed_event_keys)
        transaction.write_json(
            "data_access.json",
            {"clean": not violations, "violations": violations, "fingerprint": self.access.fingerprint},
        )

        transaction.set_phase("profiling")
        measurement = self.runtime.measure_geometry(current, event, acquisition, registry, transaction)
        transaction.write_json("geometry_measurement.json", asdict(measurement))
        plan = self.geometry.plan(event, measurement, registry)
        transaction.write_json("geometry_plan.json", plan.to_dict())
        if plan.decision in {
            GeometryDecision.REJECT.value,
            GeometryDecision.COMPRESS_AND_RETRY.value,
            GeometryDecision.EXPAND.value,
        }:
            reason = f"geometry controller decision={plan.decision}: {'; '.join(plan.reasons)}"
            self.store.reject(transaction, event=event, reason=reason)
            return {"event_key": event.event_key, "status": "rejected", "reason": reason}

        candidate_artifact_path = transaction.root / "candidate" / "artifact.json"
        candidate_registry_path = transaction.root / "candidate" / "registry.json"
        if candidate_artifact_path.exists() and candidate_registry_path.exists():
            candidate = CandidateArtifact.from_dict(
                json.loads(candidate_artifact_path.read_text(encoding="utf-8"))
            )
            if not Path(candidate.model_path).exists():
                raise FileNotFoundError(f"resumable candidate model is missing: {candidate.model_path}")
            result = ConsolidationResult(candidate, ProfileRegistry.load(candidate_registry_path))
        else:
            transaction.set_phase("consolidating")
            result = self.runtime.consolidate(current, event, acquisition, plan, registry, transaction)
            transaction.write_json("candidate/artifact.json", result.candidate.to_dict())
            result.registry.save(candidate_registry_path)
        transaction.set_phase("evaluating")
        runtime_evaluation = self.runtime.evaluate(current, event, result, transaction)
        checks = SystemChecks(
            numerical_stable=runtime_evaluation.checks.numerical_stable,
            access_audit_clean=runtime_evaluation.checks.access_audit_clean and not violations,
            within_budget=runtime_evaluation.checks.within_budget and self._within_budget(event, acquisition),
            details={**dict(runtime_evaluation.checks.details), "access_violations": violations},
            retention_stable=runtime_evaluation.checks.retention_stable,
            general_stable=runtime_evaluation.checks.general_stable,
            staleness_clean=runtime_evaluation.checks.staleness_clean,
        )
        report = self.evaluator.evaluate(
            event,
            candidate_metrics=runtime_evaluation.candidate_metrics,
            baseline_metrics=runtime_evaluation.baseline_metrics,
            checks=checks,
        )
        transaction.write_json("evaluation.json", report.to_dict())
        if not report.passed:
            self.store.reject(transaction, event=event, reason="mandatory commit gate failed", report=report)
            return {
                "event_key": event.event_key,
                "status": "rejected",
                "gates": [asdict(value) for value in report.gates],
            }
        committed = self.store.commit(
            transaction,
            event=event,
            route=route,
            candidate=result.candidate,
            registry=result.registry,
            report=report,
        )
        return {
            "event_key": event.event_key,
            "status": "committed",
            "version": committed.version,
            "commit_hash": committed.commit_hash,
            "metrics": dict(report.candidate_metrics),
        }

    def _profile_existing(
        self,
        event: LearningEvent,
        route: RoutingDecision,
        transaction: TransactionHandle,
        current: Any,
        registry: ProfileRegistry,
    ) -> Dict[str, Any]:
        transaction.set_phase("profiling_existing", policy_version=current.version)
        self.access.log(
            event=event,
            row_ids=[record.example_id for record in event.examples.records],
            purpose="profile",
        )
        result = self.runtime.profile_existing(current, event, registry, transaction)
        transaction.write_json("candidate/artifact.json", result.candidate.to_dict())
        result.registry.save(transaction.root / "candidate" / "registry.json")
        runtime_evaluation = self.runtime.evaluate(current, event, result, transaction)
        report = self.evaluator.evaluate(
            event,
            candidate_metrics=runtime_evaluation.candidate_metrics,
            baseline_metrics=runtime_evaluation.baseline_metrics,
            checks=runtime_evaluation.checks,
        )
        transaction.write_json("evaluation.json", report.to_dict())
        if not report.passed:
            self.store.reject(transaction, event=event, reason="profile-only event gate failed", report=report)
            return {"event_key": event.event_key, "status": "rejected", "reason": "profile gate failed"}
        committed = self.store.commit(
            transaction,
            event=event,
            route=route,
            candidate=result.candidate,
            registry=result.registry,
            report=report,
        )
        return {
            "event_key": event.event_key,
            "status": "committed",
            "version": committed.version,
            "commit_hash": committed.commit_hash,
            "profile_only": True,
        }

    def _acquire(
        self,
        policy: Any,
        event: LearningEvent,
        route: RoutingDecision,
        transaction: TransactionHandle,
        registry: Any,
    ) -> AcquisitionArtifact:
        protected = registry.active()
        if route.acquisition == "demonstration":
            return DemonstrationAcquirer(self.acquisition_config).acquire(
                policy=policy,
                event=event,
                output_dir=transaction.acquisition_dir,
            )
        if route.acquisition == "reward":
            verifier = build_verifier(event.verifier)
            context = build_context_provider(event.privileged_context) if event.privileged_context else None
            return OnPolicyAcquirer(self.acquisition_config).acquire(
                policy=policy,
                event=event,
                verifier=verifier,
                output_dir=transaction.acquisition_dir,
                protected_profiles=protected,
                context_provider=context,
            )
        if route.acquisition == "hybrid":
            demo_steps = max(1, int(event.budget.max_optimizer_steps * 0.35))
            reward_steps = max(1, event.budget.max_optimizer_steps - demo_steps)
            demo = DemonstrationAcquirer(self.acquisition_config).acquire(
                policy=policy,
                event=event,
                output_dir=transaction.acquisition_dir / "demonstration",
                max_steps=demo_steps,
            )
            verifier = build_verifier(event.verifier)
            context = build_context_provider(event.privileged_context) if event.privileged_context else None
            reward = OnPolicyAcquirer(self.acquisition_config).acquire(
                policy=policy,
                event=event,
                verifier=verifier,
                output_dir=transaction.acquisition_dir / "reward",
                protected_profiles=protected,
                context_provider=context,
                step_offset=demo_steps,
                max_steps=reward_steps,
            )
            merged_metrics = {f"demo_{key}": value for key, value in demo.metrics.items()}
            merged_metrics.update({f"reward_{key}": value for key, value in reward.metrics.items()})
            transaction.write_json(
                "acquisition/ledger_index.json",
                {"demonstration": demo.sample_ledger_path, "reward": reward.sample_ledger_path},
            )
            combined = replace(
                reward,
                route="hybrid",
                example_ids=tuple(sorted(set(demo.example_ids) | set(reward.example_ids))),
                metrics=merged_metrics,
                contains_optimizer_targets=True,
                metadata={"demonstration_artifact": demo.fingerprint, "reward_artifact": reward.fingerprint},
            )
            combined.save(transaction.acquisition_dir / "acquisition.json")
            return combined
        raise ValueError(f"unsupported acquisition route={route.acquisition}")

    def _within_budget(self, event: LearningEvent, artifact: AcquisitionArtifact) -> bool:
        metrics = artifact.metrics
        steps = sum(float(value) for key, value in metrics.items() if key.endswith("optimizer_steps"))
        rollouts = sum(float(value) for key, value in metrics.items() if key.endswith("rollouts"))
        tokens = sum(float(value) for key, value in metrics.items() if key.endswith("completion_tokens"))
        wall = sum(float(value) for key, value in metrics.items() if key.endswith("wall_seconds"))
        return (
            steps <= event.budget.max_optimizer_steps + 1e-9
            and rollouts <= event.budget.max_rollouts + 1e-9
            and tokens <= event.budget.max_tokens + 1e-9
            and wall <= event.budget.max_wall_seconds + 1e-9
        )

    def _committed_event_keys(self) -> Set[str]:
        keys = set()
        for row in self.store.journal():
            if row.get("status") != "committed":
                continue
            value = row.get("event_key") or row.get("source_event")
            if value and value != "bootstrap":
                keys.add(str(value))
        return keys

    def run_stream(self, source: DirectoryEventSource, *, max_events: int = 0) -> Dict[str, int]:
        processed = 0
        committed = 0
        rejected = 0
        while not max_events or processed < max_events:
            lease = source.lease()
            if lease is None:
                break
            try:
                result = self.process_event(lease.event)
            except Exception as exc:
                source.nack(lease, f"{type(exc).__name__}: {exc}")
                rejected += 1
            else:
                if result.get("status") in {"committed", "already_committed"}:
                    source.ack(lease, result)
                    committed += 1
                elif result.get("status") == "blocked":
                    source.nack(lease, str(result.get("reason", "dependency blocked")), requeue=True)
                    break
                else:
                    source.nack(lease, str(result.get("reason", "event rejected")))
                    rejected += 1
            processed += 1
        return {"processed": processed, "committed": committed, "rejected": rejected}
