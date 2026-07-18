import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from chaos.continual.access import EventDataAccessAudit
from chaos.continual.acquisition import AcquisitionConfig, DemonstrationAcquirer, OnPolicyAcquirer
from chaos.continual.algorithms import GroupRelativeAlgorithm
from chaos.continual.commit_store import TransactionStore
from chaos.continual.contexts import deterministic_context_modes
from chaos.continual.engine import ContinualLearningEngine
from chaos.continual.events import (
    AcquisitionBudget,
    DatasetRef,
    ExampleRecord,
    GateBundle,
    GateRule,
    LearningEvent,
    TargetRef,
    VerifierSpec,
)
from chaos.continual.geometry import (
    GeometryController,
    GeometryDecision,
    GeometryMeasurement,
    LayerMeasurement,
)
from chaos.continual.evaluator import CommitEvaluator, SystemChecks
from chaos.continual.profiles import ProfileRecord, ProfileRegistry
from chaos.continual.router import LearningSignalRouter
from chaos.continual.runtime import TabularContinualRuntime
from chaos.continual.runtime import TabularTemporaryPolicy
from chaos.continual.stream import DirectoryEventSource
from chaos.continual.trajectories import PolicyVersion, RolloutGroup, Trajectory, TrainingSample
from chaos.continual.verifiers import RewardResult, TrajectoryView, build_verifier, score_sync


def dataset(name, pairs):
    return DatasetRef(
        dataset_id=name,
        split="train",
        records=tuple(
            ExampleRecord(f"{name}:{index}", prompt, target)
            for index, (prompt, target) in enumerate(pairs)
        ),
    )


def capability_gates(threshold=0.75):
    return GateBundle(
        rules=(GateRule("capability", "capability", "capability", "ge", threshold),)
    )


class EventContractTests(unittest.TestCase):
    def test_event_round_trip_is_self_hashing_and_public_view_hides_target(self):
        event = LearningEvent(
            event_id="preference",
            revision=0,
            kind="reward",
            examples=dataset("preference", [("choose", "A")]),
            targets=TargetRef(visibility="verifier_only"),
            verifier=VerifierSpec("exact_match"),
        )
        restored = LearningEvent.from_dict(event.to_dict())
        self.assertEqual(restored, event)
        self.assertEqual(restored.fingerprint, event.fingerprint)
        self.assertFalse(hasattr(restored.public_examples()[0], "target"))

    def test_reward_diagnostics_reject_target_leakage(self):
        with self.assertRaisesRegex(ValueError, "prohibited"):
            RewardResult(1.0, True, True, diagnostics={"expected": "A"})

    def test_verifier_private_metadata_is_redacted_and_disguised_values_are_rejected(self):
        record = ExampleRecord(
            "revision:0",
            "prompt",
            "B",
            {"stale_output": "A", "public_route": "first-hop"},
        )
        event = LearningEvent(
            "revision_private",
            1,
            "reward",
            DatasetRef("revision_private", "train", (record,)),
            targets=TargetRef(visibility="verifier_only"),
            verifier=VerifierSpec("revision_exact"),
        )
        self.assertNotIn("stale_output", event.public_examples()[0].metadata)
        self.assertEqual(event.public_examples()[0].metadata["public_route"], "first-hop")
        with self.assertRaisesRegex(ValueError, "appears in public metadata"):
            LearningEvent(
                "leaky",
                0,
                "reward",
                DatasetRef(
                    "leaky",
                    "train",
                    (ExampleRecord("leaky:0", "prompt", "SECRET", {"innocent": "SECRET"}),),
                ),
                targets=TargetRef(visibility="verifier_only"),
                verifier=VerifierSpec("exact_match"),
            )

    def test_context_mixture_has_exact_counts(self):
        modes = deterministic_context_modes(
            100, {"full": 0.4, "compressed": 0.3, "none": 0.3}, "fixed"
        )
        self.assertEqual(modes.count("full"), 40)
        self.assertEqual(modes.count("compressed"), 30)
        self.assertEqual(modes.count("none"), 30)

    def test_revision_verifier_penalizes_the_explicit_stale_output(self):
        verifier = build_verifier(VerifierSpec("revision_exact"))
        example = ExampleRecord("revision:0", "prompt", "B", {"stale_output": "A"})
        stale = score_sync(verifier, example, TrajectoryView("prompt", "A"))
        current = score_sync(verifier, example, TrajectoryView("prompt", "B"))
        self.assertTrue(stale.stale)
        self.assertLess(stale.reward, 0.0)
        self.assertFalse(current.stale)
        self.assertTrue(current.success)

    def test_general_and_retention_checks_are_mandatory_commit_gates(self):
        event = LearningEvent(
            "gated",
            0,
            "demonstration",
            dataset("gated", [("p", "A")]),
            targets=TargetRef(),
        )
        report = CommitEvaluator().evaluate(
            event,
            candidate_metrics={},
            baseline_metrics={},
            checks=SystemChecks(
                numerical_stable=True,
                access_audit_clean=True,
                within_budget=True,
                details={},
                retention_stable=False,
                general_stable=False,
            ),
        )
        self.assertFalse(report.passed)
        failed = {gate.gate_id for gate in report.gates if not gate.passed}
        self.assertEqual(failed, {"system:retention", "system:general"})


class AlgorithmAndRouterTests(unittest.TestCase):
    def trajectory(self, reward, index):
        item = Trajectory(
            event_key="event@0",
            example_id="row",
            group_id="group",
            rollout_id=str(index),
            policy_version=PolicyVersion(0),
            prompt="prompt",
            completion=str(index),
            reward=RewardResult(reward, reward > 0, True),
            sample=TrainingSample([index + 1], [True], rollout_logprobs=[-1.0]),
        )
        return item

    def test_group_relative_credit_retains_successes_and_failures(self):
        group = RolloutGroup("event@0", "row", "group", PolicyVersion(0))
        for index, reward in enumerate((1.0, 0.0, -0.25, 0.0)):
            group.add(self.trajectory(reward, index))
        metrics = GroupRelativeAlgorithm().finalize_group(group)
        advantages = [item.sample.advantages[0] for item in group.trajectories]
        self.assertEqual(len(group.trajectories), 4)
        self.assertAlmostEqual(sum(advantages), 0.0)
        self.assertGreater(advantages[0], 0.0)
        self.assertLess(advantages[2], 0.0)
        self.assertFalse(metrics.zero_advantage)

    def test_router_uses_signal_availability_and_revision_intent(self):
        router = LearningSignalRouter()
        demo = LearningEvent(
            "demo", 0, "demonstration", dataset("demo", [("p", "A")]), targets=TargetRef()
        )
        reward = LearningEvent(
            "reward",
            0,
            "reward",
            dataset("reward", [("p", "A")]),
            targets=TargetRef(visibility="verifier_only"),
            verifier=VerifierSpec("exact_match"),
        )
        revision = LearningEvent(
            "revision",
            1,
            "revision",
            dataset("revision", [("p", "B")]),
            targets=TargetRef(visibility="verifier_only"),
            verifier=VerifierSpec("exact_match"),
            supersedes=("old",),
        )
        self.assertEqual(router.route(demo).acquisition, "demonstration")
        self.assertEqual(router.route(reward).acquisition, "reward")
        self.assertTrue(router.route(revision).requires_release)


class GeometryTests(unittest.TestCase):
    def test_hf_geometry_measures_real_overlap_union_rank_and_residual_energy(self):
        import torch

        from chaos.continual.hf_runtime import _measure_adapter_geometry

        parameter = "model.layers.0.mlp.up_proj.weight"
        delta = torch.eye(2)
        basis = torch.tensor([[1.0], [0.0]])
        measured = _measure_adapter_geometry(
            {parameter: delta},
            (0,),
            {"skill_a": {parameter: basis}, "duplicate": {parameter: basis}},
        )[0]
        self.assertAlmostEqual(measured.pressure, 2.0**0.5, places=6)
        self.assertAlmostEqual(measured.profile_overlaps["skill_a"], 0.5, places=6)
        self.assertAlmostEqual(measured.residual_energy, 0.5, places=6)
        self.assertEqual(measured.occupied_rank, 1)
        self.assertEqual(measured.dimension, 2)

    def test_explicit_release_follows_dependencies_but_overlap_alone_does_not(self):
        registry = ProfileRegistry()
        registry.register(ProfileRecord("old", "old", (), (0,), 0, "h", "e"))
        registry.register(ProfileRecord("dependent", "dependent", ("old",), (0,), 0, "h", "e"))
        registry.register(ProfileRecord("unrelated", "unrelated", (), (0,), 0, "h", "e"))
        event = LearningEvent(
            "revision",
            1,
            "revision",
            dataset("revision", [("p", "B")]),
            targets=TargetRef(visibility="verifier_only"),
            verifier=VerifierSpec("exact_match"),
            supersedes=("old",),
        )
        measurement = GeometryMeasurement(
            event.event_key,
            (
                LayerMeasurement(
                    0,
                    pressure=1.0,
                    residual_energy=0.8,
                    occupied_rank=3,
                    dimension=16,
                    profile_overlaps={"unrelated": 0.99},
                    directional_conflicts={"unrelated": 0.95},
                ),
            ),
            "base",
            "acquired",
        )
        plan = GeometryController(min_layers=1).plan(event, measurement, registry)
        self.assertEqual(plan.decision, GeometryDecision.RELEASE_AND_CONSOLIDATE.value)
        self.assertEqual(set(plan.release_profile_ids), {"old", "dependent"})
        self.assertIn("unrelated", plan.protected_profile_ids)
        self.assertIn("unrelated", plan.conflict_profile_ids)


class StoreAndStreamTests(unittest.TestCase):
    def test_recovery_removes_an_unpublished_version_without_advancing_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = TabularContinualRuntime(choices=("A", "B"))
            model = runtime.initialize_model(root / "model")
            store = TransactionStore(root / "store")
            store.initialize(model_path=model)
            orphan = store.versions / "v000001"
            orphan.mkdir()
            (orphan / "COMMITTED").write_text("{}\n", encoding="utf-8")
            recovered = TransactionStore(root / "store")
            self.assertEqual(recovered.current().version, 0)
            self.assertFalse(orphan.exists())

    def test_rejection_never_changes_current_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "weights.json").write_text("{}", encoding="utf-8")
            store = TransactionStore(root / "store")
            before = store.initialize(model_path=model)
            event = LearningEvent(
                "demo", 0, "demonstration", dataset("demo", [("p", "A")]), targets=TargetRef()
            )
            route = LearningSignalRouter().route(event)
            transaction = store.begin(event, route)
            store.reject(transaction, event=event, reason="gate failed")
            after = store.current()
            self.assertEqual(before.commit_hash, after.commit_hash)
            self.assertEqual(after.version, 0)

    def test_directory_stream_lease_and_ack_are_durable(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = DirectoryEventSource(Path(tmp))
            event = LearningEvent(
                "demo", 0, "demonstration", dataset("demo", [("p", "A")]), targets=TargetRef()
            )
            source.submit(event)
            lease = source.lease()
            self.assertEqual(lease.event, event)
            source.ack(lease, {"status": "committed"})
            self.assertEqual(source.checkpoint(), {"inbox": 0, "leased": 0, "committed": 1, "rejected": 0})

    def test_directory_stream_preserves_submission_order_not_event_name_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = DirectoryEventSource(Path(tmp))
            z_event = LearningEvent(
                "z_first", 0, "demonstration", dataset("z", [("p", "A")]), targets=TargetRef()
            )
            a_event = LearningEvent(
                "a_second", 0, "demonstration", dataset("a", [("p", "A")]), targets=TargetRef()
            )
            source.submit(z_event)
            source.submit(a_event)
            first = source.lease()
            self.assertEqual(first.event.event_key, z_event.event_key)

    def test_historical_training_rows_are_rejected_before_a_new_event_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = EventDataAccessAudit(Path(tmp) / "access.jsonl")
            first = LearningEvent(
                "first",
                0,
                "demonstration",
                dataset("shared", [("prompt", "A")]),
                targets=TargetRef(),
            )
            second = LearningEvent(
                "second",
                0,
                "demonstration",
                dataset("shared", [("prompt", "B")]),
                targets=TargetRef(),
            )
            audit.log(event=first, row_ids=("shared:0",), purpose="update")
            audit.assert_update_allowed(second, committed_event_keys=set())
            with self.assertRaisesRegex(RuntimeError, "historical training rows"):
                audit.assert_update_allowed(second)


class EndToEndEngineTests(unittest.TestCase):
    def test_completed_acquisition_restores_policy_into_a_fresh_runtime_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "completed"
            event = LearningEvent(
                "completed_demo",
                0,
                "demonstration",
                dataset("completed_demo", [("p0", "A"), ("p1", "B")]),
                targets=TargetRef(),
                budget=AcquisitionBudget(
                    max_optimizer_steps=4,
                    max_rollouts=16,
                    max_tokens=16,
                    max_wall_seconds=30,
                    group_size=1,
                    batch_size=1,
                ),
                seed=12,
            )
            trained = TabularTemporaryPolicy(
                choices=("A", "B"), committed_version=0, base_policy_hash="base"
            )
            acquirer = DemonstrationAcquirer(AcquisitionConfig(save_interval=25))
            acquirer.acquire(policy=trained, event=event, output_dir=output)
            restored = TabularTemporaryPolicy(
                choices=("A", "B"), committed_version=0, base_policy_hash="base"
            )
            acquirer.acquire(policy=restored, event=event, output_dir=output)
            self.assertEqual(restored.logits, trained.logits)
            self.assertEqual(restored.version, trained.version)

    def test_generation_errors_remain_in_the_reward_ledger(self):
        class OneFailurePolicy(TabularTemporaryPolicy):
            calls = 0

            def generate(self, prompt, sampling, seed):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("synthetic generation failure")
                return super().generate(prompt, sampling, seed=seed)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "errors"
            event = LearningEvent(
                "reward_errors",
                0,
                "reward",
                dataset("reward_errors", [("p0", "A")]),
                targets=TargetRef(visibility="verifier_only"),
                verifier=VerifierSpec("exact_match"),
                budget=AcquisitionBudget(
                    max_optimizer_steps=1,
                    max_rollouts=2,
                    max_tokens=8,
                    max_wall_seconds=30,
                    group_size=2,
                    batch_size=1,
                ),
            )
            policy = OneFailurePolicy(
                choices=("A", "B"), committed_version=0, base_policy_hash="base"
            )
            artifact = OnPolicyAcquirer(AcquisitionConfig(save_interval=25)).acquire(
                policy=policy,
                event=event,
                verifier=build_verifier(event.verifier),
                output_dir=output,
            )
            rows = [json.loads(line) for line in Path(artifact.sample_ledger_path).read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["error"], "generation:RuntimeError")
            self.assertIsNone(rows[0].get("sample"))
            self.assertEqual(artifact.metrics["generation_errors"], 1.0)

    def test_fresh_acquisition_rewinds_uncheckpointed_ledger_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fresh"
            output.mkdir(parents=True)
            (output / "demonstrations.jsonl").write_text(
                json.dumps({"metadata": {"acquisition_step": 99}}) + "\n",
                encoding="utf-8",
            )
            event = LearningEvent(
                "fresh_demo",
                0,
                "demonstration",
                dataset("fresh_demo", [("p0", "A")]),
                targets=TargetRef(),
                budget=AcquisitionBudget(
                    max_optimizer_steps=2,
                    max_rollouts=8,
                    max_tokens=8,
                    max_wall_seconds=30,
                    group_size=1,
                    batch_size=1,
                ),
            )
            policy = TabularTemporaryPolicy(
                choices=("A", "B"), committed_version=0, base_policy_hash="base"
            )
            DemonstrationAcquirer(AcquisitionConfig(save_interval=25)).acquire(
                policy=policy, event=event, output_dir=output
            )
            rows = [json.loads(line) for line in (output / "demonstrations.jsonl").read_text().splitlines()]
            self.assertEqual([row["metadata"]["acquisition_step"] for row in rows], [1, 2])

    def test_budget_limited_progress_records_last_completed_step(self):
        class SlowPolicy(TabularTemporaryPolicy):
            def update(self, trajectories, config):
                result = super().update(trajectories, config)
                time.sleep(0.02)
                return result

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "wall_limited"
            event = LearningEvent(
                "wall_limited",
                0,
                "demonstration",
                dataset("wall_limited", [("p0", "A")]),
                targets=TargetRef(),
                budget=AcquisitionBudget(
                    max_optimizer_steps=5,
                    max_rollouts=8,
                    max_tokens=8,
                    max_wall_seconds=0.005,
                    group_size=1,
                    batch_size=1,
                ),
            )
            policy = SlowPolicy(
                choices=("A", "B"), committed_version=0, base_policy_hash="base"
            )
            DemonstrationAcquirer(AcquisitionConfig(save_interval=25)).acquire(
                policy=policy, event=event, output_dir=output
            )
            progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["step"], 1)
            self.assertTrue(progress["completed"])

    def test_exact_generated_token_ids_are_not_retokenized_or_tail_truncated(self):
        from chaos.continual.hf_runtime import _completion_batch_from_ids

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 99

            def __call__(self, text, add_special_tokens=False):
                del text, add_special_tokens
                return {"input_ids": [10, 11]}

        batch = _completion_batch_from_ids(Tokenizer(), ["prompt"], [[7, 8, 9]], "cpu", 3)
        selected = batch["input_ids"][batch["completion_mask"]].tolist()
        self.assertEqual(selected, [7, 8])

    def test_profile_only_event_registers_existing_skill_without_copying_or_updating_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = TabularContinualRuntime(choices=("A", "B"), seed=2)
            model = runtime.initialize_model(root / "model")
            before_bytes = (model / "tabular_policy.json").read_bytes()
            store = TransactionStore(root / "store")
            store.initialize(model_path=model)
            engine = ContinualLearningEngine(
                store=store,
                runtime=runtime,
                geometry=GeometryController(min_layers=1),
            )
            event = LearningEvent(
                "existing_skill",
                0,
                "evaluation",
                dataset("existing_skill", [("known", "B")]),
                targets=TargetRef(visibility="verifier_only"),
                verifier=VerifierSpec("exact_match"),
                gates=GateBundle(
                    rules=(GateRule("finite", "capability", "capability", "finite"),)
                ),
            )
            result = engine.process_event(event)
            self.assertEqual(result["status"], "committed")
            self.assertTrue(result["profile_only"])
            self.assertEqual(store.current().model_path, str(model))
            self.assertEqual((model / "tabular_policy.json").read_bytes(), before_bytes)
            self.assertTrue((root / "store" / "versions" / "v000001" / "model_reference.json").exists())
            self.assertFalse((root / "store" / "versions" / "v000001" / "model").exists())
            self.assertIn("capability:existing_skill:r0", store.registry().records)
            access = json.loads((root / "store" / "data_access.jsonl").read_text().splitlines()[0])
            self.assertEqual(access["purpose"], "profile")

    def test_adapter_resume_matches_uninterrupted_demonstration_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event = LearningEvent(
                "resume_demo",
                0,
                "demonstration",
                dataset("resume_demo", [("p0", "A"), ("p1", "B")]),
                targets=TargetRef(),
                budget=AcquisitionBudget(
                    max_optimizer_steps=6,
                    max_rollouts=64,
                    max_tokens=64,
                    max_wall_seconds=30,
                    group_size=1,
                    batch_size=1,
                ),
                seed=91,
            )
            config = AcquisitionConfig(save_interval=1)

            full = TabularTemporaryPolicy(
                choices=("A", "B"),
                committed_version=0,
                base_policy_hash="base",
                seed=1,
                learning_rate=0.3,
            )
            DemonstrationAcquirer(config).acquire(
                policy=full, event=event, output_dir=root / "full", max_steps=6
            )

            interrupted = TabularTemporaryPolicy(
                choices=("A", "B"),
                committed_version=0,
                base_policy_hash="base",
                seed=1,
                learning_rate=0.3,
            )
            output = root / "resumed"
            DemonstrationAcquirer(config).acquire(
                policy=interrupted, event=event, output_dir=output, max_steps=3
            )
            (output / "acquisition.json").unlink()
            progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
            progress["completed"] = False
            (output / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
            resumed = TabularTemporaryPolicy(
                choices=("A", "B"),
                committed_version=0,
                base_policy_hash="base",
                seed=1,
                learning_rate=0.3,
            )
            DemonstrationAcquirer(config).acquire(
                policy=resumed, event=event, output_dir=output, max_steps=6
            )
            self.assertEqual(full.logits, resumed.logits)
            rows = [json.loads(line) for line in (output / "demonstrations.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual([row["metadata"]["acquisition_step"] for row in rows], list(range(1, 7)))

    def test_demo_then_verifier_only_revision_commits_one_evolving_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = TabularContinualRuntime(choices=("A", "B"), seed=11, learning_rate=0.7)
            initial_model = runtime.initialize_model(root / "initial_model")
            store = TransactionStore(root / "store")
            store.initialize(model_path=initial_model)
            engine = ContinualLearningEngine(
                store=store,
                runtime=runtime,
                geometry=GeometryController(min_layers=1),
                acquisition_config=AcquisitionConfig(),
            )
            source = DirectoryEventSource(root / "events")
            budget = AcquisitionBudget(
                max_optimizer_steps=24,
                max_rollouts=512,
                max_tokens=512,
                max_wall_seconds=30,
                group_size=4,
                batch_size=1,
            )
            first = LearningEvent(
                "skill_a",
                0,
                "demonstration",
                dataset("skill_a", [("symbol", "A")]),
                targets=TargetRef(),
                gates=capability_gates(0.75),
                budget=budget,
                seed=3,
            )
            second = LearningEvent(
                "skill_a_revision",
                1,
                "revision",
                dataset("skill_a_v2", [("symbol", "B")]),
                targets=TargetRef(visibility="verifier_only"),
                verifier=VerifierSpec("exact_match"),
                supersedes=("capability:skill_a:r0",),
                gates=capability_gates(0.75),
                budget=replace(budget, max_optimizer_steps=40),
                seed=7,
            )
            source.submit(first)
            source.submit(second)
            result = engine.run_stream(source)
            self.assertEqual(result, {"processed": 2, "committed": 2, "rejected": 0})
            self.assertEqual(store.current().version, 2)
            registry = store.registry()
            self.assertEqual(registry.records["capability:skill_a:r0"].status, "retired")
            self.assertEqual(registry.records["capability:skill_a_revision:r1"].status, "protected")
            access_rows = [
                json.loads(line)
                for line in (root / "store" / "data_access.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual({row["event_key"] for row in access_rows}, {first.event_key, second.event_key})
            second_transactions = []
            for state_path in (root / "store" / "transactions").glob("**/transaction.json"):
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertTrue((state_path.parent / "weights_pruned.json").exists())
                self.assertFalse((state_path.parent / "candidate" / "model").exists())
                if state["event_key"] == second.event_key:
                    second_transactions.append(state_path)
            self.assertEqual(len(second_transactions), 1)
            state_path = second_transactions[0]
            interrupted_state = json.loads(state_path.read_text(encoding="utf-8"))
            interrupted_state.update({"status": "running", "phase": "commit_preparing"})
            state_path.write_text(json.dumps(interrupted_state), encoding="utf-8")
            recovered = TransactionStore(root / "store")
            recovered_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(recovered.current().version, 2)
            self.assertEqual(recovered_state["status"], "committed")
            self.assertTrue(recovered_state["recovered"])


if __name__ == "__main__":
    unittest.main()
