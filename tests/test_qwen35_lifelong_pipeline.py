import json
import tempfile
import unittest
from pathlib import Path

import torch

import qwen35_lifelong_pipeline as lp


class LifelongManifestTests(unittest.TestCase):
    def make_args(self):
        args = lp.build_arg_parser().parse_args([])
        args.model_id = "test/model"
        args.seed = 1337
        args.glyph_count = 32
        args.train_samples = 64
        args.eval_samples = 16
        args.composition_train_glyphs = 16
        args.composition_train_samples = 32
        args.composition_eval_samples = 16
        return args

    def test_manifest_is_deterministic_and_self_hashing(self):
        args = self.make_args()
        first = lp.build_stage1_manifest(args)
        second = lp.build_stage1_manifest(args)
        self.assertEqual(first, second)
        self.assertEqual(first["manifest_sha256"], lp.manifest_digest(first))

    def test_composition_splits_and_policy_change(self):
        manifest = lp.build_stage1_manifest(self.make_args())
        train = set(manifest["splits"]["composition_train_glyphs"])
        evaluation = set(manifest["splits"]["composition_eval_glyphs"])
        self.assertFalse(train & evaluation)
        changed = manifest["mappings"]["changed_colors"]
        self.assertEqual(len(changed), 2)
        for color in changed:
            self.assertNotEqual(
                manifest["mappings"]["color_to_action_v1"][color],
                manifest["mappings"]["color_to_action_v2"][color],
            )

    def test_direct_training_targets_are_action_codes_only(self):
        manifest = lp.build_stage1_manifest(self.make_args())
        rows = manifest["tasks"]["composition_direct_v1"]["train"]
        self.assertTrue(rows)
        self.assertTrue(all(item["target"] in lp.ACTIONS for item in rows))
        train_glyphs = set(manifest["splits"]["composition_train_glyphs"])
        self.assertTrue(all(item["glyph"] in train_glyphs for item in rows))
        self.assertEqual(manifest["tasks"]["composition_direct_v1"]["recipe"]["max_new_tokens"], 10)

    def test_primitive_eval_uses_held_out_prompt_templates(self):
        manifest = lp.build_stage1_manifest(self.make_args())
        for task_key in ("skill_a", "skill_b_v1"):
            train_prefixes = {item["prompt"].splitlines()[0] for item in manifest["tasks"][task_key]["train"]}
            eval_prefixes = {item["prompt"].splitlines()[0] for item in manifest["tasks"][task_key]["eval"]}
            self.assertFalse(train_prefixes & eval_prefixes)

    def test_mode_schedule_has_exact_requested_counts(self):
        modes = lp.deterministic_modes(100, {"full": 0.4, "compressed": 0.3, "none": 0.3}, "test")
        self.assertEqual(modes.count("full"), 40)
        self.assertEqual(modes.count("compressed"), 30)
        self.assertEqual(modes.count("none"), 30)

    def test_runtime_hyperparameters_must_match_manifest(self):
        args = self.make_args()
        manifest = lp.build_stage1_manifest(args)
        lp.validate_frozen_hyperparameters(args, manifest)
        args.composition_lr *= 2
        with self.assertRaises(ValueError):
            lp.validate_frozen_hyperparameters(args, manifest)

    def test_manifest_requires_unfiltered_on_policy_trajectories(self):
        manifest = lp.build_stage1_manifest(self.make_args())
        verifier = manifest["verifiers"]["action"]
        self.assertEqual(verifier["trajectory_retention"], "all_on_policy")
        self.assertFalse(verifier["separate_gold_target_field_in_trajectory_artifact"])
        self.assertEqual(verifier["composition_teacher_context"], "model_generated_only")


class ProfileAndStateTests(unittest.TestCase):
    def entry(self, profile_id, dependencies=()):
        return lp.ProfileEntry(
            profile_id=profile_id,
            task_name=profile_id,
            scope={},
            dependencies=list(dependencies),
            selected_layers=[0, 1],
            checkpoint_sha256="abc",
            creation_stage="test",
            status="protected",
            tensor_path=f"{profile_id}.safetensors",
            layer_metadata={},
        )

    def test_release_closure_follows_composition_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = lp.ProfileRegistry(Path(tmp))
            registry.add_metadata(self.entry("skill_b:v1:C1"))
            registry.add_metadata(self.entry("composition:v1:C1", ["skill_b:v1:C1"]))
            registry.add_metadata(self.entry("composition:summary", ["composition:v1:C1"]))
            registry.add_metadata(self.entry("skill_b:v1:C2"))
            closure = registry.dependency_closure(["skill_b:v1:C1"])
            self.assertEqual(
                closure,
                ["composition:summary", "composition:v1:C1", "skill_b:v1:C1"],
            )
            registry.release_closure(["skill_b:v1:C1"])
            self.assertEqual(registry.entries["skill_b:v1:C2"].status, "protected")
            self.assertEqual(registry.entries["composition:v1:C1"].status, "released")

    def test_profile_safetensor_round_trip_and_registry_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = lp.ProfileRegistry(Path(tmp))
            layer = lp.qt.LayerProfile(
                layer_index=3,
                layer_name="model.layers.3.mlp.down_proj",
                activation_basis=torch.eye(3, dtype=torch.float32)[:, :2],
                gradient_basis=torch.tensor([[1.0], [2.0], [3.0]]),
                effective_act_rank=2,
                effective_grad_rank=1,
                explained_variance=0.875,
            )
            profile = lp.qt.TaskProfile(task_name="skill_a", stage_label="test", layer_profiles={3: layer})
            registry.add_profile(
                "skill_a",
                profile,
                scope={"skill": "glyph_color"},
                dependencies=[],
                selected_layers=[3],
                checkpoint_sha256="abc",
                creation_stage="11_consolidate_a",
            )
            restored = registry.load_profile("skill_a")
            self.assertTrue(torch.equal(restored.layer_profiles[3].activation_basis, layer.activation_basis))
            self.assertTrue(torch.equal(restored.layer_profiles[3].gradient_basis, layer.gradient_basis))

            temporary_path = Path(tmp) / "temporary.safetensors"
            with self.assertRaisesRegex(RuntimeError, "interrupt"):
                with registry.transaction():
                    registry.add_profile(
                        "temporary",
                        profile,
                        scope={"skill": "temporary"},
                        dependencies=["skill_a"],
                        selected_layers=[3],
                        checkpoint_sha256="def",
                        creation_stage="test",
                    )
                    self.assertTrue(temporary_path.exists())
                    raise RuntimeError("interrupt")
            self.assertNotIn("temporary", registry.entries)
            self.assertFalse(temporary_path.exists())

    def test_data_access_audit_rejects_old_training_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = lp.DataAccessAudit(Path(tmp) / "access.jsonl")
            audit.log(
                stage="10_acquire_a",
                task="skill_a",
                split="train",
                row_ids=["a:1"],
                purpose="update",
                allowed_train_tasks=["skill_a"],
            )
            with self.assertRaises(RuntimeError):
                audit.log(
                    stage="20_acquire_b",
                    task="skill_a",
                    split="train",
                    row_ids=["a:1"],
                    purpose="update",
                    allowed_train_tasks=["skill_b_v1"],
                )

    def test_data_access_audit_rejects_unknown_current_task_row(self):
        args = lp.build_arg_parser().parse_args([])
        args.model_id = "test/model"
        manifest = lp.build_stage1_manifest(args)
        with tempfile.TemporaryDirectory() as tmp:
            audit = lp.DataAccessAudit(Path(tmp) / "access.jsonl", manifest)
            with self.assertRaises(RuntimeError):
                audit.log(
                    stage="10_acquire_a",
                    task="skill_a",
                    split="train",
                    row_ids=["a:train:not-in-manifest"],
                    purpose="update",
                    allowed_train_tasks=["skill_a"],
                )

    def test_data_access_rewind_removes_discarded_update_tail(self):
        args = lp.build_arg_parser().parse_args([])
        args.model_id = "test/model"
        manifest = lp.build_stage1_manifest(args)
        valid_row = manifest["tasks"]["skill_a"]["train"][0]["row_id"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "access.jsonl"
            audit = lp.DataAccessAudit(path, manifest)
            for step in (1, 2, 3):
                audit.log(
                    stage="10_acquire_a",
                    task="skill_a",
                    split="train",
                    row_ids=[valid_row],
                    purpose="update",
                    allowed_train_tasks=["skill_a"],
                    step=step,
                )
            audit.rewind_stage("10_acquire_a", 2)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([item["step"] for item in rows], [1, 2])
            self.assertFalse(audit.validate_existing())

    def test_stage_store_rejects_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {"manifest_sha256": "one"}
            store = lp.StageStore(Path(tmp), manifest, "model", "auto")
            store.begin("00_bootstrap")
            store.commit("00_bootstrap", {"ok": True})
            self.assertIn("00_bootstrap", store.state.completed_stages)
            with self.assertRaises(ValueError):
                lp.StageStore(Path(tmp), {"manifest_sha256": "two"}, "model", "auto")

    def test_adapter_resume_persists_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = lp.StageStore(Path(tmp), {"manifest_sha256": "one"}, "model", "auto")
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            torch.manual_seed(1234)
            store.save_adapter_resume(
                "10_acquire_a",
                step=7,
                model=model,
                optimizer=optimizer,
                best_state=lp.trainable_state(model),
                best_score=(0.75, -0.5),
                extra={"block": 1},
            )
            expected_next_random = torch.rand(4)
            payload = store.load_adapter_resume("10_acquire_a")
            self.assertEqual(payload["step"], 7)
            self.assertEqual(payload["extra"], {"block": 1})
            self.assertEqual(set(payload["trainable_state"]), {"weight", "bias"})
            self.assertTrue(payload["optimizer"]["state"])
            lp.restore_rng_state(payload["rng"])
            self.assertTrue(torch.equal(torch.rand(4), expected_next_random))

            stale = store.stage_resume_dir("10_acquire_a") / "adapter_step_000008.pt"
            torch.save({"step": 8}, stale)
            self.assertEqual(store.load_adapter_resume("10_acquire_a")["step"], 7)

    def test_interrupted_adapter_training_matches_uninterrupted(self):
        torch.manual_seed(77)
        template = torch.nn.Linear(3, 2)
        initial = {key: value.detach().clone() for key, value in template.state_dict().items()}

        def create():
            model = torch.nn.Linear(3, 2)
            model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, foreach=False)
            return model, optimizer

        def advance(model, optimizer, start, stop):
            for _ in range(start, stop + 1):
                x = torch.rand(4, 3)
                y = torch.rand(4, 2)
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.mse_loss(model(x), y).backward()
                optimizer.step()

        full_model, full_optimizer = create()
        torch.manual_seed(991)
        advance(full_model, full_optimizer, 1, 6)

        with tempfile.TemporaryDirectory() as tmp:
            resumed_model, resumed_optimizer = create()
            torch.manual_seed(991)
            advance(resumed_model, resumed_optimizer, 1, 3)
            store = lp.StageStore(Path(tmp), {"manifest_sha256": "one"}, "model", "auto")
            store.save_adapter_resume(
                "10_acquire_a",
                step=3,
                model=resumed_model,
                optimizer=resumed_optimizer,
                best_state=lp.trainable_state(resumed_model),
                best_score=0.0,
                extra={},
            )
            payload = store.load_adapter_resume("10_acquire_a")

            continued_model, continued_optimizer = create()
            lp.restore_named_state(continued_model, payload["trainable_state"])
            continued_optimizer.load_state_dict(payload["optimizer"])
            lp.restore_rng_state(payload["rng"])
            advance(continued_model, continued_optimizer, 4, 6)
            for name, expected in full_model.state_dict().items():
                self.assertTrue(torch.equal(continued_model.state_dict()[name], expected), name)

    def test_full_resume_checksums_detect_corruption(self):
        class SaveableLinear(torch.nn.Linear):
            def save_pretrained(self, path, safe_serialization=True):
                del safe_serialization
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), path / "model.safetensors")

        class Tokenizer:
            def save_pretrained(self, path):
                Path(path, "tokenizer.json").write_text("{}", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            store = lp.StageStore(Path(tmp), {"manifest_sha256": "one"}, "model", "auto")
            model = SaveableLinear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
            destination = store.save_full_resume(
                "11_consolidate_a",
                2,
                model,
                Tokenizer(),
                optimizer,
                {"task_key": "skill_a"},
            )
            payload = store.load_full_resume("11_consolidate_a")
            self.assertEqual(payload["step"], 2)
            with (destination / "optimizer.pt").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaises(RuntimeError):
                store.load_full_resume("11_consolidate_a")

    def test_manifest_round_trip_validates_hash(self):
        args = lp.build_arg_parser().parse_args([])
        args.model_id = "test/model"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            lp.write_manifest(lp.build_stage1_manifest(args), path)
            loaded = lp.load_manifest(path)
            self.assertEqual(loaded["manifest_sha256"], lp.manifest_digest(loaded))
            payload = json.loads(path.read_text())
            payload["seed"] += 1
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                lp.load_manifest(path)

    def test_manifest_rejects_structural_tampering_even_when_rehashed(self):
        args = lp.build_arg_parser().parse_args([])
        args.model_id = "test/model"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            payload = lp.build_stage1_manifest(args)
            payload["stage_order"] = list(reversed(payload["stage_order"]))
            payload["manifest_sha256"] = lp.manifest_digest(payload)
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stage order"):
                lp.load_manifest(path)


class VerifierTests(unittest.TestCase):
    def test_action_reward(self):
        reward, info = lp.action_reward("A3", "A3")
        self.assertEqual(reward, 1.0)
        self.assertTrue(info["correct"])
        reward, info = lp.action_reward("A2", "A3")
        self.assertEqual(reward, 0.1)
        self.assertTrue(info["valid"])
        reward, info = lp.action_reward("nonsense", "A3")
        self.assertEqual(reward, -0.25)
        self.assertFalse(info["valid"])

    def test_completion_masks_are_right_aligned(self):
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, value, add_special_tokens=False):
                del add_special_tokens
                return {"input_ids": [2 + (ord(char) % 29) for char in str(value)]}

        batch = lp.al._prepare_completion_kl_batch(
            Tokenizer(),
            ["p", "a much longer prompt"],
            ["A3", "A3"],
            "cpu",
            12,
        )
        self.assertEqual(tuple(batch["completion_mask"].shape), (2, 12))
        self.assertTrue(torch.equal(batch["completion_mask"][0], batch["completion_mask"][1]))
        self.assertEqual(int(batch["completion_mask"].sum().item()), 4)

    def test_on_policy_trajectory_keeps_failures_without_gold_target(self):
        item = {
            "row_id": "composition:v1:train:0:direct",
            "target": "A3",
            "color": "C3",
            "glyph": "G0",
        }
        trajectory = lp.make_on_policy_trajectory(
            stage="40_acquire_composition",
            step=7,
            context_mode="none",
            rollout_index=2,
            task_key="composition_direct_v1",
            item=item,
            prompt="direct",
            teacher_prompt="direct",
            completion="invalid output",
            reward=-0.25,
            verifier={"prediction": "", "correct": False, "valid": False},
        )
        self.assertFalse(trajectory["correct"])
        self.assertEqual(trajectory["reward"], -0.25)
        self.assertNotIn("target", trajectory)
        self.assertNotIn("expected", trajectory)


if __name__ == "__main__":
    unittest.main()
