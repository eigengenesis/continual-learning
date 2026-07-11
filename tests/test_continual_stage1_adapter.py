import json
import tempfile
import unittest
from pathlib import Path

import qwen35_lifelong_pipeline as legacy

from chaos.continual.events import LearningEvent
from chaos.continual.stage1 import (
    TwoHopSelfContextProvider,
    history_profile_events_from_manifest,
    stage1_events_from_manifest,
    write_general_canary_event,
    write_stage1_events,
)


class Stage1EventAdapterTests(unittest.TestCase):
    def manifest(self):
        args = legacy.build_arg_parser().parse_args([])
        args.model_id = "test/model"
        args.glyph_count = 16
        args.train_samples = 16
        args.eval_samples = 8
        args.composition_train_glyphs = 8
        args.composition_train_samples = 16
        args.composition_eval_samples = 8
        return legacy.build_stage1_manifest(args)

    def test_stage1_becomes_generic_demo_reward_revision_stream(self):
        events = stage1_events_from_manifest(
            self.manifest(), acquisition_steps=5, composition_steps=7, group_size=2
        )
        self.assertEqual(
            [(event.event_id, event.kind) for event in events],
            [
                ("skill_a", "demonstration"),
                ("skill_b_v1", "demonstration"),
                ("composition_v1", "reward"),
                ("skill_b_v2", "revision"),
                ("composition_v2", "reward"),
            ],
        )
        composition = events[2]
        self.assertEqual(composition.targets.visibility, "verifier_only")
        self.assertEqual(composition.privileged_context.name, "two_hop_self")
        public = composition.public_examples()[0]
        self.assertFalse(hasattr(public, "target"))
        self.assertNotIn("color", public.metadata)
        self.assertNotIn("action", public.metadata)
        self.assertEqual(
            set(composition.dependencies),
            {"capability:skill_a:r0", "capability:skill_b_v1:r0"},
        )
        self.assertEqual(events[3].supersedes, ("capability:skill_b_v1:r0",))

    def test_written_events_are_frozen_and_self_hashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(self.manifest()), encoding="utf-8")
            paths = write_stage1_events(manifest_path, root / "events", acquisition_steps=3)
            self.assertEqual(len(paths), 5)
            for path in paths:
                restored = LearningEvent.from_dict(json.loads(path.read_text(encoding="utf-8")))
                self.assertTrue(restored.fingerprint)

    def test_two_hop_context_uses_policy_predictions_not_private_targets(self):
        class Output:
            def __init__(self, completion):
                self.completion = completion

        class Policy:
            def __init__(self):
                self.calls = []

            def generate(self, prompt, sampling, seed):
                self.calls.append((prompt, seed))
                return Output("C3" if len(self.calls) == 1 else "A5")

        event = stage1_events_from_manifest(self.manifest())[2]
        provider = TwoHopSelfContextProvider()
        policy = Policy()
        result = provider.build(event.public_examples()[0], "full", policy)
        self.assertEqual(len(policy.calls), 2)
        self.assertIn("INTERMEDIATE=C3; FINAL=A5", result.prompt)
        self.assertNotIn(str(event.examples.records[0].target), result.metadata)

    def test_five_skill_history_becomes_profile_only_events(self):
        history = {
            "seed": 9,
            "skills": [
                {
                    "recipe": {"name": "tooluse"},
                    "source": "fixture",
                    "train": [{"source": "call tool", "target": "ok"}],
                    "eval": [{"source": "use tool", "target": "ok"}],
                },
                {
                    "recipe": {"name": "science"},
                    "source": "fixture",
                    "train": [{"source": "science q", "target": "a"}],
                    "eval": [{"source": "science eval", "target": "a"}],
                },
            ],
        }
        events = history_profile_events_from_manifest(history)
        self.assertEqual([event.kind for event in events], ["evaluation", "evaluation"])
        self.assertEqual([event.event_id for event in events], ["history_tooluse", "history_science"])
        self.assertTrue(all(event.metadata["profile_only"] for event in events))

    def test_general_canary_becomes_a_frozen_profile_only_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canary = root / "canary.json"
            canary.write_text(
                json.dumps([{"prompt": "2+2=", "target": "4", "family": "reasoning"}]),
                encoding="utf-8",
            )
            output = write_general_canary_event(canary, root / "base_general.json", seed=5)
            event = LearningEvent.from_dict(json.loads(output.read_text(encoding="utf-8")))
            self.assertEqual(event.kind, "evaluation")
            self.assertTrue(event.metadata["profile_only"])
            self.assertEqual(event.examples.source_checksum, event.eval_examples.source_checksum)
            self.assertNotIn("target", event.public_examples()[0].metadata)


if __name__ == "__main__":
    unittest.main()
