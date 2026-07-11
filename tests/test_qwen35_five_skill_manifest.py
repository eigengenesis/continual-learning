from pathlib import Path

import qwen35_five_skill_cl_audit as audit


def _args(tmp_path: Path):
    return audit.build_arg_parser().parse_args(
        [
            "--mode",
            "build_manifest",
            "--smoke",
            "--train-samples",
            "3",
            "--eval-samples",
            "2",
            "--composition-eval-samples",
            "4",
            "--benchmarks",
            "none",
            "--manifest-path",
            str(tmp_path / "manifest.json"),
        ]
    )


def test_smoke_manifest_is_frozen_and_loadable(tmp_path):
    args = _args(tmp_path)
    manifest = audit.build_manifest(args)

    assert manifest["version"] == audit.MANIFEST_VERSION
    assert manifest["skill_order"] == list(audit.DEFAULT_SKILLS)
    assert len(manifest["skills"]) == 5
    assert all(len(skill["train"]) == 3 for skill in manifest["skills"])
    assert all(len(skill["eval"]) == 2 for skill in manifest["skills"])
    assert len(manifest["composition"]["eval"]) == 4
    assert "parts" in manifest["composition"]["eval"][0]
    assert len(manifest["composition"]["eval"][0]["parts"]) == 2

    path = audit.write_manifest(manifest, tmp_path / "manifest.json")
    loaded = audit.load_manifest(path)
    skills, composition, benchmarks = audit.tasks_from_manifest(loaded)

    assert [task.spec.name for task in skills] == list(audit.DEFAULT_SKILLS)
    assert composition is not None
    assert composition.spec.name == "composition"
    assert benchmarks == []


def test_branch_aliases_are_stable():
    assert audit.parse_branches("naive,sdft,ours") == [
        "naive_sft",
        "sdft_baseline",
        "amoeba",
    ]


def test_openmathinstruct_schema_is_accepted():
    row = {
        "problem": "What is 2 + 3?",
        "generated_solution": "2 + 3 = 5. The answer is 5.",
        "expected_answer": "5",
        "problem_source": "unit",
    }

    example = audit.make_example(audit.SKILL_RECIPES["math"], row, 0)

    assert example is not None
    assert "What is 2 + 3?" in example["prompt"]
    assert example["target"] == "2 + 3 = 5. The answer is 5."


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": str(text).split()}


def test_composition_section_parser_and_part_scorers():
    completion = """MATH ANSWER:
The final answer is 42.

SQL ANSWER:
SELECT * FROM t WHERE value > 7;
"""
    parts = [
        {"slot": "A", "skill": "math", "target": "We compute it. The answer is 42."},
        {"slot": "B", "skill": "sql", "target": "SELECT * FROM t WHERE value > 7;"},
    ]

    sections = audit.extract_generated_composition_sections(completion, parts)
    assert "42" in sections["math"]
    assert "SELECT" in sections["sql"]

    math_ok, _, math_method = audit.score_composition_part(DummyTokenizer(), "math", sections["math"], parts[0]["target"])
    sql_ok, _, sql_method = audit.score_composition_part(DummyTokenizer(), "sql", sections["sql"], parts[1]["target"])

    assert math_ok
    assert sql_ok
    assert math_method == "final_number"
    assert sql_method == "normalized_sql"


def test_old_manifest_composition_target_remains_parseable():
    parts = audit.extract_composition_parts_from_target(
        "MATH ANSWER:\n5\n\nMEDICAL ANSWER:\nA",
        "math+medical",
    )

    assert parts == [
        {"slot": "A", "skill": "math", "target": "5"},
        {"slot": "B", "skill": "medical", "target": "A"},
    ]
