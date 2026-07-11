#!/usr/bin/env python3
"""Five-skill Qwen3.5 continual-learning audit.

This runner is intentionally built around a frozen manifest:

    session 1: build manifest once
    session 2: naive SFT baseline on that manifest
    session 3: SDFT baseline on that manifest
    session 4: Amoeba/Eigenesis no-proxy run on that manifest

Every branch consumes the same prompts, targets, train split, eval split, seed,
and sample counts. That makes the headline comparison hard to nitpick.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - Colab/Kaggle normally has datasets.
    load_dataset = None

import alien_ladder_cl_audit as al
import qwen_continual_proof as qp
from standalone_latent_lora_qwen import choose_dtype, load_causal_lm, load_tokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
MANIFEST_VERSION = 1
DEFAULT_SKILLS = ("math", "code", "sql", "medical", "cowrite")
SHEN3_SKILLS = ("tooluse", "science", "medical")
TASK_PROFILES: Dict[str, Tuple[str, ...]] = {
    "five_ood": DEFAULT_SKILLS,
    "shen3": SHEN3_SKILLS,
}
BRANCHES = ("naive_sft", "sdft_baseline", "op_sdft", "amoeba")


def line(char: str = "=", width: int = 96) -> None:
    print(char * width, flush=True)


def section(title: str) -> None:
    line("=")
    print(title, flush=True)
    line("=")


def subsection(title: str) -> None:
    line("-")
    print(title, flush=True)
    line("-")


def stable_seed(label: str, offset: int = 0) -> int:
    total = int(offset)
    for idx, char in enumerate(str(label)):
        total += (idx + 1) * ord(char)
    return total


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_text(value: Any, *, max_chars: int = 4000) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()[:max_chars].strip()


def first_nonempty(row: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, (list, tuple)) and value:
                value = "\n".join(str(item) for item in value)
            text = compact_text(value)
            if text:
                return text
    return ""


def normalize_branch(value: str) -> str:
    aliases = {
        "naive": "naive_sft",
        "sft": "naive_sft",
        "naive_sft": "naive_sft",
        "sdft": "sdft_baseline",
        "tf_sdft": "sdft_baseline",
        "teacher_forced_sdft": "sdft_baseline",
        "sdft_baseline": "sdft_baseline",
        "opsdft": "op_sdft",
        "op_sdft": "op_sdft",
        "op_sdf": "op_sdft",
        "on_policy_sdft": "op_sdft",
        "on_policy_demo": "op_sdft",
        "shen": "op_sdft",
        "shen_sdft": "op_sdft",
        "amoeba": "amoeba",
        "eigenesis": "amoeba",
        "ours": "amoeba",
        "no_proxy": "amoeba",
    }
    key = str(value).strip().lower().replace("-", "_")
    if key not in aliases:
        raise ValueError(f"unknown branch {value!r}; allowed: {', '.join(BRANCHES)}")
    return aliases[key]


def parse_branches(value: str) -> List[str]:
    raw = str(value or "all").strip().lower()
    if raw in {"all", "*"}:
        return list(BRANCHES)
    out: List[str] = []
    for part in raw.split(","):
        if not part.strip():
            continue
        branch = normalize_branch(part)
        if branch not in out:
            out.append(branch)
    if not out:
        raise ValueError("--branches selected no branches")
    return out


def parse_skill_order(value: str) -> List[str]:
    order = [part.strip().lower().replace("-", "_") for part in str(value).split(",") if part.strip()]
    unknown = [name for name in order if name not in SKILL_RECIPES]
    if unknown:
        raise ValueError(f"unknown skills: {unknown}; allowed: {sorted(SKILL_RECIPES)}")
    return order


def resolve_skill_order_arg(args: argparse.Namespace) -> List[str]:
    if str(args.skill_order or "").strip():
        return parse_skill_order(args.skill_order)
    profile = str(getattr(args, "task_profile", "five_ood")).strip().lower()
    if profile not in TASK_PROFILES:
        raise ValueError(f"unknown task profile {profile!r}; allowed: {sorted(TASK_PROFILES)}")
    return list(TASK_PROFILES[profile])


@dataclass(frozen=True)
class SkillRecipe:
    name: str
    display_name: str
    dataset_id: str
    config: Optional[str]
    train_splits: Tuple[str, ...]
    eval_splits: Tuple[str, ...]
    max_new_tokens: int
    token_gate: float
    exact_gate: float
    source_keys: Tuple[str, ...] = (
        "problem",
        "question",
        "instruction",
        "prompt",
        "input",
        "query",
        "text",
        "source",
    )
    target_keys: Tuple[str, ...] = (
        "generated_solution",
        "expected_answer",
        "solution",
        "answer",
        "output",
        "response",
        "completion",
        "target",
        "query",
        "sql",
        "code",
    )


SKILL_RECIPES: Dict[str, SkillRecipe] = {
    "tooluse": SkillRecipe(
        name="tooluse",
        display_name="Tool Use",
        dataset_id="Akicou/merged-tool-use",
        config=None,
        train_splits=("train",),
        eval_splits=("train",),
        max_new_tokens=180,
        token_gate=0.35,
        exact_gate=0.05,
        source_keys=("messages", "conversations", "prompt", "instruction", "input", "query", "text"),
        target_keys=("messages", "conversations", "response", "completion", "output", "answer", "target"),
    ),
    "science": SkillRecipe(
        name="science",
        display_name="Science QA",
        dataset_id="allenai/sciq",
        config=None,
        train_splits=("train",),
        eval_splits=("validation", "test"),
        max_new_tokens=64,
        token_gate=0.45,
        exact_gate=0.20,
        source_keys=("question", "support", "prompt", "input", "text"),
        target_keys=("correct_answer", "answer", "target", "output"),
    ),
    "math": SkillRecipe(
        name="math",
        display_name="Math reasoning",
        dataset_id="nvidia/OpenMathInstruct-2",
        config=None,
        train_splits=("train",),
        eval_splits=("train",),
        max_new_tokens=96,
        token_gate=0.45,
        exact_gate=0.20,
        source_keys=("problem", "question", "instruction", "input", "prompt", "text"),
        target_keys=("generated_solution", "solution", "expected_answer", "answer", "output", "target"),
    ),
    "code": SkillRecipe(
        name="code",
        display_name="Code generation",
        dataset_id="nvidia/OpenCodeInstruct",
        config=None,
        train_splits=("train",),
        eval_splits=("train",),
        max_new_tokens=160,
        token_gate=0.35,
        exact_gate=0.05,
    ),
    "sql": SkillRecipe(
        name="sql",
        display_name="Text-to-SQL",
        dataset_id="gretelai/synthetic_text_to_sql",
        config=None,
        train_splits=("train",),
        eval_splits=("train",),
        max_new_tokens=128,
        token_gate=0.45,
        exact_gate=0.10,
        source_keys=("question", "sql_prompt", "prompt", "input", "natural_language", "text"),
        target_keys=("sql", "sql_context", "query", "answer", "output", "target"),
    ),
    "medical": SkillRecipe(
        name="medical",
        display_name="Medical QA",
        dataset_id="openlifescienceai/medmcqa",
        config=None,
        train_splits=("train",),
        eval_splits=("validation", "test"),
        max_new_tokens=48,
        token_gate=0.45,
        exact_gate=0.20,
        source_keys=("question", "input", "prompt", "text"),
        target_keys=("answer", "cop", "target", "output", "label"),
    ),
    "cowrite": SkillRecipe(
        name="cowrite",
        display_name="Co-writing",
        dataset_id="Delta-Vector/Orion-Co-Writer-51K",
        config=None,
        train_splits=("train",),
        eval_splits=("train",),
        max_new_tokens=180,
        token_gate=0.30,
        exact_gate=0.02,
        source_keys=("prompt", "instruction", "input", "messages", "conversations", "text"),
        target_keys=("response", "completion", "output", "chosen", "assistant", "target"),
    ),
    "legal": SkillRecipe(
        name="legal",
        display_name="Legal reasoning",
        dataset_id="DatologyAI/legalbench",
        config=None,
        train_splits=("train",),
        eval_splits=("test", "validation", "train"),
        max_new_tokens=80,
        token_gate=0.40,
        exact_gate=0.15,
    ),
}


BENCHMARK_RECIPES: Dict[str, SkillRecipe] = {
    "mmlu_pro": SkillRecipe(
        name="mmlu_pro",
        display_name="MMLU-Pro mini",
        dataset_id="TIGER-Lab/MMLU-Pro",
        config=None,
        train_splits=("test", "validation", "train"),
        eval_splits=("test", "validation", "train"),
        max_new_tokens=8,
        token_gate=0.20,
        exact_gate=0.20,
        source_keys=("question", "prompt", "input"),
        target_keys=("answer", "target", "label"),
    ),
    "gpqa": SkillRecipe(
        name="gpqa",
        display_name="GPQA mini",
        dataset_id="Idavidrein/gpqa",
        config="gpqa_diamond",
        train_splits=("train",),
        eval_splits=("train",),
        max_new_tokens=8,
        token_gate=0.20,
        exact_gate=0.20,
        source_keys=("Question", "question", "prompt"),
        target_keys=("Correct Answer", "answer", "target"),
    ),
}


def prompt_and_target(recipe: SkillRecipe, row: Dict[str, Any], index: int) -> Optional[Tuple[str, str, str, str]]:
    if recipe.name == "tooluse":
        prompt_text, target = extract_chat_or_instruct_pair(row)
        tools = row.get("tools", row.get("functions", row.get("available_tools", "")))
        tools_text = compact_text(tools, max_chars=2500)
        if not prompt_text:
            prompt_text = first_nonempty(row, recipe.source_keys)
        if not target:
            target = first_nonempty(row, recipe.target_keys)
        source = prompt_text if not tools_text else f"{prompt_text}\n\nAvailable tools:\n{tools_text}"
        prompt = (
            "TOOL USE\n"
            "Answer the user request. If a tool call is needed, write the tool call clearly.\n"
            f"Request:\n{source}\nResponse:\n"
        )
        return prompt, target, source, target

    if recipe.name == "science":
        question = first_nonempty(row, ("question", "prompt", "input", "text"))
        support = first_nonempty(row, ("support", "context", "passage", "rationale"))
        correct = first_nonempty(row, ("correct_answer", "answer", "target", "output"))
        distractors = [compact_text(row.get(key)) for key in ("distractor1", "distractor2", "distractor3")]
        options = [item for item in [correct, *distractors] if item]
        if len(options) >= 2:
            rng = np.random.default_rng(stable_seed(question, index + 31))
            rng.shuffle(options)
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            option_lines = [f"{letters[idx]}. {opt}" for idx, opt in enumerate(options)]
            target = letters[options.index(correct)] if correct in options else correct
            choices = "\n".join(option_lines)
            source = question if not support else f"{question}\n\nContext:\n{support}"
            prompt = f"SCIENCE QA\nChoose the best answer. Reply with only the option letter.\n{source}\n{choices}\nAnswer:\n"
        else:
            target = correct
            source = question if not support else f"{question}\n\nContext:\n{support}"
            prompt = f"SCIENCE QA\nAnswer the science question.\n{source}\nAnswer:\n"
        return prompt, target, source, target

    if recipe.name == "medical":
        question = first_nonempty(row, ("question", "input", "prompt", "text"))
        options = [compact_text(row.get(key)) for key in ("opa", "opb", "opc", "opd")]
        option_lines = [f"{letter}. {text}" for letter, text in zip("ABCD", options) if text]
        source = question + ("\n" + "\n".join(option_lines) if option_lines else "")
        cop = row.get("cop")
        if cop is not None and str(cop).strip() != "":
            try:
                idx = int(cop)
                if 0 <= idx < 4:
                    target = "ABCD"[idx]
                elif 1 <= idx <= 4:
                    target = "ABCD"[idx - 1]
                else:
                    target = compact_text(cop)
            except Exception:
                target = compact_text(cop)
        else:
            target = first_nonempty(row, recipe.target_keys)
        prompt = f"MEDICAL QA\nChoose the best answer. Reply with only the option letter.\n{source}\nAnswer:\n"
        return prompt, target, source, target

    if recipe.name == "sql":
        question = first_nonempty(row, ("question", "sql_prompt", "prompt", "input", "natural_language", "text"))
        schema = first_nonempty(row, ("schema", "context", "sql_context", "table_metadata", "create_table_statement"))
        target = first_nonempty(row, recipe.target_keys)
        source = question if not schema else f"{question}\n\nSchema:\n{schema}"
        prompt = f"TEXT TO SQL\nWrite one SQL query for the request.\nRequest:\n{source}\nSQL:\n"
        return prompt, target, source, target

    if recipe.name == "code":
        source = first_nonempty(row, ("instruction", "prompt", "question", "input", "text", "problem"))
        target = first_nonempty(row, recipe.target_keys)
        prompt = f"CODE GENERATION\nWrite correct Python code for the task.\nTask:\n{source}\nCode:\n"
        return prompt, target, source, target

    if recipe.name == "cowrite":
        prompt_text, target = extract_chat_or_instruct_pair(row)
        if not prompt_text or not target:
            prompt_text = first_nonempty(row, recipe.source_keys)
            target = first_nonempty(row, recipe.target_keys)
        prompt = f"CO-WRITING\nContinue the passage in the same style.\nPassage:\n{prompt_text}\nContinuation:\n"
        return prompt, target, prompt_text, target

    if recipe.name == "math":
        source = first_nonempty(row, ("problem", "question", "instruction", "input", "prompt", "text"))
        target = first_nonempty(row, recipe.target_keys)
        prompt = f"MATH REASONING\nSolve the problem. Give the final answer clearly.\nProblem:\n{source}\nSolution:\n"
        return prompt, target, source, target

    if recipe.name in BENCHMARK_RECIPES:
        source, target = benchmark_source_target(recipe, row)
        prompt = f"{recipe.display_name.upper()}\nAnswer the question. Reply with the answer only.\n{source}\nAnswer:\n"
        return prompt, target, source, target

    source = first_nonempty(row, recipe.source_keys)
    target = first_nonempty(row, recipe.target_keys)
    prompt = f"{recipe.display_name.upper()}\nInput:\n{source}\nAnswer:\n"
    return prompt, target, source, target


def extract_chat_or_instruct_pair(row: Dict[str, Any]) -> Tuple[str, str]:
    for key in ("messages", "conversations"):
        value = row.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                value = None
        if isinstance(value, list):
            turns: List[Dict[str, Any]] = [turn for turn in value if isinstance(turn, dict)]
            if len(turns) >= 2:
                prompt_parts: List[str] = []
                for turn in turns:
                    role = str(turn.get("role", turn.get("from", ""))).lower()
                    content = compact_text(turn.get("content", turn.get("value", "")), max_chars=3000)
                    if not content:
                        continue
                    if "assistant" in role or "gpt" in role or role == "model":
                        if prompt_parts:
                            return "\n".join(prompt_parts), content
                    else:
                        prompt_parts.append(content)
    prompt = first_nonempty(row, ("prompt", "instruction", "input", "text"))
    target = first_nonempty(row, ("response", "completion", "output", "chosen", "assistant"))
    return prompt, target


def benchmark_source_target(recipe: SkillRecipe, row: Dict[str, Any]) -> Tuple[str, str]:
    if recipe.name == "mmlu_pro":
        question = first_nonempty(row, ("question", "prompt", "input"))
        options = row.get("options")
        option_lines: List[str] = []
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                options = None
        if isinstance(options, (list, tuple)):
            option_lines = [f"{chr(65 + idx)}. {compact_text(opt, max_chars=500)}" for idx, opt in enumerate(options)]
        source = question + ("\n" + "\n".join(option_lines) if option_lines else "")
        answer = row.get("answer")
        if isinstance(answer, int):
            target = chr(65 + int(answer))
        else:
            target = compact_text(answer)
            if target.isdigit():
                idx = int(target)
                target = chr(65 + idx) if 0 <= idx < 26 else target
        return source, target

    if recipe.name == "gpqa":
        question = first_nonempty(row, ("Question", "question", "prompt"))
        target = first_nonempty(row, ("Correct Answer", "answer", "target"))
        distractors = [
            first_nonempty(row, (f"Incorrect Answer {idx}", f"incorrect_answer_{idx}"))
            for idx in range(1, 4)
        ]
        options = [target] + [item for item in distractors if item]
        rng = np.random.default_rng(stable_seed(question, 77))
        rng.shuffle(options)
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        option_lines = [f"{letters[idx]}. {opt}" for idx, opt in enumerate(options)]
        correct_idx = options.index(target) if target in options else 0
        source = question + "\n" + "\n".join(option_lines)
        return source, letters[correct_idx]

    return first_nonempty(row, recipe.source_keys), first_nonempty(row, recipe.target_keys)


ANSWER_FRIENDLY_COMPOSITION_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("tooluse", "science"),
    ("science", "medical"),
    ("tooluse", "medical"),
    ("medical", "tooluse"),
    ("math", "sql"),
    ("sql", "medical"),
    ("math", "medical"),
    ("medical", "math"),
    ("math", "code"),
    ("code", "sql"),
    ("cowrite", "math"),
)


def make_example(recipe: SkillRecipe, row: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    converted = prompt_and_target(recipe, row, index)
    if converted is None:
        return None
    prompt, target, source, raw_target = converted
    prompt = compact_text(prompt, max_chars=5000)
    target = compact_text(target, max_chars=2500)
    if not prompt or not target:
        return None
    return {
        "prompt": prompt,
        "target": target,
        "source": compact_text(source, max_chars=4000),
        "raw_target": compact_text(raw_target, max_chars=2500),
        "row_index": int(index),
    }


def synthetic_examples(recipe: SkillRecipe, count: int, *, seed: int, heldout: bool) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed + (10_000 if heldout else 0))
    out: List[Dict[str, Any]] = []
    for idx in range(int(count)):
        a = int(rng.integers(2, 99))
        b = int(rng.integers(2, 99))
        if recipe.name == "math":
            prompt = f"MATH REASONING\nSolve the problem. Give the final answer clearly.\nProblem:\nWhat is {a} + {b}?\nSolution:\n"
            target = str(a + b)
        elif recipe.name == "code":
            prompt = f"CODE GENERATION\nWrite correct Python code for the task.\nTask:\nReturn the square of {a} from a function named solve.\nCode:\n"
            target = f"def solve():\n    return {a * a}"
        elif recipe.name == "sql":
            prompt = f"TEXT TO SQL\nWrite one SQL query for the request.\nRequest:\nSelect rows from table t where value is greater than {a}.\nSQL:\n"
            target = f"SELECT * FROM t WHERE value > {a};"
        elif recipe.name == "medical":
            prompt = "MEDICAL QA\nChoose the best answer. Reply with only the option letter.\nA patient has fever. Which option is a symptom?\nA. Fever\nB. Granite\nC. Triangle\nD. SQL\nAnswer:\n"
            target = "A"
        elif recipe.name == "cowrite":
            prompt = f"CO-WRITING\nContinue the passage in the same style.\nPassage:\nThe lighthouse blinked {a} times over the quiet bay.\nContinuation:\n"
            target = f"The keeper counted {a + b} heartbeats before the fog answered back."
        else:
            prompt = f"{recipe.display_name.upper()}\nInput:\nSynthetic item {a} {b}\nAnswer:\n"
            target = str(a + b)
        out.append({"prompt": prompt, "target": target, "source": prompt, "raw_target": target, "row_index": idx})
    return out


def load_split_rows(
    recipe: SkillRecipe,
    *,
    split_candidates: Sequence[str],
    count: int,
    seed: int,
    streaming: bool,
    buffer_size: int,
) -> List[Dict[str, Any]]:
    if load_dataset is None:
        raise RuntimeError("datasets is not installed")
    errors: List[str] = []
    for split in split_candidates:
        try:
            if streaming:
                ds = load_dataset(recipe.dataset_id, recipe.config, split=split, streaming=True)
                if hasattr(ds, "shuffle"):
                    ds = ds.shuffle(seed=seed, buffer_size=int(buffer_size))
                iterator: Iterable[Any] = iter(ds)
            else:
                ds = load_dataset(recipe.dataset_id, recipe.config, split=split)
                ds = ds.shuffle(seed=seed)
                iterator = iter(ds)
            examples: List[Dict[str, Any]] = []
            scanned = 0
            max_scan = max(int(count) * 60, int(count) + 100)
            for row in iterator:
                scanned += 1
                if not isinstance(row, dict):
                    continue
                ex = make_example(recipe, row, scanned)
                if ex is not None:
                    examples.append(ex)
                if len(examples) >= int(count):
                    break
                if scanned >= max_scan:
                    break
            if len(examples) >= int(count):
                return examples[: int(count)]
            errors.append(f"{split}: only {len(examples)} usable rows")
        except Exception as exc:
            errors.append(f"{split}: {exc}")
    raise RuntimeError(f"could not load {recipe.name} from {recipe.dataset_id}: {' | '.join(errors[:5])}")


def build_skill_payload(recipe: SkillRecipe, args: argparse.Namespace, *, skill_index: int) -> Dict[str, Any]:
    if args.smoke:
        train = synthetic_examples(recipe, args.train_samples, seed=args.seed + 1000 + skill_index, heldout=False)
        eval_rows = synthetic_examples(recipe, args.eval_samples, seed=args.seed + 2000 + skill_index, heldout=True)
        source = "synthetic:smoke"
    else:
        try:
            train = load_split_rows(
                recipe,
                split_candidates=recipe.train_splits,
                count=args.train_samples,
                seed=args.seed + 1000 + skill_index,
                streaming=not args.no_streaming,
                buffer_size=args.streaming_buffer_size,
            )
            eval_rows = load_split_rows(
                recipe,
                split_candidates=recipe.eval_splits,
                count=args.eval_samples,
                seed=args.seed + 2000 + skill_index,
                streaming=not args.no_streaming,
                buffer_size=args.streaming_buffer_size,
            )
            source = recipe.dataset_id
        except Exception as exc:
            if not args.allow_synthetic_fallback:
                raise
            print(f"[manifest:{recipe.name}] WARNING: falling back to synthetic rows after load error: {exc}", flush=True)
            train = synthetic_examples(recipe, args.train_samples, seed=args.seed + 1000 + skill_index, heldout=False)
            eval_rows = synthetic_examples(recipe, args.eval_samples, seed=args.seed + 2000 + skill_index, heldout=True)
            source = f"synthetic:fallback:{exc}"
    return {
        "recipe": asdict(recipe),
        "source": source,
        "train": train,
        "eval": eval_rows,
    }


def build_composition_payload(skills: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    rng = np.random.default_rng(args.seed + 9090)
    by_name = {item["recipe"]["name"]: item for item in skills}
    available_pairs = [(a, b) for a, b in ANSWER_FRIENDLY_COMPOSITION_PAIRS if a in by_name and b in by_name]
    examples: List[Dict[str, Any]] = []
    for idx in range(int(args.composition_eval_samples)):
        if not available_pairs:
            break
        left_name, right_name = available_pairs[idx % len(available_pairs)]
        left_rows = by_name[left_name]["eval"]
        right_rows = by_name[right_name]["eval"]
        left = left_rows[int(rng.integers(0, len(left_rows)))]
        right = right_rows[int(rng.integers(0, len(right_rows)))]
        prompt = (
            "COMPOSITION TASK\n"
            "Solve both subtasks. Keep the two answers separated.\n\n"
            f"Subtask A ({left_name}):\n{left['source']}\n\n"
            f"Subtask B ({right_name}):\n{right['source']}\n\n"
            "Combined answer:\n"
        )
        target = (
            f"{left_name.upper()} ANSWER:\n{left['target']}\n\n"
            f"{right_name.upper()} ANSWER:\n{right['target']}"
        )
        examples.append(
            {
                "prompt": compact_text(prompt, max_chars=6000),
                "target": compact_text(target, max_chars=3500),
                "source": f"{left_name}+{right_name}",
                "raw_target": target,
                "pair": f"{left_name}+{right_name}",
                "parts": [
                    {
                        "slot": "A",
                        "skill": left_name,
                        "source": left["source"],
                        "target": left["target"],
                    },
                    {
                        "slot": "B",
                        "skill": right_name,
                        "source": right["source"],
                        "target": right["target"],
                    },
                ],
                "row_index": idx,
            }
        )
    recipe = SkillRecipe(
        name="composition",
        display_name="Skill composition",
        dataset_id="synthetic:paired_heldout_skill_composition",
        config=None,
        train_splits=("manifest",),
        eval_splits=("manifest",),
        max_new_tokens=220,
        token_gate=0.25,
        exact_gate=0.02,
    )
    return {"recipe": asdict(recipe), "source": recipe.dataset_id, "train": [], "eval": examples}


def build_benchmark_payloads(args: argparse.Namespace) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    names = [part.strip().lower() for part in str(args.benchmarks).split(",") if part.strip()]
    if names == ["none"]:
        return payloads
    for idx, name in enumerate(names):
        if name not in BENCHMARK_RECIPES:
            raise ValueError(f"unknown benchmark {name!r}; allowed: none,{','.join(BENCHMARK_RECIPES)}")
        recipe = BENCHMARK_RECIPES[name]
        if args.smoke:
            eval_rows = synthetic_examples(recipe, args.benchmark_eval_samples, seed=args.seed + 3000 + idx, heldout=True)
            source = "synthetic:smoke"
        else:
            eval_rows = load_split_rows(
                recipe,
                split_candidates=recipe.eval_splits,
                count=args.benchmark_eval_samples,
                seed=args.seed + 3000 + idx,
                streaming=not args.no_streaming,
                buffer_size=args.streaming_buffer_size,
            )
            source = recipe.dataset_id
        payloads.append({"recipe": asdict(recipe), "source": source, "train": [], "eval": eval_rows})
    return payloads


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    skill_names = resolve_skill_order_arg(args)
    skills = [
        build_skill_payload(SKILL_RECIPES[name], args, skill_index=idx)
        for idx, name in enumerate(skill_names)
    ]
    manifest = {
        "version": MANIFEST_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(args.seed),
        "model_id": args.model_id or DEFAULT_MODEL_ID,
        "train_samples": int(args.train_samples),
        "eval_samples": int(args.eval_samples),
        "skill_order": skill_names,
        "task_profile": str(args.task_profile),
        "skills": skills,
        "composition": build_composition_payload(skills, args) if args.build_composition else None,
        "benchmarks": build_benchmark_payloads(args),
        "notes": {
            "frozen_manifest": "Use this exact file for naive_sft, sdft_baseline/op_sdft, and amoeba branches.",
            "no_replay_claim": "Task examples are fixed before training; branch updates should not sample old-task train rows.",
        },
    }
    return manifest


def write_manifest(manifest: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return path


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("version", -1)) != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version {manifest.get('version')}; expected {MANIFEST_VERSION}")
    return manifest


def task_from_payload(payload: Dict[str, Any]) -> al.TaskData:
    recipe = payload["recipe"]
    spec = al.TaskSpec(
        name=recipe["name"],
        display_name=recipe["display_name"],
        prompt_template="{source}",
        dataset_id=recipe.get("dataset_id", ""),
        config=recipe.get("config"),
        train_split_names=tuple(recipe.get("train_splits", ("train",))),
        eval_split_names=tuple(recipe.get("eval_splits", ("validation", "test", "train"))),
        source_keys=tuple(recipe.get("source_keys", ())),
        target_keys=tuple(recipe.get("target_keys", ())),
        max_target_tokens=max(int(recipe.get("max_new_tokens", 96)), 8),
        max_source_tokens=256,
        max_new_tokens=int(recipe.get("max_new_tokens", 96)),
        train_samples=len(payload.get("train", [])),
        eval_samples=len(payload.get("eval", [])),
        token_gate=float(recipe.get("token_gate", 0.0)),
        exact_gate=float(recipe.get("exact_gate", 0.0)),
        citation=str(payload.get("source", recipe.get("dataset_id", ""))),
    )
    train = [al.AlienExample(**{k: row.get(k, "") for k in ("prompt", "target", "source", "raw_target")}) for row in payload.get("train", [])]
    eval_rows = [al.AlienExample(**{k: row.get(k, "") for k in ("prompt", "target", "source", "raw_target")}) for row in payload.get("eval", [])]
    return al.TaskData(
        spec=spec,
        train=train,
        eval=eval_rows,
        manifest={
            "source": payload.get("source"),
            "recipe": recipe,
            "eval_payload": payload.get("eval", []),
        },
    )


def tasks_from_manifest(manifest: Dict[str, Any]) -> Tuple[List[al.TaskData], Optional[al.TaskData], List[al.TaskData]]:
    skills = [task_from_payload(payload) for payload in manifest.get("skills", [])]
    composition = task_from_payload(manifest["composition"]) if manifest.get("composition") else None
    benchmarks = [task_from_payload(payload) for payload in manifest.get("benchmarks", [])]
    return skills, composition, benchmarks


def normalize_for_match(text: str) -> str:
    text = al.truncate_completion(str(text))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_composition_completion(text: str) -> str:
    text = str(text)
    for stop in ("<|endoftext|>",):
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


def normalize_compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_for_match(text)).strip()


def normalize_sql(text: str) -> str:
    text = normalize_for_match(text).rstrip(";")
    text = re.sub(r"\s*([(),=<>;+*/-])\s*", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_number(text: str) -> str:
    return str(text).strip().replace(",", "").rstrip(".")


def extract_final_number(text: str) -> str:
    text = al.truncate_completion(str(text))
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return normalize_number(boxed[-1])
    hashes = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)", text)
    if hashes:
        return normalize_number(hashes[-1])
    answerish = re.findall(
        r"(?:final answer|answer|therefore|thus)\s*(?:is|=|:)?\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if answerish:
        return normalize_number(answerish[-1])
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?", text)
    return normalize_number(numbers[-1]) if numbers else ""


def extract_choice_letter(text: str) -> str:
    text = al.truncate_completion(str(text)).strip()
    answerish = re.search(r"(?:answer|option|choice)\s*(?:is|=|:)?\s*([A-D])\b", text, flags=re.IGNORECASE)
    if answerish:
        return answerish.group(1).upper()
    first = re.search(r"\b([A-D])\b", text, flags=re.IGNORECASE)
    return first.group(1).upper() if first else ""


def extract_composition_parts_from_target(target: str, pair: str) -> List[Dict[str, str]]:
    names = [part.strip().lower() for part in str(pair).split("+") if part.strip()]
    parts: List[Dict[str, str]] = []
    for idx, name in enumerate(names):
        header = re.compile(rf"(?im)^\s*{re.escape(name)}\s+answer\s*:\s*")
        match = header.search(target)
        if not match:
            continue
        next_start = len(target)
        for other in names[idx + 1 :]:
            other_match = re.compile(rf"(?im)^\s*{re.escape(other)}\s+answer\s*:\s*").search(target, match.end())
            if other_match:
                next_start = min(next_start, other_match.start())
        parts.append({"slot": "AB"[idx] if idx < 2 else str(idx), "skill": name, "target": target[match.end() : next_start].strip()})
    return parts


def composition_parts_for_example(task: al.TaskData, idx: int, example: al.AlienExample) -> List[Dict[str, str]]:
    payloads = task.manifest.get("eval_payload", []) if isinstance(task.manifest, dict) else []
    if idx < len(payloads):
        payload_parts = payloads[idx].get("parts") if isinstance(payloads[idx], dict) else None
        if isinstance(payload_parts, list) and payload_parts:
            return [
                {
                    "slot": str(part.get("slot", "")),
                    "skill": str(part.get("skill", "")).lower(),
                    "target": str(part.get("target", "")),
                    "source": str(part.get("source", "")),
                }
                for part in payload_parts
                if isinstance(part, dict) and part.get("skill") and part.get("target")
            ]
    return extract_composition_parts_from_target(example.target, example.source)


def section_header_match(text: str, skill: str, slot: str) -> Optional[re.Match[str]]:
    skill_label = re.escape(skill)
    slot_label = re.escape(slot)
    patterns = [
        rf"(?im)^\s*{skill_label}\s+answer\s*:\s*",
        rf"(?im)^\s*{skill_label}\s*(?:solution|sql|code|response)?\s*:\s*",
        rf"(?im)^\s*(?:subtask\s*)?{slot_label}\s*(?:\([^)]+\))?\s*(?:answer|solution)?\s*:\s*",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match
    return None


def extract_generated_composition_sections(completion: str, parts: Sequence[Dict[str, str]]) -> Dict[str, str]:
    text = truncate_composition_completion(completion)
    headers: List[Tuple[int, int, str]] = []
    for idx, part in enumerate(parts):
        skill = str(part.get("skill", "")).lower()
        slot = str(part.get("slot", "AB"[idx] if idx < 2 else idx)).lower()
        match = section_header_match(text, skill, slot)
        if match:
            headers.append((match.start(), match.end(), skill))
    if headers:
        headers = sorted(headers)
        out: Dict[str, str] = {}
        for idx, (_, end, skill) in enumerate(headers):
            next_start = headers[idx + 1][0] if idx + 1 < len(headers) else len(text)
            out[skill] = text[end:next_start].strip()
        return out
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    if len(chunks) >= len(parts):
        return {str(part.get("skill", "")).lower(): chunks[idx] for idx, part in enumerate(parts)}
    return {}


def score_composition_part(tokenizer, skill: str, prediction: str, target: str) -> Tuple[bool, float, str]:
    skill = str(skill).lower()
    prediction = al.truncate_completion(prediction)
    target = str(target)
    token_acc = al.bpe_token_acc(tokenizer, prediction, target)
    exact = bool(al.prefix_exact(prediction, target))

    if skill == "medical":
        pred_letter = extract_choice_letter(prediction)
        gold_letter = extract_choice_letter(target) or normalize_compact(target)[:1].upper()
        ok = bool(pred_letter and pred_letter == gold_letter)
        return ok, 1.0 if ok else token_acc, "choice_letter"

    if skill == "math":
        pred_num = extract_final_number(prediction)
        gold_num = extract_final_number(target)
        ok = bool(pred_num and gold_num and pred_num == gold_num)
        if ok:
            return True, 1.0, "final_number"
        return exact or token_acc >= 0.65, max(float(exact), token_acc), "math_text"

    if skill == "sql":
        pred_sql = normalize_sql(prediction)
        gold_sql = normalize_sql(target)
        ok = bool(gold_sql and (pred_sql == gold_sql or gold_sql in pred_sql))
        if ok:
            return True, 1.0, "normalized_sql"
        return exact or token_acc >= 0.70, max(float(exact), token_acc), "sql_text"

    if skill == "code":
        pred_norm = normalize_compact(prediction)
        gold_norm = normalize_compact(target)
        ok = bool(gold_norm and (pred_norm == gold_norm or pred_norm.startswith(gold_norm) or gold_norm in pred_norm))
        if ok:
            return True, 1.0, "normalized_code"
        return exact or token_acc >= 0.60, max(float(exact), token_acc), "code_text"

    threshold = 0.35 if skill == "cowrite" else 0.55
    return exact or token_acc >= threshold, max(float(exact), token_acc), "text"


@torch.no_grad()
def composition_generation_metrics(
    model,
    tokenizer,
    task: al.TaskData,
    device: str,
    eval_batch_size: int,
) -> Dict[str, float]:
    prompts = [example.prompt for example in task.eval]
    completions: List[str] = []
    for start in range(0, len(prompts), eval_batch_size):
        completions.extend(
            al.generate_on_policy_completions(
                model,
                tokenizer,
                prompts[start : start + eval_batch_size],
                device,
                max_new_tokens=task.spec.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )
        )

    example_successes: List[float] = []
    format_successes: List[float] = []
    part_scores: List[float] = []
    part_corrects: List[float] = []
    pair_success: Dict[str, List[float]] = {}
    skill_correct: Dict[str, List[float]] = {}
    skill_score: Dict[str, List[float]] = {}

    for idx, (example, completion) in enumerate(zip(task.eval, completions)):
        parts = composition_parts_for_example(task, idx, example)
        sections = extract_generated_composition_sections(completion, parts)
        if not parts:
            continue
        pair = "+".join(str(part.get("skill", "")).lower() for part in parts)
        format_ok = all(str(part.get("skill", "")).lower() in sections for part in parts)
        format_successes.append(float(format_ok))
        part_oks: List[bool] = []
        for part in parts:
            skill = str(part.get("skill", "")).lower()
            target = str(part.get("target", ""))
            prediction = sections.get(skill, completion if len(parts) == 1 else "")
            ok, score, _method = score_composition_part(tokenizer, skill, prediction, target)
            part_oks.append(bool(ok))
            part_corrects.append(float(ok))
            part_scores.append(float(score))
            skill_correct.setdefault(skill, []).append(float(ok))
            skill_score.setdefault(skill, []).append(float(score))
        success = float(bool(part_oks) and all(part_oks))
        example_successes.append(success)
        pair_success.setdefault(pair, []).append(success)

    metrics: Dict[str, float] = {
        "composition_success": float(sum(example_successes) / max(len(example_successes), 1)),
        "composition_part_acc": float(sum(part_corrects) / max(len(part_corrects), 1)),
        "composition_part_score": float(sum(part_scores) / max(len(part_scores), 1)),
        "composition_format_acc": float(sum(format_successes) / max(len(format_successes), 1)),
        "composition_samples": float(len(example_successes)),
        "composition_parts": float(len(part_corrects)),
        "composition_exact": float(sum(example_successes) / max(len(example_successes), 1)),
        "composition_token_acc": float(sum(part_corrects) / max(len(part_corrects), 1)),
    }
    for pair, values in pair_success.items():
        key = "composition_pair_" + re.sub(r"[^a-z0-9]+", "_", pair.lower()).strip("_") + "_success"
        metrics[key] = float(sum(values) / max(len(values), 1))
    for skill, values in skill_correct.items():
        metrics[f"composition_{skill}_part_acc"] = float(sum(values) / max(len(values), 1))
    for skill, values in skill_score.items():
        metrics[f"composition_{skill}_part_score"] = float(sum(values) / max(len(values), 1))
    return metrics


def make_runtime_config(args: argparse.Namespace, out_dir: Path) -> qp.RuntimeConfig:
    return qp.RuntimeConfig(
        model_id=args.model_id,
        device=args.device,
        dtype=choose_dtype(args.dtype),
        local_files_only=args.local_files_only,
        resume=False,
        smoke=args.smoke,
        output_dir=out_dir,
        backup_dir=None,
        seed=args.seed,
        phase_scope="qwen35_five_skill",
        task_suite="five_skill_ood",
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        consolidation_micro_batch_size=args.micro_batch_size,
        max_seq_len=args.max_seq_len,
        gradient_checkpointing=args.gradient_checkpointing,
        wikitext_eval_samples=args.wikitext_eval_samples,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        grad_clip=args.grad_clip,
    )


def load_wikitext_eval(tokenizer, cfg: qp.RuntimeConfig, args: argparse.Namespace) -> List[torch.Tensor]:
    return qp.load_wikitext_texts(
        tokenizer,
        split=args.wikitext_eval_split,
        max_seq_len=cfg.max_seq_len,
        max_samples=cfg.wikitext_eval_samples,
        local_files_only=args.local_files_only,
    )


def collect_base_language_profile(model, tokenizer, cfg: qp.RuntimeConfig, args: argparse.Namespace, logger: al.ArtifactLogger) -> List[Any]:
    if not args.protect_base_language_profile:
        return []
    subsection("Base language geometry profile")
    chunks = qp.load_wikitext_texts(
        tokenizer,
        split=args.base_language_profile_split,
        max_seq_len=cfg.max_seq_len,
        max_samples=args.base_language_profile_samples,
        local_files_only=args.local_files_only,
    )
    batch_fn = qp.make_wikitext_batch_fn(tokenizer, chunks, cfg.device, cfg, cfg.seed + 9001)
    profile = qp._collect_profiles(model, "base_language", batch_fn)
    logger.log_event(
        "base_language_profile_collected",
        split=args.base_language_profile_split,
        samples=len(chunks),
        protected_profiles=1,
        update_proxy_batches=0,
    )
    print(
        f"[base_language_profile] split={args.base_language_profile_split} "
        f"samples={len(chunks)} update_proxy_batches=0",
        flush=True,
    )
    return [profile]


def active_eval_tasks(
    learned: Sequence[al.TaskData],
    benchmarks: Sequence[al.TaskData],
    *,
    composition: Optional[al.TaskData] = None,
    include_composition: bool = False,
) -> List[al.TaskData]:
    tasks = list(learned) + list(benchmarks)
    if include_composition and composition is not None:
        tasks.append(composition)
    return tasks


def evaluate_and_log(
    *,
    model,
    tokenizer,
    cfg: qp.RuntimeConfig,
    logger: al.ArtifactLogger,
    stage: str,
    learned: Sequence[al.TaskData],
    benchmarks: Sequence[al.TaskData],
    composition: Optional[al.TaskData],
    include_composition: bool,
    generation: bool,
) -> Dict[str, float]:
    tasks = active_eval_tasks(learned, benchmarks, composition=composition, include_composition=include_composition)
    metrics = al.evaluate_suite(model, tokenizer, tasks, cfg, do_generation=generation, include_wikitext=True)
    al.print_metrics(stage, metrics, [task.spec.name for task in tasks])
    logger.log_stage_summary(stage, metrics, old_task_examples=0, proxy_batches=0)
    return metrics


def evaluate_suite_progress(
    model,
    tokenizer,
    tasks: Sequence[al.TaskData],
    cfg: qp.RuntimeConfig,
    *,
    do_generation: bool = True,
    include_wikitext: bool = True,
    wikitext_val: Optional[List[torch.Tensor]] = None,
) -> Dict[str, float]:
    """Drop-in replacement for alien_ladder evaluate_suite with visible progress."""
    metrics: Dict[str, float] = {}
    _wikitext = wikitext_val or getattr(cfg, "_wikitext_val", None)
    if include_wikitext and _wikitext is not None:
        print(f"[eval] start wikitext chunks={len(_wikitext)}", flush=True)
        try:
            metrics.update(qp.evaluate_retention(model, tokenizer, _wikitext, cfg.device, cfg.eval_batch_size))
            print(f"[eval] done wikitext ppl={al.fmt(metrics.get('wikitext_ppl'))}", flush=True)
        except Exception as exc:
            metrics["wikitext_error"] = str(exc)
            print(f"[eval] wikitext error={exc}", flush=True)
    for idx, task in enumerate(tasks, start=1):
        print(
            f"[eval] start task={task.spec.name} {idx}/{len(tasks)} "
            f"eval_rows={len(task.eval)} generation={bool(do_generation)}",
            flush=True,
        )
        if task.spec.name == "composition":
            metrics.update(al.teacher_forced_metrics(model, tokenizer, task, cfg.device, cfg.max_seq_len, cfg.eval_batch_size))
            strict_generation = bool(do_generation or getattr(cfg, "_composition_strict_eval", True))
            if strict_generation:
                print("[eval] start composition strict generation", flush=True)
                metrics.update(composition_generation_metrics(model, tokenizer, task, cfg.device, cfg.eval_batch_size))
                print(
                    "[eval] done composition strict "
                    f"success={al.fmt(metrics.get('composition_success'))} "
                    f"part_acc={al.fmt(metrics.get('composition_part_acc'))} "
                    f"format={al.fmt(metrics.get('composition_format_acc'))}",
                    flush=True,
                )
            else:
                print("[eval] skip composition strict generation", flush=True)
        else:
            metrics.update(al.evaluate_task(model, tokenizer, task, cfg, do_generation=do_generation))
        print(
            f"[eval] done task={task.spec.name} "
            f"exact={al.fmt(metrics.get(f'{task.spec.name}_exact'))} "
            f"tok={al.fmt(metrics.get(f'{task.spec.name}_token_acc'))} "
            f"tf={al.fmt(metrics.get(f'{task.spec.name}_tf_token_acc'))} "
            f"loss={al.fmt(metrics.get(f'{task.spec.name}_tf_loss'))}",
            flush=True,
        )
    return metrics


def checkpoint_stage(model, tokenizer, out_dir: Path, stage: str, args: argparse.Namespace, metrics: Dict[str, Any]) -> None:
    al.checkpoint_model(
        model,
        tokenizer,
        out_dir,
        stage,
        args.checkpoint_policy,
        {"stage": stage, "metrics": metrics, "seed": args.seed},
    )


def scheduled_skills(skills: Sequence[al.TaskData], args: argparse.Namespace) -> Tuple[List[al.TaskData], List[Tuple[int, al.TaskData]]]:
    start = max(1, int(args.start_skill_index))
    if start > len(skills) + 1:
        raise ValueError(f"--start-skill-index={start} exceeds available skills={len(skills)}")
    learned = list(skills[: start - 1])
    scheduled = list(enumerate(skills[start - 1 :], start=start))
    return learned, scheduled


def maybe_log_base(
    *,
    model,
    tokenizer,
    skills: Sequence[al.TaskData],
    benchmarks: Sequence[al.TaskData],
    cfg: qp.RuntimeConfig,
    args: argparse.Namespace,
    logger: al.ArtifactLogger,
    out_dir: Path,
) -> None:
    if args.skip_base_eval:
        print("[base] skipped by --skip-base-eval", flush=True)
        return
    base_tasks = list(skills if args.eval_all_skills_at_base else []) + list(benchmarks)
    base_metrics = al.evaluate_suite(model, tokenizer, base_tasks, cfg, do_generation=args.generation_eval, include_wikitext=True)
    al.print_metrics("base", base_metrics, [task.spec.name for task in base_tasks])
    logger.log_stage_summary("base", base_metrics, old_task_examples=0, proxy_batches=0)
    checkpoint_stage(model, tokenizer, out_dir, "base", args, base_metrics)


def run_naive_branch(
    *,
    model,
    tokenizer,
    skills: Sequence[al.TaskData],
    benchmarks: Sequence[al.TaskData],
    composition: Optional[al.TaskData],
    cfg: qp.RuntimeConfig,
    args: argparse.Namespace,
    logger: al.ArtifactLogger,
    out_dir: Path,
) -> None:
    learned, schedule = scheduled_skills(skills, args)
    maybe_log_base(
        model=model,
        tokenizer=tokenizer,
        skills=skills,
        benchmarks=benchmarks,
        cfg=cfg,
        args=args,
        logger=logger,
        out_dir=out_dir,
    )
    for idx, task in schedule:
        learned.append(task)
        stage = f"naive_T{idx}_{task.spec.name}"
        metrics = al.consolidate_naive(
            student=model,
            tokenizer=tokenizer,
            task=task,
            active_eval_tasks=active_eval_tasks(learned, benchmarks),
            cfg=cfg,
            logger=logger,
            stage=stage,
            steps=args.skill_steps,
            lr=args.naive_lr,
        )
        checkpoint_stage(model, tokenizer, out_dir, stage, args, metrics)
    if args.composition_steps > 0 and composition is not None:
        run_composition_self_distill(
            model=model,
            tokenizer=tokenizer,
            composition=composition,
            learned=learned,
            benchmarks=benchmarks,
            cfg=cfg,
            args=args,
            logger=logger,
            out_dir=out_dir,
        )
    final_metrics = evaluate_and_log(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        logger=logger,
        stage="final_eval",
        learned=learned,
        benchmarks=benchmarks,
        composition=composition,
        include_composition=args.final_composition_eval,
        generation=args.generation_eval,
    )
    checkpoint_stage(model, tokenizer, out_dir, "final", args, final_metrics)


def train_task_teacher(
    *,
    base_model,
    tokenizer,
    task: al.TaskData,
    learned: Sequence[al.TaskData],
    protected_profiles: Sequence[Any],
    cfg: qp.RuntimeConfig,
    args: argparse.Namespace,
    logger: al.ArtifactLogger,
    stage: str,
) -> Tuple[Any, List[int], Dict[str, float]]:
    teacher = qp._clone_model(base_model, cfg.device)
    layers, _ = al.select_layers_generic(
        model=teacher,
        tokenizer=tokenizer,
        task=task,
        protected_profiles=protected_profiles,
        cfg=cfg,
        min_layers=args.min_layers,
        stage=stage,
        logger=logger,
    )
    metrics = al.train_adapter_teacher(
        model=teacher,
        tokenizer=tokenizer,
        task=task,
        active_eval_tasks=[task],
        cfg=cfg,
        logger=logger,
        stage=f"{stage}_teacher",
        selected_layers=layers,
        steps=args.teacher_steps,
        lr=args.teacher_lr,
        rank=args.teacher_rank,
        alpha=args.teacher_alpha,
        gate_init=args.teacher_gate_init,
        eval_interval=max(args.teacher_steps, 1),
    )
    return teacher, layers, metrics


def run_sdft_branch(
    *,
    model,
    tokenizer,
    skills: Sequence[al.TaskData],
    benchmarks: Sequence[al.TaskData],
    composition: Optional[al.TaskData],
    cfg: qp.RuntimeConfig,
    args: argparse.Namespace,
    logger: al.ArtifactLogger,
    out_dir: Path,
    aux_teacher_device: str,
) -> None:
    learned, schedule = scheduled_skills(skills, args)
    maybe_log_base(
        model=model,
        tokenizer=tokenizer,
        skills=skills,
        benchmarks=benchmarks,
        cfg=cfg,
        args=args,
        logger=logger,
        out_dir=out_dir,
    )
    for idx, task in schedule:
        learned.append(task)
        stage = f"sdft_T{idx}_{task.spec.name}"
        if args.sdft_mode == "on_policy_demo":
            teacher = qp._clone_model(model, aux_teacher_device)
            metrics = al.consolidate_on_policy_demo_sdft(
                student=model,
                ref_model=teacher,
                tokenizer=tokenizer,
                task=task,
                active_eval_tasks=active_eval_tasks(learned, benchmarks),
                cfg=cfg,
                logger=logger,
                stage=stage,
                steps=args.skill_steps,
                lr=args.sdft_lr,
                teacher_device=aux_teacher_device,
                do_sample=args.sdft_do_sample,
                temperature=args.sdft_temperature,
                top_p=args.sdft_top_p,
                ref_model_mixup_alpha=args.sdft_ref_model_mixup_alpha,
                ref_model_sync_steps=args.sdft_ref_model_sync_steps,
            )
            checkpoint_stage(model, tokenizer, out_dir, stage, args, metrics)
            al.release(teacher)
        else:
            teacher, _, _ = train_task_teacher(
                base_model=model,
                tokenizer=tokenizer,
                task=task,
                learned=learned,
                protected_profiles=[],
                cfg=cfg,
                args=args,
                logger=logger,
                stage=stage,
            )
            metrics = al.consolidate_sdft(
                student=model,
                teacher_new=teacher,
                tokenizer=tokenizer,
                task=task,
                active_eval_tasks=active_eval_tasks(learned, benchmarks),
                cfg=cfg,
                logger=logger,
                stage=stage,
                steps=args.skill_steps,
                lr=args.sdft_lr,
                teacher_device=aux_teacher_device,
                loss_type=args.sdft_loss,
                do_sample=args.sdft_do_sample,
                temperature=args.sdft_temperature,
                top_p=args.sdft_top_p,
            )
            checkpoint_stage(model, tokenizer, out_dir, stage, args, metrics)
            al.release(teacher)
    if args.composition_steps > 0 and composition is not None:
        run_composition_self_distill(
            model=model,
            tokenizer=tokenizer,
            composition=composition,
            learned=learned,
            benchmarks=benchmarks,
            cfg=cfg,
            args=args,
            logger=logger,
            out_dir=out_dir,
        )
    final_metrics = evaluate_and_log(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        logger=logger,
        stage="final_eval",
        learned=learned,
        benchmarks=benchmarks,
        composition=composition,
        include_composition=args.final_composition_eval,
        generation=args.generation_eval,
    )
    checkpoint_stage(model, tokenizer, out_dir, "final", args, final_metrics)


def run_amoeba_branch(
    *,
    model,
    tokenizer,
    skills: Sequence[al.TaskData],
    benchmarks: Sequence[al.TaskData],
    composition: Optional[al.TaskData],
    cfg: qp.RuntimeConfig,
    args: argparse.Namespace,
    logger: al.ArtifactLogger,
    out_dir: Path,
    aux_teacher_device: str,
) -> None:
    learned, schedule = scheduled_skills(skills, args)
    protected_profiles = collect_base_language_profile(model, tokenizer, cfg, args, logger)
    maybe_log_base(
        model=model,
        tokenizer=tokenizer,
        skills=skills,
        benchmarks=benchmarks,
        cfg=cfg,
        args=args,
        logger=logger,
        out_dir=out_dir,
    )
    for idx, task in schedule:
        old_teacher = qp._clone_model(model, aux_teacher_device)
        stage = f"amoeba_T{idx}_{task.spec.name}"
        teacher, layers, _ = train_task_teacher(
            base_model=model,
            tokenizer=tokenizer,
            task=task,
            learned=learned,
            protected_profiles=protected_profiles,
            cfg=cfg,
            args=args,
            logger=logger,
            stage=stage,
        )
        learned.append(task)
        metrics = al.consolidate_no_proxy(
            student=model,
            teacher_old=old_teacher,
            teacher_new=teacher,
            tokenizer=tokenizer,
            task=task,
            active_eval_tasks=active_eval_tasks(learned, benchmarks),
            selected_layers=layers,
            old_profiles=protected_profiles,
            cfg=cfg,
            logger=logger,
            stage=stage,
            steps=args.skill_steps,
            lr=args.amoeba_lr,
            old_kl_weight=args.no_proxy_old_kl_weight,
            old_hidden_weight=args.no_proxy_old_hidden_weight,
            new_kl_weight=args.new_kl_weight,
            new_hidden_weight=args.new_hidden_weight,
            project_gradients=args.gradient_projection,
            projection_strength=args.projection_strength,
            teacher_device=aux_teacher_device,
        )
        checkpoint_stage(model, tokenizer, out_dir, stage, args, metrics)
        protected_profiles.append(al.collect_profile(model, tokenizer, task, cfg, f"{task.spec.name}_after_T{idx}"))
        al.release(old_teacher, teacher)
    if args.composition_steps > 0 and composition is not None:
        run_composition_self_distill(
            model=model,
            tokenizer=tokenizer,
            composition=composition,
            learned=learned,
            benchmarks=benchmarks,
            cfg=cfg,
            args=args,
            logger=logger,
            out_dir=out_dir,
        )
    final_metrics = evaluate_and_log(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        logger=logger,
        stage="final_eval",
        learned=learned,
        benchmarks=benchmarks,
        composition=composition,
        include_composition=args.final_composition_eval,
        generation=args.generation_eval,
    )
    checkpoint_stage(model, tokenizer, out_dir, "final", args, final_metrics)


def run_composition_self_distill(
    *,
    model,
    tokenizer,
    composition: al.TaskData,
    learned: Sequence[al.TaskData],
    benchmarks: Sequence[al.TaskData],
    cfg: qp.RuntimeConfig,
    args: argparse.Namespace,
    logger: al.ArtifactLogger,
    out_dir: Path,
) -> None:
    subsection("LSP composition self-distillation")
    if args.composition_source == "manifest_train" and composition.train:
        train_rows = copy.deepcopy(composition.train)
        logger.log_event("composition_manifest_train_used", examples=len(train_rows))
    else:
        prompts = [row.prompt for row in composition.eval]
        completions: List[str] = []
        for start in range(0, len(prompts), cfg.eval_batch_size):
            completions.extend(
                al.generate_on_policy_completions(
                    model,
                    tokenizer,
                    prompts[start : start + cfg.eval_batch_size],
                    cfg.device,
                    max_new_tokens=composition.spec.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                )
            )
        train_rows: List[al.AlienExample] = []
        for prompt, completion in zip(prompts, completions):
            target = al.truncate_completion(completion)
            if target:
                train_rows.append(al.AlienExample(prompt=prompt, target=target, source=prompt, raw_target=target))
        if not train_rows:
            logger.log_event("composition_self_distill_skipped", reason="no nonempty self completions")
            return
    composition_train = copy.deepcopy(composition)
    composition_train.train = train_rows
    metrics = al.consolidate_naive(
        student=model,
        tokenizer=tokenizer,
        task=composition_train,
        active_eval_tasks=active_eval_tasks(learned, benchmarks, composition=composition, include_composition=True),
        cfg=cfg,
        logger=logger,
        stage="lsp_composition_self_distill",
        steps=args.composition_steps,
        lr=args.composition_lr,
    )
    checkpoint_stage(model, tokenizer, out_dir, "lsp_composition_self_distill", args, metrics)


def read_stage_summary(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def plot_comparison(out_dir: Path, skill_order: Sequence[str], branches: Sequence[str]) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    branch_labels = {
        "naive_sft": "SFT",
        "sdft_baseline": "TF-SDFT",
        "op_sdft": "OP-SDFT",
        "amoeba": "Amoeba",
    }
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    for branch in branches:
        rows = read_stage_summary(out_dir / branch / "stage_summary.csv")
        xs: List[int] = []
        ys: List[float] = []
        for row in rows:
            stage = row.get("stage", "")
            match = re.search(r"_T(\d+)_", stage)
            if not match:
                continue
            step = int(match.group(1))
            old_skill_names = list(skill_order[: max(step - 1, 0)])
            if not old_skill_names:
                continue
            values: List[float] = []
            for name in old_skill_names:
                token = safe_float(row.get(f"{name}_token_acc"))
                tf = safe_float(row.get(f"{name}_tf_token_acc"))
                values.append(max(v for v in (token, tf) if not math.isnan(v)) if not (math.isnan(token) and math.isnan(tf)) else float("nan"))
            values = [value for value in values if not math.isnan(value)]
            if values:
                xs.append(step)
                ys.append(float(sum(values) / len(values)))
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, label=branch_labels.get(branch, branch))
    ax.set_title("Old Skill Retention Across Sequential Skills")
    ax.set_xlabel("Skill step")
    ax.set_ylabel("Mean old-skill hybrid accuracy")
    ax.set_xticks(range(1, len(skill_order) + 1))
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = plot_dir / "old_skill_retention_comparison.png"
    fig.savefig(path)
    plt.close(fig)

    fig_time, ax_time = plt.subplots(figsize=(9, 5), dpi=180)
    for branch in branches:
        rows = read_stage_summary(out_dir / branch / "stage_summary.csv")
        xs_time: List[float] = []
        ys_time: List[float] = []
        elapsed = 0.0
        for row in rows:
            elapsed += max(0.0, safe_float(row.get("wall_time_sec")))
            stage = row.get("stage", "")
            match = re.search(r"_T(\d+)_", stage)
            if not match:
                continue
            step = int(match.group(1))
            old_skill_names = list(skill_order[: max(step - 1, 0)])
            eval_skill_names = old_skill_names or list(skill_order[:step])
            values: List[float] = []
            for name in eval_skill_names:
                token = safe_float(row.get(f"{name}_token_acc"))
                tf = safe_float(row.get(f"{name}_tf_token_acc"))
                values.append(max(v for v in (token, tf) if not math.isnan(v)) if not (math.isnan(token) and math.isnan(tf)) else float("nan"))
            values = [value for value in values if not math.isnan(value)]
            if values:
                xs_time.append(elapsed / 3600.0)
                ys_time.append(float(sum(values) / len(values)))
        if xs_time:
            ax_time.plot(xs_time, ys_time, marker="o", linewidth=2, label=branch_labels.get(branch, branch))
    ax_time.set_title("Retention vs Wall-Clock Cost")
    ax_time.set_xlabel("Cumulative stage wall time (hours)")
    ax_time.set_ylabel("Mean hybrid accuracy")
    ax_time.set_ylim(0, 1.02)
    ax_time.grid(alpha=0.25)
    ax_time.legend()
    fig_time.tight_layout()
    fig_time.savefig(plot_dir / "retention_vs_wall_time.png")
    plt.close(fig_time)

    comp_rows: List[Tuple[str, float, float]] = []
    for branch in branches:
        rows = read_stage_summary(out_dir / branch / "stage_summary.csv")
        final_rows = [row for row in rows if row.get("stage") == "final_eval"]
        row = final_rows[-1] if final_rows else (rows[-1] if rows else {})
        success = safe_float(row.get("composition_success"))
        part_acc = safe_float(row.get("composition_part_acc"))
        if not math.isnan(success) or not math.isnan(part_acc):
            comp_rows.append((branch, success, part_acc))
    if comp_rows:
        fig2, ax2 = plt.subplots(figsize=(8, 4.5), dpi=180)
        x = np.arange(len(comp_rows))
        width = 0.36
        success_values = [0.0 if math.isnan(row[1]) else row[1] for row in comp_rows]
        part_values = [0.0 if math.isnan(row[2]) else row[2] for row in comp_rows]
        ax2.bar(x - width / 2, success_values, width, label="both parts correct")
        ax2.bar(x + width / 2, part_values, width, label="part accuracy")
        ax2.set_title("Held-Out Skill Composition")
        ax2.set_ylabel("Accuracy")
        ax2.set_xticks(x)
        ax2.set_xticklabels([row[0] for row in comp_rows], rotation=15, ha="right")
        ax2.set_ylim(0, 1.02)
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(plot_dir / "composition_success_comparison.png")
        plt.close(fig2)
    return path


def resolve_manifest(args: argparse.Namespace) -> Path:
    path = Path(args.manifest_path).expanduser()
    if path.exists():
        return path
    if args.mode == "build_manifest" or args.build_manifest_if_missing:
        manifest = build_manifest(args)
        write_manifest(manifest, path)
        print(f"manifest_written={path}", flush=True)
        print(f"manifest_sha256={sha256_file(path)}", flush=True)
        return path
    raise FileNotFoundError(
        f"manifest not found: {path}. Run --mode build_manifest first, or pass --build-manifest-if-missing."
    )


def run_branch(branch: str, args: argparse.Namespace, manifest_path: Path, manifest: Dict[str, Any]) -> None:
    model_id = args.model_id or manifest.get("model_id") or DEFAULT_MODEL_ID
    args.model_id = model_id
    if branch == "op_sdft":
        args.sdft_mode = "on_policy_demo"
    run_root = Path(args.output_dir).expanduser() / f"qwen35_five_skill_seed{args.seed}"
    branch_dir = run_root / branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    logger = al.ArtifactLogger(branch_dir)
    logger.write_json("run_config.json", vars(args))
    logger.write_json("manifest_ref.json", {"path": str(manifest_path), "sha256": sha256_file(manifest_path)})
    cfg = make_runtime_config(args, branch_dir)
    aux_teacher_device = al.resolve_aux_device(args.teacher_device, args.device)

    section(f"QWEN3.5 FIVE-SKILL CL AUDIT :: {branch}")
    print(f"model_id={model_id}", flush=True)
    print(f"manifest={manifest_path}", flush=True)
    print(f"manifest_sha256={sha256_file(manifest_path)}", flush=True)
    print(f"device={args.device} teacher_device={aux_teacher_device} dtype={args.dtype} seed={args.seed}", flush=True)
    print(f"skill_order={', '.join(manifest.get('skill_order', []))}", flush=True)
    print("old_task_examples=0 proxy_batches=0 during branch updates", flush=True)

    tokenizer = load_tokenizer(model_id, local_files_only=args.local_files_only)
    model = load_causal_lm(model_id, device=args.device, dtype=choose_dtype(args.dtype), local_files_only=args.local_files_only)
    if args.gradient_checkpointing:
        qp._configure_gradient_checkpointing(model, True)
    wikitext_val = load_wikitext_eval(tokenizer, cfg, args)
    cfg._wikitext_val = wikitext_val
    cfg._composition_strict_eval = bool(args.composition_strict_eval)
    skills, composition, benchmarks = tasks_from_manifest(manifest)
    skills = skills[: args.max_skills]

    target_suffixes = tuple(part.strip() for part in str(args.target_suffixes).split(",") if part.strip())
    if target_suffixes:
        al.TARGET_SUFFIXES = target_suffixes
    print(f"target_suffixes={al.TARGET_SUFFIXES}", flush=True)
    al.evaluate_suite = evaluate_suite_progress

    if branch == "naive_sft":
        run_naive_branch(
            model=model,
            tokenizer=tokenizer,
            skills=skills,
            benchmarks=benchmarks,
            composition=composition,
            cfg=cfg,
            args=args,
            logger=logger,
            out_dir=branch_dir,
        )
    elif branch in {"sdft_baseline", "op_sdft"}:
        run_sdft_branch(
            model=model,
            tokenizer=tokenizer,
            skills=skills,
            benchmarks=benchmarks,
            composition=composition,
            cfg=cfg,
            args=args,
            logger=logger,
            out_dir=branch_dir,
            aux_teacher_device=aux_teacher_device,
        )
    elif branch == "amoeba":
        run_amoeba_branch(
            model=model,
            tokenizer=tokenizer,
            skills=skills,
            benchmarks=benchmarks,
            composition=composition,
            cfg=cfg,
            args=args,
            logger=logger,
            out_dir=branch_dir,
            aux_teacher_device=aux_teacher_device,
        )
    else:
        raise ValueError(f"unknown branch {branch}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3.5 0.8B five-skill OOD continual-learning audit")
    parser.add_argument("--mode", choices=("build_manifest", "run", "plot"), default="run")
    parser.add_argument("--branches", default="amoeba", help="Comma-separated branches: naive_sft,sdft_baseline,op_sdft,amoeba,all")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher-device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--manifest-path", default="outputs/qwen35_five_skill_manifest_seed1337.json")
    parser.add_argument("--build-manifest-if-missing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-synthetic-fallback", action="store_true")
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--streaming-buffer-size", type=int, default=10_000)

    parser.add_argument(
        "--task-profile",
        choices=tuple(TASK_PROFILES),
        default="five_ood",
        help="Preset task ladder. Use shen3 for Tool Use -> Science QA -> Medical QA.",
    )
    parser.add_argument("--skill-order", default="", help="Optional comma-separated override; otherwise uses --task-profile.")
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--benchmark-eval-samples", type=int, default=128)
    parser.add_argument("--benchmarks", default="mmlu_pro", help="Comma list, or none")
    parser.add_argument(
        "--build-composition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the old paired-answer composition eval in generated manifests. Use --no-build-composition for clean sequential-only ladders.",
    )
    parser.add_argument("--composition-eval-samples", type=int, default=64)
    parser.add_argument("--max-skills", type=int, default=5)
    parser.add_argument(
        "--start-skill-index",
        type=int,
        default=1,
        help="1-indexed first skill to train. Use with --model-id pointing at the previous final checkpoint for modular Kaggle resumes.",
    )
    parser.add_argument(
        "--skip-base-eval",
        action="store_true",
        help="Skip base eval/logging when resuming from a previous final checkpoint.",
    )

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--wikitext-eval-samples", type=int, default=64)
    parser.add_argument("--wikitext-eval-split", default="validation")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--generation-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-all-skills-at-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-composition-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--composition-strict-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run generated pairwise composition scoring even when generic generation eval is disabled.",
    )

    parser.add_argument("--skill-steps", type=int, default=400)
    parser.add_argument("--teacher-steps", type=int, default=400)
    parser.add_argument("--composition-steps", type=int, default=0)
    parser.add_argument("--naive-lr", type=float, default=1e-5)
    parser.add_argument("--sdft-lr", type=float, default=1e-5)
    parser.add_argument("--amoeba-lr", type=float, default=1e-5)
    parser.add_argument("--composition-lr", type=float, default=8e-6)
    parser.add_argument(
        "--composition-source",
        choices=("self", "manifest_train"),
        default="self",
        help="Source for the LSP composition phase. self uses generated completions; manifest_train uses held-out composition train rows from the manifest.",
    )
    parser.add_argument("--teacher-lr", type=float, default=8e-5)
    parser.add_argument("--teacher-rank", type=int, default=32)
    parser.add_argument("--teacher-alpha", type=float, default=64.0)
    parser.add_argument("--teacher-gate-init", type=float, default=-1.5)
    parser.add_argument("--min-layers", type=int, default=8)
    parser.add_argument("--target-suffixes", default="mlp.down_proj,mlp.up_proj")

    parser.add_argument("--sdft-loss", choices=("forward_kl", "ce"), default="forward_kl")
    parser.add_argument(
        "--sdft-mode",
        choices=("teacher_forced", "on_policy_demo"),
        default="teacher_forced",
        help="teacher_forced is the stable local distillation baseline; on_policy_demo follows the Shen et al. demo-conditioned on-policy mechanism.",
    )
    parser.add_argument("--sdft-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sdft-temperature", type=float, default=0.7)
    parser.add_argument("--sdft-top-p", type=float, default=0.95)
    parser.add_argument("--sdft-ref-model-mixup-alpha", type=float, default=0.01)
    parser.add_argument("--sdft-ref-model-sync-steps", type=int, default=1)

    parser.add_argument("--protect-base-language-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-language-profile-samples", type=int, default=64)
    parser.add_argument("--base-language-profile-split", default="train")
    parser.add_argument("--no-proxy-old-kl-weight", type=float, default=0.75)
    parser.add_argument("--no-proxy-old-hidden-weight", type=float, default=18.0)
    parser.add_argument("--new-kl-weight", type=float, default=1.0)
    parser.add_argument("--new-hidden-weight", type=float, default=0.5)
    parser.add_argument("--gradient-projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projection-strength", type=float, default=1.0)

    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--checkpoint-policy",
        choices=("none", "state", "pretrained", "final_state", "final_pretrained"),
        default="state",
        help="Use final_pretrained on Kaggle to save only checkpoints/final plus lightweight CSV/JSON artifacts.",
    )
    return parser


def run() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.mode == "build_manifest":
        manifest = build_manifest(args)
        path = write_manifest(manifest, Path(args.manifest_path).expanduser())
        print(f"manifest_written={path}", flush=True)
        print(f"manifest_sha256={sha256_file(path)}", flush=True)
        print(f"skills={', '.join(manifest['skill_order'])}", flush=True)
        return

    manifest_path = resolve_manifest(args)
    manifest = load_manifest(manifest_path)
    branches = parse_branches(args.branches)
    if args.mode == "plot":
        run_root = Path(args.output_dir).expanduser() / f"qwen35_five_skill_seed{args.seed}"
        path = plot_comparison(run_root, manifest.get("skill_order", []), branches)
        print(f"plot_written={path}" if path else "plot_skipped=matplotlib_unavailable", flush=True)
        return

    for branch in branches:
        run_branch(branch, args, manifest_path, manifest)
    if len(branches) > 1:
        run_root = Path(args.output_dir).expanduser() / f"qwen35_five_skill_seed{args.seed}"
        path = plot_comparison(run_root, manifest.get("skill_order", []), branches)
        if path:
            print(f"plot_written={path}", flush=True)


if __name__ == "__main__":
    run()
