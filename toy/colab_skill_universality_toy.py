#!/usr/bin/env python3
"""
Toy universality benchmark for skill-family locality.

Goal:
Test whether related anchor families provide a statistically cleaner advantage
than plain base or unrelated anchors across many toy skill families, using
curve metrics rather than one harsh endpoint.

For each candidate skill, the benchmark evaluates:
    - plain base
    - related anchor
    - every unrelated anchor

Metrics:
    - zero_shot_answer_acc
    - early_answer_acc
    - best_answer_acc
    - auc_answer_acc
    - final_answer_acc
    - learn_gain
    - pressure_corr_to_anchor (for anchored conditions)

Run:
    python colab_skill_universality_toy.py

Examples:
    python colab_skill_universality_toy.py --seeds 1337
    python colab_skill_universality_toy.py --seeds 1337,2027,31415

Smoke:
    CHAOS_SMOKE=1 python colab_skill_universality_toy.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import colab_skill_affinity_toy as toy
import colab_skill_anchor_affinity_toy as anchor_toy
import colab_water_weights_benchmark as ww


ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "skill_universality_toy_results.json"
COND_CSV_PATH = ROOT / "skill_universality_toy_conditions.csv"
COMP_CSV_PATH = ROOT / "skill_universality_toy_comparisons.csv"

FAMILY_SPECS = {
    "global_order": {
        "anchor": "reverse",
        "candidates": ["rotate_left", "rotate_right"],
    },
    "local_order": {
        "anchor": "swap_pairs",
        "candidates": ["block_reverse3", "block_rotate3"],
    },
    "uniform_substitution": {
        "anchor": "shift",
        "candidates": ["shift_twice", "reflect"],
    },
    "patterned_substitution": {
        "anchor": "alt_shift",
        "candidates": ["alt_shift_twice", "odd_shift"],
    },
}


@dataclass
class ConditionRow:
    seed: int
    family: str
    candidate: str
    condition_name: str
    relation: str
    pressure_cos_to_anchor: float
    pressure_corr_to_anchor: float
    pressure_jaccard_to_anchor: float
    zero_shot_answer_acc: float
    early_answer_acc: float
    best_answer_acc: float
    auc_answer_acc: float
    best_step: int
    final_answer_acc: float
    final_problem_acc: float
    final_seq_acc: float
    learn_gain: float
    text_loss_with_adapter: float
    text_acc_with_adapter: float


@dataclass
class ComparisonRow:
    seed: int
    family: str
    candidate: str
    related_minus_base_zero: float
    related_minus_base_early: float
    related_minus_base_best: float
    related_minus_base_auc: float
    related_minus_base_final: float
    related_minus_base_gain: float
    related_minus_unrelated_pressure_corr: float
    related_minus_unrelated_zero: float
    related_minus_unrelated_early: float
    related_minus_unrelated_best: float
    related_minus_unrelated_auc: float
    related_minus_unrelated_final: float
    related_minus_unrelated_gain: float


def parse_args() -> argparse.Namespace:
    smoke = os.environ.get("CHAOS_SMOKE", "0") == "1"
    parser = argparse.ArgumentParser(description="Toy universality benchmark for skill-family locality")
    parser.add_argument("--seeds", type=str, default="1337")
    parser.add_argument("--base-steps", type=int, default=12 if smoke else toy.DEFAULT_BASE_STEPS)
    parser.add_argument("--anchor-steps", type=int, default=18 if smoke else (140 if torch.cuda.is_available() else 100))
    parser.add_argument("--task-steps", type=int, default=18 if smoke else toy.DEFAULT_TASK_STEPS)
    parser.add_argument("--batch-size", type=int, default=4 if smoke else toy.DEFAULT_BATCH_SIZE)
    parser.add_argument("--source-len-train", type=int, default=6 if smoke else 9)
    parser.add_argument("--source-len-eval", type=int, default=7 if smoke else 12)
    parser.add_argument("--text-eval-batches", type=int, default=2 if smoke else toy.DEFAULT_TEXT_EVAL_BATCHES)
    parser.add_argument("--task-eval-batches", type=int, default=2 if smoke else toy.DEFAULT_TASK_EVAL_BATCHES)
    parser.add_argument("--probe-batches", type=int, default=2 if smoke else toy.DEFAULT_PROBE_BATCHES)
    parser.add_argument("--projector-batches", type=int, default=2 if smoke else toy.DEFAULT_PROJECTOR_BATCHES)
    parser.add_argument("--base-lr", type=float, default=ww.BASE_LR)
    parser.add_argument("--anchor-lr", type=float, default=2.5e-4)
    parser.add_argument("--adapter-lr", type=float, default=2e-3)
    parser.add_argument("--latent-strength", type=float, default=0.75)
    parser.add_argument("--anchor-text-period", type=int, default=4)
    parser.add_argument("--candidate-eval-period", type=int, default=6 if smoke else 24)
    parser.add_argument("--early-step", type=int, default=4 if smoke else 32)
    parser.add_argument("--frontier-k", type=int, default=4)
    parser.add_argument("--text-corpus-chars", type=int, default=4_000 if smoke else 80_000)
    parser.add_argument("--json-path", type=Path, default=JSON_PATH)
    parser.add_argument("--condition-csv-path", type=Path, default=COND_CSV_PATH)
    parser.add_argument("--comparison-csv-path", type=Path, default=COMP_CSV_PATH)
    return parser.parse_args()


def parse_seeds(spec: str) -> List[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def safe_mean(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return float("nan")
    return float(np.mean(clean))


def sign_test_p_value(num_trials: int, num_positive: int) -> float:
    if num_trials <= 0:
        return float("nan")
    tail = sum(math.comb(num_trials, wins) for wins in range(num_positive, num_trials + 1))
    return float(tail / (2 ** num_trials))


def family_verdict(
    *,
    mean_related_minus_base_auc: float,
    mean_related_minus_unrelated_auc: float,
    mean_related_minus_unrelated_final: float,
    final_unrelated_wins: int,
    total_cases: int,
) -> str:
    if total_cases <= 0:
        return "no_data"
    if (
        mean_related_minus_base_auc > 0.02
        and mean_related_minus_unrelated_auc > 0.01
        and mean_related_minus_unrelated_final > 0.01
        and final_unrelated_wins >= math.ceil(total_cases / 2)
    ):
        return "positive"
    if mean_related_minus_base_auc > 0.0 or mean_related_minus_unrelated_auc > 0.0 or mean_related_minus_unrelated_final > 0.0:
        return "mixed"
    return "weak"


def save_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_candidate_condition(
    *,
    seed: int,
    family: str,
    candidate: str,
    condition_name: str,
    relation: str,
    vocab_size: int,
    stoi: Dict[str, int],
    condition_state: Dict[str, torch.Tensor],
    condition_projectors: Dict[str, torch.Tensor],
    condition_pressure_profile: Dict[str, float] | None,
    condition_frontier: Sequence[str] | None,
    val_data: torch.Tensor,
    cfg: argparse.Namespace,
) -> ConditionRow:
    model = toy.restore_base_model(vocab_size, condition_state)
    eval_batches = toy.make_fixed_task_batches(
        stoi,
        candidate,
        seed + 70_000 + anchor_toy.stable_offset(condition_name) + anchor_toy.stable_offset(candidate),
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    probe_batches = toy.make_fixed_task_batches(
        stoi,
        candidate,
        seed + 80_000 + anchor_toy.stable_offset(condition_name) + anchor_toy.stable_offset(candidate),
        num_batches=cfg.probe_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    zero_metrics = toy.evaluate_task(model, eval_batches)
    candidate_pressure_profile = toy.normalize_profile(toy.mean_group_pressure_profile(model, probe_batches))
    candidate_frontier = toy.top_profile_keys(candidate_pressure_profile, cfg.frontier_k)

    if condition_pressure_profile is None or condition_frontier is None:
        pressure_cos = float("nan")
        pressure_corr = float("nan")
        pressure_jaccard = float("nan")
    else:
        pressure_cos = toy.cosine_similarity(candidate_pressure_profile, condition_pressure_profile)
        pressure_corr = toy.centered_similarity(candidate_pressure_profile, condition_pressure_profile)
        pressure_jaccard = toy.frontier_jaccard(candidate_frontier, condition_frontier)

    model.set_adapters_enabled(True)
    model.set_latent_free_projectors(condition_projectors, cfg.latent_strength)
    ww.set_requires_grad(ww.all_base_params(model), False)
    ww.set_requires_grad(ww.all_adapter_params(model), True)
    optimizer = torch.optim.AdamW(
        [param for param in ww.all_adapter_params(model) if param.requires_grad],
        lr=cfg.adapter_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )

    eval_trace: List[Tuple[int, float, float, float]] = [
        (
            0,
            float(zero_metrics["answer_acc"]),
            float(zero_metrics["problem_acc"]),
            float(zero_metrics["seq_acc"]),
        )
    ]
    for step in range(1, cfg.task_steps + 1):
        batch = toy.make_task_batch(
            stoi,
            candidate,
            seed + 90_000 + anchor_toy.stable_offset(condition_name) + anchor_toy.stable_offset(candidate),
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        toy.train_step(model, optimizer, batch)
        if step % cfg.candidate_eval_period == 0 or step == cfg.early_step or step == cfg.task_steps:
            metrics = toy.evaluate_task(model, eval_batches)
            eval_trace.append(
                (
                    step,
                    float(metrics["answer_acc"]),
                    float(metrics["problem_acc"]),
                    float(metrics["seq_acc"]),
                )
            )

    final_metrics = toy.evaluate_task(model, eval_batches)
    text_eval_positions = toy.make_text_positions(
        len(val_data),
        cfg.text_eval_batches,
        min(cfg.batch_size, ww.TEXT_EVAL_BATCH),
        seed + 100_000 + anchor_toy.stable_offset(condition_name) + anchor_toy.stable_offset(candidate),
    )
    text_metrics = ww.evaluate_text(model, val_data, text_eval_positions)

    answer_trace = [answer for _step, answer, _problem, _seq in eval_trace]
    best_step, best_answer, _best_problem, _best_seq = max(eval_trace, key=lambda row: row[1])
    early_candidates = [row for row in eval_trace if row[0] >= cfg.early_step]
    early_answer = early_candidates[0][1] if early_candidates else answer_trace[-1]
    auc_answer = float(np.mean(answer_trace))

    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ConditionRow(
        seed=seed,
        family=family,
        candidate=candidate,
        condition_name=condition_name,
        relation=relation,
        pressure_cos_to_anchor=float(pressure_cos),
        pressure_corr_to_anchor=float(pressure_corr),
        pressure_jaccard_to_anchor=float(pressure_jaccard),
        zero_shot_answer_acc=float(zero_metrics["answer_acc"]),
        early_answer_acc=float(early_answer),
        best_answer_acc=float(best_answer),
        auc_answer_acc=float(auc_answer),
        best_step=int(best_step),
        final_answer_acc=float(final_metrics["answer_acc"]),
        final_problem_acc=float(final_metrics["problem_acc"]),
        final_seq_acc=float(final_metrics["seq_acc"]),
        learn_gain=float(final_metrics["answer_acc"] - zero_metrics["answer_acc"]),
        text_loss_with_adapter=float(text_metrics["loss"]),
        text_acc_with_adapter=float(text_metrics["acc"]),
    )


def build_comparison_rows(rows: Sequence[ConditionRow]) -> List[ComparisonRow]:
    grouped: Dict[Tuple[int, str, str], List[ConditionRow]] = {}
    for row in rows:
        grouped.setdefault((row.seed, row.family, row.candidate), []).append(row)

    comparisons: List[ComparisonRow] = []
    for (seed, family, candidate), group in grouped.items():
        base_row = next(row for row in group if row.relation == "base")
        related_row = next(row for row in group if row.relation == "related")
        unrelated_rows = [row for row in group if row.relation == "unrelated"]

        def unrelated_mean(attr: str) -> float:
            return safe_mean([getattr(row, attr) for row in unrelated_rows])

        comparisons.append(
            ComparisonRow(
                seed=seed,
                family=family,
                candidate=candidate,
                related_minus_base_zero=related_row.zero_shot_answer_acc - base_row.zero_shot_answer_acc,
                related_minus_base_early=related_row.early_answer_acc - base_row.early_answer_acc,
                related_minus_base_best=related_row.best_answer_acc - base_row.best_answer_acc,
                related_minus_base_auc=related_row.auc_answer_acc - base_row.auc_answer_acc,
                related_minus_base_final=related_row.final_answer_acc - base_row.final_answer_acc,
                related_minus_base_gain=related_row.learn_gain - base_row.learn_gain,
                related_minus_unrelated_pressure_corr=related_row.pressure_corr_to_anchor - unrelated_mean("pressure_corr_to_anchor"),
                related_minus_unrelated_zero=related_row.zero_shot_answer_acc - unrelated_mean("zero_shot_answer_acc"),
                related_minus_unrelated_early=related_row.early_answer_acc - unrelated_mean("early_answer_acc"),
                related_minus_unrelated_best=related_row.best_answer_acc - unrelated_mean("best_answer_acc"),
                related_minus_unrelated_auc=related_row.auc_answer_acc - unrelated_mean("auc_answer_acc"),
                related_minus_unrelated_final=related_row.final_answer_acc - unrelated_mean("final_answer_acc"),
                related_minus_unrelated_gain=related_row.learn_gain - unrelated_mean("learn_gain"),
            )
        )
    return comparisons


def main() -> None:
    cfg = parse_args()
    seeds = parse_seeds(cfg.seeds)
    condition_rows: List[ConditionRow] = []
    comparison_rows: List[ComparisonRow] = []
    per_seed_meta: Dict[str, object] = {}

    print("=" * 78)
    print("TOY UNIVERSALITY BENCHMARK")
    print("=" * 78)
    print(
        f"device={ww.DEVICE} seeds={seeds} base_steps={cfg.base_steps} "
        f"anchor_steps={cfg.anchor_steps} task_steps={cfg.task_steps}"
    )
    print(f"families={list(FAMILY_SPECS.keys())}")

    for seed in seeds:
        toy.set_seed(seed)
        cfg.seed = seed
        joint_text = ww.EMBEDDED_FALLBACK_TEXT[: cfg.text_corpus_chars]
        stoi, _ = toy.build_joint_vocab(joint_text)
        train_data, val_data = toy.split_text_corpus(stoi, cfg.text_corpus_chars)
        vocab_size = len(stoi)

        print("-" * 78)
        print(f"[seed {seed}] preparing base")
        base_state, base_text_metrics, base_projectors = toy.pretrain_base_model(vocab_size, train_data, val_data, cfg)
        print(
            f"[seed {seed}] base text_loss={base_text_metrics['loss']:.4f} "
            f"text_acc={base_text_metrics['acc']:.3f}"
        )

        anchor_states: Dict[str, Dict[str, torch.Tensor]] = {}
        anchor_projectors: Dict[str, Dict[str, torch.Tensor]] = {}
        anchor_runs: Dict[str, anchor_toy.AnchorRun] = {}
        for family, spec in FAMILY_SPECS.items():
            anchor_state, projectors, anchor_run = anchor_toy.train_anchor_base(
                anchor_name=family,
                anchor_task=str(spec["anchor"]),
                vocab_size=vocab_size,
                stoi=stoi,
                base_state=base_state,
                train_data=train_data,
                val_data=val_data,
                cfg=cfg,
            )
            anchor_states[family] = anchor_state
            anchor_projectors[family] = projectors
            anchor_runs[family] = anchor_run
            print(
                f"[seed {seed}] [{family}] anchor={spec['anchor']} "
                f"eval_answer={anchor_run.eval_answer_acc:.3f} text_acc={anchor_run.text_acc:.3f}"
            )

        base_row_count_before = len(condition_rows)
        for family, spec in FAMILY_SPECS.items():
            related_anchor = family
            for candidate in spec["candidates"]:
                row = run_candidate_condition(
                    seed=seed,
                    family=family,
                    candidate=str(candidate),
                    condition_name="plain_base",
                    relation="base",
                    vocab_size=vocab_size,
                    stoi=stoi,
                    condition_state=base_state,
                    condition_projectors=base_projectors,
                    condition_pressure_profile=None,
                    condition_frontier=None,
                    val_data=val_data,
                    cfg=cfg,
                )
                condition_rows.append(row)

                related_row = run_candidate_condition(
                    seed=seed,
                    family=family,
                    candidate=str(candidate),
                    condition_name=related_anchor,
                    relation="related",
                    vocab_size=vocab_size,
                    stoi=stoi,
                    condition_state=anchor_states[related_anchor],
                    condition_projectors=anchor_projectors[related_anchor],
                    condition_pressure_profile=anchor_runs[related_anchor].pressure_profile,
                    condition_frontier=anchor_runs[related_anchor].pressure_frontier,
                    val_data=val_data,
                    cfg=cfg,
                )
                condition_rows.append(related_row)

                for other_family in FAMILY_SPECS.keys():
                    if other_family == related_anchor:
                        continue
                    unrelated_row = run_candidate_condition(
                        seed=seed,
                        family=family,
                        candidate=str(candidate),
                        condition_name=other_family,
                        relation="unrelated",
                        vocab_size=vocab_size,
                        stoi=stoi,
                        condition_state=anchor_states[other_family],
                        condition_projectors=anchor_projectors[other_family],
                        condition_pressure_profile=anchor_runs[other_family].pressure_profile,
                        condition_frontier=anchor_runs[other_family].pressure_frontier,
                        val_data=val_data,
                        cfg=cfg,
                    )
                    condition_rows.append(unrelated_row)

                print(
                    f"[seed {seed}] {family}/{candidate} "
                    f"base={row.final_answer_acc:.3f} related={related_row.final_answer_acc:.3f}"
                )

        seed_condition_rows = condition_rows[base_row_count_before:]
        seed_comparisons = build_comparison_rows(seed_condition_rows)
        comparison_rows.extend(seed_comparisons)
        per_seed_meta[str(seed)] = {
            "base_text_metrics": base_text_metrics,
            "anchors": {family: asdict(run) for family, run in anchor_runs.items()},
            "comparison_summary": {
                "mean_related_minus_base_early": safe_mean([row.related_minus_base_early for row in seed_comparisons]),
                "mean_related_minus_base_auc": safe_mean([row.related_minus_base_auc for row in seed_comparisons]),
                "mean_related_minus_base_final": safe_mean([row.related_minus_base_final for row in seed_comparisons]),
                "mean_related_minus_unrelated_early": safe_mean([row.related_minus_unrelated_early for row in seed_comparisons]),
                "mean_related_minus_unrelated_auc": safe_mean([row.related_minus_unrelated_auc for row in seed_comparisons]),
                "mean_related_minus_unrelated_final": safe_mean([row.related_minus_unrelated_final for row in seed_comparisons]),
            },
        }

    total_cases = len(comparison_rows)
    mean_related_minus_base_early = safe_mean([row.related_minus_base_early for row in comparison_rows])
    mean_related_minus_base_auc = safe_mean([row.related_minus_base_auc for row in comparison_rows])
    mean_related_minus_base_final = safe_mean([row.related_minus_base_final for row in comparison_rows])
    mean_related_minus_unrelated_pressure_corr = safe_mean([row.related_minus_unrelated_pressure_corr for row in comparison_rows])
    mean_related_minus_unrelated_early = safe_mean([row.related_minus_unrelated_early for row in comparison_rows])
    mean_related_minus_unrelated_auc = safe_mean([row.related_minus_unrelated_auc for row in comparison_rows])
    mean_related_minus_unrelated_final = safe_mean([row.related_minus_unrelated_final for row in comparison_rows])
    early_base_wins = sum(row.related_minus_base_early > 0.0 for row in comparison_rows)
    auc_base_wins = sum(row.related_minus_base_auc > 0.0 for row in comparison_rows)
    final_base_wins = sum(row.related_minus_base_final > 0.0 for row in comparison_rows)
    early_unrelated_wins = sum(row.related_minus_unrelated_early > 0.0 for row in comparison_rows)
    auc_unrelated_wins = sum(row.related_minus_unrelated_auc > 0.0 for row in comparison_rows)
    final_unrelated_wins = sum(row.related_minus_unrelated_final > 0.0 for row in comparison_rows)
    family_groups: Dict[str, List[ComparisonRow]] = defaultdict(list)
    for row in comparison_rows:
        family_groups[row.family].append(row)

    family_summary: Dict[str, Dict[str, object]] = {}
    for family, rows in family_groups.items():
        fam_cases = len(rows)
        fam_mean_related_minus_base_auc = safe_mean([row.related_minus_base_auc for row in rows])
        fam_mean_related_minus_base_final = safe_mean([row.related_minus_base_final for row in rows])
        fam_mean_related_minus_unrelated_auc = safe_mean([row.related_minus_unrelated_auc for row in rows])
        fam_mean_related_minus_unrelated_final = safe_mean([row.related_minus_unrelated_final for row in rows])
        fam_mean_related_minus_unrelated_pressure_corr = safe_mean([row.related_minus_unrelated_pressure_corr for row in rows])
        fam_auc_base_wins = sum(row.related_minus_base_auc > 0.0 for row in rows)
        fam_final_base_wins = sum(row.related_minus_base_final > 0.0 for row in rows)
        fam_auc_unrelated_wins = sum(row.related_minus_unrelated_auc > 0.0 for row in rows)
        fam_final_unrelated_wins = sum(row.related_minus_unrelated_final > 0.0 for row in rows)
        family_summary[family] = {
            "cases": fam_cases,
            "mean_related_minus_base_auc": fam_mean_related_minus_base_auc,
            "mean_related_minus_base_final": fam_mean_related_minus_base_final,
            "mean_related_minus_unrelated_pressure_corr": fam_mean_related_minus_unrelated_pressure_corr,
            "mean_related_minus_unrelated_auc": fam_mean_related_minus_unrelated_auc,
            "mean_related_minus_unrelated_final": fam_mean_related_minus_unrelated_final,
            "auc_base_wins": fam_auc_base_wins,
            "final_base_wins": fam_final_base_wins,
            "auc_unrelated_wins": fam_auc_unrelated_wins,
            "final_unrelated_wins": fam_final_unrelated_wins,
            "verdict": family_verdict(
                mean_related_minus_base_auc=fam_mean_related_minus_base_auc,
                mean_related_minus_unrelated_auc=fam_mean_related_minus_unrelated_auc,
                mean_related_minus_unrelated_final=fam_mean_related_minus_unrelated_final,
                final_unrelated_wins=fam_final_unrelated_wins,
                total_cases=fam_cases,
            ),
        }

    p_values = {
        "early_base": sign_test_p_value(total_cases, early_base_wins),
        "auc_base": sign_test_p_value(total_cases, auc_base_wins),
        "final_base": sign_test_p_value(total_cases, final_base_wins),
        "early_unrelated": sign_test_p_value(total_cases, early_unrelated_wins),
        "auc_unrelated": sign_test_p_value(total_cases, auc_unrelated_wins),
        "final_unrelated": sign_test_p_value(total_cases, final_unrelated_wins),
    }
    if mean_related_minus_unrelated_final > 0.01 and final_unrelated_wins >= math.ceil(total_cases / 2) and p_values["final_unrelated"] < 0.01:
        overall_verdict = "positive"
    elif mean_related_minus_base_auc > 0.02 and mean_related_minus_unrelated_auc > 0.0:
        overall_verdict = "mixed_positive"
    else:
        overall_verdict = "weak_or_inconclusive"

    print("=" * 78)
    print("AGGREGATE SUMMARY")
    print("=" * 78)
    print(f"cases={total_cases}")
    print(
        f"related-base early={mean_related_minus_base_early:+.3f} "
        f"auc={mean_related_minus_base_auc:+.3f} final={mean_related_minus_base_final:+.3f}"
    )
    print(
        f"related-unrelated pressure_corr={mean_related_minus_unrelated_pressure_corr:+.3f} "
        f"early={mean_related_minus_unrelated_early:+.3f} "
        f"auc={mean_related_minus_unrelated_auc:+.3f} "
        f"final={mean_related_minus_unrelated_final:+.3f}"
    )
    print(
        f"wins over base: early={early_base_wins}/{total_cases} "
        f"auc={auc_base_wins}/{total_cases} final={final_base_wins}/{total_cases}"
    )
    print(
        f"wins over unrelated mean: early={early_unrelated_wins}/{total_cases} "
        f"auc={auc_unrelated_wins}/{total_cases} final={final_unrelated_wins}/{total_cases}"
    )
    print(
        f"sign-test p-values: base(early={p_values['early_base']:.4g}, auc={p_values['auc_base']:.4g}, final={p_values['final_base']:.4g}) "
        f"unrelated(early={p_values['early_unrelated']:.4g}, auc={p_values['auc_unrelated']:.4g}, final={p_values['final_unrelated']:.4g})"
    )
    print("-" * 78)
    print("FAMILY VERDICTS")
    for family in FAMILY_SPECS.keys():
        summary = family_summary[family]
        print(
            f"{family:>22}  verdict={summary['verdict']:<8} "
            f"rb_auc={float(summary['mean_related_minus_base_auc']):+0.3f} "
            f"ru_auc={float(summary['mean_related_minus_unrelated_auc']):+0.3f} "
            f"ru_final={float(summary['mean_related_minus_unrelated_final']):+0.3f} "
            f"wins={int(summary['final_unrelated_wins'])}/{int(summary['cases'])}"
        )
    print("-" * 78)
    print(f"OVERALL VERDICT: {overall_verdict}")

    cfg.json_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.json_path.write_text(
        json.dumps(
            {
                "config": vars(cfg),
                "families": FAMILY_SPECS,
                "per_seed": per_seed_meta,
                "aggregate": {
                    "cases": total_cases,
                    "mean_related_minus_base_early": mean_related_minus_base_early,
                    "mean_related_minus_base_auc": mean_related_minus_base_auc,
                    "mean_related_minus_base_final": mean_related_minus_base_final,
                    "mean_related_minus_unrelated_pressure_corr": mean_related_minus_unrelated_pressure_corr,
                    "mean_related_minus_unrelated_early": mean_related_minus_unrelated_early,
                    "mean_related_minus_unrelated_auc": mean_related_minus_unrelated_auc,
                    "mean_related_minus_unrelated_final": mean_related_minus_unrelated_final,
                    "early_base_wins": early_base_wins,
                    "auc_base_wins": auc_base_wins,
                    "final_base_wins": final_base_wins,
                    "early_unrelated_wins": early_unrelated_wins,
                    "auc_unrelated_wins": auc_unrelated_wins,
                    "final_unrelated_wins": final_unrelated_wins,
                    "sign_test_p_values": p_values,
                    "overall_verdict": overall_verdict,
                },
                "family_summary": family_summary,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    save_csv(cfg.condition_csv_path, [asdict(row) for row in condition_rows])
    save_csv(cfg.comparison_csv_path, [asdict(row) for row in comparison_rows])
    print(f"saved: {cfg.json_path}")
    print(f"saved: {cfg.condition_csv_path}")
    print(f"saved: {cfg.comparison_csv_path}")


if __name__ == "__main__":
    main()
