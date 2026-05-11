#!/usr/bin/env python3
"""
Compositional expansion rescue lab.

Question:
When a fixed-size ABCD model hits a C-vs-D tradeoff frontier, can a one-block
propagation expansion rescue D without overwriting C?
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None

import colab_compositional_transfer_lab as comp
import colab_layer_expansion_lateral_lab as expand
import colab_layer_expansion_lateral_propagation_lab as prop
import colab_phase_reversibility_lab as phase
import colab_water_weights_benchmark as ww
import colab_water_weights_lateral_consolidation_benchmark as lateral


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "compositional_expansion_lab_results.csv"
PLOT_DIR = ROOT / "compositional_expansion_plots"
PARETO_PLOT_PATH = PLOT_DIR / "compositional_expansion_pareto.png"
LAB_SEEDS = list(comp.LAB_SEEDS)
EXPANDED_BRANCH = phase.ABC_CONSOLIDATION_BRANCH
EXPANDED_FRONTIER_MIN_BRACKET = 0.92
EXPANDED_FRONTIER_MIN_REVERSE = 0.20
EXPANDED_FRONTIER_REVERSE_MARGIN = 0.03
EXPANDED_FRONTIER_SORT_MARGIN = 0.02
EXPANDED_FRONTIER_BRACKET_SLACK = 0.03
EXPANDED_FRONTIER_REVERSE_SLACK = 0.03
EXPANDED_GUARD_REVERSE_SLACK = 0.02
EXPANDED_SAFE_REVERSE_SLACK = 0.00
EXPANDED_TRADEOFF_ABSOLUTE_MIN_REVERSE = 0.14
EXPANDED_TRADEOFF_BASE_REVERSE_DROP = 0.05
EXPANDED_TRADEOFF_DROP_PER_SORT_GAIN = 0.45
EXPANDED_TRADEOFF_MAX_REVERSE_DROP = 0.12
EXPANDED_TRADEOFF_MIN_SORT_GAIN = 0.04
EXPANDED_TRADEOFF_MAX_SORT_REGRESSION = 0.02
EXPANDED_TRADEOFF_TEXT_SLACK = 0.20
EXPANDED_BALANCED_REVERSE_SLACK = 0.03
EXPANDED_BALANCED_BRACKET_SLACK = 0.02
EXPANDED_BALANCED_TEXT_SLACK = 0.08
EXPANDED_BALANCED_SORT_SLACK = 0.04
EXPANDED_BALANCED_BRANCH_MIN_SORT = 0.30
EXPANDED_BALANCED_BRANCH_FRONTIER_SORT_SLACK = 0.05
EXPANDED_BALANCED_BRANCH_REVERSE_SLACK = 0.05
EXPANDED_BALANCED_BRANCH_TEXT_SLACK = 0.12
EXPANDED_BEST_D_MIN_BRACKET = 0.90
EXPANDED_BEST_D_BRACKET_SLACK = 0.08
EXPANDED_BEST_D_TEXT_SLACK = 0.20
EXPANDED_BEST_D_MIN_REVERSE = 0.12
EXPANDED_BALANCED_TEACHER_MIN_BRACKET = 0.35
EXPANDED_BALANCED_TEACHER_TEXT_SLACK = 0.20
EXPANDED_BALANCED_TEACHER_MIN_REVERSE = 0.12
EXPANDED_BALANCED_TEACHER_MIN_SORT_GAIN = 0.05
EXPANDED_BALANCED_TEACHER_FINAL_REVERSE_SLACK = 0.30
EXPANDED_BALANCED_TEACHER_FINAL_SORT_SLACK = 0.25
EXPANDED_BALANCED_TEACHER_TARGET_SORT_OFFSET = 0.25
EXPANDED_BALANCED_CONSOLIDATION_REVERSE_BOOST = 1.75
EXPANDED_BALANCED_CONSOLIDATION_AUX_SCALE = 1.80
EXPANDED_BALANCED_CONSOLIDATION_AUX_PERIOD = 1
EXPANDED_BALANCED_POLISH_STEPS = 120
EXPANDED_BALANCED_POLISH_SORT_WEIGHT_SCALE = 0.70
EXPANDED_BALANCED_POLISH_OLD_WEIGHT_SCALE = 1.40
EXPANDED_BALANCED_POLISH_REVERSE_BOOST = 1.80
EXPANDED_BALANCED_POLISH_AUX_SCALE = 2.00
EXPANDED_BALANCED_POLISH_OLD_PERIOD = 1
EXPANDED_BALANCED_POLISH_MAX_SORT_DROP = 0.08
EXPANDED_FINAL_BRACKET_SLACK = 0.05
EXPANDED_FINAL_TEXT_SLACK = 0.20
EXPANDED_FINAL_REVERSE_SLACK = 0.10
EXPANDED_TEACHER_MIN_BRACKET = 0.90
EXPANDED_TEACHER_TEXT_SLACK = 0.12
EXPANDED_TEACHER_ABSOLUTE_MIN_REVERSE = 0.10
EXPANDED_TEACHER_BASE_REVERSE_DROP = 0.08
EXPANDED_TEACHER_DROP_PER_SORT_GAIN = 0.45
EXPANDED_TEACHER_MAX_REVERSE_DROP = 0.24
EXPANDED_TEACHER_MIN_SORT_GAIN = 0.05
FIXED_MIXED_FRONTIER_MIN_BRACKET = 0.90
FIXED_MIXED_FRONTIER_MIN_SORT = 0.35
FIXED_MIXED_FRONTIER_MIN_REVERSE = 0.10
FIXED_MIXED_FALLBACK_MIN_BRACKET = 0.85
FIXED_MIXED_FALLBACK_MIN_SORT = 0.18
FIXED_MIXED_FALLBACK_MIN_REVERSE = 0.05
EXPANDED_BEST_D_MIN_SORT = 0.25
EXPANDED_BEST_D_POLISH_MAX_SORT_DROP = 0.15
EXPANDED_HEADLINE_COLLAPSE_SORT_MAX = 0.12
EXPANDED_HEADLINE_COLLAPSE_REVERSE_MIN = 0.98
EXPANDED_HEADLINE_MIXED_SORT_FLOOR = 0.25


@dataclass
class FixedCompositionalResult:
    base_a_metrics: Dict[str, float]
    base_ab: phase.StageResult
    reverse_ab: phase.StageResult
    base_abc: phase.StageResult
    base_abc_metrics: Dict[str, float]
    teacher_d_ab: phase.StageResult
    teacher_d_ab_metrics: Dict[str, float]
    base_abd: phase.StageResult
    base_abd_metrics: Dict[str, float]
    teacher_d_abc: phase.StageResult
    teacher_d_abc_metrics: Dict[str, float]
    base_abcd_consolidated: phase.StageResult
    fixed_mixed_frontier_metrics: Dict[str, float]
    base_abcd: phase.StageResult
    base_abcd_metrics: Dict[str, float]


@dataclass
class ExpandedStageSelection:
    selected: phase.StageResult
    best_d: phase.StageResult
    balanced: phase.StageResult | None


@dataclass
class ExpandedTeacherSelection:
    best_d: phase.StageResult
    balanced: phase.StageResult
    selected: phase.StageResult


def merged_metrics(*parts: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in parts:
        out.update(part)
    return out


def evaluate_external_checkpoint(
    checkpoint: Dict[str, object],
    vocab_size: int,
    tasks: Sequence[phase.TaskSpec],
) -> Dict[str, float]:
    model, optimizer, _forward = expand.restore_external_teacher(vocab_size, checkpoint)
    metrics = phase.evaluate_world(model, tasks)
    del model, optimizer, _forward
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def print_stage(label: str, metrics: Dict[str, float]) -> None:
    print(
        f"{label:32s} "
        f"{metrics.get('bracket_seq', float('nan')):11.3f} "
        f"{metrics.get('text_loss', float('nan')):10.3f} "
        f"{metrics.get('reverse_problem_acc', float('nan')):10.3f} "
        f"{metrics.get('sort_problem_acc', float('nan')):10.3f}"
    )


def make_expanded_frontier_reference(
    base_abc_metrics: Dict[str, float],
    fixed_abcd_metrics: Dict[str, float],
) -> Dict[str, float]:
    reference = dict(fixed_abcd_metrics)
    reference["bracket_seq"] = max(
        float(fixed_abcd_metrics.get("bracket_seq", 0.0)),
        EXPANDED_FRONTIER_MIN_BRACKET,
    )
    reference["text_loss"] = float(fixed_abcd_metrics.get("text_loss", base_abc_metrics.get("text_loss", float("inf"))))
    reference["reverse_problem_acc"] = max(
        float(fixed_abcd_metrics.get("reverse_problem_acc", 0.0)) + EXPANDED_FRONTIER_REVERSE_MARGIN,
        EXPANDED_FRONTIER_MIN_REVERSE,
    )
    return reference


def expanded_meets_fixed_frontier(
    metrics: Dict[str, float],
    fixed_abcd_metrics: Dict[str, float],
) -> bool:
    return (
        float(metrics.get("bracket_seq", 0.0))
        >= float(fixed_abcd_metrics.get("bracket_seq", 0.0)) - EXPANDED_FRONTIER_BRACKET_SLACK
        and float(metrics.get("reverse_problem_acc", 0.0))
        >= float(fixed_abcd_metrics.get("reverse_problem_acc", 0.0)) - EXPANDED_FRONTIER_REVERSE_SLACK
        and float(metrics.get("sort_problem_acc", metrics.get("arith_problem_acc", 0.0)))
        >= float(fixed_abcd_metrics.get("sort_problem_acc", fixed_abcd_metrics.get("arith_problem_acc", 0.0)))
        + EXPANDED_FRONTIER_SORT_MARGIN
    )


def sort_prob(metrics: Dict[str, float]) -> float:
    return float(metrics.get("sort_problem_acc", metrics.get("arith_problem_acc", 0.0)))


def reverse_prob(metrics: Dict[str, float]) -> float:
    return float(metrics.get("reverse_problem_acc", 0.0))


def bracket_seq(metrics: Dict[str, float]) -> float:
    return float(metrics.get("bracket_seq", 0.0))


def text_loss(metrics: Dict[str, float]) -> float:
    return float(metrics.get("text_loss", float("inf")))


def has_meaningful_sort(metrics: Dict[str, float], floor: float = FIXED_MIXED_FRONTIER_MIN_SORT) -> bool:
    return sort_prob(metrics) >= floor


def fixed_reverse_floor(fixed_abcd_metrics: Dict[str, float], slack: float) -> float:
    return max(
        EXPANDED_FRONTIER_MIN_REVERSE,
        float(fixed_abcd_metrics.get("reverse_problem_acc", 0.0)) - slack,
    )


def expanded_tradeoff_reverse_floor(
    metrics: Dict[str, float],
    fixed_abcd_metrics: Dict[str, float],
) -> float:
    sort_gain = max(0.0, sort_prob(metrics) - sort_prob(fixed_abcd_metrics))
    allowed_drop = min(
        EXPANDED_TRADEOFF_MAX_REVERSE_DROP,
        EXPANDED_TRADEOFF_BASE_REVERSE_DROP + EXPANDED_TRADEOFF_DROP_PER_SORT_GAIN * sort_gain,
    )
    return max(
        EXPANDED_TRADEOFF_ABSOLUTE_MIN_REVERSE,
        reverse_prob(fixed_abcd_metrics) - allowed_drop,
    )


def expanded_tradeoff_candidate_ok(
    metrics: Dict[str, float],
    retention_reference: Dict[str, float],
    fixed_abcd_metrics: Dict[str, float],
    keep_reverse: bool,
) -> bool:
    bracket_floor = max(
        EXPANDED_FRONTIER_MIN_BRACKET,
        bracket_seq(retention_reference) - EXPANDED_FRONTIER_BRACKET_SLACK,
    )
    text_ceiling = text_loss(retention_reference) + EXPANDED_TRADEOFF_TEXT_SLACK
    if bracket_seq(metrics) < bracket_floor:
        return False
    if text_loss(metrics) > text_ceiling:
        return False
    if sort_prob(metrics) < sort_prob(fixed_abcd_metrics) - EXPANDED_TRADEOFF_MAX_SORT_REGRESSION:
        return False
    if keep_reverse and reverse_prob(metrics) < expanded_tradeoff_reverse_floor(metrics, fixed_abcd_metrics):
        return False
    return True


def expanded_candidate_priority(
    metrics: Dict[str, float],
    retention_reference: Dict[str, float],
    fixed_abcd_metrics: Dict[str, float],
    keep_reverse: bool,
) -> tuple[float, float, float, float, float, float]:
    tradeoff_ok = expanded_tradeoff_candidate_ok(metrics, retention_reference, fixed_abcd_metrics, keep_reverse)
    sort_gain = sort_prob(metrics) - sort_prob(fixed_abcd_metrics)
    reverse_margin = reverse_prob(metrics) - expanded_tradeoff_reverse_floor(metrics, fixed_abcd_metrics)
    return (
        1.0 if tradeoff_ok else 0.0,
        sort_gain,
        reverse_margin,
        reverse_prob(metrics),
        bracket_seq(metrics),
        -text_loss(metrics),
    )


def expanded_balanced_candidate_ok(
    metrics: Dict[str, float],
    fixed_abcd_metrics: Dict[str, float],
) -> bool:
    if bracket_seq(metrics) < bracket_seq(fixed_abcd_metrics) - EXPANDED_BALANCED_BRACKET_SLACK:
        return False
    if text_loss(metrics) > text_loss(fixed_abcd_metrics) + EXPANDED_BALANCED_TEXT_SLACK:
        return False
    if reverse_prob(metrics) < reverse_prob(fixed_abcd_metrics) - EXPANDED_BALANCED_REVERSE_SLACK:
        return False
    if sort_prob(metrics) < sort_prob(fixed_abcd_metrics) - EXPANDED_BALANCED_SORT_SLACK:
        return False
    return True


def expanded_balanced_priority(metrics: Dict[str, float]) -> tuple[float, float, float, float]:
    return (
        reverse_prob(metrics),
        sort_prob(metrics),
        bracket_seq(metrics),
        -text_loss(metrics),
    )


def expanded_balanced_branch_sort_floor(
    fixed_frontier_metrics: Dict[str, float],
    fixed_final_metrics: Dict[str, float],
) -> float:
    return max(
        EXPANDED_BALANCED_BRANCH_MIN_SORT,
        sort_prob(fixed_frontier_metrics) - EXPANDED_BALANCED_BRANCH_FRONTIER_SORT_SLACK,
        sort_prob(fixed_final_metrics) - EXPANDED_BALANCED_TEACHER_FINAL_SORT_SLACK,
    )


def expanded_balanced_branch_reverse_floor(fixed_final_metrics: Dict[str, float]) -> float:
    return max(
        EXPANDED_BALANCED_TEACHER_MIN_REVERSE,
        reverse_prob(fixed_final_metrics) - EXPANDED_BALANCED_BRANCH_REVERSE_SLACK,
    )


def expanded_balanced_branch_candidate_ok(
    metrics: Dict[str, float],
    fixed_frontier_metrics: Dict[str, float],
    fixed_final_metrics: Dict[str, float],
) -> bool:
    if bracket_seq(metrics) < bracket_seq(fixed_final_metrics) - EXPANDED_BALANCED_BRACKET_SLACK:
        return False
    if text_loss(metrics) > text_loss(fixed_final_metrics) + EXPANDED_BALANCED_BRANCH_TEXT_SLACK:
        return False
    if reverse_prob(metrics) < expanded_balanced_branch_reverse_floor(fixed_final_metrics):
        return False
    if sort_prob(metrics) < expanded_balanced_branch_sort_floor(fixed_frontier_metrics, fixed_final_metrics):
        return False
    return True


def expanded_balanced_branch_priority(
    metrics: Dict[str, float],
    fixed_frontier_metrics: Dict[str, float],
    fixed_final_metrics: Dict[str, float],
) -> tuple[float, float, float, float, float]:
    return (
        1.0 if expanded_balanced_branch_candidate_ok(metrics, fixed_frontier_metrics, fixed_final_metrics) else 0.0,
        reverse_prob(metrics),
        sort_prob(metrics),
        bracket_seq(metrics),
        -text_loss(metrics),
    )


def expanded_teacher_reverse_floor(
    metrics: Dict[str, float],
    fixed_abcd_metrics: Dict[str, float],
) -> float:
    sort_gain = max(0.0, sort_prob(metrics) - sort_prob(fixed_abcd_metrics))
    allowed_drop = min(
        EXPANDED_TEACHER_MAX_REVERSE_DROP,
        EXPANDED_TEACHER_BASE_REVERSE_DROP + EXPANDED_TEACHER_DROP_PER_SORT_GAIN * sort_gain,
    )
    return max(
        EXPANDED_TEACHER_ABSOLUTE_MIN_REVERSE,
        reverse_prob(fixed_abcd_metrics) - allowed_drop,
    )


def expanded_teacher_candidate_ok(
    metrics: Dict[str, float],
    retention_reference: Dict[str, float],
    fixed_abcd_metrics: Dict[str, float],
) -> bool:
    if sort_prob(metrics) < sort_prob(fixed_abcd_metrics) + EXPANDED_TEACHER_MIN_SORT_GAIN:
        return False
    if bracket_seq(metrics) < max(EXPANDED_TEACHER_MIN_BRACKET, bracket_seq(retention_reference) - 0.08):
        return False
    if text_loss(metrics) > text_loss(retention_reference) + EXPANDED_TEACHER_TEXT_SLACK:
        return False
    if reverse_prob(metrics) < expanded_teacher_reverse_floor(metrics, fixed_abcd_metrics):
        return False
    return True


def expanded_teacher_priority(
    metrics: Dict[str, float],
    retention_reference: Dict[str, float],
    fixed_abcd_metrics: Dict[str, float],
) -> tuple[float, float, float, float, float]:
    return (
        sort_prob(metrics) - sort_prob(fixed_abcd_metrics),
        reverse_prob(metrics) - expanded_teacher_reverse_floor(metrics, fixed_abcd_metrics),
        reverse_prob(metrics),
        bracket_seq(metrics),
        -text_loss(metrics),
    )


def expanded_balanced_teacher_candidate_ok(
    metrics: Dict[str, float],
    retention_reference: Dict[str, float],
    fixed_frontier_metrics: Dict[str, float],
    fixed_final_metrics: Dict[str, float],
) -> bool:
    sort_floor = max(
        expanded_balanced_branch_sort_floor(fixed_frontier_metrics, fixed_final_metrics),
        sort_prob(fixed_frontier_metrics) + EXPANDED_BALANCED_TEACHER_MIN_SORT_GAIN,
    )
    if sort_prob(metrics) < sort_floor:
        return False
    if bracket_seq(metrics) < max(
        EXPANDED_BALANCED_TEACHER_MIN_BRACKET,
        bracket_seq(retention_reference) - 0.55,
    ):
        return False
    if text_loss(metrics) > text_loss(retention_reference) + EXPANDED_BALANCED_TEACHER_TEXT_SLACK:
        return False
    if reverse_prob(metrics) < max(
        EXPANDED_BALANCED_TEACHER_MIN_REVERSE,
        reverse_prob(fixed_final_metrics) - EXPANDED_BALANCED_TEACHER_FINAL_REVERSE_SLACK,
    ):
        return False
    return True


def expanded_balanced_teacher_priority(
    metrics: Dict[str, float],
    fixed_frontier_metrics: Dict[str, float],
    fixed_final_metrics: Dict[str, float],
) -> tuple[float, float, float, float, float]:
    target_sort = max(
        expanded_balanced_branch_sort_floor(fixed_frontier_metrics, fixed_final_metrics),
        min(sort_prob(fixed_final_metrics), sort_prob(fixed_frontier_metrics) + EXPANDED_BALANCED_TEACHER_TARGET_SORT_OFFSET),
    )
    return (
        reverse_prob(metrics),
        -abs(sort_prob(metrics) - target_sort),
        sort_prob(metrics),
        bracket_seq(metrics),
        -text_loss(metrics),
    )


def choose_expanded_teacher_stage(
    raw_stage: phase.StageResult,
    safe_stage: phase.StageResult,
    tradeoff_stage: phase.StageResult | None,
    balanced_stage: phase.StageResult | None,
    retention_reference: Dict[str, float],
    fixed_abcd_metrics: Dict[str, float],
) -> phase.StageResult:
    if tradeoff_stage is not None and has_meaningful_sort(tradeoff_stage.metrics):
        return tradeoff_stage
    if balanced_stage is not None and has_meaningful_sort(balanced_stage.metrics):
        return balanced_stage
    if expanded_teacher_candidate_ok(raw_stage.metrics, retention_reference, fixed_abcd_metrics) and has_meaningful_sort(
        raw_stage.metrics
    ):
        return raw_stage
    fallback = comp.choose_sort_teacher_stage(raw_stage, safe_stage, retention_reference)
    if has_meaningful_sort(fallback.metrics):
        return fallback
    if balanced_stage is not None:
        return balanced_stage
    if tradeoff_stage is not None:
        return tradeoff_stage
    return fallback


def choose_expanded_balanced_teacher_stage(
    raw_stage: phase.StageResult,
    safe_stage: phase.StageResult,
    tradeoff_stage: phase.StageResult | None,
    balanced_stage: phase.StageResult | None,
) -> phase.StageResult:
    mixed_sort_floor = EXPANDED_BALANCED_BRANCH_MIN_SORT
    if balanced_stage is not None and has_meaningful_sort(balanced_stage.metrics, mixed_sort_floor):
        return balanced_stage
    if tradeoff_stage is not None and has_meaningful_sort(tradeoff_stage.metrics, mixed_sort_floor):
        return tradeoff_stage
    fallback_candidates = [
        stage
        for stage in (safe_stage, raw_stage)
        if stage is not None and has_meaningful_sort(stage.metrics, mixed_sort_floor)
    ]
    if fallback_candidates:
        return choose_highest_reverse_stage(fallback_candidates)
    fallback = choose_highest_reverse_stage([safe_stage, raw_stage])
    if has_meaningful_sort(fallback.metrics, mixed_sort_floor):
        return fallback
    if balanced_stage is not None:
        return balanced_stage
    if tradeoff_stage is not None:
        return tradeoff_stage
    return fallback


def expanded_guard_candidate_ok(
    metrics: Dict[str, float],
    retention_reference: Dict[str, float],
    fixed_abcd_metrics: Dict[str, float],
    keep_reverse: bool,
) -> bool:
    if not comp.sort_transfer_guard_candidate_ok(metrics, retention_reference, keep_reverse):
        return False
    if keep_reverse and reverse_prob(metrics) < expanded_tradeoff_reverse_floor(metrics, fixed_abcd_metrics):
        return False
    return True


def expanded_safe_candidate_ok(
    metrics: Dict[str, float],
    retention_reference: Dict[str, float],
    fixed_abcd_metrics: Dict[str, float],
    keep_reverse: bool,
) -> bool:
    if not comp.sort_transfer_candidate_ok(metrics, retention_reference, keep_reverse):
        return False
    if keep_reverse and reverse_prob(metrics) < expanded_tradeoff_reverse_floor(metrics, fixed_abcd_metrics):
        return False
    return True


def expanded_polish_candidate_allowed(
    metrics: Dict[str, float],
    retention_reference: Dict[str, float],
    fixed_abcd_metrics: Dict[str, float],
    keep_reverse: bool,
) -> bool:
    if expanded_tradeoff_candidate_ok(metrics, retention_reference, fixed_abcd_metrics, keep_reverse):
        return True
    if sort_prob(metrics) > sort_prob(fixed_abcd_metrics) + EXPANDED_FRONTIER_SORT_MARGIN:
        return False
    if bracket_seq(metrics) < bracket_seq(retention_reference) - EXPANDED_FRONTIER_BRACKET_SLACK:
        return False
    if text_loss(metrics) > text_loss(retention_reference) + EXPANDED_TRADEOFF_TEXT_SLACK:
        return False
    return reverse_prob(metrics) >= reverse_prob(fixed_abcd_metrics)


def choose_highest_reverse_stage(stages: Sequence[phase.StageResult | None]) -> phase.StageResult:
    available = [stage for stage in stages if stage is not None]
    if not available:
        raise ValueError("no stages available")
    best = available[0]
    for stage in available[1:]:
        stage_metrics = stage.metrics
        best_metrics = best.metrics
        if reverse_prob(stage_metrics) > reverse_prob(best_metrics) + 1e-12:
            best = stage
            continue
        if reverse_prob(best_metrics) > reverse_prob(stage_metrics) + 1e-12:
            continue
        if float(stage_metrics.get("bracket_seq", 0.0)) > float(best_metrics.get("bracket_seq", 0.0)) + 1e-12:
            best = stage
            continue
        if float(best_metrics.get("bracket_seq", 0.0)) > float(stage_metrics.get("bracket_seq", 0.0)) + 1e-12:
            continue
        if float(stage_metrics.get("text_loss", float("inf"))) < float(best_metrics.get("text_loss", float("inf"))) - 1e-12:
            best = stage
    return best


def clone_stage_result(stage: phase.StageResult, label: str | None = None) -> phase.StageResult:
    return phase.StageResult(
        label=label or stage.label,
        checkpoint=stage.checkpoint,
        metrics=dict(stage.metrics),
        replay_count=stage.replay_count,
        replay_budget=stage.replay_budget,
        z_viscosity_steps=stage.z_viscosity_steps,
        latent_projection_steps=stage.latent_projection_steps,
        old_batch_count=stage.old_batch_count,
        old_batch_budget=stage.old_batch_budget,
        base_only_verified=stage.base_only_verified,
    )


def fixed_mixed_frontier_candidate_ok(metrics: Dict[str, float]) -> bool:
    return (
        bracket_seq(metrics) >= FIXED_MIXED_FRONTIER_MIN_BRACKET
        and sort_prob(metrics) >= FIXED_MIXED_FRONTIER_MIN_SORT
        and reverse_prob(metrics) >= FIXED_MIXED_FRONTIER_MIN_REVERSE
    )


def fixed_soft_mixed_frontier_candidate_ok(metrics: Dict[str, float]) -> bool:
    return (
        bracket_seq(metrics) >= FIXED_MIXED_FALLBACK_MIN_BRACKET
        and sort_prob(metrics) >= FIXED_MIXED_FALLBACK_MIN_SORT
        and reverse_prob(metrics) >= FIXED_MIXED_FALLBACK_MIN_REVERSE
    )


def mixed_frontier_priority(metrics: Dict[str, float]) -> tuple[float, float, float, float, float]:
    return (
        min(sort_prob(metrics), reverse_prob(metrics)),
        sort_prob(metrics),
        reverse_prob(metrics),
        bracket_seq(metrics),
        -text_loss(metrics),
    )


def choose_fixed_mixed_frontier_metrics(candidates: Sequence[Dict[str, float]]) -> Dict[str, float]:
    viable = [dict(metrics) for metrics in candidates if fixed_mixed_frontier_candidate_ok(metrics)]
    if not viable:
        soft_viable = [dict(metrics) for metrics in candidates if fixed_soft_mixed_frontier_candidate_ok(metrics)]
        if soft_viable:
            return max(soft_viable, key=mixed_frontier_priority)
        noncollapse = [
            dict(metrics)
            for metrics in candidates
            if sort_prob(metrics) > EXPANDED_HEADLINE_COLLAPSE_SORT_MAX
            and reverse_prob(metrics) >= FIXED_MIXED_FALLBACK_MIN_REVERSE
        ]
        if noncollapse:
            return max(noncollapse, key=mixed_frontier_priority)
        return max([dict(metrics) for metrics in candidates], key=mixed_frontier_priority)
    return max(viable, key=mixed_frontier_priority)


def collect_sort_transfer_candidate_metrics(stage: phase.StageResult) -> List[Dict[str, float]]:
    candidates: List[Dict[str, float]] = [dict(stage.metrics)]
    metadata = stage.checkpoint.get(comp.SORT_TRANSFER_CANDIDATE_METRICS_KEY)
    if not isinstance(metadata, dict):
        return candidates
    for key in ("selected", "raw", "guard", "safe"):
        metrics = metadata.get(key)
        if isinstance(metrics, dict):
            candidates.append(dict(metrics))
    return candidates


def expanded_best_d_candidate_ok(
    metrics: Dict[str, float],
    retention_reference: Dict[str, float],
    keep_reverse: bool,
) -> bool:
    bracket_floor = max(
        EXPANDED_BEST_D_MIN_BRACKET,
        bracket_seq(retention_reference) - EXPANDED_BEST_D_BRACKET_SLACK,
    )
    if bracket_seq(metrics) < bracket_floor:
        return False
    if text_loss(metrics) > text_loss(retention_reference) + EXPANDED_BEST_D_TEXT_SLACK:
        return False
    if sort_prob(metrics) < EXPANDED_BEST_D_MIN_SORT:
        return False
    if keep_reverse and reverse_prob(metrics) < EXPANDED_BEST_D_MIN_REVERSE:
        return False
    return True


def expanded_best_d_priority(metrics: Dict[str, float]) -> tuple[float, float, float, float]:
    return (
        sort_prob(metrics),
        reverse_prob(metrics),
        bracket_seq(metrics),
        -text_loss(metrics),
    )


def expanded_beats_fixed_final(
    metrics: Dict[str, float],
    fixed_final_metrics: Dict[str, float],
) -> bool:
    return (
        bracket_seq(metrics) >= bracket_seq(fixed_final_metrics) - EXPANDED_FINAL_BRACKET_SLACK
        and text_loss(metrics) <= text_loss(fixed_final_metrics) + EXPANDED_FINAL_TEXT_SLACK
        and reverse_prob(metrics) >= reverse_prob(fixed_final_metrics) - EXPANDED_FINAL_REVERSE_SLACK
        and sort_prob(metrics) >= sort_prob(fixed_final_metrics)
    )


def expanded_headline_priority(
    metrics: Dict[str, float],
    fixed_final_metrics: Dict[str, float],
) -> tuple[float, float, float, float, float, float]:
    sort_delta = sort_prob(metrics) - sort_prob(fixed_final_metrics)
    reverse_delta = reverse_prob(metrics) - reverse_prob(fixed_final_metrics)
    bracket_delta = bracket_seq(metrics) - bracket_seq(fixed_final_metrics)
    text_delta = text_loss(fixed_final_metrics) - text_loss(metrics)
    return (
        1.0 if expanded_beats_fixed_final(metrics, fixed_final_metrics) else 0.0,
        1.0 if expanded_balanced_candidate_ok(metrics, fixed_final_metrics) else 0.0,
        sort_delta,
        reverse_delta,
        bracket_delta,
        text_delta,
    )


def is_pure_c_collapse(metrics: Dict[str, float]) -> bool:
    return (
        reverse_prob(metrics) >= EXPANDED_HEADLINE_COLLAPSE_REVERSE_MIN
        and sort_prob(metrics) <= EXPANDED_HEADLINE_COLLAPSE_SORT_MAX
    )


def choose_expanded_headline_stage(
    fallback_stage: phase.StageResult,
    best_d_stage: phase.StageResult,
    balanced_stage: phase.StageResult | None,
    fixed_final_metrics: Dict[str, float],
) -> phase.StageResult:
    candidates = [fallback_stage, best_d_stage]
    if balanced_stage is not None:
        candidates.append(balanced_stage)
    return max(
        candidates,
        key=lambda stage: expanded_headline_priority(stage.metrics, fixed_final_metrics),
    )


def choose_expanded_best_d_branch_stage(
    fallback_stage: phase.StageResult,
    best_d_stage: phase.StageResult,
    retention_reference: Dict[str, float],
    keep_reverse: bool,
    preserved_stage: phase.StageResult | None = None,
) -> phase.StageResult:
    if (
        preserved_stage is not None
        and has_meaningful_sort(preserved_stage.metrics, EXPANDED_BEST_D_MIN_SORT)
        and (
            not has_meaningful_sort(best_d_stage.metrics, EXPANDED_BEST_D_MIN_SORT)
            or sort_prob(best_d_stage.metrics)
            < sort_prob(preserved_stage.metrics) - EXPANDED_BEST_D_POLISH_MAX_SORT_DROP
        )
        and (
            not has_meaningful_sort(fallback_stage.metrics, EXPANDED_BEST_D_MIN_SORT)
            or sort_prob(fallback_stage.metrics)
            < sort_prob(preserved_stage.metrics) - EXPANDED_BEST_D_POLISH_MAX_SORT_DROP
        )
    ):
        return preserved_stage
    if expanded_best_d_candidate_ok(best_d_stage.metrics, retention_reference, keep_reverse) and has_meaningful_sort(
        best_d_stage.metrics,
        EXPANDED_BEST_D_MIN_SORT,
    ):
        return best_d_stage
    if has_meaningful_sort(fallback_stage.metrics, EXPANDED_BEST_D_MIN_SORT):
        return fallback_stage
    if preserved_stage is not None and has_meaningful_sort(preserved_stage.metrics, EXPANDED_BEST_D_MIN_SORT):
        return preserved_stage
    if expanded_best_d_priority(best_d_stage.metrics) > expanded_best_d_priority(fallback_stage.metrics):
        return best_d_stage
    return fallback_stage


def choose_expanded_balanced_branch_stage(
    fallback_stage: phase.StageResult,
    balanced_stage: phase.StageResult | None,
    fixed_frontier_metrics: Dict[str, float],
    fixed_final_metrics: Dict[str, float],
    preserved_stage: phase.StageResult | None = None,
) -> phase.StageResult:
    candidates = [stage for stage in (preserved_stage, balanced_stage, fallback_stage) if stage is not None]
    valid = [
        stage
        for stage in candidates
        if expanded_balanced_branch_candidate_ok(stage.metrics, fixed_frontier_metrics, fixed_final_metrics)
    ]
    if valid:
        return max(
            valid,
            key=lambda stage: expanded_balanced_branch_priority(
                stage.metrics,
                fixed_frontier_metrics,
                fixed_final_metrics,
            ),
        )
    if preserved_stage is not None and has_meaningful_sort(
        preserved_stage.metrics,
        expanded_balanced_branch_sort_floor(fixed_frontier_metrics, fixed_final_metrics),
    ):
        return preserved_stage
    if balanced_stage is not None and has_meaningful_sort(
        balanced_stage.metrics,
        expanded_balanced_branch_sort_floor(fixed_frontier_metrics, fixed_final_metrics),
    ):
        return balanced_stage
    if has_meaningful_sort(
        fallback_stage.metrics,
        expanded_balanced_branch_sort_floor(fixed_frontier_metrics, fixed_final_metrics),
    ):
        return fallback_stage
    if balanced_stage is not None:
        return balanced_stage
    if preserved_stage is not None:
        return preserved_stage
    return fallback_stage


def choose_expanded_global_stage(
    candidates: Sequence[phase.StageResult | None],
    fixed_final_metrics: Dict[str, float],
) -> phase.StageResult:
    available = [stage for stage in candidates if stage is not None]
    if not available:
        raise ValueError("no expanded stages available")
    mixed_candidates = [
        stage
        for stage in available
        if not is_pure_c_collapse(stage.metrics)
        and has_meaningful_sort(stage.metrics, max(EXPANDED_HEADLINE_MIXED_SORT_FLOOR, sort_prob(fixed_final_metrics) + 0.05))
    ]
    if mixed_candidates:
        available = mixed_candidates
    return max(
        available,
        key=lambda stage: expanded_headline_priority(stage.metrics, fixed_final_metrics),
    )


def expanded_beats_fixed_frontier_both(
    metrics: Dict[str, float],
    fixed_frontier_metrics: Dict[str, float],
) -> bool:
    return (
        sort_prob(metrics) >= sort_prob(fixed_frontier_metrics)
        and reverse_prob(metrics) >= reverse_prob(fixed_frontier_metrics)
    )


def choose_expanded_branch_selection(
    best_d_selection: ExpandedStageSelection,
    balanced_selection: ExpandedStageSelection,
    fixed_final_metrics: Dict[str, float],
    fixed_frontier_metrics: Dict[str, float],
) -> phase.StageResult:
    balanced_candidates = [
        stage
        for stage in (balanced_selection.selected, balanced_selection.balanced)
        if stage is not None
        and expanded_beats_fixed_frontier_both(stage.metrics, fixed_frontier_metrics)
    ]
    if balanced_candidates:
        return max(
            balanced_candidates,
            key=lambda stage: expanded_balanced_branch_priority(
                stage.metrics,
                fixed_frontier_metrics,
                fixed_final_metrics,
            ),
        )
    return choose_expanded_global_stage(
        [
            best_d_selection.selected,
            best_d_selection.best_d,
            balanced_selection.selected,
            balanced_selection.balanced,
        ],
        fixed_final_metrics,
    )


def choose_expanded_sort_transfer_stage(
    raw_stage: phase.StageResult,
    guard_stage: phase.StageResult | None,
    safe_stage: phase.StageResult | None,
    balanced_stage: phase.StageResult | None,
    retention_reference: Dict[str, float],
    fixed_abcd_metrics: Dict[str, float],
    keep_reverse: bool,
) -> phase.StageResult:
    stages = [raw_stage]
    if guard_stage is not None:
        stages.append(guard_stage)
    if safe_stage is not None:
        stages.append(safe_stage)

    balanced_ok = (
        balanced_stage is not None
        and expanded_balanced_candidate_ok(balanced_stage.metrics, fixed_abcd_metrics)
    )

    tradeoff_stages = [
        stage
        for stage in stages
        if expanded_tradeoff_candidate_ok(stage.metrics, retention_reference, fixed_abcd_metrics, keep_reverse)
    ]
    if tradeoff_stages:
        best_tradeoff = max(
            tradeoff_stages,
            key=lambda stage: expanded_candidate_priority(
                stage.metrics,
                retention_reference,
                fixed_abcd_metrics,
                keep_reverse,
            ),
        )
        if sort_prob(best_tradeoff.metrics) >= sort_prob(fixed_abcd_metrics) + EXPANDED_TRADEOFF_MIN_SORT_GAIN:
            return best_tradeoff
        if guard_stage is None and safe_stage is None:
            return best_tradeoff

    filtered_raw = (
        raw_stage
        if expanded_polish_candidate_allowed(raw_stage.metrics, retention_reference, fixed_abcd_metrics, keep_reverse)
        else None
    )
    filtered_guard = (
        guard_stage
        if guard_stage is not None
        and expanded_polish_candidate_allowed(guard_stage.metrics, retention_reference, fixed_abcd_metrics, keep_reverse)
        else None
    )
    filtered_safe = (
        safe_stage
        if safe_stage is not None
        and expanded_polish_candidate_allowed(safe_stage.metrics, retention_reference, fixed_abcd_metrics, keep_reverse)
        else None
    )
    if filtered_raw is None and filtered_guard is None and filtered_safe is None:
        fallback = choose_highest_reverse_stage([balanced_stage, safe_stage, guard_stage, raw_stage])
        return fallback
    if filtered_raw is None:
        chosen = choose_highest_reverse_stage([balanced_stage, filtered_safe, filtered_guard])
        return chosen
    chosen = comp.choose_sort_transfer_stage(
        filtered_raw,
        filtered_guard,
        filtered_safe,
        retention_reference,
        keep_reverse,
    )
    if balanced_ok:
        assert balanced_stage is not None
        if reverse_prob(chosen.metrics) < reverse_prob(fixed_abcd_metrics) and (
            sort_prob(balanced_stage.metrics) >= sort_prob(chosen.metrics) - EXPANDED_BALANCED_SORT_SLACK
        ):
            return balanced_stage
        if expanded_balanced_priority(balanced_stage.metrics) > expanded_balanced_priority(chosen.metrics) and (
            sort_prob(balanced_stage.metrics) >= sort_prob(chosen.metrics) - 1e-12
        ):
            return balanced_stage
    if tradeoff_stages:
        best_tradeoff = max(
            tradeoff_stages,
            key=lambda stage: expanded_candidate_priority(
                stage.metrics,
                retention_reference,
                fixed_abcd_metrics,
                keep_reverse,
            ),
        )
        if expanded_candidate_priority(
            best_tradeoff.metrics, retention_reference, fixed_abcd_metrics, keep_reverse
        ) > expanded_candidate_priority(
            chosen.metrics, retention_reference, fixed_abcd_metrics, keep_reverse
        ):
            return best_tradeoff
    return chosen


def train_propagation_sort_teacher_candidates(
    vocab_size: int,
    prop_checkpoint: Dict[str, object],
    old_checkpoint: Dict[str, object],
    bracket_task: phase.TaskSpec,
    text_task: phase.TaskSpec,
    reverse_task: phase.TaskSpec,
    sort_task: phase.TaskSpec,
    eval_tasks: Sequence[phase.TaskSpec],
    retention_reference: Dict[str, float],
    fixed_frontier_metrics: Dict[str, float],
    fixed_final_metrics: Dict[str, float],
) -> ExpandedTeacherSelection:
    model, _optimizer = prop.restore_prop_checkpoint(vocab_size, prop_checkpoint, load_optimizer=False)
    model.set_adapters_enabled(False)
    model.clear_latent_free_projectors()
    ww.set_requires_grad(prop.prop_all_base_params(model), False)
    ww.set_requires_grad(prop.prop_trainable_params(model), True)
    optimizer = prop.make_prop_optimizer(model, prop.PROP_NEW_BLOCK_LR, params=prop.prop_trainable_params(model))

    old_teacher, _old_teacher_opt = phase.restore_phase_checkpoint(vocab_size, old_checkpoint, load_optimizer=False)
    phase.set_model_base_only(old_teacher)
    ww.set_requires_grad(old_teacher.parameters(), False)

    prop.set_prop_probe_mode(model)
    final_metrics = phase.evaluate_world(model, eval_tasks)
    raw_best_metrics = dict(final_metrics)
    raw_best_checkpoint = prop.make_prop_checkpoint(model, optimizer)
    safe_best_metrics = dict(final_metrics)
    safe_best_checkpoint = raw_best_checkpoint
    tradeoff_best_metrics = (
        dict(final_metrics)
        if expanded_teacher_candidate_ok(final_metrics, retention_reference, fixed_frontier_metrics)
        else None
    )
    tradeoff_best_checkpoint = raw_best_checkpoint
    balanced_best_metrics = (
        dict(final_metrics)
        if expanded_balanced_teacher_candidate_ok(
            final_metrics,
            retention_reference,
            fixed_frontier_metrics,
            fixed_final_metrics,
        )
        else None
    )
    balanced_best_checkpoint = raw_best_checkpoint
    print(f"[prop_attach:D_from_ABC] lateral-propagation sort attach for {prop.PROP_ATTACH_STEPS} steps")

    for step in range(1, prop.PROP_ATTACH_STEPS + 1):
        batch = sort_task.sample_train_batch(step)
        compat_task = comp.pick_compositional_retention_task(
            bracket_task,
            text_task,
            reverse_task,
            step - 1,
            final_metrics,
            retention_reference,
        )
        compat_batch = compat_task.sample_anchor_batch(500_000 + step)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        sort_logits, _sort_loss = model(batch.x, batch.y)
        loss = phase.task_loss_from_logits(sort_task, sort_logits, batch)

        with torch.no_grad():
            teacher_logits, _teacher_loss = old_teacher(compat_batch.x, compat_batch.y)
        compat_logits, _compat_loss = model(compat_batch.x, compat_batch.y)
        compat_scale = prop.prop_compat_scale(step, prop.PROP_ATTACH_STEPS)
        compat_boost = comp.SORT_TEACHER_REVERSE_COMPAT_BOOST if compat_task is reverse_task else 1.0
        loss = (
            loss
            + (compat_scale * compat_boost) * phase.task_loss_from_logits(compat_task, compat_logits, compat_batch)
            + (compat_scale * 0.85 * compat_boost) * lateral.distill_kl(compat_logits, teacher_logits)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prop.prop_trainable_params(model), ww.GRAD_CLIP)
        optimizer.step()

        if ww.is_probe_step(step) or step == prop.PROP_ATTACH_STEPS:
            prop.set_prop_probe_mode(model)
            final_metrics = phase.evaluate_world(model, eval_tasks)
            if step % prop.PROP_ATTACH_LOG_INTERVAL == 0 or step == prop.PROP_ATTACH_STEPS:
                print(
                    f"[prop_attach:D_from_ABC] step={step:04d}/{prop.PROP_ATTACH_STEPS} "
                    f"{comp.summarize_sort_metrics(final_metrics)} "
                    f"old_gate={float(model.prop_block.old_gate().item()):.4f} "
                    f"consensus={float(model.prop_block.consensus_gate().item()):.4f}"
                )
            if comp.better_sort_teacher_raw_candidate(final_metrics, raw_best_metrics):
                raw_best_metrics = dict(final_metrics)
                raw_best_checkpoint = prop.make_prop_checkpoint(model, optimizer)
            if (
                comp.sort_teacher_candidate_ok(final_metrics, retention_reference)
                and comp.better_sort_transfer_candidate(final_metrics, safe_best_metrics, True, retention_reference)
            ):
                safe_best_metrics = dict(final_metrics)
                safe_best_checkpoint = prop.make_prop_checkpoint(model, optimizer)
            if expanded_teacher_candidate_ok(final_metrics, retention_reference, fixed_frontier_metrics):
                if tradeoff_best_metrics is None or expanded_teacher_priority(
                    final_metrics, retention_reference, fixed_frontier_metrics
                ) > expanded_teacher_priority(
                    tradeoff_best_metrics, retention_reference, fixed_frontier_metrics
                ):
                    tradeoff_best_metrics = dict(final_metrics)
                    tradeoff_best_checkpoint = prop.make_prop_checkpoint(model, optimizer)
            if expanded_balanced_teacher_candidate_ok(
                final_metrics,
                retention_reference,
                fixed_frontier_metrics,
                fixed_final_metrics,
            ):
                if balanced_best_metrics is None or expanded_balanced_teacher_priority(
                    final_metrics,
                    fixed_frontier_metrics,
                    fixed_final_metrics,
                ) > expanded_balanced_teacher_priority(
                    balanced_best_metrics,
                    fixed_frontier_metrics,
                    fixed_final_metrics,
                ):
                    balanced_best_metrics = dict(final_metrics)
                    balanced_best_checkpoint = prop.make_prop_checkpoint(model, optimizer)

    del model, _optimizer, optimizer, old_teacher, _old_teacher_opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_stage = phase.StageResult(
        label="expanded_teacher_D_from_ABC_raw",
        checkpoint=raw_best_checkpoint,
        metrics=raw_best_metrics,
        base_only_verified=True,
    )
    safe_stage = phase.StageResult(
        label="expanded_teacher_D_from_ABC_safe",
        checkpoint=safe_best_checkpoint,
        metrics=safe_best_metrics,
        base_only_verified=True,
    )
    tradeoff_stage = (
        phase.StageResult(
            label="expanded_best_D_teacher",
            checkpoint=tradeoff_best_checkpoint,
            metrics=tradeoff_best_metrics,
            base_only_verified=True,
        )
        if tradeoff_best_metrics is not None
        else None
    )
    balanced_stage = (
        phase.StageResult(
            label="expanded_balanced_teacher",
            checkpoint=balanced_best_checkpoint,
            metrics=balanced_best_metrics,
            base_only_verified=True,
        )
        if balanced_best_metrics is not None
        else None
    )
    best_d_stage = choose_expanded_teacher_stage(
        raw_stage,
        safe_stage,
        tradeoff_stage,
        balanced_stage,
        retention_reference,
        fixed_frontier_metrics,
    )
    selected_balanced_stage = choose_expanded_balanced_teacher_stage(
        raw_stage,
        safe_stage,
        tradeoff_stage,
        balanced_stage,
    )
    return ExpandedTeacherSelection(
        best_d=best_d_stage,
        balanced=selected_balanced_stage,
        selected=best_d_stage,
    )


def consolidate_expanded_sort_transfer(
    label: str,
    branch_name: str,
    student_checkpoint: Dict[str, object],
    old_teacher_checkpoint: Dict[str, object],
    new_teacher_checkpoint: Dict[str, object],
    bracket_task: phase.TaskSpec,
    text_task: phase.TaskSpec,
    reverse_task: phase.TaskSpec | None,
    sort_task: phase.TaskSpec,
    eval_tasks: Sequence[phase.TaskSpec],
    vocab_size: int,
    selection_reference: Dict[str, float],
    fixed_frontier_metrics: Dict[str, float],
    fixed_final_metrics: Dict[str, float],
    keep_reverse: bool,
    profile: str = "best_d",
) -> ExpandedStageSelection:
    new_teacher, _new_teacher_opt, new_teacher_forward = expand.restore_external_teacher(
        vocab_size, new_teacher_checkpoint
    )

    old_teacher, _old_teacher_opt = phase.restore_phase_checkpoint(vocab_size, old_teacher_checkpoint, load_optimizer=False)
    phase.set_model_base_only(old_teacher)
    ww.set_requires_grad(old_teacher.parameters(), False)

    student, _student_opt = expand.restore_layer_checkpoint(vocab_size, student_checkpoint, load_optimizer=False)
    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()
    ww.set_requires_grad(expand.model_all_adapter_params(student), False)
    ww.set_requires_grad(expand.model_all_base_params(student), True)
    optimizer = expand.make_layer_optimizer(
        student,
        lateral.CONSOLIDATION_LR,
        params=expand.model_all_base_params(student),
    )

    old_batch_budget = phase.consolidation_expected_old_batches(branch_name)
    old_batch_count = 0
    balanced_profile = profile == "balanced"
    expand.set_expanded_probe_mode(student)
    final_metrics = phase.evaluate_world(student, eval_tasks)
    raw_best_metrics = dict(final_metrics)
    raw_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
    guard_best_metrics = (
        dict(final_metrics)
        if expanded_guard_candidate_ok(final_metrics, selection_reference, fixed_frontier_metrics, keep_reverse)
        else None
    )
    guard_best_checkpoint = raw_best_checkpoint
    safe_best_metrics = (
        dict(final_metrics)
        if expanded_safe_candidate_ok(final_metrics, selection_reference, fixed_frontier_metrics, keep_reverse)
        else None
    )
    safe_best_checkpoint = raw_best_checkpoint
    balanced_best_metrics = (
        dict(final_metrics)
        if (
            expanded_balanced_branch_candidate_ok(final_metrics, fixed_frontier_metrics, fixed_final_metrics)
            if balanced_profile
            else expanded_balanced_candidate_ok(final_metrics, fixed_final_metrics)
        )
        else None
    )
    balanced_best_checkpoint = raw_best_checkpoint
    best_d_metrics = (
        dict(final_metrics)
        if expanded_best_d_candidate_ok(final_metrics, selection_reference, keep_reverse)
        else None
    )
    best_d_checkpoint = raw_best_checkpoint

    print(f"[consolidate:{label}] branch={branch_name} for {lateral.CONSOLIDATION_STEPS} steps")
    for step in range(1, lateral.CONSOLIDATION_STEPS + 1):
        student.train()
        student.set_adapters_enabled(False)
        student.clear_latent_free_projectors()
        old_step = phase.consolidation_old_batch_schedule(branch_name, step)
        if old_step:
            current_task = comp.pick_compositional_retention_task(
                bracket_task,
                text_task,
                reverse_task if keep_reverse else None,
                old_batch_count,
                final_metrics,
                selection_reference,
            )
            batch = current_task.sample_anchor_batch(old_batch_count)
            teacher_model = old_teacher
            teacher_forward = lateral.forward_with_block_outputs
            old_batch_count += 1
        else:
            current_task = sort_task
            batch_sampler = sort_task.sample_consolidation_batch or sort_task.sample_train_batch
            batch = batch_sampler(step)
            teacher_model = new_teacher
            teacher_forward = new_teacher_forward

        with torch.no_grad():
            teacher_logits, _teacher_loss, teacher_states = teacher_forward(teacher_model, batch, detach=True)

        optimizer.zero_grad(set_to_none=True)
        student_logits, _student_loss, student_states = expand.forward_with_block_outputs(student, batch, detach=False)
        task_loss = phase.task_loss_from_logits(current_task, student_logits, batch)
        task_weight, kl_weight, hidden_weight = phase.consolidation_weights_for_step(branch_name, old_step, step)
        if balanced_profile and keep_reverse and reverse_task is not None and current_task is reverse_task:
            task_weight *= EXPANDED_BALANCED_CONSOLIDATION_REVERSE_BOOST
            kl_weight *= EXPANDED_BALANCED_CONSOLIDATION_REVERSE_BOOST
            hidden_weight *= EXPANDED_BALANCED_CONSOLIDATION_REVERSE_BOOST
        loss = (
            task_weight * task_loss
            + kl_weight * lateral.distill_kl(student_logits, teacher_logits)
            + hidden_weight * lateral.hidden_lateral_loss(student_states, teacher_states)
        )
        light_retention = keep_reverse and comp.use_light_sort_transfer_retention(final_metrics, keep_reverse)
        aux_reverse_period = (
            comp.SORT_TRANSFER_BALANCED_AUX_REVERSE_PERIOD
            if light_retention
            else comp.SORT_TRANSFER_AUX_REVERSE_PERIOD
        )
        aux_reverse_scale = comp.SORT_TRANSFER_BALANCED_AUX_SCALE if light_retention else 1.0
        if balanced_profile:
            aux_reverse_period = EXPANDED_BALANCED_CONSOLIDATION_AUX_PERIOD
            aux_reverse_scale *= EXPANDED_BALANCED_CONSOLIDATION_AUX_SCALE
        if keep_reverse and reverse_task is not None and not old_step and step % aux_reverse_period == 0:
            reverse_batch = reverse_task.sample_anchor_batch(975_000 + step)
            with torch.no_grad():
                reverse_teacher_logits, _reverse_teacher_loss, reverse_teacher_states = lateral.forward_with_block_outputs(
                    old_teacher, reverse_batch, detach=True
                )
            reverse_student_logits, _reverse_student_loss, reverse_student_states = expand.forward_with_block_outputs(
                student, reverse_batch, detach=False
            )
            reverse_task_loss = phase.task_loss_from_logits(reverse_task, reverse_student_logits, reverse_batch)
            loss = (
                loss
                + (comp.SORT_TRANSFER_AUX_REVERSE_TASK_WEIGHT * aux_reverse_scale) * reverse_task_loss
                + (comp.SORT_TRANSFER_AUX_REVERSE_KL_WEIGHT * aux_reverse_scale)
                * lateral.distill_kl(reverse_student_logits, reverse_teacher_logits)
                + (comp.SORT_TRANSFER_AUX_REVERSE_HIDDEN_WEIGHT * aux_reverse_scale)
                * lateral.hidden_lateral_loss(reverse_student_states, reverse_teacher_states)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expand.model_all_base_params(student), ww.GRAD_CLIP)
        optimizer.step()

        if step % comp.SORT_TRANSFER_EVAL_INTERVAL == 0 or step == lateral.CONSOLIDATION_STEPS:
            expand.set_expanded_probe_mode(student)
            final_metrics = phase.evaluate_world(student, eval_tasks)
            if step % comp.SORT_TRANSFER_LOG_INTERVAL == 0 or step == lateral.CONSOLIDATION_STEPS:
                print(
                    f"[consolidate:{label}] step={step:04d}/{lateral.CONSOLIDATION_STEPS} "
                    f"{comp.summarize_sort_metrics(final_metrics)} "
                    f"old_batches={old_batch_count}/{old_batch_budget} "
                    f"viscosity={ww.INITIAL_Z_VISCOSITY:.3f}"
                )
            if comp.better_sort_transfer_candidate(final_metrics, raw_best_metrics, keep_reverse, selection_reference):
                raw_best_metrics = dict(final_metrics)
                raw_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
            if (
                expanded_guard_candidate_ok(final_metrics, selection_reference, fixed_frontier_metrics, keep_reverse)
                and comp.better_sort_transfer_candidate(final_metrics, guard_best_metrics, keep_reverse, selection_reference)
            ):
                guard_best_metrics = dict(final_metrics)
                guard_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
            if (
                expanded_safe_candidate_ok(final_metrics, selection_reference, fixed_frontier_metrics, keep_reverse)
                and comp.better_sort_transfer_candidate(final_metrics, safe_best_metrics, keep_reverse, selection_reference)
            ):
                safe_best_metrics = dict(final_metrics)
                safe_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
            if (
                expanded_balanced_branch_candidate_ok(final_metrics, fixed_frontier_metrics, fixed_final_metrics)
                if balanced_profile
                else expanded_balanced_candidate_ok(final_metrics, fixed_final_metrics)
            ):
                if balanced_best_metrics is None or (
                    expanded_balanced_branch_priority(final_metrics, fixed_frontier_metrics, fixed_final_metrics)
                    if balanced_profile
                    else expanded_balanced_priority(final_metrics)
                ) > (
                    expanded_balanced_branch_priority(balanced_best_metrics, fixed_frontier_metrics, fixed_final_metrics)
                    if balanced_profile
                    else expanded_balanced_priority(balanced_best_metrics)
                ):
                    balanced_best_metrics = dict(final_metrics)
                    balanced_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
            if expanded_best_d_candidate_ok(final_metrics, selection_reference, keep_reverse):
                if best_d_metrics is None or expanded_best_d_priority(final_metrics) > expanded_best_d_priority(best_d_metrics):
                    best_d_metrics = dict(final_metrics)
                    best_d_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))

    expand.set_expanded_probe_mode(student)
    final_metrics = phase.evaluate_world(student, eval_tasks)
    if comp.better_sort_transfer_candidate(final_metrics, raw_best_metrics, keep_reverse, selection_reference):
        raw_best_metrics = dict(final_metrics)
        raw_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
    if (
        expanded_guard_candidate_ok(final_metrics, selection_reference, fixed_frontier_metrics, keep_reverse)
        and comp.better_sort_transfer_candidate(final_metrics, guard_best_metrics, keep_reverse, selection_reference)
    ):
        guard_best_metrics = dict(final_metrics)
        guard_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
    if (
        expanded_safe_candidate_ok(final_metrics, selection_reference, fixed_frontier_metrics, keep_reverse)
        and comp.better_sort_transfer_candidate(final_metrics, safe_best_metrics, keep_reverse, selection_reference)
    ):
        safe_best_metrics = dict(final_metrics)
        safe_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
    if (
        expanded_balanced_branch_candidate_ok(final_metrics, fixed_frontier_metrics, fixed_final_metrics)
        if balanced_profile
        else expanded_balanced_candidate_ok(final_metrics, fixed_final_metrics)
    ):
        if balanced_best_metrics is None or (
            expanded_balanced_branch_priority(final_metrics, fixed_frontier_metrics, fixed_final_metrics)
            if balanced_profile
            else expanded_balanced_priority(final_metrics)
        ) > (
            expanded_balanced_branch_priority(balanced_best_metrics, fixed_frontier_metrics, fixed_final_metrics)
            if balanced_profile
            else expanded_balanced_priority(balanced_best_metrics)
        ):
            balanced_best_metrics = dict(final_metrics)
            balanced_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
    if expanded_best_d_candidate_ok(final_metrics, selection_reference, keep_reverse):
        if best_d_metrics is None or expanded_best_d_priority(final_metrics) > expanded_best_d_priority(best_d_metrics):
            best_d_metrics = dict(final_metrics)
            best_d_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))

    del (
        new_teacher,
        _new_teacher_opt,
        new_teacher_forward,
        old_teacher,
        _old_teacher_opt,
        student,
        _student_opt,
        optimizer,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_stage = phase.StageResult(
        label=label,
        checkpoint=raw_best_checkpoint,
        metrics=raw_best_metrics,
        old_batch_count=old_batch_count,
        old_batch_budget=old_batch_budget,
        base_only_verified=True,
    )
    guard_stage = (
        phase.StageResult(
            label=label,
            checkpoint=guard_best_checkpoint,
            metrics=guard_best_metrics,
            old_batch_count=old_batch_count,
            old_batch_budget=old_batch_budget,
            base_only_verified=True,
        )
        if guard_best_metrics is not None
        else None
    )
    safe_stage = (
        phase.StageResult(
            label=label,
            checkpoint=safe_best_checkpoint,
            metrics=safe_best_metrics,
            old_batch_count=old_batch_count,
            old_batch_budget=old_batch_budget,
            base_only_verified=True,
        )
        if safe_best_metrics is not None
        else None
    )
    balanced_stage = (
        phase.StageResult(
            label=label,
            checkpoint=balanced_best_checkpoint,
            metrics=balanced_best_metrics,
            old_batch_count=old_batch_count,
            old_batch_budget=old_batch_budget,
            base_only_verified=True,
        )
        if balanced_best_metrics is not None
        else None
    )
    best_d_stage = phase.StageResult(
        label=label,
        checkpoint=best_d_checkpoint,
        metrics=best_d_metrics if best_d_metrics is not None else raw_best_metrics,
        old_batch_count=old_batch_count,
        old_batch_budget=old_batch_budget,
        base_only_verified=True,
    )
    fallback_stage = choose_expanded_sort_transfer_stage(
        raw_stage,
        guard_stage,
        safe_stage,
        None,
        selection_reference,
        fixed_frontier_metrics,
        keep_reverse,
    )
    if balanced_profile:
        selected_stage = choose_expanded_balanced_branch_stage(
            fallback_stage,
            balanced_stage,
            fixed_frontier_metrics,
            fixed_final_metrics,
        )
    else:
        selected_stage = choose_expanded_best_d_branch_stage(
            fallback_stage,
            best_d_stage,
            selection_reference,
            keep_reverse,
        )
    return ExpandedStageSelection(selected=selected_stage, best_d=best_d_stage, balanced=balanced_stage)


def polish_expanded_sort_transfer(
    label: str,
    initial_selection: ExpandedStageSelection,
    old_teacher_checkpoint: Dict[str, object],
    vocab_size: int,
    bracket_task: phase.TaskSpec,
    text_task: phase.TaskSpec,
    reverse_task: phase.TaskSpec | None,
    sort_task: phase.TaskSpec,
    eval_tasks: Sequence[phase.TaskSpec],
    teacher_checkpoint: Dict[str, object],
    teacher_reference: Dict[str, float],
    retention_reference: Dict[str, float],
    fixed_frontier_metrics: Dict[str, float],
    fixed_final_metrics: Dict[str, float],
    keep_reverse: bool,
    profile: str = "best_d",
) -> ExpandedStageSelection:
    initial_stage = initial_selection.selected
    balanced_profile = profile == "balanced"
    preserved_best_d_stage = clone_stage_result(initial_selection.best_d, label=label)
    preserved_balanced_stage = (
        clone_stage_result(initial_selection.balanced, label=label)
        if balanced_profile and initial_selection.balanced is not None
        else (clone_stage_result(initial_stage, label=label) if balanced_profile else None)
    )
    if expanded_meets_fixed_frontier(initial_stage.metrics, fixed_frontier_metrics):
        return ExpandedStageSelection(
            selected=clone_stage_result(initial_stage, label=label),
            best_d=clone_stage_result(initial_selection.best_d, label=label),
            balanced=clone_stage_result(initial_selection.balanced, label=label) if initial_selection.balanced is not None else None,
        )

    if not comp.needs_sort_transfer_polish(
        initial_stage.metrics,
        teacher_reference,
        retention_reference,
        keep_reverse,
    ):
        return ExpandedStageSelection(
            selected=clone_stage_result(initial_stage, label=label),
            best_d=clone_stage_result(initial_selection.best_d, label=label),
            balanced=clone_stage_result(initial_selection.balanced, label=label) if initial_selection.balanced is not None else None,
        )

    new_teacher, _new_teacher_opt, new_teacher_forward = expand.restore_external_teacher(
        vocab_size, teacher_checkpoint
    )

    old_teacher, _old_teacher_opt = phase.restore_phase_checkpoint(vocab_size, old_teacher_checkpoint, load_optimizer=False)
    phase.set_model_base_only(old_teacher)
    ww.set_requires_grad(old_teacher.parameters(), False)

    student, _student_opt = expand.restore_layer_checkpoint(vocab_size, initial_stage.checkpoint, load_optimizer=False)
    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()
    ww.set_requires_grad(expand.model_all_adapter_params(student), False)
    ww.set_requires_grad(expand.model_all_base_params(student), True)
    optimizer = expand.make_layer_optimizer(
        student,
        lateral.CONSOLIDATION_LR * phase.ARITH_TRANSFER_POLISH_LR_SCALE,
        params=expand.model_all_base_params(student),
    )

    light_retention = keep_reverse and comp.use_light_sort_transfer_retention(initial_stage.metrics, keep_reverse)
    old_period = phase.ARITH_TRANSFER_POLISH_OLD_PERIOD
    if keep_reverse and not light_retention:
        old_period = phase.ARITH_TRANSFER_RETENTION_OLD_PERIOD
    elif phase.retention_is_weak(initial_stage.metrics, retention_reference):
        old_period = phase.ARITH_TRANSFER_RETENTION_OLD_PERIOD
    if balanced_profile:
        old_period = min(old_period, EXPANDED_BALANCED_POLISH_OLD_PERIOD)
    aux_reverse_period = (
        comp.SORT_TRANSFER_BALANCED_AUX_REVERSE_PERIOD
        if light_retention
        else comp.SORT_TRANSFER_AUX_REVERSE_PERIOD
    )
    aux_reverse_scale = comp.SORT_TRANSFER_BALANCED_AUX_SCALE if light_retention else 1.0
    if balanced_profile:
        aux_reverse_period = 1
        aux_reverse_scale *= EXPANDED_BALANCED_POLISH_AUX_SCALE
    polish_steps = EXPANDED_BALANCED_POLISH_STEPS if balanced_profile else comp.SORT_TRANSFER_POLISH_STEPS
    old_batch_budget = sum(
        1 for step in range(1, polish_steps + 1) if step % old_period == 0
    )
    old_batch_count = 0
    expand.set_expanded_probe_mode(student)
    final_metrics = phase.evaluate_world(student, eval_tasks)
    raw_best_metrics = dict(initial_stage.metrics)
    raw_best_checkpoint = initial_stage.checkpoint
    guard_best_metrics = (
        dict(initial_stage.metrics)
        if expanded_guard_candidate_ok(initial_stage.metrics, retention_reference, fixed_frontier_metrics, keep_reverse)
        else None
    )
    guard_best_checkpoint = initial_stage.checkpoint
    safe_best_metrics = (
        dict(initial_stage.metrics)
        if expanded_safe_candidate_ok(initial_stage.metrics, retention_reference, fixed_frontier_metrics, keep_reverse)
        else None
    )
    safe_best_checkpoint = initial_stage.checkpoint
    balanced_best_metrics = (
        dict(initial_selection.balanced.metrics)
        if initial_selection.balanced is not None
        else (
            dict(initial_stage.metrics)
            if (
                expanded_balanced_branch_candidate_ok(initial_stage.metrics, fixed_frontier_metrics, fixed_final_metrics)
                if balanced_profile
                else expanded_balanced_candidate_ok(initial_stage.metrics, fixed_final_metrics)
            )
            else None
        )
    )
    balanced_best_checkpoint = (
        initial_selection.balanced.checkpoint
        if initial_selection.balanced is not None
        else initial_stage.checkpoint
    )
    best_d_metrics = (
        dict(initial_selection.best_d.metrics)
        if expanded_best_d_candidate_ok(initial_selection.best_d.metrics, retention_reference, keep_reverse)
        else None
    )
    best_d_checkpoint = initial_selection.best_d.checkpoint

    print(f"[transfer:{label}] expanded base-only sort transfer polish for {polish_steps} steps")
    for step in range(1, polish_steps + 1):
        student.train()
        student.set_adapters_enabled(False)
        student.clear_latent_free_projectors()
        old_step = step % old_period == 0
        if old_step:
            current_task = comp.pick_compositional_retention_task(
                bracket_task,
                text_task,
                reverse_task if keep_reverse else None,
                old_batch_count,
                final_metrics,
                retention_reference,
            )
            batch = current_task.sample_anchor_batch(900_000 + old_batch_count)
            teacher_model = old_teacher
            teacher_forward = lateral.forward_with_block_outputs
            old_batch_count += 1
            boost = (
                phase.ARITH_TRANSFER_POLISH_RETENTION_BOOST
                if phase.retention_is_weak(final_metrics, retention_reference)
                else 1.0
            )
            if keep_reverse and current_task is reverse_task:
                boost = max(
                    boost,
                    comp.SORT_TRANSFER_BALANCED_REVERSE_BOOST if light_retention else 1.35,
                )
                if balanced_profile:
                    boost = max(boost, EXPANDED_BALANCED_POLISH_REVERSE_BOOST)
            task_weight = phase.ARITH_TRANSFER_POLISH_OLD_TASK_WEIGHT * boost
            kl_weight = phase.ARITH_TRANSFER_POLISH_OLD_KL_WEIGHT * boost
            hidden_weight = phase.ARITH_TRANSFER_POLISH_OLD_HIDDEN_WEIGHT * boost
            if balanced_profile:
                task_weight *= EXPANDED_BALANCED_POLISH_OLD_WEIGHT_SCALE
                kl_weight *= EXPANDED_BALANCED_POLISH_OLD_WEIGHT_SCALE
                hidden_weight *= EXPANDED_BALANCED_POLISH_OLD_WEIGHT_SCALE
        else:
            current_task = sort_task
            batch_sampler = sort_task.sample_consolidation_batch or sort_task.sample_train_batch
            batch = batch_sampler(900_000 + step)
            teacher_model = new_teacher
            teacher_forward = new_teacher_forward
            task_weight = phase.ARITH_TRANSFER_POLISH_TASK_WEIGHT
            kl_weight = phase.ARITH_TRANSFER_POLISH_KL_WEIGHT
            hidden_weight = phase.ARITH_TRANSFER_POLISH_HIDDEN_WEIGHT
            if balanced_profile:
                task_weight *= EXPANDED_BALANCED_POLISH_SORT_WEIGHT_SCALE
                kl_weight *= EXPANDED_BALANCED_POLISH_SORT_WEIGHT_SCALE
                hidden_weight *= EXPANDED_BALANCED_POLISH_SORT_WEIGHT_SCALE

        with torch.no_grad():
            teacher_logits, _teacher_loss, teacher_states = teacher_forward(teacher_model, batch, detach=True)
        optimizer.zero_grad(set_to_none=True)
        student_logits, _student_loss, student_states = expand.forward_with_block_outputs(student, batch, detach=False)
        task_loss = phase.task_loss_from_logits(current_task, student_logits, batch)
        loss = (
            task_weight * task_loss
            + kl_weight * lateral.distill_kl(student_logits, teacher_logits)
            + hidden_weight * lateral.hidden_lateral_loss(student_states, teacher_states)
        )
        if keep_reverse and reverse_task is not None and not old_step and step % aux_reverse_period == 0:
            reverse_batch = reverse_task.sample_anchor_batch(950_000 + step)
            with torch.no_grad():
                reverse_teacher_logits, _reverse_teacher_loss, reverse_teacher_states = lateral.forward_with_block_outputs(
                    old_teacher, reverse_batch, detach=True
                )
            reverse_student_logits, _reverse_student_loss, reverse_student_states = expand.forward_with_block_outputs(
                student, reverse_batch, detach=False
            )
            reverse_task_loss = phase.task_loss_from_logits(reverse_task, reverse_student_logits, reverse_batch)
            loss = (
                loss
                + (comp.SORT_TRANSFER_AUX_REVERSE_TASK_WEIGHT * aux_reverse_scale) * reverse_task_loss
                + (comp.SORT_TRANSFER_AUX_REVERSE_KL_WEIGHT * aux_reverse_scale)
                * lateral.distill_kl(reverse_student_logits, reverse_teacher_logits)
                + (comp.SORT_TRANSFER_AUX_REVERSE_HIDDEN_WEIGHT * aux_reverse_scale)
                * lateral.hidden_lateral_loss(reverse_student_states, reverse_teacher_states)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expand.model_all_base_params(student), ww.GRAD_CLIP)
        optimizer.step()

        if step % comp.SORT_TRANSFER_EVAL_INTERVAL == 0 or step == polish_steps:
            expand.set_expanded_probe_mode(student)
            final_metrics = phase.evaluate_world(student, eval_tasks)
            if step % comp.SORT_TRANSFER_LOG_INTERVAL == 0 or step == polish_steps:
                print(
                    f"[transfer:{label}] step={step:04d}/{polish_steps} "
                    f"{comp.summarize_sort_metrics(final_metrics)} "
                    f"old_batches={old_batch_count}/{old_batch_budget}"
                )
            if (
                expanded_polish_candidate_allowed(
                    final_metrics, retention_reference, fixed_frontier_metrics, keep_reverse
                )
                and comp.better_sort_transfer_candidate(final_metrics, raw_best_metrics, keep_reverse, retention_reference)
            ):
                raw_best_metrics = dict(final_metrics)
                raw_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
            if (
                expanded_guard_candidate_ok(final_metrics, retention_reference, fixed_frontier_metrics, keep_reverse)
                and expanded_polish_candidate_allowed(
                    final_metrics, retention_reference, fixed_frontier_metrics, keep_reverse
                )
                and comp.better_sort_transfer_candidate(final_metrics, guard_best_metrics, keep_reverse, retention_reference)
            ):
                guard_best_metrics = dict(final_metrics)
                guard_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
            if (
                expanded_safe_candidate_ok(final_metrics, retention_reference, fixed_frontier_metrics, keep_reverse)
                and expanded_polish_candidate_allowed(
                    final_metrics, retention_reference, fixed_frontier_metrics, keep_reverse
                )
                and comp.better_sort_transfer_candidate(final_metrics, safe_best_metrics, keep_reverse, retention_reference)
            ):
                safe_best_metrics = dict(final_metrics)
                safe_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
            if (
                expanded_balanced_branch_candidate_ok(final_metrics, fixed_frontier_metrics, fixed_final_metrics)
                if balanced_profile
                else expanded_balanced_candidate_ok(final_metrics, fixed_final_metrics)
            ):
                if balanced_best_metrics is None or (
                    expanded_balanced_branch_priority(final_metrics, fixed_frontier_metrics, fixed_final_metrics)
                    if balanced_profile
                    else expanded_balanced_priority(final_metrics)
                ) > (
                    expanded_balanced_branch_priority(balanced_best_metrics, fixed_frontier_metrics, fixed_final_metrics)
                    if balanced_profile
                    else expanded_balanced_priority(balanced_best_metrics)
                ):
                    balanced_best_metrics = dict(final_metrics)
                    balanced_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
            if expanded_best_d_candidate_ok(final_metrics, retention_reference, keep_reverse):
                if best_d_metrics is None or expanded_best_d_priority(final_metrics) > expanded_best_d_priority(best_d_metrics):
                    best_d_metrics = dict(final_metrics)
                    best_d_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))

    expand.set_expanded_probe_mode(student)
    final_metrics = phase.evaluate_world(student, eval_tasks)
    if (
        expanded_polish_candidate_allowed(
            final_metrics, retention_reference, fixed_frontier_metrics, keep_reverse
        )
        and comp.better_sort_transfer_candidate(final_metrics, raw_best_metrics, keep_reverse, retention_reference)
    ):
        raw_best_metrics = dict(final_metrics)
        raw_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
    if (
        expanded_guard_candidate_ok(final_metrics, retention_reference, fixed_frontier_metrics, keep_reverse)
        and expanded_polish_candidate_allowed(
            final_metrics, retention_reference, fixed_frontier_metrics, keep_reverse
        )
        and comp.better_sort_transfer_candidate(final_metrics, guard_best_metrics, keep_reverse, retention_reference)
    ):
        guard_best_metrics = dict(final_metrics)
        guard_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
    if (
        expanded_safe_candidate_ok(final_metrics, retention_reference, fixed_frontier_metrics, keep_reverse)
        and expanded_polish_candidate_allowed(
            final_metrics, retention_reference, fixed_frontier_metrics, keep_reverse
        )
        and comp.better_sort_transfer_candidate(final_metrics, safe_best_metrics, keep_reverse, retention_reference)
    ):
        safe_best_metrics = dict(final_metrics)
        safe_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
    if (
        expanded_balanced_branch_candidate_ok(final_metrics, fixed_frontier_metrics, fixed_final_metrics)
        if balanced_profile
        else expanded_balanced_candidate_ok(final_metrics, fixed_final_metrics)
    ):
        if balanced_best_metrics is None or (
            expanded_balanced_branch_priority(final_metrics, fixed_frontier_metrics, fixed_final_metrics)
            if balanced_profile
            else expanded_balanced_priority(final_metrics)
        ) > (
            expanded_balanced_branch_priority(balanced_best_metrics, fixed_frontier_metrics, fixed_final_metrics)
            if balanced_profile
            else expanded_balanced_priority(balanced_best_metrics)
        ):
            balanced_best_metrics = dict(final_metrics)
            balanced_best_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))
    if expanded_best_d_candidate_ok(final_metrics, retention_reference, keep_reverse):
        if best_d_metrics is None or expanded_best_d_priority(final_metrics) > expanded_best_d_priority(best_d_metrics):
            best_d_metrics = dict(final_metrics)
            best_d_checkpoint = expand.make_layer_checkpoint(student, optimizer, len(student.blocks))

    del (
        new_teacher,
        _new_teacher_opt,
        new_teacher_forward,
        old_teacher,
        _old_teacher_opt,
        student,
        _student_opt,
        optimizer,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_stage = phase.StageResult(
        label=label,
        checkpoint=raw_best_checkpoint,
        metrics=raw_best_metrics,
        old_batch_count=old_batch_count,
        old_batch_budget=old_batch_budget,
        base_only_verified=True,
    )
    guard_stage = (
        phase.StageResult(
            label=label,
            checkpoint=guard_best_checkpoint,
            metrics=guard_best_metrics,
            old_batch_count=old_batch_count,
            old_batch_budget=old_batch_budget,
            base_only_verified=True,
        )
        if guard_best_metrics is not None
        else None
    )
    safe_stage = (
        phase.StageResult(
            label=label,
            checkpoint=safe_best_checkpoint,
            metrics=safe_best_metrics,
            old_batch_count=old_batch_count,
            old_batch_budget=old_batch_budget,
            base_only_verified=True,
        )
        if safe_best_metrics is not None
        else None
    )
    balanced_stage = (
        phase.StageResult(
            label=label,
            checkpoint=balanced_best_checkpoint,
            metrics=balanced_best_metrics,
            old_batch_count=old_batch_count,
            old_batch_budget=old_batch_budget,
            base_only_verified=True,
        )
        if balanced_best_metrics is not None
        else None
    )
    if (
        balanced_profile
        and preserved_balanced_stage is not None
        and (
            balanced_stage is None
            or expanded_balanced_branch_priority(
                preserved_balanced_stage.metrics,
                fixed_frontier_metrics,
                fixed_final_metrics,
            ) > expanded_balanced_branch_priority(
                balanced_stage.metrics,
                fixed_frontier_metrics,
                fixed_final_metrics,
            )
            or sort_prob(balanced_stage.metrics)
            < sort_prob(preserved_balanced_stage.metrics) - EXPANDED_BALANCED_POLISH_MAX_SORT_DROP
        )
    ):
        balanced_stage = preserved_balanced_stage
    best_d_stage = phase.StageResult(
        label=label,
        checkpoint=best_d_checkpoint,
        metrics=best_d_metrics if best_d_metrics is not None else raw_best_metrics,
        old_batch_count=old_batch_count,
        old_batch_budget=old_batch_budget,
        base_only_verified=True,
    )
    fallback_stage = choose_expanded_sort_transfer_stage(
        raw_stage,
        guard_stage,
        safe_stage,
        None,
        retention_reference,
        fixed_frontier_metrics,
        keep_reverse,
    )
    if balanced_profile:
        selected_stage = choose_expanded_balanced_branch_stage(
            fallback_stage,
            balanced_stage,
            fixed_frontier_metrics,
            fixed_final_metrics,
            preserved_stage=preserved_balanced_stage,
        )
    else:
        selected_stage = choose_expanded_best_d_branch_stage(
            fallback_stage,
            best_d_stage,
            retention_reference,
            keep_reverse,
            preserved_stage=preserved_best_d_stage,
        )
    return ExpandedStageSelection(selected=selected_stage, best_d=best_d_stage, balanced=balanced_stage)


def write_rows(rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "seed",
        "stage",
        "bracket_seq",
        "text_loss",
        "reverse_problem_acc",
        "sort_problem_acc",
        "reverse_acc",
        "sort_acc",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pareto_plot(rows: List[Dict[str, object]]) -> None:
    if plt is None:
        return

    stage_styles = {
        "base_ABD_unified": {"color": "#444444", "marker": "D", "size": 90},
        "fixed_mixed_frontier_ABCD": {"color": "#b22222", "marker": "s", "size": 85},
        "fixed_base_ABCD_unified": {"color": "#e15759", "marker": "o", "size": 90},
        "expanded_best_D_teacher": {"color": "#1f77b4", "marker": "x", "size": 80},
        "expanded_balanced_teacher": {"color": "#4e79a7", "marker": "x", "size": 80},
        "expanded_best_D_ABCD": {"color": "#2ca02c", "marker": "^", "size": 95},
        "expanded_balanced_ABCD": {"color": "#59a14f", "marker": "P", "size": 100},
        "expanded_base_ABCD_unified": {"color": "#0b6e4f", "marker": "*", "size": 170},
    }
    label_alias = {
        "base_ABD_unified": "ABD upper bound",
        "fixed_mixed_frontier_ABCD": "Fixed frontier",
        "fixed_base_ABCD_unified": "Fixed final",
        "expanded_best_D_teacher": "Expanded best-D teacher",
        "expanded_balanced_teacher": "Expanded balanced teacher",
        "expanded_best_D_ABCD": "Expanded best-D",
        "expanded_balanced_ABCD": "Expanded balanced",
        "expanded_base_ABCD_unified": "Expanded headline",
    }

    plot_rows = [
        row
        for row in rows
        if row["stage"] in stage_styles
        and isinstance(row.get("reverse_problem_acc"), (int, float))
        and isinstance(row.get("sort_problem_acc"), (int, float))
    ]
    if not plot_rows:
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    seen_labels: set[str] = set()
    for row in plot_rows:
        stage = str(row["stage"])
        seed = int(row["seed"])
        x = float(row["reverse_problem_acc"])
        y = float(row["sort_problem_acc"])
        style = stage_styles[stage]
        legend_label = label_alias[stage] if stage not in seen_labels else None
        seen_labels.add(stage)
        ax.scatter(
            x,
            y,
            s=style["size"],
            c=style["color"],
            marker=style["marker"],
            alpha=0.9,
            label=legend_label,
        )
        ax.annotate(
            f"{label_alias[stage]} ({seed})",
            (x, y),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            color=style["color"],
        )

    def line_points(stage_names: Sequence[str]) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for name in stage_names:
            match = next((row for row in plot_rows if row["stage"] == name), None)
            if match is not None:
                points.append(
                    (
                        float(match["reverse_problem_acc"]),
                        float(match["sort_problem_acc"]),
                    )
                )
        return points

    fixed_line = line_points(["fixed_mixed_frontier_ABCD", "fixed_base_ABCD_unified"])
    expanded_line = line_points(["expanded_best_D_ABCD", "expanded_balanced_ABCD", "expanded_base_ABCD_unified"])
    if len(fixed_line) >= 2:
        xs, ys = zip(*fixed_line)
        ax.plot(xs, ys, color="#e15759", linewidth=1.5, alpha=0.7)
    if len(expanded_line) >= 2:
        xs, ys = zip(*expanded_line)
        ax.plot(xs, ys, color="#2ca02c", linewidth=1.5, alpha=0.7)

    ax.set_title("Compositional Expansion Pareto Frontier")
    ax.set_xlabel("Reverse Retention (C)")
    ax.set_ylabel("Sort Transfer (D)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(PARETO_PLOT_PATH, dpi=180)
    plt.close(fig)


def run_fixed_compositional_pipeline(
    seed: int,
    vocab_size: int,
    stoi: Dict[str, int],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    bracket_task: phase.TaskSpec,
    text_task: phase.TaskSpec,
    reverse_train_task: phase.TaskSpec,
    reverse_probe_task: phase.TaskSpec,
    sort_task: phase.TaskSpec,
    bracket_eval_batches: Sequence[ww.Batch],
) -> FixedCompositionalResult:
    anchor = ww.train_old_skill(
        vocab_size,
        stoi,
        seed,
        bracket_eval_batches,
        ww.make_fixed_bracket_batches(stoi, seed + 40_000, 1, ww.PROBE_BATCH)[0],
    )
    anchor_a = phase.anchor_from_old_skill(anchor)
    base_a_metrics = comp.evaluate_checkpoint(anchor_a.checkpoint, vocab_size, [bracket_task, text_task, reverse_probe_task])

    teacher_b = phase.attach_latent_teacher(
        "B_text",
        anchor_a,
        vocab_size,
        [bracket_task],
        text_task,
        [bracket_task, text_task],
        seed,
    )
    base_ab = phase.consolidate_dual_teacher(
        "AB",
        phase.AB_CONSOLIDATION_BRANCH,
        anchor_a.checkpoint,
        anchor_a,
        anchor_a.checkpoint,
        [bracket_task],
        teacher_b.checkpoint,
        {block: projector.to(ww.DEVICE) for block, projector in anchor_a.latent_free_projectors.items()},
        text_task,
        [bracket_task, text_task],
        vocab_size,
    )
    reverse_ab = phase.reverse_extract_to_base_a(
        "AB_to_A",
        base_ab.checkpoint,
        anchor_a.checkpoint,
        base_ab.metrics,
        anchor_a,
        bracket_task,
        text_task,
        [bracket_task, text_task],
        vocab_size,
    )

    anchor_ab = phase.collect_world_anchor("AB", base_ab.checkpoint, [bracket_task, text_task], vocab_size)
    teacher_c_attach = phase.attach_latent_teacher(
        "C_reverse",
        anchor_ab,
        vocab_size,
        [bracket_task, text_task],
        reverse_train_task,
        [bracket_task, text_task, reverse_train_task],
        seed + 77,
    )
    teacher_c = phase.sharpen_arithmetic_teacher(
        "C_reverse",
        teacher_c_attach,
        anchor_ab,
        vocab_size,
        [bracket_task, text_task],
        reverse_train_task,
        [bracket_task, text_task, reverse_train_task],
    )
    base_abc_consolidated = phase.consolidate_dual_teacher(
        "ABC",
        phase.ABC_CONSOLIDATION_BRANCH,
        anchor_ab.checkpoint,
        anchor_ab,
        anchor_ab.checkpoint,
        [bracket_task, text_task],
        teacher_c.checkpoint,
        {block: projector.to(ww.DEVICE) for block, projector in anchor_ab.latent_free_projectors.items()},
        reverse_train_task,
        [bracket_task, text_task, reverse_train_task],
        vocab_size,
        selection_reference=base_ab.metrics,
    )
    base_abc = phase.polish_arithmetic_transfer(
        "ABC",
        base_abc_consolidated,
        anchor_ab,
        vocab_size,
        [bracket_task, text_task],
        reverse_train_task,
        [bracket_task, text_task, reverse_train_task],
        teacher_c.checkpoint,
        {block: projector.to(ww.DEVICE) for block, projector in anchor_ab.latent_free_projectors.items()},
        teacher_c.metrics,
        base_ab.metrics,
    )
    base_abc_metrics = comp.evaluate_checkpoint(
        base_abc.checkpoint,
        vocab_size,
        [bracket_task, text_task, reverse_probe_task],
    )

    teacher_d_ab_attach = phase.attach_latent_teacher(
        "D_sort_from_AB",
        anchor_ab,
        vocab_size,
        [bracket_task, text_task],
        sort_task,
        [bracket_task, text_task, sort_task],
        seed + 101,
    )
    teacher_d_ab = phase.sharpen_arithmetic_teacher(
        "D_sort_from_AB",
        teacher_d_ab_attach,
        anchor_ab,
        vocab_size,
        [bracket_task, text_task],
        sort_task,
        [bracket_task, text_task, sort_task],
    )
    base_abd_consolidated = phase.consolidate_dual_teacher(
        "ABD",
        phase.ABC_CONSOLIDATION_BRANCH,
        anchor_ab.checkpoint,
        anchor_ab,
        anchor_ab.checkpoint,
        [bracket_task, text_task],
        teacher_d_ab.checkpoint,
        {block: projector.to(ww.DEVICE) for block, projector in anchor_ab.latent_free_projectors.items()},
        sort_task,
        [bracket_task, text_task, reverse_probe_task, sort_task],
        vocab_size,
        selection_reference=base_ab.metrics,
    )
    base_abd = comp.polish_sort_transfer(
        "ABD",
        base_abd_consolidated,
        anchor_ab.checkpoint,
        vocab_size,
        bracket_task,
        text_task,
        None,
        sort_task,
        [bracket_task, text_task, reverse_probe_task, sort_task],
        teacher_d_ab.checkpoint,
        {block: projector.to(ww.DEVICE) for block, projector in anchor_ab.latent_free_projectors.items()},
        teacher_d_ab.metrics,
        base_ab.metrics,
        keep_reverse=False,
    )
    base_abd_metrics = merged_metrics(
        comp.evaluate_checkpoint(base_abd.checkpoint, vocab_size, [bracket_task, text_task, sort_task]),
        comp.evaluate_checkpoint(base_abd.checkpoint, vocab_size, [reverse_probe_task]),
    )

    anchor_abc = phase.collect_world_anchor(
        "ABC",
        base_abc.checkpoint,
        [bracket_task, text_task, reverse_train_task],
        vocab_size,
    )
    teacher_d_abc_attach = comp.attach_sort_teacher_with_retention(
        "D_sort_from_ABC",
        anchor_abc,
        vocab_size,
        bracket_task,
        text_task,
        reverse_train_task,
        sort_task,
        [bracket_task, text_task, reverse_probe_task, sort_task],
        base_abc_metrics,
        seed + 202,
    )
    teacher_d_abc = comp.sharpen_sort_teacher_with_retention(
        "D_sort_from_ABC",
        teacher_d_abc_attach,
        anchor_abc,
        vocab_size,
        bracket_task,
        text_task,
        reverse_train_task,
        sort_task,
        [bracket_task, text_task, reverse_probe_task, sort_task],
        base_abc_metrics,
    )
    base_abcd_consolidated = comp.consolidate_sort_transfer(
        "ABCD",
        phase.ABC_CONSOLIDATION_BRANCH,
        anchor_abc.checkpoint,
        anchor_abc,
        anchor_abc.checkpoint,
        [bracket_task, text_task, reverse_train_task],
        teacher_d_abc.checkpoint,
        {block: projector.to(ww.DEVICE) for block, projector in anchor_abc.latent_free_projectors.items()},
        bracket_task,
        text_task,
        reverse_train_task,
        sort_task,
        [bracket_task, text_task, reverse_probe_task, sort_task],
        vocab_size,
        selection_reference=base_abc_metrics,
        keep_reverse=True,
    )
    base_abcd = comp.polish_sort_transfer(
        "ABCD",
        base_abcd_consolidated,
        anchor_abc.checkpoint,
        vocab_size,
        bracket_task,
        text_task,
        reverse_train_task,
        sort_task,
        [bracket_task, text_task, reverse_probe_task, sort_task],
        teacher_d_abc.checkpoint,
        {block: projector.to(ww.DEVICE) for block, projector in anchor_abc.latent_free_projectors.items()},
        teacher_d_abc.metrics,
        base_abc_metrics,
        keep_reverse=True,
    )
    base_abcd_metrics = merged_metrics(
        comp.evaluate_checkpoint(base_abcd.checkpoint, vocab_size, [bracket_task, text_task, sort_task]),
        comp.evaluate_checkpoint(base_abcd.checkpoint, vocab_size, [reverse_probe_task]),
    )
    fixed_mixed_frontier_metrics = choose_fixed_mixed_frontier_metrics(
        collect_sort_transfer_candidate_metrics(base_abcd_consolidated)
        + collect_sort_transfer_candidate_metrics(base_abcd)
    )

    teacher_d_ab_metrics = merged_metrics(
        dict(teacher_d_ab.metrics),
        comp.evaluate_checkpoint(
            teacher_d_ab.checkpoint,
            vocab_size,
            [reverse_probe_task],
            adapters_enabled=True,
            latent_projectors=anchor_ab.latent_free_projectors,
            latent_strength=phase.ARITH_SHARPEN_PROJECTION,
        ),
    )
    teacher_d_abc_metrics = merged_metrics(
        dict(teacher_d_abc.metrics),
        comp.evaluate_checkpoint(
            teacher_d_abc.checkpoint,
            vocab_size,
            [reverse_probe_task],
            adapters_enabled=True,
            latent_projectors=anchor_abc.latent_free_projectors,
            latent_strength=phase.ARITH_SHARPEN_PROJECTION,
        ),
    )

    return FixedCompositionalResult(
        base_a_metrics=base_a_metrics,
        base_ab=base_ab,
        reverse_ab=reverse_ab,
        base_abc=base_abc,
        base_abc_metrics=base_abc_metrics,
        teacher_d_ab=teacher_d_ab,
        teacher_d_ab_metrics=teacher_d_ab_metrics,
        base_abd=base_abd,
        base_abd_metrics=base_abd_metrics,
        teacher_d_abc=teacher_d_abc,
        teacher_d_abc_metrics=teacher_d_abc_metrics,
        base_abcd_consolidated=base_abcd_consolidated,
        fixed_mixed_frontier_metrics=fixed_mixed_frontier_metrics,
        base_abcd=base_abcd,
        base_abcd_metrics=base_abcd_metrics,
    )


def append_stage_row(rows: List[Dict[str, object]], seed: int, label: str, metrics: Dict[str, float]) -> None:
    rows.append(
        {
            "seed": seed,
            "stage": label,
            "bracket_seq": metrics.get("bracket_seq", float("nan")),
            "text_loss": metrics.get("text_loss", float("nan")),
            "reverse_problem_acc": metrics.get("reverse_problem_acc", float("nan")),
            "sort_problem_acc": metrics.get("sort_problem_acc", metrics.get("arith_problem_acc", float("nan"))),
            "reverse_acc": metrics.get("reverse_acc", float("nan")),
            "sort_acc": metrics.get("sort_acc", metrics.get("arith_acc", float("nan"))),
        }
    )


def run_seed(
    seed: int,
    vocab_size: int,
    stoi: Dict[str, int],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> List[Dict[str, object]]:
    print("\n" + "#" * 78)
    print(f"BEGIN COMPOSITIONAL EXPANSION SEED {seed}")
    print("#" * 78)
    ww.set_seed(seed)

    text_eval_positions = phase.make_text_eval_positions(len(val_data), seed)
    bracket_eval_batches = ww.make_fixed_bracket_batches(
        stoi, seed + 30_000, ww.BRACKET_EVAL_BATCHES, ww.BRACKET_EVAL_BATCH
    )
    reverse_eval_batches = phase.make_fixed_arithmetic_batches(
        stoi, seed + 31_000, ww.BRACKET_EVAL_BATCHES, ww.BRACKET_EVAL_BATCH
    )
    sort_eval_batches = comp.make_fixed_sort_batches(
        stoi, seed + 32_000, ww.BRACKET_EVAL_BATCHES, ww.BRACKET_EVAL_BATCH
    )

    bracket_task = phase.make_bracket_task(stoi, bracket_eval_batches, seed)
    text_task = phase.make_text_task(train_data, val_data, text_eval_positions, seed)
    reverse_train_task = phase.make_arithmetic_task(stoi, reverse_eval_batches, seed)
    reverse_probe_task = comp.make_reverse_probe_task(reverse_eval_batches)
    sort_task = comp.make_sort_task(stoi, sort_eval_batches, seed)

    fixed = run_fixed_compositional_pipeline(
        seed,
        vocab_size,
        stoi,
        train_data,
        val_data,
        bracket_task,
        text_task,
        reverse_train_task,
        reverse_probe_task,
        sort_task,
        bracket_eval_batches,
    )

    expanded_identity = expand.build_expanded_identity_checkpoint(vocab_size, fixed.base_abc.checkpoint)
    prop_identity = prop.build_prop_identity_checkpoint(vocab_size, fixed.base_abc.checkpoint)
    expanded_teacher_selection = train_propagation_sort_teacher_candidates(
        vocab_size,
        prop_identity,
        fixed.base_abc.checkpoint,
        bracket_task,
        text_task,
        reverse_train_task,
        sort_task,
        [bracket_task, text_task, reverse_probe_task, sort_task],
        fixed.base_abc_metrics,
        fixed.fixed_mixed_frontier_metrics,
        fixed.base_abcd_metrics,
    )
    expanded_best_d_teacher_metrics = evaluate_external_checkpoint(
        expanded_teacher_selection.best_d.checkpoint,
        vocab_size,
        [bracket_task, text_task, reverse_probe_task, sort_task],
    )
    expanded_balanced_teacher_metrics = evaluate_external_checkpoint(
        expanded_teacher_selection.balanced.checkpoint,
        vocab_size,
        [bracket_task, text_task, reverse_probe_task, sort_task],
    )
    expanded_frontier_reference = make_expanded_frontier_reference(
        fixed.base_abc_metrics,
        fixed.fixed_mixed_frontier_metrics,
    )
    expanded_best_d_consolidated = consolidate_expanded_sort_transfer(
        "expanded_best_D_ABCD",
        EXPANDED_BRANCH,
        expanded_identity,
        fixed.base_abc.checkpoint,
        expanded_teacher_selection.best_d.checkpoint,
        bracket_task,
        text_task,
        reverse_train_task,
        sort_task,
        [bracket_task, text_task, reverse_probe_task, sort_task],
        vocab_size,
        selection_reference=expanded_frontier_reference,
        fixed_frontier_metrics=fixed.fixed_mixed_frontier_metrics,
        fixed_final_metrics=fixed.base_abcd_metrics,
        keep_reverse=True,
        profile="best_d",
    )
    expanded_best_d_selection = polish_expanded_sort_transfer(
        "expanded_best_D_ABCD",
        expanded_best_d_consolidated,
        fixed.base_abc.checkpoint,
        vocab_size,
        bracket_task,
        text_task,
        reverse_train_task,
        sort_task,
        [bracket_task, text_task, reverse_probe_task, sort_task],
        expanded_teacher_selection.best_d.checkpoint,
        expanded_teacher_selection.best_d.metrics,
        expanded_frontier_reference,
        fixed.fixed_mixed_frontier_metrics,
        fixed.base_abcd_metrics,
        keep_reverse=True,
        profile="best_d",
    )
    expanded_balanced_identity = expand.build_expanded_identity_checkpoint(vocab_size, fixed.base_abc.checkpoint)
    expanded_balanced_consolidated = consolidate_expanded_sort_transfer(
        "expanded_balanced_ABCD",
        EXPANDED_BRANCH,
        expanded_balanced_identity,
        fixed.base_abc.checkpoint,
        expanded_teacher_selection.balanced.checkpoint,
        bracket_task,
        text_task,
        reverse_train_task,
        sort_task,
        [bracket_task, text_task, reverse_probe_task, sort_task],
        vocab_size,
        selection_reference=expanded_frontier_reference,
        fixed_frontier_metrics=fixed.fixed_mixed_frontier_metrics,
        fixed_final_metrics=fixed.base_abcd_metrics,
        keep_reverse=True,
        profile="balanced",
    )
    expanded_balanced_selection = polish_expanded_sort_transfer(
        "expanded_balanced_ABCD",
        expanded_balanced_consolidated,
        fixed.base_abc.checkpoint,
        vocab_size,
        bracket_task,
        text_task,
        reverse_train_task,
        sort_task,
        [bracket_task, text_task, reverse_probe_task, sort_task],
        expanded_teacher_selection.balanced.checkpoint,
        expanded_teacher_selection.balanced.metrics,
        expanded_frontier_reference,
        fixed.fixed_mixed_frontier_metrics,
        fixed.base_abcd_metrics,
        keep_reverse=True,
        profile="balanced",
    )
    expanded_abcd = choose_expanded_branch_selection(
        expanded_best_d_selection,
        expanded_balanced_selection,
        fixed.base_abcd_metrics,
        fixed.fixed_mixed_frontier_metrics,
    )
    expanded_abcd_metrics = merged_metrics(
        evaluate_external_checkpoint(expanded_abcd.checkpoint, vocab_size, [bracket_task, text_task, sort_task]),
        evaluate_external_checkpoint(expanded_abcd.checkpoint, vocab_size, [reverse_probe_task]),
    )
    expanded_best_d_metrics = merged_metrics(
        evaluate_external_checkpoint(expanded_best_d_selection.best_d.checkpoint, vocab_size, [bracket_task, text_task, sort_task]),
        evaluate_external_checkpoint(expanded_best_d_selection.best_d.checkpoint, vocab_size, [reverse_probe_task]),
    )
    expanded_balanced_stage = (
        expanded_balanced_selection.balanced
        if expanded_balanced_selection.balanced is not None
        else expanded_balanced_selection.selected
    )
    expanded_balanced_metrics = merged_metrics(
        evaluate_external_checkpoint(expanded_balanced_stage.checkpoint, vocab_size, [bracket_task, text_task, sort_task]),
        evaluate_external_checkpoint(expanded_balanced_stage.checkpoint, vocab_size, [reverse_probe_task]),
    )

    print("\n" + "=" * 78)
    print(f"SEED {seed} COMPOSITIONAL EXPANSION COMPARISON")
    print("=" * 78)
    print(f"{'stage':32s} {'bracket_seq':>11s} {'text_loss':>10s} {'rev_prob':>10s} {'sort_prob':>10s}")
    stage_rows = [
        ("base_ABC_unified", fixed.base_abc_metrics),
        ("base_ABD_unified", fixed.base_abd_metrics),
        ("fixed_teacher_D_from_ABC", fixed.teacher_d_abc_metrics),
        ("fixed_mixed_frontier_ABCD", fixed.fixed_mixed_frontier_metrics),
        ("fixed_base_ABCD_unified", fixed.base_abcd_metrics),
        ("expanded_best_D_teacher", expanded_best_d_teacher_metrics),
        ("expanded_balanced_teacher", expanded_balanced_teacher_metrics),
        ("expanded_best_D_ABCD", expanded_best_d_metrics),
        ("expanded_balanced_ABCD", expanded_balanced_metrics),
    ]
    stage_rows.append(("expanded_base_ABCD_unified", expanded_abcd_metrics))
    for label, metrics in stage_rows:
        print_stage(label, metrics)
    print(
        "Expanded headline vs fixed final: "
        f"sort_prob={expanded_abcd_metrics.get('sort_problem_acc', 0.0) - fixed.base_abcd_metrics.get('sort_problem_acc', 0.0):+.3f} "
        f"reverse_prob={expanded_abcd_metrics.get('reverse_problem_acc', 0.0) - fixed.base_abcd_metrics.get('reverse_problem_acc', 0.0):+.3f} "
        f"bracket={expanded_abcd_metrics.get('bracket_seq', 0.0) - fixed.base_abcd_metrics.get('bracket_seq', 0.0):+.3f}"
    )
    print(
        "Expanded best-D vs fixed final: "
        f"sort_prob={expanded_best_d_metrics.get('sort_problem_acc', 0.0) - fixed.base_abcd_metrics.get('sort_problem_acc', 0.0):+.3f} "
        f"reverse_prob={expanded_best_d_metrics.get('reverse_problem_acc', 0.0) - fixed.base_abcd_metrics.get('reverse_problem_acc', 0.0):+.3f} "
        f"bracket={expanded_best_d_metrics.get('bracket_seq', 0.0) - fixed.base_abcd_metrics.get('bracket_seq', 0.0):+.3f}"
    )
    print(
        "Expanded headline vs fixed frontier: "
        f"sort_prob={expanded_abcd_metrics.get('sort_problem_acc', 0.0) - fixed.fixed_mixed_frontier_metrics.get('sort_problem_acc', 0.0):+.3f} "
        f"reverse_prob={expanded_abcd_metrics.get('reverse_problem_acc', 0.0) - fixed.fixed_mixed_frontier_metrics.get('reverse_problem_acc', 0.0):+.3f} "
        f"bracket={expanded_abcd_metrics.get('bracket_seq', 0.0) - fixed.fixed_mixed_frontier_metrics.get('bracket_seq', 0.0):+.3f}"
    )
    print(
        "Gap to ABD upper bound: "
        f"fixed_sort_gap={fixed.base_abd_metrics.get('sort_problem_acc', 0.0) - fixed.base_abcd_metrics.get('sort_problem_acc', 0.0):+.3f} "
        f"expanded_sort_gap={fixed.base_abd_metrics.get('sort_problem_acc', 0.0) - expanded_abcd_metrics.get('sort_problem_acc', 0.0):+.3f}"
    )
    print(
        "C retention through D: "
        f"fixed_reverse_delta={fixed.base_abcd_metrics.get('reverse_problem_acc', 0.0) - fixed.base_abc_metrics.get('reverse_problem_acc', 0.0):+.3f} "
        f"expanded_reverse_delta={expanded_abcd_metrics.get('reverse_problem_acc', 0.0) - fixed.base_abc_metrics.get('reverse_problem_acc', 0.0):+.3f}"
    )
    print("=" * 78)

    rows: List[Dict[str, object]] = []
    append_stage_row(rows, seed, "base_ABC_unified", fixed.base_abc_metrics)
    append_stage_row(rows, seed, "base_ABD_unified", fixed.base_abd_metrics)
    append_stage_row(rows, seed, "fixed_teacher_D_from_ABC", fixed.teacher_d_abc_metrics)
    append_stage_row(rows, seed, "fixed_mixed_frontier_ABCD", fixed.fixed_mixed_frontier_metrics)
    append_stage_row(rows, seed, "fixed_base_ABCD_unified", fixed.base_abcd_metrics)
    append_stage_row(rows, seed, "expanded_best_D_teacher", expanded_best_d_teacher_metrics)
    append_stage_row(rows, seed, "expanded_balanced_teacher", expanded_balanced_teacher_metrics)
    append_stage_row(rows, seed, "expanded_best_D_ABCD", expanded_best_d_metrics)
    append_stage_row(rows, seed, "expanded_balanced_ABCD", expanded_balanced_metrics)
    append_stage_row(rows, seed, "expanded_base_ABCD_unified", expanded_abcd_metrics)
    return rows


def main() -> None:
    ww.set_seed(LAB_SEEDS[0])
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 78)
    print("COMPOSITIONAL EXPANSION RESCUE LAB")
    print("=" * 78)
    print("Question: can an expanded ABCD model rescue D transfer once fixed-size ABCD hits a C-vs-D tradeoff frontier?")
    print(f"Device: {ww.DEVICE}")
    print(f"Seeds: {LAB_SEEDS}")
    print(f"Phase A: bracket until seq>={ww.OLD_READY_SEQ:.2f} or {ww.PHASE_A_MAX_STEPS} steps")
    print(f"Phase B: latent text attach for {ww.PHASE_B_STEPS} steps")
    print(f"Phase C: dual-teacher consolidation for {lateral.CONSOLIDATION_STEPS} steps")
    print(f"Phase D: latent reversal attach for {ww.PHASE_B_STEPS} steps")
    print(f"Phase D2: reversal sharpen for {phase.ARITH_SHARPEN_STEPS} steps")
    print(f"Phase E: dual-teacher consolidation for {lateral.CONSOLIDATION_STEPS} steps")
    print(f"Phase E2: base-only reversal transfer polish for {phase.ARITH_TRANSFER_POLISH_STEPS} steps")
    print(f"Phase F: fixed-size sort attach/sharpen for {ww.PHASE_B_STEPS}+{phase.ARITH_SHARPEN_STEPS} steps")
    print(f"Phase G: fixed-size ABCD consolidation/polish for {lateral.CONSOLIDATION_STEPS}+{phase.ARITH_TRANSFER_POLISH_STEPS} steps")
    print(f"Phase H: propagation expansion attach for {prop.PROP_ATTACH_STEPS} steps")
    print(
        "Phase I: expanded ABCD consolidation/polish "
        f"(best-D {lateral.CONSOLIDATION_STEPS}+{comp.SORT_TRANSFER_POLISH_STEPS}, "
        f"balanced {lateral.CONSOLIDATION_STEPS}+{EXPANDED_BALANCED_POLISH_STEPS} steps)"
    )
    print(
        f"Model: d={ww.D_MODEL}, fixed_layers={ww.N_LAYER}, expanded_layers={ww.N_LAYER + expand.EXTRA_BLOCKS}, "
        f"heads={ww.N_HEAD}, block={ww.BLOCK_SIZE}, adapter_rank={ww.ADAPTER_RANK}"
    )
    print(
        f"Sort D expansion path: propagation_iters={prop.PROP_ITERATIONS}, "
        f"new_block_lr={prop.PROP_NEW_BLOCK_LR:.2e}, compat_scale=({prop.PROP_COMPAT_EARLY:.2f},{prop.PROP_COMPAT_MID:.2f},{prop.PROP_COMPAT_LATE:.2f})"
    )

    text = ww.download_or_load_text()
    stoi, _itos = phase.build_joint_vocab(text)
    encoded = ww.encode(text, stoi)
    split = int(0.95 * len(encoded))
    train_data = encoded[:split]
    val_data = encoded[split:]
    print(f"Vocab size: {len(stoi)}")
    print(f"Train tokens: {train_data.numel():,} | Val tokens: {val_data.numel():,}")

    all_rows: List[Dict[str, object]] = []
    start = time.time()
    for seed in LAB_SEEDS:
        all_rows.extend(run_seed(seed, len(stoi), stoi, train_data, val_data))
    write_rows(all_rows)
    write_pareto_plot(all_rows)
    print(f"CSV saved to: {CSV_PATH}")
    if plt is not None:
        print(f"Pareto plot saved to: {PARETO_PLOT_PATH}")
    print("=" * 78)
    print(f"Total wall time: {phase.format_seconds(time.time() - start)}")


if __name__ == "__main__":
    main()
