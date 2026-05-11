#!/usr/bin/env python3
"""
Compositional transfer lab for Water Weights.

Question:
Can we show A -> B -> C -> D without collapse, and does learning C improve
later transfer of D compared with attaching D directly from AB?
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import colab_phase_reversibility_lab as phase
import colab_water_weights_benchmark as ww
import colab_water_weights_lateral_consolidation_benchmark as lateral


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "compositional_transfer_lab_results.csv"

LAB_SEEDS = list(phase.LAB_SEEDS[:1])
SORT_ALPHABET = phase.ARITH_CHARS[:-2]
SORT_EVAL_LEN = 4 if ww.D_MODEL <= 256 else 6
SORT_ATTACH_STAGES = (2, 3, SORT_EVAL_LEN)
SORT_ANSWER_WEIGHT = 6.0
SORT_CONTEXT_WEIGHT = 0.10
SORT_SINGLE_PROBLEM = True
SORT_RANDOM_PAD = False
SORT_TRANSFER_TARGET_FRACTION = 0.85
SORT_TRANSFER_BRACKET_SLACK = 0.04
SORT_TRANSFER_TEXT_SLACK = 0.55
SORT_TRANSFER_REVERSE_FRACTION = 0.85
SORT_TRANSFER_REVERSE_ABS_SLACK = 0.10
SORT_TRANSFER_POLISH_STEPS = phase.ARITH_TRANSFER_POLISH_STEPS
SORT_TRANSFER_AUX_REVERSE_PERIOD = 2
SORT_TRANSFER_AUX_REVERSE_TASK_WEIGHT = 1.25
SORT_TRANSFER_AUX_REVERSE_KL_WEIGHT = 0.45
SORT_TRANSFER_AUX_REVERSE_HIDDEN_WEIGHT = 0.30
SORT_TRANSFER_MIN_REVERSE_RATIO = 0.20
SORT_TRANSFER_RAW_SORT_ADVANTAGE = 0.12
SORT_TRANSFER_RAW_JOINT_ADVANTAGE = 0.04
SORT_TRANSFER_SAFE_SORT_FLOOR = 0.20
SORT_TRANSFER_RAW_BRACKET_SLACK = 0.10
SORT_TRANSFER_RAW_MIN_BRACKET = 0.85
SORT_TRANSFER_RAW_TEXT_SLACK = 0.18
SORT_TRANSFER_BALANCED_SORT_FLOOR = 0.60
SORT_TRANSFER_BALANCED_REVERSE_FLOOR = 0.20
SORT_TRANSFER_BALANCED_REVERSE_BOOST = 1.05
SORT_TRANSFER_BALANCED_AUX_REVERSE_PERIOD = 4
SORT_TRANSFER_BALANCED_AUX_SCALE = 0.50
SORT_TRANSFER_EVAL_INTERVAL = max(4, lateral.CONSOLIDATION_EVAL_INTERVAL // 2)
SORT_TRANSFER_LOG_INTERVAL = max(8, lateral.CONSOLIDATION_LOG_INTERVAL // 2)
SORT_TEACHER_BRACKET_SLACK = 0.08
SORT_TEACHER_TEXT_SLACK = 0.90
SORT_TEACHER_REVERSE_COMPAT_BOOST = 1.50
SORT_TEACHER_MIN_REVERSE_RATIO = 0.10
SORT_TRANSFER_CANDIDATE_METRICS_KEY = "_sort_transfer_candidate_metrics"


def build_sort_stream(rng: np.random.Generator, length: int = SORT_EVAL_LEN) -> Tuple[List[str], List[bool]]:
    token_ids = rng.integers(0, len(SORT_ALPHABET), size=length)
    source = "".join(SORT_ALPHABET[int(idx)] for idx in token_ids)
    prompt = f"{source}|"
    rhs = f"{''.join(sorted(source))}\n"
    episode = list(prompt + rhs)
    flags = [False] * len(prompt) + [True] * length + [False]

    total_len = ww.BLOCK_SIZE + 1
    pad_len = max(total_len - len(episode), 0)
    if SORT_RANDOM_PAD and pad_len > 0:
        left_pad = int(rng.integers(0, pad_len + 1))
        right_pad = pad_len - left_pad
    else:
        left_pad = 0
        right_pad = pad_len
    chars = ([" "] * left_pad) + episode + ([" "] * right_pad)
    critical = ([False] * left_pad) + flags + ([False] * right_pad)
    return chars[:total_len], critical[:total_len]


def make_sort_batch(
    stoi: Dict[str, int],
    seed: int,
    index: int,
    batch_size: int = ww.BATCH_SIZE,
    length: int = SORT_EVAL_LEN,
) -> ww.Batch:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for offset in range(batch_size):
        rng = np.random.default_rng(seed * 3_333_331 + index * 10_007 + offset)
        chars, critical = build_sort_stream(rng, length=length)
        ids = torch.tensor([stoi[ch] for ch in chars], dtype=torch.long)
        crit = torch.tensor(critical, dtype=torch.bool)
        xs.append(ids[:-1])
        ys.append(ids[1:])
        masks.append(crit[1:])
    return ww.Batch(
        torch.stack(xs).to(ww.DEVICE),
        torch.stack(ys).to(ww.DEVICE),
        torch.stack(masks).to(ww.DEVICE),
    )


def make_fixed_sort_batches(
    stoi: Dict[str, int],
    seed: int,
    num_batches: int,
    batch_size: int,
    length: int = SORT_EVAL_LEN,
) -> List[ww.Batch]:
    return [make_sort_batch(stoi, seed, index, batch_size=batch_size, length=length) for index in range(num_batches)]


def sort_length_for_step(step: int, total_steps: int = ww.PHASE_B_STEPS) -> int:
    if len(SORT_ATTACH_STAGES) == 1:
        return SORT_ATTACH_STAGES[0]
    progress = step / max(total_steps, 1)
    if progress < 0.25:
        return SORT_ATTACH_STAGES[0]
    if progress < 0.65:
        return SORT_ATTACH_STAGES[1]
    return SORT_ATTACH_STAGES[2]


def sort_weighted_loss(logits: torch.Tensor, batch: ww.Batch) -> torch.Tensor:
    if batch.critical_mask is None:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.y.reshape(-1))
    token_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.y.reshape(-1), reduction="none").view_as(batch.y)
    weights = torch.where(
        batch.critical_mask,
        torch.full_like(token_loss, SORT_ANSWER_WEIGHT),
        torch.full_like(token_loss, SORT_CONTEXT_WEIGHT),
    )
    return (token_loss * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_sort(model: ww.TinyGPT, batches: List[ww.Batch]) -> Dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    answer_correct = 0
    answer_total = 0
    problem_correct = 0
    problem_total = 0
    seq_correct = 0
    seq_total = 0
    for batch in batches:
        logits, loss = model(batch.x, batch.y)
        losses.append(float(loss.item()))
        preds = logits.argmax(dim=-1)
        correct += int((preds == batch.y).sum().item())
        total += int(batch.y.numel())
        assert batch.critical_mask is not None
        hits = (preds == batch.y) & batch.critical_mask
        answer_correct += int(hits.sum().item())
        answer_total += int(batch.critical_mask.sum().item())
        for row_hits, row_mask in zip(hits, batch.critical_mask):
            start = None
            for idx, flagged in enumerate(row_mask.tolist()):
                if flagged and start is None:
                    start = idx
                elif not flagged and start is not None:
                    segment = row_hits[start:idx]
                    problem_correct += int(bool(segment.all().item()))
                    problem_total += 1
                    start = None
            if start is not None:
                segment = row_hits[start:]
                problem_correct += int(bool(segment.all().item()))
                problem_total += 1
        seq_ok = ((preds == batch.y) | (~batch.critical_mask)).all(dim=1)
        seq_correct += int(seq_ok.sum().item())
        seq_total += int(seq_ok.numel())
    model.train()
    return {
        "loss": float(np.mean(losses)),
        "acc": correct / max(total, 1),
        "answer_acc": answer_correct / max(answer_total, 1),
        "problem_acc": problem_correct / max(problem_total, 1),
        "seq_acc": seq_correct / max(seq_total, 1),
    }


def make_sort_task(
    stoi: Dict[str, int],
    sort_eval_batches: List[ww.Batch],
    seed: int,
) -> phase.TaskSpec:
    def evaluate(model: ww.TinyGPT) -> Dict[str, float]:
        metrics = evaluate_sort(model, sort_eval_batches)
        return {
            "arith_loss": metrics["loss"],
            "arith_acc": metrics["answer_acc"],
            "arith_problem_acc": metrics["problem_acc"],
            "arith_seq": metrics["seq_acc"],
            "sort_loss": metrics["loss"],
            "sort_acc": metrics["answer_acc"],
            "sort_problem_acc": metrics["problem_acc"],
            "sort_seq": metrics["seq_acc"],
        }

    return phase.TaskSpec(
        name="arith",
        sample_train_batch=lambda step: make_sort_batch(
            stoi, seed + 51_000, step, length=sort_length_for_step(step)
        ),
        sample_anchor_batch=lambda index: make_sort_batch(
            stoi, seed + 52_000, index, length=SORT_EVAL_LEN
        ),
        evaluate=evaluate,
        primary_key="arith_problem_acc",
        goal="max",
        sample_consolidation_batch=lambda step: make_sort_batch(
            stoi, seed + 53_000, step, length=SORT_EVAL_LEN
        ),
        loss_fn=sort_weighted_loss,
    )


def make_reverse_probe_task(reverse_eval_batches: List[ww.Batch]) -> phase.TaskSpec:
    def evaluate(model: ww.TinyGPT) -> Dict[str, float]:
        metrics = phase.evaluate_arithmetic(model, reverse_eval_batches)
        return {
            "reverse_loss": metrics["loss"],
            "reverse_acc": metrics["answer_acc"],
            "reverse_problem_acc": metrics["problem_acc"],
            "reverse_seq": metrics["seq_acc"],
        }

    sample = reverse_eval_batches[0]
    return phase.TaskSpec(
        name="reverse_probe",
        sample_train_batch=lambda _step: sample,
        sample_anchor_batch=lambda _index: sample,
        evaluate=evaluate,
        primary_key="reverse_problem_acc",
        goal="max",
    )


def evaluate_checkpoint(
    checkpoint: Dict[str, object],
    vocab_size: int,
    tasks: Sequence[phase.TaskSpec],
    *,
    adapters_enabled: bool = False,
    latent_projectors: Dict[str, torch.Tensor] | None = None,
    latent_strength: float = 1.0,
) -> Dict[str, float]:
    model, _optimizer = phase.restore_phase_checkpoint(vocab_size, checkpoint, load_optimizer=False)
    if adapters_enabled:
        phase.set_model_probe_mode(model, True, latent_projectors, latent_strength)
    else:
        phase.set_model_base_only(model)
    metrics = phase.evaluate_world(model, tasks)
    del model, _optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def merged_metrics(*parts: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in parts:
        out.update(part)
    return out


def print_stage(label: str, metrics: Dict[str, float]) -> None:
    print(
        f"{label:28s} "
        f"{metrics.get('bracket_seq', float('nan')):11.3f} "
        f"{metrics.get('text_loss', float('nan')):10.3f} "
        f"{metrics.get('reverse_problem_acc', float('nan')):10.3f} "
        f"{metrics.get('sort_problem_acc', float('nan')):10.3f}"
    )


def summarize_sort_metrics(metrics: Dict[str, float]) -> str:
    parts = []
    if "bracket_seq" in metrics:
        parts.append(f"bracket_seq={metrics['bracket_seq']:.3f}")
    if "text_loss" in metrics:
        parts.append(f"text_loss={metrics['text_loss']:.3f}")
    if "reverse_problem_acc" in metrics:
        parts.append(f"reverse_prob={metrics['reverse_problem_acc']:.3f}")
    if "reverse_acc" in metrics:
        parts.append(f"reverse_acc={metrics['reverse_acc']:.3f}")
    if "sort_problem_acc" in metrics:
        parts.append(f"sort_prob={metrics['sort_problem_acc']:.3f}")
    if "sort_acc" in metrics:
        parts.append(f"sort_acc={metrics['sort_acc']:.3f}")
    return " ".join(parts)


def reverse_retention_floor(retention_reference: Dict[str, float]) -> float:
    reference = float(retention_reference.get("reverse_problem_acc", 0.0))
    if reference <= 0.0:
        return 0.0
    return max(0.0, min(reference * SORT_TRANSFER_REVERSE_FRACTION, reference - SORT_TRANSFER_REVERSE_ABS_SLACK))


def reverse_retention_ratio(metrics: Dict[str, float], retention_reference: Dict[str, float]) -> float:
    reference = float(retention_reference.get("reverse_problem_acc", 0.0))
    current = float(metrics.get("reverse_problem_acc", 0.0))
    if reference <= 1e-9:
        return current
    return max(0.0, min(1.0, current / reference))


def needs_sort_transfer_polish(
    metrics: Dict[str, float],
    teacher_reference: Dict[str, float],
    retention_reference: Dict[str, float],
    keep_reverse: bool,
) -> bool:
    teacher_prob = float(teacher_reference.get("sort_problem_acc", teacher_reference.get("arith_problem_acc", 0.0)))
    teacher_acc = float(teacher_reference.get("sort_acc", teacher_reference.get("arith_acc", 0.0)))
    new_task_weak = False
    if teacher_prob > 0.0 or teacher_acc > 0.0:
        new_task_weak = (
            float(metrics.get("sort_problem_acc", metrics.get("arith_problem_acc", 0.0)))
            < teacher_prob * SORT_TRANSFER_TARGET_FRACTION
            or float(metrics.get("sort_acc", metrics.get("arith_acc", 0.0)))
            < teacher_acc * SORT_TRANSFER_TARGET_FRACTION
        )
    reverse_weak = keep_reverse and (
        float(metrics.get("reverse_problem_acc", 0.0)) < reverse_retention_floor(retention_reference)
    )
    return new_task_weak or reverse_weak or phase.retention_is_weak(metrics, retention_reference)


def sort_transfer_candidate_ok(
    metrics: Dict[str, float],
    retention_reference: Dict[str, float],
    keep_reverse: bool,
) -> bool:
    bracket_floor = float(retention_reference.get("bracket_seq", 0.0)) - SORT_TRANSFER_BRACKET_SLACK
    text_ceiling = float(retention_reference.get("text_loss", float("inf"))) + SORT_TRANSFER_TEXT_SLACK
    if float(metrics.get("bracket_seq", 0.0)) < bracket_floor:
        return False
    if float(metrics.get("text_loss", float("inf"))) > text_ceiling:
        return False
    if keep_reverse and float(metrics.get("reverse_problem_acc", 0.0)) < reverse_retention_floor(retention_reference):
        return False
    return True


def sort_teacher_candidate_ok(metrics: Dict[str, float], retention_reference: Dict[str, float]) -> bool:
    bracket_floor = float(retention_reference.get("bracket_seq", 0.0)) - SORT_TEACHER_BRACKET_SLACK
    text_ceiling = float(retention_reference.get("text_loss", float("inf"))) + SORT_TEACHER_TEXT_SLACK
    if float(metrics.get("bracket_seq", 0.0)) < bracket_floor:
        return False
    if float(metrics.get("text_loss", float("inf"))) > text_ceiling:
        return False
    if float(metrics.get("reverse_problem_acc", 0.0)) < reverse_retention_floor(retention_reference):
        return False
    return True


def sort_transfer_guard_candidate_ok(
    metrics: Dict[str, float],
    retention_reference: Dict[str, float],
    keep_reverse: bool,
) -> bool:
    bracket_floor = max(
        SORT_TRANSFER_RAW_MIN_BRACKET,
        float(retention_reference.get("bracket_seq", 0.0)) - SORT_TRANSFER_RAW_BRACKET_SLACK,
    )
    text_ceiling = float(retention_reference.get("text_loss", float("inf"))) + SORT_TRANSFER_RAW_TEXT_SLACK
    if float(metrics.get("bracket_seq", 0.0)) < bracket_floor:
        return False
    if float(metrics.get("text_loss", float("inf"))) > text_ceiling:
        return False
    if keep_reverse and reverse_retention_ratio(metrics, retention_reference) < SORT_TRANSFER_MIN_REVERSE_RATIO:
        return False
    return True


def choose_sort_teacher_stage(
    raw_stage: phase.StageResult,
    safe_stage: phase.StageResult,
    retention_reference: Dict[str, float],
) -> phase.StageResult:
    raw_metrics = raw_stage.metrics
    safe_metrics = safe_stage.metrics
    raw_sort = float(raw_metrics.get("sort_problem_acc", raw_metrics.get("arith_problem_acc", 0.0)))
    safe_sort = float(safe_metrics.get("sort_problem_acc", safe_metrics.get("arith_problem_acc", 0.0)))
    raw_rev = reverse_retention_ratio(raw_metrics, retention_reference)
    safe_rev = reverse_retention_ratio(safe_metrics, retention_reference)
    raw_joint = min(raw_sort, raw_rev)
    safe_joint = min(safe_sort, safe_rev)
    raw_ok = sort_teacher_candidate_ok(raw_metrics, retention_reference)
    safe_ok = sort_teacher_candidate_ok(safe_metrics, retention_reference)
    raw_retention_ok = raw_rev >= SORT_TEACHER_MIN_REVERSE_RATIO
    if raw_retention_ok and raw_sort >= safe_sort + 0.10:
        return raw_stage
    if raw_retention_ok and raw_joint >= safe_joint + 0.05:
        return raw_stage
    if safe_ok and safe_joint >= raw_joint - 0.01:
        return safe_stage
    if raw_ok and raw_joint >= safe_joint + 0.02:
        return raw_stage
    if safe_ok:
        return safe_stage
    return raw_stage


def choose_sort_transfer_stage(
    raw_stage: phase.StageResult,
    guard_stage: phase.StageResult | None,
    safe_stage: phase.StageResult | None,
    retention_reference: Dict[str, float],
    keep_reverse: bool,
) -> phase.StageResult:
    if safe_stage is None and guard_stage is None:
        return raw_stage
    if guard_stage is None:
        return safe_stage if safe_stage is not None else raw_stage
    if safe_stage is None:
        return guard_stage

    raw_metrics = guard_stage.metrics
    safe_metrics = safe_stage.metrics
    raw_sort = float(raw_metrics.get("sort_problem_acc", raw_metrics.get("arith_problem_acc", 0.0)))
    safe_sort = float(safe_metrics.get("sort_problem_acc", safe_metrics.get("arith_problem_acc", 0.0)))

    if not keep_reverse:
        return safe_stage if safe_sort >= raw_sort - 0.01 else raw_stage

    raw_rev = reverse_retention_ratio(raw_metrics, retention_reference)
    safe_rev = reverse_retention_ratio(safe_metrics, retention_reference)
    raw_joint = min(raw_sort, raw_rev)
    safe_joint = min(safe_sort, safe_rev)
    raw_retention_ok = raw_rev >= SORT_TRANSFER_MIN_REVERSE_RATIO

    if raw_retention_ok and safe_sort < SORT_TRANSFER_SAFE_SORT_FLOOR and raw_sort > safe_sort + 0.05:
        return guard_stage
    if raw_retention_ok and raw_sort >= safe_sort + SORT_TRANSFER_RAW_SORT_ADVANTAGE:
        return guard_stage
    if raw_retention_ok and raw_joint >= safe_joint + SORT_TRANSFER_RAW_JOINT_ADVANTAGE:
        return guard_stage
    if safe_joint >= raw_joint - 0.01 and safe_sort >= raw_sort - 0.05:
        return safe_stage
    if raw_retention_ok and raw_sort > safe_sort + 1e-12:
        return guard_stage
    return safe_stage


def annotate_sort_transfer_candidates(
    selected_stage: phase.StageResult,
    raw_stage: phase.StageResult,
    guard_stage: phase.StageResult | None,
    safe_stage: phase.StageResult | None,
) -> phase.StageResult:
    checkpoint = dict(selected_stage.checkpoint)
    checkpoint[SORT_TRANSFER_CANDIDATE_METRICS_KEY] = {
        "selected": dict(selected_stage.metrics),
        "raw": dict(raw_stage.metrics),
        "guard": dict(guard_stage.metrics) if guard_stage is not None else None,
        "safe": dict(safe_stage.metrics) if safe_stage is not None else None,
    }
    selected_stage.checkpoint = checkpoint
    return selected_stage


def better_sort_teacher_raw_candidate(
    candidate: Dict[str, float],
    incumbent: Dict[str, float] | None,
) -> bool:
    if incumbent is None:
        return True
    cand_sort = float(candidate.get("sort_problem_acc", candidate.get("arith_problem_acc", 0.0)))
    inc_sort = float(incumbent.get("sort_problem_acc", incumbent.get("arith_problem_acc", 0.0)))
    if cand_sort > inc_sort + 1e-12:
        return True
    if inc_sort > cand_sort + 1e-12:
        return False
    cand_rev = float(candidate.get("reverse_problem_acc", 0.0))
    inc_rev = float(incumbent.get("reverse_problem_acc", 0.0))
    if cand_rev > inc_rev + 1e-12:
        return True
    if inc_rev > cand_rev + 1e-12:
        return False
    cand_text = float(candidate.get("text_loss", float("inf")))
    inc_text = float(incumbent.get("text_loss", float("inf")))
    if cand_text < inc_text - 1e-12:
        return True
    if inc_text < cand_text - 1e-12:
        return False
    return float(candidate.get("bracket_seq", 0.0)) > float(incumbent.get("bracket_seq", 0.0)) + 1e-12


def better_sort_transfer_candidate(
    candidate: Dict[str, float],
    incumbent: Dict[str, float] | None,
    keep_reverse: bool,
    retention_reference: Dict[str, float],
) -> bool:
    if incumbent is None:
        return True
    cand_sort = float(candidate.get("sort_problem_acc", candidate.get("arith_problem_acc", 0.0)))
    inc_sort = float(incumbent.get("sort_problem_acc", incumbent.get("arith_problem_acc", 0.0)))
    if keep_reverse:
        cand_rev = reverse_retention_ratio(candidate, retention_reference)
        inc_rev = reverse_retention_ratio(incumbent, retention_reference)
        cand_joint = min(cand_sort, cand_rev)
        inc_joint = min(inc_sort, inc_rev)
        if cand_joint > inc_joint + 1e-12:
            return True
        if inc_joint > cand_joint + 1e-12:
            return False
        cand_pair = 0.5 * (cand_sort + cand_rev)
        inc_pair = 0.5 * (inc_sort + inc_rev)
        if cand_pair > inc_pair + 1e-12:
            return True
        if inc_pair > cand_pair + 1e-12:
            return False
    else:
        if cand_sort > inc_sort + 1e-12:
            return True
        if inc_sort > cand_sort + 1e-12:
            return False
    cand_acc = float(candidate.get("sort_acc", candidate.get("arith_acc", 0.0)))
    inc_acc = float(incumbent.get("sort_acc", incumbent.get("arith_acc", 0.0)))
    if cand_acc > inc_acc + 1e-12:
        return True
    if inc_acc > cand_acc + 1e-12:
        return False
    cand_text = float(candidate.get("text_loss", float("inf")))
    inc_text = float(incumbent.get("text_loss", float("inf")))
    if cand_text < inc_text - 1e-12:
        return True
    if inc_text < cand_text - 1e-12:
        return False
    return float(candidate.get("bracket_seq", 0.0)) > float(incumbent.get("bracket_seq", 0.0)) + 1e-12


def pick_compositional_retention_task(
    bracket_task: phase.TaskSpec,
    text_task: phase.TaskSpec,
    reverse_task: phase.TaskSpec | None,
    cycle_index: int,
    metrics: Dict[str, float],
    retention_reference: Dict[str, float],
) -> phase.TaskSpec:
    bracket_gap = float(retention_reference.get("bracket_seq", 0.0)) - float(metrics.get("bracket_seq", 0.0))
    text_gap = float(metrics.get("text_loss", float("inf"))) - float(retention_reference.get("text_loss", float("inf")))
    reverse_gap = float("-inf")
    if reverse_task is not None:
        reverse_gap = reverse_retention_floor(retention_reference) - float(metrics.get("reverse_problem_acc", 0.0))
    candidates = [
        ("bracket", bracket_gap, bracket_task),
        ("text", text_gap, text_task),
    ]
    if reverse_task is not None:
        candidates.append(("reverse", reverse_gap, reverse_task))
    weak = max(candidates, key=lambda item: item[1])
    if weak[1] > 1e-6:
        return weak[2]
    cycle = [bracket_task, text_task] + ([reverse_task] if reverse_task is not None else [])
    return cycle[cycle_index % len(cycle)]


def use_light_sort_transfer_retention(
    metrics: Dict[str, float],
    keep_reverse: bool,
) -> bool:
    if not keep_reverse:
        return True
    return (
        float(metrics.get("sort_problem_acc", metrics.get("arith_problem_acc", 0.0)))
        >= SORT_TRANSFER_BALANCED_SORT_FLOOR
        and float(metrics.get("reverse_problem_acc", 0.0)) >= SORT_TRANSFER_BALANCED_REVERSE_FLOOR
    )


def polish_sort_transfer(
    label: str,
    initial_stage: phase.StageResult,
    old_teacher_checkpoint: Dict[str, object],
    vocab_size: int,
    bracket_task: phase.TaskSpec,
    text_task: phase.TaskSpec,
    reverse_task: phase.TaskSpec | None,
    sort_task: phase.TaskSpec,
    eval_tasks: Sequence[phase.TaskSpec],
    teacher_checkpoint: Dict[str, object],
    teacher_projectors: Dict[str, torch.Tensor],
    teacher_reference: Dict[str, float],
    retention_reference: Dict[str, float],
    keep_reverse: bool,
) -> phase.StageResult:
    if not needs_sort_transfer_polish(initial_stage.metrics, teacher_reference, retention_reference, keep_reverse):
        return phase.StageResult(
            label=label,
            checkpoint=initial_stage.checkpoint,
            metrics=dict(initial_stage.metrics),
            old_batch_count=0,
            old_batch_budget=0,
            base_only_verified=True,
        )

    new_teacher_model, _new_teacher_opt = phase.restore_phase_checkpoint(vocab_size, teacher_checkpoint, load_optimizer=False)
    lateral.set_teacher_mode(new_teacher_model, teacher_projectors, latent_strength=1.0)
    ww.set_requires_grad(new_teacher_model.parameters(), False)

    old_teacher_model, _old_teacher_opt = phase.restore_phase_checkpoint(vocab_size, old_teacher_checkpoint, load_optimizer=False)
    phase.set_model_base_only(old_teacher_model)
    ww.set_requires_grad(old_teacher_model.parameters(), False)

    student, _student_opt = phase.restore_phase_checkpoint(vocab_size, initial_stage.checkpoint, load_optimizer=False)
    lateral.set_student_base_only(student, trainable_base=True)
    optimizer = lateral.make_optimizer_for_base(student, lateral.CONSOLIDATION_LR * phase.ARITH_TRANSFER_POLISH_LR_SCALE)

    light_retention = keep_reverse and use_light_sort_transfer_retention(initial_stage.metrics, keep_reverse)
    old_period = phase.ARITH_TRANSFER_POLISH_OLD_PERIOD
    if keep_reverse and not light_retention:
        old_period = phase.ARITH_TRANSFER_RETENTION_OLD_PERIOD
    elif phase.retention_is_weak(initial_stage.metrics, retention_reference):
        old_period = phase.ARITH_TRANSFER_RETENTION_OLD_PERIOD
    aux_reverse_period = SORT_TRANSFER_BALANCED_AUX_REVERSE_PERIOD if light_retention else SORT_TRANSFER_AUX_REVERSE_PERIOD
    aux_reverse_scale = SORT_TRANSFER_BALANCED_AUX_SCALE if light_retention else 1.0
    old_batch_budget = sum(1 for step in range(1, SORT_TRANSFER_POLISH_STEPS + 1) if step % old_period == 0)
    old_batch_count = 0
    final_metrics = phase.evaluate_world(student, eval_tasks)
    raw_best_metrics = dict(initial_stage.metrics)
    raw_best_checkpoint = initial_stage.checkpoint
    guard_best_metrics = (
        dict(initial_stage.metrics)
        if sort_transfer_guard_candidate_ok(initial_stage.metrics, retention_reference, keep_reverse)
        else None
    )
    guard_best_checkpoint = initial_stage.checkpoint
    safe_best_metrics = (
        dict(initial_stage.metrics)
        if sort_transfer_candidate_ok(initial_stage.metrics, retention_reference, keep_reverse)
        else None
    )
    safe_best_checkpoint = initial_stage.checkpoint

    print(f"[transfer:{label}] base-only sort transfer polish for {SORT_TRANSFER_POLISH_STEPS} steps")
    for step in range(1, SORT_TRANSFER_POLISH_STEPS + 1):
        old_step = step % old_period == 0
        if old_step:
            current_task = pick_compositional_retention_task(
                bracket_task,
                text_task,
                reverse_task if keep_reverse else None,
                old_batch_count,
                final_metrics,
                retention_reference,
            )
            batch = current_task.sample_anchor_batch(900_000 + old_batch_count)
            teacher_model = old_teacher_model
            old_batch_count += 1
            boost = phase.ARITH_TRANSFER_POLISH_RETENTION_BOOST if phase.retention_is_weak(final_metrics, retention_reference) else 1.0
            if keep_reverse and current_task is reverse_task:
                boost = max(
                    boost,
                    SORT_TRANSFER_BALANCED_REVERSE_BOOST if light_retention else 1.35,
                )
            task_weight = phase.ARITH_TRANSFER_POLISH_OLD_TASK_WEIGHT * boost
            kl_weight = phase.ARITH_TRANSFER_POLISH_OLD_KL_WEIGHT * boost
            hidden_weight = phase.ARITH_TRANSFER_POLISH_OLD_HIDDEN_WEIGHT * boost
        else:
            current_task = sort_task
            batch_sampler = sort_task.sample_consolidation_batch or sort_task.sample_train_batch
            batch = batch_sampler(900_000 + step)
            teacher_model = new_teacher_model
            lateral.set_teacher_mode(teacher_model, teacher_projectors, latent_strength=1.0)
            task_weight = phase.ARITH_TRANSFER_POLISH_TASK_WEIGHT
            kl_weight = phase.ARITH_TRANSFER_POLISH_KL_WEIGHT
            hidden_weight = phase.ARITH_TRANSFER_POLISH_HIDDEN_WEIGHT

        with torch.no_grad():
            teacher_logits, _teacher_loss, teacher_states = lateral.forward_with_block_outputs(
                teacher_model, batch, detach=True
            )
        optimizer.zero_grad(set_to_none=True)
        student_logits, _student_loss, student_states = lateral.forward_with_block_outputs(student, batch, detach=False)
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
                    old_teacher_model, reverse_batch, detach=True
                )
            reverse_student_logits, _reverse_student_loss, reverse_student_states = lateral.forward_with_block_outputs(
                student, reverse_batch, detach=False
            )
            reverse_task_loss = phase.task_loss_from_logits(reverse_task, reverse_student_logits, reverse_batch)
            loss = (
                loss
                + (SORT_TRANSFER_AUX_REVERSE_TASK_WEIGHT * aux_reverse_scale) * reverse_task_loss
                + (SORT_TRANSFER_AUX_REVERSE_KL_WEIGHT * aux_reverse_scale)
                * lateral.distill_kl(reverse_student_logits, reverse_teacher_logits)
                + (SORT_TRANSFER_AUX_REVERSE_HIDDEN_WEIGHT * aux_reverse_scale)
                * lateral.hidden_lateral_loss(reverse_student_states, reverse_teacher_states)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ww.all_base_params(student), ww.GRAD_CLIP)
        optimizer.step()

        if step % SORT_TRANSFER_EVAL_INTERVAL == 0 or step == SORT_TRANSFER_POLISH_STEPS:
            lateral.set_student_base_only(student, trainable_base=False)
            final_metrics = phase.evaluate_world(student, eval_tasks)
            if step % SORT_TRANSFER_LOG_INTERVAL == 0 or step == SORT_TRANSFER_POLISH_STEPS:
                print(
                    f"[transfer:{label}] step={step:04d}/{SORT_TRANSFER_POLISH_STEPS} "
                    f"{summarize_sort_metrics(final_metrics)} old_batches={old_batch_count}/{old_batch_budget}"
                )
            if better_sort_transfer_candidate(final_metrics, raw_best_metrics, keep_reverse, retention_reference):
                raw_best_metrics = dict(final_metrics)
                raw_best_checkpoint = phase.make_phase_checkpoint(student, optimizer)
            if (
                sort_transfer_guard_candidate_ok(final_metrics, retention_reference, keep_reverse)
                and better_sort_transfer_candidate(final_metrics, guard_best_metrics, keep_reverse, retention_reference)
            ):
                guard_best_metrics = dict(final_metrics)
                guard_best_checkpoint = phase.make_phase_checkpoint(student, optimizer)
            if (
                sort_transfer_candidate_ok(final_metrics, retention_reference, keep_reverse)
                and better_sort_transfer_candidate(final_metrics, safe_best_metrics, keep_reverse, retention_reference)
            ):
                safe_best_metrics = dict(final_metrics)
                safe_best_checkpoint = phase.make_phase_checkpoint(student, optimizer)
            lateral.set_student_base_only(student, trainable_base=True)

    lateral.set_student_base_only(student, trainable_base=False)
    final_metrics = phase.evaluate_world(student, eval_tasks)
    if better_sort_transfer_candidate(final_metrics, raw_best_metrics, keep_reverse, retention_reference):
        raw_best_metrics = dict(final_metrics)
        raw_best_checkpoint = phase.make_phase_checkpoint(student, optimizer)
    if (
        sort_transfer_guard_candidate_ok(final_metrics, retention_reference, keep_reverse)
        and better_sort_transfer_candidate(final_metrics, guard_best_metrics, keep_reverse, retention_reference)
    ):
        guard_best_metrics = dict(final_metrics)
        guard_best_checkpoint = phase.make_phase_checkpoint(student, optimizer)
    if (
        sort_transfer_candidate_ok(final_metrics, retention_reference, keep_reverse)
        and better_sort_transfer_candidate(final_metrics, safe_best_metrics, keep_reverse, retention_reference)
    ):
        safe_best_metrics = dict(final_metrics)
        safe_best_checkpoint = phase.make_phase_checkpoint(student, optimizer)

    del (
        new_teacher_model,
        _new_teacher_opt,
        old_teacher_model,
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
    selected_stage = choose_sort_transfer_stage(raw_stage, guard_stage, safe_stage, retention_reference, keep_reverse)
    return annotate_sort_transfer_candidates(selected_stage, raw_stage, guard_stage, safe_stage)


def consolidate_sort_transfer(
    label: str,
    branch_name: str,
    student_checkpoint: Dict[str, object],
    old_anchor: phase.AnchorBundle,
    old_teacher_checkpoint: Dict[str, object],
    old_tasks: Sequence[phase.TaskSpec],
    new_teacher_checkpoint: Dict[str, object],
    new_teacher_projectors: Dict[str, torch.Tensor],
    bracket_task: phase.TaskSpec,
    text_task: phase.TaskSpec,
    reverse_task: phase.TaskSpec | None,
    sort_task: phase.TaskSpec,
    eval_tasks: Sequence[phase.TaskSpec],
    vocab_size: int,
    selection_reference: Dict[str, float],
    keep_reverse: bool,
) -> phase.StageResult:
    new_teacher_model, _new_teacher_opt = phase.restore_phase_checkpoint(
        vocab_size, new_teacher_checkpoint, load_optimizer=False
    )
    lateral.set_teacher_mode(new_teacher_model, new_teacher_projectors, latent_strength=1.0)
    ww.set_requires_grad(new_teacher_model.parameters(), False)

    old_teacher_model, _old_teacher_opt = phase.restore_phase_checkpoint(
        vocab_size, old_teacher_checkpoint, load_optimizer=False
    )
    phase.set_model_base_only(old_teacher_model)
    ww.set_requires_grad(old_teacher_model.parameters(), False)

    student, _student_opt = phase.restore_phase_checkpoint(vocab_size, student_checkpoint, load_optimizer=False)
    lateral.set_student_base_only(student, trainable_base=True)
    optimizer = lateral.make_optimizer_for_base(student, lateral.CONSOLIDATION_LR)

    old_batch_budget = phase.consolidation_expected_old_batches(branch_name)
    old_batch_count = 0
    final_metrics = phase.evaluate_world(student, eval_tasks)
    raw_best_metrics = dict(final_metrics)
    raw_best_checkpoint = phase.make_phase_checkpoint(student, optimizer)
    guard_best_metrics = (
        dict(final_metrics)
        if sort_transfer_guard_candidate_ok(final_metrics, selection_reference, keep_reverse)
        else None
    )
    guard_best_checkpoint = raw_best_checkpoint
    safe_best_metrics = (
        dict(final_metrics)
        if sort_transfer_candidate_ok(final_metrics, selection_reference, keep_reverse)
        else None
    )
    safe_best_checkpoint = raw_best_checkpoint

    print(f"[consolidate:{label}] branch={branch_name} for {lateral.CONSOLIDATION_STEPS} steps")
    for step in range(1, lateral.CONSOLIDATION_STEPS + 1):
        old_step = phase.consolidation_old_batch_schedule(branch_name, step)
        if old_step:
            current_task = pick_compositional_retention_task(
                bracket_task,
                text_task,
                reverse_task if keep_reverse else None,
                old_batch_count,
                final_metrics,
                selection_reference,
            )
            batch = current_task.sample_anchor_batch(old_batch_count)
            teacher_model = old_teacher_model
            old_batch_count += 1
        else:
            current_task = sort_task
            batch_sampler = sort_task.sample_consolidation_batch or sort_task.sample_train_batch
            batch = batch_sampler(step)
            teacher_model = new_teacher_model
            lateral.set_teacher_mode(teacher_model, new_teacher_projectors, latent_strength=1.0)

        with torch.no_grad():
            teacher_logits, _teacher_loss, teacher_states = lateral.forward_with_block_outputs(
                teacher_model, batch, detach=True
            )

        optimizer.zero_grad(set_to_none=True)
        student_logits, _student_loss, student_states = lateral.forward_with_block_outputs(student, batch, detach=False)
        task_loss = phase.task_loss_from_logits(current_task, student_logits, batch)
        task_weight, kl_weight, hidden_weight = phase.consolidation_weights_for_step(branch_name, old_step, step)
        loss = (
            task_weight * task_loss
            + kl_weight * lateral.distill_kl(student_logits, teacher_logits)
            + hidden_weight * lateral.hidden_lateral_loss(student_states, teacher_states)
        )
        light_retention = keep_reverse and use_light_sort_transfer_retention(final_metrics, keep_reverse)
        aux_reverse_period = SORT_TRANSFER_BALANCED_AUX_REVERSE_PERIOD if light_retention else SORT_TRANSFER_AUX_REVERSE_PERIOD
        aux_reverse_scale = SORT_TRANSFER_BALANCED_AUX_SCALE if light_retention else 1.0
        if keep_reverse and reverse_task is not None and not old_step and step % aux_reverse_period == 0:
            reverse_batch = reverse_task.sample_anchor_batch(975_000 + step)
            with torch.no_grad():
                reverse_teacher_logits, _reverse_teacher_loss, reverse_teacher_states = lateral.forward_with_block_outputs(
                    old_teacher_model, reverse_batch, detach=True
                )
            reverse_student_logits, _reverse_student_loss, reverse_student_states = lateral.forward_with_block_outputs(
                student, reverse_batch, detach=False
            )
            reverse_task_loss = phase.task_loss_from_logits(reverse_task, reverse_student_logits, reverse_batch)
            loss = (
                loss
                + (SORT_TRANSFER_AUX_REVERSE_TASK_WEIGHT * aux_reverse_scale) * reverse_task_loss
                + (SORT_TRANSFER_AUX_REVERSE_KL_WEIGHT * aux_reverse_scale)
                * lateral.distill_kl(reverse_student_logits, reverse_teacher_logits)
                + (SORT_TRANSFER_AUX_REVERSE_HIDDEN_WEIGHT * aux_reverse_scale)
                * lateral.hidden_lateral_loss(reverse_student_states, reverse_teacher_states)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ww.all_base_params(student), ww.GRAD_CLIP)
        optimizer.step()

        if step % SORT_TRANSFER_EVAL_INTERVAL == 0 or step == lateral.CONSOLIDATION_STEPS:
            lateral.set_student_base_only(student, trainable_base=False)
            final_metrics = phase.evaluate_world(student, eval_tasks)
            if step % SORT_TRANSFER_LOG_INTERVAL == 0 or step == lateral.CONSOLIDATION_STEPS:
                print(
                    f"[consolidate:{label}] step={step:04d}/{lateral.CONSOLIDATION_STEPS} "
                    f"{summarize_sort_metrics(final_metrics)} "
                    f"old_batches={old_batch_count}/{old_batch_budget} "
                    f"viscosity={ww.INITIAL_Z_VISCOSITY:.3f}"
                )
            if better_sort_transfer_candidate(final_metrics, raw_best_metrics, keep_reverse, selection_reference):
                raw_best_metrics = dict(final_metrics)
                raw_best_checkpoint = phase.make_phase_checkpoint(student, optimizer)
            if (
                sort_transfer_guard_candidate_ok(final_metrics, selection_reference, keep_reverse)
                and better_sort_transfer_candidate(final_metrics, guard_best_metrics, keep_reverse, selection_reference)
            ):
                guard_best_metrics = dict(final_metrics)
                guard_best_checkpoint = phase.make_phase_checkpoint(student, optimizer)
            if (
                sort_transfer_candidate_ok(final_metrics, selection_reference, keep_reverse)
                and better_sort_transfer_candidate(final_metrics, safe_best_metrics, keep_reverse, selection_reference)
            ):
                safe_best_metrics = dict(final_metrics)
                safe_best_checkpoint = phase.make_phase_checkpoint(student, optimizer)
            lateral.set_student_base_only(student, trainable_base=True)

    lateral.set_student_base_only(student, trainable_base=False)
    final_metrics = phase.evaluate_world(student, eval_tasks)
    if better_sort_transfer_candidate(final_metrics, raw_best_metrics, keep_reverse, selection_reference):
        raw_best_metrics = dict(final_metrics)
        raw_best_checkpoint = phase.make_phase_checkpoint(student, optimizer)
    if (
        sort_transfer_guard_candidate_ok(final_metrics, selection_reference, keep_reverse)
        and better_sort_transfer_candidate(final_metrics, guard_best_metrics, keep_reverse, selection_reference)
    ):
        guard_best_metrics = dict(final_metrics)
        guard_best_checkpoint = phase.make_phase_checkpoint(student, optimizer)
    if (
        sort_transfer_candidate_ok(final_metrics, selection_reference, keep_reverse)
        and better_sort_transfer_candidate(final_metrics, safe_best_metrics, keep_reverse, selection_reference)
    ):
        safe_best_metrics = dict(final_metrics)
        safe_best_checkpoint = phase.make_phase_checkpoint(student, optimizer)

    del (
        new_teacher_model,
        _new_teacher_opt,
        old_teacher_model,
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
    selected_stage = choose_sort_transfer_stage(raw_stage, guard_stage, safe_stage, selection_reference, keep_reverse)
    return annotate_sort_transfer_candidates(selected_stage, raw_stage, guard_stage, safe_stage)


def attach_sort_teacher_with_retention(
    label: str,
    anchor: phase.AnchorBundle,
    vocab_size: int,
    bracket_task: phase.TaskSpec,
    text_task: phase.TaskSpec,
    reverse_task: phase.TaskSpec,
    sort_task: phase.TaskSpec,
    eval_tasks: Sequence[phase.TaskSpec],
    retention_reference: Dict[str, float],
    seed: int,
) -> phase.StageResult:
    adapter_rank_override = phase.c_attach_adapter_rank(sort_task)
    phase_checkpoint = dict(anchor.checkpoint)
    if adapter_rank_override is not None:
        phase_checkpoint["adapter_rank_override"] = adapter_rank_override
    model, optimizer = phase.restore_phase_checkpoint(vocab_size, phase_checkpoint, load_optimizer=False)
    latent_projectors = {block: projector.to(ww.DEVICE) for block, projector in anchor.latent_free_projectors.items()}

    old_teacher_model, _old_teacher_opt = phase.restore_phase_checkpoint(vocab_size, anchor.checkpoint, load_optimizer=False)
    phase.set_model_base_only(old_teacher_model)
    ww.set_requires_grad(old_teacher_model.parameters(), False)

    print(f"[teacher:{label}] momentum reset + old-world reminiscence")
    phase.run_mixed_reminiscence(model, optimizer, [bracket_task, text_task, reverse_task], seed)
    ww.configure_branch_trainability(model, "water_weights_latent_adapter_only", anchor.old_frontier)
    optimizer = lateral.make_optimizer_for_adapter(model)

    replay_budget = int(ww.PHASE_B_STEPS * phase.replay_budget_fraction_for_task(sort_task))
    replay_count = 0
    current_latent_projection = ww.INITIAL_LATENT_PROJECTION
    latent_projection_steps = 0
    final_metrics = phase.evaluate_world(model, eval_tasks)
    raw_best_metrics = dict(final_metrics)
    raw_best_checkpoint = phase.make_phase_checkpoint(model, optimizer, adapter_rank_override=adapter_rank_override)
    safe_best_metrics = dict(final_metrics)
    safe_best_checkpoint = raw_best_checkpoint

    for step in range(1, ww.PHASE_B_STEPS + 1):
        replay_this_step = ww.should_replay("water_weights_latent_adapter_only", step, replay_count, replay_budget)
        if replay_this_step:
            current_task = pick_compositional_retention_task(
                bracket_task, text_task, reverse_task, replay_count, final_metrics, retention_reference
            )
            batch = current_task.sample_anchor_batch(replay_count)
            replay_count += 1
        else:
            current_task = sort_task
            batch = sort_task.sample_train_batch(step)

        latent_strength = ww.latent_projection_strength_for_branch(
            "water_weights_latent_adapter_only",
            current_latent_projection,
        )
        model.set_adapters_enabled(True)
        model.set_latent_free_projectors(latent_projectors, latent_strength)
        latent_projection_steps += int(latent_strength > 0.0)

        if current_task is sort_task:
            compat_task = pick_compositional_retention_task(
                bracket_task, text_task, reverse_task, step - 1, final_metrics, retention_reference
            )
            compat_batch = compat_task.sample_anchor_batch(100_000 + step)
            model.train()
            ww.set_optimizer_lrs(optimizer, ww.BASE_LR, True)
            optimizer.zero_grad(set_to_none=True)

            sort_logits, _ = model(batch.x, batch.y)
            loss = phase.task_loss_from_logits(sort_task, sort_logits, batch)

            with torch.no_grad():
                teacher_logits, _teacher_loss, teacher_states = lateral.forward_with_block_outputs(
                    old_teacher_model, compat_batch, detach=True
                )
            compat_logits, _compat_loss, compat_states = lateral.forward_with_block_outputs(
                model, compat_batch, detach=False
            )
            compat_task_loss = phase.task_loss_from_logits(compat_task, compat_logits, compat_batch)
            compat_task_w, compat_kl_w, compat_hidden_w = phase.compat_weights_for_step(step, ww.PHASE_B_STEPS)
            compat_boost = SORT_TEACHER_REVERSE_COMPAT_BOOST if compat_task is reverse_task else 1.0
            loss = (
                loss
                + (compat_task_w * compat_boost) * compat_task_loss
                + (compat_kl_w * compat_boost) * lateral.distill_kl(compat_logits, teacher_logits)
                + (compat_hidden_w * compat_boost) * lateral.hidden_lateral_loss(compat_states, teacher_states)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ww.all_adapter_params(model), ww.GRAD_CLIP)
            optimizer.step()
        else:
            phase.task_train_one_step(model, optimizer, batch, current_task, base_lr=ww.BASE_LR, adapter_fluid=True)

        if ww.is_probe_step(step):
            model.set_adapters_enabled(True)
            model.set_latent_free_projectors(latent_projectors, latent_strength)
            final_metrics = phase.evaluate_world(model, eval_tasks)
            shock = phase.compute_anchor_shock(model, anchor, [bracket_task, text_task, reverse_task])
            current_latent_projection = ww.latent_projection_from_shock(shock)
            current_latent_projection = phase.relax_projection_for_arithmetic(step, current_latent_projection, final_metrics)
            if step % ww.BRANCH_LOG_INTERVAL == 0 or step == ww.PHASE_B_STEPS:
                print(
                    f"[teacher:{label}] step={step:04d}/{ww.PHASE_B_STEPS} "
                    f"{summarize_sort_metrics(final_metrics)} z_shock={shock:.2f} "
                    f"latent_proj={latent_strength:.2f} replay={replay_count}/{replay_budget}"
                )
            if better_sort_teacher_raw_candidate(final_metrics, raw_best_metrics):
                raw_best_metrics = dict(final_metrics)
                raw_best_checkpoint = phase.make_phase_checkpoint(
                    model, optimizer, adapter_rank_override=adapter_rank_override
                )
            if (
                sort_teacher_candidate_ok(final_metrics, retention_reference)
                and better_sort_transfer_candidate(final_metrics, safe_best_metrics, True, retention_reference)
            ):
                safe_best_metrics = dict(final_metrics)
                safe_best_checkpoint = phase.make_phase_checkpoint(
                    model, optimizer, adapter_rank_override=adapter_rank_override
                )

    del model, optimizer, old_teacher_model, _old_teacher_opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_stage = phase.StageResult(
        label=label,
        checkpoint=raw_best_checkpoint,
        metrics=raw_best_metrics,
        replay_count=replay_count,
        replay_budget=replay_budget,
        latent_projection_steps=latent_projection_steps,
        base_only_verified=False,
    )
    safe_stage = phase.StageResult(
        label=label,
        checkpoint=safe_best_checkpoint,
        metrics=safe_best_metrics,
        replay_count=replay_count,
        replay_budget=replay_budget,
        latent_projection_steps=latent_projection_steps,
        base_only_verified=False,
    )
    return choose_sort_teacher_stage(raw_stage, safe_stage, retention_reference)


def sharpen_sort_teacher_with_retention(
    label: str,
    initial_stage: phase.StageResult,
    anchor: phase.AnchorBundle,
    vocab_size: int,
    bracket_task: phase.TaskSpec,
    text_task: phase.TaskSpec,
    reverse_task: phase.TaskSpec,
    sort_task: phase.TaskSpec,
    eval_tasks: Sequence[phase.TaskSpec],
    retention_reference: Dict[str, float],
) -> phase.StageResult:
    if not phase.needs_arith_sharpen(initial_stage.metrics):
        return phase.StageResult(
            label=label,
            checkpoint=initial_stage.checkpoint,
            metrics=dict(initial_stage.metrics),
            latent_projection_steps=0,
            base_only_verified=False,
        )

    model, _optimizer = phase.restore_phase_checkpoint(vocab_size, initial_stage.checkpoint, load_optimizer=False)
    ww.configure_branch_trainability(model, "water_weights_latent_adapter_only", anchor.old_frontier)
    optimizer = lateral.make_optimizer_for_adapter(model)
    latent_projectors = {block: projector.to(ww.DEVICE) for block, projector in anchor.latent_free_projectors.items()}

    old_teacher_model, _old_teacher_opt = phase.restore_phase_checkpoint(vocab_size, anchor.checkpoint, load_optimizer=False)
    phase.set_model_base_only(old_teacher_model)
    ww.set_requires_grad(old_teacher_model.parameters(), False)

    adapter_rank_override = int(initial_stage.checkpoint.get("adapter_rank_override", 0) or 0)
    current_latent_projection = phase.ARITH_SHARPEN_PROJECTION
    latent_projection_steps = 0
    final_metrics = phase.evaluate_world(model, eval_tasks)
    raw_best_metrics = dict(initial_stage.metrics)
    raw_best_checkpoint = initial_stage.checkpoint
    safe_best_metrics = dict(initial_stage.metrics)
    safe_best_checkpoint = initial_stage.checkpoint

    print(f"[sharpen:{label}] sort+reverse adapter sharpening for {phase.ARITH_SHARPEN_STEPS} steps")
    for step in range(1, phase.ARITH_SHARPEN_STEPS + 1):
        batch = sort_task.sample_train_batch(200_000 + step)
        compat_task = pick_compositional_retention_task(
            bracket_task, text_task, reverse_task, step - 1, final_metrics, retention_reference
        )
        compat_batch = compat_task.sample_anchor_batch(300_000 + step)

        model.train()
        model.set_adapters_enabled(True)
        model.set_latent_free_projectors(latent_projectors, current_latent_projection)
        latent_projection_steps += 1
        ww.set_optimizer_lrs(optimizer, ww.BASE_LR * phase.ARITH_SHARPEN_LR_SCALE, True)
        optimizer.zero_grad(set_to_none=True)

        sort_logits, _ = model(batch.x, batch.y)
        loss = phase.task_loss_from_logits(sort_task, sort_logits, batch)

        with torch.no_grad():
            teacher_logits, _teacher_loss, teacher_states = lateral.forward_with_block_outputs(
                old_teacher_model, compat_batch, detach=True
            )
        compat_logits, _compat_loss, compat_states = lateral.forward_with_block_outputs(
            model, compat_batch, detach=False
        )
        compat_task_loss = phase.task_loss_from_logits(compat_task, compat_logits, compat_batch)
        compat_boost = phase.sharpen_compat_boost(final_metrics)
        if compat_task is reverse_task:
            compat_boost *= SORT_TEACHER_REVERSE_COMPAT_BOOST
        loss = (
            loss
            + (phase.ARITH_SHARPEN_COMPAT_TASK_WEIGHT * compat_boost) * compat_task_loss
            + (phase.ARITH_SHARPEN_COMPAT_KL_WEIGHT * compat_boost) * lateral.distill_kl(compat_logits, teacher_logits)
            + (phase.ARITH_SHARPEN_COMPAT_HIDDEN_WEIGHT * compat_boost)
            * lateral.hidden_lateral_loss(compat_states, teacher_states)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ww.all_adapter_params(model), ww.GRAD_CLIP)
        optimizer.step()

        if ww.is_probe_step(step) or step == phase.ARITH_SHARPEN_STEPS:
            model.set_adapters_enabled(True)
            model.set_latent_free_projectors(latent_projectors, current_latent_projection)
            final_metrics = phase.evaluate_world(model, eval_tasks)
            if step % ww.BRANCH_LOG_INTERVAL == 0 or step == phase.ARITH_SHARPEN_STEPS:
                print(
                    f"[sharpen:{label}] step={step:04d}/{phase.ARITH_SHARPEN_STEPS} "
                    f"{summarize_sort_metrics(final_metrics)} latent_proj={current_latent_projection:.2f}"
                )
            if better_sort_teacher_raw_candidate(final_metrics, raw_best_metrics):
                raw_best_metrics = dict(final_metrics)
                raw_best_checkpoint = phase.make_phase_checkpoint(
                    model, optimizer, adapter_rank_override=adapter_rank_override or None
                )
            if (
                sort_teacher_candidate_ok(final_metrics, retention_reference)
                and better_sort_transfer_candidate(final_metrics, safe_best_metrics, True, retention_reference)
            ):
                safe_best_metrics = dict(final_metrics)
                safe_best_checkpoint = phase.make_phase_checkpoint(
                    model, optimizer, adapter_rank_override=adapter_rank_override or None
                )
                if (
                    float(final_metrics.get("sort_problem_acc", 0.0)) >= phase.ARITH_SHARPEN_TARGET_PROB
                    and float(final_metrics.get("sort_acc", 0.0)) >= phase.ARITH_SHARPEN_TARGET_ACC
                ):
                    break

    del model, _optimizer, optimizer, old_teacher_model, _old_teacher_opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_stage = phase.StageResult(
        label=label,
        checkpoint=raw_best_checkpoint,
        metrics=raw_best_metrics,
        latent_projection_steps=latent_projection_steps,
        base_only_verified=False,
    )
    safe_stage = phase.StageResult(
        label=label,
        checkpoint=safe_best_checkpoint,
        metrics=safe_best_metrics,
        latent_projection_steps=latent_projection_steps,
        base_only_verified=False,
    )
    return choose_sort_teacher_stage(raw_stage, safe_stage, retention_reference)


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


def run_seed(
    seed: int,
    vocab_size: int,
    stoi: Dict[str, int],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> List[Dict[str, object]]:
    print("\n" + "#" * 78)
    print(f"BEGIN COMPOSITIONAL TRANSFER SEED {seed}")
    print("#" * 78)
    ww.set_seed(seed)

    text_eval_positions = phase.make_text_eval_positions(len(val_data), seed)
    bracket_eval_batches = ww.make_fixed_bracket_batches(
        stoi, seed + 30_000, ww.BRACKET_EVAL_BATCHES, ww.BRACKET_EVAL_BATCH
    )
    reverse_eval_batches = phase.make_fixed_arithmetic_batches(
        stoi, seed + 31_000, ww.BRACKET_EVAL_BATCHES, ww.BRACKET_EVAL_BATCH
    )
    sort_eval_batches = make_fixed_sort_batches(
        stoi, seed + 32_000, ww.BRACKET_EVAL_BATCHES, ww.BRACKET_EVAL_BATCH
    )

    bracket_task = phase.make_bracket_task(stoi, bracket_eval_batches, seed)
    text_task = phase.make_text_task(train_data, val_data, text_eval_positions, seed)
    reverse_train_task = phase.make_arithmetic_task(stoi, reverse_eval_batches, seed)
    reverse_probe_task = make_reverse_probe_task(reverse_eval_batches)
    sort_task = make_sort_task(stoi, sort_eval_batches, seed)

    anchor = ww.train_old_skill(
        vocab_size,
        stoi,
        seed,
        bracket_eval_batches,
        ww.make_fixed_bracket_batches(stoi, seed + 40_000, 1, ww.PROBE_BATCH)[0],
    )
    anchor_a = phase.anchor_from_old_skill(anchor)
    base_a_metrics = evaluate_checkpoint(anchor_a.checkpoint, vocab_size, [bracket_task, text_task, reverse_probe_task])

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
    base_abc_metrics = evaluate_checkpoint(
        base_abc.checkpoint, vocab_size, [bracket_task, text_task, reverse_probe_task]
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
    base_abd = polish_sort_transfer(
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
        evaluate_checkpoint(base_abd.checkpoint, vocab_size, [bracket_task, text_task, sort_task]),
        evaluate_checkpoint(base_abd.checkpoint, vocab_size, [reverse_probe_task]),
    )

    anchor_abc = phase.collect_world_anchor(
        "ABC", base_abc.checkpoint, [bracket_task, text_task, reverse_train_task], vocab_size
    )
    teacher_d_abc_attach = attach_sort_teacher_with_retention(
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
    teacher_d_abc = sharpen_sort_teacher_with_retention(
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
    base_abcd_consolidated = consolidate_sort_transfer(
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
    base_abcd = polish_sort_transfer(
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
        evaluate_checkpoint(base_abcd.checkpoint, vocab_size, [bracket_task, text_task, sort_task]),
        evaluate_checkpoint(base_abcd.checkpoint, vocab_size, [reverse_probe_task]),
    )

    teacher_d_ab_metrics = merged_metrics(
        dict(teacher_d_ab.metrics),
        evaluate_checkpoint(
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
        evaluate_checkpoint(
            teacher_d_abc.checkpoint,
            vocab_size,
            [reverse_probe_task],
            adapters_enabled=True,
            latent_projectors=anchor_abc.latent_free_projectors,
            latent_strength=phase.ARITH_SHARPEN_PROJECTION,
        ),
    )

    print("\n" + "=" * 78)
    print(f"SEED {seed} COMPOSITIONAL TRANSFER RESULT")
    print("=" * 78)
    print(f"{'stage':28s} {'bracket_seq':>11s} {'text_loss':>10s} {'rev_prob':>10s} {'sort_prob':>10s}")
    print_stage("base_A_anchor", base_a_metrics)
    print_stage("base_AB_unified", base_ab.metrics)
    print_stage("reverse_text_extract", reverse_ab.metrics)
    print_stage("base_ABC_unified", base_abc_metrics)
    print_stage("teacher_D_from_AB", teacher_d_ab_metrics)
    print_stage("base_ABD_unified", base_abd_metrics)
    print_stage("teacher_D_from_ABC", teacher_d_abc_metrics)
    print_stage("base_ABCD_unified", base_abcd_metrics)
    print(
        "Compositional transfer gain: "
        f"teacher_sort_prob={teacher_d_abc_metrics.get('sort_problem_acc', 0.0) - teacher_d_ab_metrics.get('sort_problem_acc', 0.0):+.3f} "
        f"base_sort_prob={base_abcd_metrics.get('sort_problem_acc', 0.0) - base_abd_metrics.get('sort_problem_acc', 0.0):+.3f}"
    )
    print(
        "C retention through D: "
        f"reverse_prob_delta={base_abcd_metrics.get('reverse_problem_acc', 0.0) - base_abc_metrics.get('reverse_problem_acc', 0.0):+.3f} "
        f"bracket_delta={base_abcd_metrics.get('bracket_seq', 0.0) - base_abc_metrics.get('bracket_seq', 0.0):+.3f}"
    )
    print("=" * 78)

    rows: List[Dict[str, object]] = []
    for label, metrics in [
        ("base_A_anchor", base_a_metrics),
        ("base_AB_unified", base_ab.metrics),
        ("reverse_text_extract", reverse_ab.metrics),
        ("base_ABC_unified", base_abc_metrics),
        ("teacher_D_from_AB", teacher_d_ab_metrics),
        ("base_ABD_unified", base_abd_metrics),
        ("teacher_D_from_ABC", teacher_d_abc_metrics),
        ("base_ABCD_unified", base_abcd_metrics),
    ]:
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
    return rows


def main() -> None:
    ww.set_seed(LAB_SEEDS[0])
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 78)
    print("COMPOSITIONAL TRANSFER LAB")
    print("=" * 78)
    print("Question: does C improve later D transfer, and can A->B->C->D unify without collapse?")
    print(f"Device: {ww.DEVICE}")
    print(f"Seeds: {LAB_SEEDS}")
    print(f"Phase A: bracket until seq>={ww.OLD_READY_SEQ:.2f} or {ww.PHASE_A_MAX_STEPS} steps")
    print(f"Phase B: latent text attach for {ww.PHASE_B_STEPS} steps")
    print(f"Phase C: dual-teacher consolidation for {lateral.CONSOLIDATION_STEPS} steps")
    print(f"Phase D: latent reversal attach for {ww.PHASE_B_STEPS} steps")
    print(f"Phase D2: reversal sharpen for {phase.ARITH_SHARPEN_STEPS} steps")
    print(f"Phase E: dual-teacher consolidation for {lateral.CONSOLIDATION_STEPS} steps")
    print(f"Phase E2: base-only reversal transfer polish for {phase.ARITH_TRANSFER_POLISH_STEPS} steps")
    print(f"Phase F: direct-vs-composed sort attach for {ww.PHASE_B_STEPS} steps")
    print(f"Phase F2: sort sharpen for {phase.ARITH_SHARPEN_STEPS} steps")
    print(f"Phase G: final D consolidation/polish for {lateral.CONSOLIDATION_STEPS}+{phase.ARITH_TRANSFER_POLISH_STEPS} steps")
    print(
        f"Model: d={ww.D_MODEL}, layers={ww.N_LAYER}, heads={ww.N_HEAD}, "
        f"block={ww.BLOCK_SIZE}, adapter_rank={ww.ADAPTER_RANK}"
    )
    print(
        f"Reversal C: eval_len={phase.ARITH_EVAL_DIGITS}, curriculum={phase.ARITH_ATTACH_DIGIT_STAGES}; "
        f"Sort D: eval_len={SORT_EVAL_LEN}, curriculum={SORT_ATTACH_STAGES}"
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
    print(f"CSV saved to: {CSV_PATH}")
    print("=" * 78)
    print(f"Total wall time: {phase.format_seconds(time.time() - start)}")


if __name__ == "__main__":
    main()
