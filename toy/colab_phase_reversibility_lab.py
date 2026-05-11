#!/usr/bin/env python3
"""
Phase-reversibility research lab for Water Weights.

Run:
    python colab_phase_reversibility_lab.py

Hidden smoke mode:
    CHAOS_SMOKE=1 python colab_phase_reversibility_lab.py

Core question:
Can task skills be phase-separated into latent adapters, laterally
consolidated back into a base-only model, extended to a third task, and then
partially reversed with an adapter extraction pass?

Experiments in one script:
    1. A -> B attach + dual-teacher consolidation into base_AB
    2. Reverse extraction: base_AB + adapter -> base_A-like overlay
    3. A -> B -> C attach + consolidation into base_ABC
    4. Geometry: teacher-space overlap and unified-base transfer overlap
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import colab_water_weights_benchmark as ww
import colab_water_weights_lateral_consolidation_benchmark as lateral

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "phase_reversibility_lab_results.csv"
GEOM_CSV_PATH = ROOT / "phase_reversibility_geometry.csv"
PLOT_DIR = ROOT / "phase_reversibility_plots"

ARITH_CHARS = "01234567|\n"
ARITH_EVAL_DIGITS = 4 if ww.D_MODEL <= 256 else 6
ARITH_ATTACH_DIGIT_STAGES = (2, 3, ARITH_EVAL_DIGITS)
ARITH_ANSWER_WEIGHT = 6.0
ARITH_CONTEXT_WEIGHT = 0.10
ARITH_SINGLE_PROBLEM = True
ARITH_RANDOM_PAD = False
ARITH_REPLAY_BUDGET_FRACTION = 0.04
ARITH_ESCAPE_WARMUP = 250
ARITH_ESCAPE_MID = 600
ARITH_ESCAPE_LATE = 900
ARITH_ESCAPE_LEVELS = (0.65, 0.35, 0.15)
ARITH_ADAPTER_RANK = max(ww.ADAPTER_RANK * 2, 24)
ARITH_OLD_COMPAT_TASK_WEIGHT = 0.50
ARITH_OLD_COMPAT_KL_WEIGHT = 0.50
ARITH_OLD_COMPAT_HIDDEN_WEIGHT = 0.15
ARITH_OLD_COMPAT_MID_FACTOR = 0.35
ARITH_OLD_COMPAT_LATE_FACTOR = 0.10
ARITH_STRONG_ANSWER_ACC = 0.98
ARITH_MIXED_ANSWER_ACC = 0.85
ARITH_SHARPEN_STEPS = max(12, ww.PHASE_B_STEPS // 3)
ARITH_SHARPEN_TARGET_PROB = 0.30 if ARITH_EVAL_DIGITS <= 4 else 0.20
ARITH_SHARPEN_TARGET_ACC = 0.92 if ARITH_EVAL_DIGITS <= 4 else 0.85
ARITH_SHARPEN_LR_SCALE = 0.75
ARITH_SHARPEN_PROJECTION = ARITH_ESCAPE_LEVELS[-1]
ARITH_SHARPEN_COMPAT_TASK_WEIGHT = 0.10
ARITH_SHARPEN_COMPAT_KL_WEIGHT = 0.10
ARITH_SHARPEN_COMPAT_HIDDEN_WEIGHT = 0.03
ARITH_SHARPEN_COMPAT_MID_BOOST = 2.0
ARITH_SHARPEN_COMPAT_LATE_BOOST = 4.0
ARITH_SHARPEN_BRACKET_SLACK = 0.04
ARITH_SHARPEN_TEXT_SLACK = 0.10
ARITH_TRANSFER_POLISH_STEPS = max(8, lateral.CONSOLIDATION_STEPS // 3)
ARITH_TRANSFER_TARGET_FRACTION = 0.85
ARITH_TRANSFER_POLISH_LR_SCALE = 0.60
ARITH_TRANSFER_POLISH_OLD_PERIOD = 4
ARITH_TRANSFER_RETENTION_OLD_PERIOD = 2
ARITH_TRANSFER_POLISH_TASK_WEIGHT = 3.50
ARITH_TRANSFER_POLISH_KL_WEIGHT = 1.50
ARITH_TRANSFER_POLISH_HIDDEN_WEIGHT = 1.00
ARITH_TRANSFER_POLISH_OLD_TASK_WEIGHT = 2.50
ARITH_TRANSFER_POLISH_OLD_KL_WEIGHT = 0.90
ARITH_TRANSFER_POLISH_OLD_HIDDEN_WEIGHT = 0.60
ARITH_TRANSFER_POLISH_RETENTION_BOOST = 1.50
ARITH_TRANSFER_TRIGGER_BRACKET_SLACK = 0.02
ARITH_TRANSFER_TRIGGER_TEXT_SLACK = 0.04
ARITH_TRANSFER_POLISH_BRACKET_SLACK = 0.02
ARITH_TRANSFER_POLISH_TEXT_SLACK = 0.05
REVERSE_STEPS = lateral.CONSOLIDATION_STEPS
REVERSE_LOG_INTERVAL = lateral.CONSOLIDATION_LOG_INTERVAL
REVERSE_EVAL_INTERVAL = lateral.CONSOLIDATION_EVAL_INTERVAL
GEOM_BATCHES = 8 if torch.cuda.is_available() else 4
GEOM_ACT_RANK = min(ww.LATENT_ACT_RANK, 16 if torch.cuda.is_available() else 6)
GEOM_GRAD_RANK = min(ww.LATENT_GRAD_RANK, 8 if torch.cuda.is_available() else 4)
LAB_SEEDS = list(ww.SEEDS[:1])
AB_CONSOLIDATION_BRANCH = "dual_lateral_balanced"
ABC_CONSOLIDATION_BRANCH = "dual_lateral_reversal_transfer"
REVERSE_LATENT_PROJECTION = 1.0
REVERSE_TEXT_BATCH_PERIOD = 4
REVERSE_BRACKET_TASK_WEIGHT = 1.50
REVERSE_BRACKET_KL_WEIGHT = 1.00
REVERSE_BRACKET_HIDDEN_WEIGHT = 0.40
REVERSE_TEXT_KL_WEIGHT = 2.00
REVERSE_TEXT_HIDDEN_WEIGHT = 0.60
REVERSE_TEXT_LOGIT_MSE_WEIGHT = 0.25
REVERSE_SELECTION_BRACKET_SLACK = 0.01
ABC_ARITH_TRANSFER_EARLY = 0.55
ABC_ARITH_TRANSFER_LATE = 0.85
ABC_ARITH_SELECT_BRACKET_SLACK = 0.03
ABC_ARITH_SELECT_TEXT_SLACK = 0.10
ABC_ARITH_TRANSFER_BRACKET_SLACK = 0.08
ABC_ARITH_TRANSFER_TEXT_SLACK = 0.90


@dataclass
class TaskSpec:
    name: str
    sample_train_batch: Callable[[int], ww.Batch]
    sample_anchor_batch: Callable[[int], ww.Batch]
    evaluate: Callable[[ww.TinyGPT], Dict[str, float]]
    primary_key: str
    goal: str  # "max" or "min"
    sample_consolidation_batch: Callable[[int], ww.Batch] | None = None
    loss_fn: Callable[[torch.Tensor, ww.Batch], torch.Tensor] | None = None


@dataclass
class AnchorBundle:
    label: str
    checkpoint: Dict[str, object]
    old_frontier: List[str]
    block_anchor_z: Dict[str, float]
    grad_basis: Dict[str, torch.Tensor]
    latent_free_projectors: Dict[str, torch.Tensor]


@dataclass
class StageResult:
    label: str
    checkpoint: Dict[str, object]
    metrics: Dict[str, float]
    replay_count: int = 0
    replay_budget: int = 0
    z_viscosity_steps: int = 0
    latent_projection_steps: int = 0
    old_batch_count: int = 0
    old_batch_budget: int = 0
    base_only_verified: bool = False


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {sec:02d}s"
    return f"{minutes:02d}m {sec:02d}s"


def build_joint_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text + ww.BRACKET_CHARS + ARITH_CHARS))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def make_phase_checkpoint(
    model: ww.TinyGPT,
    optimizer: torch.optim.Optimizer,
    adapter_rank_override: int | None = None,
) -> Dict[str, object]:
    checkpoint = ww.make_checkpoint(model, optimizer)
    if adapter_rank_override is not None:
        checkpoint["adapter_rank_override"] = int(adapter_rank_override)
    return checkpoint


def restore_phase_checkpoint(
    vocab_size: int,
    checkpoint: Dict[str, object],
    load_optimizer: bool = False,
) -> Tuple[ww.TinyGPT, torch.optim.Optimizer]:
    adapter_rank_override = int(checkpoint.get("adapter_rank_override", 0) or 0)
    if adapter_rank_override <= 0 or adapter_rank_override == ww.ADAPTER_RANK:
        return ww.restore_model_from_checkpoint(vocab_size, checkpoint, load_optimizer=load_optimizer)

    original_rank = ww.ADAPTER_RANK
    ww.ADAPTER_RANK = adapter_rank_override
    try:
        model = ww.TinyGPT(vocab_size).to(ww.DEVICE)
    finally:
        ww.ADAPTER_RANK = original_rank

    target_state = model.state_dict()
    source_state = checkpoint["model"]  # type: ignore[index]
    resized_state = {}
    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            resized_state[key] = target_tensor
            continue
        source_tensor = source_tensor.to(dtype=target_tensor.dtype)
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            resized_state[key] = source_tensor
            continue
        blended = target_tensor.clone()
        slices = tuple(slice(0, min(s, t)) for s, t in zip(source_tensor.shape, target_tensor.shape))
        blended[slices] = source_tensor[slices]
        resized_state[key] = blended
    model.load_state_dict(resized_state, strict=False)

    optimizer = ww.make_optimizer(model)
    if load_optimizer:
        try:
            optimizer.load_state_dict(ww.tensor_tree_to_cpu(checkpoint["optimizer"]))  # type: ignore[index]
            ww.optimizer_to_device(optimizer, ww.DEVICE)
        except Exception:
            pass
    return model, optimizer


def text_batch_from_seed(data: torch.Tensor, seed: int, index: int, batch_size: int = ww.BATCH_SIZE) -> ww.Batch:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed * 1_111_111 + index)
    starts = torch.randint(0, data.size(0) - ww.BLOCK_SIZE - 1, (batch_size,), generator=gen)
    return ww.text_batch_from_positions(data, starts)


def make_text_eval_positions(val_len: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 33_000_000)
    return torch.randint(
        0,
        val_len - ww.BLOCK_SIZE - 1,
        (ww.TEXT_EVAL_BATCHES, ww.TEXT_EVAL_BATCH),
        generator=gen,
    )


def build_arithmetic_stream(rng: np.random.Generator, digits: int = ARITH_EVAL_DIGITS) -> Tuple[List[str], List[bool]]:
    seq_len = digits
    alphabet = ARITH_CHARS[:-2]
    token_ids = rng.integers(0, len(alphabet), size=seq_len)
    source = "".join(alphabet[int(idx)] for idx in token_ids)
    prompt = f"{source}|"
    rhs = f"{source[::-1]}\n"
    episode = list(prompt + rhs)
    flags = [False] * len(prompt) + [True] * seq_len + [False]

    if not ARITH_SINGLE_PROBLEM:
        chars: List[str] = []
        critical: List[bool] = []
        while len(chars) < ww.BLOCK_SIZE + 1:
            chars.extend(episode)
            critical.extend(flags)
        return chars[: ww.BLOCK_SIZE + 1], critical[: ww.BLOCK_SIZE + 1]

    total_len = ww.BLOCK_SIZE + 1
    pad_len = max(total_len - len(episode), 0)
    if ARITH_RANDOM_PAD and pad_len > 0:
        left_pad = int(rng.integers(0, pad_len + 1))
        right_pad = pad_len - left_pad
    else:
        left_pad = 0
        right_pad = pad_len
    chars = ([" "] * left_pad) + episode + ([" "] * right_pad)
    critical = ([False] * left_pad) + flags + ([False] * right_pad)
    return chars[:total_len], critical[:total_len]


def make_arithmetic_batch(
    stoi: Dict[str, int],
    seed: int,
    index: int,
    batch_size: int = ww.BATCH_SIZE,
    digits: int = ARITH_EVAL_DIGITS,
) -> ww.Batch:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for offset in range(batch_size):
        rng = np.random.default_rng(seed * 2_222_223 + index * 10_003 + offset)
        chars, critical = build_arithmetic_stream(rng, digits=digits)
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


def make_fixed_arithmetic_batches(
    stoi: Dict[str, int],
    seed: int,
    num_batches: int,
    batch_size: int,
    digits: int = ARITH_EVAL_DIGITS,
) -> List[ww.Batch]:
    return [make_arithmetic_batch(stoi, seed, index, batch_size=batch_size, digits=digits) for index in range(num_batches)]


def arithmetic_digits_for_step(step: int, total_steps: int = ww.PHASE_B_STEPS) -> int:
    if len(ARITH_ATTACH_DIGIT_STAGES) == 1:
        return ARITH_ATTACH_DIGIT_STAGES[0]
    progress = step / max(total_steps, 1)
    if len(ARITH_ATTACH_DIGIT_STAGES) == 2:
        return ARITH_ATTACH_DIGIT_STAGES[0] if progress < 0.40 else ARITH_ATTACH_DIGIT_STAGES[1]
    if progress < 0.25:
        return ARITH_ATTACH_DIGIT_STAGES[0]
    if progress < 0.65:
        return ARITH_ATTACH_DIGIT_STAGES[1]
    return ARITH_ATTACH_DIGIT_STAGES[2]


def arithmetic_weighted_loss(logits: torch.Tensor, batch: ww.Batch) -> torch.Tensor:
    if batch.critical_mask is None:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.y.reshape(-1))
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch.y.reshape(-1),
        reduction="none",
    ).reshape_as(batch.y)
    mask = batch.critical_mask
    weights = torch.where(
        mask,
        torch.full_like(token_loss, ARITH_ANSWER_WEIGHT),
        torch.full_like(token_loss, ARITH_CONTEXT_WEIGHT),
    )
    return (token_loss * weights).sum() / weights.sum().clamp_min(1.0)


def task_loss_from_logits(task: TaskSpec, logits: torch.Tensor, batch: ww.Batch) -> torch.Tensor:
    if task.loss_fn is not None:
        return task.loss_fn(logits, batch)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.y.reshape(-1))


def task_train_one_step(
    model: ww.TinyGPT,
    optimizer: torch.optim.Optimizer,
    batch: ww.Batch,
    task: TaskSpec,
    base_lr: float,
    adapter_fluid: bool,
) -> float:
    model.train()
    ww.set_optimizer_lrs(optimizer, base_lr, adapter_fluid)
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(batch.x, batch.y)
    loss = task_loss_from_logits(task, logits, batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), ww.GRAD_CLIP)
    optimizer.step()
    return float(loss.item())


def replay_budget_fraction_for_task(task: TaskSpec) -> float:
    if task.name == "arith":
        return ARITH_REPLAY_BUDGET_FRACTION
    return ww.REPLAY_BUDGET_FRACTION


def relax_projection_for_arithmetic(
    step: int,
    current_projection: float,
    metrics: Dict[str, float],
) -> float:
    answer_acc = float(metrics.get("arith_acc", 0.0))
    problem_acc = float(metrics.get("arith_problem_acc", 0.0))
    relaxed = current_projection
    if step >= ARITH_ESCAPE_WARMUP and answer_acc < 0.50 and problem_acc < 0.05:
        relaxed = min(relaxed, ARITH_ESCAPE_LEVELS[0])
    if step >= ARITH_ESCAPE_MID and answer_acc < 0.70 and problem_acc < 0.15:
        relaxed = min(relaxed, ARITH_ESCAPE_LEVELS[1])
    if step >= ARITH_ESCAPE_LATE and answer_acc < 0.82 and problem_acc < 0.30:
        relaxed = min(relaxed, ARITH_ESCAPE_LEVELS[2])
    return relaxed


def attach_should_preserve_old_world(new_task: TaskSpec, old_tasks: Sequence[TaskSpec]) -> bool:
    return new_task.name == "arith" and len(old_tasks) > 0


def c_attach_adapter_rank(new_task: TaskSpec) -> int | None:
    if new_task.name == "arith":
        return ARITH_ADAPTER_RANK
    return None


def compat_weights_for_step(step: int, total_steps: int) -> Tuple[float, float, float]:
    factor = 1.0
    if step >= int(total_steps * 0.50):
        factor = ARITH_OLD_COMPAT_MID_FACTOR
    if step >= int(total_steps * 0.80):
        factor = ARITH_OLD_COMPAT_LATE_FACTOR
    return (
        ARITH_OLD_COMPAT_TASK_WEIGHT * factor,
        ARITH_OLD_COMPAT_KL_WEIGHT * factor,
        ARITH_OLD_COMPAT_HIDDEN_WEIGHT * factor,
    )


def is_reversal_transfer_branch(branch_name: str) -> bool:
    return branch_name in {"dual_lateral_arith_transfer", "dual_lateral_reversal_transfer"}


def sharpen_compat_boost(metrics: Dict[str, float]) -> float:
    problem_acc = float(metrics.get("arith_problem_acc", 0.0))
    answer_acc = float(metrics.get("arith_acc", 0.0))
    if problem_acc >= 0.50 or answer_acc >= 0.80:
        return ARITH_SHARPEN_COMPAT_LATE_BOOST
    if problem_acc >= 0.10 or answer_acc >= 0.50:
        return ARITH_SHARPEN_COMPAT_MID_BOOST
    return 1.0


def needs_arith_sharpen(metrics: Dict[str, float]) -> bool:
    return (
        float(metrics.get("arith_problem_acc", 0.0)) < ARITH_SHARPEN_TARGET_PROB
        or float(metrics.get("arith_acc", 0.0)) < ARITH_SHARPEN_TARGET_ACC
    )


def arith_sharpen_candidate_ok(metrics: Dict[str, float], reference: Dict[str, float]) -> bool:
    bracket_floor = float(reference.get("bracket_seq", 0.0)) - ARITH_SHARPEN_BRACKET_SLACK
    text_ceiling = float(reference.get("text_loss", float("inf"))) + ARITH_SHARPEN_TEXT_SLACK
    return (
        float(metrics.get("bracket_seq", 0.0)) >= bracket_floor
        and float(metrics.get("text_loss", float("inf"))) <= text_ceiling
    )


def consolidation_old_batch_schedule(branch_name: str, step: int) -> bool:
    if is_reversal_transfer_branch(branch_name):
        if step <= int(lateral.CONSOLIDATION_STEPS * ABC_ARITH_TRANSFER_EARLY):
            return step % 3 == 1
        if step <= int(lateral.CONSOLIDATION_STEPS * ABC_ARITH_TRANSFER_LATE):
            return step % 2 == 1
        return step % 3 != 0
    return lateral.old_batch_schedule(branch_name, step)


def consolidation_expected_old_batches(branch_name: str) -> int:
    return sum(1 for step in range(1, lateral.CONSOLIDATION_STEPS + 1) if consolidation_old_batch_schedule(branch_name, step))


def consolidation_weights_for_step(branch_name: str, old_step: bool, step: int) -> Tuple[float, float, float]:
    if is_reversal_transfer_branch(branch_name):
        early = step <= int(lateral.CONSOLIDATION_STEPS * ABC_ARITH_TRANSFER_EARLY)
        late = step > int(lateral.CONSOLIDATION_STEPS * ABC_ARITH_TRANSFER_LATE)
        if old_step:
            if late:
                return 3.0, 1.00, 0.75
            if early:
                return 2.0, 0.75, 0.50
            return 2.5, 0.90, 0.60
        if early:
            return 3.0, 1.50, 1.25
        if late:
            return 1.75, 0.75, 0.50
        return 2.5, 1.20, 0.90
    return lateral.consolidation_weights(branch_name, old_step)


def arith_consolidation_candidate_ok(metrics: Dict[str, float], reference: Dict[str, float]) -> bool:
    bracket_floor = float(reference.get("bracket_seq", 0.0)) - ABC_ARITH_SELECT_BRACKET_SLACK
    text_ceiling = float(reference.get("text_loss", float("inf"))) + ABC_ARITH_SELECT_TEXT_SLACK
    return (
        float(metrics.get("bracket_seq", 0.0)) >= bracket_floor
        and float(metrics.get("text_loss", float("inf"))) <= text_ceiling
    )


def arith_transfer_ready_candidate_ok(metrics: Dict[str, float], reference: Dict[str, float]) -> bool:
    bracket_floor = float(reference.get("bracket_seq", 0.0)) - ABC_ARITH_TRANSFER_BRACKET_SLACK
    text_ceiling = float(reference.get("text_loss", float("inf"))) + ABC_ARITH_TRANSFER_TEXT_SLACK
    return (
        float(metrics.get("bracket_seq", 0.0)) >= bracket_floor
        and float(metrics.get("text_loss", float("inf"))) <= text_ceiling
    )


def task_uses_transfer_selector(task: TaskSpec) -> bool:
    return task.name in {"arith", "reverse"}


def retention_is_weak(metrics: Dict[str, float], retention_reference: Dict[str, float]) -> bool:
    return (
        float(metrics.get("bracket_seq", 0.0))
        < float(retention_reference.get("bracket_seq", 0.0)) - ARITH_TRANSFER_TRIGGER_BRACKET_SLACK
        or float(metrics.get("text_loss", float("inf")))
        > float(retention_reference.get("text_loss", float("inf"))) + ARITH_TRANSFER_TRIGGER_TEXT_SLACK
    )


def needs_arith_transfer_polish(
    metrics: Dict[str, float],
    teacher_reference: Dict[str, float],
    retention_reference: Dict[str, float],
) -> bool:
    teacher_prob = float(teacher_reference.get("arith_problem_acc", 0.0))
    teacher_acc = float(teacher_reference.get("arith_acc", 0.0))
    new_task_weak = False
    if teacher_prob > 0.0 or teacher_acc > 0.0:
        new_task_weak = (
            float(metrics.get("arith_problem_acc", 0.0)) < teacher_prob * ARITH_TRANSFER_TARGET_FRACTION
            or float(metrics.get("arith_acc", 0.0)) < teacher_acc * ARITH_TRANSFER_TARGET_FRACTION
        )
    return new_task_weak or retention_is_weak(metrics, retention_reference)


def arith_transfer_candidate_ok(metrics: Dict[str, float], retention_reference: Dict[str, float]) -> bool:
    bracket_floor = float(retention_reference.get("bracket_seq", 0.0)) - ARITH_TRANSFER_POLISH_BRACKET_SLACK
    text_ceiling = float(retention_reference.get("text_loss", float("inf"))) + ARITH_TRANSFER_POLISH_TEXT_SLACK
    return (
        float(metrics.get("bracket_seq", 0.0)) >= bracket_floor
        and float(metrics.get("text_loss", float("inf"))) <= text_ceiling
    )


def reverse_extract_candidate_ok(metrics: Dict[str, float], retention_reference: Dict[str, float]) -> bool:
    bracket_floor = float(retention_reference.get("bracket_seq", 0.0)) - REVERSE_SELECTION_BRACKET_SLACK
    return float(metrics.get("bracket_seq", 0.0)) >= bracket_floor


def better_reverse_candidate(candidate: Dict[str, float], incumbent: Dict[str, float] | None) -> bool:
    if incumbent is None:
        return True
    cand_text = float(candidate.get("text_loss", float("-inf")))
    inc_text = float(incumbent.get("text_loss", float("-inf")))
    if cand_text > inc_text + 1e-12:
        return True
    if abs(cand_text - inc_text) > 1e-12:
        return False
    return float(candidate.get("bracket_seq", 0.0)) > float(incumbent.get("bracket_seq", 0.0)) + 1e-12


def better_arith_candidate(candidate: Dict[str, float], incumbent: Dict[str, float] | None) -> bool:
    if incumbent is None:
        return True
    cand_prob = float(candidate.get("arith_problem_acc", 0.0))
    inc_prob = float(incumbent.get("arith_problem_acc", 0.0))
    if cand_prob > inc_prob + 1e-12:
        return True
    if inc_prob > cand_prob + 1e-12:
        return False
    cand_acc = float(candidate.get("arith_acc", 0.0))
    inc_acc = float(incumbent.get("arith_acc", 0.0))
    if cand_acc > inc_acc + 1e-12:
        return True
    if inc_acc > cand_acc + 1e-12:
        return False
    return float(candidate.get("text_loss", float("inf"))) < float(incumbent.get("text_loss", float("inf")))


def pick_old_retention_task(
    old_tasks: Sequence[TaskSpec],
    cycle_index: int,
    metrics: Dict[str, float] | None = None,
    retention_reference: Dict[str, float] | None = None,
) -> TaskSpec:
    if old_tasks and metrics is not None and retention_reference is not None:
        bracket_weak = float(metrics.get("bracket_seq", 0.0)) < (
            float(retention_reference.get("bracket_seq", 0.0)) - ARITH_TRANSFER_TRIGGER_BRACKET_SLACK
        )
        text_weak = float(metrics.get("text_loss", float("inf"))) > (
            float(retention_reference.get("text_loss", float("inf"))) + ARITH_TRANSFER_TRIGGER_TEXT_SLACK
        )
        bracket_task = next((task for task in old_tasks if task.name == "bracket"), None)
        text_task = next((task for task in old_tasks if task.name == "text"), None)
        if bracket_weak and text_weak and bracket_task is not None and text_task is not None:
            return text_task if cycle_index % 2 else bracket_task
        if text_weak and text_task is not None:
            return text_task
        if bracket_weak and bracket_task is not None:
            return bracket_task
    return old_tasks[cycle_index % len(old_tasks)]


@torch.no_grad()
def evaluate_arithmetic(model: ww.TinyGPT, batches: List[ww.Batch]) -> Dict[str, float]:
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


def make_bracket_task(
    stoi: Dict[str, int],
    bracket_eval_batches: List[ww.Batch],
    seed: int,
) -> TaskSpec:
    def evaluate(model: ww.TinyGPT) -> Dict[str, float]:
        metrics = ww.evaluate_bracket(model, bracket_eval_batches)
        return {
            "bracket_loss": metrics["loss"],
            "bracket_close": metrics["close_acc"],
            "bracket_seq": metrics["seq_acc"],
        }

    return TaskSpec(
        name="bracket",
        sample_train_batch=lambda step: ww.bracket_train_batch_for_step(stoi, seed + 1_000, step),
        sample_anchor_batch=lambda index: ww.replay_batch_for_index(stoi, seed + 2_000, index),
        evaluate=evaluate,
        primary_key="bracket_seq",
        goal="max",
    )


def make_text_task(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    text_eval_positions: torch.Tensor,
    seed: int,
) -> TaskSpec:
    def evaluate(model: ww.TinyGPT) -> Dict[str, float]:
        metrics = ww.evaluate_text(model, val_data, text_eval_positions)
        return {
            "text_loss": metrics["loss"],
            "text_acc": metrics["acc"],
        }

    return TaskSpec(
        name="text",
        sample_train_batch=lambda step: text_batch_from_seed(train_data, seed + 3_000, step),
        sample_anchor_batch=lambda index: text_batch_from_seed(train_data, seed + 4_000, index),
        evaluate=evaluate,
        primary_key="text_loss",
        goal="min",
    )


def make_arithmetic_task(
    stoi: Dict[str, int],
    arithmetic_eval_batches: List[ww.Batch],
    seed: int,
) -> TaskSpec:
    def evaluate(model: ww.TinyGPT) -> Dict[str, float]:
        metrics = evaluate_arithmetic(model, arithmetic_eval_batches)
        return {
            "arith_loss": metrics["loss"],
            "arith_acc": metrics["answer_acc"],
            "arith_problem_acc": metrics["problem_acc"],
            "arith_seq": metrics["seq_acc"],
            "reverse_loss": metrics["loss"],
            "reverse_acc": metrics["answer_acc"],
            "reverse_problem_acc": metrics["problem_acc"],
            "reverse_seq": metrics["seq_acc"],
        }

    return TaskSpec(
        name="arith",
        sample_train_batch=lambda step: make_arithmetic_batch(
            stoi,
            seed + 5_000,
            step,
            digits=arithmetic_digits_for_step(step),
        ),
        sample_anchor_batch=lambda index: make_arithmetic_batch(
            stoi,
            seed + 6_000,
            index,
            digits=ARITH_EVAL_DIGITS,
        ),
        evaluate=evaluate,
        primary_key="arith_problem_acc",
        goal="max",
        sample_consolidation_batch=lambda step: make_arithmetic_batch(
            stoi,
            seed + 7_000,
            step,
            digits=ARITH_EVAL_DIGITS,
        ),
        loss_fn=arithmetic_weighted_loss,
    )


def evaluate_world(model: ww.TinyGPT, tasks: Sequence[TaskSpec]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for task in tasks:
        metrics.update(task.evaluate(model))
    return metrics


def summarize_metrics(metrics: Dict[str, float]) -> str:
    parts = []
    if "bracket_seq" in metrics:
        parts.append(f"bracket_seq={metrics['bracket_seq']:.3f}")
    if "text_loss" in metrics:
        parts.append(f"text_loss={metrics['text_loss']:.3f}")
    if "arith_acc" in metrics:
        parts.append(f"rev_acc={metrics['arith_acc']:.3f}")
    if "arith_problem_acc" in metrics:
        parts.append(f"rev_prob={metrics['arith_problem_acc']:.3f}")
    if "arith_seq" in metrics:
        parts.append(f"rev_seq={metrics['arith_seq']:.3f}")
    return " ".join(parts)


def anchor_from_old_skill(anchor: ww.AnchorInfo) -> AnchorBundle:
    return AnchorBundle(
        label="A",
        checkpoint=anchor.checkpoint,
        old_frontier=list(anchor.old_frontier),
        block_anchor_z=dict(anchor.block_anchor_z),
        grad_basis={block: basis.clone() for block, basis in anchor.grad_basis.items()},
        latent_free_projectors={
            block: projector.clone() for block, projector in anchor.latent_free_projectors.items()
        },
    )


def _task_block_z_map(model: ww.TinyGPT, batch: ww.Batch) -> Dict[str, float]:
    param_z, act_z = ww.probe_z(model, batch)
    return ww.block_combined_z_map(param_z, act_z)


def compute_anchor_shock(model: ww.TinyGPT, anchor: AnchorBundle, tasks: Sequence[TaskSpec]) -> float:
    shocks = []
    for task in tasks:
        current_block_z = _task_block_z_map(model, task.sample_anchor_batch(0))
        block_shocks = []
        for block in anchor.old_frontier:
            current = max(current_block_z.get(block, 1e-12), 1e-12)
            base = max(anchor.block_anchor_z.get(block, 1e-12), 1e-12)
            block_shocks.append(abs(math.log2(current / base)))
        if block_shocks:
            shocks.append(float(np.mean(block_shocks)))
    return float(np.mean(shocks)) if shocks else 0.0


def run_mixed_reminiscence(
    model: ww.TinyGPT,
    optimizer: torch.optim.Optimizer,
    tasks: Sequence[TaskSpec],
    seed: int,
) -> None:
    if not tasks:
        return
    model.set_adapters_enabled(False)
    ww.set_requires_grad(ww.all_adapter_params(model), False)
    for step in range(1, ww.REMINISCENCE_STEPS + 1):
        task = tasks[(seed + step - 1) % len(tasks)]
        batch = task.sample_anchor_batch(50_000 + step)
        ww.train_one_step(model, optimizer, batch, base_lr=ww.REMINISCENCE_LR, adapter_fluid=False)


def collect_mixed_gradient_basis(
    model: ww.TinyGPT,
    tasks: Sequence[TaskSpec],
    blocks: Sequence[str],
) -> Dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    rows: Dict[str, List[torch.Tensor]] = {block: [] for block in blocks}
    for index in range(ww.GRAD_ANCHOR_BATCHES):
        task = tasks[index % len(tasks)]
        batch = task.sample_anchor_batch(index)
        model.zero_grad(set_to_none=True)
        logits, _ = model(batch.x, batch.y)
        loss = task_loss_from_logits(task, logits, batch)
        loss.backward()
        for block in blocks:
            rows[block].append(ww.flatten_grads(ww.base_block_params(model, block)).detach().cpu())
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return {block: ww.make_low_rank_basis(block_rows, ww.GRAD_ANCHOR_RANK) for block, block_rows in rows.items()}


def collect_mixed_latent_free_projectors(
    model: ww.TinyGPT,
    tasks: Sequence[TaskSpec],
    blocks: Sequence[str],
) -> Dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    act_cov = {block: torch.zeros(ww.D_MODEL, ww.D_MODEL) for block in blocks}
    grad_cov = {block: torch.zeros(ww.D_MODEL, ww.D_MODEL) for block in blocks}
    counts = {block: 0 for block in blocks}

    for index in range(ww.LATENT_ANCHOR_BATCHES):
        task = tasks[index % len(tasks)]
        batch = task.sample_anchor_batch(index)
        model.zero_grad(set_to_none=True)
        logits, _, activations = model(batch.x, batch.y, return_activations=True)
        loss = task_loss_from_logits(task, logits, batch)
        loss.backward()
        for block in blocks:
            block_index = int(block[1:])
            activation = activations[block_index].detach().reshape(-1, ww.D_MODEL).float().cpu()
            gradient = activations[block_index].grad
            if gradient is None:
                gradient_flat = torch.zeros_like(activation)
            else:
                gradient_flat = gradient.detach().reshape(-1, ww.D_MODEL).float().cpu()
            act_cov[block] += activation.t() @ activation / max(activation.shape[0], 1)
            grad_cov[block] += gradient_flat.t() @ gradient_flat / max(gradient_flat.shape[0], 1)
            counts[block] += 1

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    projectors = {}
    for block in blocks:
        denom = max(counts[block], 1)
        projectors[block] = ww.free_projector_from_covariances(
            act_cov[block] / denom,
            grad_cov[block] / denom,
        )
    return projectors


def collect_world_anchor(
    label: str,
    checkpoint: Dict[str, object],
    tasks: Sequence[TaskSpec],
    vocab_size: int,
) -> AnchorBundle:
    model, _optimizer = restore_phase_checkpoint(vocab_size, checkpoint, load_optimizer=False)
    block_maps = []
    for task in tasks:
        block_maps.append(_task_block_z_map(model, task.sample_anchor_batch(0)))
    combined = {}
    for block in ww.block_keys():
        logs = [math.log(max(block_map.get(block, 1e-12), 1e-12)) for block_map in block_maps]
        combined[block] = float(math.exp(float(np.mean(np.asarray(logs, dtype=float)))))
    old_frontier = ww.top_blocks(combined)
    print(
        f"[anchor:{label}] collecting mixed gradient basis for {'+'.join(old_frontier)} "
        f"over {len(tasks)} task(s)"
    )
    grad_basis = collect_mixed_gradient_basis(model, tasks, old_frontier)
    print(
        f"[anchor:{label}] collecting mixed latent occupied-space basis for all blocks "
        f"over {len(tasks)} task(s)"
    )
    latent_free_projectors = collect_mixed_latent_free_projectors(model, tasks, ww.block_keys())
    del model, _optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return AnchorBundle(
        label=label,
        checkpoint=checkpoint,
        old_frontier=old_frontier,
        block_anchor_z=combined,
        grad_basis=grad_basis,
        latent_free_projectors=latent_free_projectors,
    )


def attach_latent_teacher(
    label: str,
    anchor: AnchorBundle,
    vocab_size: int,
    old_tasks: Sequence[TaskSpec],
    new_task: TaskSpec,
    eval_tasks: Sequence[TaskSpec],
    seed: int,
) -> StageResult:
    adapter_rank_override = c_attach_adapter_rank(new_task)
    phase_checkpoint = dict(anchor.checkpoint)
    if adapter_rank_override is not None:
        phase_checkpoint["adapter_rank_override"] = adapter_rank_override
    model, optimizer = restore_phase_checkpoint(vocab_size, phase_checkpoint, load_optimizer=False)
    latent_projectors = {block: projector.to(ww.DEVICE) for block, projector in anchor.latent_free_projectors.items()}
    old_teacher_model = None
    old_teacher_opt = None
    preserve_old_world = attach_should_preserve_old_world(new_task, old_tasks)
    if preserve_old_world:
        old_teacher_model, old_teacher_opt = restore_phase_checkpoint(
            vocab_size, anchor.checkpoint, load_optimizer=False
        )
        set_model_base_only(old_teacher_model)
        ww.set_requires_grad(old_teacher_model.parameters(), False)
    print(f"[teacher:{label}] momentum reset + old-world reminiscence")
    run_mixed_reminiscence(model, optimizer, old_tasks, seed)
    ww.configure_branch_trainability(model, "water_weights_latent_adapter_only", anchor.old_frontier)
    optimizer = lateral.make_optimizer_for_adapter(model)

    replay_budget = int(ww.PHASE_B_STEPS * replay_budget_fraction_for_task(new_task))
    replay_count = 0
    current_latent_projection = ww.INITIAL_LATENT_PROJECTION
    latent_projection_steps = 0
    final_metrics = evaluate_world(model, eval_tasks)

    for step in range(1, ww.PHASE_B_STEPS + 1):
        replay_this_step = ww.should_replay("water_weights_latent_adapter_only", step, replay_count, replay_budget)
        if replay_this_step:
            current_task = old_tasks[replay_count % len(old_tasks)]
            batch = current_task.sample_anchor_batch(replay_count)
            replay_count += 1
        else:
            current_task = new_task
            batch = new_task.sample_train_batch(step)

        latent_strength = ww.latent_projection_strength_for_branch(
            "water_weights_latent_adapter_only",
            current_latent_projection,
        )
        model.set_adapters_enabled(True)
        model.set_latent_free_projectors(latent_projectors, latent_strength)
        latent_projection_steps += int(latent_strength > 0.0)
        if preserve_old_world and current_task.name == new_task.name:
            compat_task = old_tasks[(step - 1) % len(old_tasks)]
            compat_batch = compat_task.sample_anchor_batch(100_000 + step)
            model.train()
            ww.set_optimizer_lrs(optimizer, ww.BASE_LR, True)
            optimizer.zero_grad(set_to_none=True)

            new_logits, _ = model(batch.x, batch.y)
            loss = task_loss_from_logits(current_task, new_logits, batch)

            with torch.no_grad():
                teacher_logits, _teacher_loss, teacher_states = lateral.forward_with_block_outputs(
                    old_teacher_model, compat_batch, detach=True
                )
            compat_logits, _compat_loss, compat_states = lateral.forward_with_block_outputs(
                model, compat_batch, detach=False
            )
            compat_task_loss = task_loss_from_logits(compat_task, compat_logits, compat_batch)
            compat_task_w, compat_kl_w, compat_hidden_w = compat_weights_for_step(step, ww.PHASE_B_STEPS)
            loss = (
                loss
                + compat_task_w * compat_task_loss
                + compat_kl_w * lateral.distill_kl(compat_logits, teacher_logits)
                + compat_hidden_w * lateral.hidden_lateral_loss(compat_states, teacher_states)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ww.all_adapter_params(model), ww.GRAD_CLIP)
            optimizer.step()
        else:
            task_train_one_step(model, optimizer, batch, current_task, base_lr=ww.BASE_LR, adapter_fluid=True)

        if ww.is_probe_step(step):
            model.set_adapters_enabled(True)
            model.set_latent_free_projectors(latent_projectors, latent_strength)
            final_metrics = evaluate_world(model, eval_tasks)
            shock = compute_anchor_shock(model, anchor, old_tasks)
            current_latent_projection = ww.latent_projection_from_shock(shock)
            if new_task.name == "arith":
                current_latent_projection = relax_projection_for_arithmetic(
                    step, current_latent_projection, final_metrics
                )
            if step % ww.BRANCH_LOG_INTERVAL == 0 or step == ww.PHASE_B_STEPS:
                print(
                    f"[teacher:{label}] step={step:04d}/{ww.PHASE_B_STEPS} "
                    f"{summarize_metrics(final_metrics)} z_shock={shock:.2f} "
                    f"latent_proj={latent_strength:.2f} replay={replay_count}/{replay_budget}"
                )

    checkpoint = make_phase_checkpoint(model, optimizer, adapter_rank_override=adapter_rank_override)
    del model, optimizer, old_teacher_model, old_teacher_opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return StageResult(
        label=label,
        checkpoint=checkpoint,
        metrics=final_metrics,
        replay_count=replay_count,
        replay_budget=replay_budget,
        latent_projection_steps=latent_projection_steps,
        base_only_verified=False,
    )


def sharpen_arithmetic_teacher(
    label: str,
    initial_stage: StageResult,
    anchor: AnchorBundle,
    vocab_size: int,
    old_tasks: Sequence[TaskSpec],
    arith_task: TaskSpec,
    eval_tasks: Sequence[TaskSpec],
) -> StageResult:
    if not needs_arith_sharpen(initial_stage.metrics):
        return StageResult(
            label=label,
            checkpoint=initial_stage.checkpoint,
            metrics=dict(initial_stage.metrics),
            latent_projection_steps=0,
            base_only_verified=False,
        )

    model, _optimizer = restore_phase_checkpoint(vocab_size, initial_stage.checkpoint, load_optimizer=False)
    ww.configure_branch_trainability(model, "water_weights_latent_adapter_only", anchor.old_frontier)
    optimizer = lateral.make_optimizer_for_adapter(model)
    latent_projectors = {block: projector.to(ww.DEVICE) for block, projector in anchor.latent_free_projectors.items()}

    old_teacher_model, _old_teacher_opt = restore_phase_checkpoint(vocab_size, anchor.checkpoint, load_optimizer=False)
    set_model_base_only(old_teacher_model)
    ww.set_requires_grad(old_teacher_model.parameters(), False)

    adapter_rank_override = int(initial_stage.checkpoint.get("adapter_rank_override", 0) or 0)
    current_latent_projection = ARITH_SHARPEN_PROJECTION
    latent_projection_steps = 0
    final_metrics = evaluate_world(model, eval_tasks)
    best_metrics = dict(initial_stage.metrics)
    best_checkpoint = initial_stage.checkpoint
    best_problem = float(best_metrics.get("arith_problem_acc", 0.0))
    best_answer = float(best_metrics.get("arith_acc", 0.0))

    print(f"[sharpen:{label}] reversal-only adapter sharpening for {ARITH_SHARPEN_STEPS} steps")
    for step in range(1, ARITH_SHARPEN_STEPS + 1):
        batch = arith_task.sample_train_batch(200_000 + step)
        compat_task = old_tasks[(step - 1) % len(old_tasks)]
        compat_batch = compat_task.sample_anchor_batch(300_000 + step)

        model.train()
        model.set_adapters_enabled(True)
        model.set_latent_free_projectors(latent_projectors, current_latent_projection)
        latent_projection_steps += 1
        ww.set_optimizer_lrs(optimizer, ww.BASE_LR * ARITH_SHARPEN_LR_SCALE, True)
        optimizer.zero_grad(set_to_none=True)

        new_logits, _ = model(batch.x, batch.y)
        loss = task_loss_from_logits(arith_task, new_logits, batch)

        with torch.no_grad():
            teacher_logits, _teacher_loss, teacher_states = lateral.forward_with_block_outputs(
                old_teacher_model, compat_batch, detach=True
            )
        compat_logits, _compat_loss, compat_states = lateral.forward_with_block_outputs(
            model, compat_batch, detach=False
        )
        compat_task_loss = task_loss_from_logits(compat_task, compat_logits, compat_batch)
        compat_boost = sharpen_compat_boost(final_metrics)
        loss = (
            loss
            + (ARITH_SHARPEN_COMPAT_TASK_WEIGHT * compat_boost) * compat_task_loss
            + (ARITH_SHARPEN_COMPAT_KL_WEIGHT * compat_boost) * lateral.distill_kl(compat_logits, teacher_logits)
            + (ARITH_SHARPEN_COMPAT_HIDDEN_WEIGHT * compat_boost)
            * lateral.hidden_lateral_loss(compat_states, teacher_states)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ww.all_adapter_params(model), ww.GRAD_CLIP)
        optimizer.step()

        if ww.is_probe_step(step) or step == ARITH_SHARPEN_STEPS:
            model.set_adapters_enabled(True)
            model.set_latent_free_projectors(latent_projectors, current_latent_projection)
            final_metrics = evaluate_world(model, eval_tasks)
            if step % ww.BRANCH_LOG_INTERVAL == 0 or step == ARITH_SHARPEN_STEPS:
                print(
                    f"[sharpen:{label}] step={step:04d}/{ARITH_SHARPEN_STEPS} "
                    f"{summarize_metrics(final_metrics)} latent_proj={current_latent_projection:.2f}"
                )

            problem_acc = float(final_metrics.get("arith_problem_acc", 0.0))
            answer_acc = float(final_metrics.get("arith_acc", 0.0))
            if arith_sharpen_candidate_ok(final_metrics, initial_stage.metrics):
                better = (problem_acc > best_problem + 1e-12) or (
                    abs(problem_acc - best_problem) <= 1e-12 and answer_acc > best_answer + 1e-12
                )
                if better:
                    best_problem = problem_acc
                    best_answer = answer_acc
                    best_metrics = dict(final_metrics)
                    best_checkpoint = make_phase_checkpoint(
                        model, optimizer, adapter_rank_override=adapter_rank_override or None
                    )
                if (
                    problem_acc >= ARITH_SHARPEN_TARGET_PROB
                    and answer_acc >= ARITH_SHARPEN_TARGET_ACC
                ):
                    break

    del model, _optimizer, optimizer, old_teacher_model, _old_teacher_opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return StageResult(
        label=label,
        checkpoint=best_checkpoint,
        metrics=best_metrics,
        latent_projection_steps=latent_projection_steps,
        base_only_verified=False,
    )


def polish_arithmetic_transfer(
    label: str,
    initial_stage: StageResult,
    anchor_ab: AnchorBundle,
    vocab_size: int,
    old_tasks: Sequence[TaskSpec],
    arith_task: TaskSpec,
    eval_tasks: Sequence[TaskSpec],
    teacher_checkpoint: Dict[str, object],
    teacher_projectors: Dict[str, torch.Tensor],
    teacher_reference: Dict[str, float],
    retention_reference: Dict[str, float],
) -> StageResult:
    if not needs_arith_transfer_polish(initial_stage.metrics, teacher_reference, retention_reference):
        return StageResult(
            label=label,
            checkpoint=initial_stage.checkpoint,
            metrics=dict(initial_stage.metrics),
            old_batch_count=0,
            old_batch_budget=0,
            base_only_verified=True,
        )

    new_teacher_model, _new_teacher_opt = restore_phase_checkpoint(
        vocab_size, teacher_checkpoint, load_optimizer=False
    )
    lateral.set_teacher_mode(new_teacher_model, teacher_projectors, latent_strength=1.0)
    ww.set_requires_grad(new_teacher_model.parameters(), False)

    old_teacher_model, _old_teacher_opt = restore_phase_checkpoint(
        vocab_size, anchor_ab.checkpoint, load_optimizer=False
    )
    set_model_base_only(old_teacher_model)
    ww.set_requires_grad(old_teacher_model.parameters(), False)

    student, _student_opt = restore_phase_checkpoint(vocab_size, initial_stage.checkpoint, load_optimizer=False)
    lateral.set_student_base_only(student, trainable_base=True)
    optimizer = lateral.make_optimizer_for_base(student, lateral.CONSOLIDATION_LR * ARITH_TRANSFER_POLISH_LR_SCALE)

    old_period = (
        ARITH_TRANSFER_RETENTION_OLD_PERIOD
        if retention_is_weak(initial_stage.metrics, retention_reference)
        else ARITH_TRANSFER_POLISH_OLD_PERIOD
    )
    old_batch_budget = sum(1 for step in range(1, ARITH_TRANSFER_POLISH_STEPS + 1) if step % old_period == 0)
    old_batch_count = 0
    final_metrics = evaluate_world(student, eval_tasks)
    best_metrics = dict(initial_stage.metrics)
    best_checkpoint = initial_stage.checkpoint

    print(f"[transfer:{label}] base-only reversal transfer polish for {ARITH_TRANSFER_POLISH_STEPS} steps")
    for step in range(1, ARITH_TRANSFER_POLISH_STEPS + 1):
        old_step = step % old_period == 0
        if old_step:
            current_task = pick_old_retention_task(old_tasks, old_batch_count, final_metrics, retention_reference)
            batch = current_task.sample_anchor_batch(700_000 + old_batch_count)
            teacher_model = old_teacher_model
            old_batch_count += 1
            boost = ARITH_TRANSFER_POLISH_RETENTION_BOOST if retention_is_weak(final_metrics, retention_reference) else 1.0
            task_weight = ARITH_TRANSFER_POLISH_OLD_TASK_WEIGHT * boost
            kl_weight = ARITH_TRANSFER_POLISH_OLD_KL_WEIGHT * boost
            hidden_weight = ARITH_TRANSFER_POLISH_OLD_HIDDEN_WEIGHT * boost
        else:
            current_task = arith_task
            batch_sampler = arith_task.sample_consolidation_batch or arith_task.sample_train_batch
            batch = batch_sampler(700_000 + step)
            teacher_model = new_teacher_model
            lateral.set_teacher_mode(teacher_model, teacher_projectors, latent_strength=1.0)
            task_weight = ARITH_TRANSFER_POLISH_TASK_WEIGHT
            kl_weight = ARITH_TRANSFER_POLISH_KL_WEIGHT
            hidden_weight = ARITH_TRANSFER_POLISH_HIDDEN_WEIGHT

        with torch.no_grad():
            teacher_logits, _teacher_loss, teacher_states = lateral.forward_with_block_outputs(
                teacher_model, batch, detach=True
            )

        optimizer.zero_grad(set_to_none=True)
        student_logits, _student_loss, student_states = lateral.forward_with_block_outputs(student, batch, detach=False)
        task_loss = task_loss_from_logits(current_task, student_logits, batch)
        loss = (
            task_weight * task_loss
            + kl_weight * lateral.distill_kl(student_logits, teacher_logits)
            + hidden_weight * lateral.hidden_lateral_loss(student_states, teacher_states)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ww.all_base_params(student), ww.GRAD_CLIP)
        optimizer.step()

        if step % lateral.CONSOLIDATION_EVAL_INTERVAL == 0 or step == ARITH_TRANSFER_POLISH_STEPS:
            lateral.set_student_base_only(student, trainable_base=False)
            final_metrics = evaluate_world(student, eval_tasks)
            if step % lateral.CONSOLIDATION_LOG_INTERVAL == 0 or step == ARITH_TRANSFER_POLISH_STEPS:
                print(
                    f"[transfer:{label}] step={step:04d}/{ARITH_TRANSFER_POLISH_STEPS} "
                    f"{summarize_metrics(final_metrics)} old_batches={old_batch_count}/{old_batch_budget}"
                )
            if (
                arith_transfer_candidate_ok(final_metrics, retention_reference)
                and better_arith_candidate(final_metrics, best_metrics)
            ):
                best_metrics = dict(final_metrics)
                best_checkpoint = make_phase_checkpoint(student, optimizer)
            lateral.set_student_base_only(student, trainable_base=True)

    lateral.set_student_base_only(student, trainable_base=False)
    final_metrics = evaluate_world(student, eval_tasks)
    if (
        arith_transfer_candidate_ok(final_metrics, retention_reference)
        and better_arith_candidate(final_metrics, best_metrics)
    ):
        best_metrics = dict(final_metrics)
        best_checkpoint = make_phase_checkpoint(student, optimizer)

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
    return StageResult(
        label=label,
        checkpoint=best_checkpoint,
        metrics=best_metrics,
        old_batch_count=old_batch_count,
        old_batch_budget=old_batch_budget,
        base_only_verified=True,
    )


def set_model_base_only(model: ww.TinyGPT) -> None:
    lateral.set_student_base_only(model, trainable_base=False)
    model.eval()


def set_model_probe_mode(
    model: ww.TinyGPT,
    adapters_enabled: bool,
    latent_projectors: Dict[str, torch.Tensor] | None = None,
    latent_strength: float = 1.0,
) -> None:
    model.set_adapters_enabled(adapters_enabled)
    if adapters_enabled and latent_projectors is not None:
        model.set_latent_free_projectors(latent_projectors, latent_strength)
    else:
        model.clear_latent_free_projectors()
    model.eval()


def consolidate_dual_teacher(
    label: str,
    branch_name: str,
    student_checkpoint: Dict[str, object],
    old_anchor: AnchorBundle,
    old_teacher_checkpoint: Dict[str, object],
    old_tasks: Sequence[TaskSpec],
    new_teacher_checkpoint: Dict[str, object],
    new_teacher_projectors: Dict[str, torch.Tensor],
    new_task: TaskSpec,
    eval_tasks: Sequence[TaskSpec],
    vocab_size: int,
    selection_reference: Dict[str, float] | None = None,
) -> StageResult:
    new_teacher_model, _new_teacher_opt = restore_phase_checkpoint(
        vocab_size, new_teacher_checkpoint, load_optimizer=False
    )
    lateral.set_teacher_mode(new_teacher_model, new_teacher_projectors, latent_strength=1.0)
    ww.set_requires_grad(new_teacher_model.parameters(), False)

    old_teacher_model, _old_teacher_opt = restore_phase_checkpoint(
        vocab_size, old_teacher_checkpoint, load_optimizer=False
    )
    set_model_base_only(old_teacher_model)
    ww.set_requires_grad(old_teacher_model.parameters(), False)

    student, _student_opt = restore_phase_checkpoint(vocab_size, student_checkpoint, load_optimizer=False)
    lateral.set_student_base_only(student, trainable_base=True)
    if branch_name == "dual_lateral_freeze_frontier":
        for block in old_anchor.old_frontier:
            ww.set_requires_grad(ww.base_block_params(student, block), False)
    optimizer = lateral.make_optimizer_for_base(
        student,
        lateral.CONSOLIDATION_STRONG_LR if branch_name == "dual_lateral_strong" else lateral.CONSOLIDATION_LR,
    )

    old_batch_budget = consolidation_expected_old_batches(branch_name)
    old_batch_count = 0
    current_z_viscosity = ww.INITIAL_Z_VISCOSITY
    z_viscosity_steps = 0
    final_metrics = evaluate_world(student, eval_tasks)
    best_metrics: Dict[str, float] | None = None
    best_checkpoint: Dict[str, object] | None = None
    transfer_metrics: Dict[str, float] | None = None
    transfer_checkpoint: Dict[str, object] | None = None
    print(f"[consolidate:{label}] branch={branch_name} for {lateral.CONSOLIDATION_STEPS} steps")

    for step in range(1, lateral.CONSOLIDATION_STEPS + 1):
        old_step = consolidation_old_batch_schedule(branch_name, step)
        if old_step:
            current_task = pick_old_retention_task(old_tasks, old_batch_count, final_metrics, selection_reference)
            batch = current_task.sample_anchor_batch(old_batch_count)
            teacher_model = old_teacher_model
            old_batch_count += 1
        else:
            current_task = new_task
            batch_sampler = new_task.sample_consolidation_batch or new_task.sample_train_batch
            batch = batch_sampler(step)
            teacher_model = new_teacher_model
            lateral.set_teacher_mode(teacher_model, new_teacher_projectors, latent_strength=1.0)

        with torch.no_grad():
            teacher_logits, _teacher_loss, teacher_states = lateral.forward_with_block_outputs(
                teacher_model, batch, detach=True
            )

        optimizer.zero_grad(set_to_none=True)
        student_logits, _student_loss, student_states = lateral.forward_with_block_outputs(student, batch, detach=False)
        task_loss = task_loss_from_logits(current_task, student_logits, batch)
        task_weight, kl_weight, hidden_weight = consolidation_weights_for_step(branch_name, old_step, step)
        loss = (
            task_weight * task_loss
            + kl_weight * lateral.distill_kl(student_logits, teacher_logits)
            + hidden_weight * lateral.hidden_lateral_loss(student_states, teacher_states)
        )
        loss.backward()
        if lateral.apply_consolidation_viscosity(student, old_anchor, branch_name, current_z_viscosity):
            z_viscosity_steps += 1
        torch.nn.utils.clip_grad_norm_(ww.all_base_params(student), ww.GRAD_CLIP)
        optimizer.step()

        if step % lateral.CONSOLIDATION_EVAL_INTERVAL == 0 or step == lateral.CONSOLIDATION_STEPS:
            lateral.set_student_base_only(student, trainable_base=False)
            final_metrics = evaluate_world(student, eval_tasks)
            if branch_name in {"dual_lateral_z_viscous", "dual_lateral_strong"}:
                current_z_viscosity = ww.viscosity_from_shock(compute_anchor_shock(student, old_anchor, old_tasks))
            if step % lateral.CONSOLIDATION_LOG_INTERVAL == 0 or step == lateral.CONSOLIDATION_STEPS:
                print(
                    f"[consolidate:{label}] step={step:04d}/{lateral.CONSOLIDATION_STEPS} "
                    f"{summarize_metrics(final_metrics)} "
                    f"old_batches={old_batch_count}/{old_batch_budget} "
                    f"viscosity={current_z_viscosity:.3f}"
                )
            if (
                selection_reference is not None
                and task_uses_transfer_selector(new_task)
                and arith_consolidation_candidate_ok(final_metrics, selection_reference)
                and better_arith_candidate(final_metrics, best_metrics)
            ):
                best_metrics = dict(final_metrics)
                best_checkpoint = make_phase_checkpoint(student, optimizer)
            if (
                selection_reference is not None
                and task_uses_transfer_selector(new_task)
                and arith_transfer_ready_candidate_ok(final_metrics, selection_reference)
                and better_arith_candidate(final_metrics, transfer_metrics)
            ):
                transfer_metrics = dict(final_metrics)
                transfer_checkpoint = make_phase_checkpoint(student, optimizer)
            lateral.set_student_base_only(student, trainable_base=True)

    lateral.set_student_base_only(student, trainable_base=False)
    final_metrics = evaluate_world(student, eval_tasks)
    if (
        selection_reference is not None
        and task_uses_transfer_selector(new_task)
        and arith_consolidation_candidate_ok(final_metrics, selection_reference)
        and better_arith_candidate(final_metrics, best_metrics)
    ):
        best_metrics = dict(final_metrics)
        best_checkpoint = make_phase_checkpoint(student, optimizer)
    if (
        selection_reference is not None
        and task_uses_transfer_selector(new_task)
        and arith_transfer_ready_candidate_ok(final_metrics, selection_reference)
        and better_arith_candidate(final_metrics, transfer_metrics)
    ):
        transfer_metrics = dict(final_metrics)
        transfer_checkpoint = make_phase_checkpoint(student, optimizer)
    if transfer_checkpoint is not None:
        checkpoint = transfer_checkpoint
        final_metrics = transfer_metrics if transfer_metrics is not None else final_metrics
    elif best_checkpoint is not None:
        checkpoint = best_checkpoint
        final_metrics = best_metrics if best_metrics is not None else final_metrics
    else:
        checkpoint = make_phase_checkpoint(student, optimizer)
    base_only_verified = not any(param.requires_grad for param in ww.all_adapter_params(student))

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
    return StageResult(
        label=label,
        checkpoint=checkpoint,
        metrics=final_metrics,
        old_batch_count=old_batch_count,
        old_batch_budget=old_batch_budget,
        z_viscosity_steps=z_viscosity_steps,
        base_only_verified=base_only_verified,
    )


def reverse_extract_to_base_a(
    label: str,
    unified_checkpoint: Dict[str, object],
    teacher_checkpoint: Dict[str, object],
    retention_reference: Dict[str, float],
    anchor_a: AnchorBundle,
    bracket_task: TaskSpec,
    text_task: TaskSpec,
    eval_tasks: Sequence[TaskSpec],
    vocab_size: int,
) -> StageResult:
    teacher_model, _teacher_opt = restore_phase_checkpoint(vocab_size, teacher_checkpoint, load_optimizer=False)
    set_model_base_only(teacher_model)
    ww.set_requires_grad(teacher_model.parameters(), False)

    student, _student_opt = restore_phase_checkpoint(vocab_size, unified_checkpoint, load_optimizer=False)
    ww.configure_branch_trainability(student, "water_weights_latent_adapter_only", anchor_a.old_frontier)
    optimizer = lateral.make_optimizer_for_adapter(student)
    latent_projectors = {block: projector.to(ww.DEVICE) for block, projector in anchor_a.latent_free_projectors.items()}
    final_metrics = evaluate_world(student, eval_tasks)
    best_metrics = dict(final_metrics)
    best_checkpoint = unified_checkpoint

    print(f"[reverse:{label}] adapter-only extraction for {REVERSE_STEPS} steps")
    for step in range(1, REVERSE_STEPS + 1):
        text_step = step % REVERSE_TEXT_BATCH_PERIOD != 1
        task = text_task if text_step else bracket_task
        batch = task.sample_train_batch(step)
        student.set_adapters_enabled(True)
        student.set_latent_free_projectors(latent_projectors, REVERSE_LATENT_PROJECTION)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_logits, _teacher_loss, teacher_states = lateral.forward_with_block_outputs(
                teacher_model, batch, detach=True
            )
        student_logits, task_loss, student_states = lateral.forward_with_block_outputs(student, batch, detach=False)
        if text_step:
            loss = (
                REVERSE_TEXT_KL_WEIGHT * lateral.distill_kl(student_logits, teacher_logits)
                + REVERSE_TEXT_HIDDEN_WEIGHT * lateral.hidden_lateral_loss(student_states, teacher_states)
                + REVERSE_TEXT_LOGIT_MSE_WEIGHT * F.mse_loss(student_logits, teacher_logits)
            )
        else:
            loss = (
                REVERSE_BRACKET_TASK_WEIGHT * task_loss
                + REVERSE_BRACKET_KL_WEIGHT * lateral.distill_kl(student_logits, teacher_logits)
                + REVERSE_BRACKET_HIDDEN_WEIGHT * lateral.hidden_lateral_loss(student_states, teacher_states)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ww.all_adapter_params(student), ww.GRAD_CLIP)
        optimizer.step()

        if step % REVERSE_EVAL_INTERVAL == 0 or step == REVERSE_STEPS:
            student.set_adapters_enabled(True)
            student.set_latent_free_projectors(latent_projectors, REVERSE_LATENT_PROJECTION)
            final_metrics = evaluate_world(student, eval_tasks)
            if step % REVERSE_LOG_INTERVAL == 0 or step == REVERSE_STEPS:
                print(
                    f"[reverse:{label}] step={step:04d}/{REVERSE_STEPS} "
                    f"{summarize_metrics(final_metrics)} latent_proj={REVERSE_LATENT_PROJECTION:.2f}"
                )
            if (
                reverse_extract_candidate_ok(final_metrics, retention_reference)
                and better_reverse_candidate(final_metrics, best_metrics)
            ):
                best_metrics = dict(final_metrics)
                best_checkpoint = make_phase_checkpoint(student, optimizer)

    checkpoint = best_checkpoint
    final_metrics = best_metrics
    del teacher_model, _teacher_opt, student, _student_opt, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return StageResult(
        label=label,
        checkpoint=checkpoint,
        metrics=final_metrics,
        latent_projection_steps=REVERSE_STEPS,
        base_only_verified=False,
    )


def collect_task_subspaces(
    checkpoint: Dict[str, object],
    vocab_size: int,
    task: TaskSpec,
    adapters_enabled: bool,
    latent_projectors: Dict[str, torch.Tensor] | None = None,
    latent_strength: float = 1.0,
) -> Dict[str, torch.Tensor]:
    model, _optimizer = restore_phase_checkpoint(vocab_size, checkpoint, load_optimizer=False)
    set_model_probe_mode(model, adapters_enabled, latent_projectors, latent_strength)

    act_cov = {block: torch.zeros(ww.D_MODEL, ww.D_MODEL) for block in ww.block_keys()}
    grad_cov = {block: torch.zeros(ww.D_MODEL, ww.D_MODEL) for block in ww.block_keys()}
    counts = {block: 0 for block in ww.block_keys()}

    for index in range(GEOM_BATCHES):
        batch = task.sample_anchor_batch(index)
        model.zero_grad(set_to_none=True)
        logits, _, activations = model(batch.x, batch.y, return_activations=True)
        loss = task_loss_from_logits(task, logits, batch)
        loss.backward()
        for block in ww.block_keys():
            block_index = int(block[1:])
            activation = activations[block_index].detach().reshape(-1, ww.D_MODEL).float().cpu()
            gradient = activations[block_index].grad
            if gradient is None:
                gradient_flat = torch.zeros_like(activation)
            else:
                gradient_flat = gradient.detach().reshape(-1, ww.D_MODEL).float().cpu()
            act_cov[block] += activation.t() @ activation / max(activation.shape[0], 1)
            grad_cov[block] += gradient_flat.t() @ gradient_flat / max(gradient_flat.shape[0], 1)
            counts[block] += 1

    spaces = {}
    for block in ww.block_keys():
        denom = max(counts[block], 1)
        act_basis = ww.top_cov_basis(act_cov[block] / denom, GEOM_ACT_RANK)
        grad_basis = ww.top_cov_basis(grad_cov[block] / denom, GEOM_GRAD_RANK)
        spaces[block] = ww.orthonormalize_columns([act_basis, grad_basis])

    del model, _optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return spaces


def subspace_pair_metrics(space_a: Dict[str, torch.Tensor], space_b: Dict[str, torch.Tensor]) -> Dict[str, float]:
    overlaps = []
    max_cosines = []
    min_angles = []
    for block in ww.block_keys():
        basis_a = space_a[block]
        basis_b = space_b[block]
        if basis_a.numel() == 0 or basis_b.numel() == 0:
            continue
        singular = torch.linalg.svdvals(basis_a.t() @ basis_b)
        if singular.numel() == 0:
            continue
        rank_norm = max(min(basis_a.shape[1], basis_b.shape[1]), 1)
        overlaps.append(float((singular.square().sum() / rank_norm).item()))
        peak = float(torch.clamp(singular.max(), 0.0, 1.0).item())
        max_cosines.append(peak)
        min_angles.append(float(math.degrees(math.acos(min(max(peak, 0.0), 1.0)))))
    if not overlaps:
        return {"overlap": float("nan"), "max_cos": float("nan"), "min_angle_deg": float("nan")}
    return {
        "overlap": float(np.mean(overlaps)),
        "max_cos": float(np.mean(max_cosines)),
        "min_angle_deg": float(np.mean(min_angles)),
    }


def save_heatmap(matrix: np.ndarray, labels: Sequence[str], title: str, path: Path) -> None:
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    image = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=max(float(matrix.max()), 1e-6))
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def geometry_report(
    seed: int,
    vocab_size: int,
    anchor_a: AnchorBundle,
    teacher_b: StageResult,
    anchor_ab: AnchorBundle,
    teacher_c: StageResult,
    base_abc: StageResult,
    bracket_task: TaskSpec,
    text_task: TaskSpec,
    arith_task: TaskSpec,
) -> List[Dict[str, object]]:
    teacher_spaces = {
        "A": collect_task_subspaces(anchor_a.checkpoint, vocab_size, bracket_task, adapters_enabled=False),
        "B": collect_task_subspaces(
            teacher_b.checkpoint,
            vocab_size,
            text_task,
            adapters_enabled=True,
            latent_projectors=anchor_a.latent_free_projectors,
        ),
        "C": collect_task_subspaces(
            teacher_c.checkpoint,
            vocab_size,
            arith_task,
            adapters_enabled=True,
            latent_projectors=anchor_ab.latent_free_projectors,
        ),
    }
    unified_spaces = {
        "A": collect_task_subspaces(base_abc.checkpoint, vocab_size, bracket_task, adapters_enabled=False),
        "B": collect_task_subspaces(base_abc.checkpoint, vocab_size, text_task, adapters_enabled=False),
        "C": collect_task_subspaces(base_abc.checkpoint, vocab_size, arith_task, adapters_enabled=False),
    }

    labels = ["A", "B", "C"]
    teacher_overlap = np.eye(len(labels))
    unified_overlap = np.eye(len(labels))
    rows: List[Dict[str, object]] = []

    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            teacher_metrics = subspace_pair_metrics(teacher_spaces[left], teacher_spaces[right])
            unified_metrics = subspace_pair_metrics(unified_spaces[left], unified_spaces[right])
            teacher_overlap[i, j] = teacher_metrics["overlap"]
            unified_overlap[i, j] = unified_metrics["overlap"]
            rows.append(
                {
                    "seed": seed,
                    "matrix": "teacher_overlap",
                    "left": left,
                    "right": right,
                    **teacher_metrics,
                }
            )
            rows.append(
                {
                    "seed": seed,
                    "matrix": "unified_overlap",
                    "left": left,
                    "right": right,
                    **unified_metrics,
                }
            )

    transfer = {}
    for label in labels:
        metrics = subspace_pair_metrics(teacher_spaces[label], unified_spaces[label])
        transfer[label] = metrics
        rows.append({"seed": seed, "matrix": "teacher_to_unified", "left": label, "right": label, **metrics})

    print("\n" + "-" * 78)
    print(f"SEED {seed} GEOMETRY")
    print("-" * 78)
    print("Teacher-space overlap (mean projector overlap):")
    for i, left in enumerate(labels):
        values = " ".join(f"{teacher_overlap[i, j]:.2f}" for j in range(len(labels)))
        print(f"  {left}: {values}")
    print("Unified-base task overlap (mean projector overlap):")
    for i, left in enumerate(labels):
        values = " ".join(f"{unified_overlap[i, j]:.2f}" for j in range(len(labels)))
        print(f"  {left}: {values}")
    print(
        "Same-task teacher->unified overlap: "
        + ", ".join(f"{label}={transfer[label]['overlap']:.2f}" for label in labels)
    )

    save_heatmap(
        teacher_overlap,
        labels,
        f"Teacher Task Overlap Seed {seed}",
        PLOT_DIR / f"teacher_overlap_seed_{seed}.png",
    )
    save_heatmap(
        unified_overlap,
        labels,
        f"Unified Task Overlap Seed {seed}",
        PLOT_DIR / f"unified_overlap_seed_{seed}.png",
    )
    return rows


def print_seed_summary(
    seed: int,
    anchor_metrics: Dict[str, float],
    teacher_b: StageResult,
    base_ab: StageResult,
    reverse_b: StageResult,
    teacher_c_attach: StageResult,
    teacher_c: StageResult,
    base_abc_consolidated: StageResult,
    base_abc: StageResult,
) -> None:
    print("\n" + "=" * 78)
    print(f"SEED {seed} PHASE-REVERSIBILITY RESULT")
    print("=" * 78)
    print(
        f"{'stage':30s} {'bracket_seq':>11s} {'text_loss':>10s} {'rev_acc':>10s} {'rev_prob':>10s} {'rev_seq':>10s} "
        f"{'replay':>9s} {'old_batch':>10s}"
    )

    def row(label: str, metrics: Dict[str, float], replay: str = "-", old_batch: str = "-") -> None:
        bracket_seq = metrics.get("bracket_seq", float("nan"))
        text_loss = metrics.get("text_loss", float("nan"))
        arith_acc = metrics.get("arith_acc", float("nan"))
        arith_problem_acc = metrics.get("arith_problem_acc", float("nan"))
        arith_seq = metrics.get("arith_seq", float("nan"))
        print(
            f"{label:30s} "
            f"{bracket_seq:11.3f} {text_loss:10.3f} {arith_acc:10.3f} {arith_problem_acc:10.3f} {arith_seq:10.3f} "
            f"{replay:>9s} {old_batch:>10s}"
        )

    row("base_A_anchor", anchor_metrics)
    row(
        "teacher_B_latent",
        teacher_b.metrics,
        replay=f"{teacher_b.replay_count}/{teacher_b.replay_budget}",
    )
    row(
        "base_AB_unified",
        base_ab.metrics,
        old_batch=f"{base_ab.old_batch_count}/{base_ab.old_batch_budget}",
    )
    row("reverse_text_extract", reverse_b.metrics)
    row(
        "teacher_C_attach",
        teacher_c_attach.metrics,
        replay=f"{teacher_c_attach.replay_count}/{teacher_c_attach.replay_budget}",
    )
    row(
        "teacher_C_sharpen",
        teacher_c.metrics,
        replay=f"{teacher_c.replay_count}/{teacher_c.replay_budget}",
    )
    row(
        "base_ABC_consolidated",
        base_abc_consolidated.metrics,
        old_batch=f"{base_abc_consolidated.old_batch_count}/{base_abc_consolidated.old_batch_budget}",
    )
    row(
        "base_ABC_unified",
        base_abc.metrics,
        old_batch=f"{base_abc.old_batch_count}/{base_abc.old_batch_budget}",
    )

    print(
        "Text removal on reverse extraction: "
        f"{reverse_b.metrics['text_loss'] - base_ab.metrics['text_loss']:+.3f} loss "
        f"while bracket changes by {reverse_b.metrics['bracket_seq'] - base_ab.metrics['bracket_seq']:+.3f}"
    )
    print(
        "Reversal sharpening delta: "
        f"rev_prob={teacher_c.metrics['arith_problem_acc'] - teacher_c_attach.metrics['arith_problem_acc']:+.3f} "
        f"rev_acc={teacher_c.metrics['arith_acc'] - teacher_c_attach.metrics['arith_acc']:+.3f} "
        f"while bracket changes by {teacher_c.metrics['bracket_seq'] - teacher_c_attach.metrics['bracket_seq']:+.3f}"
    )
    print(
        "Reversal transfer delta: "
        f"rev_prob={base_abc.metrics['arith_problem_acc'] - base_abc_consolidated.metrics['arith_problem_acc']:+.3f} "
        f"rev_acc={base_abc.metrics['arith_acc'] - base_abc_consolidated.metrics['arith_acc']:+.3f} "
        f"while bracket changes by {base_abc.metrics['bracket_seq'] - base_abc_consolidated.metrics['bracket_seq']:+.3f}"
    )
    print("=" * 78)


def result_rows_for_seed(
    seed: int,
    anchor_metrics: Dict[str, float],
    teacher_b: StageResult,
    base_ab: StageResult,
    reverse_b: StageResult,
    teacher_c_attach: StageResult,
    teacher_c: StageResult,
    base_abc_consolidated: StageResult,
    base_abc: StageResult,
) -> List[Dict[str, object]]:
    stages = {
        "base_A_anchor": anchor_metrics,
        "teacher_B_latent": teacher_b.metrics,
        "base_AB_unified": base_ab.metrics,
        "reverse_text_extract": reverse_b.metrics,
        "teacher_C_attach": teacher_c_attach.metrics,
        "teacher_C_sharpen": teacher_c.metrics,
        "base_ABC_consolidated": base_abc_consolidated.metrics,
        "base_ABC_unified": base_abc.metrics,
    }
    extras = {
        "teacher_B_latent": teacher_b,
        "base_AB_unified": base_ab,
        "reverse_text_extract": reverse_b,
        "teacher_C_attach": teacher_c_attach,
        "teacher_C_sharpen": teacher_c,
        "base_ABC_consolidated": base_abc_consolidated,
        "base_ABC_unified": base_abc,
    }
    rows = []
    for label, metrics in stages.items():
        extra = extras.get(label)
        rows.append(
            {
                "seed": seed,
                "stage": label,
                "bracket_seq": metrics.get("bracket_seq", float("nan")),
                "bracket_close": metrics.get("bracket_close", float("nan")),
                "text_loss": metrics.get("text_loss", float("nan")),
                "text_acc": metrics.get("text_acc", float("nan")),
                "arith_seq": metrics.get("arith_seq", float("nan")),
                "arith_acc": metrics.get("arith_acc", float("nan")),
                "arith_problem_acc": metrics.get("arith_problem_acc", float("nan")),
                "reverse_seq": metrics.get("reverse_seq", metrics.get("arith_seq", float("nan"))),
                "reverse_acc": metrics.get("reverse_acc", metrics.get("arith_acc", float("nan"))),
                "reverse_problem_acc": metrics.get("reverse_problem_acc", metrics.get("arith_problem_acc", float("nan"))),
                "replay_count": 0 if extra is None else extra.replay_count,
                "replay_budget": 0 if extra is None else extra.replay_budget,
                "old_batch_count": 0 if extra is None else extra.old_batch_count,
                "old_batch_budget": 0 if extra is None else extra.old_batch_budget,
                "z_viscosity_steps": 0 if extra is None else extra.z_viscosity_steps,
                "latent_projection_steps": 0 if extra is None else extra.latent_projection_steps,
                "base_only_verified": 0 if extra is None else int(extra.base_only_verified),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else float("nan")


def summarize_all(result_rows: List[Dict[str, object]], geometry_rows: List[Dict[str, object]]) -> None:
    by_stage: Dict[str, List[Dict[str, object]]] = {}
    for row in result_rows:
        by_stage.setdefault(str(row["stage"]), []).append(row)

    seeds_run = len({int(row["seed"]) for row in result_rows})
    ready_count = sum(1 for row in by_stage.get("base_A_anchor", []) if float(row["bracket_seq"]) >= ww.OLD_READY_SEQ)

    print("\n" + "=" * 78)
    print("PHASE-REVERSIBILITY SUMMARY ACROSS SEEDS")
    print("=" * 78)
    print(f"Seeds run: {seeds_run}")
    print(f"Old bracket seq>={ww.OLD_READY_SEQ:.2f} reached before attach: {ready_count}/{seeds_run}")

    def stage_mean(stage: str, key: str) -> float:
        return safe_mean([float(row[key]) for row in by_stage.get(stage, [])])

    print("\nMean stage metrics:")
    for stage in [
        "base_A_anchor",
        "teacher_B_latent",
        "base_AB_unified",
        "reverse_text_extract",
        "teacher_C_attach",
        "teacher_C_sharpen",
        "base_ABC_consolidated",
        "base_ABC_unified",
    ]:
        print(
            f"  {stage:24s} "
            f"bracket_seq={stage_mean(stage, 'bracket_seq'):.3f} "
            f"text_loss={stage_mean(stage, 'text_loss'):.3f} "
            f"rev_acc={stage_mean(stage, 'arith_acc'):.3f} "
            f"rev_prob={stage_mean(stage, 'arith_problem_acc'):.3f} "
            f"rev_seq={stage_mean(stage, 'arith_seq'):.3f}"
        )

    base_ab_bracket = [float(row["bracket_seq"]) for row in by_stage["base_AB_unified"]]
    reverse_bracket = [float(row["bracket_seq"]) for row in by_stage["reverse_text_extract"]]
    base_ab_text = [float(row["text_loss"]) for row in by_stage["base_AB_unified"]]
    reverse_text = [float(row["text_loss"]) for row in by_stage["reverse_text_extract"]]
    base_a_text = [float(row["text_loss"]) for row in by_stage["base_A_anchor"]]
    base_abc_bracket = [float(row["bracket_seq"]) for row in by_stage["base_ABC_unified"]]
    base_abc_text = [float(row["text_loss"]) for row in by_stage["base_ABC_unified"]]
    teacher_c_attach_arith_acc = [float(row["arith_acc"]) for row in by_stage["teacher_C_attach"]]
    teacher_c_attach_arith_prob = [float(row["arith_problem_acc"]) for row in by_stage["teacher_C_attach"]]
    teacher_c_arith_acc = [float(row["arith_acc"]) for row in by_stage["teacher_C_sharpen"]]
    teacher_c_arith_prob = [float(row["arith_problem_acc"]) for row in by_stage["teacher_C_sharpen"]]
    base_abc_consolidated_arith_acc = [float(row["arith_acc"]) for row in by_stage["base_ABC_consolidated"]]
    base_abc_consolidated_arith_prob = [float(row["arith_problem_acc"]) for row in by_stage["base_ABC_consolidated"]]
    base_abc_arith_acc = [float(row["arith_acc"]) for row in by_stage["base_ABC_unified"]]
    base_abc_arith_prob = [float(row["arith_problem_acc"]) for row in by_stage["base_ABC_unified"]]
    base_abc_arith = [float(row["arith_seq"]) for row in by_stage["base_ABC_unified"]]

    removal_fraction = []
    for base_loss, extracted_loss, anchor_loss in zip(base_ab_text, reverse_text, base_a_text):
        denom = max(anchor_loss - base_loss, 1e-12)
        removal_fraction.append((extracted_loss - base_loss) / denom)

    print("\nKey effects:")
    print(
        f"  Reverse extraction text removal fraction vs base_A: "
        f"{safe_mean(removal_fraction):.3f}"
    )
    print(
        f"  Reverse extraction bracket delta vs base_AB: "
        f"{safe_mean([r - b for r, b in zip(reverse_bracket, base_ab_bracket)]):+.3f}"
    )
    print(
        f"  Tri-task unified base_ABC: "
        f"bracket_seq={safe_mean(base_abc_bracket):.3f} "
        f"text_loss={safe_mean(base_abc_text):.3f} "
        f"rev_acc={safe_mean(base_abc_arith_acc):.3f} "
        f"rev_prob={safe_mean(base_abc_arith_prob):.3f} "
        f"rev_seq={safe_mean(base_abc_arith):.3f}"
    )
    print(
        f"  Teacher C reversal attach quality: "
        f"rev_acc={safe_mean(teacher_c_attach_arith_acc):.3f} "
        f"rev_prob={safe_mean(teacher_c_attach_arith_prob):.3f}"
    )
    print(
        f"  Teacher C sharpened quality: "
        f"rev_acc={safe_mean(teacher_c_arith_acc):.3f} "
        f"rev_prob={safe_mean(teacher_c_arith_prob):.3f}"
    )
    print(
        f"  Reversal sharpen delta: "
        f"rev_acc={safe_mean([s - a for s, a in zip(teacher_c_arith_acc, teacher_c_attach_arith_acc)]):+.3f} "
        f"rev_prob={safe_mean([s - a for s, a in zip(teacher_c_arith_prob, teacher_c_attach_arith_prob)]):+.3f}"
    )
    print(
        f"  Reversal transfer delta: "
        f"rev_acc={safe_mean([u - c for u, c in zip(base_abc_arith_acc, base_abc_consolidated_arith_acc)]):+.3f} "
        f"rev_prob={safe_mean([u - c for u, c in zip(base_abc_arith_prob, base_abc_consolidated_arith_prob)]):+.3f}"
    )

    if geometry_rows:
        def geom_values(matrix: str, left: str, right: str, key: str) -> List[float]:
            return [
                float(row[key])
                for row in geometry_rows
                if row["matrix"] == matrix and row["left"] == left and row["right"] == right
            ]

        print("\nGeometry:")
        for pair in [("A", "B"), ("A", "C"), ("B", "C")]:
            teacher_overlap = safe_mean(geom_values("teacher_overlap", pair[0], pair[1], "overlap"))
            unified_overlap = safe_mean(geom_values("unified_overlap", pair[0], pair[1], "overlap"))
            print(
                f"  {pair[0]} vs {pair[1]} teacher_overlap={teacher_overlap:.3f} "
                f"unified_overlap={unified_overlap:.3f}"
            )
        transfer = {
            label: safe_mean(geom_values("teacher_to_unified", label, label, "overlap"))
            for label in ["A", "B", "C"]
        }
        print(
            "  Same-task teacher->unified overlap: "
            + ", ".join(f"{label}={value:.3f}" for label, value in transfer.items())
        )

    arith_teacher_ok = safe_mean(teacher_c_arith_prob) >= 0.10 or safe_mean(teacher_c_arith_acc) >= ARITH_MIXED_ANSWER_ACC
    base_abc_ok = safe_mean(base_abc_bracket) >= 0.90 and safe_mean(base_abc_arith_prob) >= 0.30
    base_abc_mixed = safe_mean(base_abc_bracket) >= 0.85 and safe_mean(base_abc_arith_prob) >= 0.15
    reverse_ok = safe_mean(removal_fraction) >= 0.75
    if ready_count < seeds_run:
        interpretation = (
            "mechanics-only/diagnostic: not enough seeds learned the starting old skill, so the full claims "
            "are not scored from this run."
        )
    elif not arith_teacher_ok:
        interpretation = (
            "diagnostic: reversal attach is still underpowered, so the tri-task result is bottlenecked "
            "before consolidation."
        )
    elif base_abc_ok and reverse_ok:
        interpretation = (
            "strong pass: the unified base keeps prior skills, absorbs a third task, "
            "and the earlier text skill is partially reversible with an adapter extraction."
        )
    elif base_abc_ok or base_abc_mixed:
        interpretation = (
            "mixed-positive: tri-task consolidation works, but reverse extraction is not yet clean enough."
        )
    else:
        interpretation = (
            "diagnostic: at least one of tri-task consolidation or reverse extraction still needs work."
        )
    print(f"\nInterpretation: {interpretation}")
    print(f"CSV saved to: {CSV_PATH}")
    print(f"Geometry CSV saved to: {GEOM_CSV_PATH}")
    if plt is not None:
        print(f"Plots saved to: {PLOT_DIR}")
    print("=" * 78)


def run_seed(
    seed: int,
    vocab_size: int,
    stoi: Dict[str, int],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    print("\n" + "#" * 78)
    print(f"BEGIN PHASE-REVERSIBILITY SEED {seed}")
    print("#" * 78)
    ww.set_seed(seed)

    text_eval_positions = make_text_eval_positions(len(val_data), seed)
    bracket_eval_batches = ww.make_fixed_bracket_batches(
        stoi, seed + 30_000, ww.BRACKET_EVAL_BATCHES, ww.BRACKET_EVAL_BATCH
    )
    arithmetic_eval_batches = make_fixed_arithmetic_batches(
        stoi, seed + 31_000, ww.BRACKET_EVAL_BATCHES, ww.BRACKET_EVAL_BATCH
    )

    bracket_task = make_bracket_task(stoi, bracket_eval_batches, seed)
    text_task = make_text_task(train_data, val_data, text_eval_positions, seed)
    arith_task = make_arithmetic_task(stoi, arithmetic_eval_batches, seed)

    anchor = ww.train_old_skill(
        vocab_size,
        stoi,
        seed,
        bracket_eval_batches,
        ww.make_fixed_bracket_batches(stoi, seed + 40_000, 1, ww.PROBE_BATCH)[0],
    )
    anchor_a = anchor_from_old_skill(anchor)
    base_a_model, _base_a_opt = restore_phase_checkpoint(vocab_size, anchor_a.checkpoint, load_optimizer=False)
    lateral.set_student_base_only(base_a_model, trainable_base=False)
    anchor_metrics = evaluate_world(base_a_model, [bracket_task, text_task])
    del base_a_model, _base_a_opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher_b = attach_latent_teacher(
        "B_text",
        anchor_a,
        vocab_size,
        [bracket_task],
        text_task,
        [bracket_task, text_task],
        seed,
    )
    base_ab = consolidate_dual_teacher(
        "AB",
        AB_CONSOLIDATION_BRANCH,
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
    reverse_b = reverse_extract_to_base_a(
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

    anchor_ab = collect_world_anchor("AB", base_ab.checkpoint, [bracket_task, text_task], vocab_size)
    teacher_c_attach = attach_latent_teacher(
        "C_reverse",
        anchor_ab,
        vocab_size,
        [bracket_task, text_task],
        arith_task,
        [bracket_task, text_task, arith_task],
        seed + 77,
    )
    teacher_c = sharpen_arithmetic_teacher(
        "C_reverse",
        teacher_c_attach,
        anchor_ab,
        vocab_size,
        [bracket_task, text_task],
        arith_task,
        [bracket_task, text_task, arith_task],
    )
    base_abc_consolidated = consolidate_dual_teacher(
        "ABC",
        ABC_CONSOLIDATION_BRANCH,
        anchor_ab.checkpoint,
        anchor_ab,
        anchor_ab.checkpoint,
        [bracket_task, text_task],
        teacher_c.checkpoint,
        {block: projector.to(ww.DEVICE) for block, projector in anchor_ab.latent_free_projectors.items()},
        arith_task,
        [bracket_task, text_task, arith_task],
        vocab_size,
        selection_reference=base_ab.metrics,
    )
    base_abc = polish_arithmetic_transfer(
        "ABC",
        base_abc_consolidated,
        anchor_ab,
        vocab_size,
        [bracket_task, text_task],
        arith_task,
        [bracket_task, text_task, arith_task],
        teacher_c.checkpoint,
        {block: projector.to(ww.DEVICE) for block, projector in anchor_ab.latent_free_projectors.items()},
        teacher_c.metrics,
        base_ab.metrics,
    )

    geometry_rows = geometry_report(
        seed,
        vocab_size,
        anchor_a,
        teacher_b,
        anchor_ab,
        teacher_c,
        base_abc,
        bracket_task,
        text_task,
        arith_task,
    )

    if ww.SMOKE:
        assert base_ab.base_only_verified, "base_AB is not base-only"
        assert base_abc.base_only_verified, "base_ABC is not base-only"
        assert teacher_b.replay_count > 0, "teacher B did not replay old task"
        assert teacher_c_attach.replay_count > 0, "teacher C did not replay old world"
        assert reverse_b.latent_projection_steps == REVERSE_STEPS, "reverse extraction latent projection did not stay active"
        assert any(row["matrix"] == "teacher_to_unified" for row in geometry_rows), "geometry transfer rows missing"

    print_seed_summary(
        seed,
        anchor_metrics,
        teacher_b,
        base_ab,
        reverse_b,
        teacher_c_attach,
        teacher_c,
        base_abc_consolidated,
        base_abc,
    )
    return result_rows_for_seed(
        seed,
        anchor_metrics,
        teacher_b,
        base_ab,
        reverse_b,
        teacher_c_attach,
        teacher_c,
        base_abc_consolidated,
        base_abc,
    ), geometry_rows


def main() -> None:
    ww.set_seed(LAB_SEEDS[0])
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 78)
    print("PHASE-REVERSIBILITY LAB: TRI-TASK + GEOMETRY + REVERSE EXTRACTION")
    print("=" * 78)
    print("Question: can task manifolds be separated, consolidated, measured, and partially reversed?")
    print(f"Device: {ww.DEVICE}")
    print(f"Seeds: {LAB_SEEDS}")
    print(f"Phase A: bracket until seq>={ww.OLD_READY_SEQ:.2f} or {ww.PHASE_A_MAX_STEPS} steps")
    print(f"Phase B: latent text attach for {ww.PHASE_B_STEPS} steps")
    print(f"Phase C: dual-teacher consolidation for {lateral.CONSOLIDATION_STEPS} steps")
    print(f"Phase D: latent reversal attach for {ww.PHASE_B_STEPS} steps")
    print(f"Phase D2: reversal sharpen for {ARITH_SHARPEN_STEPS} steps")
    print(f"Phase E: dual-teacher consolidation for {lateral.CONSOLIDATION_STEPS} steps")
    print(f"Phase E2: base-only reversal transfer polish for {ARITH_TRANSFER_POLISH_STEPS} steps")
    print(f"Reverse extraction: adapter-only for {REVERSE_STEPS} steps")
    print(
        f"Model: d={ww.D_MODEL}, layers={ww.N_LAYER}, heads={ww.N_HEAD}, "
        f"block={ww.BLOCK_SIZE}, adapter_rank={ww.ADAPTER_RANK}"
    )
    print(
        f"Geometry: batches={GEOM_BATCHES}, act_rank={GEOM_ACT_RANK}, grad_rank={GEOM_GRAD_RANK}; "
        f"AB branch={AB_CONSOLIDATION_BRANCH}, ABC branch={ABC_CONSOLIDATION_BRANCH}"
    )
    print(
        f"Reversal: eval_len={ARITH_EVAL_DIGITS}, attach_curriculum={ARITH_ATTACH_DIGIT_STAGES}, "
        f"answer_weight={ARITH_ANSWER_WEIGHT:.2f}, context_weight={ARITH_CONTEXT_WEIGHT:.2f}, "
        f"single_problem={ARITH_SINGLE_PROBLEM}, replay_frac={ARITH_REPLAY_BUDGET_FRACTION:.2f}, "
        f"escape_levels={ARITH_ESCAPE_LEVELS}, adapter_rank={ARITH_ADAPTER_RANK}, old_compat=("
        f"{ARITH_OLD_COMPAT_TASK_WEIGHT:.2f},{ARITH_OLD_COMPAT_KL_WEIGHT:.2f},{ARITH_OLD_COMPAT_HIDDEN_WEIGHT:.2f}), "
        f"compat_anneal=({ARITH_OLD_COMPAT_MID_FACTOR:.2f},{ARITH_OLD_COMPAT_LATE_FACTOR:.2f}), "
        f"sharpen=(steps={ARITH_SHARPEN_STEPS}, target_prob={ARITH_SHARPEN_TARGET_PROB:.2f}, "
        f"target_acc={ARITH_SHARPEN_TARGET_ACC:.2f}, lr_scale={ARITH_SHARPEN_LR_SCALE:.2f}), "
        f"transfer=(steps={ARITH_TRANSFER_POLISH_STEPS}, target_fraction={ARITH_TRANSFER_TARGET_FRACTION:.2f}, "
        f"lr_scale={ARITH_TRANSFER_POLISH_LR_SCALE:.2f}, old_period={ARITH_TRANSFER_POLISH_OLD_PERIOD})"
    )

    text = ww.download_or_load_text()
    stoi, _itos = build_joint_vocab(text)
    encoded = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split = int(len(encoded) * ww.TRAIN_FRACTION)
    train_data = encoded[:split]
    val_data = encoded[split:]
    print(f"Vocab size: {len(stoi)}")
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    all_rows: List[Dict[str, object]] = []
    geometry_rows: List[Dict[str, object]] = []
    start = time.time()
    for seed in LAB_SEEDS:
        seed_rows, seed_geom = run_seed(seed, len(stoi), stoi, train_data, val_data)
        all_rows.extend(seed_rows)
        geometry_rows.extend(seed_geom)
        write_csv(CSV_PATH, all_rows)
        write_csv(GEOM_CSV_PATH, geometry_rows)

    summarize_all(all_rows, geometry_rows)
    write_csv(CSV_PATH, all_rows)
    write_csv(GEOM_CSV_PATH, geometry_rows)
    print(f"Total wall time: {format_seconds(time.time() - start)}")


if __name__ == "__main__":
    main()
