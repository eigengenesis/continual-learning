#!/usr/bin/env python3
"""
Water Weights lateral-consolidation benchmark.

Run:
    python colab_water_weights_lateral_consolidation_benchmark.py

Hidden smoke mode:
    CHAOS_SMOKE=1 python colab_water_weights_lateral_consolidation_benchmark.py

Core question:
Can we learn a new skill in solid-base latent adapters, then laterally
propagate that skill back into a base-only student so adapters can be removed
without catastrophic forgetting?

Cycle tested:
    SOLID old base
      -> ATTACH text skill in latent-routed adapters
      -> LATERAL CONSOLIDATION into a base-only student
      -> SOLID unified v2 base
"""

from __future__ import annotations

import copy
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import colab_water_weights_benchmark as ww


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "water_weights_lateral_consolidation_results.csv"
SMOKE = os.environ.get("CHAOS_SMOKE", "0") == "1"

CONSOLIDATION_STEPS = 600 if torch.cuda.is_available() else 160
CONSOLIDATION_LOG_INTERVAL = 100 if torch.cuda.is_available() else 40
CONSOLIDATION_EVAL_INTERVAL = 100 if torch.cuda.is_available() else 40
CONSOLIDATION_OLD_FRACTION = 0.50
CONSOLIDATION_LR = 1.0e-4
CONSOLIDATION_STRONG_LR = 7.5e-5
DISTILL_TEMPERATURE = 2.0

if SMOKE:
    CONSOLIDATION_STEPS = min(CONSOLIDATION_STEPS, 36)
    CONSOLIDATION_LOG_INTERVAL = min(CONSOLIDATION_LOG_INTERVAL, 12)
    CONSOLIDATION_EVAL_INTERVAL = min(CONSOLIDATION_EVAL_INTERVAL, 12)


CONSOLIDATION_BRANCHES = [
    "no_consolidation",
    "dual_lateral_balanced",
    "dual_lateral_freeze_frontier",
    "dual_lateral_z_viscous",
    "dual_lateral_strong",
]


@dataclass
class TeacherResult:
    checkpoint: Dict[str, object]
    rows: List[ww.BranchRow]
    final_old_seq: float
    final_old_close: float
    final_text_loss: float
    final_text_acc: float
    old_seq_auc: float
    text_loss_auc: float
    replay_count: int
    replay_budget: int
    latent_projection_steps: int


@dataclass
class ConsolidationResult:
    name: str
    final_old_seq: float
    final_old_close: float
    final_text_loss: float
    final_text_acc: float
    old_seq_auc: float
    text_loss_auc: float
    replay_count: int
    replay_budget: int
    z_viscosity_steps: int
    base_train_steps: int
    adapter_enabled: bool
    base_only_verified: bool


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    clean = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if clean.size == 0:
        return float("nan"), float("nan")
    return float(clean.mean()), float(clean.std(ddof=0))


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    if not math.isfinite(mean):
        return "nan +/- nan"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def format_step(value: float) -> str:
    if not math.isfinite(value):
        return "miss"
    return str(int(value))


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {sec:02d}s"
    return f"{minutes:02d}m {sec:02d}s"


def bootstrap_ci(values: Sequence[float], samples: int = ww.BOOTSTRAP_SAMPLES) -> Tuple[float, float]:
    clean = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(12345)
    means = []
    for _ in range(samples):
        sample = rng.choice(clean, size=clean.size, replace=True)
        means.append(float(sample.mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def format_ci(lo: float, hi: float, digits: int = 3) -> str:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return "[nan, nan]"
    return f"[{lo:+.{digits}f}, {hi:+.{digits}f}]"


def set_trainable(params: Iterable[torch.nn.Parameter], enabled: bool) -> None:
    for param in params:
        param.requires_grad = enabled


def set_teacher_mode(
    model: ww.TinyGPT,
    latent_projectors: Dict[str, torch.Tensor],
    latent_strength: float = 1.0,
) -> None:
    model.set_adapters_enabled(True)
    model.set_latent_free_projectors(latent_projectors, latent_strength)
    model.eval()


def set_student_base_only(model: ww.TinyGPT, trainable_base: bool) -> None:
    model.set_adapters_enabled(False)
    model.clear_latent_free_projectors()
    set_trainable(ww.all_adapter_params(model), False)
    set_trainable(ww.all_base_params(model), trainable_base)


def forward_with_block_outputs(
    model: ww.TinyGPT,
    batch: ww.Batch,
    detach: bool,
):
    outputs: List[torch.Tensor] = []
    hooks = []

    def save_output(_module, _inputs, output):
        outputs.append(output.detach() if detach else output)

    for block in model.blocks:
        hooks.append(block.register_forward_hook(save_output))
    try:
        logits, loss = model(batch.x, batch.y)
    finally:
        for hook in hooks:
            hook.remove()
    return logits, loss, outputs


def distill_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    temp = DISTILL_TEMPERATURE
    s = student_logits.reshape(-1, student_logits.size(-1)) / temp
    t = teacher_logits.reshape(-1, teacher_logits.size(-1)) / temp
    return F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="batchmean") * (temp * temp)


def hidden_lateral_loss(student_states: Sequence[torch.Tensor], teacher_states: Sequence[torch.Tensor]) -> torch.Tensor:
    if not student_states or not teacher_states:
        return torch.tensor(0.0, device=ww.DEVICE)
    losses = []
    for student, teacher in zip(student_states, teacher_states):
        losses.append(F.mse_loss(student, teacher.detach()))
    return torch.stack(losses).mean()


def z_shock_for_model(model: ww.TinyGPT, anchor: ww.AnchorInfo, bracket_probe_batch: ww.Batch) -> float:
    old_z, old_act_z = ww.probe_z(model, bracket_probe_batch)
    current_block_z = ww.block_combined_z_map(old_z, old_act_z)
    shocks = []
    for block in anchor.old_frontier:
        current = max(current_block_z.get(block, 1e-12), 1e-12)
        base = max(anchor.block_anchor_z.get(block, 1e-12), 1e-12)
        shocks.append(abs(math.log2(current / base)))
    return float(np.mean(shocks)) if shocks else 0.0


def old_batch_schedule(branch_name: str, step: int) -> bool:
    if branch_name == "dual_lateral_strong":
        return step % 3 != 0
    if branch_name in {
        "dual_lateral_balanced",
        "dual_lateral_freeze_frontier",
        "dual_lateral_z_viscous",
    }:
        return step % 2 == 1
    return False


def expected_old_batches(branch_name: str) -> int:
    return sum(1 for step in range(1, CONSOLIDATION_STEPS + 1) if old_batch_schedule(branch_name, step))


def consolidation_weights(branch_name: str, old_step: bool) -> Tuple[float, float, float]:
    if branch_name == "dual_lateral_strong":
        return (4.0, 1.25, 1.00) if old_step else (0.75, 0.50, 0.20)
    if branch_name == "dual_lateral_z_viscous":
        return (2.5, 1.00, 0.75) if old_step else (1.00, 0.50, 0.20)
    if branch_name in {"dual_lateral_balanced", "dual_lateral_freeze_frontier"}:
        return (2.0, 0.75, 0.50) if old_step else (1.00, 0.50, 0.20)
    return 1.0, 0.50, 0.20


def apply_consolidation_viscosity(
    model: ww.TinyGPT,
    anchor: ww.AnchorInfo,
    branch_name: str,
    current_z_viscosity: float,
) -> bool:
    if branch_name not in {"dual_lateral_z_viscous", "dual_lateral_strong"}:
        return False
    multiplier = min(current_z_viscosity, 0.10 if branch_name == "dual_lateral_strong" else 1.0)
    for block in anchor.old_frontier:
        ww.multiply_grads(ww.base_block_params(model, block), multiplier)
    return True


def make_optimizer_for_base(model: ww.TinyGPT, lr: float) -> torch.optim.Optimizer:
    params = [param for param in ww.all_base_params(model) if param.requires_grad]
    return torch.optim.AdamW(params, lr=lr, betas=ww.BETAS, weight_decay=ww.WEIGHT_DECAY)


def make_optimizer_for_adapter(model: ww.TinyGPT) -> torch.optim.Optimizer:
    params = [param for param in ww.all_adapter_params(model) if param.requires_grad]
    return torch.optim.AdamW(params, lr=ww.BASE_LR * ww.ADAPTER_LR_MULT, betas=ww.BETAS, weight_decay=ww.WEIGHT_DECAY)


def probe_adapter_teacher(
    model: ww.TinyGPT,
    val_data: torch.Tensor,
    text_eval_positions: torch.Tensor,
    bracket_eval_batches: List[ww.Batch],
    bracket_probe_batch: ww.Batch,
    anchor: ww.AnchorInfo,
    step: int,
    replay_count: int,
    latent_projection: float,
) -> ww.BranchRow:
    return ww.probe_branch_row(
        model,
        val_data,
        text_eval_positions,
        bracket_eval_batches,
        bracket_probe_batch,
        anchor,
        step,
        viscosity=1.0,
        replay_count=replay_count,
        adapter_enabled=True,
        grad_projection=0.0,
        latent_projection=latent_projection,
    )


def train_latent_adapter_teacher(
    anchor: ww.AnchorInfo,
    vocab_size: int,
    stoi: Dict[str, int],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    text_train_positions: torch.Tensor,
    text_eval_positions: torch.Tensor,
    bracket_eval_batches: List[ww.Batch],
    bracket_probe_batch: ww.Batch,
    seed: int,
) -> TeacherResult:
    model, optimizer = ww.restore_model_from_checkpoint(vocab_size, anchor.checkpoint, load_optimizer=False)
    latent_projectors = {block: projector.to(ww.DEVICE) for block, projector in anchor.latent_free_projectors.items()}

    print("[teacher:latent_adapter_only] momentum reset + old-data reminiscence")
    ww.run_reminiscence(model, optimizer, stoi, seed)
    ww.configure_branch_trainability(model, "water_weights_latent_adapter_only", anchor.old_frontier)
    assert not any(param.requires_grad for param in ww.all_base_params(model)), "teacher base should be frozen"
    assert any(param.requires_grad for param in ww.all_adapter_params(model)), "teacher adapters should be trainable"
    optimizer = make_optimizer_for_adapter(model)

    replay_budget = int(ww.PHASE_B_STEPS * ww.REPLAY_BUDGET_FRACTION)
    replay_count = 0
    current_latent_projection = ww.INITIAL_LATENT_PROJECTION
    latent_projection_steps = 0
    rows: List[ww.BranchRow] = []
    model.set_latent_free_projectors(
        latent_projectors,
        ww.latent_projection_strength_for_branch("water_weights_latent_adapter_only", current_latent_projection),
    )
    rows.append(
        probe_adapter_teacher(
            model,
            val_data,
            text_eval_positions,
            bracket_eval_batches,
            bracket_probe_batch,
            anchor,
            0,
            replay_count,
            current_latent_projection,
        )
    )

    for step in range(1, ww.PHASE_B_STEPS + 1):
        replay_this_step = ww.should_replay("water_weights_latent_adapter_only", step, replay_count, replay_budget)
        if replay_this_step:
            batch = ww.replay_batch_for_index(stoi, seed, replay_count)
            replay_count += 1
        else:
            batch = ww.text_batch_from_positions(train_data, text_train_positions[step - 1])

        latent_strength = ww.latent_projection_strength_for_branch(
            "water_weights_latent_adapter_only",
            current_latent_projection,
        )
        model.set_latent_free_projectors(latent_projectors, latent_strength)
        latent_projection_steps += int(latent_strength > 0.0)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(batch.x, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ww.all_adapter_params(model), ww.GRAD_CLIP)
        optimizer.step()

        if ww.is_probe_step(step):
            row = probe_adapter_teacher(
                model,
                val_data,
                text_eval_positions,
                bracket_eval_batches,
                bracket_probe_batch,
                anchor,
                step,
                replay_count,
                latent_strength,
            )
            rows.append(row)
            current_latent_projection = ww.latent_projection_from_shock(row.z_shock)
            if step % ww.BRANCH_LOG_INTERVAL == 0 or step == ww.PHASE_B_STEPS:
                print(
                    f"[teacher:latent_adapter_only] step={step:04d}/{ww.PHASE_B_STEPS} "
                    f"old_seq={row.old_seq_acc:.3f} text_loss={row.text_loss:.3f} "
                    f"z_shock={row.z_shock:.2f} latent_proj={latent_strength:.2f} "
                    f"replay={replay_count}/{replay_budget}"
                )

    final = rows[-1]
    result = TeacherResult(
        checkpoint=ww.make_checkpoint(model, optimizer),
        rows=rows,
        final_old_seq=final.old_seq_acc,
        final_old_close=final.old_close_acc,
        final_text_loss=final.text_loss,
        final_text_acc=final.text_acc,
        old_seq_auc=ww.auc_rows(rows, "old_seq_acc"),
        text_loss_auc=ww.auc_rows(rows, "text_loss"),
        replay_count=replay_count,
        replay_budget=replay_budget,
        latent_projection_steps=latent_projection_steps,
    )
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def evaluate_base_only(
    model: ww.TinyGPT,
    val_data: torch.Tensor,
    text_eval_positions: torch.Tensor,
    bracket_eval_batches: List[ww.Batch],
) -> Tuple[float, float, float, float]:
    model.set_adapters_enabled(False)
    model.clear_latent_free_projectors()
    old = ww.evaluate_bracket(model, bracket_eval_batches)
    text = ww.evaluate_text(model, val_data, text_eval_positions)
    return old["seq_acc"], old["close_acc"], text["loss"], text["acc"]


def no_consolidation_result(
    teacher: TeacherResult,
    anchor: ww.AnchorInfo,
    vocab_size: int,
    val_data: torch.Tensor,
    text_eval_positions: torch.Tensor,
    bracket_eval_batches: List[ww.Batch],
) -> ConsolidationResult:
    model, _optimizer = ww.restore_model_from_checkpoint(vocab_size, teacher.checkpoint, load_optimizer=False)
    old_seq, old_close, text_loss, text_acc = evaluate_base_only(model, val_data, text_eval_positions, bracket_eval_batches)
    del model, _optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ConsolidationResult(
        name="no_consolidation",
        final_old_seq=old_seq,
        final_old_close=old_close,
        final_text_loss=text_loss,
        final_text_acc=text_acc,
        old_seq_auc=old_seq,
        text_loss_auc=text_loss,
        replay_count=0,
        replay_budget=expected_old_batches("dual_lateral_balanced"),
        z_viscosity_steps=0,
        base_train_steps=0,
        adapter_enabled=False,
        base_only_verified=True,
    )


def consolidate_student(
    branch_name: str,
    teacher: TeacherResult,
    anchor: ww.AnchorInfo,
    vocab_size: int,
    stoi: Dict[str, int],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    text_train_positions: torch.Tensor,
    text_eval_positions: torch.Tensor,
    bracket_eval_batches: List[ww.Batch],
    bracket_probe_batch: ww.Batch,
    seed: int,
) -> ConsolidationResult:
    adapter_teacher_model, _adapter_teacher_opt = ww.restore_model_from_checkpoint(
        vocab_size, teacher.checkpoint, load_optimizer=False
    )
    latent_projectors = {block: projector.to(ww.DEVICE) for block, projector in anchor.latent_free_projectors.items()}
    set_teacher_mode(adapter_teacher_model, latent_projectors, latent_strength=1.0)
    set_trainable(adapter_teacher_model.parameters(), False)

    old_teacher_model, _old_teacher_opt = ww.restore_model_from_checkpoint(
        vocab_size, anchor.checkpoint, load_optimizer=False
    )
    set_student_base_only(old_teacher_model, trainable_base=False)
    old_teacher_model.eval()
    set_trainable(old_teacher_model.parameters(), False)

    student, _student_opt = ww.restore_model_from_checkpoint(vocab_size, anchor.checkpoint, load_optimizer=False)
    set_student_base_only(student, trainable_base=True)
    if branch_name == "dual_lateral_freeze_frontier":
        for block in anchor.old_frontier:
            set_trainable(ww.base_block_params(student, block), False)
    optimizer = make_optimizer_for_base(
        student,
        CONSOLIDATION_STRONG_LR if branch_name == "dual_lateral_strong" else CONSOLIDATION_LR,
    )

    old_batch_budget = expected_old_batches(branch_name)
    old_batch_count = 0
    rows_old_seq = []
    rows_text_loss = []
    z_viscosity_steps = 0
    current_z_viscosity = ww.INITIAL_Z_VISCOSITY

    print(
        f"[consolidate:{branch_name}] dual-teacher base-only student training for "
        f"{CONSOLIDATION_STEPS} steps"
    )

    for step in range(1, CONSOLIDATION_STEPS + 1):
        old_step = old_batch_schedule(branch_name, step)
        if old_step:
            batch = ww.replay_batch_for_index(stoi, seed + 50_000, old_batch_count)
            old_batch_count += 1
            active_teacher = old_teacher_model
            active_teacher.set_adapters_enabled(False)
            active_teacher.clear_latent_free_projectors()
        else:
            position_index = (step - 1) % text_train_positions.shape[0]
            batch = ww.text_batch_from_positions(train_data, text_train_positions[position_index])
            active_teacher = adapter_teacher_model
            set_teacher_mode(active_teacher, latent_projectors, latent_strength=1.0)

        with torch.no_grad():
            teacher_logits, _teacher_loss, teacher_states = forward_with_block_outputs(
                active_teacher, batch, detach=True
            )

        optimizer.zero_grad(set_to_none=True)
        student_logits, task_loss, student_states = forward_with_block_outputs(student, batch, detach=False)
        task_weight, kl_weight, hidden_weight = consolidation_weights(branch_name, old_step)
        kl_loss = distill_kl(student_logits, teacher_logits)
        hidden_loss = hidden_lateral_loss(student_states, teacher_states)
        loss = task_weight * task_loss + kl_weight * kl_loss + hidden_weight * hidden_loss
        loss.backward()
        if apply_consolidation_viscosity(student, anchor, branch_name, current_z_viscosity):
            z_viscosity_steps += 1
        torch.nn.utils.clip_grad_norm_(ww.all_base_params(student), ww.GRAD_CLIP)
        optimizer.step()

        if step % CONSOLIDATION_EVAL_INTERVAL == 0 or step == CONSOLIDATION_STEPS:
            old_seq, _old_close, text_loss, _text_acc = evaluate_base_only(
                student, val_data, text_eval_positions, bracket_eval_batches
            )
            rows_old_seq.append((step, old_seq))
            rows_text_loss.append((step, text_loss))
            if branch_name in {"dual_lateral_z_viscous", "dual_lateral_strong"}:
                current_z_viscosity = ww.viscosity_from_shock(z_shock_for_model(student, anchor, bracket_probe_batch))
            if step % CONSOLIDATION_LOG_INTERVAL == 0 or step == CONSOLIDATION_STEPS:
                print(
                    f"[consolidate:{branch_name}] step={step:04d}/{CONSOLIDATION_STEPS} "
                    f"old_seq={old_seq:.3f} text_loss={text_loss:.3f} "
                    f"old_batches={old_batch_count}/{old_batch_budget} "
                    f"viscosity={current_z_viscosity:.3f}"
                )

    final_old_seq, final_old_close, final_text_loss, final_text_acc = evaluate_base_only(
        student, val_data, text_eval_positions, bracket_eval_batches
    )
    old_auc = float(np.mean([value for _step, value in rows_old_seq])) if rows_old_seq else final_old_seq
    text_auc = float(np.mean([value for _step, value in rows_text_loss])) if rows_text_loss else final_text_loss
    base_only_verified = (
        not any(param.requires_grad for param in ww.all_adapter_params(student))
        and not student.blocks[0].adapter_enabled
    )

    del (
        adapter_teacher_model,
        _adapter_teacher_opt,
        old_teacher_model,
        _old_teacher_opt,
        student,
        _student_opt,
        optimizer,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ConsolidationResult(
        name=branch_name,
        final_old_seq=final_old_seq,
        final_old_close=final_old_close,
        final_text_loss=final_text_loss,
        final_text_acc=final_text_acc,
        old_seq_auc=old_auc,
        text_loss_auc=text_auc,
        replay_count=old_batch_count,
        replay_budget=old_batch_budget,
        z_viscosity_steps=z_viscosity_steps,
        base_train_steps=CONSOLIDATION_STEPS,
        adapter_enabled=False,
        base_only_verified=base_only_verified,
    )


def print_seed_summary(
    seed: int,
    anchor: ww.AnchorInfo,
    teacher: TeacherResult,
    baseline_base_text_loss: float,
    results: Dict[str, ConsolidationResult],
) -> None:
    print("\n" + "=" * 78)
    print(f"SEED {seed} LATERAL CONSOLIDATION RESULT")
    print("=" * 78)
    print(
        f"Anchor: step={anchor.step} ready={anchor.reached_ready} "
        f"old_seq={anchor.old_seq_acc:.3f} frontier={'+'.join(anchor.old_frontier)}"
    )
    print(
        f"Adapter teacher: old_auc={teacher.old_seq_auc:.3f} final_old={teacher.final_old_seq:.3f} "
        f"text_loss={teacher.final_text_loss:.3f} replay={teacher.replay_count}/{teacher.replay_budget}"
    )
    print(f"Anchor base-only text_loss before consolidation: {baseline_base_text_loss:.3f}")
    print(
        f"{'branch':28s} {'base_old':>8s} {'base_text':>9s} {'old_auc':>8s} "
        f"{'text_auc':>8s} {'old_batch':>9s} {'z_visc':>7s} {'base_only':>9s}"
    )
    for name in CONSOLIDATION_BRANCHES:
        result = results[name]
        print(
            f"{name:28s} {result.final_old_seq:8.3f} {result.final_text_loss:9.3f} "
            f"{result.old_seq_auc:8.3f} {result.text_loss_auc:8.3f} "
            f"{result.replay_count:3d}/{result.replay_budget:<5d} {result.z_viscosity_steps:7d} "
            f"{'yes' if result.base_only_verified else 'no':>9s}"
        )
    best = max((results[name] for name in CONSOLIDATION_BRANCHES if name != "no_consolidation"), key=lambda r: r.final_old_seq)
    best_text = min((results[name] for name in CONSOLIDATION_BRANCHES if name != "no_consolidation"), key=lambda r: r.final_text_loss)
    print(f"Best base-old branch: {best.name} old_seq={best.final_old_seq:.3f} text_loss={best.final_text_loss:.3f}")
    print(f"Best base-text branch: {best_text.name} old_seq={best_text.final_old_seq:.3f} text_loss={best_text.final_text_loss:.3f}")
    print("=" * 78)


def result_to_rows(
    seed: int,
    anchor: ww.AnchorInfo,
    teacher: TeacherResult,
    baseline_base_text_loss: float,
    results: Dict[str, ConsolidationResult],
) -> List[Dict[str, object]]:
    rows = []
    for name in CONSOLIDATION_BRANCHES:
        result = results[name]
        rows.append(
            {
                "seed": seed,
                "branch": name,
                "old_ready": int(anchor.reached_ready),
                "anchor_step": anchor.step,
                "anchor_old_seq": anchor.old_seq_acc,
                "anchor_old_close": anchor.old_close_acc,
                "old_frontier": "+".join(anchor.old_frontier),
                "teacher_old_seq_auc": teacher.old_seq_auc,
                "teacher_final_old_seq": teacher.final_old_seq,
                "teacher_final_text_loss": teacher.final_text_loss,
                "anchor_base_text_loss": baseline_base_text_loss,
                "base_final_old_seq": result.final_old_seq,
                "base_final_old_close": result.final_old_close,
                "base_final_text_loss": result.final_text_loss,
                "base_final_text_acc": result.final_text_acc,
                "base_old_seq_auc": result.old_seq_auc,
                "base_text_loss_auc": result.text_loss_auc,
                "old_batch_count": result.replay_count,
                "old_batch_budget": result.replay_budget,
                "replay_count": result.replay_count,
                "replay_budget": result.replay_budget,
                "z_viscosity_steps": result.z_viscosity_steps,
                "base_train_steps": result.base_train_steps,
                "base_only_verified": int(result.base_only_verified),
            }
        )
    return rows


def write_csv(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
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
    print(f"BEGIN LATERAL CONSOLIDATION SEED {seed}")
    print("#" * 78)
    ww.set_seed(seed)

    text_train_positions, text_eval_positions = ww.make_text_positions(len(train_data), len(val_data), seed)
    bracket_eval_batches = ww.make_fixed_bracket_batches(
        stoi, seed + 30_000, ww.BRACKET_EVAL_BATCHES, ww.BRACKET_EVAL_BATCH
    )
    bracket_probe_batch = ww.make_fixed_bracket_batches(stoi, seed + 40_000, 1, ww.PROBE_BATCH)[0]

    anchor = ww.train_old_skill(vocab_size, stoi, seed, bracket_eval_batches, bracket_probe_batch)
    base_model, base_opt = ww.restore_model_from_checkpoint(vocab_size, anchor.checkpoint, load_optimizer=False)
    baseline_base_text_loss = ww.evaluate_text(base_model, val_data, text_eval_positions)["loss"]
    del base_model, base_opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher = train_latent_adapter_teacher(
        anchor,
        vocab_size,
        stoi,
        train_data,
        val_data,
        text_train_positions,
        text_eval_positions,
        bracket_eval_batches,
        bracket_probe_batch,
        seed,
    )

    results: Dict[str, ConsolidationResult] = {
        "no_consolidation": no_consolidation_result(
            teacher, anchor, vocab_size, val_data, text_eval_positions, bracket_eval_batches
        )
    }
    for branch_name in CONSOLIDATION_BRANCHES:
        if branch_name == "no_consolidation":
            continue
        results[branch_name] = consolidate_student(
            branch_name,
            teacher,
            anchor,
            vocab_size,
            stoi,
            train_data,
            val_data,
            text_train_positions,
            text_eval_positions,
            bracket_eval_batches,
            bracket_probe_batch,
            seed,
        )

    if SMOKE:
        assert set(results) == set(CONSOLIDATION_BRANCHES), "not all consolidation branches ran"
        assert results["no_consolidation"].base_train_steps == 0, "no-consolidation branch trained"
        assert results["dual_lateral_balanced"].replay_count > 0, "balanced branch used no old batches"
        assert results["dual_lateral_strong"].replay_count > results["dual_lateral_balanced"].replay_count, (
            "strong branch should bias toward more old batches"
        )
        assert results["dual_lateral_freeze_frontier"].replay_count > 0, "freeze-frontier branch used no old batches"
        assert results["dual_lateral_z_viscous"].z_viscosity_steps > 0, "Z-viscous branch did not apply viscosity"
        assert all(result.base_only_verified for result in results.values()), "adapter-off base-only check failed"
        assert teacher.latent_projection_steps > 0, "adapter teacher did not use latent projection"

    print_seed_summary(seed, anchor, teacher, baseline_base_text_loss, results)
    return result_to_rows(seed, anchor, teacher, baseline_base_text_loss, results)


def summarize_all(rows: List[Dict[str, object]]) -> None:
    by_branch: Dict[str, List[Dict[str, object]]] = {name: [] for name in CONSOLIDATION_BRANCHES}
    for row in rows:
        by_branch[str(row["branch"])].append(row)
    seeds = sorted({int(row["seed"]) for row in rows})

    print("\n" + "=" * 78)
    print("LATERAL CONSOLIDATION SUMMARY ACROSS SEEDS")
    print("=" * 78)
    print(f"Seeds run: {len(seeds)}")
    ready = [row for row in by_branch["no_consolidation"] if int(row["old_ready"]) == 1]
    print(f"Old bracket seq>=0.90 reached before attachment: {len(ready)}/{len(seeds)}")

    print("\nMean base-only final old_seq by branch:")
    for name in CONSOLIDATION_BRANCHES:
        mean, std = mean_std([float(row["base_final_old_seq"]) for row in by_branch[name]])
        print(f"  {name:28s} {format_mean_std(mean, std)}")

    print("\nMean base-only final text_loss by branch:")
    for name in CONSOLIDATION_BRANCHES:
        mean, std = mean_std([float(row["base_final_text_loss"]) for row in by_branch[name]])
        print(f"  {name:28s} {format_mean_std(mean, std)}")

    def values(branch: str, key: str) -> List[float]:
        return [float(row[key]) for row in by_branch[branch]]

    no_old = values("no_consolidation", "base_final_old_seq")
    no_text = values("no_consolidation", "base_final_text_loss")
    balanced_old = values("dual_lateral_balanced", "base_final_old_seq")
    balanced_text = values("dual_lateral_balanced", "base_final_text_loss")
    freeze_old = values("dual_lateral_freeze_frontier", "base_final_old_seq")
    freeze_text = values("dual_lateral_freeze_frontier", "base_final_text_loss")
    z_old = values("dual_lateral_z_viscous", "base_final_old_seq")
    z_text = values("dual_lateral_z_viscous", "base_final_text_loss")
    strong_old = values("dual_lateral_strong", "base_final_old_seq")
    strong_text = values("dual_lateral_strong", "base_final_text_loss")
    teacher_old = values("no_consolidation", "teacher_final_old_seq")
    teacher_text = values("no_consolidation", "teacher_final_text_loss")

    balanced_old_gain = [b - n for b, n in zip(balanced_old, no_old)]
    freeze_old_gain = [f - n for f, n in zip(freeze_old, no_old)]
    z_old_gain = [z - n for z, n in zip(z_old, no_old)]
    strong_old_gain = [s - n for s, n in zip(strong_old, no_old)]
    balanced_text_gain = [n - b for b, n in zip(balanced_text, no_text)]
    freeze_text_gain = [n - f for f, n in zip(freeze_text, no_text)]
    z_text_gain = [n - z for z, n in zip(z_text, no_text)]
    strong_text_gain = [n - s for s, n in zip(strong_text, no_text)]
    z_teacher_old_gap = [z - t for z, t in zip(z_old, teacher_old)]
    z_teacher_text_gap = [z - t for z, t in zip(z_text, teacher_text)]

    print("\nPrimary consolidation effects:")
    print(f"  Dual balanced old_seq gain over no-consolidation: {format_mean_std(*mean_std(balanced_old_gain))}")
    print(f"  Dual freeze-frontier old_seq gain over no-consolidation: {format_mean_std(*mean_std(freeze_old_gain))}")
    print(f"  Dual Z-viscous old_seq gain over no-consolidation: {format_mean_std(*mean_std(z_old_gain))}")
    print(f"  Dual strong old_seq gain over no-consolidation: {format_mean_std(*mean_std(strong_old_gain))}")
    print(f"  Dual balanced text_loss improvement over no-consolidation: {format_mean_std(*mean_std(balanced_text_gain))}")
    print(f"  Dual freeze-frontier text_loss improvement over no-consolidation: {format_mean_std(*mean_std(freeze_text_gain))}")
    print(f"  Dual Z-viscous text_loss improvement over no-consolidation: {format_mean_std(*mean_std(z_text_gain))}")
    print(f"  Dual strong text_loss improvement over no-consolidation: {format_mean_std(*mean_std(strong_text_gain))}")
    print(f"  Z-viscous bootstrap 95% CI old_seq gain: {format_ci(*bootstrap_ci(z_old_gain))}")
    print(f"  Z-viscous base old_seq minus adapter-teacher old_seq: {format_mean_std(*mean_std(z_teacher_old_gap))}")
    print(f"  Z-viscous base text_loss minus adapter-teacher text_loss: {format_mean_std(*mean_std(z_teacher_text_gap))}")

    z_old_mean, _ = mean_std(z_old)
    z_text_improve_mean, _ = mean_std(z_text_gain)
    if len(ready) < len(seeds):
        interpretation = "mechanics-only/diagnostic: not enough seeds learned the old skill, so retention is not scored."
    elif z_old_mean >= 0.80 and z_text_improve_mean > 0.0:
        interpretation = "strong pass: lateral consolidation produced a base-only v2 with retained old skill and improved text."
    elif z_text_improve_mean > 0.0:
        interpretation = "mixed: consolidation improved base-only text but old-skill retention still needs tighter protection."
    else:
        interpretation = "diagnostic/fail: lateral consolidation did not yet improve base-only text without losing old skill."
    print(f"\nInterpretation: {interpretation}")
    print(f"CSV saved to: {CSV_PATH}")
    print("=" * 78)


def main() -> None:
    ww.set_seed(ww.SEEDS[0])
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 78)
    print("WATER WEIGHTS: LATERAL CONSOLIDATION BENCHMARK")
    print("=" * 78)
    print("Question: can dual-teacher lateral consolidation unify adapter skills into a base-only v2?")
    print(f"Device: {ww.DEVICE}")
    print(f"Seeds: {ww.SEEDS}")
    print(f"Phase A: bracket until seq>={ww.OLD_READY_SEQ:.2f} or {ww.PHASE_A_MAX_STEPS} steps")
    print(f"Phase B: solid-base latent adapter attach for {ww.PHASE_B_STEPS} steps")
    print(f"Phase C: base-only dual-teacher lateral consolidation for {CONSOLIDATION_STEPS} steps")
    print(
        f"Model: d={ww.D_MODEL}, layers={ww.N_LAYER}, heads={ww.N_HEAD}, "
        f"block={ww.BLOCK_SIZE}, adapter_rank={ww.ADAPTER_RANK}"
    )
    print(f"Consolidation branches: {CONSOLIDATION_BRANCHES}")
    print(
        "Primary test: old batches imitate the frozen old base, text batches imitate the adapter teacher; "
        "after consolidation, adapters are disabled and only the base is evaluated."
    )

    text = ww.download_or_load_text()
    stoi, _itos = ww.build_vocab(text)
    encoded = ww.encode(text, stoi)
    split = int(len(encoded) * ww.TRAIN_FRACTION)
    train_data = encoded[:split]
    val_data = encoded[split:]
    print(f"Vocab size: {len(stoi)}")
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    all_rows: List[Dict[str, object]] = []
    start = time.time()
    for seed in ww.SEEDS:
        all_rows.extend(run_seed(seed, len(stoi), stoi, train_data, val_data))
        write_csv(all_rows)
    summarize_all(all_rows)
    write_csv(all_rows)
    print(f"Total wall time: {format_seconds(time.time() - start)}")


if __name__ == "__main__":
    main()
