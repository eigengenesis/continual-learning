#!/usr/bin/env python3
"""
Small calibrated toy benchmark for the claim that replay is not the main thing
carrying retention in this continual-learning pipeline.

This benchmark intentionally uses easy sequence transforms so the old skill can
actually reach mastery before we ask retention questions.

Pipeline:
    text pretrain -> old base skill A -> adapter teacher for new skill B
    -> base-only consolidation

Variants:
    1. full_pipeline        : teacher replay on, consolidation old batches on
    2. no_teacher_replay    : teacher replay off, consolidation old batches on
    3. no_consolidation_old : teacher replay on, consolidation old batches off
    4. no_replay_anywhere   : teacher replay off, consolidation old batches off

Default tasks are from the same order family:
    A = reverse
    B = rotate_left

Run:
    python colab_replay_independence_toy.py

Fast:
    python colab_replay_independence_toy.py --fast
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import colab_layer_expansion_lateral_lab as expand
import colab_layer_expansion_lateral_propagation_lab as prop
import colab_skill_affinity_toy as toy
import colab_water_weights_benchmark as ww


ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "replay_independence_toy_results.json"
CSV_PATH = ROOT / "replay_independence_toy_results.csv"


@dataclass
class AnchorResult:
    checkpoint: Dict[str, torch.Tensor]
    projectors: Dict[str, torch.Tensor]
    grad_basis: Dict[str, torch.Tensor]
    grad_importance: Dict[str, torch.Tensor]
    old_answer_acc: float
    old_seq_acc: float
    text_loss: float
    text_acc: float
    steps_used: int
    reached_ready: bool


@dataclass
class TeacherResult:
    checkpoint: Dict[str, torch.Tensor]
    replay_enabled: bool
    replay_count: int
    replay_budget: int
    old_answer_acc: float
    old_seq_acc: float
    new_answer_acc: float
    new_seq_acc: float
    text_loss: float
    text_acc: float


@dataclass
class ConsolidationResult:
    variant: str
    model_family: str
    teacher_replay_enabled: bool
    consolidation_old_enabled: bool
    hard_old_grad_proj: float
    teacher_replay_count: int
    teacher_replay_budget: int
    consolidation_old_count: int
    consolidation_old_budget: int
    old_answer_acc: float
    old_seq_acc: float
    new_answer_acc: float
    new_seq_acc: float
    text_loss: float
    text_acc: float
    balanced_mean: float
    balanced_geom: float


@dataclass
class ModelCheckpointResult:
    checkpoint: Dict[str, object]
    model_family: str
    old_answer_acc: float
    old_seq_acc: float
    new_answer_acc: float
    new_seq_acc: float
    text_loss: float
    text_acc: float


def parse_args() -> argparse.Namespace:
    smoke = False
    parser = argparse.ArgumentParser(description="Toy replay-independence ablation")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--old-task", type=str, default="reverse")
    parser.add_argument("--new-task", type=str, default="rotate_left")
    parser.add_argument("--old-tag", type=str, default="o")
    parser.add_argument("--new-tag", type=str, default="n")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--text-corpus-chars", type=int, default=16_000 if not smoke else 6_000)
    parser.add_argument("--base-steps", type=int, default=60 if torch.cuda.is_available() else 30)
    parser.add_argument("--anchor-steps", type=int, default=280 if torch.cuda.is_available() else 160)
    parser.add_argument("--task-steps", type=int, default=180 if torch.cuda.is_available() else 120)
    parser.add_argument("--consolidation-steps", type=int, default=180 if torch.cuda.is_available() else 100)
    parser.add_argument("--batch-size", type=int, default=min(ww.BATCH_SIZE, 12 if torch.cuda.is_available() else 8))
    parser.add_argument("--source-len-train", type=int, default=4)
    parser.add_argument("--source-len-eval", type=int, default=5)
    parser.add_argument("--text-eval-batches", type=int, default=4)
    parser.add_argument("--task-eval-batches", type=int, default=6)
    parser.add_argument("--projector-batches", type=int, default=4)
    parser.add_argument("--grad-basis-batches", type=int, default=8)
    parser.add_argument("--anchor-eval-period", type=int, default=20)
    parser.add_argument("--teacher-eval-period", type=int, default=24)
    parser.add_argument("--consolidation-eval-period", type=int, default=24)
    parser.add_argument("--base-lr", type=float, default=ww.BASE_LR)
    parser.add_argument("--anchor-lr", type=float, default=3e-4)
    parser.add_argument("--adapter-lr", type=float, default=2e-3)
    parser.add_argument("--consolidation-lr", type=float, default=1.2e-4)
    parser.add_argument("--latent-strength", type=float, default=0.75)
    parser.add_argument("--teacher-replay-period", type=int, default=4)
    parser.add_argument("--teacher-replay-fraction", type=float, default=0.12)
    parser.add_argument("--consolidation-old-period", type=int, default=2)
    parser.add_argument("--kl-weight", type=float, default=0.75)
    parser.add_argument("--old-kl-weight", type=float, default=0.75)
    parser.add_argument("--old-answer-threshold", type=float, default=0.98)
    parser.add_argument("--old-seq-threshold", type=float, default=0.90)
    parser.add_argument("--hard-old-grad-proj", type=float, default=0.0)
    parser.add_argument("--include-expansion", action="store_true")
    parser.add_argument("--include-prop-expansion", action="store_true")
    parser.add_argument("--include-ln-heal", action="store_true")
    parser.add_argument("--include-joint-distill", action="store_true")
    parser.add_argument("--include-prop-post-consolidation", action="store_true")
    parser.add_argument("--include-prop-post-ln-heal", action="store_true")
    parser.add_argument("--include-amoeba", action="store_true",
                        help="Run water-weights (amoeba) sweep: compress low-importance old dims")
    parser.add_argument("--include-microinject", action="store_true",
                        help="Run microinject sweep: core/fringe/null gradient routing")
    parser.add_argument("--include-amoeba-only", action="store_true",
                        help="True zero-replay: new-task only + amoeba projection, no teachers")
    parser.add_argument("--include-activation-anchor", action="store_true",
                        help="True zero-replay: new-task + amoeba + hidden state anchor regularization")
    parser.add_argument("--activation-anchor-probes", type=int, default=16,
                        help="Number of probe batches to save hidden states from")
    parser.add_argument("--activation-anchor-weight", type=float, default=0.5,
                        help="Weight of hidden state anchor loss relative to task loss")
    parser.add_argument("--include-lateral-merge", action="store_true",
                        help="True zero-replay: Phase 1 train adapter, Phase 2 distill into dense base with Amoeba")
    parser.add_argument("--lateral-proxy-weight", type=float, default=1.0,
                        help="Weight of generic text proxy distillation during lateral merge")
    parser.add_argument("--include-nonlinear-amoeba", action="store_true",
                        help="True zero-replay with activation-space Amoeba: prevents noise injection via input-space gradient projection")
    parser.add_argument("--activation-rank", type=int, default=None,
                        help="Rank of activation basis for nonlinear amoeba (default: d_model // 2)")
    parser.add_argument("--expansion-steps", type=int, default=180 if torch.cuda.is_available() else 120)
    parser.add_argument("--expansion-lr", type=float, default=ww.BASE_LR * 2.0)
    parser.add_argument("--expansion-gate-floor-logit", type=float, default=-4.2)
    parser.add_argument("--ln-heal-steps", type=int, default=30)
    parser.add_argument("--ln-heal-lr", type=float, default=8e-4)
    parser.add_argument("--json-path", type=Path, default=JSON_PATH)
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    return parser.parse_args()


def normalize_output_paths(cfg: argparse.Namespace) -> None:
    """Allow --json-path/--csv-path to be either files or output directories."""
    if cfg.json_path.exists() and cfg.json_path.is_dir():
        cfg.json_path = cfg.json_path / JSON_PATH.name
    elif cfg.json_path.suffix.lower() != ".json":
        cfg.json_path = cfg.json_path / JSON_PATH.name

    if cfg.csv_path.exists() and cfg.csv_path.is_dir():
        cfg.csv_path = cfg.csv_path / CSV_PATH.name
    elif cfg.csv_path.suffix.lower() != ".csv":
        cfg.csv_path = cfg.csv_path / CSV_PATH.name


@contextmanager
def fast_mode(cfg: argparse.Namespace) -> Iterator[None]:
    if not cfg.fast:
        yield
        return
    saved = {
        "base_steps": cfg.base_steps,
        "anchor_steps": cfg.anchor_steps,
        "task_steps": cfg.task_steps,
        "consolidation_steps": cfg.consolidation_steps,
        "expansion_steps": cfg.expansion_steps,
        "task_eval_batches": cfg.task_eval_batches,
        "text_eval_batches": cfg.text_eval_batches,
        "projector_batches": cfg.projector_batches,
        "grad_basis_batches": cfg.grad_basis_batches,
        "anchor_eval_period": cfg.anchor_eval_period,
        "teacher_eval_period": cfg.teacher_eval_period,
        "consolidation_eval_period": cfg.consolidation_eval_period,
    }
    try:
        cfg.base_steps = min(cfg.base_steps, 36)
        cfg.anchor_steps = min(cfg.anchor_steps, 160)
        cfg.task_steps = min(cfg.task_steps, 100)
        cfg.consolidation_steps = min(cfg.consolidation_steps, 80)
        cfg.expansion_steps = min(cfg.expansion_steps, 100)
        cfg.task_eval_batches = min(cfg.task_eval_batches, 4)
        cfg.text_eval_batches = min(cfg.text_eval_batches, 3)
        cfg.projector_batches = min(cfg.projector_batches, 3)
        cfg.grad_basis_batches = min(cfg.grad_basis_batches, 4)
        cfg.anchor_eval_period = min(cfg.anchor_eval_period, 16)
        cfg.teacher_eval_period = min(cfg.teacher_eval_period, 16)
        cfg.consolidation_eval_period = min(cfg.consolidation_eval_period, 16)
        yield
    finally:
        for key, value in saved.items():
            setattr(cfg, key, value)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def task_batches(
    stoi: Dict[str, int],
    task_name: str,
    task_tag: str,
    seed: int,
    *,
    num_batches: int,
    batch_size: int,
    source_len: int,
) -> List[ww.Batch]:
    return [
        make_task_batch(
            stoi,
            task_name,
            task_tag,
            seed,
            index,
            batch_size=batch_size,
            source_len=source_len,
        )
        for index in range(num_batches)
    ]


def build_task_stream(
    stoi: Dict[str, int],
    task_name: str,
    task_tag: str,
    rng: np.random.Generator,
    source_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(task_tag) != 1:
        raise ValueError(f"task_tag must be a single character, got {task_tag!r}")
    if task_tag not in stoi:
        raise ValueError(f"task_tag {task_tag!r} missing from vocabulary")
    token_ids = rng.integers(0, len(toy.TASK_ALPHABET), size=source_len)
    source = "".join(toy.TASK_ALPHABET[int(idx)] for idx in token_ids)
    target = toy.task_transform(task_name, source)
    prompt = f"{task_tag}{source}|"
    answer = f"{target}\n"
    episode = list(prompt + answer)
    flags = [False] * len(prompt) + [True] * len(target) + [False]

    total_len = ww.BLOCK_SIZE + 1
    pad_len = max(total_len - len(episode), 0)
    left_pad = int(rng.integers(0, pad_len + 1)) if pad_len > 0 else 0
    right_pad = pad_len - left_pad
    chars = ([" "] * left_pad) + episode + ([" "] * right_pad)
    critical = ([False] * left_pad) + flags + ([False] * right_pad)
    ids = torch.tensor([stoi[ch] for ch in chars[:total_len]], dtype=torch.long)
    mask = torch.tensor(critical[:total_len], dtype=torch.bool)
    return ids[:-1], ids[1:], mask[1:]


def make_task_batch(
    stoi: Dict[str, int],
    task_name: str,
    task_tag: str,
    seed: int,
    index: int,
    *,
    batch_size: int,
    source_len: int,
) -> ww.Batch:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for offset in range(batch_size):
        rng = np.random.default_rng(seed * 1_000_003 + index * 10_007 + offset)
        x, y, mask = build_task_stream(stoi, task_name, task_tag, rng, source_len)
        xs.append(x)
        ys.append(y)
        masks.append(mask)
    return ww.Batch(
        torch.stack(xs).to(ww.DEVICE),
        torch.stack(ys).to(ww.DEVICE),
        torch.stack(masks).to(ww.DEVICE),
    )


def eval_task(model: ww.TinyGPT, batches: Sequence[ww.Batch]) -> Dict[str, float]:
    return toy.evaluate_task(model, batches)


def eval_text(model: ww.TinyGPT, val_data: torch.Tensor, cfg: argparse.Namespace) -> Dict[str, float]:
    positions = toy.make_text_positions(
        len(val_data),
        cfg.text_eval_batches,
        min(cfg.batch_size, ww.TEXT_EVAL_BATCH),
        cfg.seed + 90_000,
    )
    return ww.evaluate_text(model, val_data, positions)


def collect_task_gradient_basis(
    model: ww.TinyGPT,
    stoi: Dict[str, int],
    task_name: str,
    task_tag: str,
    seed: int,
    cfg: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    rows: Dict[str, List[torch.Tensor]] = {block: [] for block in ww.block_keys()}
    for index in range(cfg.grad_basis_batches):
        batch = make_task_batch(
            stoi,
            task_name,
            task_tag,
            seed,
            index,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_eval,
        )
        model.zero_grad(set_to_none=True)
        _, loss = model(batch.x, batch.y)
        assert loss is not None
        loss.backward()
        for block in ww.block_keys():
            rows[block].append(ww.flatten_grads(ww.base_block_params(model, block)).detach().cpu())
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return {block: ww.make_low_rank_basis(block_rows, ww.GRAD_ANCHOR_RANK) for block, block_rows in rows.items()}


def collect_task_gradient_membrane(
    model: ww.TinyGPT,
    stoi: Dict[str, int],
    task_name: str,
    task_tag: str,
    seed: int,
    cfg: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    was_training = model.training
    model.eval()
    rows: Dict[str, List[torch.Tensor]] = {block: [] for block in ww.block_keys()}
    for index in range(cfg.grad_basis_batches):
        batch = make_task_batch(
            stoi,
            task_name,
            task_tag,
            seed,
            index,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_eval,
        )
        model.zero_grad(set_to_none=True)
        _, loss = model(batch.x, batch.y)
        assert loss is not None
        loss.backward()
        for block in ww.block_keys():
            rows[block].append(ww.flatten_grads(ww.base_block_params(model, block)).detach().cpu())
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    basis_by_block: Dict[str, torch.Tensor] = {}
    importance_by_block: Dict[str, torch.Tensor] = {}
    for block, block_rows in rows.items():
        if not block_rows:
            basis_by_block[block] = torch.empty(0, 0)
            importance_by_block[block] = torch.empty(0)
            continue
        normalized: List[torch.Tensor] = []
        for row in block_rows:
            row = row.float().cpu()
            normalized.append(row / row.norm().clamp_min(1e-12))
        matrix = torch.stack(normalized, dim=0)
        if matrix.shape[0] == 1:
            basis = matrix[:1].contiguous()
            importance = torch.ones(1, dtype=basis.dtype)
            basis_by_block[block] = basis
            importance_by_block[block] = importance
            continue
        gram = matrix @ matrix.t()
        eigvals, eigvecs = torch.linalg.eigh(gram)
        order = torch.argsort(eigvals, descending=True)
        chosen = order[: min(ww.GRAD_ANCHOR_RANK, matrix.shape[0])]
        topvals = eigvals[chosen].clamp_min(1e-12)
        maxval = topvals.max().clamp_min(1e-12)
        basis_rows: List[torch.Tensor] = []
        importance_rows: List[torch.Tensor] = []
        for idx, value in zip(chosen, topvals):
            vector = eigvecs[:, idx]
            basis = (vector @ matrix) / value.sqrt()
            basis = basis / basis.norm().clamp_min(1e-12)
            basis_rows.append(basis)
            importance_rows.append((value / maxval).reshape(1))
        basis_by_block[block] = torch.stack(basis_rows, dim=0)
        importance_by_block[block] = torch.cat(importance_rows, dim=0)
    return basis_by_block, importance_by_block


def project_gradient_away_weighted(
    flat: torch.Tensor,
    basis: torch.Tensor,
    importance: torch.Tensor,
    strength: float,
    *,
    power: float,
    floor: float,
) -> torch.Tensor:
    if strength <= 0.0 or basis.numel() == 0 or importance.numel() == 0:
        return flat
    basis = basis.to(device=flat.device, dtype=flat.dtype)
    importance = importance.to(device=flat.device, dtype=flat.dtype)
    coeff = torch.mv(basis, flat)
    weights = floor + (1.0 - floor) * importance.clamp(0.0, 1.0).pow(power)
    projected = torch.mv(basis.t(), weights * coeff)
    return flat - min(max(strength, 0.0), 1.0) * projected


def project_block_gradients_weighted(
    model: ww.TinyGPT,
    basis_by_block: Dict[str, torch.Tensor],
    importance_by_block: Dict[str, torch.Tensor],
    projection_strength: float,
    *,
    power: float,
    floor: float,
) -> None:
    if projection_strength <= 0.0:
        return
    for block, basis in basis_by_block.items():
        importance = importance_by_block.get(block)
        if importance is None:
            continue
        params = ww.base_block_params(model, block)
        flat = ww.flatten_grads(params)
        safe = project_gradient_away_weighted(flat, basis, importance, projection_strength, power=power, floor=floor)
        ww.assign_flat_grads(params, safe)


def microinject_gradient(
    flat: torch.Tensor,
    basis: torch.Tensor,
    importance: torch.Tensor,
    *,
    core_threshold: float,
    core_scale: float,
    fringe_scale: float,
    null_boost: float,
    preserve_norm: bool,
) -> torch.Tensor:
    if basis.numel() == 0 or importance.numel() == 0:
        return null_boost * flat
    basis = basis.to(device=flat.device, dtype=flat.dtype)
    importance = importance.to(device=flat.device, dtype=flat.dtype).clamp(0.0, 1.0)
    coeff = torch.mv(basis, flat)
    recon = torch.mv(basis.t(), coeff)
    core_mask = (importance >= core_threshold).to(dtype=flat.dtype)
    fringe_mask = 1.0 - core_mask
    core = torch.mv(basis.t(), core_mask * coeff)
    fringe = torch.mv(basis.t(), fringe_mask * coeff)
    null = flat - recon
    mixed = core_scale * core + fringe_scale * fringe + null_boost * null
    if preserve_norm:
        base_norm = flat.norm().clamp_min(1e-12)
        mixed_norm = mixed.norm().clamp_min(1e-12)
        mixed = mixed * (base_norm / mixed_norm)
    return mixed


def microinject_block_gradients(
    model: ww.TinyGPT,
    basis_by_block: Dict[str, torch.Tensor],
    importance_by_block: Dict[str, torch.Tensor],
    *,
    core_threshold: float,
    core_scale: float,
    fringe_scale: float,
    null_boost: float,
    preserve_norm: bool,
) -> None:
    for block, basis in basis_by_block.items():
        importance = importance_by_block.get(block)
        if importance is None:
            continue
        params = ww.base_block_params(model, block)
        flat = ww.flatten_grads(params)
        safe = microinject_gradient(
            flat,
            basis,
            importance,
            core_threshold=core_threshold,
            core_scale=core_scale,
            fringe_scale=fringe_scale,
            null_boost=null_boost,
            preserve_norm=preserve_norm,
        )
        ww.assign_flat_grads(params, safe)


def kl_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature * temperature)


def restore_model(vocab_size: int, state: Dict[str, torch.Tensor]) -> ww.TinyGPT:
    model = ww.TinyGPT(vocab_size).to(ww.DEVICE)
    model.load_state_dict(state)
    return model


def make_result(
    *,
    variant: str,
    model_family: str,
    teacher_replay_enabled: bool,
    consolidation_old_enabled: bool,
    hard_old_grad_proj: float,
    teacher_replay_count: int,
    teacher_replay_budget: int,
    consolidation_old_count: int,
    consolidation_old_budget: int,
    old_metrics: Dict[str, float],
    new_metrics: Dict[str, float],
    text_metrics: Dict[str, float],
) -> ConsolidationResult:
    old_answer = float(old_metrics["answer_acc"])
    new_answer = float(new_metrics["answer_acc"])
    return ConsolidationResult(
        variant=variant,
        model_family=model_family,
        teacher_replay_enabled=teacher_replay_enabled,
        consolidation_old_enabled=consolidation_old_enabled,
        hard_old_grad_proj=hard_old_grad_proj,
        teacher_replay_count=teacher_replay_count,
        teacher_replay_budget=teacher_replay_budget,
        consolidation_old_count=consolidation_old_count,
        consolidation_old_budget=consolidation_old_budget,
        old_answer_acc=old_answer,
        old_seq_acc=float(old_metrics["seq_acc"]),
        new_answer_acc=new_answer,
        new_seq_acc=float(new_metrics["seq_acc"]),
        text_loss=float(text_metrics["loss"]),
        text_acc=float(text_metrics["acc"]),
        balanced_mean=0.5 * (old_answer + new_answer),
        balanced_geom=math.sqrt(max(old_answer, 0.0) * max(new_answer, 0.0)),
    )


def _row_summary(row: ConsolidationResult) -> Dict[str, object]:
    return {
        "variant": row.variant,
        "family": row.model_family,
        "teacher_replay": f"{row.teacher_replay_count}/{row.teacher_replay_budget}",
        "old_mix": f"{row.consolidation_old_count}/{row.consolidation_old_budget}",
        "old_answer": row.old_answer_acc,
        "new_answer": row.new_answer_acc,
        "balanced_mean": row.balanced_mean,
        "balanced_geom": row.balanced_geom,
        "text_loss": row.text_loss,
    }


def _find_row(rows: Sequence[ConsolidationResult], variant: str) -> ConsolidationResult | None:
    return next((row for row in rows if row.variant == variant), None)


def _best_row(rows: Sequence[ConsolidationResult]) -> ConsolidationResult | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (row.balanced_geom, row.balanced_mean))


def build_evidence_tables(
    base_text_metrics: Dict[str, float],
    anchor: AnchorResult,
    rows: Sequence[ConsolidationResult],
    cfg: argparse.Namespace,
) -> Dict[str, object]:
    replay_variants = [
        "full_pipeline",
        "no_teacher_replay",
        "no_consolidation_old",
        "no_replay_anywhere",
    ]
    replay_rows = [row for name in replay_variants if (row := _find_row(rows, name)) is not None]
    nonlinear_rows = [
        row
        for row in rows
        if row.model_family == "base_nonlinear_amoeba" and row.consolidation_old_count == 0
    ]
    best_nonlinear = _best_row(nonlinear_rows)
    if best_nonlinear is not None:
        replay_rows.append(best_nonlinear)

    synergy_variants = [
        "nl_proj_only",
        "nl_dual10",
        "nl_thmatch10",
        "nl_th10_d10",
        "nl_th10_d10_no_proj",
        "nl_th10_d10_no_text",
    ]
    synergy_rows = [row for name in synergy_variants if (row := _find_row(rows, name)) is not None]

    full = _find_row(rows, "full_pipeline")
    no_teacher = _find_row(rows, "no_teacher_replay")
    no_consolidation_old = _find_row(rows, "no_consolidation_old")
    no_replay = _find_row(rows, "no_replay_anywhere")
    combo = _find_row(rows, "nl_th10_d10")
    components = [row for row in (_find_row(rows, "nl_dual10"), _find_row(rows, "nl_thmatch10")) if row is not None]
    best_component = _best_row(components)

    teacher_replay_irrelevant = False
    if full is not None and no_teacher is not None:
        teacher_replay_irrelevant = (
            abs(no_teacher.old_answer_acc - full.old_answer_acc) <= 0.03
            and abs(no_teacher.new_answer_acc - full.new_answer_acc) <= 0.03
            and abs(no_teacher.balanced_geom - full.balanced_geom) <= 0.03
        )

    old_signal_location_matters = False
    if no_consolidation_old is not None and no_replay is not None:
        old_signal_location_matters = (
            no_consolidation_old.old_answer_acc <= 0.20
            and no_replay.old_answer_acc <= 0.20
            and max(no_consolidation_old.new_answer_acc, no_replay.new_answer_acc) >= 0.80
        )

    hidden_match_synergy = False
    if combo is not None and best_component is not None:
        hidden_match_synergy = combo.balanced_geom >= best_component.balanced_geom + 0.02

    proxy_high = best_nonlinear is not None and best_nonlinear.balanced_geom >= 0.82

    return {
        "config": {
            "old_task": cfg.old_task,
            "new_task": cfg.new_task,
            "source_len_train": cfg.source_len_train,
            "source_len_eval": cfg.source_len_eval,
            "device": ww.DEVICE,
        },
        "base_text_metrics": base_text_metrics,
        "anchor": {
            "old_answer_acc": anchor.old_answer_acc,
            "old_seq_acc": anchor.old_seq_acc,
            "reached_ready": anchor.reached_ready,
        },
        "toy_replay_table": [_row_summary(row) for row in replay_rows],
        "toy_synergy_table": [_row_summary(row) for row in synergy_rows],
        "acceptance": {
            "teacher_replay_irrelevant": teacher_replay_irrelevant,
            "old_signal_location_matters": old_signal_location_matters,
            "proxy_replay_free_geom_ge_0_82": proxy_high,
            "hidden_match_dual_synergy": hidden_match_synergy,
        },
        "best_nonlinear_amoeba": None if best_nonlinear is None else _row_summary(best_nonlinear),
        "best_hidden_match_component": None if best_component is None else _row_summary(best_component),
        "hidden_match_dual": None if combo is None else _row_summary(combo),
        "callout": (
            "Generic text is needed in this toy because the symbolic task barely exercises the text "
            "model's base circuits. For LLMs, fine-tuning data is already text, so the proxy role is "
            "more naturally available, though broad generic anchors may still help."
        ),
    }


def _print_evidence_rows(title: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    print(title)
    print("-" * 78)
    print(f"{'variant':<24} {'old_mix':>8} {'old':>8} {'new':>8} {'bal':>8} {'geom':>8} {'text_loss':>10}")
    for row in rows:
        print(
            f"{str(row['variant']):<24} {str(row['old_mix']):>8} "
            f"{float(row['old_answer']):>8.3f} {float(row['new_answer']):>8.3f} "
            f"{float(row['balanced_mean']):>8.3f} {float(row['balanced_geom']):>8.3f} "
            f"{float(row['text_loss']):>10.3f}"
        )
    print("-" * 78)


def prepare_base(cfg: argparse.Namespace) -> Tuple[int, Dict[str, int], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
    toy.set_seed(cfg.seed)
    joint_text = ww.EMBEDDED_FALLBACK_TEXT[: cfg.text_corpus_chars] + cfg.old_tag + cfg.new_tag
    stoi, _ = toy.build_joint_vocab(joint_text)
    train_data, val_data = toy.split_text_corpus(stoi, cfg.text_corpus_chars)
    vocab_size = len(stoi)
    base_state, base_text_metrics, _ = toy.pretrain_base_model(vocab_size, train_data, val_data, cfg)
    return vocab_size, stoi, train_data, val_data, base_state, base_text_metrics


def train_old_anchor(
    vocab_size: int,
    stoi: Dict[str, int],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    base_state: Dict[str, torch.Tensor],
    cfg: argparse.Namespace,
) -> AnchorResult:
    model = restore_model(vocab_size, base_state)
    model.set_adapters_enabled(False)
    model.clear_latent_free_projectors()
    ww.set_requires_grad(ww.all_base_params(model), True)
    ww.set_requires_grad(ww.all_adapter_params(model), False)
    optimizer = torch.optim.AdamW(
        ww.all_base_params(model),
        lr=cfg.anchor_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )

    eval_batches = task_batches(
        stoi,
        cfg.old_task,
        cfg.old_tag,
        cfg.seed + 10_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    reached_ready = False
    steps_used = 0
    metrics = eval_task(model, eval_batches)
    for step in range(1, cfg.anchor_steps + 1):
        batch = make_task_batch(
            stoi,
            cfg.old_task,
            cfg.old_tag,
            cfg.seed + 20_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        toy.train_step(model, optimizer, batch)
        steps_used = step
        if step % cfg.anchor_eval_period == 0 or step == cfg.anchor_steps:
            metrics = eval_task(model, eval_batches)
            if (
                float(metrics["answer_acc"]) >= float(cfg.old_answer_threshold)
                and float(metrics["seq_acc"]) >= float(cfg.old_seq_threshold)
            ):
                reached_ready = True
                break

    positions = toy.make_text_positions(
        len(train_data),
        cfg.projector_batches,
        cfg.batch_size,
        cfg.seed + 30_000,
    )
    projectors = toy.collect_text_latent_free_projectors(model, train_data, positions, ww.block_keys())
    grad_basis, grad_importance = collect_task_gradient_membrane(model, stoi, cfg.old_task, cfg.old_tag, cfg.seed + 35_000, cfg)
    text_metrics = eval_text(model, val_data, cfg)
    checkpoint = ww.tensor_tree_to_cpu(model.state_dict())
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return AnchorResult(
        checkpoint=checkpoint,
        projectors=projectors,
        grad_basis=grad_basis,
        grad_importance=grad_importance,
        old_answer_acc=float(metrics["answer_acc"]),
        old_seq_acc=float(metrics["seq_acc"]),
        text_loss=float(text_metrics["loss"]),
        text_acc=float(text_metrics["acc"]),
        steps_used=steps_used,
        reached_ready=reached_ready,
    )


def train_teacher_b(
    vocab_size: int,
    stoi: Dict[str, int],
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    cfg: argparse.Namespace,
    *,
    replay_enabled: bool,
) -> TeacherResult:
    model = restore_model(vocab_size, old_anchor.checkpoint)
    model.set_adapters_enabled(True)
    model.set_latent_free_projectors(old_anchor.projectors, cfg.latent_strength)
    ww.set_requires_grad(ww.all_base_params(model), False)
    ww.set_requires_grad(ww.all_adapter_params(model), True)
    optimizer = torch.optim.AdamW(
        [param for param in ww.all_adapter_params(model) if param.requires_grad],
        lr=cfg.adapter_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )

    old_eval_batches = task_batches(
        stoi,
        cfg.old_task,
        cfg.old_tag,
        cfg.seed + 40_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    new_eval_batches = task_batches(
        stoi,
        cfg.new_task,
        cfg.new_tag,
        cfg.seed + 50_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )

    replay_budget = int(cfg.task_steps * cfg.teacher_replay_fraction) if replay_enabled else 0
    replay_count = 0
    for step in range(1, cfg.task_steps + 1):
        replay_this_step = (
            replay_enabled
            and cfg.teacher_replay_period > 0
            and step % cfg.teacher_replay_period == 0
            and replay_count < replay_budget
        )
        if replay_this_step:
            batch = make_task_batch(
                stoi,
                cfg.old_task,
                cfg.old_tag,
                cfg.seed + 60_000,
                replay_count,
                batch_size=cfg.batch_size,
                source_len=cfg.source_len_train,
            )
            replay_count += 1
        else:
            batch = make_task_batch(
                stoi,
                cfg.new_task,
                cfg.new_tag,
                cfg.seed + 70_000,
                step,
                batch_size=cfg.batch_size,
                source_len=cfg.source_len_train,
            )
        toy.train_step(model, optimizer, batch)

    old_metrics = eval_task(model, old_eval_batches)
    new_metrics = eval_task(model, new_eval_batches)
    text_metrics = eval_text(model, val_data, cfg)
    checkpoint = ww.tensor_tree_to_cpu(model.state_dict())
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return TeacherResult(
        checkpoint=checkpoint,
        replay_enabled=replay_enabled,
        replay_count=replay_count,
        replay_budget=replay_budget,
        old_answer_acc=float(old_metrics["answer_acc"]),
        old_seq_acc=float(old_metrics["seq_acc"]),
        new_answer_acc=float(new_metrics["answer_acc"]),
        new_seq_acc=float(new_metrics["seq_acc"]),
        text_loss=float(text_metrics["loss"]),
        text_acc=float(text_metrics["acc"]),
    )


def train_joint_dual_teacher_no_replay(
    vocab_size: int,
    stoi: Dict[str, int],
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    cfg: argparse.Namespace,
    *,
    variant: str = "joint_no_replay_dual_teacher",
    hard_old_grad_proj: float = 0.0,
) -> ConsolidationResult:
    teacher_new = train_teacher_b(
        vocab_size,
        stoi,
        val_data,
        old_anchor,
        cfg,
        replay_enabled=False,
    )
    teacher_old_model = restore_model(vocab_size, old_anchor.checkpoint)
    teacher_old_model.set_adapters_enabled(False)
    teacher_old_model.clear_latent_free_projectors()
    ww.set_requires_grad(teacher_old_model.parameters(), False)

    teacher_new_model = restore_model(vocab_size, teacher_new.checkpoint)
    teacher_new_model.set_adapters_enabled(True)
    teacher_new_model.set_latent_free_projectors(old_anchor.projectors, cfg.latent_strength)
    ww.set_requires_grad(teacher_new_model.parameters(), False)

    student = restore_model(vocab_size, old_anchor.checkpoint)
    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()
    ww.set_requires_grad(ww.all_adapter_params(student), False)
    ww.set_requires_grad(ww.all_base_params(student), True)
    optimizer = torch.optim.AdamW(
        ww.all_base_params(student),
        lr=cfg.consolidation_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )

    for step in range(1, cfg.consolidation_steps + 1):
        old_batch = make_task_batch(
            stoi,
            cfg.old_task,
            cfg.old_tag,
            cfg.seed + 112_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        new_batch = make_task_batch(
            stoi,
            cfg.new_task,
            cfg.new_tag,
            cfg.seed + 113_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        optimizer.zero_grad(set_to_none=True)
        student_old_logits, _ = student(old_batch.x, old_batch.y)
        student_new_logits, _ = student(new_batch.x, new_batch.y)
        with torch.no_grad():
            teacher_old_logits, _ = teacher_old_model(old_batch.x, old_batch.y)
            teacher_new_logits, _ = teacher_new_model(new_batch.x, new_batch.y)
        loss_old = toy.task_weighted_loss(student_old_logits, old_batch) + cfg.old_kl_weight * kl_divergence(student_old_logits, teacher_old_logits)
        loss_new = toy.task_weighted_loss(student_new_logits, new_batch) + cfg.kl_weight * kl_divergence(student_new_logits, teacher_new_logits)
        loss = 0.5 * (loss_old + loss_new)
        loss.backward()
        if hard_old_grad_proj > 0.0:
            ww.project_block_gradients(student, old_anchor.grad_basis, hard_old_grad_proj)
        torch.nn.utils.clip_grad_norm_(ww.all_base_params(student), ww.GRAD_CLIP)
        optimizer.step()

    old_eval_batches = task_batches(
        stoi,
        cfg.old_task,
        cfg.old_tag,
        cfg.seed + 114_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    new_eval_batches = task_batches(
        stoi,
        cfg.new_task,
        cfg.new_tag,
        cfg.seed + 115_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    old_metrics = eval_task(student, old_eval_batches)
    new_metrics = eval_task(student, new_eval_batches)
    text_metrics = eval_text(student, val_data, cfg)
    del teacher_old_model, teacher_new_model, student, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return make_result(
        variant=variant,
        model_family="base_joint",
        teacher_replay_enabled=False,
        consolidation_old_enabled=False,
        hard_old_grad_proj=hard_old_grad_proj,
        teacher_replay_count=0,
        teacher_replay_budget=0,
        consolidation_old_count=cfg.consolidation_steps,
        consolidation_old_budget=cfg.consolidation_steps,
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        text_metrics=text_metrics,
    )


def train_joint_dual_teacher_water_weights(
    vocab_size: int,
    stoi: Dict[str, int],
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    cfg: argparse.Namespace,
    *,
    variant: str,
    projection_strength: float,
    water_power: float,
    water_floor: float,
    ln_heal_steps: int = 0,
) -> ConsolidationResult:
    teacher_new = train_teacher_b(
        vocab_size,
        stoi,
        val_data,
        old_anchor,
        cfg,
        replay_enabled=False,
    )
    teacher_old_model = restore_model(vocab_size, old_anchor.checkpoint)
    teacher_old_model.set_adapters_enabled(False)
    teacher_old_model.clear_latent_free_projectors()
    ww.set_requires_grad(teacher_old_model.parameters(), False)

    teacher_new_model = restore_model(vocab_size, teacher_new.checkpoint)
    teacher_new_model.set_adapters_enabled(True)
    teacher_new_model.set_latent_free_projectors(old_anchor.projectors, cfg.latent_strength)
    ww.set_requires_grad(teacher_new_model.parameters(), False)

    student = restore_model(vocab_size, old_anchor.checkpoint)
    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()
    ww.set_requires_grad(ww.all_adapter_params(student), False)
    ww.set_requires_grad(ww.all_base_params(student), True)
    optimizer = torch.optim.AdamW(
        ww.all_base_params(student),
        lr=cfg.consolidation_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )

    for step in range(1, cfg.consolidation_steps + 1):
        old_batch = make_task_batch(
            stoi,
            cfg.old_task,
            cfg.old_tag,
            cfg.seed + 270_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        new_batch = make_task_batch(
            stoi,
            cfg.new_task,
            cfg.new_tag,
            cfg.seed + 280_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        optimizer.zero_grad(set_to_none=True)
        student_old_logits, _ = student(old_batch.x, old_batch.y)
        student_new_logits, _ = student(new_batch.x, new_batch.y)
        with torch.no_grad():
            teacher_old_logits, _ = teacher_old_model(old_batch.x, old_batch.y)
            teacher_new_logits, _ = teacher_new_model(new_batch.x, new_batch.y)
        loss_old = toy.task_weighted_loss(student_old_logits, old_batch) + cfg.old_kl_weight * kl_divergence(student_old_logits, teacher_old_logits)
        loss_new = toy.task_weighted_loss(student_new_logits, new_batch) + cfg.kl_weight * kl_divergence(student_new_logits, teacher_new_logits)
        loss = 0.5 * (loss_old + loss_new)
        loss.backward()
        project_block_gradients_weighted(
            student,
            old_anchor.grad_basis,
            old_anchor.grad_importance,
            projection_strength,
            power=water_power,
            floor=water_floor,
        )
        torch.nn.utils.clip_grad_norm_(ww.all_base_params(student), ww.GRAD_CLIP)
        optimizer.step()

    if ln_heal_steps > 0:
        student.set_adapters_enabled(False)
        student.clear_latent_free_projectors()
        ww.set_requires_grad(student.parameters(), False)
        ln_params = layernorm_params(student)
        ww.set_requires_grad(ln_params, True)
        ln_optimizer = torch.optim.AdamW(
            [param for param in ln_params if param.requires_grad],
            lr=cfg.ln_heal_lr,
            betas=ww.BETAS,
            weight_decay=0.0,
        )
        for step in range(1, ln_heal_steps + 1):
            old_batch = make_task_batch(
                stoi,
                cfg.old_task,
                cfg.old_tag,
                cfg.seed + 281_000,
                step,
                batch_size=cfg.batch_size,
                source_len=cfg.source_len_train,
            )
            new_batch = make_task_batch(
                stoi,
                cfg.new_task,
                cfg.new_tag,
                cfg.seed + 282_000,
                step,
                batch_size=cfg.batch_size,
                source_len=cfg.source_len_train,
            )
            ln_optimizer.zero_grad(set_to_none=True)
            student_old_logits, _ = student(old_batch.x, old_batch.y)
            student_new_logits, _ = student(new_batch.x, new_batch.y)
            with torch.no_grad():
                teacher_old_logits, _ = teacher_old_model(old_batch.x, old_batch.y)
                teacher_new_logits, _ = teacher_new_model(new_batch.x, new_batch.y)
            loss = kl_divergence(student_old_logits, teacher_old_logits) + kl_divergence(student_new_logits, teacher_new_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ln_params, ww.GRAD_CLIP)
            ln_optimizer.step()
        del ln_optimizer

    old_eval_batches = task_batches(
        stoi,
        cfg.old_task,
        cfg.old_tag,
        cfg.seed + 283_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    new_eval_batches = task_batches(
        stoi,
        cfg.new_task,
        cfg.new_tag,
        cfg.seed + 284_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    old_metrics = eval_task(student, old_eval_batches)
    new_metrics = eval_task(student, new_eval_batches)
    text_metrics = eval_text(student, val_data, cfg)
    del teacher_old_model, teacher_new_model, student, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return make_result(
        variant=variant,
        model_family="base_water",
        teacher_replay_enabled=False,
        consolidation_old_enabled=False,
        hard_old_grad_proj=projection_strength,
        teacher_replay_count=0,
        teacher_replay_budget=0,
        consolidation_old_count=cfg.consolidation_steps + max(ln_heal_steps, 0),
        consolidation_old_budget=cfg.consolidation_steps + max(ln_heal_steps, 0),
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        text_metrics=text_metrics,
    )


def train_joint_dual_teacher_amoeba_schedule(
    vocab_size: int,
    stoi: Dict[str, int],
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    cfg: argparse.Namespace,
    *,
    variant: str,
    phases: Sequence[Tuple[int, float, float, float, float]],
) -> ConsolidationResult:
    teacher_new = train_teacher_b(
        vocab_size,
        stoi,
        val_data,
        old_anchor,
        cfg,
        replay_enabled=False,
    )
    teacher_old_model = restore_model(vocab_size, old_anchor.checkpoint)
    teacher_old_model.set_adapters_enabled(False)
    teacher_old_model.clear_latent_free_projectors()
    ww.set_requires_grad(teacher_old_model.parameters(), False)

    teacher_new_model = restore_model(vocab_size, teacher_new.checkpoint)
    teacher_new_model.set_adapters_enabled(True)
    teacher_new_model.set_latent_free_projectors(old_anchor.projectors, cfg.latent_strength)
    ww.set_requires_grad(teacher_new_model.parameters(), False)

    student = restore_model(vocab_size, old_anchor.checkpoint)
    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()
    ww.set_requires_grad(ww.all_adapter_params(student), False)
    ww.set_requires_grad(ww.all_base_params(student), True)
    optimizer = torch.optim.AdamW(
        ww.all_base_params(student),
        lr=cfg.consolidation_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )

    step_index = 0
    total_steps = 0
    for phase_steps, projection_strength, power, floor, lr_scale in phases:
        for group in optimizer.param_groups:
            group["lr"] = cfg.consolidation_lr * lr_scale
        for _ in range(phase_steps):
            step_index += 1
            total_steps += 1
            old_batch = make_task_batch(
                stoi,
                cfg.old_task,
                cfg.old_tag,
                cfg.seed + 296_000,
                step_index,
                batch_size=cfg.batch_size,
                source_len=cfg.source_len_train,
            )
            new_batch = make_task_batch(
                stoi,
                cfg.new_task,
                cfg.new_tag,
                cfg.seed + 297_000,
                step_index,
                batch_size=cfg.batch_size,
                source_len=cfg.source_len_train,
            )
            optimizer.zero_grad(set_to_none=True)
            student_old_logits, _ = student(old_batch.x, old_batch.y)
            student_new_logits, _ = student(new_batch.x, new_batch.y)
            with torch.no_grad():
                teacher_old_logits, _ = teacher_old_model(old_batch.x, old_batch.y)
                teacher_new_logits, _ = teacher_new_model(new_batch.x, new_batch.y)
            loss_old = toy.task_weighted_loss(student_old_logits, old_batch) + cfg.old_kl_weight * kl_divergence(student_old_logits, teacher_old_logits)
            loss_new = toy.task_weighted_loss(student_new_logits, new_batch) + cfg.kl_weight * kl_divergence(student_new_logits, teacher_new_logits)
            loss = 0.5 * (loss_old + loss_new)
            loss.backward()
            if projection_strength > 0.0:
                project_block_gradients_weighted(
                    student,
                    old_anchor.grad_basis,
                    old_anchor.grad_importance,
                    projection_strength,
                    power=power,
                    floor=floor,
                )
            torch.nn.utils.clip_grad_norm_(ww.all_base_params(student), ww.GRAD_CLIP)
            optimizer.step()

    old_eval_batches = task_batches(
        stoi,
        cfg.old_task,
        cfg.old_tag,
        cfg.seed + 298_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    new_eval_batches = task_batches(
        stoi,
        cfg.new_task,
        cfg.new_tag,
        cfg.seed + 299_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    old_metrics = eval_task(student, old_eval_batches)
    new_metrics = eval_task(student, new_eval_batches)
    text_metrics = eval_text(student, val_data, cfg)
    del teacher_old_model, teacher_new_model, student, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return make_result(
        variant=variant,
        model_family="base_water_schedule",
        teacher_replay_enabled=False,
        consolidation_old_enabled=False,
        hard_old_grad_proj=0.0,
        teacher_replay_count=0,
        teacher_replay_budget=0,
        consolidation_old_count=total_steps,
        consolidation_old_budget=total_steps,
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        text_metrics=text_metrics,
    )


def collect_activation_anchors(
    model: ww.TinyGPT,
    train_data: torch.Tensor,
    num_probes: int,
    batch_size: int,
    seed: int,
) -> List[List[torch.Tensor]]:
    """Run probe inputs through model and save per-block hidden states.

    Returns a list of probe results, where each probe result is a list of
    tensors (one per block) containing the hidden state [batch, seq, d_model].
    """
    was_training = model.training
    model.eval()
    
    # Ensure gradients are enabled so retain_grad() inside TinyGPT doesn't crash
    ww.set_requires_grad(ww.all_base_params(model), True)
    
    anchors: List[List[torch.Tensor]] = []
    rng = np.random.default_rng(seed)
    data_len = len(train_data)
    for probe_idx in range(num_probes):
        # Generate random text positions for probe
        starts = rng.integers(0, data_len - ww.BLOCK_SIZE - 1, size=batch_size)
        x = torch.stack(
            [train_data[int(s): int(s) + ww.BLOCK_SIZE] for s in starts], dim=0
        ).to(ww.DEVICE)
        y = torch.stack(
            [train_data[int(s) + 1: int(s) + ww.BLOCK_SIZE + 1] for s in starts], dim=0
        ).to(ww.DEVICE)
        
        _, _, activations = model(x, y, return_activations=True)
        anchors.append([act.detach().cpu() for act in activations])
        
    ww.set_requires_grad(ww.all_base_params(model), False)
    if was_training:
        model.train()
    return anchors


def activation_anchor_loss(
    model: ww.TinyGPT,
    probe_inputs: List[Tuple[torch.Tensor, torch.Tensor]],
    saved_activations: List[List[torch.Tensor]],
) -> torch.Tensor:
    """Compute MSE between current and saved hidden states on probe inputs."""
    total_loss = torch.tensor(0.0, device=ww.DEVICE)
    count = 0
    for (x, y), saved_acts in zip(probe_inputs, saved_activations):
        _, _, current_acts = model(x, y, return_activations=True)
        for layer_idx, (current, saved) in enumerate(zip(current_acts, saved_acts)):
            saved_dev = saved.to(device=current.device, dtype=current.dtype)
            total_loss = total_loss + F.mse_loss(current, saved_dev)
            count += 1
    if count > 0:
        total_loss = total_loss / count
    return total_loss


def train_amoeba_only_no_teacher(
    vocab_size: int,
    stoi: Dict[str, int],
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    cfg: argparse.Namespace,
    *,
    variant: str,
    phases: Sequence[Tuple[int, float, float, float, float]],
) -> ConsolidationResult:
    """True zero-replay path 1: new-task-only training with staged amoeba projection.

    No old-task data. No teachers. No KL. Just raw task loss on new data,
    with the amoeba membrane protecting old-skill geometry.
    """
    student = restore_model(vocab_size, old_anchor.checkpoint)
    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()
    ww.set_requires_grad(ww.all_adapter_params(student), False)
    ww.set_requires_grad(ww.all_base_params(student), True)
    optimizer = torch.optim.AdamW(
        ww.all_base_params(student),
        lr=cfg.consolidation_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )

    step_index = 0
    total_steps = 0
    for phase_steps, projection_strength, power, floor, lr_scale in phases:
        for group in optimizer.param_groups:
            group["lr"] = cfg.consolidation_lr * lr_scale
        for _ in range(phase_steps):
            step_index += 1
            total_steps += 1
            new_batch = make_task_batch(
                stoi,
                cfg.new_task,
                cfg.new_tag,
                cfg.seed + 310_000,
                step_index,
                batch_size=cfg.batch_size,
                source_len=cfg.source_len_train,
            )
            optimizer.zero_grad(set_to_none=True)
            student_logits, _ = student(new_batch.x, new_batch.y)
            loss = toy.task_weighted_loss(student_logits, new_batch)
            loss.backward()
            if projection_strength > 0.0:
                project_block_gradients_weighted(
                    student,
                    old_anchor.grad_basis,
                    old_anchor.grad_importance,
                    projection_strength,
                    power=power,
                    floor=floor,
                )
            torch.nn.utils.clip_grad_norm_(ww.all_base_params(student), ww.GRAD_CLIP)
            optimizer.step()

    old_eval_batches = task_batches(
        stoi, cfg.old_task, cfg.old_tag, cfg.seed + 311_000,
        num_batches=cfg.task_eval_batches, batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    new_eval_batches = task_batches(
        stoi, cfg.new_task, cfg.new_tag, cfg.seed + 312_000,
        num_batches=cfg.task_eval_batches, batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    old_metrics = eval_task(student, old_eval_batches)
    new_metrics = eval_task(student, new_eval_batches)
    text_metrics = eval_text(student, val_data, cfg)
    del student, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return make_result(
        variant=variant,
        model_family="base_amoeba_only",
        teacher_replay_enabled=False,
        consolidation_old_enabled=False,
        hard_old_grad_proj=0.0,
        teacher_replay_count=0,
        teacher_replay_budget=0,
        consolidation_old_count=0,
        consolidation_old_budget=0,
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        text_metrics=text_metrics,
    )


def train_amoeba_activation_anchor(
    vocab_size: int,
    stoi: Dict[str, int],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    cfg: argparse.Namespace,
    *,
    variant: str,
    phases: Sequence[Tuple[int, float, float, float, float]],
    anchor_weight: float,
    anchor_period: int = 4,
) -> ConsolidationResult:
    """True zero-replay path 2: new-task training + amoeba + hidden state anchor.

    No old-task data at all. Instead of using old-task batches to preserve the
    old skill, this saves hidden-state snapshots from the anchor model on random
    text probes. During training, periodically compute MSE between current and
    saved hidden states to geometrically anchor the old-skill circuits.
    """
    # Collect activation anchors from the old-skill model on random text probes
    anchor_model = restore_model(vocab_size, old_anchor.checkpoint)
    anchor_model.set_adapters_enabled(False)
    anchor_model.clear_latent_free_projectors()
    anchor_model.eval()

    num_probes = cfg.activation_anchor_probes
    saved_anchors = collect_activation_anchors(
        anchor_model, train_data, num_probes, cfg.batch_size, cfg.seed + 320_000,
    )

    # Also save the probe inputs so we can re-feed them during training
    rng_replay = np.random.default_rng(cfg.seed + 320_000)
    data_len = len(train_data)
    probe_inputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(num_probes):
        starts = rng_replay.integers(0, data_len - ww.BLOCK_SIZE - 1, size=cfg.batch_size)
        x = torch.stack(
            [train_data[int(s): int(s) + ww.BLOCK_SIZE] for s in starts], dim=0
        ).to(ww.DEVICE)
        y = torch.stack(
            [train_data[int(s) + 1: int(s) + ww.BLOCK_SIZE + 1] for s in starts], dim=0
        ).to(ww.DEVICE)
        probe_inputs.append((x, y))

    del anchor_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Train student
    student = restore_model(vocab_size, old_anchor.checkpoint)
    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()
    ww.set_requires_grad(ww.all_adapter_params(student), False)
    ww.set_requires_grad(ww.all_base_params(student), True)
    optimizer = torch.optim.AdamW(
        ww.all_base_params(student),
        lr=cfg.consolidation_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )

    step_index = 0
    total_steps = 0
    for phase_steps, projection_strength, power, floor, lr_scale in phases:
        for group in optimizer.param_groups:
            group["lr"] = cfg.consolidation_lr * lr_scale
        for _ in range(phase_steps):
            step_index += 1
            total_steps += 1
            new_batch = make_task_batch(
                stoi,
                cfg.new_task,
                cfg.new_tag,
                cfg.seed + 330_000,
                step_index,
                batch_size=cfg.batch_size,
                source_len=cfg.source_len_train,
            )
            optimizer.zero_grad(set_to_none=True)
            student_logits, _ = student(new_batch.x, new_batch.y)
            task_loss = toy.task_weighted_loss(student_logits, new_batch)

            # Periodically add activation anchor regularization
            if anchor_weight > 0.0 and step_index % anchor_period == 0:
                anchor_loss = activation_anchor_loss(
                    student, probe_inputs, saved_anchors,
                )
                loss = task_loss + anchor_weight * anchor_loss
            else:
                loss = task_loss

            loss.backward()
            if projection_strength > 0.0:
                project_block_gradients_weighted(
                    student,
                    old_anchor.grad_basis,
                    old_anchor.grad_importance,
                    projection_strength,
                    power=power,
                    floor=floor,
                )
            torch.nn.utils.clip_grad_norm_(ww.all_base_params(student), ww.GRAD_CLIP)
            optimizer.step()

    old_eval_batches = task_batches(
        stoi, cfg.old_task, cfg.old_tag, cfg.seed + 331_000,
        num_batches=cfg.task_eval_batches, batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    new_eval_batches = task_batches(
        stoi, cfg.new_task, cfg.new_tag, cfg.seed + 332_000,
        num_batches=cfg.task_eval_batches, batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    old_metrics = eval_task(student, old_eval_batches)
    new_metrics = eval_task(student, new_eval_batches)
    text_metrics = eval_text(student, val_data, cfg)
    del student, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return make_result(
        variant=variant,
        model_family="base_act_anchor",
        teacher_replay_enabled=False,
        consolidation_old_enabled=False,
        hard_old_grad_proj=0.0,
        teacher_replay_count=0,
        teacher_replay_budget=0,
        consolidation_old_count=0,
        consolidation_old_budget=0,
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        text_metrics=text_metrics,
    )


def train_zero_replay_lateral_merge(
    vocab_size: int,
    stoi: Dict[str, int],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    cfg: argparse.Namespace,
    *,
    variant: str,
    phases: Sequence[Tuple[int, float, float, float, float]],
    proxy_weight: float = 1.0,
) -> ConsolidationResult:
    """True zero-replay path 3: Lateral Merge.
    
    Phase 1: Train an adapter on the new task while the base is completely frozen.
             This guarantees perfectly zero forgetting because base is untouched.
    Phase 2: Distill the Teacher (FrozenBase + TrainedAdapter) into a Student (DenseBase).
             Uses Amoeba projection to protect the old skill's linear geometry during distillation.
             Uses New Task data, plus Proxy Text data to keep LayerNorms calibrated.
    """
    # Phase 1: Train isolated adapter
    teacher_new = train_teacher_b(
        vocab_size, stoi, val_data, old_anchor, cfg, replay_enabled=False,
    )
    
    # Phase 2 Setup: Teacher has adapters ON, is frozen
    teacher = restore_model(vocab_size, teacher_new.checkpoint)
    teacher.set_adapters_enabled(True)
    teacher.set_latent_free_projectors(old_anchor.projectors, cfg.latent_strength)
    ww.set_requires_grad(teacher.parameters(), False)
    teacher.eval()
    
    # Phase 2 Setup: Student is dense base, unfrozen
    student = restore_model(vocab_size, old_anchor.checkpoint)
    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()
    ww.set_requires_grad(ww.all_adapter_params(student), False)
    ww.set_requires_grad(ww.all_base_params(student), True)
    
    optimizer = torch.optim.AdamW(
        ww.all_base_params(student),
        lr=cfg.consolidation_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )
    
    rng_proxy = np.random.default_rng(cfg.seed + 340_000)
    data_len = len(train_data)
    
    step_index = 0
    total_steps = 0
    for phase_steps, projection_strength, power, floor, lr_scale in phases:
        for group in optimizer.param_groups:
            group["lr"] = cfg.consolidation_lr * lr_scale
        for _ in range(phase_steps):
            step_index += 1
            total_steps += 1
            
            # Form New Task batch
            new_batch = make_task_batch(
                stoi,
                cfg.new_task,
                cfg.new_tag,
                cfg.seed + 350_000,
                step_index,
                batch_size=cfg.batch_size,
                source_len=cfg.source_len_train,
            )
            
            # Form Proxy Text batch
            starts = rng_proxy.integers(0, data_len - ww.BLOCK_SIZE - 1, size=cfg.batch_size)
            px = torch.stack([train_data[int(s): int(s) + ww.BLOCK_SIZE] for s in starts], dim=0).to(ww.DEVICE)
            py = torch.stack([train_data[int(s) + 1: int(s) + ww.BLOCK_SIZE + 1] for s in starts], dim=0).to(ww.DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            # New Task Distillation
            student_new_logits, _ = student(new_batch.x, new_batch.y)
            with torch.no_grad():
                teacher_new_logits, _ = teacher(new_batch.x, new_batch.y)
            loss_new = toy.task_weighted_loss(student_new_logits, new_batch) + cfg.kl_weight * kl_divergence(student_new_logits, teacher_new_logits)
            loss = loss_new
            
            # Proxy Text Distillation (anchors the non-linearities for old skill)
            if proxy_weight > 0.0:
                student_proxy_logits, _ = student(px, py)
                with torch.no_grad():
                    teacher_proxy_logits, _ = teacher(px, py)
                loss_proxy = kl_divergence(student_proxy_logits, teacher_proxy_logits)
                loss = loss + proxy_weight * loss_proxy
                
            loss.backward()
            
            # Amoeba Projection
            if projection_strength > 0.0:
                project_block_gradients_weighted(
                    student,
                    old_anchor.grad_basis,
                    old_anchor.grad_importance,
                    projection_strength,
                    power=power,
                    floor=floor,
                )
            torch.nn.utils.clip_grad_norm_(ww.all_base_params(student), ww.GRAD_CLIP)
            optimizer.step()

    old_eval_batches = task_batches(
        stoi, cfg.old_task, cfg.old_tag, cfg.seed + 351_000,
        num_batches=cfg.task_eval_batches, batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    new_eval_batches = task_batches(
        stoi, cfg.new_task, cfg.new_tag, cfg.seed + 352_000,
        num_batches=cfg.task_eval_batches, batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    old_metrics = eval_task(student, old_eval_batches)
    new_metrics = eval_task(student, new_eval_batches)
    text_metrics = eval_text(student, val_data, cfg)
    
    del student, teacher, teacher_new, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return make_result(
        variant=variant,
        model_family="base_lateral_merge",
        teacher_replay_enabled=False,
        consolidation_old_enabled=False,
        hard_old_grad_proj=0.0,
        teacher_replay_count=0,
        teacher_replay_budget=0,
        consolidation_old_count=0,
        consolidation_old_budget=0,
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        text_metrics=text_metrics,
    )


def _covariance_svd(X: torch.Tensor, rank: int) -> torch.Tensor:
    """Compute top-k eigenvectors of centered covariance of X [n_samples, dim]."""
    X = X - X.mean(dim=0, keepdim=True)
    dim = X.shape[1]
    cov = (X.t() @ X) / max(X.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    chosen = order[:min(rank, dim)]
    return eigvecs[:, chosen].t().contiguous()  # [rank, dim]


def collect_per_layer_activation_basis(
    model: ww.TinyGPT,
    stoi: Dict[str, int],
    task_name: str,
    task_tag: str,
    seed: int,
    cfg: argparse.Namespace,
    rank: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Collect per-layer input activation covariance for ALL linear layers.

    Hooks every linear layer individually:
      - b{i}.attn.qkv   input: ln1(x)          dim=d_model
      - b{i}.attn.proj   input: attn_output      dim=d_model
      - b{i}.mlp.fc1     input: ln2(x+attn)      dim=d_model
      - b{i}.mlp.fc2     input: gelu(fc1_out)     dim=4*d_model

    Returns dict mapping layer key -> V_old [rank, input_dim].
    """
    d_model = ww.D_MODEL
    if rank is None:
        rank = d_model // 2

    was_training = model.training
    model.eval()

    layer_activations: Dict[str, List[torch.Tensor]] = {}
    hooks = []

    for idx, block in enumerate(model.blocks):
        targets = [
            (f"b{idx}.attn.qkv", block.attn.qkv),
            (f"b{idx}.attn.proj", block.attn.proj),
            (f"b{idx}.mlp.fc1", block.mlp.fc1),
            (f"b{idx}.mlp.fc2", block.mlp.fc2),
        ]
        for layer_key, module in targets:
            layer_activations[layer_key] = []
            def make_hook(key: str):
                def hook_fn(mod, inp, out):
                    x = inp[0].detach().float()
                    # Flatten all dims except the last (feature dim)
                    layer_activations[key].append(x.reshape(-1, x.shape[-1]).cpu())
                return hook_fn
            h = module.register_forward_hook(make_hook(layer_key))
            hooks.append(h)

    # Run old-task data through the model
    num_probes = max(cfg.grad_basis_batches, 4)
    for index in range(num_probes):
        batch = make_task_batch(
            stoi, task_name, task_tag, seed, index,
            batch_size=cfg.batch_size, source_len=cfg.source_len_eval,
        )
        with torch.no_grad():
            model(batch.x, batch.y)

    for h in hooks:
        h.remove()

    # Compute per-layer activation basis
    basis_by_layer: Dict[str, torch.Tensor] = {}
    for layer_key, act_list in layer_activations.items():
        if not act_list:
            basis_by_layer[layer_key] = torch.empty(0, 0)
            continue
        X = torch.cat(act_list, dim=0)
        input_dim = X.shape[1]
        effective_rank = min(rank, input_dim)
        basis_by_layer[layer_key] = _covariance_svd(X, effective_rank)

    if was_training:
        model.train()
    return basis_by_layer


# Keep the old block-level version for backward compat
def collect_activation_basis(
    model: ww.TinyGPT,
    stoi: Dict[str, int],
    task_name: str,
    task_tag: str,
    seed: int,
    cfg: argparse.Namespace,
    rank: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Block-level activation basis (legacy). See collect_per_layer_activation_basis."""
    d_model = ww.D_MODEL
    if rank is None:
        rank = d_model // 2
    was_training = model.training
    model.eval()
    block_activations: Dict[str, List[torch.Tensor]] = {block: [] for block in ww.block_keys()}
    hooks = []
    for idx, block in enumerate(model.blocks):
        block_name = f"b{idx}"
        def make_hook(name: str):
            def hook_fn(module, input, output):
                x = input[0].detach()
                block_activations[name].append(x.reshape(-1, d_model).float().cpu())
            return hook_fn
        h = block.register_forward_hook(make_hook(block_name))
        hooks.append(h)
    num_probes = max(cfg.grad_basis_batches, 4)
    for index in range(num_probes):
        batch = make_task_batch(stoi, task_name, task_tag, seed, index,
                                batch_size=cfg.batch_size, source_len=cfg.source_len_eval)
        with torch.no_grad():
            model(batch.x, batch.y)
    for h in hooks:
        h.remove()
    basis_by_block: Dict[str, torch.Tensor] = {}
    for block_name, act_list in block_activations.items():
        if not act_list:
            basis_by_block[block_name] = torch.empty(0, d_model)
            continue
        X = torch.cat(act_list, dim=0)
        basis_by_block[block_name] = _covariance_svd(X, min(rank, d_model))
    if was_training:
        model.train()
    return basis_by_block


def project_all_weight_grads_activation_space(
    model: ww.TinyGPT,
    act_basis_by_layer: Dict[str, torch.Tensor],
    strength: float,
) -> None:
    """Project ALL linear layer weight gradients in their per-layer activation space.

    Targets every linear layer: qkv, proj, fc1, fc2.
    For each, projects gradient rows away from old-task input directions so that
    ΔW @ x_old ≈ 0 for that layer's specific input distribution.
    """
    if strength <= 0.0:
        return

    for idx, block in enumerate(model.blocks):
        targets = [
            (f"b{idx}.attn.qkv", block.attn.qkv.weight),
            (f"b{idx}.attn.proj", block.attn.proj.weight),
            (f"b{idx}.mlp.fc1", block.mlp.fc1.weight),
            (f"b{idx}.mlp.fc2", block.mlp.fc2.weight),
        ]
        for layer_key, w in targets:
            V_old = act_basis_by_layer.get(layer_key)
            if V_old is None or V_old.numel() == 0 or w.grad is None:
                continue
            V_old = V_old.to(device=w.device, dtype=torch.float32)
            g = w.grad.float()
            # g: [out, in], V_old: [rank, in]
            # Project: g' = g - strength * g @ V^T @ V
            proj = g @ V_old.t() @ V_old
            w.grad.copy_((g - strength * proj).to(w.grad.dtype))


# Keep old version for backward compat
def project_weight_grad_activation_space(
    model: ww.TinyGPT,
    act_basis_by_block: Dict[str, torch.Tensor],
    strength: float,
) -> None:
    """Block-level activation projection (legacy). See project_all_weight_grads_activation_space."""
    if strength <= 0.0:
        return
    for idx, block in enumerate(model.blocks):
        block_name = f"b{idx}"
        V_old = act_basis_by_block.get(block_name)
        if V_old is None or V_old.numel() == 0:
            continue
        V_old = V_old.to(device=ww.DEVICE, dtype=torch.float32)
        for w in [block.attn.qkv.weight, block.mlp.fc1.weight]:
            if w.grad is None:
                continue
            g = w.grad.float()
            if g.dim() == 2:
                proj = g @ V_old.t() @ V_old
                w.grad.copy_((g - strength * proj).to(w.grad.dtype))


def train_nonlinear_amoeba_lateral_merge(
    vocab_size: int,
    stoi: Dict[str, int],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    cfg: argparse.Namespace,
    *,
    variant: str,
    phases: Sequence[Tuple[int, float, float, float, float]],
    proxy_weight: float = 1.0,
    freeze_ln: bool = True,
    activation_rank: int | None = None,
    per_phase_rank: Sequence[int] | None = None,
    old_anchor_weight: float = 0.0,
    hidden_match_weight: float = 0.0,
    task_hidden_match_weight: float = 0.0,
    task_kl_weight: float = 0.0,
) -> ConsolidationResult:
    """True zero-replay with FULLY SEALED Non-Linear Amoeba.

    All leaks sealed:
    1. Embeddings (token + position) and head: FROZEN (never change)
    2. LayerNorm (all γ, β): FROZEN during projection phases
    3. ALL linear layers (qkv, proj, fc1, fc2): activation-space projected per-layer
    4. Parameter-space Amoeba on all block params

    Only trainable during projection: block linear weights (with dual projection).
    Only trainable during polish: block linear weights + LayerNorm.
    Embeddings + head NEVER trained — kept at old-anchor values.

    Uses lateral merge: distill Teacher(Base+Adapter) -> Student(DenseBase)
    with only new-task data + generic text proxy. Zero old-task data.
    """
    # Collect PER-LAYER activation basis from old-task data DURING ANCHORING
    # If per_phase_rank is used, collect at max rank and we'll slice per-phase
    max_rank = activation_rank if activation_rank is not None else cfg.activation_rank
    if per_phase_rank is not None:
        max_rank = max(per_phase_rank)
    anchor_model = restore_model(vocab_size, old_anchor.checkpoint)
    anchor_model.set_adapters_enabled(False)
    anchor_model.clear_latent_free_projectors()
    act_basis_full = collect_per_layer_activation_basis(
        anchor_model, stoi, cfg.old_task, cfg.old_tag,
        cfg.seed + 360_000, cfg,
        rank=max_rank,
    )
    del anchor_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 1: Train isolated adapter on new task (base frozen)
    teacher_new = train_teacher_b(
        vocab_size, stoi, val_data, old_anchor, cfg, replay_enabled=False,
    )

    # Phase 2 Setup: Teacher = Base + Trained Adapter
    teacher = restore_model(vocab_size, teacher_new.checkpoint)
    teacher.set_adapters_enabled(True)
    teacher.set_latent_free_projectors(old_anchor.projectors, cfg.latent_strength)
    ww.set_requires_grad(teacher.parameters(), False)
    teacher.eval()

    # Old Anchor model for dual-anchor distillation and hidden state matching
    old_anchor_model = None
    if old_anchor_weight > 0.0 or hidden_match_weight > 0.0 or task_hidden_match_weight > 0.0 or task_kl_weight > 0.0:
        old_anchor_model = restore_model(vocab_size, old_anchor.checkpoint)
        old_anchor_model.set_adapters_enabled(False)
        old_anchor_model.clear_latent_free_projectors()
        ww.set_requires_grad(old_anchor_model.parameters(), False)
        old_anchor_model.eval()

    # Student = Dense Base (from old anchor)
    student = restore_model(vocab_size, old_anchor.checkpoint)
    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()

    # SEAL: Freeze embeddings, position embeddings, head — NEVER trained
    ww.set_requires_grad(list(student.token_embedding.parameters()), False)
    ww.set_requires_grad(list(student.position_embedding.parameters()), False)
    ww.set_requires_grad(list(student.head.parameters()), False)

    # Freeze adapters
    ww.set_requires_grad(ww.all_adapter_params(student), False)

    # Identify non-linear params (LayerNorm γ, β) for conditional freezing
    ln_param_ids = set()
    for block in student.blocks:
        for p in block.ln1.parameters():
            ln_param_ids.add(id(p))
        for p in block.ln2.parameters():
            ln_param_ids.add(id(p))
    for p in student.ln_f.parameters():
        ln_param_ids.add(id(p))

    # Enable block base params (linear weights + LN)
    for block in student.blocks:
        for p in block.base_parameters():
            p.requires_grad = True

    # Collect trainable params for optimizer (block base params only)
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.consolidation_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )

    rng_proxy = np.random.default_rng(cfg.seed + 370_000)
    data_len = len(train_data)

    step_index = 0
    for phase_idx, (phase_steps, projection_strength, power, floor, lr_scale) in enumerate(phases):
        for group in optimizer.param_groups:
            group["lr"] = cfg.consolidation_lr * lr_scale

        # Determine activation basis for this phase (potentially rank-sliced)
        if per_phase_rank is not None and phase_idx < len(per_phase_rank):
            phase_rank = per_phase_rank[phase_idx]
            act_basis = {k: v[:phase_rank] if v.numel() > 0 else v
                         for k, v in act_basis_full.items()}
        else:
            act_basis = act_basis_full

        # Freeze/unfreeze LayerNorm based on projection phase
        if freeze_ln and projection_strength > 0.0:
            for p in trainable:
                if id(p) in ln_param_ids:
                    p.requires_grad = False
        else:
            for p in trainable:
                if id(p) in ln_param_ids:
                    p.requires_grad = True

        for _ in range(phase_steps):
            step_index += 1

            # New Task batch
            new_batch = make_task_batch(
                stoi, cfg.new_task, cfg.new_tag,
                cfg.seed + 380_000, step_index,
                batch_size=cfg.batch_size, source_len=cfg.source_len_train,
            )

            # Proxy Text batch
            starts = rng_proxy.integers(0, data_len - ww.BLOCK_SIZE - 1, size=cfg.batch_size)
            px = torch.stack([train_data[int(s): int(s) + ww.BLOCK_SIZE] for s in starts], dim=0).to(ww.DEVICE)
            py = torch.stack([train_data[int(s) + 1: int(s) + ww.BLOCK_SIZE + 1] for s in starts], dim=0).to(ww.DEVICE)

            optimizer.zero_grad(set_to_none=True)

            # Distill from Teacher on new task
            student_new_logits, _ = student(new_batch.x, new_batch.y)
            with torch.no_grad():
                teacher_new_logits, _ = teacher(new_batch.x, new_batch.y)
            loss = toy.task_weighted_loss(student_new_logits, new_batch) + cfg.kl_weight * kl_divergence(student_new_logits, teacher_new_logits)

            # Task-data KL anchor: match old anchor OUTPUT on new-task data (no separate corpus needed)
            if old_anchor_model is not None and task_kl_weight > 0.0:
                with torch.no_grad():
                    old_anchor_task_logits, _ = old_anchor_model(new_batch.x, new_batch.y)
                loss = loss + task_kl_weight * kl_divergence(student_new_logits, old_anchor_task_logits)

            # Proxy text distillation (anchors non-linearities)
            if proxy_weight > 0.0:
                student_proxy_logits, _ = student(px, py)
                with torch.no_grad():
                    teacher_proxy_logits, _ = teacher(px, py)
                loss = loss + proxy_weight * kl_divergence(student_proxy_logits, teacher_proxy_logits)

            # Dual-anchor: also match old anchor on proxy text (active pull-back)
            if old_anchor_model is not None and old_anchor_weight > 0.0:
                if proxy_weight <= 0.0:
                    student_proxy_logits, _ = student(px, py)
                with torch.no_grad():
                    old_anchor_proxy_logits, _ = old_anchor_model(px, py)
                loss = loss + old_anchor_weight * kl_divergence(student_proxy_logits, old_anchor_proxy_logits)

            # Hidden State Matching: match internal representations at every layer
            if old_anchor_model is not None and hidden_match_weight > 0.0:
                # Collect student hidden states on proxy text
                student_hiddens = []
                s_hooks = []
                for blk in student.blocks:
                    def make_s_hook(store):
                        def hook_fn(mod, inp, out):
                            store.append(inp[0])  # residual stream entering block
                        return hook_fn
                    h = blk.register_forward_hook(make_s_hook(student_hiddens))
                    s_hooks.append(h)
                student(px, py)  # already computed above but we need hooks active
                for h in s_hooks:
                    h.remove()

                # Collect anchor hidden states on proxy text
                anchor_hiddens = []
                a_hooks = []
                for blk in old_anchor_model.blocks:
                    def make_a_hook(store):
                        def hook_fn(mod, inp, out):
                            store.append(inp[0].detach())
                        return hook_fn
                    h = blk.register_forward_hook(make_a_hook(anchor_hiddens))
                    a_hooks.append(h)
                with torch.no_grad():
                    old_anchor_model(px, py)
                for h in a_hooks:
                    h.remove()

                # MSE between hidden states at each layer
                hmatch_loss = torch.tensor(0.0, device=ww.DEVICE)
                for sh, ah in zip(student_hiddens, anchor_hiddens):
                    hmatch_loss = hmatch_loss + torch.nn.functional.mse_loss(sh, ah)
                loss = loss + hidden_match_weight * hmatch_loss

            # Hidden State Matching on NEW-TASK DATA (exercises shared circuits)
            if old_anchor_model is not None and task_hidden_match_weight > 0.0:
                # Collect student hidden states on new-task input
                student_task_hiddens = []
                st_hooks = []
                for blk in student.blocks:
                    def make_st_hook(store):
                        def hook_fn(mod, inp, out):
                            store.append(inp[0])
                        return hook_fn
                    h = blk.register_forward_hook(make_st_hook(student_task_hiddens))
                    st_hooks.append(h)
                student(new_batch.x, new_batch.y)  # forward on new-task data with hooks
                for h in st_hooks:
                    h.remove()

                # Collect anchor hidden states on new-task input
                anchor_task_hiddens = []
                at_hooks = []
                for blk in old_anchor_model.blocks:
                    def make_at_hook(store):
                        def hook_fn(mod, inp, out):
                            store.append(inp[0].detach())
                        return hook_fn
                    h = blk.register_forward_hook(make_at_hook(anchor_task_hiddens))
                    at_hooks.append(h)
                with torch.no_grad():
                    old_anchor_model(new_batch.x, new_batch.y)
                for h in at_hooks:
                    h.remove()

                # MSE between hidden states at each layer on task data
                thmatch_loss = torch.tensor(0.0, device=ww.DEVICE)
                n_layers = len(student_task_hiddens)
                for layer_idx, (sh, ah) in enumerate(zip(student_task_hiddens, anchor_task_hiddens)):
                    # Weight early layers more (shared preprocessing) vs later (task-specific)
                    layer_weight = 1.0 - 0.5 * (layer_idx / max(n_layers - 1, 1))
                    thmatch_loss = thmatch_loss + layer_weight * torch.nn.functional.mse_loss(sh, ah)
                loss = loss + task_hidden_match_weight * thmatch_loss

            loss.backward()

            # SEALED NON-LINEAR AMOEBA: per-layer activation-space projection
            if projection_strength > 0.0:
                project_all_weight_grads_activation_space(student, act_basis, projection_strength)

            # LINEAR AMOEBA: parameter-space projection
            if projection_strength > 0.0:
                project_block_gradients_weighted(
                    student,
                    old_anchor.grad_basis,
                    old_anchor.grad_importance,
                    projection_strength,
                    power=power,
                    floor=floor,
                )

            torch.nn.utils.clip_grad_norm_(trainable, ww.GRAD_CLIP)
            optimizer.step()

    # Ensure all params set for eval
    student.eval()

    old_eval_batches = task_batches(
        stoi, cfg.old_task, cfg.old_tag, cfg.seed + 381_000,
        num_batches=cfg.task_eval_batches, batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    new_eval_batches = task_batches(
        stoi, cfg.new_task, cfg.new_tag, cfg.seed + 382_000,
        num_batches=cfg.task_eval_batches, batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    old_metrics = eval_task(student, old_eval_batches)
    new_metrics = eval_task(student, new_eval_batches)
    text_metrics = eval_text(student, val_data, cfg)

    del student, teacher, teacher_new, optimizer
    if old_anchor_model is not None:
        del old_anchor_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return make_result(
        variant=variant,
        model_family="base_nonlinear_amoeba",
        teacher_replay_enabled=False,
        consolidation_old_enabled=False,
        hard_old_grad_proj=0.0,
        teacher_replay_count=0,
        teacher_replay_budget=0,
        consolidation_old_count=0,
        consolidation_old_budget=0,
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        text_metrics=text_metrics,
    )


def train_joint_dual_teacher_microinject(
    vocab_size: int,
    stoi: Dict[str, int],
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    cfg: argparse.Namespace,
    *,
    variant: str,
    core_threshold: float,
    core_scale: float,
    fringe_scale: float,
    null_boost: float,
    preserve_norm: bool,
    ln_heal_steps: int = 0,
) -> ConsolidationResult:
    teacher_new = train_teacher_b(
        vocab_size,
        stoi,
        val_data,
        old_anchor,
        cfg,
        replay_enabled=False,
    )
    teacher_old_model = restore_model(vocab_size, old_anchor.checkpoint)
    teacher_old_model.set_adapters_enabled(False)
    teacher_old_model.clear_latent_free_projectors()
    ww.set_requires_grad(teacher_old_model.parameters(), False)

    teacher_new_model = restore_model(vocab_size, teacher_new.checkpoint)
    teacher_new_model.set_adapters_enabled(True)
    teacher_new_model.set_latent_free_projectors(old_anchor.projectors, cfg.latent_strength)
    ww.set_requires_grad(teacher_new_model.parameters(), False)

    student = restore_model(vocab_size, old_anchor.checkpoint)
    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()
    ww.set_requires_grad(ww.all_adapter_params(student), False)
    ww.set_requires_grad(ww.all_base_params(student), True)
    optimizer = torch.optim.AdamW(
        ww.all_base_params(student),
        lr=cfg.consolidation_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )

    for step in range(1, cfg.consolidation_steps + 1):
        old_batch = make_task_batch(
            stoi,
            cfg.old_task,
            cfg.old_tag,
            cfg.seed + 290_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        new_batch = make_task_batch(
            stoi,
            cfg.new_task,
            cfg.new_tag,
            cfg.seed + 291_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        optimizer.zero_grad(set_to_none=True)
        student_old_logits, _ = student(old_batch.x, old_batch.y)
        student_new_logits, _ = student(new_batch.x, new_batch.y)
        with torch.no_grad():
            teacher_old_logits, _ = teacher_old_model(old_batch.x, old_batch.y)
            teacher_new_logits, _ = teacher_new_model(new_batch.x, new_batch.y)
        loss_old = toy.task_weighted_loss(student_old_logits, old_batch) + cfg.old_kl_weight * kl_divergence(student_old_logits, teacher_old_logits)
        loss_new = toy.task_weighted_loss(student_new_logits, new_batch) + cfg.kl_weight * kl_divergence(student_new_logits, teacher_new_logits)
        loss = 0.5 * (loss_old + loss_new)
        loss.backward()
        microinject_block_gradients(
            student,
            old_anchor.grad_basis,
            old_anchor.grad_importance,
            core_threshold=core_threshold,
            core_scale=core_scale,
            fringe_scale=fringe_scale,
            null_boost=null_boost,
            preserve_norm=preserve_norm,
        )
        torch.nn.utils.clip_grad_norm_(ww.all_base_params(student), ww.GRAD_CLIP)
        optimizer.step()

    if ln_heal_steps > 0:
        student.set_adapters_enabled(False)
        student.clear_latent_free_projectors()
        ww.set_requires_grad(student.parameters(), False)
        ln_params = layernorm_params(student)
        ww.set_requires_grad(ln_params, True)
        ln_optimizer = torch.optim.AdamW(
            [param for param in ln_params if param.requires_grad],
            lr=cfg.ln_heal_lr,
            betas=ww.BETAS,
            weight_decay=0.0,
        )
        for step in range(1, ln_heal_steps + 1):
            old_batch = make_task_batch(
                stoi,
                cfg.old_task,
                cfg.old_tag,
                cfg.seed + 292_000,
                step,
                batch_size=cfg.batch_size,
                source_len=cfg.source_len_train,
            )
            new_batch = make_task_batch(
                stoi,
                cfg.new_task,
                cfg.new_tag,
                cfg.seed + 293_000,
                step,
                batch_size=cfg.batch_size,
                source_len=cfg.source_len_train,
            )
            ln_optimizer.zero_grad(set_to_none=True)
            student_old_logits, _ = student(old_batch.x, old_batch.y)
            student_new_logits, _ = student(new_batch.x, new_batch.y)
            with torch.no_grad():
                teacher_old_logits, _ = teacher_old_model(old_batch.x, old_batch.y)
                teacher_new_logits, _ = teacher_new_model(new_batch.x, new_batch.y)
            loss = kl_divergence(student_old_logits, teacher_old_logits) + kl_divergence(student_new_logits, teacher_new_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ln_params, ww.GRAD_CLIP)
            ln_optimizer.step()
        del ln_optimizer

    old_eval_batches = task_batches(
        stoi,
        cfg.old_task,
        cfg.old_tag,
        cfg.seed + 294_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    new_eval_batches = task_batches(
        stoi,
        cfg.new_task,
        cfg.new_tag,
        cfg.seed + 295_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    old_metrics = eval_task(student, old_eval_batches)
    new_metrics = eval_task(student, new_eval_batches)
    text_metrics = eval_text(student, val_data, cfg)
    del teacher_old_model, teacher_new_model, student, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return make_result(
        variant=variant,
        model_family="base_microinject",
        teacher_replay_enabled=False,
        consolidation_old_enabled=False,
        hard_old_grad_proj=0.0,
        teacher_replay_count=0,
        teacher_replay_budget=0,
        consolidation_old_count=cfg.consolidation_steps + max(ln_heal_steps, 0),
        consolidation_old_budget=cfg.consolidation_steps + max(ln_heal_steps, 0),
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        text_metrics=text_metrics,
    )


def consolidate_student(
    vocab_size: int,
    stoi: Dict[str, int],
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    teacher_b: TeacherResult,
    cfg: argparse.Namespace,
    *,
    variant: str,
    consolidation_old_enabled: bool,
    hard_old_grad_proj: float = 0.0,
) -> ConsolidationResult:
    teacher_old = restore_model(vocab_size, old_anchor.checkpoint)
    teacher_old.set_adapters_enabled(False)
    teacher_old.clear_latent_free_projectors()
    ww.set_requires_grad(teacher_old.parameters(), False)

    teacher_new = restore_model(vocab_size, teacher_b.checkpoint)
    teacher_new.set_adapters_enabled(True)
    teacher_new.set_latent_free_projectors(old_anchor.projectors, cfg.latent_strength)
    ww.set_requires_grad(teacher_new.parameters(), False)

    student = restore_model(vocab_size, old_anchor.checkpoint)
    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()
    ww.set_requires_grad(ww.all_adapter_params(student), False)
    ww.set_requires_grad(ww.all_base_params(student), True)
    optimizer = torch.optim.AdamW(
        ww.all_base_params(student),
        lr=cfg.consolidation_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )

    old_eval_batches = task_batches(
        stoi,
        cfg.old_task,
        cfg.old_tag,
        cfg.seed + 80_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    new_eval_batches = task_batches(
        stoi,
        cfg.new_task,
        cfg.new_tag,
        cfg.seed + 90_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )

    old_count = 0
    old_budget = 0
    if consolidation_old_enabled and cfg.consolidation_old_period > 0:
        old_budget = sum(1 for step in range(1, cfg.consolidation_steps + 1) if step % cfg.consolidation_old_period == 1)

    for step in range(1, cfg.consolidation_steps + 1):
        use_old = consolidation_old_enabled and cfg.consolidation_old_period > 0 and step % cfg.consolidation_old_period == 1
        if use_old:
            batch = make_task_batch(
                stoi,
                cfg.old_task,
                cfg.old_tag,
                cfg.seed + 100_000,
                old_count,
                batch_size=cfg.batch_size,
                source_len=cfg.source_len_train,
            )
            teacher = teacher_old
            kl_weight = cfg.old_kl_weight
            old_count += 1
        else:
            batch = make_task_batch(
                stoi,
                cfg.new_task,
                cfg.new_tag,
                cfg.seed + 110_000,
                step,
                batch_size=cfg.batch_size,
                source_len=cfg.source_len_train,
            )
            teacher = teacher_new
            kl_weight = cfg.kl_weight

        optimizer.zero_grad(set_to_none=True)
        student_logits, _ = student(batch.x, batch.y)
        with torch.no_grad():
            teacher_logits, _ = teacher(batch.x, batch.y)
        task_loss = toy.task_weighted_loss(student_logits, batch)
        loss = task_loss + kl_weight * kl_divergence(student_logits, teacher_logits)
        loss.backward()
        if not use_old and hard_old_grad_proj > 0.0:
            ww.project_block_gradients(student, old_anchor.grad_basis, hard_old_grad_proj)
        torch.nn.utils.clip_grad_norm_(ww.all_base_params(student), ww.GRAD_CLIP)
        optimizer.step()

    old_metrics = eval_task(student, old_eval_batches)
    new_metrics = eval_task(student, new_eval_batches)
    text_metrics = eval_text(student, val_data, cfg)
    del teacher_old, teacher_new, student, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return make_result(
        variant=variant,
        model_family="base_only",
        teacher_replay_enabled=teacher_b.replay_enabled,
        consolidation_old_enabled=consolidation_old_enabled,
        hard_old_grad_proj=hard_old_grad_proj,
        teacher_replay_count=teacher_b.replay_count,
        teacher_replay_budget=teacher_b.replay_budget,
        consolidation_old_count=old_count,
        consolidation_old_budget=old_budget,
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        text_metrics=text_metrics,
    )


def train_prop_expanded_teacher(
    vocab_size: int,
    stoi: Dict[str, int],
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    cfg: argparse.Namespace,
) -> ModelCheckpointResult:
    checkpoint = {
        "model": old_anchor.checkpoint,
    }
    model, optimizer = prop.restore_prop_checkpoint(vocab_size, checkpoint, load_optimizer=False)
    prop.prime_prop_block_from_last_base(model)
    model.set_adapters_enabled(False)
    model.clear_latent_free_projectors()
    ww.set_requires_grad(prop.prop_all_base_params(model), False)
    ww.set_requires_grad(prop.prop_trainable_params(model), True)
    optimizer = prop.make_prop_optimizer(model, cfg.expansion_lr, params=prop.prop_trainable_params(model))

    for step in range(1, cfg.expansion_steps + 1):
        batch = make_task_batch(
            stoi,
            cfg.new_task,
            cfg.new_tag,
            cfg.seed + 170_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        model.train()
        model.set_adapters_enabled(False)
        model.clear_latent_free_projectors()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch.x, batch.y)
        loss = toy.task_weighted_loss(logits, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prop.prop_trainable_params(model), ww.GRAD_CLIP)
        optimizer.step()

    old_eval_batches = task_batches(
        stoi,
        cfg.old_task,
        cfg.old_tag,
        cfg.seed + 150_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    new_eval_batches = task_batches(
        stoi,
        cfg.new_task,
        cfg.new_tag,
        cfg.seed + 160_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    model.eval()
    old_metrics = eval_task(model, old_eval_batches)
    new_metrics = eval_task(model, new_eval_batches)
    text_metrics = eval_text(model, val_data, cfg)
    output = ModelCheckpointResult(
        checkpoint=prop.make_prop_checkpoint(model, optimizer),
        model_family="expanded_prop",
        old_answer_acc=float(old_metrics["answer_acc"]),
        old_seq_acc=float(old_metrics["seq_acc"]),
        new_answer_acc=float(new_metrics["answer_acc"]),
        new_seq_acc=float(new_metrics["seq_acc"]),
        text_loss=float(text_metrics["loss"]),
        text_acc=float(text_metrics["acc"]),
    )
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def train_expanded_no_replay(
    vocab_size: int,
    stoi: Dict[str, int],
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    cfg: argparse.Namespace,
) -> ConsolidationResult:
    checkpoint = {
        "model": old_anchor.checkpoint,
        "n_layer_override": ww.N_LAYER + expand.EXTRA_BLOCKS,
    }
    model, _optimizer = expand.restore_layer_checkpoint(vocab_size, checkpoint, load_optimizer=False)
    expand.prime_new_block_from_last_base(model)
    model.set_adapters_enabled(False)
    model.clear_latent_free_projectors()
    ww.set_requires_grad(expand.model_all_base_params(model), False)
    ww.set_requires_grad(expand.model_all_adapter_params(model), False)
    ww.set_requires_grad(expand.model_new_block_params(model), True)
    optimizer = expand.make_layer_optimizer(model, cfg.expansion_lr, params=expand.model_new_block_params(model))

    old_eval_batches = task_batches(
        stoi,
        cfg.old_task,
        cfg.old_tag,
        cfg.seed + 120_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    new_eval_batches = task_batches(
        stoi,
        cfg.new_task,
        cfg.new_tag,
        cfg.seed + 130_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )

    for step in range(1, cfg.expansion_steps + 1):
        batch = make_task_batch(
            stoi,
            cfg.new_task,
            cfg.new_tag,
            cfg.seed + 140_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        model.train()
        model.set_adapters_enabled(False)
        model.clear_latent_free_projectors()
        expand.enforce_new_block_gate_floor(model, cfg.expansion_gate_floor_logit)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch.x, batch.y)
        loss = toy.task_weighted_loss(logits, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expand.model_new_block_params(model), ww.GRAD_CLIP)
        optimizer.step()

    model.eval()
    old_metrics = eval_task(model, old_eval_batches)
    new_metrics = eval_task(model, new_eval_batches)
    text_metrics = eval_text(model, val_data, cfg)
    del model, _optimizer, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ConsolidationResult(
        variant="no_replay_expand",
        model_family="expanded_gated",
        teacher_replay_enabled=False,
        consolidation_old_enabled=False,
        hard_old_grad_proj=0.0,
        teacher_replay_count=0,
        teacher_replay_budget=0,
        consolidation_old_count=0,
        consolidation_old_budget=0,
        old_answer_acc=float(old_metrics["answer_acc"]),
        old_seq_acc=float(old_metrics["seq_acc"]),
        new_answer_acc=float(new_metrics["answer_acc"]),
        new_seq_acc=float(new_metrics["seq_acc"]),
        text_loss=float(text_metrics["loss"]),
        text_acc=float(text_metrics["acc"]),
        balanced_mean=0.5 * (float(old_metrics["answer_acc"]) + float(new_metrics["answer_acc"])),
        balanced_geom=math.sqrt(max(float(old_metrics["answer_acc"]), 0.0) * max(float(new_metrics["answer_acc"]), 0.0)),
    )


def train_prop_expanded_no_replay(
    vocab_size: int,
    stoi: Dict[str, int],
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    cfg: argparse.Namespace,
) -> ConsolidationResult:
    teacher = train_prop_expanded_teacher(vocab_size, stoi, val_data, old_anchor, cfg)
    return make_result(
        variant="no_replay_prop_expand",
        model_family=teacher.model_family,
        teacher_replay_enabled=False,
        consolidation_old_enabled=False,
        hard_old_grad_proj=0.0,
        teacher_replay_count=0,
        teacher_replay_budget=0,
        consolidation_old_count=0,
        consolidation_old_budget=0,
        old_metrics={"answer_acc": teacher.old_answer_acc, "seq_acc": teacher.old_seq_acc},
        new_metrics={"answer_acc": teacher.new_answer_acc, "seq_acc": teacher.new_seq_acc},
        text_metrics={"loss": teacher.text_loss, "acc": teacher.text_acc},
    )


def train_prop_expand_then_consolidate(
    vocab_size: int,
    stoi: Dict[str, int],
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    cfg: argparse.Namespace,
) -> ConsolidationResult:
    teacher_prop = train_prop_expanded_teacher(vocab_size, stoi, val_data, old_anchor, cfg)
    teacher_old_model = restore_model(vocab_size, old_anchor.checkpoint)
    teacher_old_model.set_adapters_enabled(False)
    teacher_old_model.clear_latent_free_projectors()
    ww.set_requires_grad(teacher_old_model.parameters(), False)

    teacher_new_model, _teacher_opt = prop.restore_prop_checkpoint(vocab_size, teacher_prop.checkpoint, load_optimizer=False)
    teacher_new_model.set_adapters_enabled(False)
    teacher_new_model.clear_latent_free_projectors()
    ww.set_requires_grad(teacher_new_model.parameters(), False)

    student = restore_model(vocab_size, old_anchor.checkpoint)
    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()
    ww.set_requires_grad(ww.all_adapter_params(student), False)
    ww.set_requires_grad(ww.all_base_params(student), True)
    optimizer = torch.optim.AdamW(
        ww.all_base_params(student),
        lr=cfg.consolidation_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )

    for step in range(1, cfg.consolidation_steps + 1):
        old_batch = make_task_batch(
            stoi,
            cfg.old_task,
            cfg.old_tag,
            cfg.seed + 230_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        new_batch = make_task_batch(
            stoi,
            cfg.new_task,
            cfg.new_tag,
            cfg.seed + 240_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        optimizer.zero_grad(set_to_none=True)
        student_old_logits, _ = student(old_batch.x, old_batch.y)
        student_new_logits, _ = student(new_batch.x, new_batch.y)
        with torch.no_grad():
            teacher_old_logits, _ = teacher_old_model(old_batch.x, old_batch.y)
            teacher_new_logits, _ = teacher_new_model(new_batch.x, new_batch.y)
        loss_old = toy.task_weighted_loss(student_old_logits, old_batch) + cfg.old_kl_weight * kl_divergence(student_old_logits, teacher_old_logits)
        loss_new = toy.task_weighted_loss(student_new_logits, new_batch) + cfg.kl_weight * kl_divergence(student_new_logits, teacher_new_logits)
        loss = 0.5 * (loss_old + loss_new)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ww.all_base_params(student), ww.GRAD_CLIP)
        optimizer.step()

    old_eval_batches = task_batches(
        stoi,
        cfg.old_task,
        cfg.old_tag,
        cfg.seed + 241_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    new_eval_batches = task_batches(
        stoi,
        cfg.new_task,
        cfg.new_tag,
        cfg.seed + 242_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    old_metrics = eval_task(student, old_eval_batches)
    new_metrics = eval_task(student, new_eval_batches)
    text_metrics = eval_text(student, val_data, cfg)
    del teacher_old_model, teacher_new_model, _teacher_opt, student, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return make_result(
        variant="prop_expand_then_consolidate",
        model_family="base_from_prop",
        teacher_replay_enabled=False,
        consolidation_old_enabled=False,
        hard_old_grad_proj=0.0,
        teacher_replay_count=0,
        teacher_replay_budget=0,
        consolidation_old_count=cfg.consolidation_steps,
        consolidation_old_budget=cfg.consolidation_steps,
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        text_metrics=text_metrics,
    )


def layernorm_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if ".ln" in name or name.startswith("ln_f."):
            params.append(param)
    return params


def train_hard_project_ln_heal(
    vocab_size: int,
    stoi: Dict[str, int],
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    cfg: argparse.Namespace,
) -> ConsolidationResult:
    teacher_new = train_teacher_b(
        vocab_size,
        stoi,
        val_data,
        old_anchor,
        cfg,
        replay_enabled=False,
    )
    teacher_old_model = restore_model(vocab_size, old_anchor.checkpoint)
    teacher_old_model.set_adapters_enabled(False)
    teacher_old_model.clear_latent_free_projectors()
    ww.set_requires_grad(teacher_old_model.parameters(), False)

    teacher_new_model = restore_model(vocab_size, teacher_new.checkpoint)
    teacher_new_model.set_adapters_enabled(True)
    teacher_new_model.set_latent_free_projectors(old_anchor.projectors, cfg.latent_strength)
    ww.set_requires_grad(teacher_new_model.parameters(), False)

    student = restore_model(vocab_size, old_anchor.checkpoint)
    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()
    ww.set_requires_grad(ww.all_adapter_params(student), False)
    ww.set_requires_grad(ww.all_base_params(student), True)
    optimizer = torch.optim.AdamW(
        ww.all_base_params(student),
        lr=cfg.consolidation_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )

    for step in range(1, cfg.consolidation_steps + 1):
        batch = make_task_batch(
            stoi,
            cfg.new_task,
            cfg.new_tag,
            cfg.seed + 180_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        optimizer.zero_grad(set_to_none=True)
        student_logits, _ = student(batch.x, batch.y)
        with torch.no_grad():
            teacher_logits, _ = teacher_new_model(batch.x, batch.y)
        task_loss = toy.task_weighted_loss(student_logits, batch)
        loss = task_loss + cfg.kl_weight * kl_divergence(student_logits, teacher_logits)
        loss.backward()
        if cfg.hard_old_grad_proj > 0.0:
            ww.project_block_gradients(student, old_anchor.grad_basis, cfg.hard_old_grad_proj)
        torch.nn.utils.clip_grad_norm_(ww.all_base_params(student), ww.GRAD_CLIP)
        optimizer.step()

    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()
    ww.set_requires_grad(student.parameters(), False)
    ln_params = layernorm_params(student)
    ww.set_requires_grad(ln_params, True)
    ln_optimizer = torch.optim.AdamW(
        [param for param in ln_params if param.requires_grad],
        lr=cfg.ln_heal_lr,
        betas=ww.BETAS,
        weight_decay=0.0,
    )

    for step in range(1, cfg.ln_heal_steps + 1):
        old_batch = make_task_batch(
            stoi,
            cfg.old_task,
            cfg.old_tag,
            cfg.seed + 190_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        new_batch = make_task_batch(
            stoi,
            cfg.new_task,
            cfg.new_tag,
            cfg.seed + 200_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        ln_optimizer.zero_grad(set_to_none=True)
        student_old_logits, _ = student(old_batch.x, old_batch.y)
        student_new_logits, _ = student(new_batch.x, new_batch.y)
        with torch.no_grad():
            teacher_old_logits, _ = teacher_old_model(old_batch.x, old_batch.y)
            teacher_new_logits, _ = teacher_new_model(new_batch.x, new_batch.y)
        loss = (
            kl_divergence(student_old_logits, teacher_old_logits)
            + kl_divergence(student_new_logits, teacher_new_logits)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ln_params, ww.GRAD_CLIP)
        ln_optimizer.step()

    old_eval_batches = task_batches(
        stoi,
        cfg.old_task,
        cfg.old_tag,
        cfg.seed + 210_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    new_eval_batches = task_batches(
        stoi,
        cfg.new_task,
        cfg.new_tag,
        cfg.seed + 220_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    student.eval()
    old_metrics = eval_task(student, old_eval_batches)
    new_metrics = eval_task(student, new_eval_batches)
    text_metrics = eval_text(student, val_data, cfg)
    del teacher_old_model, teacher_new_model, student, optimizer, ln_optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ConsolidationResult(
        variant="hard_project_ln_heal",
        model_family="base_ln_heal",
        teacher_replay_enabled=False,
        consolidation_old_enabled=False,
        hard_old_grad_proj=cfg.hard_old_grad_proj,
        teacher_replay_count=0,
        teacher_replay_budget=0,
        consolidation_old_count=0,
        consolidation_old_budget=0,
        old_answer_acc=float(old_metrics["answer_acc"]),
        old_seq_acc=float(old_metrics["seq_acc"]),
        new_answer_acc=float(new_metrics["answer_acc"]),
        new_seq_acc=float(new_metrics["seq_acc"]),
        text_loss=float(text_metrics["loss"]),
        text_acc=float(text_metrics["acc"]),
        balanced_mean=0.5 * (float(old_metrics["answer_acc"]) + float(new_metrics["answer_acc"])),
        balanced_geom=math.sqrt(max(float(old_metrics["answer_acc"]), 0.0) * max(float(new_metrics["answer_acc"]), 0.0)),
    )


def train_prop_expand_ln_heal(
    vocab_size: int,
    stoi: Dict[str, int],
    val_data: torch.Tensor,
    old_anchor: AnchorResult,
    cfg: argparse.Namespace,
) -> ConsolidationResult:
    teacher_prop = train_prop_expanded_teacher(vocab_size, stoi, val_data, old_anchor, cfg)
    teacher_old_model = restore_model(vocab_size, old_anchor.checkpoint)
    teacher_old_model.set_adapters_enabled(False)
    teacher_old_model.clear_latent_free_projectors()
    ww.set_requires_grad(teacher_old_model.parameters(), False)

    teacher_new_model, _teacher_opt = prop.restore_prop_checkpoint(vocab_size, teacher_prop.checkpoint, load_optimizer=False)
    teacher_new_model.set_adapters_enabled(False)
    teacher_new_model.clear_latent_free_projectors()
    ww.set_requires_grad(teacher_new_model.parameters(), False)

    student, _student_opt = prop.restore_prop_checkpoint(vocab_size, teacher_prop.checkpoint, load_optimizer=False)
    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()
    ww.set_requires_grad(student.parameters(), False)
    ln_params = layernorm_params(student)
    ww.set_requires_grad(ln_params, True)
    optimizer = torch.optim.AdamW(
        [param for param in ln_params if param.requires_grad],
        lr=cfg.ln_heal_lr,
        betas=ww.BETAS,
        weight_decay=0.0,
    )

    for step in range(1, cfg.ln_heal_steps + 1):
        old_batch = make_task_batch(
            stoi,
            cfg.old_task,
            cfg.old_tag,
            cfg.seed + 250_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        new_batch = make_task_batch(
            stoi,
            cfg.new_task,
            cfg.new_tag,
            cfg.seed + 260_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        optimizer.zero_grad(set_to_none=True)
        student_old_logits, _ = student(old_batch.x, old_batch.y)
        student_new_logits, _ = student(new_batch.x, new_batch.y)
        with torch.no_grad():
            teacher_old_logits, _ = teacher_old_model(old_batch.x, old_batch.y)
            teacher_new_logits, _ = teacher_new_model(new_batch.x, new_batch.y)
        loss = kl_divergence(student_old_logits, teacher_old_logits) + kl_divergence(student_new_logits, teacher_new_logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ln_params, ww.GRAD_CLIP)
        optimizer.step()

    old_eval_batches = task_batches(
        stoi,
        cfg.old_task,
        cfg.old_tag,
        cfg.seed + 261_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    new_eval_batches = task_batches(
        stoi,
        cfg.new_task,
        cfg.new_tag,
        cfg.seed + 262_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    old_metrics = eval_task(student, old_eval_batches)
    new_metrics = eval_task(student, new_eval_batches)
    text_metrics = eval_text(student, val_data, cfg)
    del teacher_old_model, teacher_new_model, _teacher_opt, student, _student_opt, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return make_result(
        variant="prop_expand_ln_heal",
        model_family="expanded_prop_ln",
        teacher_replay_enabled=False,
        consolidation_old_enabled=False,
        hard_old_grad_proj=0.0,
        teacher_replay_count=0,
        teacher_replay_budget=0,
        consolidation_old_count=cfg.ln_heal_steps,
        consolidation_old_budget=cfg.ln_heal_steps,
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        text_metrics=text_metrics,
    )


def print_summary(base_text_metrics: Dict[str, float], anchor: AnchorResult, rows: Sequence[ConsolidationResult], cfg: argparse.Namespace) -> None:
    evidence = build_evidence_tables(base_text_metrics, anchor, rows, cfg)
    print("=" * 78)
    print("TOY REPLAY-INDEPENDENCE CONTINUAL-LEARNING BENCHMARK")
    print("=" * 78)
    print(
        f"device={ww.DEVICE} old_task={cfg.old_task} new_task={cfg.new_task} "
        f"anchor_steps={cfg.anchor_steps} task_steps={cfg.task_steps} consolidation_steps={cfg.consolidation_steps}"
    )
    print(
        f"base_text_loss={base_text_metrics['loss']:.3f} anchor_old_answer={anchor.old_answer_acc:.3f} "
        f"anchor_old_seq={anchor.old_seq_acc:.3f} ready={anchor.reached_ready}"
    )
    print("-" * 78)
    print(
        f"{'variant':<24} {'family':<13} {'teach_rep':>9} {'old_mix':>8} {'hard_p':>8} {'old_ans':>8} {'new_ans':>8} {'bal':>8} {'geom':>8} {'text_loss':>10}"
    )
    for row in rows:
        print(
            f"{row.variant:<24} {row.model_family:<13} {row.teacher_replay_count:>4d}/{row.teacher_replay_budget:<4d} "
            f"{row.consolidation_old_count:>3d}/{row.consolidation_old_budget:<3d} {row.hard_old_grad_proj:>8.2f} "
            f"{row.old_answer_acc:>8.3f} {row.new_answer_acc:>8.3f} {row.balanced_mean:>8.3f} {row.balanced_geom:>8.3f} {row.text_loss:>10.3f}"
        )
    full = next(row for row in rows if row.variant == "full_pipeline")
    no_teacher = next(row for row in rows if row.variant == "no_teacher_replay")
    none = next(row for row in rows if row.variant == "no_replay_anywhere")
    print("-" * 78)
    print(
        f"no_teacher_replay vs full: "
        f"old_answer {no_teacher.old_answer_acc - full.old_answer_acc:+.3f}, "
        f"new_answer {no_teacher.new_answer_acc - full.new_answer_acc:+.3f}, "
        f"balanced {no_teacher.balanced_mean - full.balanced_mean:+.3f}, "
        f"geom {no_teacher.balanced_geom - full.balanced_geom:+.3f}"
    )
    print(
        f"no_replay_anywhere vs full: "
        f"old_answer {none.old_answer_acc - full.old_answer_acc:+.3f}, "
        f"new_answer {none.new_answer_acc - full.new_answer_acc:+.3f}, "
        f"balanced {none.balanced_mean - full.balanced_mean:+.3f}, "
        f"geom {none.balanced_geom - full.balanced_geom:+.3f}, "
        f"text_loss {none.text_loss - full.text_loss:+.3f}"
    )
    if (
        abs(no_teacher.old_answer_acc - full.old_answer_acc) <= 0.03
        and abs(no_teacher.new_answer_acc - full.new_answer_acc) <= 0.03
        and abs(no_teacher.balanced_mean - full.balanced_mean) <= 0.03
    ):
        print("signal: teacher-stage replay did not matter in this toy benchmark.")
    if none.balanced_geom > full.balanced_geom + 0.05 and none.old_answer_acc >= 0.50:
        print("signal: replay-free produced a better balanced CL score than the replay-heavy path once task identities were separated.")
    elif none.old_answer_acc >= anchor.old_answer_acc - 0.05 and none.new_answer_acc > 0.0:
        print("mixed: replay-free still works meaningfully, but replay may still help enough that we should not call it irrelevant.")
    else:
        print("diagnostic: no-replay-anywhere still changed the old/new tradeoff too much to claim replay-independence outright.")
    fixed_rows = [row for row in rows if row.model_family.startswith("base")]
    expanded_rows = [row for row in rows if row.model_family.startswith("expanded")]
    if fixed_rows:
        best_fixed = max(fixed_rows, key=lambda row: (row.balanced_geom, row.balanced_mean))
        print(
            f"best fixed-size: {best_fixed.variant} old={best_fixed.old_answer_acc:.3f} "
            f"new={best_fixed.new_answer_acc:.3f} bal={best_fixed.balanced_mean:.3f} geom={best_fixed.balanced_geom:.3f}"
        )
    if expanded_rows:
        best_expanded = max(expanded_rows, key=lambda row: (row.balanced_geom, row.balanced_mean))
        print(
            f"best expansion: {best_expanded.variant} old={best_expanded.old_answer_acc:.3f} "
            f"new={best_expanded.new_answer_acc:.3f} bal={best_expanded.balanced_mean:.3f} geom={best_expanded.balanced_geom:.3f}"
        )
    print("=" * 78)
    print("BLOG / PAPER EVIDENCE TABLES")
    print("=" * 78)
    _print_evidence_rows("Toy Replay Table", evidence["toy_replay_table"])
    _print_evidence_rows("Toy Hidden-Match Synergy Table", evidence["toy_synergy_table"])
    acceptance = evidence["acceptance"]
    print(
        "acceptance: "
        f"teacher_replay_irrelevant={acceptance['teacher_replay_irrelevant']} "
        f"old_signal_location_matters={acceptance['old_signal_location_matters']} "
        f"proxy_replay_free_geom_ge_0_82={acceptance['proxy_replay_free_geom_ge_0_82']} "
        f"hidden_match_dual_synergy={acceptance['hidden_match_dual_synergy']}"
    )
    print(f"callout: {evidence['callout']}")


def main() -> None:
    cfg = parse_args()
    normalize_output_paths(cfg)
    with fast_mode(cfg):
        vocab_size, stoi, train_data, val_data, base_state, base_text_metrics = prepare_base(cfg)
        anchor = train_old_anchor(vocab_size, stoi, train_data, val_data, base_state, cfg)
        if not anchor.reached_ready:
            payload = {
                "valid_retention_test": False,
                "reason": "Old anchor did not reach readiness.",
                "config": vars(cfg),
                "base_text_metrics": base_text_metrics,
                "anchor": asdict(anchor),
            }
            cfg.json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            write_csv(cfg.csv_path, [{"status": "invalid_anchor", "reason": payload["reason"], "old_answer_acc": anchor.old_answer_acc, "old_seq_acc": anchor.old_seq_acc}])
            print("=" * 78)
            print("TOY REPLAY-INDEPENDENCE CONTINUAL-LEARNING BENCHMARK")
            print("=" * 78)
            print(payload["reason"])
            print(f"saved: {cfg.json_path}")
            print(f"saved: {cfg.csv_path}")
            return

        configs = [
            ("full_pipeline", True, True, 0.0),
            ("no_teacher_replay", False, True, 0.0),
            ("no_consolidation_old", True, False, 0.0),
            ("no_replay_anywhere", False, False, 0.0),
        ]
        if cfg.hard_old_grad_proj > 0.0:
            configs.append(("no_replay_hard_project", False, False, cfg.hard_old_grad_proj))
        rows: List[ConsolidationResult] = []
        for variant, teacher_replay_enabled, consolidation_old_enabled, hard_old_grad_proj in configs:
            print(
                f"[{variant}] teacher_replay={teacher_replay_enabled} "
                f"consolidation_old={consolidation_old_enabled} hard_old_grad_proj={hard_old_grad_proj:.2f}"
            )
            teacher_b = train_teacher_b(
                vocab_size,
                stoi,
                val_data,
                anchor,
                cfg,
                replay_enabled=teacher_replay_enabled,
            )
            result = consolidate_student(
                vocab_size,
                stoi,
                val_data,
                anchor,
                teacher_b,
                cfg,
                variant=variant,
                consolidation_old_enabled=consolidation_old_enabled,
                hard_old_grad_proj=hard_old_grad_proj,
            )
            rows.append(result)
        if cfg.include_expansion:
            print("[no_replay_expand] frozen_old=True extra_block=1 replay=0")
            rows.append(train_expanded_no_replay(vocab_size, stoi, val_data, anchor, cfg))
        if cfg.include_prop_expansion:
            print("[no_replay_prop_expand] frozen_old=True prop_block=1 replay=0")
            rows.append(train_prop_expanded_no_replay(vocab_size, stoi, val_data, anchor, cfg))
        if cfg.include_ln_heal:
            print(f"[hard_project_ln_heal] hard_old_grad_proj={cfg.hard_old_grad_proj:.2f} ln_heal_steps={cfg.ln_heal_steps}")
            rows.append(train_hard_project_ln_heal(vocab_size, stoi, val_data, anchor, cfg))
        if cfg.include_joint_distill:
            print(f"[joint_no_replay_dual_teacher] hard_old_grad_proj={cfg.hard_old_grad_proj:.2f} old+new every step")
            rows.append(
                train_joint_dual_teacher_no_replay(
                    vocab_size,
                    stoi,
                    val_data,
                    anchor,
                    cfg,
                    hard_old_grad_proj=cfg.hard_old_grad_proj,
                )
            )
        if cfg.include_prop_post_consolidation:
            print("[prop_expand_then_consolidate] prop teacher -> base joint consolidate")
            rows.append(train_prop_expand_then_consolidate(vocab_size, stoi, val_data, anchor, cfg))
        if cfg.include_prop_post_ln_heal:
            print("[prop_expand_ln_heal] prop teacher -> expanded LN-only heal")
            rows.append(train_prop_expand_ln_heal(vocab_size, stoi, val_data, anchor, cfg))

        # ── Amoeba / Water-Weights sweep ──────────────────────────────────
        if cfg.include_amoeba:
            amoeba_configs = [
                # (variant_name, projection_strength, power, floor, ln_heal_steps)
                # power controls how sharply importance drops off
                # floor is the minimum protection even for lowest-importance dims
                # low floor + high power = aggressive amoeba (compress old harshly)
                # high floor + low power = gentle amoeba (protect old more)
                ("amoeba_gentle",     1.0, 1.0, 0.3, 0),
                ("amoeba_moderate",   1.0, 2.0, 0.1, 0),
                ("amoeba_aggressive", 1.0, 3.0, 0.0, 0),
                ("amoeba_moderate_ln", 1.0, 2.0, 0.1, cfg.ln_heal_steps),
                ("amoeba_aggressive_ln", 1.0, 3.0, 0.0, cfg.ln_heal_steps),
            ]
            for vname, proj_str, power, floor, ln_steps in amoeba_configs:
                print(f"[{vname}] strength={proj_str} power={power} floor={floor} ln_heal={ln_steps}")
                rows.append(
                    train_joint_dual_teacher_water_weights(
                        vocab_size, stoi, val_data, anchor, cfg,
                        variant=vname,
                        projection_strength=proj_str,
                        water_power=power,
                        water_floor=floor,
                        ln_heal_steps=ln_steps,
                    )
                )

        # ── Microinject sweep ─────────────────────────────────────────────
        if cfg.include_microinject:
            microinject_configs = [
                # (variant, core_threshold, core_scale, fringe_scale, null_boost, preserve_norm, ln_heal)
                # core_threshold: importance above this = fully protected
                # core_scale: multiplier on protected gradient component
                # fringe_scale: multiplier on low-importance (compressible) gradient component
                # null_boost: multiplier on null-space (free) gradient component
                ("micro_conservative", 0.3, 0.0, 0.5, 1.5, True, 0),
                ("micro_balanced",     0.5, 0.0, 0.8, 2.0, True, 0),
                ("micro_aggressive",   0.7, 0.0, 1.0, 3.0, True, 0),
                ("micro_balanced_ln",  0.5, 0.0, 0.8, 2.0, True, cfg.ln_heal_steps),
                ("micro_aggressive_ln", 0.7, 0.0, 1.0, 3.0, True, cfg.ln_heal_steps),
            ]
            for vname, ct, cs, fs, nb, pn, ln_steps in microinject_configs:
                print(f"[{vname}] core_t={ct} core_s={cs} fringe_s={fs} null_b={nb} norm={pn} ln={ln_steps}")
                rows.append(
                    train_joint_dual_teacher_microinject(
                        vocab_size, stoi, val_data, anchor, cfg,
                        variant=vname,
                        core_threshold=ct,
                        core_scale=cs,
                        fringe_scale=fs,
                        null_boost=nb,
                        preserve_norm=pn,
                        ln_heal_steps=ln_steps,
                    )
                )

        # ── True Zero-Replay: Amoeba-Only (no teacher, no old data) ──────
        if cfg.include_amoeba_only:
            # Same staged schedule as the winning amoeba config
            amoeba_only_configs = [
                (
                    "zero_replay_amoeba",
                    [
                        (120, 1.0, 1.0, 0.3, 1.0),   # gentle
                        (240, 1.0, 2.0, 0.1, 1.0),   # moderate
                        (40, 0.0, 1.0, 0.0, 0.2),    # polish (no projection, low LR)
                    ],
                ),
                (
                    "zero_replay_amoeba_long",
                    [
                        (180, 1.0, 1.0, 0.3, 1.0),   # gentle longer
                        (360, 1.0, 2.0, 0.1, 1.0),   # moderate longer
                        (60, 0.0, 1.0, 0.0, 0.2),    # polish
                    ],
                ),
                (
                    "zero_replay_amoeba_tight",
                    [
                        (60, 1.0, 1.0, 0.5, 1.0),    # gentle but higher floor
                        (300, 1.0, 2.0, 0.2, 1.0),   # moderate with more protection
                        (40, 0.0, 1.0, 0.0, 0.2),    # polish
                    ],
                ),
            ]
            for vname, phases in amoeba_only_configs:
                total = sum(p[0] for p in phases)
                print(f"[{vname}] phases={len(phases)} total_steps={total} NO old data, NO teachers")
                rows.append(
                    train_amoeba_only_no_teacher(
                        vocab_size, stoi, val_data, anchor, cfg,
                        variant=vname,
                        phases=phases,
                    )
                )

        # ── True Zero-Replay: Activation Anchor ──────────────────────────
        if cfg.include_activation_anchor:
            act_anchor_configs = [
                (
                    "zero_replay_act_anchor",
                    [
                        (120, 1.0, 1.0, 0.3, 1.0),
                        (240, 1.0, 2.0, 0.1, 1.0),
                        (40, 0.0, 1.0, 0.0, 0.2),
                    ],
                    cfg.activation_anchor_weight,
                    4,  # anchor every 4 steps
                ),
                (
                    "zero_replay_act_anchor_strong",
                    [
                        (120, 1.0, 1.0, 0.3, 1.0),
                        (240, 1.0, 2.0, 0.1, 1.0),
                        (40, 0.0, 1.0, 0.0, 0.2),
                    ],
                    cfg.activation_anchor_weight * 2.0,
                    2,  # anchor every 2 steps
                ),
                (
                    "zero_replay_act_anchor_tight",
                    [
                        (60, 1.0, 1.0, 0.5, 1.0),
                        (300, 1.0, 2.0, 0.2, 1.0),
                        (40, 0.0, 1.0, 0.0, 0.2),
                    ],
                    cfg.activation_anchor_weight,
                    4,
                ),
            ]
            for vname, phases, aw, ap in act_anchor_configs:
                total = sum(p[0] for p in phases)
                print(f"[{vname}] phases={len(phases)} total={total} anchor_weight={aw} anchor_period={ap}")
                rows.append(
                    train_amoeba_activation_anchor(
                        vocab_size, stoi, train_data, val_data, anchor, cfg,
                        variant=vname,
                        phases=phases,
                        anchor_weight=aw,
                        anchor_period=ap,
                    )
                )

        # ── True Zero-Replay: Lateral Merge ──────────────────────────────
        if cfg.include_lateral_merge:
            lateral_merge_configs = [
                (
                    "zero_replay_lateral_merge",
                    [
                        (120, 1.0, 1.0, 0.3, 1.0),
                        (240, 1.0, 2.0, 0.1, 1.0),
                        (40, 0.0, 1.0, 0.0, 0.2), # polish
                    ],
                    cfg.lateral_proxy_weight,
                ),
                (
                    "zero_replay_lateral_merge_no_proxy",
                    [
                        (120, 1.0, 1.0, 0.3, 1.0),
                        (240, 1.0, 2.0, 0.1, 1.0),
                        (40, 0.0, 1.0, 0.0, 0.2),
                    ],
                    0.0, # no generic text anchor
                ),
                (
                    "zero_replay_lateral_merge_tight",
                    [
                        (60, 1.0, 1.0, 0.5, 1.0),
                        (300, 1.0, 2.0, 0.2, 1.0),
                        (40, 0.0, 1.0, 0.0, 0.2),
                    ],
                    cfg.lateral_proxy_weight,
                ),
            ]
            for vname, phases, pw in lateral_merge_configs:
                total = sum(p[0] for p in phases)
                print(f"[{vname}] phases={len(phases)} total={total} proxy_weight={pw}")
                rows.append(
                    train_zero_replay_lateral_merge(
                        vocab_size, stoi, train_data, val_data, anchor, cfg,
                        variant=vname,
                        phases=phases,
                        proxy_weight=pw,
                    )
                )

        # ── True Zero-Replay: Non-Linear Amoeba ───────────────────────
        if cfg.include_nonlinear_amoeba:
            d = ww.D_MODEL
            is_big = d >= 192
            if cfg.fast:
                # Keep --fast useful as a reporting-code smoke test. The real
                # canonical run still uses the full nonlinear schedule above.
                s1, s2, s3 = 24, 48, 8
            else:
                s1 = 300 if is_big else 120
                s2 = 600 if is_big else 240
                s3 = 100 if is_big else 40

            # (name, phases, proxy, freeze_ln, rank, ppr, dual, hmatch, thmatch, task_kl)
            nl_configs = [
                # ── With proxy text (baseline proof) ──
                ("nl_proj_only",
                 [(s1, 1.0, 1.0, 0.3, 1.0), (s2, 1.0, 2.0, 0.1, 1.0), (s3, 0.0, 1.0, 0.0, 0.2)],
                 1.0, True, (d * 2) // 3, None, 0.0, 0.0, 0.0, 0.0),
                ("nl_dual10",
                 [(s1, 1.0, 1.0, 0.3, 1.0), (s2, 1.0, 2.0, 0.1, 1.0), (s3, 0.0, 1.0, 0.0, 0.2)],
                 1.0, True, (d * 2) // 3, None, 10.0, 0.0, 0.0, 0.0),
                ("nl_th10_d10",
                 [(s1, 1.0, 1.0, 0.3, 1.0), (s2, 1.0, 2.0, 0.1, 1.0), (s3, 0.0, 1.0, 0.0, 0.2)],
                 1.0, True, (d * 2) // 3, None, 10.0, 0.0, 10.0, 0.0),

                # ── NO PROXY: chase 1/1 ──
                ("nl_tkl04_th30_no_proxy",
                 [(s1, 1.0, 1.0, 0.3, 1.0), (s2, 1.0, 2.0, 0.1, 1.0), (s3, 0.0, 1.0, 0.0, 0.2)],
                 0.0, True, (d * 2) // 3, None, 0.0, 0.0, 30.0, 0.4),
                ("nl_tkl05_th30_no_proxy",
                 [(s1, 1.0, 1.0, 0.3, 1.0), (s2, 1.0, 2.0, 0.1, 1.0), (s3, 0.0, 1.0, 0.0, 0.2)],
                 0.0, True, (d * 2) // 3, None, 0.0, 0.0, 30.0, 0.5),
                ("nl_tkl07_th30_no_proxy",
                 [(s1, 1.0, 1.0, 0.3, 1.0), (s2, 1.0, 2.0, 0.1, 1.0), (s3, 0.0, 1.0, 0.0, 0.2)],
                 0.0, True, (d * 2) // 3, None, 0.0, 0.0, 30.0, 0.7),
                ("nl_tkl10_th30_no_proxy",
                 [(s1, 1.0, 1.0, 0.3, 1.0), (s2, 1.0, 2.0, 0.1, 1.0), (s3, 0.0, 1.0, 0.0, 0.2)],
                 0.0, True, (d * 2) // 3, None, 0.0, 0.0, 30.0, 1.0),
                ("nl_tkl15_th30_no_proxy",
                 [(s1, 1.0, 1.0, 0.3, 1.0), (s2, 1.0, 2.0, 0.1, 1.0), (s3, 0.0, 1.0, 0.0, 0.2)],
                 0.0, True, (d * 2) // 3, None, 0.0, 0.0, 30.0, 1.5),
            ]
            for vname, phases, pw, fln, rank_val, ppr, oaw, hmw, thmw, tkl in nl_configs:
                total = sum(p[0] for p in phases)
                rank_desc = f"per_phase={ppr}" if ppr else f"rank={rank_val}"
                extras = []
                if oaw > 0: extras.append(f"dual={oaw}")
                if hmw > 0: extras.append(f"hmatch={hmw}")
                if thmw > 0: extras.append(f"thmatch={thmw}")
                if tkl > 0: extras.append(f"tkl={tkl}")
                if pw == 0: extras.append("NO_PROXY")
                extra_desc = " " + " ".join(extras) if extras else ""
                print(f"[{vname}] d={d} phases={len(phases)} total={total} {rank_desc}{extra_desc}")
                rows.append(
                    train_nonlinear_amoeba_lateral_merge(
                        vocab_size, stoi, train_data, val_data, anchor, cfg,
                        variant=vname,
                        phases=phases,
                        proxy_weight=pw,
                        freeze_ln=fln,
                        activation_rank=rank_val,
                        per_phase_rank=ppr,
                        old_anchor_weight=oaw,
                        hidden_match_weight=hmw,
                        task_hidden_match_weight=thmw,
                        task_kl_weight=tkl,
                    )
                )

        print_summary(base_text_metrics, anchor, rows, cfg)
        evidence = build_evidence_tables(base_text_metrics, anchor, rows, cfg)
        payload = {
            "valid_retention_test": True,
            "config": vars(cfg),
            "base_text_metrics": base_text_metrics,
            "anchor": asdict(anchor),
            "rows": [asdict(row) for row in rows],
            "evidence": evidence,
        }
        cfg.json_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        write_csv(cfg.csv_path, [asdict(row) for row in rows])
        write_csv(cfg.csv_path.with_name(f"{cfg.csv_path.stem}_replay_table.csv"), evidence["toy_replay_table"])
        write_csv(cfg.csv_path.with_name(f"{cfg.csv_path.stem}_synergy_table.csv"), evidence["toy_synergy_table"])
        print(f"saved: {cfg.json_path}")
        print(f"saved: {cfg.csv_path}")
        print(f"saved: {cfg.csv_path.with_name(f'{cfg.csv_path.stem}_replay_table.csv')}")
        print(f"saved: {cfg.csv_path.with_name(f'{cfg.csv_path.stem}_synergy_table.csv')}")


if __name__ == "__main__":
    main()
