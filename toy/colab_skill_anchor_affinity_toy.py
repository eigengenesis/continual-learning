#!/usr/bin/env python3
"""
Stronger toy probe for latent skill locality via anchor families.

Core question:
If a tiny base model already contains an anchor skill from one family, does a
new related skill:
    1. align more strongly with that anchor's pressure profile,
    2. show better zero-shot readiness, and
    3. learn faster / better than when started from an unrelated anchor base?

This is stronger than the generic pairwise probe because it controls for task
difficulty: each candidate skill is evaluated against a related and unrelated
anchor base.

Run:
    python colab_skill_anchor_affinity_toy.py

Fast smoke:
    CHAOS_SMOKE=1 python colab_skill_anchor_affinity_toy.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import colab_skill_affinity_toy as toy
import colab_water_weights_benchmark as ww


ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "skill_anchor_affinity_toy_results.json"
CSV_PATH = ROOT / "skill_anchor_affinity_toy_rows.csv"

ANCHOR_TASKS = {
    "order_anchor": "reverse",
    "substitution_anchor": "shift",
}

CANDIDATE_TO_RELATED_ANCHOR = {
    "rotate_left": "order_anchor",
    "shift_twice": "substitution_anchor",
}


@dataclass
class AnchorRun:
    name: str
    task_name: str
    pressure_frontier: List[str]
    pressure_profile: Dict[str, float]
    eval_answer_acc: float
    eval_problem_acc: float
    eval_seq_acc: float
    eval_loss: float
    text_loss: float
    text_acc: float


@dataclass
class CandidateOnAnchor:
    anchor_name: str
    anchor_task: str
    candidate_name: str
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


def stable_offset(label: str) -> int:
    return sum((index + 1) * ord(ch) for index, ch in enumerate(label))


def parse_args() -> argparse.Namespace:
    smoke = os.environ.get("CHAOS_SMOKE", "0") == "1"
    parser = argparse.ArgumentParser(description="Anchor-family toy skill-affinity probe")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--base-steps", type=int, default=18 if smoke else toy.DEFAULT_BASE_STEPS)
    parser.add_argument("--anchor-steps", type=int, default=28 if smoke else (140 if torch.cuda.is_available() else 100))
    parser.add_argument("--task-steps", type=int, default=28 if smoke else toy.DEFAULT_TASK_STEPS)
    parser.add_argument("--batch-size", type=int, default=4 if smoke else toy.DEFAULT_BATCH_SIZE)
    parser.add_argument("--source-len-train", type=int, default=6 if smoke else 8)
    parser.add_argument("--source-len-eval", type=int, default=7 if smoke else 10)
    parser.add_argument("--text-eval-batches", type=int, default=2 if smoke else toy.DEFAULT_TEXT_EVAL_BATCHES)
    parser.add_argument("--task-eval-batches", type=int, default=2 if smoke else toy.DEFAULT_TASK_EVAL_BATCHES)
    parser.add_argument("--probe-batches", type=int, default=2 if smoke else toy.DEFAULT_PROBE_BATCHES)
    parser.add_argument("--projector-batches", type=int, default=2 if smoke else toy.DEFAULT_PROJECTOR_BATCHES)
    parser.add_argument("--base-lr", type=float, default=ww.BASE_LR)
    parser.add_argument("--anchor-lr", type=float, default=2.5e-4)
    parser.add_argument("--adapter-lr", type=float, default=2e-3)
    parser.add_argument("--latent-strength", type=float, default=0.75)
    parser.add_argument("--anchor-text-period", type=int, default=4)
    parser.add_argument("--candidate-eval-period", type=int, default=12 if smoke else 24)
    parser.add_argument("--early-step", type=int, default=8 if smoke else 32)
    parser.add_argument("--frontier-k", type=int, default=4)
    parser.add_argument("--text-corpus-chars", type=int, default=6_000 if smoke else 80_000)
    parser.add_argument("--json-path", type=Path, default=JSON_PATH)
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    return parser.parse_args()


def train_anchor_base(
    *,
    anchor_name: str,
    anchor_task: str,
    vocab_size: int,
    stoi: Dict[str, int],
    base_state: Dict[str, torch.Tensor],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], AnchorRun]:
    model = toy.restore_base_model(vocab_size, base_state)
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

    anchor_offset = stable_offset(anchor_name)
    text_train_positions = toy.make_text_positions(len(train_data), max(cfg.anchor_steps, 1), cfg.batch_size, cfg.seed + anchor_offset + 100)
    text_ptr = 0
    for step in range(cfg.anchor_steps):
        use_text = cfg.anchor_text_period > 0 and ((step + 1) % cfg.anchor_text_period == 0)
        if use_text:
            batch = ww.text_batch_from_positions(train_data, text_train_positions[text_ptr % len(text_train_positions)])
            toy.train_step(
                model,
                optimizer,
                batch,
                loss_fn=lambda logits, b: F.cross_entropy(logits.reshape(-1, logits.size(-1)), b.y.reshape(-1)),
            )
            text_ptr += 1
        else:
            batch = toy.make_task_batch(
                stoi,
                anchor_task,
                cfg.seed + 20_000 + anchor_offset,
                step,
                batch_size=cfg.batch_size,
                source_len=cfg.source_len_train,
            )
            toy.train_step(model, optimizer, batch)

    eval_batches = toy.make_fixed_task_batches(
        stoi,
        anchor_task,
        cfg.seed + 30_000 + anchor_offset,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    task_metrics = toy.evaluate_task(model, eval_batches)
    probe_batches = toy.make_fixed_task_batches(
        stoi,
        anchor_task,
        cfg.seed + 40_000 + anchor_offset,
        num_batches=cfg.probe_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    pressure_profile = toy.normalize_profile(toy.mean_group_pressure_profile(model, probe_batches))
    pressure_frontier = toy.top_profile_keys(pressure_profile, cfg.frontier_k)

    text_eval_positions = toy.make_text_positions(
        len(val_data),
        cfg.text_eval_batches,
        min(cfg.batch_size, ww.TEXT_EVAL_BATCH),
        cfg.seed + 50_000 + anchor_offset,
    )
    text_metrics = ww.evaluate_text(model, val_data, text_eval_positions)
    projector_positions = toy.make_text_positions(
        len(train_data),
        cfg.projector_batches,
        cfg.batch_size,
        cfg.seed + 60_000 + anchor_offset,
    )
    projectors = toy.collect_text_latent_free_projectors(model, train_data, projector_positions, ww.block_keys())
    anchor_state = ww.tensor_tree_to_cpu(model.state_dict())

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return anchor_state, projectors, AnchorRun(
        name=anchor_name,
        task_name=anchor_task,
        pressure_frontier=pressure_frontier,
        pressure_profile=pressure_profile,
        eval_answer_acc=float(task_metrics["answer_acc"]),
        eval_problem_acc=float(task_metrics["problem_acc"]),
        eval_seq_acc=float(task_metrics["seq_acc"]),
        eval_loss=float(task_metrics["loss"]),
        text_loss=float(text_metrics["loss"]),
        text_acc=float(text_metrics["acc"]),
    )


def save_anchor_csv(path: Path, rows: Sequence[CandidateOnAnchor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "anchor_name",
                "anchor_task",
                "candidate_name",
                "relation",
                "pressure_cos_to_anchor",
                "pressure_corr_to_anchor",
                "pressure_jaccard_to_anchor",
                "zero_shot_answer_acc",
                "early_answer_acc",
                "best_answer_acc",
                "auc_answer_acc",
                "best_step",
                "final_answer_acc",
                "final_problem_acc",
                "final_seq_acc",
                "learn_gain",
                "text_loss_with_adapter",
                "text_acc_with_adapter",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run_candidate_on_anchor(
    *,
    anchor_name: str,
    anchor_task: str,
    candidate_name: str,
    related_anchor: str,
    vocab_size: int,
    stoi: Dict[str, int],
    anchor_state: Dict[str, torch.Tensor],
    anchor_projectors: Dict[str, torch.Tensor],
    anchor_run: AnchorRun,
    val_data: torch.Tensor,
    cfg: argparse.Namespace,
) -> CandidateOnAnchor:
    model = toy.restore_base_model(vocab_size, anchor_state)
    eval_batches = toy.make_fixed_task_batches(
        stoi,
        candidate_name,
        cfg.seed + 70_000 + stable_offset(anchor_name) + stable_offset(candidate_name),
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    pressure_probe_batches = toy.make_fixed_task_batches(
        stoi,
        candidate_name,
        cfg.seed + 80_000 + stable_offset(anchor_name) + stable_offset(candidate_name),
        num_batches=cfg.probe_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    zero_shot_metrics = toy.evaluate_task(model, eval_batches)
    candidate_pressure_profile = toy.normalize_profile(toy.mean_group_pressure_profile(model, pressure_probe_batches))
    candidate_pressure_frontier = toy.top_profile_keys(candidate_pressure_profile, cfg.frontier_k)

    model.set_adapters_enabled(True)
    model.set_latent_free_projectors(anchor_projectors, cfg.latent_strength)
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
            float(zero_shot_metrics["answer_acc"]),
            float(zero_shot_metrics["problem_acc"]),
            float(zero_shot_metrics["seq_acc"]),
        )
    ]
    for step in range(1, cfg.task_steps + 1):
        batch = toy.make_task_batch(
            stoi,
            candidate_name,
            cfg.seed + 90_000 + stable_offset(anchor_name) + stable_offset(candidate_name),
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        toy.train_step(model, optimizer, batch)
        if step % cfg.candidate_eval_period == 0 or step == cfg.task_steps or step == cfg.early_step:
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
        cfg.seed + 100_000 + stable_offset(anchor_name) + stable_offset(candidate_name),
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

    return CandidateOnAnchor(
        anchor_name=anchor_name,
        anchor_task=anchor_task,
        candidate_name=candidate_name,
        relation="related" if anchor_name == related_anchor else "control",
        pressure_cos_to_anchor=toy.cosine_similarity(candidate_pressure_profile, anchor_run.pressure_profile),
        pressure_corr_to_anchor=toy.centered_similarity(candidate_pressure_profile, anchor_run.pressure_profile),
        pressure_jaccard_to_anchor=toy.frontier_jaccard(candidate_pressure_frontier, anchor_run.pressure_frontier),
        zero_shot_answer_acc=float(zero_shot_metrics["answer_acc"]),
        early_answer_acc=float(early_answer),
        best_answer_acc=float(best_answer),
        auc_answer_acc=float(auc_answer),
        best_step=int(best_step),
        final_answer_acc=float(final_metrics["answer_acc"]),
        final_problem_acc=float(final_metrics["problem_acc"]),
        final_seq_acc=float(final_metrics["seq_acc"]),
        learn_gain=float(final_metrics["answer_acc"] - zero_shot_metrics["answer_acc"]),
        text_loss_with_adapter=float(text_metrics["loss"]),
        text_acc_with_adapter=float(text_metrics["acc"]),
    )


def main() -> None:
    cfg = parse_args()
    toy.set_seed(cfg.seed)

    joint_text = ww.EMBEDDED_FALLBACK_TEXT[: cfg.text_corpus_chars]
    stoi, _ = toy.build_joint_vocab(joint_text)
    train_data, val_data = toy.split_text_corpus(stoi, cfg.text_corpus_chars)
    vocab_size = len(stoi)

    print("=" * 78)
    print("ANCHOR-FAMILY TOY SKILL AFFINITY PROBE")
    print("=" * 78)
    print(
        f"device={ww.DEVICE} vocab={vocab_size} "
        f"base_steps={cfg.base_steps} anchor_steps={cfg.anchor_steps} task_steps={cfg.task_steps}"
    )
    print(f"anchors={ANCHOR_TASKS} candidates={CANDIDATE_TO_RELATED_ANCHOR}")

    base_state, base_text_metrics, _ = toy.pretrain_base_model(vocab_size, train_data, val_data, cfg)
    print(f"[base] text_loss={base_text_metrics['loss']:.4f} text_acc={base_text_metrics['acc']:.3f}")

    anchor_states: Dict[str, Dict[str, torch.Tensor]] = {}
    anchor_projectors: Dict[str, Dict[str, torch.Tensor]] = {}
    anchor_runs: Dict[str, AnchorRun] = {}
    for anchor_name, anchor_task in ANCHOR_TASKS.items():
        anchor_state, projectors, anchor_run = train_anchor_base(
            anchor_name=anchor_name,
            anchor_task=anchor_task,
            vocab_size=vocab_size,
            stoi=stoi,
            base_state=base_state,
            train_data=train_data,
            val_data=val_data,
            cfg=cfg,
        )
        anchor_states[anchor_name] = anchor_state
        anchor_projectors[anchor_name] = projectors
        anchor_runs[anchor_name] = anchor_run
        print(
            f"[{anchor_name}] task={anchor_task} eval_answer={anchor_run.eval_answer_acc:.3f} "
            f"text_acc={anchor_run.text_acc:.3f} frontier={'+'.join(anchor_run.pressure_frontier)}"
        )

    rows: List[CandidateOnAnchor] = []
    by_candidate: Dict[str, List[CandidateOnAnchor]] = {name: [] for name in CANDIDATE_TO_RELATED_ANCHOR}
    for candidate_name, related_anchor in CANDIDATE_TO_RELATED_ANCHOR.items():
        for anchor_name, anchor_task in ANCHOR_TASKS.items():
            row = run_candidate_on_anchor(
                anchor_name=anchor_name,
                anchor_task=anchor_task,
                candidate_name=candidate_name,
                related_anchor=related_anchor,
                vocab_size=vocab_size,
                stoi=stoi,
                anchor_state=anchor_states[anchor_name],
                anchor_projectors=anchor_projectors[anchor_name],
                anchor_run=anchor_runs[anchor_name],
                val_data=val_data,
                cfg=cfg,
            )
            rows.append(row)
            by_candidate[candidate_name].append(row)
            print(
                f"[{candidate_name} on {anchor_name}] relation={row.relation:<7} "
                f"pressure_corr={row.pressure_corr_to_anchor:.3f} "
                f"zero={row.zero_shot_answer_acc:.3f} early={row.early_answer_acc:.3f} "
                f"best={row.best_answer_acc:.3f}@{row.best_step} final={row.final_answer_acc:.3f} "
                f"gain={row.learn_gain:+.3f}"
            )

    pressure_gaps: List[float] = []
    pressure_corr_gaps: List[float] = []
    zero_gaps: List[float] = []
    early_gaps: List[float] = []
    best_gaps: List[float] = []
    auc_gaps: List[float] = []
    final_gaps: List[float] = []
    gain_gaps: List[float] = []
    print("-" * 78)
    for candidate_name, candidate_rows in by_candidate.items():
        related_row = next(row for row in candidate_rows if row.relation == "related")
        control_row = next(row for row in candidate_rows if row.relation == "control")
        pressure_gap = related_row.pressure_cos_to_anchor - control_row.pressure_cos_to_anchor
        pressure_corr_gap = related_row.pressure_corr_to_anchor - control_row.pressure_corr_to_anchor
        zero_gap = related_row.zero_shot_answer_acc - control_row.zero_shot_answer_acc
        early_gap = related_row.early_answer_acc - control_row.early_answer_acc
        best_gap = related_row.best_answer_acc - control_row.best_answer_acc
        auc_gap = related_row.auc_answer_acc - control_row.auc_answer_acc
        final_gap = related_row.final_answer_acc - control_row.final_answer_acc
        gain_gap = related_row.learn_gain - control_row.learn_gain
        pressure_gaps.append(pressure_gap)
        pressure_corr_gaps.append(pressure_corr_gap)
        zero_gaps.append(zero_gap)
        early_gaps.append(early_gap)
        best_gaps.append(best_gap)
        auc_gaps.append(auc_gap)
        final_gaps.append(final_gap)
        gain_gaps.append(gain_gap)
        print(
            f"{candidate_name:>12} related-minus-control "
            f"pressure={pressure_gap:+.3f} corr={pressure_corr_gap:+.3f} "
            f"zero={zero_gap:+.3f} early={early_gap:+.3f} best={best_gap:+.3f} "
            f"auc={auc_gap:+.3f} final={final_gap:+.3f} gain={gain_gap:+.3f}"
        )

    mean_pressure_gap = float(np.mean(pressure_gaps)) if pressure_gaps else float("nan")
    mean_pressure_corr_gap = float(np.mean(pressure_corr_gaps)) if pressure_corr_gaps else float("nan")
    mean_zero_gap = float(np.mean(zero_gaps)) if zero_gaps else float("nan")
    mean_early_gap = float(np.mean(early_gaps)) if early_gaps else float("nan")
    mean_best_gap = float(np.mean(best_gaps)) if best_gaps else float("nan")
    mean_auc_gap = float(np.mean(auc_gaps)) if auc_gaps else float("nan")
    mean_final_gap = float(np.mean(final_gaps)) if final_gaps else float("nan")
    mean_gain_gap = float(np.mean(gain_gaps)) if gain_gaps else float("nan")
    signal_margin = 0.02
    print("-" * 78)
    print(
        f"mean related-minus-control pressure={mean_pressure_gap:+.3f} "
        f"corr={mean_pressure_corr_gap:+.3f} zero={mean_zero_gap:+.3f} "
        f"early={mean_early_gap:+.3f} best={mean_best_gap:+.3f} auc={mean_auc_gap:+.3f} "
        f"final={mean_final_gap:+.3f} gain={mean_gain_gap:+.3f}"
    )
    if (
        math.isfinite(mean_final_gap)
        and math.isfinite(mean_pressure_corr_gap)
        and math.isfinite(mean_zero_gap)
    ):
        if mean_final_gap > signal_margin and (mean_pressure_corr_gap > 0.0 or mean_zero_gap > 0.0):
            print("signal: related anchor bases help candidate skills more than unrelated anchors.")
        elif (mean_zero_gap > signal_margin or mean_early_gap > signal_margin or mean_best_gap > signal_margin or mean_auc_gap > signal_margin) and mean_final_gap <= 0.0:
            print("trajectory signal: related anchors help early readiness or sample efficiency, but later unconstrained adapter training washes that advantage out.")
        elif mean_final_gap > 0.0 or mean_pressure_corr_gap > 0.0 or mean_zero_gap > 0.0 or mean_early_gap > 0.0 or mean_best_gap > 0.0 or mean_auc_gap > 0.0 or mean_gain_gap > 0.0:
            print("mixed signal: some family-locality evidence exists, but it is not fully clean yet.")
        else:
            print("no positive anchor-locality gap yet under this toy setup.")

    cfg.json_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.json_path.write_text(
        json.dumps(
            {
                "config": vars(cfg),
                "base_text_metrics": base_text_metrics,
                "anchor_runs": {name: asdict(run) for name, run in anchor_runs.items()},
                "rows": [asdict(row) for row in rows],
                "summary": {
                    "mean_pressure_gap_related_minus_control": mean_pressure_gap,
                    "mean_pressure_corr_gap_related_minus_control": mean_pressure_corr_gap,
                    "mean_zero_gap_related_minus_control": mean_zero_gap,
                    "mean_early_gap_related_minus_control": mean_early_gap,
                    "mean_best_gap_related_minus_control": mean_best_gap,
                    "mean_auc_gap_related_minus_control": mean_auc_gap,
                    "mean_final_gap_related_minus_control": mean_final_gap,
                    "mean_gain_gap_related_minus_control": mean_gain_gap,
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    save_anchor_csv(cfg.csv_path, rows)
    print(f"saved: {cfg.json_path}")
    print(f"saved: {cfg.csv_path}")


if __name__ == "__main__":
    main()
