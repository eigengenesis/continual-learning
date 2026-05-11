#!/usr/bin/env python3
"""Predictive/causal sanity test for Z tomography.

This script tests Z as an independent claim, not as part of the full Amoeba
continual-learning pipeline.

Core falsification:
  1. Compute Z tomography on a task before adapter training.
  2. Train matched LoRA teachers on top-Z, random, and bottom-Z layer sets.
  3. If Z is real, top-Z should learn faster/better than random/bottom-Z.

Optional negative control:
  Train top-Z on shuffled prompt/target pairs and evaluate on the real task. A
  strong Z story should not turn label noise into a real generated skill.

Artifacts:
  outputs/z_tomography_falsification_seed{seed}/
    run_config.json
    task_manifest.json
    z_branch_manifest.json
    z_falsification_summary.csv
    z_falsification_verdict.json
    curves.csv
    stage_summary.csv
    z_tomography.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

import qwen_continual_proof as qp
from alien_ladder_cl_audit import (
    AlienExample,
    ArtifactLogger,
    TaskData,
    build_task_specs,
    evaluate_task,
    focus_metrics,
    load_hf_task,
    load_proof_v2_record_task,
    load_synthetic_listops_task,
    make_runtime_config,
    print_metrics,
    release,
    select_layers_generic,
    train_adapter_teacher,
)
from standalone_latent_lora_qwen import choose_dtype, default_model_id, load_causal_lm, load_tokenizer


def line(char: str = "=") -> None:
    print(char * 96, flush=True)


def section(title: str) -> None:
    line("=")
    print(title, flush=True)
    line("=")


def subsection(title: str) -> None:
    line("-")
    print(title, flush=True)
    line("-")


def fmt(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def build_loader_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        scan_config=args.scan_config,
        cogs_config=args.cogs_config,
        geoquery_config=args.geoquery_config,
        scan_max_target_tokens=args.scan_max_target_tokens,
        cogs_max_target_tokens=args.cogs_max_target_tokens,
        geoquery_max_target_tokens=args.geoquery_max_target_tokens,
        task_train_samples=args.task_train_samples,
        task_eval_samples=args.task_eval_samples,
        seed=args.seed,
    )


def load_task(args: argparse.Namespace, tokenizer) -> TaskData:
    if args.task == "listops":
        return load_synthetic_listops_task(args.seed + 101, args.task_train_samples, args.task_eval_samples)
    if args.task == "proof_v2_record":
        return load_proof_v2_record_task(tokenizer, args.seed + 102, args.task_train_samples, args.task_eval_samples)

    loader_args = build_loader_args(args)
    scan_spec, cogs_spec, geo_spec = build_task_specs(loader_args)
    if args.task == "scan":
        return load_hf_task(scan_spec, tokenizer, args.seed + 103)
    if args.task == "cogs":
        return load_hf_task(cogs_spec, tokenizer, args.seed + 104)
    if args.task == "geoquery":
        return load_hf_task(geo_spec, tokenizer, args.seed + 105)
    raise ValueError(f"unknown task {args.task!r}")


def shuffled_label_task(task: TaskData, seed: int) -> TaskData:
    rng = np.random.default_rng(seed)
    targets = [item.target for item in task.train]
    raw_targets = [item.raw_target for item in task.train]
    perm = np.arange(len(targets))
    rng.shuffle(perm)
    shuffled_train: List[AlienExample] = []
    for idx, item in enumerate(task.train):
        target_idx = int(perm[idx])
        shuffled_train.append(
            AlienExample(
                prompt=item.prompt,
                target=targets[target_idx],
                source=item.source,
                raw_target=raw_targets[target_idx],
            )
        )
    spec = replace(
        task.spec,
        name=f"{task.spec.name}_shuffled",
        display_name=f"{task.spec.display_name} shuffled-label control",
    )
    return TaskData(
        spec=spec,
        train=shuffled_train,
        eval=task.eval,
        manifest={**task.manifest, "negative_control": "train_targets_shuffled_eval_real"},
    )


def random_target_task(task: TaskData, seed: int) -> TaskData:
    rng = np.random.default_rng(seed)
    vocab = [f"ZZ{idx:02d}" for idx in range(16)]
    random_train: List[AlienExample] = []
    for item in task.train:
        length = int(rng.integers(4, 10))
        target = " ".join(str(vocab[int(rng.integers(0, len(vocab)))]) for _ in range(length))
        random_train.append(
            AlienExample(
                prompt=item.prompt,
                target=target,
                source=item.source,
                raw_target=target,
            )
        )
    spec = replace(
        task.spec,
        name=f"{task.spec.name}_random_targets",
        display_name=f"{task.spec.display_name} random-target control",
        max_target_tokens=max(task.spec.max_target_tokens, 32),
    )
    return TaskData(
        spec=spec,
        train=random_train,
        eval=task.eval,
        manifest={**task.manifest, "negative_control": "train_random_targets_eval_real"},
    )


def pressure_by_layer(tomography: Any) -> Dict[int, float]:
    return {
        int(item.layer_index): float(item.learning_pressure)
        for item in getattr(tomography, "layer_saturations", [])
    }


def sorted_layers_by_pressure(tomography: Any) -> List[int]:
    return [
        int(item.layer_index)
        for item in sorted(
            getattr(tomography, "layer_saturations", []),
            key=lambda item: float(item.learning_pressure),
            reverse=True,
        )
    ]


def branch_layer_sets(
    *,
    all_layers: Sequence[int],
    n_layers: int,
    random_trials: int,
    seed: int,
    random_exclude_layers: Sequence[int] = (),
) -> List[Tuple[str, List[int]]]:
    ordered = list(int(v) for v in all_layers)
    if not ordered:
        raise RuntimeError("tomography returned no layers")
    n = min(int(n_layers), len(ordered))
    branches: List[Tuple[str, List[int]]] = [
        ("top_z", ordered[:n]),
        ("bottom_z", list(reversed(ordered[-n:]))),
    ]
    rng = np.random.default_rng(seed)
    excluded = set(int(v) for v in random_exclude_layers)
    random_pool = [layer for layer in ordered if layer not in excluded]
    if len(random_pool) < n:
        random_pool = [layer for layer in ordered if layer not in set(ordered[:n])]
    if len(random_pool) < n:
        random_pool = ordered
    for trial in range(int(random_trials)):
        sample = sorted(int(v) for v in rng.choice(random_pool, size=n, replace=False).tolist())
        branches.append((f"random_{trial + 1}", sample))
    return branches


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_delta(row: Dict[str, Any], task_name: str, key: str, base: Dict[str, float]) -> float:
    metric = f"{task_name}_{key}"
    return float(row.get(metric, float("nan"))) - float(base.get(metric, float("nan")))


def train_branch(
    *,
    base_model,
    tokenizer,
    task: TaskData,
    eval_task: TaskData,
    cfg: qp.RuntimeConfig,
    logger: ArtifactLogger,
    branch_name: str,
    layers: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    subsection(f"Branch: {branch_name} layers={list(layers)}")
    model = qp._clone_model(base_model, cfg.device)
    started = time.time()
    metrics = train_adapter_teacher(
        model=model,
        tokenizer=tokenizer,
        task=task,
        active_eval_tasks=[eval_task],
        cfg=cfg,
        logger=logger,
        stage=f"ztest_{branch_name}",
        selected_layers=layers,
        steps=args.steps,
        lr=args.teacher_lr,
        rank=args.rank,
        alpha=args.alpha,
        gate_init=args.gate_init,
        eval_interval=args.eval_interval,
    )
    row = {
        "branch": branch_name,
        "layers": ",".join(str(v) for v in layers),
        "layer_count": len(layers),
        "wall_time_sec": time.time() - started,
        **metrics,
    }
    release(model)
    return row


def verdict(summary_rows: Sequence[Dict[str, Any]], task_name: str, base_metrics: Dict[str, float]) -> Dict[str, Any]:
    by_branch = {str(row["branch"]): row for row in summary_rows}
    top = by_branch.get("top_z")
    bottom = by_branch.get("bottom_z")
    randoms = [row for name, row in by_branch.items() if name.startswith("random_")]
    if top is None or bottom is None:
        return {"passed": False, "reason": "missing top_z or bottom_z branch"}

    token_key = f"{task_name}_token_acc"
    exact_key = f"{task_name}_exact"
    tf_loss_key = f"{task_name}_tf_loss"

    top_tok = float(top.get(token_key, float("nan")))
    bottom_tok = float(bottom.get(token_key, float("nan")))
    random_tok_values = [float(row.get(token_key, float("nan"))) for row in randoms]
    random_tok_mean = float(np.nanmean(random_tok_values)) if random_tok_values else float("nan")
    random_tok_max = float(np.nanmax(random_tok_values)) if random_tok_values else float("nan")

    top_loss = float(top.get(tf_loss_key, float("nan")))
    bottom_loss = float(bottom.get(tf_loss_key, float("nan")))
    random_loss_values = [float(row.get(tf_loss_key, float("nan"))) for row in randoms]
    random_loss_mean = float(np.nanmean(random_loss_values)) if random_loss_values else float("nan")
    random_loss_min = float(np.nanmin(random_loss_values)) if random_loss_values else float("nan")

    top_exact = float(top.get(exact_key, float("nan")))
    bottom_exact = float(bottom.get(exact_key, float("nan")))
    random_exact_values = [float(row.get(exact_key, float("nan"))) for row in randoms]
    random_exact_mean = float(np.nanmean(random_exact_values)) if random_exact_values else float("nan")
    random_exact_max = float(np.nanmax(random_exact_values)) if random_exact_values else float("nan")
    base_tok = float(base_metrics.get(token_key, float("nan")))
    base_loss = float(base_metrics.get(tf_loss_key, float("nan")))

    top_beats_bottom_generated = bool((top_tok >= bottom_tok + 1e-9) or (top_exact >= bottom_exact + 1e-9))
    top_beats_random_generated = bool(
        not randoms
        or (top_tok >= random_tok_mean + 1e-9)
        or (top_exact >= random_exact_mean + 1e-9)
    )
    top_beats_bottom_loss = bool(top_loss <= bottom_loss - 1e-9)
    top_beats_random_loss = bool(
        not randoms or top_loss <= random_loss_mean - 1e-9
    )
    learns_vs_base = bool(
        (math.isfinite(base_tok) and top_tok > base_tok + 0.02)
        or (math.isfinite(base_loss) and top_loss < base_loss * 0.90)
        or top_exact > 0.0
    )
    generated_pass = bool(top_beats_bottom_generated and top_beats_random_generated and learns_vs_base)
    loss_only_signal = bool(top_beats_bottom_loss and top_beats_random_loss and learns_vs_base)
    passed = bool(generated_pass)
    status = "PASS" if generated_pass else ("PARTIAL" if loss_only_signal else "FAIL")
    return {
        "passed": passed,
        "status": status,
        "generated_pass": generated_pass,
        "loss_only_signal": loss_only_signal,
        "top_beats_bottom_generated": top_beats_bottom_generated,
        "top_beats_random_generated": top_beats_random_generated,
        "top_beats_bottom_loss": top_beats_bottom_loss,
        "top_beats_random_loss": top_beats_random_loss,
        "top_learns_vs_base": learns_vs_base,
        "top_token_acc": top_tok,
        "bottom_token_acc": bottom_tok,
        "random_token_acc_mean": random_tok_mean,
        "random_token_acc_max": random_tok_max,
        "top_exact": top_exact,
        "bottom_exact": bottom_exact,
        "random_exact_mean": random_exact_mean,
        "random_exact_max": random_exact_max,
        "top_tf_loss": top_loss,
        "bottom_tf_loss": bottom_loss,
        "random_tf_loss_mean": random_loss_mean,
        "random_tf_loss_min": random_loss_min,
        "base_token_acc": base_tok,
        "base_tf_loss": base_loss,
        "interpretation": (
            "PASS: top-Z beat matched controls on generated acquisition."
            if generated_pass
            else (
                "PARTIAL: top-Z showed a teacher-forced loss edge, but did not beat matched controls on generated acquisition."
                if loss_only_signal
                else "FAIL: top-Z did not beat matched controls in this run."
            )
        ),
    }


def run() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_id = args.model_id or default_model_id(args.local_files_only)
    out_dir = Path(args.output_dir).expanduser() / f"z_tomography_falsification_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = ArtifactLogger(out_dir)
    logger.write_json("run_config.json", vars(args) | {"model_id_resolved": model_id})

    section("Z TOMOGRAPHY FALSIFICATION AUDIT")
    print(f"model_id={model_id}", flush=True)
    print(f"device={args.device} dtype={args.dtype} seed={args.seed}", flush=True)
    print(f"task={args.task} branches=top_z | bottom_z | random_x{args.random_trials}", flush=True)
    print("stdout is a convenience artifact; CSV/JSON are the paper artifacts.", flush=True)

    tokenizer = load_tokenizer(model_id, local_files_only=args.local_files_only)
    model = load_causal_lm(
        model_id,
        device=args.device,
        dtype=choose_dtype(args.dtype),
        local_files_only=args.local_files_only,
    )
    cfg = make_runtime_config(args, out_dir)

    task = load_task(args, tokenizer)
    logger.write_json("task_manifest.json", task.manifest)
    print(
        f"loaded_task={task.spec.name} train={len(task.train)} eval={len(task.eval)} "
        f"max_target_tokens={task.spec.max_target_tokens}",
        flush=True,
    )

    subsection("Base evaluation")
    base_metrics = evaluate_task(model, tokenizer, task, cfg, do_generation=True)
    print_metrics("base", base_metrics, [task.spec.name])
    logger.log_stage_summary("base", base_metrics)

    subsection("Pre-training Z tomography")
    layers, tomography = select_layers_generic(
        model=model,
        tokenizer=tokenizer,
        task=task,
        protected_profiles=[],
        cfg=cfg,
        min_layers=args.num_layers,
        stage="z_falsification",
        logger=logger,
    )
    del layers
    ordered_layers = sorted_layers_by_pressure(tomography)
    pressures = pressure_by_layer(tomography)
    branches = branch_layer_sets(
        all_layers=ordered_layers,
        n_layers=args.num_layers,
        random_trials=args.random_trials,
        seed=args.seed + 777,
        random_exclude_layers=ordered_layers[: args.num_layers] if args.random_excludes_top_z else (),
    )
    top_set = set(ordered_layers[: min(args.num_layers, len(ordered_layers))])
    bottom_set = set(ordered_layers[-min(args.num_layers, len(ordered_layers)):])
    branch_manifest = [
        {
            "branch": name,
            "layers": layers,
            "top_z_overlap": len(set(layers) & top_set),
            "bottom_z_overlap": len(set(layers) & bottom_set),
            "mean_learning_pressure": float(np.mean([pressures.get(int(v), 0.0) for v in layers])),
            "sum_learning_pressure": float(np.sum([pressures.get(int(v), 0.0) for v in layers])),
        }
        for name, layers in branches
    ]
    logger.write_json("z_branch_manifest.json", {"branches": branch_manifest})
    for item in branch_manifest:
        print(
            f"[branch_layers] {item['branch']:10s} layers={item['layers']} "
            f"top_overlap={item['top_z_overlap']} bottom_overlap={item['bottom_z_overlap']} "
            f"mean_pressure={fmt(item['mean_learning_pressure'])}",
            flush=True,
        )

    section("MATCHED LORA TRAINING")
    summary_rows: List[Dict[str, Any]] = []
    for name, layer_set in branches:
        row = train_branch(
            base_model=model,
            tokenizer=tokenizer,
            task=task,
            eval_task=task,
            cfg=cfg,
            logger=logger,
            branch_name=name,
            layers=layer_set,
            args=args,
        )
        row["base_token_delta"] = metric_delta(row, task.spec.name, "token_acc", base_metrics)
        row["base_exact_delta"] = metric_delta(row, task.spec.name, "exact", base_metrics)
        summary_rows.append(row)

    if args.negative_control != "none":
        title = "SHUFFLED-LABEL NEGATIVE CONTROL" if args.negative_control == "shuffled" else "RANDOM-TARGET NEGATIVE CONTROL"
        section(title)
        control_task = (
            shuffled_label_task(task, args.seed + 909)
            if args.negative_control == "shuffled"
            else random_target_task(task, args.seed + 910)
        )
        shuffled_layers, shuffled_tomography = select_layers_generic(
            model=model,
            tokenizer=tokenizer,
            task=control_task,
            protected_profiles=[],
            cfg=cfg,
            min_layers=args.num_layers,
            stage=f"z_falsification_{args.negative_control}",
            logger=logger,
        )
        del shuffled_layers
        shuffled_ordered = sorted_layers_by_pressure(shuffled_tomography)
        shuffled_top = shuffled_ordered[: min(args.num_layers, len(shuffled_ordered))]
        row = train_branch(
            base_model=model,
            tokenizer=tokenizer,
            task=control_task,
            eval_task=task,
            cfg=cfg,
            logger=logger,
            branch_name=f"top_z_{args.negative_control}_control",
            layers=shuffled_top,
            args=args,
        )
        row["base_token_delta"] = metric_delta(row, task.spec.name, "token_acc", base_metrics)
        row["base_exact_delta"] = metric_delta(row, task.spec.name, "exact", base_metrics)
        row["negative_control"] = control_task.manifest.get("negative_control", args.negative_control)
        summary_rows.append(row)

    summary_path = out_dir / "z_falsification_summary.csv"
    write_rows(summary_path, summary_rows)

    verdict_payload = verdict(summary_rows, task.spec.name, base_metrics)
    with (out_dir / "z_falsification_verdict.json").open("w", encoding="utf-8") as handle:
        json.dump(verdict_payload, handle, indent=2, sort_keys=True, default=str)

    section("Z TOMOGRAPHY VERDICT")
    for row in summary_rows:
        print_metrics(str(row["branch"]), row, [task.spec.name])
    print(json.dumps(verdict_payload, indent=2, sort_keys=True), flush=True)
    print(f"artifacts={out_dir}", flush=True)

    release(model)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predictive falsification test for Z tomography")
    parser.add_argument("--model-id", default="", help="HF model id; default is Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="retained for RuntimeConfig compatibility")

    parser.add_argument("--task", choices=("listops", "proof_v2_record", "scan", "cogs", "geoquery"), default="listops")
    parser.add_argument("--task-train-samples", type=int, default=256)
    parser.add_argument("--task-eval-samples", type=int, default=64)
    parser.add_argument("--scan-config", default="simple")
    parser.add_argument("--cogs-config", default=None)
    parser.add_argument("--geoquery-config", default=None)
    parser.add_argument("--scan-max-target-tokens", type=int, default=64)
    parser.add_argument("--cogs-max-target-tokens", type=int, default=48)
    parser.add_argument("--geoquery-max-target-tokens", type=int, default=80)

    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--random-trials", type=int, default=2)
    parser.add_argument("--random-excludes-top-z", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--negative-control",
        choices=("none", "shuffled", "random_targets"),
        default="random_targets",
        help="Noise control evaluated on the real task. random_targets is stricter than shuffled for structured targets.",
    )

    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--teacher-lr", type=float, default=8e-5)
    parser.add_argument("--gate-init", type=float, default=-1.5)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--wikitext-eval-samples", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=50)

    # Compatibility fields consumed by alien_ladder_cl_audit.make_runtime_config.
    parser.add_argument("--checkpoint-policy", choices=("none", "state", "pretrained"), default="none")
    parser.add_argument("--max-ppl-ratio", type=float, default=1.12)
    return parser


if __name__ == "__main__":
    run()
