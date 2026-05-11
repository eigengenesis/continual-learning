#!/usr/bin/env python3
"""Z tomography occupancy/protection audit.

This is the corrected Z test.

The predictive falsification script asks whether top-Z layers learn a new
adapter fastest. That is not the claim used by the Amoeba/Water Weights
pipeline. The pipeline claim is:

  High old-task Z/pressure marks occupied old-skill directions. Protecting
  those directions during a later task should retain the old skill better than
  protecting low-Z or random matched layers.

Pipeline:
  base
  -> train old-skill teacher
  -> consolidate old skill into base weights
  -> compute old-skill Z tomography + occupied bases
  -> train new-skill teacher from old-skilled base
  -> consolidate new skill with branch-specific old protection:
       no_anchor, global_kl_only, protect_top_z, protect_bottom_z, protect_random

No branch uses old-task examples during the new-skill consolidation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

import qwen_continual_proof as qp
from alien_ladder_cl_audit import (
    AlienExample,
    ArtifactLogger,
    TaskData,
    build_task_specs,
    collect_profile,
    consolidate_no_proxy,
    evaluate_suite,
    hidden_alignment_to_student_device,
    kl_divergence_to_student_device,
    load_hf_task,
    load_proof_v2_record_task,
    load_synthetic_listops_task,
    make_runtime_config,
    move_batch,
    print_metrics,
    release,
    require_generated_acquisition,
    select_layers_generic,
    stable_seed,
    train_adapter_teacher,
)
from qwen_cl_desiderata_audit import project_old_occupied_gradients
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
        writer.writerows(rows)


def make_old_conflict_task(old_task: TaskData, args: argparse.Namespace) -> TaskData:
    """Adversarial overwrite task.

    Reuses the exact old-task prompts but trains a contradictory constant answer.
    This is not a real downstream skill; it is a stress-test that creates
    controlled destructive interference so protection mechanisms can be tested.
    """
    target = str(args.conflict_target)
    train = [
        AlienExample(prompt=item.prompt, target=target, source=item.source, raw_target=target)
        for item in old_task.train
    ]
    eval_examples = [
        AlienExample(prompt=item.prompt, target=target, source=item.source, raw_target=target)
        for item in old_task.eval
    ]
    spec = replace(
        old_task.spec,
        name="old_conflict",
        display_name=f"{old_task.spec.display_name} conflict target",
        max_target_tokens=max(old_task.spec.max_target_tokens, 8),
        max_new_tokens=max(8, min(old_task.spec.max_new_tokens, 16)),
        token_gate=0.75,
        exact_gate=0.50,
        citation="Synthetic adversarial conflict task generated from the old task prompts.",
    )
    return TaskData(
        spec=spec,
        train=train,
        eval=eval_examples,
        manifest={
            **old_task.manifest,
            "name": "old_conflict",
            "dataset_id": "synthetic:old_prompt_constant_conflict",
            "old_task": old_task.spec.name,
            "conflict_target": target,
            "purpose": "controlled forgetting stress-test; not a real benchmark skill",
        },
    )


def load_named_task(name: str, args: argparse.Namespace, tokenizer, seed: int) -> TaskData:
    if name == "listops":
        return load_synthetic_listops_task(seed, args.task_train_samples, args.task_eval_samples)
    if name == "proof_v2_record":
        return load_proof_v2_record_task(tokenizer, seed, args.task_train_samples, args.task_eval_samples)
    loader_args = argparse.Namespace(
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
    scan_spec, cogs_spec, geo_spec = build_task_specs(loader_args)
    if name == "scan":
        return load_hf_task(scan_spec, tokenizer, seed)
    if name == "cogs":
        return load_hf_task(cogs_spec, tokenizer, seed)
    if name == "geoquery":
        return load_hf_task(geo_spec, tokenizer, seed)
    raise ValueError(f"unknown task: {name}")


def random_layer_set(
    ordered_layers: Sequence[int],
    *,
    n_layers: int,
    seed: int,
    exclude: Sequence[int],
) -> List[int]:
    rng = np.random.default_rng(seed)
    excluded = set(int(v) for v in exclude)
    pool = [int(v) for v in ordered_layers if int(v) not in excluded]
    if len(pool) < n_layers:
        pool = list(int(v) for v in ordered_layers)
    return sorted(int(v) for v in rng.choice(pool, size=min(n_layers, len(pool)), replace=False).tolist())


def branch_layer_sets(tomography: Any, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[int], List[int], List[int]]:
    ordered = sorted_layers_by_pressure(tomography)
    if not ordered:
        raise RuntimeError("old-skill tomography returned no layers")
    n = min(int(args.num_layers), len(ordered))
    top = list(ordered[:n])
    bottom = list(reversed(ordered[-n:]))
    random_layers = random_layer_set(ordered, n_layers=n, seed=args.seed + 8800, exclude=top + bottom)
    pressures = pressure_by_layer(tomography)
    rows = []
    for name, layers in (
        ("top_z", top),
        ("bottom_z", bottom),
        ("random", random_layers),
    ):
        rows.append(
            {
                "branch": name,
                "layers": ",".join(str(v) for v in layers),
                "mean_learning_pressure": float(np.mean([pressures.get(int(v), 0.0) for v in layers])),
                "sum_learning_pressure": float(np.sum([pressures.get(int(v), 0.0) for v in layers])),
            }
        )
    return rows, top, bottom, random_layers


def train_new_with_branch_protection(
    *,
    student,
    teacher_old,
    teacher_new,
    tokenizer,
    new_task: TaskData,
    active_eval_tasks: Sequence[TaskData],
    new_alignment_layers: Sequence[int],
    protection_layers: Sequence[int],
    old_profiles: Sequence[Any],
    cfg: qp.RuntimeConfig,
    logger: ArtifactLogger,
    stage: str,
    steps: int,
    lr: float,
    old_kl_weight: float,
    old_hidden_weight: float,
    new_kl_weight: float,
    new_hidden_weight: float,
    project_gradients: bool,
    projection_strength: float,
    teacher_device: str,
) -> Dict[str, float]:
    qp._freeze_model(teacher_old)
    qp._freeze_model(teacher_new)
    teacher_old.to(teacher_device)
    teacher_new.to(teacher_device)
    qp._unfreeze_model(student)
    if cfg.device.startswith("cuda"):
        qp._configure_gradient_checkpointing(student, cfg.gradient_checkpointing)
    params = qp._trainable_params(student)
    if not params:
        raise RuntimeError(f"{stage}: student has no trainable params")
    optimizer = torch.optim.AdamW(params, lr=float(lr))
    batch_fn = new_task.make_batch(tokenizer, cfg.device, cfg.batch_size, cfg.max_seq_len, cfg.seed + stable_seed(stage, 9100))
    start = time.time()
    student.train()
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = batch_fn(step)
        split = qp._split_tensor_batch(batch, cfg.consolidation_micro_batch_size)
        loss_value = 0.0
        for micro in split:
            student_outputs = student(**micro, output_hidden_states=True, use_cache=False)
            teacher_micro = move_batch(micro, teacher_device)
            with torch.no_grad():
                old_outputs = teacher_old(**teacher_micro, output_hidden_states=True, use_cache=False)
                new_outputs = teacher_new(**teacher_micro, output_hidden_states=True, use_cache=False)
            new_kl = kl_divergence_to_student_device(student_outputs.logits, new_outputs.logits)
            old_kl = kl_divergence_to_student_device(student_outputs.logits, old_outputs.logits)
            new_hidden = hidden_alignment_to_student_device(
                student_outputs,
                new_outputs,
                list(new_alignment_layers),
                micro["input_ids"].device,
            )
            old_hidden = hidden_alignment_to_student_device(
                student_outputs,
                old_outputs,
                list(protection_layers),
                micro["input_ids"].device,
            )
            loss = (
                student_outputs.loss
                + float(new_kl_weight) * new_kl
                + float(new_hidden_weight) * new_hidden
                + float(old_kl_weight) * old_kl
                + float(old_hidden_weight) * old_hidden
            ) / max(1, len(split))
            loss.backward()
            loss_value += float(loss.item())
        projected_modules = 0
        if project_gradients and protection_layers:
            projected_modules = project_old_occupied_gradients(
                student,
                list(old_profiles),
                list(protection_layers),
                strength=float(projection_strength),
            )
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        if step % cfg.log_interval == 0 or step == steps:
            print(
                f"[{stage}] step={step:04d}/{steps} loss={loss_value:.4f} "
                f"old_task_examples=0 proxy_batches=0 old_kl={old_kl_weight:.3f} "
                f"old_hidden={old_hidden_weight:.3f} protection_layers={list(protection_layers)} "
                f"projected_modules={projected_modules}",
                flush=True,
            )
            logger.log_curve(
                stage,
                step,
                loss=loss_value,
                old_task_examples=0,
                proxy_batches=0,
                projected_modules=projected_modules,
            )
    metrics = evaluate_suite(student, tokenizer, active_eval_tasks, cfg, do_generation=True, include_wikitext=True)
    print_metrics(stage, metrics, [task.spec.name for task in active_eval_tasks])
    logger.log_stage_summary(
        stage,
        metrics,
        wall_time_sec=time.time() - start,
        old_task_examples=0,
        proxy_batches=0,
        protection_layers=",".join(str(v) for v in protection_layers),
    )
    print(f"[{stage}] wall_time_sec={time.time() - start:.1f}", flush=True)
    return metrics


def verdict(
    rows: Sequence[Dict[str, Any]],
    old_task: TaskData,
    new_task: TaskData,
    base_old_metrics: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    by = {str(row["branch"]): row for row in rows}
    old_key = f"{old_task.spec.name}_token_acc"
    old_exact_key = f"{old_task.spec.name}_exact"
    new_key = f"{new_task.spec.name}_token_acc"
    top = by.get("protect_top_z", {})
    bottom = by.get("protect_bottom_z", {})
    random = by.get("protect_random", {})
    no_anchor = by.get("no_anchor", {})
    global_kl = by.get("global_kl_only", {})
    top_old = float(top.get(old_key, 0.0))
    bottom_old = float(bottom.get(old_key, 0.0))
    random_old = float(random.get(old_key, 0.0))
    no_anchor_old = float(no_anchor.get(old_key, 0.0))
    global_old = float(global_kl.get(old_key, 0.0))
    top_new = float(top.get(new_key, 0.0))
    no_anchor_new = float(no_anchor.get(new_key, 0.0))
    global_new = float(global_kl.get(new_key, 0.0))
    bottom_new = float(bottom.get(new_key, 0.0))
    random_new = float(random.get(new_key, 0.0))
    base_old = float(base_old_metrics.get(old_key, 0.0))
    forgetting_drop = base_old - no_anchor_old
    forgetting_present = bool(forgetting_drop >= float(args.min_forgetting_drop))
    best_control_old = max(bottom_old, random_old, global_old, no_anchor_old)
    layer_control_best = max(bottom_old, random_old)
    gain_margin = max(float(args.protection_margin), float(args.min_protection_gain))
    top_beats_layer_controls = bool(top_old >= layer_control_best + gain_margin)
    top_beats_all_controls = bool(top_old >= best_control_old + gain_margin)
    preserves_nonzero_old = bool(top_old >= float(args.min_protected_old_token_acc))
    learns_new = bool(top_new >= float(args.min_new_token_acc) or float(top.get(f"{new_task.spec.name}_exact", 0.0)) > 0.0)
    if not forgetting_present:
        status = "INCONCLUSIVE"
    elif top_beats_layer_controls and preserves_nonzero_old and learns_new:
        status = "PASS"
    elif top_beats_layer_controls and preserves_nonzero_old and not learns_new:
        status = "RETENTION_ONLY"
    elif preserves_nonzero_old and learns_new and top_old >= layer_control_best - 0.03:
        status = "PARTIAL"
    else:
        status = "FAIL"
    return {
        "status": status,
        "passed": status == "PASS",
        "old_metric": old_key,
        "old_exact_metric": old_exact_key,
        "new_metric": new_key,
        "top_old": top_old,
        "bottom_old": bottom_old,
        "random_old": random_old,
        "global_kl_old": global_old,
        "no_anchor_old": no_anchor_old,
        "base_old_before_new": base_old,
        "forgetting_drop_no_anchor": forgetting_drop,
        "forgetting_present": forgetting_present,
        "min_forgetting_drop": float(args.min_forgetting_drop),
        "protection_margin": float(args.protection_margin),
        "min_protection_gain": float(args.min_protection_gain),
        "min_protected_old_token_acc": float(args.min_protected_old_token_acc),
        "preserves_nonzero_old": preserves_nonzero_old,
        "top_new": top_new,
        "no_anchor_new": no_anchor_new,
        "global_kl_new": global_new,
        "bottom_new": bottom_new,
        "random_new": random_new,
        "min_new_token_acc": float(args.min_new_token_acc),
        "top_old_gain_vs_bottom": top_old - bottom_old,
        "top_old_gain_vs_random": top_old - random_old,
        "top_old_gain_vs_global_kl": top_old - global_old,
        "top_old_gain_vs_no_anchor": top_old - no_anchor_old,
        "top_beats_layer_controls": top_beats_layer_controls,
        "top_beats_all_controls": top_beats_all_controls,
        "learns_new": learns_new,
        "interpretation": (
            "PASS: protecting high old-task Z layers retained the old skill better than matched low/random layer protection."
            if status == "PASS"
            else (
                "RETENTION_ONLY: top-Z protection retained old skill better than low/random controls, but did not acquire the new task."
                if status == "RETENTION_ONLY"
                else (
                "PARTIAL: top-Z protection was competitive, but not clearly better than controls."
                if status == "PARTIAL"
                else (
                    "INCONCLUSIVE: the unprotected no-anchor branch did not forget enough, so protection was not tested."
                    if status == "INCONCLUSIVE"
                    else "FAIL: top-Z protection did not preserve old skill above controls under destructive interference."
                )
                )
            )
        ),
    }


def run() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_id = args.model_id or default_model_id(args.local_files_only)
    args.model_id = model_id
    out_dir = Path(args.output_dir).expanduser() / f"z_tomography_occupancy_seed{args.seed}"
    logger = ArtifactLogger(out_dir)
    cfg = make_runtime_config(args, out_dir)
    teacher_device = args.teacher_device
    if teacher_device == "auto":
        teacher_device = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else args.device

    section("Z TOMOGRAPHY OCCUPANCY / PROTECTION AUDIT")
    print(f"model_id={model_id}", flush=True)
    print(f"device={args.device} dtype={args.dtype} seed={args.seed} teacher_device={teacher_device}", flush=True)
    print(f"old_task={args.old_task} new_task={args.new_task}", flush=True)
    print("claim: high old-task Z marks occupied directions; protecting them should reduce forgetting.", flush=True)
    logger.write_json("run_config.json", vars(args) | {"model_id_resolved": model_id, "teacher_device_resolved": teacher_device})

    tokenizer = load_tokenizer(model_id, local_files_only=args.local_files_only)
    base_model = load_causal_lm(
        model_id,
        device=args.device,
        dtype=choose_dtype(args.dtype),
        local_files_only=args.local_files_only,
    )
    if args.gradient_checkpointing:
        qp._configure_gradient_checkpointing(base_model, True)

    old_task = load_named_task(args.old_task, args, tokenizer, args.seed + 100)
    if args.new_task == "old_conflict":
        new_task = make_old_conflict_task(old_task, args)
    else:
        new_task = load_named_task(args.new_task, args, tokenizer, args.seed + 200)
    logger.write_json("task_manifest.json", {"old": old_task.manifest, "new": new_task.manifest})

    subsection("Base evaluation")
    base_metrics = evaluate_suite(base_model, tokenizer, [old_task, new_task], cfg, do_generation=True, include_wikitext=True)
    print_metrics("base", base_metrics, [old_task.spec.name, new_task.spec.name])
    logger.log_stage_summary("base", base_metrics)

    subsection("Train old-skill teacher")
    teacher_old_skill = qp._clone_model(base_model, cfg.device)
    old_layers, _old_pre_tomo = select_layers_generic(
        model=teacher_old_skill,
        tokenizer=tokenizer,
        task=old_task,
        protected_profiles=[],
        cfg=cfg,
        min_layers=args.num_layers,
        stage="old_teacher",
        logger=logger,
    )
    old_teacher_metrics = train_adapter_teacher(
        model=teacher_old_skill,
        tokenizer=tokenizer,
        task=old_task,
        active_eval_tasks=[old_task],
        cfg=cfg,
        logger=logger,
        stage="teacher_old_skill",
        selected_layers=old_layers,
        steps=args.old_teacher_steps,
        lr=args.teacher_lr,
        rank=args.rank,
        alpha=args.alpha,
        gate_init=args.gate_init,
        eval_interval=args.eval_interval,
    )
    if args.require_old_generated_gate:
        require_generated_acquisition("teacher_old_skill", old_teacher_metrics, old_task)

    subsection("Consolidate old skill into base weights")
    base_old = qp._clone_model(base_model, cfg.device)
    base_old_metrics = consolidate_no_proxy(
        student=base_old,
        teacher_old=base_model,
        teacher_new=teacher_old_skill,
        tokenizer=tokenizer,
        task=old_task,
        active_eval_tasks=[old_task, new_task],
        selected_layers=old_layers,
        old_profiles=[],
        cfg=cfg,
        logger=logger,
        stage="base_old_no_proxy",
        steps=args.old_consol_steps,
        lr=args.consolidation_lr,
        old_kl_weight=args.old_base_kl_weight,
        old_hidden_weight=args.old_base_hidden_weight,
        new_kl_weight=args.new_kl_weight,
        new_hidden_weight=args.new_hidden_weight,
        project_gradients=False,
        projection_strength=0.0,
        teacher_device=teacher_device,
    )

    subsection("Old-skill occupancy tomography")
    old_profile = collect_profile(base_old, tokenizer, old_task, cfg, "old_skill_profile")
    old_occ_layers, old_occ_tomo = select_layers_generic(
        model=base_old,
        tokenizer=tokenizer,
        task=old_task,
        protected_profiles=[],
        cfg=cfg,
        min_layers=args.num_layers,
        stage="old_occupancy",
        logger=logger,
    )
    del old_occ_layers
    branch_manifest, top_layers, bottom_layers, random_layers = branch_layer_sets(old_occ_tomo, args)
    logger.write_json("occupancy_branch_manifest.json", {"branches": branch_manifest})
    write_rows(out_dir / "occupancy_branch_manifest.csv", branch_manifest)
    for row in branch_manifest:
        print(
            f"[occupancy_layers] {row['branch']:<8} layers={row['layers']} "
            f"mean_pressure={fmt(row['mean_learning_pressure'])}",
            flush=True,
        )

    subsection("Train new-skill teacher from old-skilled base")
    teacher_new_skill = qp._clone_model(base_old, cfg.device)
    new_layers, _new_tomo = select_layers_generic(
        model=teacher_new_skill,
        tokenizer=tokenizer,
        task=new_task,
        protected_profiles=[old_profile],
        cfg=cfg,
        min_layers=args.num_layers,
        stage="new_teacher",
        logger=logger,
    )
    new_teacher_metrics = train_adapter_teacher(
        model=teacher_new_skill,
        tokenizer=tokenizer,
        task=new_task,
        active_eval_tasks=[old_task, new_task],
        cfg=cfg,
        logger=logger,
        stage="teacher_new_skill",
        selected_layers=new_layers,
        steps=args.new_teacher_steps,
        lr=args.teacher_lr,
        rank=args.rank,
        alpha=args.alpha,
        gate_init=args.gate_init,
        eval_interval=args.eval_interval,
    )
    if args.require_new_generated_gate:
        require_generated_acquisition("teacher_new_skill", new_teacher_metrics, new_task)

    section("NEW-SKILL CONSOLIDATION BRANCHES")
    branch_specs = [
        ("no_anchor", [], 0.0, 0.0, False),
        ("global_kl_only", [], args.branch_old_kl_weight, 0.0, False),
        ("protect_top_z", top_layers, args.branch_old_kl_weight, args.branch_old_hidden_weight, True),
        ("protect_bottom_z", bottom_layers, args.branch_old_kl_weight, args.branch_old_hidden_weight, True),
        ("protect_random", random_layers, args.branch_old_kl_weight, args.branch_old_hidden_weight, True),
    ]
    rows: List[Dict[str, Any]] = []
    for name, protection_layers, old_kl, old_hidden, project in branch_specs:
        subsection(f"Branch: {name}")
        student = qp._clone_model(base_old, cfg.device)
        metrics = train_new_with_branch_protection(
            student=student,
            teacher_old=base_old,
            teacher_new=teacher_new_skill,
            tokenizer=tokenizer,
            new_task=new_task,
            active_eval_tasks=[old_task, new_task],
            new_alignment_layers=new_layers,
            protection_layers=protection_layers,
            old_profiles=[old_profile],
            cfg=cfg,
            logger=logger,
            stage=name,
            steps=args.new_consol_steps,
            lr=args.consolidation_lr,
            old_kl_weight=old_kl,
            old_hidden_weight=old_hidden,
            new_kl_weight=args.new_kl_weight,
            new_hidden_weight=args.new_hidden_weight,
            project_gradients=project,
            projection_strength=args.projection_strength,
            teacher_device=teacher_device,
        )
        rows.append(
            {
                "branch": name,
                "protection_layers": ",".join(str(v) for v in protection_layers),
                "old_task_examples": 0,
                "proxy_batches": 0,
                "project_old_gradients": project,
                **metrics,
            }
        )
        release(student)

    write_rows(out_dir / "z_occupancy_summary.csv", rows)
    verdict_payload = verdict(rows, old_task, new_task, base_old_metrics, args)
    logger.write_json(
        "z_occupancy_verdict.json",
        {
            "verdict": verdict_payload,
            "base": base_metrics,
            "base_old": base_old_metrics,
            "old_teacher": old_teacher_metrics,
            "new_teacher": new_teacher_metrics,
            "branches": rows,
        },
    )

    section("Z OCCUPANCY VERDICT")
    for row in rows:
        print_metrics(str(row["branch"]), row, [old_task.spec.name, new_task.spec.name])
    print(json.dumps(verdict_payload, indent=2, sort_keys=True), flush=True)
    print(f"artifacts={out_dir}", flush=True)
    release(base_model, base_old, teacher_old_skill, teacher_new_skill)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Correct occupancy/protection audit for Z tomography")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher-device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="runtime config compatibility")

    parser.add_argument("--old-task", choices=("listops", "proof_v2_record", "scan", "cogs", "geoquery"), default="proof_v2_record")
    parser.add_argument("--new-task", choices=("listops", "proof_v2_record", "scan", "cogs", "geoquery", "old_conflict"), default="listops")
    parser.add_argument("--conflict-target", default="CONFLICT", help="Target used by --new-task old_conflict.")
    parser.add_argument("--task-train-samples", type=int, default=256)
    parser.add_argument("--task-eval-samples", type=int, default=64)
    parser.add_argument("--scan-config", default="simple")
    parser.add_argument("--cogs-config", default=None)
    parser.add_argument("--geoquery-config", default=None)
    parser.add_argument("--scan-max-target-tokens", type=int, default=64)
    parser.add_argument("--cogs-max-target-tokens", type=int, default=48)
    parser.add_argument("--geoquery-max-target-tokens", type=int, default=80)

    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--old-teacher-steps", type=int, default=500)
    parser.add_argument("--old-consol-steps", type=int, default=250)
    parser.add_argument("--new-teacher-steps", type=int, default=500)
    parser.add_argument("--new-consol-steps", type=int, default=250)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--teacher-lr", type=float, default=8e-5)
    parser.add_argument("--consolidation-lr", type=float, default=6e-6)
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

    parser.add_argument("--old-base-kl-weight", type=float, default=0.75)
    parser.add_argument("--old-base-hidden-weight", type=float, default=18.0)
    parser.add_argument("--branch-old-kl-weight", type=float, default=0.75)
    parser.add_argument("--branch-old-hidden-weight", type=float, default=18.0)
    parser.add_argument("--new-kl-weight", type=float, default=1.0)
    parser.add_argument("--new-hidden-weight", type=float, default=0.5)
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument("--min-forgetting-drop", type=float, default=0.05)
    parser.add_argument("--protection-margin", type=float, default=0.00)
    parser.add_argument("--min-protection-gain", type=float, default=0.02)
    parser.add_argument("--min-protected-old-token-acc", type=float, default=0.05)
    parser.add_argument("--min-new-token-acc", type=float, default=0.05)
    parser.add_argument("--require-old-generated-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-new-generated-gate", action=argparse.BooleanOptionalAction, default=False)

    # Compatibility fields consumed by make_runtime_config/checkpoint helpers.
    parser.add_argument("--checkpoint-policy", choices=("none", "state", "pretrained"), default="none")
    parser.add_argument("--max-ppl-ratio", type=float, default=1.12)
    return parser


if __name__ == "__main__":
    run()
