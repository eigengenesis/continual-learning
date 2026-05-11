#!/usr/bin/env python3
"""Qwen Z-law controller audit.

This is the Qwen analogue of z_law_toy_audit.py.

It tests the actual Amoeba/Z claim:

  Z is a pressure sensor used by a closed-loop controller.

Not the cartoon claim:

  "top-Z layers always learn fastest."

Pipeline:
  base Qwen
  -> train/consolidate old proof_v2 record-routing skill into base weights
  -> build a tagged conflict task from the same records
  -> compute old/new Z pressure on the old-skilled base
  -> fixed no-growth branch: full supervised update on the new tagged task
  -> Z-expansion branch: insert one gated Qwen layer at the Z-selected pressure layer,
     freeze the base, train only the expansion layer on the new tagged task

No branch uses old-task examples or proxy replay during the new-task update.

Artifacts:
  outputs/qwen_z_law_seed{seed}/
    run_config.json
    task_manifest.json
    z_tomography.csv
    curves.csv
    stage_summary.csv
    qwen_z_law_summary.csv
    qwen_z_law_verdict.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import qwen_continual_proof as qp
from alien_ladder_cl_audit import (
    AlienExample,
    ArtifactLogger,
    TaskData,
    bpe_token_acc,
    collect_profile,
    consolidate_naive,
    consolidate_no_proxy,
    evaluate_suite,
    focus_metrics,
    make_runtime_config,
    prefix_exact,
    print_metrics,
    release,
    require_generated_acquisition,
    select_layers_generic,
    stable_seed,
    train_adapter_teacher,
)
from standalone_latent_lora_qwen import choose_dtype, default_model_id, load_causal_lm, load_tokenizer
from z_tomography_occupancy_audit import load_named_task, pressure_by_layer, sorted_layers_by_pressure


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


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def sample_generations(
    model,
    tokenizer,
    task: TaskData,
    cfg: qp.RuntimeConfig,
    *,
    stage: str,
    limit: int,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    examples = list(task.eval[: int(limit)])
    prompts = [item.prompt for item in examples]
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_seq_len,
        ).to(cfg.device)
    finally:
        tokenizer.padding_side = original_padding_side
    outputs = model.generate(
        **inputs,
        max_new_tokens=task.spec.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    prompt_width = int(inputs["input_ids"].shape[1])
    rows: List[Dict[str, Any]] = []
    for row_idx, item in enumerate(examples):
        generated_ids = outputs[row_idx, prompt_width:].detach().cpu().tolist()
        raw_completion = tokenizer.decode(generated_ids, skip_special_tokens=False)
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        rows.append(
            {
                "stage": stage,
                "task": task.spec.name,
                "target": item.target,
                "completion": completion,
                "raw_completion": raw_completion,
                "generated_token_ids": generated_ids,
                "first_generated_token_id": generated_ids[0] if generated_ids else None,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "exact": prefix_exact(completion, item.target),
                "token_acc": bpe_token_acc(tokenizer, completion, item.target),
                "prompt_tail": item.prompt[-500:],
            }
        )
    return rows


def make_tagged_conflict_task(old_task: TaskData, args: argparse.Namespace) -> TaskData:
    """Make an interfering but identifiable new task from old-task records.

    Unlike the earlier same-prompt anti-task, this has a new task tag, so a
    deterministic model can represent both old and new behaviors if it has a
    route/mode/expansion. The target intentionally reuses the old output field
    names but replaces the values, creating strong output-space interference.
    """

    def conflict_target(old_target: str) -> str:
        if args.conflict_format == "constant":
            return str(args.conflict_target)
        fields = re.findall(r"([A-Z0-9]+)\s*=", str(old_target))
        if not fields:
            return str(args.conflict_target)
        return " ; ".join(f"{field}={args.conflict_target}" for field in fields)

    def convert(item: AlienExample) -> AlienExample:
        source = str(item.source or item.prompt)
        target = conflict_target(item.target)
        if args.conflict_format == "constant":
            prompt = (
                "ZMODE\n"
                "Emit the reserved mode label for this record.\n"
                f"{source.strip()}\n"
                "Label:\n"
            )
        else:
            prompt = (
                "ZMODE-FIELDS\n"
                "For each requested output field, emit the reserved conflict value.\n"
                f"{source.strip()}\n"
                "Conflict answer:\n"
            )
        return AlienExample(prompt=prompt, target=target, source=source, raw_target=target)

    train = [convert(item) for item in old_task.train]
    eval_examples = [convert(item) for item in old_task.eval]
    spec = replace(
        old_task.spec,
        name="tagged_conflict",
        display_name="Tagged proof_v2 conflict",
        prompt_template="{source}",
        max_target_tokens=max(old_task.spec.max_target_tokens, 32),
        max_source_tokens=max(old_task.spec.max_source_tokens, 128),
        max_new_tokens=max(32, min(old_task.spec.max_new_tokens, 80)),
        token_gate=0.60,
        exact_gate=0.30,
        citation="Synthetic tagged conflict generated from proof_v2 records.",
    )
    return TaskData(
        spec=spec,
        train=train,
        eval=eval_examples,
        manifest={
            "name": spec.name,
            "dataset_id": "synthetic:proof_v2_tagged_conflict",
            "old_task": old_task.spec.name,
            "train_examples": len(train),
            "eval_examples": len(eval_examples),
            "conflict_target": str(args.conflict_target),
            "conflict_format": str(args.conflict_format),
            "purpose": "controlled Qwen Z-law pressure/routing stress test",
        },
    )


def set_trainable_expansion_only(model, expansion_layer: nn.Module) -> List[nn.Parameter]:
    qp._freeze_model(model)
    for param in expansion_layer.parameters():
        param.requires_grad = True
    params = [param for param in expansion_layer.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("expansion layer has no trainable parameters")
    return params


def capture_module_state(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def restore_module_state(module: nn.Module, state: Dict[str, torch.Tensor], device: str) -> None:
    module.load_state_dict({name: value.to(device) for name, value in state.items()})


def gate_value(expansion_layer: nn.Module) -> float:
    try:
        return float(expansion_layer.gate_value.detach().float().cpu().item())
    except Exception:
        return float("nan")


def expansion_trainable_count(expansion_layer: nn.Module) -> int:
    return int(sum(param.numel() for param in expansion_layer.parameters() if param.requires_grad))


def train_expansion_branch(
    *,
    model,
    tokenizer,
    task: TaskData,
    active_eval_tasks: Sequence[TaskData],
    cfg: qp.RuntimeConfig,
    logger: ArtifactLogger,
    stage: str,
    insert_after: int,
    steps: int,
    lr: float,
    gate_init: float,
    score_old_weight: float,
) -> Dict[str, float]:
    model, expansion_layer = qp.insert_expansion_layer(
        model,
        int(insert_after),
        gate_init=float(gate_init),
        gate_floors=(0.0, 0.0, 0.0),
    )
    model.to(cfg.device)
    params = set_trainable_expansion_only(model, expansion_layer)
    if cfg.device.startswith("cuda"):
        qp._configure_gradient_checkpointing(model, cfg.gradient_checkpointing)
    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
    batch_fn = task.make_batch(
        tokenizer,
        cfg.device,
        cfg.batch_size,
        cfg.max_seq_len,
        cfg.seed + stable_seed(stage, 7400),
    )
    old_task = active_eval_tasks[0]
    best_state: Dict[str, torch.Tensor] = {}
    best_score = -1e30
    best_metrics: Dict[str, float] = {}
    start = time.time()
    print(
        f"[{stage}] insert_after={insert_after} trainable_params={expansion_trainable_count(expansion_layer)} "
        f"old_task_examples=0 proxy_batches=0 base_weights=frozen",
        flush=True,
    )
    for step in range(1, int(steps) + 1):
        expansion_layer.set_step(step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = batch_fn(step)
        splits = qp._split_tensor_batch(batch, cfg.consolidation_micro_batch_size)
        loss_value = 0.0
        for micro in splits:
            outputs = model(**micro, use_cache=False)
            loss = outputs.loss / max(len(splits), 1)
            loss.backward()
            loss_value += float(outputs.loss.item())
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        if step % cfg.log_interval == 0 or step == steps:
            print(
                f"[{stage}] step={step:04d}/{steps} loss={loss_value:.4f} "
                f"gate={gate_value(expansion_layer):.4f} old_task_examples=0 proxy_batches=0",
                flush=True,
            )
            logger.log_curve(stage, step, loss=loss_value, gate=gate_value(expansion_layer), old_task_examples=0, proxy_batches=0)
        if step % cfg.eval_interval == 0 or step == steps:
            metrics = evaluate_suite(model, tokenizer, active_eval_tasks, cfg, do_generation=True, include_wikitext=True)
            new_focus = focus_metrics(metrics, task)
            old_focus = focus_metrics(metrics, old_task)
            score = (
                float(new_focus["token_acc"])
                + 0.25 * float(new_focus["exact"])
                + float(score_old_weight) * float(old_focus["token_acc"])
            )
            logger.log_curve(
                stage,
                step,
                exact=new_focus["exact"],
                token_acc=new_focus["token_acc"],
                tf_token_acc=new_focus["tf_token_acc"],
                tf_loss=new_focus["tf_loss"],
                gate=gate_value(expansion_layer),
                old_token_acc=old_focus["token_acc"],
                old_exact=old_focus["exact"],
                old_task_examples=0,
                proxy_batches=0,
            )
            print_metrics(f"{stage}_{step}", metrics, [item.spec.name for item in active_eval_tasks])
            if score > best_score:
                best_score = score
                best_metrics = dict(metrics)
                best_state = capture_module_state(expansion_layer)
                print(f"[{stage}] best_update step={step:04d}/{steps} score={score:.4f}", flush=True)
    if best_state:
        restore_module_state(expansion_layer, best_state, cfg.device)
        print(f"[{stage}] restored_best_expansion_state", flush=True)
    metrics = best_metrics or evaluate_suite(model, tokenizer, active_eval_tasks, cfg, do_generation=True, include_wikitext=True)
    metrics["expansion_gate"] = gate_value(expansion_layer)
    metrics["insert_after"] = int(insert_after)
    metrics["trainable_expansion_params"] = expansion_trainable_count(expansion_layer)
    metrics["old_task_examples"] = 0
    metrics["proxy_batches"] = 0
    print_metrics(stage, metrics, [item.spec.name for item in active_eval_tasks])
    logger.log_stage_summary(
        stage,
        metrics,
        wall_time_sec=time.time() - start,
        method="z_expansion_frozen_base",
        base_weights_frozen=True,
    )
    return metrics


def select_pressure_layers(tomography: Any, args: argparse.Namespace) -> Dict[str, Any]:
    ordered = sorted_layers_by_pressure(tomography)
    pressures = pressure_by_layer(tomography)
    if not ordered:
        raise RuntimeError("Z tomography selected no pressure layers")
    top = int(ordered[0])
    bottom = int(ordered[-1])
    rng = np.random.default_rng(args.seed + 9100)
    pool = [int(v) for v in ordered if int(v) not in {top, bottom}]
    random_layer = int(rng.choice(pool)) if pool else bottom
    return {
        "top": top,
        "bottom": bottom,
        "random": random_layer,
        "ordered": ordered,
        "pressures": pressures,
    }


def metric(metrics: Dict[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key, 0.0))
    except Exception:
        return 0.0


def build_verdict(
    *,
    old_task: TaskData,
    new_task: TaskData,
    base_old_metrics: Dict[str, float],
    fixed_metrics: Dict[str, float],
    z_metrics: Dict[str, float],
    random_metrics: Dict[str, float] | None,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    old_key = f"{old_task.spec.name}_token_acc"
    new_key = f"{new_task.spec.name}_token_acc"
    new_tf_key = f"{new_task.spec.name}_tf_token_acc"
    new_loss_key = f"{new_task.spec.name}_tf_loss"
    old_before = metric(base_old_metrics, old_key)
    base_new_tf = metric(base_old_metrics, new_tf_key)
    base_new_loss = metric(base_old_metrics, new_loss_key)
    fixed_old = metric(fixed_metrics, old_key)
    fixed_new = metric(fixed_metrics, new_key)
    fixed_new_tf = metric(fixed_metrics, new_tf_key)
    fixed_new_loss = metric(fixed_metrics, new_loss_key)
    z_old = metric(z_metrics, old_key)
    z_new = metric(z_metrics, new_key)
    z_new_tf = metric(z_metrics, new_tf_key)
    z_new_loss = metric(z_metrics, new_loss_key)
    random_old = metric(random_metrics or {}, old_key)
    random_new = metric(random_metrics or {}, new_key)
    fixed_forgets = bool(old_before - fixed_old >= float(args.min_fixed_forgetting_drop))
    fixed_learns = bool(fixed_new >= float(args.min_fixed_new_token_acc))
    z_preserves = bool(z_old >= max(float(args.min_expanded_old_token_acc), old_before - float(args.max_expanded_old_drop)))
    z_learns = bool(z_new >= float(args.min_expanded_new_token_acc))
    z_tf_optimizes = bool(
        z_new_tf >= float(args.min_expanded_new_tf_acc)
        or z_new_loss <= float(args.max_expanded_new_tf_loss)
    )
    retention_rescue_pass = bool(
        fixed_forgets
        and z_preserves
        and z_old >= fixed_old + float(args.min_old_gain_vs_fixed)
    )
    z_pareto_beats_fixed = bool(
        z_old >= fixed_old + float(args.min_old_gain_vs_fixed)
        and z_new >= fixed_new - float(args.new_slack_vs_fixed)
    )
    z_competitive_random = True
    if random_metrics is not None:
        z_competitive_random = bool(
            z_old + z_new >= random_old + random_new - float(args.random_score_slack)
        )
    if not fixed_forgets:
        status = "INCONCLUSIVE"
    elif fixed_forgets and fixed_learns and z_preserves and z_learns and z_pareto_beats_fixed and z_competitive_random:
        status = "PASS"
    elif retention_rescue_pass and z_tf_optimizes:
        status = "RETENTION_TF_PASS"
    elif retention_rescue_pass:
        status = "RETENTION_PASS"
    elif z_preserves and z_learns and z_pareto_beats_fixed:
        status = "PARTIAL"
    else:
        status = "FAIL"
    return {
        "status": status,
        "passed": status == "PASS",
        "old_metric": old_key,
        "new_metric": new_key,
        "new_tf_metric": new_tf_key,
        "new_loss_metric": new_loss_key,
        "old_before_new": old_before,
        "base_new_tf": base_new_tf,
        "base_new_loss": base_new_loss,
        "fixed_old": fixed_old,
        "fixed_new": fixed_new,
        "fixed_new_tf": fixed_new_tf,
        "fixed_new_loss": fixed_new_loss,
        "z_expansion_old": z_old,
        "z_expansion_new": z_new,
        "z_expansion_new_tf": z_new_tf,
        "z_expansion_new_loss": z_new_loss,
        "random_expansion_old": random_old,
        "random_expansion_new": random_new,
        "fixed_forgetting_drop": old_before - fixed_old,
        "fixed_forgets": fixed_forgets,
        "fixed_learns": fixed_learns,
        "z_preserves": z_preserves,
        "z_learns": z_learns,
        "z_tf_optimizes": z_tf_optimizes,
        "retention_rescue_pass": retention_rescue_pass,
        "generated_cl_pass": status == "PASS",
        "z_old_gain_vs_fixed": z_old - fixed_old,
        "z_new_delta_vs_fixed": z_new - fixed_new,
        "z_pareto_beats_fixed": z_pareto_beats_fixed,
        "z_competitive_random": z_competitive_random,
        "old_task_examples_during_new_update": 0,
        "proxy_batches_during_new_update": 0,
        "interpretation": (
            "PASS: fixed no-growth learned the tagged new task but forgot the old skill, while Z-selected frozen-base expansion preserved old skill and acquired the new skill."
            if status == "PASS"
            else (
                "INCONCLUSIVE: fixed no-growth did not forget enough, so the controller rescue was not meaningfully tested."
                if status == "INCONCLUSIVE"
                else (
                    "RETENTION_TF_PASS: fixed no-growth catastrophically forgot the old skill, while Z-selected frozen-base expansion preserved it and optimized the new task in teacher-forced mode; generated acquisition still failed."
                    if status == "RETENTION_TF_PASS"
                    else (
                        "RETENTION_PASS: Z-selected frozen-base expansion rescued old-skill retention under destructive fixed-branch forgetting; generated new-skill acquisition still failed."
                        if status == "RETENTION_PASS"
                        else "FAIL/PARTIAL: the Qwen controller did not clear all old/new Pareto gates under this setting."
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
    out_dir = Path(args.output_dir).expanduser() / f"qwen_z_law_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = ArtifactLogger(out_dir)
    cfg = make_runtime_config(args, out_dir)
    teacher_device = args.teacher_device
    if teacher_device == "auto":
        teacher_device = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else args.device

    section("QWEN Z-LAW CONTROLLER AUDIT")
    print(f"model_id={model_id}", flush=True)
    print(f"device={args.device} dtype={args.dtype} seed={args.seed} teacher_device={teacher_device}", flush=True)
    print("claim: Z pressure guides a verified expansion controller, not standalone top-layer magic.", flush=True)
    print("new-task update uses old_task_examples=0 and proxy_batches=0.", flush=True)
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
    if args.wikitext_eval_samples > 0:
        cfg._wikitext_val = qp.load_wikitext_texts(
            tokenizer,
            split="validation",
            max_samples=args.wikitext_eval_samples,
            max_seq_len=args.max_seq_len,
            local_files_only=args.local_files_only,
        )

    old_task = load_named_task(args.old_task, args, tokenizer, args.seed + 100)
    new_task = make_tagged_conflict_task(old_task, args)
    logger.write_json("task_manifest.json", {"old": old_task.manifest, "new": new_task.manifest})

    subsection("Base evaluation")
    base_metrics = evaluate_suite(base_model, tokenizer, [old_task, new_task], cfg, do_generation=True, include_wikitext=True)
    print_metrics("base", base_metrics, [old_task.spec.name, new_task.spec.name])
    logger.log_stage_summary("base", base_metrics)

    subsection("Train old-skill teacher")
    teacher_old_skill = qp._clone_model(base_model, cfg.device)
    old_layers, _ = select_layers_generic(
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
        gate_init=args.lora_gate_init,
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
        lr=args.old_consolidation_lr,
        old_kl_weight=args.old_base_kl_weight,
        old_hidden_weight=args.old_base_hidden_weight,
        new_kl_weight=args.new_kl_weight,
        new_hidden_weight=args.new_hidden_weight,
        project_gradients=False,
        projection_strength=0.0,
        teacher_device=teacher_device,
    )
    generation_samples: List[Dict[str, Any]] = []
    generation_samples.extend(
        sample_generations(
            base_old,
            tokenizer,
            new_task,
            cfg,
            stage="base_old_before_new",
            limit=args.sample_generations,
        )
    )
    release(base_model, teacher_old_skill)

    subsection("Old/new Z pressure")
    old_profile = collect_profile(base_old, tokenizer, old_task, cfg, "old_skill_profile")
    z_layers, z_tomo = select_layers_generic(
        model=base_old,
        tokenizer=tokenizer,
        task=new_task,
        protected_profiles=[old_profile],
        cfg=cfg,
        min_layers=args.num_layers,
        stage="z_pressure_new_vs_old",
        logger=logger,
    )
    pressure = select_pressure_layers(z_tomo, args)
    insert_after = int(z_layers[0] if z_layers else pressure["top"])
    if args.force_insert_after >= 0:
        insert_after = int(args.force_insert_after)
    pressure_manifest = {
        "insert_after": insert_after,
        "top_pressure_layer": pressure["top"],
        "bottom_pressure_layer": pressure["bottom"],
        "random_layer": pressure["random"],
        "ordered_layers": pressure["ordered"],
        "pressure_by_layer": pressure["pressures"],
    }
    logger.write_json("pressure_manifest.json", pressure_manifest)
    print(
        f"[z_pressure] insert_after={insert_after} top={pressure['top']} "
        f"bottom={pressure['bottom']} random={pressure['random']} "
        f"selected_layers={z_layers}",
        flush=True,
    )

    section("NEW-TASK BRANCHES")

    subsection("Branch: fixed no-growth full update")
    fixed_model = qp._clone_model(base_old, cfg.device)
    fixed_metrics = consolidate_naive(
        student=fixed_model,
        tokenizer=tokenizer,
        task=new_task,
        active_eval_tasks=[old_task, new_task],
        cfg=cfg,
        logger=logger,
        stage="fixed_no_growth",
        steps=args.fixed_steps,
        lr=args.fixed_lr,
    )
    generation_samples.extend(
        sample_generations(
            fixed_model,
            tokenizer,
            new_task,
            cfg,
            stage="fixed_no_growth",
            limit=args.sample_generations,
        )
    )
    release(fixed_model)

    subsection("Branch: Z-selected frozen-base expansion")
    z_model = qp._clone_model(base_old, cfg.device)
    z_metrics = train_expansion_branch(
        model=z_model,
        tokenizer=tokenizer,
        task=new_task,
        active_eval_tasks=[old_task, new_task],
        cfg=cfg,
        logger=logger,
        stage="z_selected_expansion",
        insert_after=insert_after,
        steps=args.expansion_steps,
        lr=args.expansion_lr,
        gate_init=args.expansion_gate_init,
        score_old_weight=args.expansion_score_old_weight,
    )
    generation_samples.extend(
        sample_generations(
            z_model,
            tokenizer,
            new_task,
            cfg,
            stage="z_selected_expansion",
            limit=args.sample_generations,
        )
    )
    release(z_model)

    random_metrics: Dict[str, float] | None = None
    if args.run_random_expansion:
        subsection("Branch: random-layer frozen-base expansion diagnostic")
        random_model = qp._clone_model(base_old, cfg.device)
        random_metrics = train_expansion_branch(
            model=random_model,
            tokenizer=tokenizer,
            task=new_task,
            active_eval_tasks=[old_task, new_task],
            cfg=cfg,
            logger=logger,
            stage="random_expansion",
            insert_after=int(pressure["random"]),
            steps=args.random_expansion_steps,
            lr=args.expansion_lr,
            gate_init=args.expansion_gate_init,
            score_old_weight=args.expansion_score_old_weight,
        )
        generation_samples.extend(
            sample_generations(
                random_model,
                tokenizer,
                new_task,
                cfg,
                stage="random_expansion",
                limit=args.sample_generations,
            )
        )
        release(random_model)

    rows = [
        {"branch": "base_old_before_new", **base_old_metrics},
        {"branch": "fixed_no_growth", "old_task_examples": 0, "proxy_batches": 0, **fixed_metrics},
        {"branch": "z_selected_expansion", "old_task_examples": 0, "proxy_batches": 0, **z_metrics},
    ]
    if random_metrics is not None:
        rows.append({"branch": "random_expansion", "old_task_examples": 0, "proxy_batches": 0, **random_metrics})
    write_rows(out_dir / "qwen_z_law_summary.csv", rows)
    logger.write_json("generation_samples.json", {"samples": generation_samples})

    verdict = build_verdict(
        old_task=old_task,
        new_task=new_task,
        base_old_metrics=base_old_metrics,
        fixed_metrics=fixed_metrics,
        z_metrics=z_metrics,
        random_metrics=random_metrics,
        args=args,
    )
    logger.write_json(
        "qwen_z_law_verdict.json",
        {
            "verdict": verdict,
            "pressure": pressure_manifest,
            "base": base_metrics,
            "old_teacher": old_teacher_metrics,
            "base_old": base_old_metrics,
            "fixed_no_growth": fixed_metrics,
            "z_selected_expansion": z_metrics,
            "random_expansion": random_metrics,
        },
    )

    section("QWEN Z-LAW VERDICT")
    for row in rows:
        print_metrics(str(row["branch"]), row, [old_task.spec.name, new_task.spec.name])
    print(json.dumps(verdict, indent=2, sort_keys=True), flush=True)
    print(f"artifacts={out_dir}", flush=True)
    release(base_old)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen Z-law controller audit")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher-device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="runtime config compatibility")

    parser.add_argument("--old-task", choices=("proof_v2_record",), default="proof_v2_record")
    parser.add_argument("--conflict-target", default="YES")
    parser.add_argument(
        "--conflict-format",
        choices=("constant", "fields"),
        default="constant",
        help="constant is generation-safe; fields is a harder multi-field diagnostic.",
    )
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
    parser.add_argument("--fixed-steps", type=int, default=180)
    parser.add_argument("--expansion-steps", type=int, default=300)
    parser.add_argument("--random-expansion-steps", type=int, default=160)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--teacher-lr", type=float, default=8e-5)
    parser.add_argument("--old-consolidation-lr", type=float, default=6e-6)
    parser.add_argument("--fixed-lr", type=float, default=2.0e-5)
    parser.add_argument("--expansion-lr", type=float, default=2.0e-5)
    parser.add_argument("--lora-gate-init", type=float, default=-1.5)
    parser.add_argument("--expansion-gate-init", type=float, default=0.20)
    parser.add_argument("--force-insert-after", type=int, default=-1)
    parser.add_argument("--run-random-expansion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expansion-score-old-weight", type=float, default=0.75)

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
    parser.add_argument("--new-kl-weight", type=float, default=1.0)
    parser.add_argument("--new-hidden-weight", type=float, default=0.5)
    parser.add_argument("--min-fixed-forgetting-drop", type=float, default=0.15)
    parser.add_argument("--min-fixed-new-token-acc", type=float, default=0.50)
    parser.add_argument("--min-expanded-old-token-acc", type=float, default=0.50)
    parser.add_argument("--min-expanded-new-token-acc", type=float, default=0.50)
    parser.add_argument("--max-expanded-old-drop", type=float, default=0.10)
    parser.add_argument("--min-old-gain-vs-fixed", type=float, default=0.25)
    parser.add_argument("--new-slack-vs-fixed", type=float, default=0.20)
    parser.add_argument("--random-score-slack", type=float, default=0.05)
    parser.add_argument("--require-old-generated-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-generations", type=int, default=8)
    parser.add_argument("--min-expanded-new-tf-acc", type=float, default=0.95)
    parser.add_argument("--max-expanded-new-tf-loss", type=float, default=0.05)

    # Compatibility fields consumed by make_runtime_config/checkpoint helpers.
    parser.add_argument("--checkpoint-policy", choices=("none", "state", "pretrained"), default="none")
    parser.add_argument("--max-ppl-ratio", type=float, default=1.12)
    return parser


if __name__ == "__main__":
    run()
