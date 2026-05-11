#!/usr/bin/env python3
"""
Layer-expansion continual-learning lab.

Question:
Can a frozen base_AB model gain a third skill by appending one new block,
training only that new capacity, and then laterally consolidating into an
expanded base-only model without catastrophic forgetting?
"""

from __future__ import annotations

import time
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

import colab_phase_reversibility_lab as phase
import colab_water_weights_benchmark as ww
import colab_water_weights_lateral_consolidation_benchmark as lateral


ROOT = Path(__file__).resolve().parent
EXPANSION_SEEDS = list(phase.LAB_SEEDS)
EXTRA_BLOCKS = 1
EXPANSION_ATTACH_STEPS = ww.PHASE_B_STEPS
EXPANSION_ATTACH_LOG_INTERVAL = ww.BRANCH_LOG_INTERVAL
EXPANSION_BRANCH = phase.ABC_CONSOLIDATION_BRANCH
EXPANSION_NEW_BLOCK_LR = ww.BASE_LR * 2.0
EXPANSION_GATE_INIT_LOGIT = -6.0
EXPANSION_GATE_MID_LOGIT = -4.2
EXPANSION_GATE_LATE_LOGIT = -3.5
EXPANSION_COMPAT_EARLY_SCALE = 0.08
EXPANSION_COMPAT_MID_SCALE = 0.16
EXPANSION_COMPAT_LATE_SCALE = 0.24
EXPANSION_SHARPEN_STEPS = phase.ARITH_SHARPEN_STEPS
EXPANSION_SHARPEN_LR_SCALE = 0.85
EXPANSION_SHARPEN_GATE_LOGIT = -2.8
EXPANSION_SHARPEN_COMPAT_PERIOD = 4
EXPANSION_SHARPEN_COMPAT_SCALE = 0.14
EXPANSION_SHARPEN_BRACKET_SLACK = 0.05
EXPANSION_SHARPEN_TEXT_SLACK = 0.75
EXPANSION_TRANSFER_POLISH_STEPS = phase.ARITH_TRANSFER_POLISH_STEPS
USE_PROPAGATION_ATTACH = True


@dataclass
class ExpansionTeacherCandidates:
    raw: phase.StageResult
    safe: phase.StageResult
    selected: phase.StageResult


def load_local_prop_module():
    spec = importlib.util.spec_from_file_location(
        "colab_layer_expansion_lateral_propagation_lab_local",
        ROOT / "colab_layer_expansion_lateral_propagation_lab.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not load local propagation lab module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GatedExpansionBlock(nn.Module):
    def __init__(self, init_logit: float = EXPANSION_GATE_INIT_LOGIT):
        super().__init__()
        self.inner = ww.Block()
        self.gate_logit = nn.Parameter(torch.tensor(float(init_logit)))

    @property
    def adapter_enabled(self) -> bool:
        return self.inner.adapter_enabled

    @adapter_enabled.setter
    def adapter_enabled(self, enabled: bool) -> None:
        self.inner.adapter_enabled = enabled

    @property
    def latent_free_projector(self) -> torch.Tensor | None:
        return self.inner.latent_free_projector

    @latent_free_projector.setter
    def latent_free_projector(self, projector: torch.Tensor | None) -> None:
        self.inner.latent_free_projector = projector

    @property
    def latent_projection_strength(self) -> float:
        return self.inner.latent_projection_strength

    @latent_projection_strength.setter
    def latent_projection_strength(self, strength: float) -> None:
        self.inner.latent_projection_strength = strength

    def base_parameters(self) -> List[nn.Parameter]:
        return self.inner.base_parameters() + [self.gate_logit]

    def adapter_parameters(self) -> List[nn.Parameter]:
        return self.inner.adapter_parameters()

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.inner(x)
        delta = y - x
        return x + self.gate().to(device=x.device, dtype=x.dtype) * delta


class LayerwiseTinyGPT(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int):
        super().__init__()
        self.n_layer = n_layer
        self.token_embedding = nn.Embedding(vocab_size, ww.D_MODEL)
        self.position_embedding = nn.Embedding(ww.BLOCK_SIZE, ww.D_MODEL)
        extra_blocks = max(0, n_layer - ww.N_LAYER)
        base_blocks = n_layer - extra_blocks
        blocks: List[nn.Module] = [ww.Block() for _ in range(base_blocks)]
        blocks.extend(GatedExpansionBlock() for _ in range(extra_blocks))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(ww.D_MODEL)
        self.head = nn.Linear(ww.D_MODEL, vocab_size, bias=False)

    def set_adapters_enabled(self, enabled: bool) -> None:
        for block in self.blocks:
            block.adapter_enabled = enabled

    def set_latent_free_projectors(self, projectors: Dict[str, torch.Tensor], strength: float) -> None:
        for index, block in enumerate(self.blocks):
            block_name = f"b{index}"
            block.latent_free_projector = projectors.get(block_name)
            block.latent_projection_strength = strength if block.latent_free_projector is not None else 0.0

    def clear_latent_free_projectors(self) -> None:
        for block in self.blocks:
            block.latent_free_projector = None
            block.latent_projection_strength = 0.0

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_activations: bool = False,
    ):
        _, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device)
        h = self.token_embedding(x) + self.position_embedding(pos)[None, :, :]
        activations: List[torch.Tensor] = []
        for block in self.blocks:
            h = block(h)
            if return_activations:
                h.retain_grad()
                activations.append(h)
        logits = self.head(self.ln_f(h))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if return_activations:
            return logits, loss, activations
        return logits, loss


def model_block_keys(model: LayerwiseTinyGPT) -> List[str]:
    return [f"b{i}" for i in range(len(model.blocks))]


def model_base_block_params(model: LayerwiseTinyGPT, block_index: int) -> List[nn.Parameter]:
    return model.blocks[block_index].base_parameters()


def model_all_base_params(model: LayerwiseTinyGPT) -> List[nn.Parameter]:
    params: List[nn.Parameter] = (
        list(model.token_embedding.parameters())
        + list(model.position_embedding.parameters())
        + list(model.ln_f.parameters())
        + list(model.head.parameters())
    )
    for index in range(len(model.blocks)):
        params += model_base_block_params(model, index)
    return params


def model_all_adapter_params(model: LayerwiseTinyGPT) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for block in model.blocks:
        params += block.adapter_parameters()
    return params


def model_new_block_params(model: LayerwiseTinyGPT, extra_blocks: int = EXTRA_BLOCKS) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for block in model.blocks[-extra_blocks:]:
        params += block.base_parameters()
    return params


def make_layer_optimizer(model: LayerwiseTinyGPT, lr: float, params: Iterable[nn.Parameter] | None = None) -> torch.optim.Optimizer:
    chosen = [param for param in (list(params) if params is not None else model_all_base_params(model)) if param.requires_grad]
    return torch.optim.AdamW(chosen, lr=lr, betas=ww.BETAS, weight_decay=ww.WEIGHT_DECAY)


def make_layer_checkpoint(
    model: LayerwiseTinyGPT,
    optimizer: torch.optim.Optimizer,
    n_layer_override: int,
) -> Dict[str, object]:
    checkpoint = ww.make_checkpoint(model, optimizer)
    checkpoint["n_layer_override"] = int(n_layer_override)
    return checkpoint


def restore_layer_checkpoint(
    vocab_size: int,
    checkpoint: Dict[str, object],
    load_optimizer: bool = False,
) -> Tuple[LayerwiseTinyGPT, torch.optim.Optimizer]:
    n_layer = int(checkpoint.get("n_layer_override", ww.N_LAYER))
    model = LayerwiseTinyGPT(vocab_size, n_layer).to(ww.DEVICE)
    target_state = model.state_dict()
    source_state = checkpoint["model"]  # type: ignore[index]
    merged = {}
    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            merged[key] = target_tensor
            continue
        source_tensor = source_tensor.to(dtype=target_tensor.dtype)
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            merged[key] = source_tensor
            continue
        blended = target_tensor.clone()
        slices = tuple(slice(0, min(s, t)) for s, t in zip(source_tensor.shape, target_tensor.shape))
        blended[slices] = source_tensor[slices]
        merged[key] = blended
    model.load_state_dict(merged, strict=False)
    optimizer = make_layer_optimizer(model, ww.BASE_LR)
    if load_optimizer:
        try:
            optimizer.load_state_dict(ww.tensor_tree_to_cpu(checkpoint["optimizer"]))  # type: ignore[index]
            ww.optimizer_to_device(optimizer, ww.DEVICE)
        except Exception:
            pass
    return model, optimizer


def expansion_compat_scale_for_step(step: int, total_steps: int) -> float:
    if step >= int(total_steps * 0.80):
        return EXPANSION_COMPAT_LATE_SCALE
    if step >= int(total_steps * 0.50):
        return EXPANSION_COMPAT_MID_SCALE
    return EXPANSION_COMPAT_EARLY_SCALE


def expansion_gate_floor_for_step(step: int, total_steps: int) -> float:
    if step >= int(total_steps * 0.80):
        return EXPANSION_GATE_LATE_LOGIT
    if step >= int(total_steps * 0.50):
        return EXPANSION_GATE_MID_LOGIT
    return EXPANSION_GATE_INIT_LOGIT


def prime_new_block_from_last_base(model: LayerwiseTinyGPT) -> None:
    if EXTRA_BLOCKS <= 0:
        return
    if len(model.blocks) <= EXTRA_BLOCKS:
        return
    source_block = model.blocks[len(model.blocks) - EXTRA_BLOCKS - 1]
    source_state = source_block.state_dict()
    for block in model.blocks[-EXTRA_BLOCKS:]:
        if not isinstance(block, GatedExpansionBlock):
            continue
        with torch.no_grad():
            block.inner.load_state_dict(source_state, strict=False)
            block.gate_logit.fill_(EXPANSION_GATE_INIT_LOGIT)


def build_expanded_identity_checkpoint(vocab_size: int, checkpoint: Dict[str, object]) -> Dict[str, object]:
    model, optimizer = restore_layer_checkpoint(
        vocab_size,
        {"model": checkpoint["model"], "optimizer": checkpoint["optimizer"], "n_layer_override": ww.N_LAYER + EXTRA_BLOCKS},
        load_optimizer=False,
    )
    prime_new_block_from_last_base(model)
    output = make_layer_checkpoint(model, optimizer, len(model.blocks))
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def set_expanded_probe_mode(model: LayerwiseTinyGPT) -> None:
    model.set_adapters_enabled(False)
    model.clear_latent_free_projectors()
    model.eval()


def mean_new_block_gate(model: LayerwiseTinyGPT) -> float:
    values = [float(block.gate().item()) for block in model.blocks[-EXTRA_BLOCKS:] if isinstance(block, GatedExpansionBlock)]
    if not values:
        return 1.0
    return float(sum(values) / len(values))


def enforce_new_block_gate_floor(model: LayerwiseTinyGPT, min_logit: float) -> None:
    with torch.no_grad():
        for block in model.blocks[-EXTRA_BLOCKS:]:
            if isinstance(block, GatedExpansionBlock):
                floor = torch.full_like(block.gate_logit, min_logit)
                block.gate_logit.copy_(torch.maximum(block.gate_logit, floor))


def forward_with_block_outputs(
    model,
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


def restore_external_teacher(
    vocab_size: int,
    checkpoint: Dict[str, object],
):
    model_kind = str(checkpoint.get("model_kind", "layer"))
    if model_kind == "propagation_expansion":
        prop = load_local_prop_module()

        model, optimizer = prop.restore_prop_checkpoint(vocab_size, checkpoint, load_optimizer=False)
        prop.set_prop_probe_mode(model)
        ww.set_requires_grad(model.parameters(), False)
        return model, optimizer, prop.forward_prop_with_block_outputs

    model, optimizer = restore_layer_checkpoint(vocab_size, checkpoint, load_optimizer=False)
    set_expanded_probe_mode(model)
    ww.set_requires_grad(model.parameters(), False)
    return model, optimizer, forward_with_block_outputs


def choose_expansion_teacher_candidate(
    raw_stage: phase.StageResult,
    safe_stage: phase.StageResult,
    retention_reference: Dict[str, float],
) -> phase.StageResult:
    raw_prob = float(raw_stage.metrics.get("arith_problem_acc", 0.0))
    safe_prob = float(safe_stage.metrics.get("arith_problem_acc", 0.0))
    raw_bracket_floor = float(retention_reference.get("bracket_seq", 0.0)) - (EXPANSION_SHARPEN_BRACKET_SLACK * 2.0)
    raw_text_ceiling = float(retention_reference.get("text_loss", float("inf"))) + (EXPANSION_SHARPEN_TEXT_SLACK * 2.0)
    raw_ok = (
        float(raw_stage.metrics.get("bracket_seq", 0.0)) >= raw_bracket_floor
        and float(raw_stage.metrics.get("text_loss", float("inf"))) <= raw_text_ceiling
    )
    if raw_ok and raw_prob >= safe_prob + 0.01:
        return raw_stage
    if safe_prob >= raw_prob - 0.01:
        return safe_stage
    return raw_stage


def expanded_attach_new_block_teacher(
    vocab_size: int,
    expanded_checkpoint: Dict[str, object],
    old_checkpoint: Dict[str, object],
    old_tasks: Sequence[phase.TaskSpec],
    new_task: phase.TaskSpec,
    eval_tasks: Sequence[phase.TaskSpec],
) -> phase.StageResult:
    model, _optimizer = restore_layer_checkpoint(vocab_size, expanded_checkpoint, load_optimizer=False)
    model.set_adapters_enabled(False)
    model.clear_latent_free_projectors()
    ww.set_requires_grad(model_all_base_params(model), False)
    ww.set_requires_grad(model_all_adapter_params(model), False)
    ww.set_requires_grad(model_new_block_params(model), True)
    optimizer = make_layer_optimizer(model, EXPANSION_NEW_BLOCK_LR, params=model_new_block_params(model))

    old_teacher, _old_teacher_opt = phase.restore_phase_checkpoint(vocab_size, old_checkpoint, load_optimizer=False)
    phase.set_model_base_only(old_teacher)
    ww.set_requires_grad(old_teacher.parameters(), False)

    final_metrics = phase.evaluate_world(model, eval_tasks)
    retention_reference = dict(final_metrics)
    best_metrics = dict(final_metrics)
    best_checkpoint = make_layer_checkpoint(model, optimizer, len(model.blocks))
    best_prob = float(best_metrics.get("arith_problem_acc", 0.0))
    best_acc = float(best_metrics.get("arith_acc", 0.0))

    print(f"[expand:C_block] frozen-old + new-block attach for {EXPANSION_ATTACH_STEPS} steps")
    for step in range(1, EXPANSION_ATTACH_STEPS + 1):
        batch = new_task.sample_train_batch(step)
        compat_task = phase.pick_old_retention_task(old_tasks, step - 1, final_metrics, retention_reference)
        compat_batch = compat_task.sample_anchor_batch(400_000 + step)
        model.train()
        model.set_adapters_enabled(False)
        model.clear_latent_free_projectors()
        enforce_new_block_gate_floor(model, expansion_gate_floor_for_step(step, EXPANSION_ATTACH_STEPS))
        optimizer.zero_grad(set_to_none=True)

        new_logits, _ = model(batch.x, batch.y)
        loss = phase.task_loss_from_logits(new_task, new_logits, batch)

        with torch.no_grad():
            teacher_logits, _teacher_loss, teacher_states = forward_with_block_outputs(
                old_teacher, compat_batch, detach=True
            )
        compat_logits, _compat_loss, compat_states = forward_with_block_outputs(model, compat_batch, detach=False)
        compat_task_loss = phase.task_loss_from_logits(compat_task, compat_logits, compat_batch)
        compat_task_w, compat_kl_w, compat_hidden_w = phase.compat_weights_for_step(step, EXPANSION_ATTACH_STEPS)
        compat_scale = expansion_compat_scale_for_step(step, EXPANSION_ATTACH_STEPS)
        loss = (
            loss
            + compat_scale * compat_task_w * compat_task_loss
            + compat_scale * compat_kl_w * lateral.distill_kl(compat_logits, teacher_logits)
            + compat_scale * compat_hidden_w * lateral.hidden_lateral_loss(compat_states, teacher_states)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_new_block_params(model), ww.GRAD_CLIP)
        optimizer.step()

        if ww.is_probe_step(step) or step == EXPANSION_ATTACH_STEPS:
            set_expanded_probe_mode(model)
            final_metrics = phase.evaluate_world(model, eval_tasks)
            if step % EXPANSION_ATTACH_LOG_INTERVAL == 0 or step == EXPANSION_ATTACH_STEPS:
                print(
                    f"[expand:C_block] step={step:04d}/{EXPANSION_ATTACH_STEPS} "
                    f"{phase.summarize_metrics(final_metrics)} gate={mean_new_block_gate(model):.4f}"
                )
            if (
                phase.arith_consolidation_candidate_ok(final_metrics, retention_reference)
                and phase.better_arith_candidate(final_metrics, best_metrics)
            ):
                best_metrics = dict(final_metrics)
                best_prob = float(best_metrics.get("arith_problem_acc", 0.0))
                best_acc = float(best_metrics.get("arith_acc", 0.0))
                best_checkpoint = make_layer_checkpoint(model, optimizer, len(model.blocks))
            elif (
                float(final_metrics.get("arith_problem_acc", 0.0)) > best_prob + 1e-12
                or (
                    abs(float(final_metrics.get("arith_problem_acc", 0.0)) - best_prob) <= 1e-12
                    and float(final_metrics.get("arith_acc", 0.0)) > best_acc + 1e-12
                )
            ):
                best_metrics = dict(final_metrics)
                best_prob = float(best_metrics.get("arith_problem_acc", 0.0))
                best_acc = float(best_metrics.get("arith_acc", 0.0))
                best_checkpoint = make_layer_checkpoint(model, optimizer, len(model.blocks))

    del model, _optimizer, optimizer, old_teacher, _old_teacher_opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return phase.StageResult(
        label="expanded_C_teacher",
        checkpoint=best_checkpoint,
        metrics=best_metrics,
        replay_count=0,
        replay_budget=0,
        base_only_verified=True,
    )


def expansion_sharpen_candidate_ok(metrics: Dict[str, float], reference: Dict[str, float]) -> bool:
    bracket_floor = float(reference.get("bracket_seq", 0.0)) - EXPANSION_SHARPEN_BRACKET_SLACK
    text_ceiling = float(reference.get("text_loss", float("inf"))) + EXPANSION_SHARPEN_TEXT_SLACK
    return (
        float(metrics.get("bracket_seq", 0.0)) >= bracket_floor
        and float(metrics.get("text_loss", float("inf"))) <= text_ceiling
    )


def sharpen_expanded_new_block_teacher(
    vocab_size: int,
    initial_stage: phase.StageResult,
    old_checkpoint: Dict[str, object],
    old_tasks: Sequence[phase.TaskSpec],
    new_task: phase.TaskSpec,
    eval_tasks: Sequence[phase.TaskSpec],
    retention_reference: Dict[str, float],
) -> ExpansionTeacherCandidates:
    if not phase.needs_arith_sharpen(initial_stage.metrics):
        stage = phase.StageResult(
            label="expanded_C_teacher_safe",
            checkpoint=initial_stage.checkpoint,
            metrics=dict(initial_stage.metrics),
            replay_count=0,
            replay_budget=0,
            base_only_verified=True,
        )
        return ExpansionTeacherCandidates(raw=stage, safe=stage, selected=stage)

    model, _optimizer = restore_layer_checkpoint(vocab_size, initial_stage.checkpoint, load_optimizer=False)
    model.set_adapters_enabled(False)
    model.clear_latent_free_projectors()
    ww.set_requires_grad(model_all_base_params(model), False)
    ww.set_requires_grad(model_all_adapter_params(model), False)
    ww.set_requires_grad(model_new_block_params(model), True)
    optimizer = make_layer_optimizer(
        model,
        EXPANSION_NEW_BLOCK_LR * EXPANSION_SHARPEN_LR_SCALE,
        params=model_new_block_params(model),
    )

    old_teacher, _old_teacher_opt = phase.restore_phase_checkpoint(vocab_size, old_checkpoint, load_optimizer=False)
    phase.set_model_base_only(old_teacher)
    ww.set_requires_grad(old_teacher.parameters(), False)

    final_metrics = phase.evaluate_world(model, eval_tasks)
    raw_best_metrics = dict(initial_stage.metrics)
    raw_best_checkpoint = initial_stage.checkpoint
    safe_best_metrics = dict(initial_stage.metrics)
    safe_best_checkpoint = initial_stage.checkpoint
    print(f"[sharpen:C_block] new-block reversal sharpening for {EXPANSION_SHARPEN_STEPS} steps")

    for step in range(1, EXPANSION_SHARPEN_STEPS + 1):
        batch = new_task.sample_train_batch(600_000 + step)
        use_compat = step % EXPANSION_SHARPEN_COMPAT_PERIOD == 0
        compat_task = phase.pick_old_retention_task(old_tasks, step - 1, final_metrics, retention_reference)
        compat_batch = compat_task.sample_anchor_batch(700_000 + step)

        model.train()
        model.set_adapters_enabled(False)
        model.clear_latent_free_projectors()
        enforce_new_block_gate_floor(model, EXPANSION_SHARPEN_GATE_LOGIT)
        optimizer.zero_grad(set_to_none=True)

        new_logits, _ = model(batch.x, batch.y)
        loss = phase.task_loss_from_logits(new_task, new_logits, batch)

        if use_compat:
            with torch.no_grad():
                teacher_logits, _teacher_loss, teacher_states = forward_with_block_outputs(
                    old_teacher, compat_batch, detach=True
                )
            compat_logits, _compat_loss, compat_states = forward_with_block_outputs(model, compat_batch, detach=False)
            compat_task_loss = phase.task_loss_from_logits(compat_task, compat_logits, compat_batch)
            loss = (
                loss
                + EXPANSION_SHARPEN_COMPAT_SCALE * compat_task_loss
                + (EXPANSION_SHARPEN_COMPAT_SCALE * 0.75) * lateral.distill_kl(compat_logits, teacher_logits)
                + (EXPANSION_SHARPEN_COMPAT_SCALE * 0.45) * lateral.hidden_lateral_loss(compat_states, teacher_states)
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_new_block_params(model), ww.GRAD_CLIP)
        optimizer.step()

        if ww.is_probe_step(step) or step == EXPANSION_SHARPEN_STEPS:
            set_expanded_probe_mode(model)
            final_metrics = phase.evaluate_world(model, eval_tasks)
            if step % EXPANSION_ATTACH_LOG_INTERVAL == 0 or step == EXPANSION_SHARPEN_STEPS:
                print(
                    f"[sharpen:C_block] step={step:04d}/{EXPANSION_SHARPEN_STEPS} "
                    f"{phase.summarize_metrics(final_metrics)} gate={mean_new_block_gate(model):.4f}"
                )
            if (
                expansion_sharpen_candidate_ok(final_metrics, retention_reference)
                and phase.better_arith_candidate(final_metrics, safe_best_metrics)
            ):
                safe_best_metrics = dict(final_metrics)
                safe_best_checkpoint = make_layer_checkpoint(model, optimizer, len(model.blocks))
            if phase.better_arith_candidate(final_metrics, raw_best_metrics):
                raw_best_metrics = dict(final_metrics)
                raw_best_checkpoint = make_layer_checkpoint(model, optimizer, len(model.blocks))

    del model, _optimizer, optimizer, old_teacher, _old_teacher_opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    raw_stage = phase.StageResult(
        label="expanded_C_teacher_raw",
        checkpoint=raw_best_checkpoint,
        metrics=raw_best_metrics,
        replay_count=0,
        replay_budget=0,
        base_only_verified=True,
    )
    safe_stage = phase.StageResult(
        label="expanded_C_teacher_safe",
        checkpoint=safe_best_checkpoint,
        metrics=safe_best_metrics,
        replay_count=0,
        replay_budget=0,
        base_only_verified=True,
    )
    return ExpansionTeacherCandidates(
        raw=raw_stage,
        safe=safe_stage,
        selected=choose_expansion_teacher_candidate(raw_stage, safe_stage, retention_reference),
    )


def consolidate_expanded_student(
    vocab_size: int,
    expanded_student_checkpoint: Dict[str, object],
    old_teacher_checkpoint: Dict[str, object],
    expanded_teacher_checkpoint: Dict[str, object],
    old_tasks: Sequence[phase.TaskSpec],
    new_task: phase.TaskSpec,
    eval_tasks: Sequence[phase.TaskSpec],
    selection_reference: Dict[str, float],
) -> phase.StageResult:
    new_teacher, _new_teacher_opt, new_teacher_forward = restore_external_teacher(vocab_size, expanded_teacher_checkpoint)

    old_teacher, _old_teacher_opt = phase.restore_phase_checkpoint(vocab_size, old_teacher_checkpoint, load_optimizer=False)
    phase.set_model_base_only(old_teacher)
    ww.set_requires_grad(old_teacher.parameters(), False)

    student, _student_opt = restore_layer_checkpoint(vocab_size, expanded_student_checkpoint, load_optimizer=False)
    student.set_adapters_enabled(False)
    student.clear_latent_free_projectors()
    ww.set_requires_grad(model_all_adapter_params(student), False)
    ww.set_requires_grad(model_all_base_params(student), True)
    optimizer = make_layer_optimizer(student, lateral.CONSOLIDATION_LR)

    old_batch_budget = phase.consolidation_expected_old_batches(EXPANSION_BRANCH)
    old_batch_count = 0
    final_metrics = phase.evaluate_world(student, eval_tasks)
    best_metrics: Dict[str, float] | None = None
    best_checkpoint: Dict[str, object] | None = None
    print(f"[expand_consolidate:C_block] branch={EXPANSION_BRANCH} for {lateral.CONSOLIDATION_STEPS} steps")

    for step in range(1, lateral.CONSOLIDATION_STEPS + 1):
        old_step = phase.consolidation_old_batch_schedule(EXPANSION_BRANCH, step)
        if old_step:
            current_task = phase.pick_old_retention_task(old_tasks, old_batch_count, final_metrics, selection_reference)
            batch = current_task.sample_anchor_batch(old_batch_count)
            teacher_model = old_teacher
            old_batch_count += 1
        else:
            current_task = new_task
            batch_sampler = new_task.sample_consolidation_batch or new_task.sample_train_batch
            batch = batch_sampler(step)
            teacher_model = new_teacher
            teacher_forward = new_teacher_forward

        if old_step:
            teacher_forward = forward_with_block_outputs
        with torch.no_grad():
            teacher_logits, _teacher_loss, teacher_states = teacher_forward(teacher_model, batch, detach=True)
        optimizer.zero_grad(set_to_none=True)
        student_logits, _student_loss, student_states = forward_with_block_outputs(student, batch, detach=False)
        task_loss = phase.task_loss_from_logits(current_task, student_logits, batch)
        task_weight, kl_weight, hidden_weight = phase.consolidation_weights_for_step(EXPANSION_BRANCH, old_step, step)
        loss = (
            task_weight * task_loss
            + kl_weight * lateral.distill_kl(student_logits, teacher_logits)
            + hidden_weight * lateral.hidden_lateral_loss(student_states, teacher_states)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_all_base_params(student), ww.GRAD_CLIP)
        optimizer.step()

        if step % lateral.CONSOLIDATION_EVAL_INTERVAL == 0 or step == lateral.CONSOLIDATION_STEPS:
            set_expanded_probe_mode(student)
            final_metrics = phase.evaluate_world(student, eval_tasks)
            if step % lateral.CONSOLIDATION_LOG_INTERVAL == 0 or step == lateral.CONSOLIDATION_STEPS:
                print(
                    f"[expand_consolidate:C_block] step={step:04d}/{lateral.CONSOLIDATION_STEPS} "
                    f"{phase.summarize_metrics(final_metrics)} old_batches={old_batch_count}/{old_batch_budget}"
                )
            if (
                phase.arith_consolidation_candidate_ok(final_metrics, selection_reference)
                and phase.better_arith_candidate(final_metrics, best_metrics)
            ):
                best_metrics = dict(final_metrics)
                best_checkpoint = make_layer_checkpoint(student, optimizer, len(student.blocks))

    set_expanded_probe_mode(student)
    final_metrics = phase.evaluate_world(student, eval_tasks)
    if (
        phase.arith_consolidation_candidate_ok(final_metrics, selection_reference)
        and phase.better_arith_candidate(final_metrics, best_metrics)
    ):
        best_metrics = dict(final_metrics)
        best_checkpoint = make_layer_checkpoint(student, optimizer, len(student.blocks))
    checkpoint = best_checkpoint if best_checkpoint is not None else make_layer_checkpoint(student, optimizer, len(student.blocks))
    if best_metrics is not None:
        final_metrics = best_metrics

    del new_teacher, _new_teacher_opt, old_teacher, _old_teacher_opt, student, _student_opt, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return phase.StageResult(
        label="expanded_base_abc",
        checkpoint=checkpoint,
        metrics=final_metrics,
        old_batch_count=old_batch_count,
        old_batch_budget=old_batch_budget,
        base_only_verified=True,
    )


def polish_expanded_transfer(
    vocab_size: int,
    initial_stage: phase.StageResult,
    old_teacher_checkpoint: Dict[str, object],
    expanded_teacher_checkpoint: Dict[str, object],
    old_tasks: Sequence[phase.TaskSpec],
    new_task: phase.TaskSpec,
    eval_tasks: Sequence[phase.TaskSpec],
    teacher_reference: Dict[str, float],
    retention_reference: Dict[str, float],
) -> phase.StageResult:
    if not phase.needs_arith_transfer_polish(initial_stage.metrics, teacher_reference, retention_reference):
        return phase.StageResult(
            label="expanded_base_abc",
            checkpoint=initial_stage.checkpoint,
            metrics=dict(initial_stage.metrics),
            old_batch_count=0,
            old_batch_budget=0,
            base_only_verified=True,
        )

    new_teacher, _new_teacher_opt, new_teacher_forward = restore_external_teacher(vocab_size, expanded_teacher_checkpoint)

    old_teacher, _old_teacher_opt = phase.restore_phase_checkpoint(vocab_size, old_teacher_checkpoint, load_optimizer=False)
    phase.set_model_base_only(old_teacher)
    ww.set_requires_grad(old_teacher.parameters(), False)

    student, _student_opt = restore_layer_checkpoint(vocab_size, initial_stage.checkpoint, load_optimizer=False)
    set_expanded_probe_mode(student)
    ww.set_requires_grad(model_all_adapter_params(student), False)
    ww.set_requires_grad(model_all_base_params(student), True)
    optimizer = make_layer_optimizer(student, lateral.CONSOLIDATION_LR * phase.ARITH_TRANSFER_POLISH_LR_SCALE)

    old_period = (
        phase.ARITH_TRANSFER_RETENTION_OLD_PERIOD
        if phase.retention_is_weak(initial_stage.metrics, retention_reference)
        else phase.ARITH_TRANSFER_POLISH_OLD_PERIOD
    )
    old_batch_budget = sum(1 for step in range(1, EXPANSION_TRANSFER_POLISH_STEPS + 1) if step % old_period == 0)
    old_batch_count = 0
    final_metrics = phase.evaluate_world(student, eval_tasks)
    best_metrics = dict(initial_stage.metrics)
    best_checkpoint = initial_stage.checkpoint

    print(f"[expand_transfer:C_block] base-only transfer polish for {EXPANSION_TRANSFER_POLISH_STEPS} steps")
    for step in range(1, EXPANSION_TRANSFER_POLISH_STEPS + 1):
        old_step = step % old_period == 0
        if old_step:
            current_task = phase.pick_old_retention_task(old_tasks, old_batch_count, final_metrics, retention_reference)
            batch = current_task.sample_anchor_batch(800_000 + old_batch_count)
            teacher_model = old_teacher
            old_batch_count += 1
            boost = (
                phase.ARITH_TRANSFER_POLISH_RETENTION_BOOST
                if phase.retention_is_weak(final_metrics, retention_reference)
                else 1.0
            )
            task_weight = phase.ARITH_TRANSFER_POLISH_OLD_TASK_WEIGHT * boost
            kl_weight = phase.ARITH_TRANSFER_POLISH_OLD_KL_WEIGHT * boost
            hidden_weight = phase.ARITH_TRANSFER_POLISH_OLD_HIDDEN_WEIGHT * boost
        else:
            current_task = new_task
            batch_sampler = new_task.sample_consolidation_batch or new_task.sample_train_batch
            batch = batch_sampler(800_000 + step)
            teacher_model = new_teacher
            teacher_forward = new_teacher_forward
            task_weight = phase.ARITH_TRANSFER_POLISH_TASK_WEIGHT
            kl_weight = phase.ARITH_TRANSFER_POLISH_KL_WEIGHT
            hidden_weight = phase.ARITH_TRANSFER_POLISH_HIDDEN_WEIGHT

        if old_step:
            teacher_forward = forward_with_block_outputs
        with torch.no_grad():
            teacher_logits, _teacher_loss, teacher_states = teacher_forward(teacher_model, batch, detach=True)
        optimizer.zero_grad(set_to_none=True)
        student_logits, _student_loss, student_states = forward_with_block_outputs(student, batch, detach=False)
        task_loss = phase.task_loss_from_logits(current_task, student_logits, batch)
        loss = (
            task_weight * task_loss
            + kl_weight * lateral.distill_kl(student_logits, teacher_logits)
            + hidden_weight * lateral.hidden_lateral_loss(student_states, teacher_states)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_all_base_params(student), ww.GRAD_CLIP)
        optimizer.step()

        if step % lateral.CONSOLIDATION_EVAL_INTERVAL == 0 or step == EXPANSION_TRANSFER_POLISH_STEPS:
            set_expanded_probe_mode(student)
            final_metrics = phase.evaluate_world(student, eval_tasks)
            if step % lateral.CONSOLIDATION_LOG_INTERVAL == 0 or step == EXPANSION_TRANSFER_POLISH_STEPS:
                print(
                    f"[expand_transfer:C_block] step={step:04d}/{EXPANSION_TRANSFER_POLISH_STEPS} "
                    f"{phase.summarize_metrics(final_metrics)} old_batches={old_batch_count}/{old_batch_budget}"
                )
            if (
                phase.arith_transfer_candidate_ok(final_metrics, retention_reference)
                and phase.better_arith_candidate(final_metrics, best_metrics)
            ):
                best_metrics = dict(final_metrics)
                best_checkpoint = make_layer_checkpoint(student, optimizer, len(student.blocks))

    set_expanded_probe_mode(student)
    final_metrics = phase.evaluate_world(student, eval_tasks)
    if (
        phase.arith_transfer_candidate_ok(final_metrics, retention_reference)
        and phase.better_arith_candidate(final_metrics, best_metrics)
    ):
        best_metrics = dict(final_metrics)
        best_checkpoint = make_layer_checkpoint(student, optimizer, len(student.blocks))

    del new_teacher, _new_teacher_opt, old_teacher, _old_teacher_opt, student, _student_opt, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return phase.StageResult(
        label="expanded_base_abc",
        checkpoint=best_checkpoint,
        metrics=best_metrics,
        old_batch_count=old_batch_count,
        old_batch_budget=old_batch_budget,
        base_only_verified=True,
    )


def print_seed_summary(
    seed: int,
    anchor_metrics: Dict[str, float],
    base_ab: phase.StageResult,
    expanded_attach: phase.StageResult,
    expanded_teacher_candidates: ExpansionTeacherCandidates,
    expanded_base_consolidated: phase.StageResult,
    expanded_base: phase.StageResult,
) -> None:
    print("\n" + "=" * 78)
    print(f"SEED {seed} LAYER-EXPANSION RESULT")
    print("=" * 78)
    print(
        f"{'stage':28s} {'bracket_seq':>11s} {'text_loss':>10s} {'rev_acc':>10s} {'rev_prob':>10s} {'rev_seq':>10s}"
    )
    for label, metrics in [
        ("base_A_anchor", anchor_metrics),
        ("base_AB_unified", base_ab.metrics),
        ("expanded_C_attach", expanded_attach.metrics),
        ("expanded_C_teacher_raw", expanded_teacher_candidates.raw.metrics),
        ("expanded_C_teacher_safe", expanded_teacher_candidates.safe.metrics),
        ("expanded_C_teacher_used", expanded_teacher_candidates.selected.metrics),
        ("expanded_base_abc_consolidated", expanded_base_consolidated.metrics),
        ("expanded_base_abc", expanded_base.metrics),
    ]:
        print(
            f"{label:28s} "
            f"{metrics.get('bracket_seq', float('nan')):11.3f} "
            f"{metrics.get('text_loss', float('nan')):10.3f} "
            f"{metrics.get('arith_acc', float('nan')):10.3f} "
            f"{metrics.get('arith_problem_acc', float('nan')):10.3f} "
            f"{metrics.get('arith_seq', float('nan')):10.3f}"
        )
    teacher_delta_label = "Expansion teacher delta" if USE_PROPAGATION_ATTACH else "Expansion sharpen delta"
    print(
        f"{teacher_delta_label}: "
        f"rev_prob={expanded_teacher_candidates.raw.metrics.get('arith_problem_acc', 0.0) - expanded_attach.metrics.get('arith_problem_acc', 0.0):+.3f} "
        f"rev_acc={expanded_teacher_candidates.raw.metrics.get('arith_acc', 0.0) - expanded_attach.metrics.get('arith_acc', 0.0):+.3f} "
        f"bracket={expanded_teacher_candidates.raw.metrics.get('bracket_seq', 0.0) - expanded_attach.metrics.get('bracket_seq', 0.0):+.3f}"
    )
    print(
        "Expansion safety gap: "
        f"rev_prob={expanded_teacher_candidates.safe.metrics.get('arith_problem_acc', 0.0) - expanded_teacher_candidates.raw.metrics.get('arith_problem_acc', 0.0):+.3f} "
        f"rev_acc={expanded_teacher_candidates.safe.metrics.get('arith_acc', 0.0) - expanded_teacher_candidates.raw.metrics.get('arith_acc', 0.0):+.3f} "
        f"bracket={expanded_teacher_candidates.safe.metrics.get('bracket_seq', 0.0) - expanded_teacher_candidates.raw.metrics.get('bracket_seq', 0.0):+.3f}"
    )
    print(
        "Expansion consolidation delta: "
        f"rev_prob={expanded_base.metrics.get('arith_problem_acc', 0.0) - expanded_teacher_candidates.selected.metrics.get('arith_problem_acc', 0.0):+.3f} "
        f"rev_acc={expanded_base.metrics.get('arith_acc', 0.0) - expanded_teacher_candidates.selected.metrics.get('arith_acc', 0.0):+.3f} "
        f"bracket={expanded_base.metrics.get('bracket_seq', 0.0) - expanded_teacher_candidates.selected.metrics.get('bracket_seq', 0.0):+.3f}"
    )
    print(
        "Expansion polish delta: "
        f"rev_prob={expanded_base.metrics.get('arith_problem_acc', 0.0) - expanded_base_consolidated.metrics.get('arith_problem_acc', 0.0):+.3f} "
        f"rev_acc={expanded_base.metrics.get('arith_acc', 0.0) - expanded_base_consolidated.metrics.get('arith_acc', 0.0):+.3f} "
        f"bracket={expanded_base.metrics.get('bracket_seq', 0.0) - expanded_base_consolidated.metrics.get('bracket_seq', 0.0):+.3f}"
    )
    print("=" * 78)


def main() -> None:
    prop = load_local_prop_module()

    ww.set_seed(EXPANSION_SEEDS[0])
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 78)
    print("LAYER-EXPANSION LATERAL LAB")
    print("=" * 78)
    print("Question: can a frozen base_AB append one new block, learn a new skill, and laterally unify into an expanded base-only model?")
    print(f"Device: {ww.DEVICE}")
    print(f"Seeds: {EXPANSION_SEEDS}")
    print(f"Phase A: bracket until seq>={ww.OLD_READY_SEQ:.2f} or {ww.PHASE_A_MAX_STEPS} steps")
    print(f"Phase B: latent text attach for {ww.PHASE_B_STEPS} steps")
    print(f"Phase C: dual-teacher consolidation for {lateral.CONSOLIDATION_STEPS} steps")
    if USE_PROPAGATION_ATTACH:
        print(f"Phase D: recurrent lateral-propagation attach for {prop.PROP_ATTACH_STEPS} steps")
        print("Phase D2: propagation teacher candidate selection (raw/safe)")
    else:
        print(f"Phase D: extra-block reversal attach for {EXPANSION_ATTACH_STEPS} steps")
        print(f"Phase D2: new-block reversal sharpen for {EXPANSION_SHARPEN_STEPS} steps")
    print(f"Phase E: expanded dual-teacher consolidation for {lateral.CONSOLIDATION_STEPS} steps")
    print(f"Phase E2: expanded base-only transfer polish for {EXPANSION_TRANSFER_POLISH_STEPS} steps")
    print(
        f"Model: d={ww.D_MODEL}, base_layers={ww.N_LAYER}, extra_layers={EXTRA_BLOCKS}, "
        f"block={ww.BLOCK_SIZE}, adapter_rank={ww.ADAPTER_RANK}"
    )
    print(
        f"Expansion: new_block_lr={EXPANSION_NEW_BLOCK_LR:.2e}, gate_init={torch.sigmoid(torch.tensor(EXPANSION_GATE_INIT_LOGIT)).item():.4f}, "
        f"gate_floor=({torch.sigmoid(torch.tensor(EXPANSION_GATE_MID_LOGIT)).item():.4f},{torch.sigmoid(torch.tensor(EXPANSION_GATE_LATE_LOGIT)).item():.4f}), "
        f"compat_scale=({EXPANSION_COMPAT_EARLY_SCALE:.2f},{EXPANSION_COMPAT_MID_SCALE:.2f},{EXPANSION_COMPAT_LATE_SCALE:.2f})"
    )

    text = ww.download_or_load_text()
    stoi, _itos = phase.build_joint_vocab(text)
    encoded = ww.encode(text, stoi)
    split = int(0.95 * len(encoded))
    train_data = encoded[:split]
    val_data = encoded[split:]
    print(f"Vocab size: {len(stoi)}")
    print(f"Train tokens: {train_data.numel():,} | Val tokens: {val_data.numel():,}")

    all_start = time.time()
    for seed in EXPANSION_SEEDS:
        print("\n" + "#" * 78)
        print(f"BEGIN EXPANSION SEED {seed}")
        print("#" * 78)
        ww.set_seed(seed)
        text_eval_positions = phase.make_text_eval_positions(len(val_data), seed)
        bracket_eval_batches = ww.make_fixed_bracket_batches(
            stoi, seed + 30_000, ww.BRACKET_EVAL_BATCHES, ww.BRACKET_EVAL_BATCH
        )
        bracket_probe_batch = ww.replay_batch_for_index(stoi, seed + 40_000, 0, batch_size=ww.BRACKET_EVAL_BATCH)
        arithmetic_eval_batches = phase.make_fixed_arithmetic_batches(
            stoi,
            seed + 50_000,
            ww.BRACKET_EVAL_BATCHES,
            ww.BRACKET_EVAL_BATCH,
            digits=phase.ARITH_EVAL_DIGITS,
        )

        bracket_task = phase.make_bracket_task(stoi, bracket_eval_batches, seed)
        text_task = phase.make_text_task(train_data, val_data, text_eval_positions, seed)
        arith_task = phase.make_arithmetic_task(stoi, arithmetic_eval_batches, seed)

        old_anchor = ww.train_old_skill(len(stoi), stoi, seed, bracket_eval_batches, bracket_probe_batch)
        anchor_a = phase.anchor_from_old_skill(old_anchor)
        anchor_model, anchor_optimizer = phase.restore_phase_checkpoint(len(stoi), old_anchor.checkpoint, load_optimizer=False)
        anchor_text_metrics = phase.evaluate_world(anchor_model, [text_task])
        anchor_metrics = {
            "bracket_seq": old_anchor.old_seq_acc,
            "bracket_close": old_anchor.old_close_acc,
            "bracket_loss": old_anchor.old_loss,
            "text_loss": anchor_text_metrics.get("text_loss", float("nan")),
        }
        del anchor_model, anchor_optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        teacher_b = phase.attach_latent_teacher(
            "B_text",
            anchor_a,
            len(stoi),
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
            len(stoi),
        )

        expanded_identity = build_expanded_identity_checkpoint(len(stoi), base_ab.checkpoint)
        if USE_PROPAGATION_ATTACH:
            prop_identity = prop.build_prop_identity_checkpoint(len(stoi), base_ab.checkpoint)
            prop_candidates = prop.train_propagation_teacher_candidates(
                len(stoi),
                prop_identity,
                base_ab.checkpoint,
                [bracket_task, text_task],
                arith_task,
                [bracket_task, text_task, arith_task],
                base_ab.metrics,
            )
            expanded_attach = prop_candidates.raw
            expanded_teacher = ExpansionTeacherCandidates(
                raw=prop_candidates.raw,
                safe=prop_candidates.safe,
                selected=choose_expansion_teacher_candidate(prop_candidates.raw, prop_candidates.safe, base_ab.metrics),
            )
        else:
            expanded_attach = expanded_attach_new_block_teacher(
                len(stoi),
                expanded_identity,
                base_ab.checkpoint,
                [bracket_task, text_task],
                arith_task,
                [bracket_task, text_task, arith_task],
            )
            expanded_teacher = sharpen_expanded_new_block_teacher(
                len(stoi),
                expanded_attach,
                base_ab.checkpoint,
                [bracket_task, text_task],
                arith_task,
                [bracket_task, text_task, arith_task],
                base_ab.metrics,
            )
        expanded_base = consolidate_expanded_student(
            len(stoi),
            expanded_identity,
            base_ab.checkpoint,
            expanded_teacher.selected.checkpoint,
            [bracket_task, text_task],
            arith_task,
            [bracket_task, text_task, arith_task],
            selection_reference=base_ab.metrics,
        )
        expanded_base_polished = polish_expanded_transfer(
            len(stoi),
            expanded_base,
            base_ab.checkpoint,
            expanded_teacher.selected.checkpoint,
            [bracket_task, text_task],
            arith_task,
            [bracket_task, text_task, arith_task],
            expanded_teacher.selected.metrics,
            base_ab.metrics,
        )

        print_seed_summary(
            seed,
            anchor_metrics,
            base_ab,
            expanded_attach,
            expanded_teacher,
            expanded_base,
            expanded_base_polished,
        )

    print("=" * 78)
    print(f"Total wall time: {phase.format_seconds(time.time() - all_start)}")


if __name__ == "__main__":
    main()
