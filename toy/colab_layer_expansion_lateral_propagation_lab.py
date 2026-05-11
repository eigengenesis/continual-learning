#!/usr/bin/env python3
"""
Lateral-propagation expansion attach test.

Question:
Does a true recurrent lateral-propagation extra block learn the new reversal
skill better than the current gated residual expansion block before
consolidation?
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
LAB_SEEDS = list(phase.LAB_SEEDS[:1])
PROP_ATTACH_STEPS = ww.PHASE_B_STEPS
PROP_ATTACH_LOG_INTERVAL = ww.BRANCH_LOG_INTERVAL
PROP_NEW_BLOCK_LR = ww.BASE_LR * 2.0
PROP_ITERATIONS = 3
PROP_OLD_MSG_INIT = -3.5
PROP_SELF_MSG_INIT = -4.2
PROP_CONSENSUS_INIT = -2.9
PROP_COMPAT_EARLY = 0.06
PROP_COMPAT_MID = 0.12
PROP_COMPAT_LATE = 0.20
PROP_MODEL_KIND = "propagation_expansion"


def load_local_expand_module():
    spec = importlib.util.spec_from_file_location(
        "colab_layer_expansion_lateral_lab_local",
        ROOT / "colab_layer_expansion_lateral_lab.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not load local expansion lab module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class PropTeacherPair:
    raw: phase.StageResult
    safe: phase.StageResult


class LateralPropagationBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = ww.Block()
        self.old_norm = nn.LayerNorm(ww.D_MODEL)
        self.new_norm = nn.LayerNorm(ww.D_MODEL)
        self.old_to_new = nn.Linear(ww.D_MODEL, ww.D_MODEL, bias=False)
        self.new_to_new = nn.Linear(ww.D_MODEL, ww.D_MODEL, bias=False)
        self.old_gate_logit = nn.Parameter(torch.tensor(float(PROP_OLD_MSG_INIT)))
        self.self_gate_logit = nn.Parameter(torch.tensor(float(PROP_SELF_MSG_INIT)))
        self.consensus_logit = nn.Parameter(torch.tensor(float(PROP_CONSENSUS_INIT)))

    def base_parameters(self) -> List[nn.Parameter]:
        return (
            self.inner.base_parameters()
            + list(self.old_norm.parameters())
            + list(self.new_norm.parameters())
            + list(self.old_to_new.parameters())
            + list(self.new_to_new.parameters())
            + [self.old_gate_logit, self.self_gate_logit, self.consensus_logit]
        )

    def old_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.old_gate_logit)

    def self_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.self_gate_logit)

    def consensus_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.consensus_logit)

    def forward(self, old_state: torch.Tensor, new_state: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if new_state is None:
            new_state = old_state
        for _ in range(PROP_ITERATIONS):
            message = (
                self.old_gate().to(device=old_state.device, dtype=old_state.dtype)
                * self.old_to_new(self.old_norm(old_state))
                + self.self_gate().to(device=old_state.device, dtype=old_state.dtype)
                * self.new_to_new(self.new_norm(new_state))
            )
            proposal = new_state + message
            updated = self.inner(proposal)
            new_state = 0.5 * new_state + 0.5 * updated
        alpha = self.consensus_gate().to(device=old_state.device, dtype=old_state.dtype)
        consensus = (1.0 - alpha) * old_state + alpha * new_state
        return consensus, new_state


class PropagationExpansionTinyGPT(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, ww.D_MODEL)
        self.position_embedding = nn.Embedding(ww.BLOCK_SIZE, ww.D_MODEL)
        self.blocks = nn.ModuleList([ww.Block() for _ in range(ww.N_LAYER)])
        self.prop_block = LateralPropagationBlock()
        self.ln_f = nn.LayerNorm(ww.D_MODEL)
        self.head = nn.Linear(ww.D_MODEL, vocab_size, bias=False)

    def set_adapters_enabled(self, enabled: bool) -> None:
        for block in self.blocks:
            block.adapter_enabled = enabled
        self.prop_block.inner.adapter_enabled = enabled

    def clear_latent_free_projectors(self) -> None:
        for block in self.blocks:
            block.latent_free_projector = None
            block.latent_projection_strength = 0.0
        self.prop_block.inner.latent_free_projector = None
        self.prop_block.inner.latent_projection_strength = 0.0

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
                if h.requires_grad:
                    h.retain_grad()
                activations.append(h)
        consensus, new_state = self.prop_block(h)
        if return_activations:
            if consensus.requires_grad:
                consensus.retain_grad()
            activations.append(consensus)
        logits = self.head(self.ln_f(consensus))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if return_activations:
            return logits, loss, activations
        return logits, loss


def prop_trainable_params(model: PropagationExpansionTinyGPT) -> List[nn.Parameter]:
    return list(model.prop_block.base_parameters())


def prop_all_base_params(model: PropagationExpansionTinyGPT) -> List[nn.Parameter]:
    params: List[nn.Parameter] = (
        list(model.token_embedding.parameters())
        + list(model.position_embedding.parameters())
        + list(model.ln_f.parameters())
        + list(model.head.parameters())
    )
    for block in model.blocks:
        params += block.base_parameters()
    params += model.prop_block.base_parameters()
    return params


def make_prop_optimizer(
    model: PropagationExpansionTinyGPT,
    lr: float,
    params: Iterable[nn.Parameter] | None = None,
) -> torch.optim.Optimizer:
    chosen = [param for param in (list(params) if params is not None else prop_all_base_params(model)) if param.requires_grad]
    return torch.optim.AdamW(chosen, lr=lr, betas=ww.BETAS, weight_decay=ww.WEIGHT_DECAY)


def make_prop_checkpoint(model: PropagationExpansionTinyGPT, optimizer: torch.optim.Optimizer) -> Dict[str, object]:
    checkpoint = ww.make_checkpoint(model, optimizer)
    checkpoint["model_kind"] = PROP_MODEL_KIND
    return checkpoint


def restore_prop_checkpoint(
    vocab_size: int,
    checkpoint: Dict[str, object],
    load_optimizer: bool = False,
) -> Tuple[PropagationExpansionTinyGPT, torch.optim.Optimizer]:
    model = PropagationExpansionTinyGPT(vocab_size).to(ww.DEVICE)
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
    optimizer = make_prop_optimizer(model, ww.BASE_LR)
    if load_optimizer:
        try:
            optimizer.load_state_dict(ww.tensor_tree_to_cpu(checkpoint["optimizer"]))  # type: ignore[index]
            ww.optimizer_to_device(optimizer, ww.DEVICE)
        except Exception:
            pass
    return model, optimizer


def prime_prop_block_from_last_base(model: PropagationExpansionTinyGPT) -> None:
    source_state = model.blocks[-1].state_dict()
    with torch.no_grad():
        model.prop_block.inner.load_state_dict(source_state, strict=False)


def build_prop_identity_checkpoint(vocab_size: int, checkpoint: Dict[str, object]) -> Dict[str, object]:
    model, optimizer = restore_prop_checkpoint(vocab_size, checkpoint, load_optimizer=False)
    prime_prop_block_from_last_base(model)
    output = make_prop_checkpoint(model, optimizer)
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def set_prop_probe_mode(model: PropagationExpansionTinyGPT) -> None:
    model.set_adapters_enabled(False)
    model.clear_latent_free_projectors()
    model.eval()


def forward_prop_with_block_outputs(
    model: PropagationExpansionTinyGPT,
    batch: ww.Batch,
    detach: bool,
):
    logits, loss, activations = model(batch.x, batch.y, return_activations=True)
    if detach:
        activations = [activation.detach() for activation in activations]
    return logits, loss, activations


def prop_compat_scale(step: int, total_steps: int) -> float:
    if step >= int(total_steps * 0.80):
        return PROP_COMPAT_LATE
    if step >= int(total_steps * 0.50):
        return PROP_COMPAT_MID
    return PROP_COMPAT_EARLY


def evaluate_prop_world(model: PropagationExpansionTinyGPT, tasks: Sequence[phase.TaskSpec]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for task in tasks:
        metrics.update(task.evaluate(model))
    return metrics


def train_propagation_teacher_candidates(
    vocab_size: int,
    prop_checkpoint: Dict[str, object],
    old_checkpoint: Dict[str, object],
    old_tasks: Sequence[phase.TaskSpec],
    new_task: phase.TaskSpec,
    eval_tasks: Sequence[phase.TaskSpec],
    retention_reference: Dict[str, float],
) -> PropTeacherPair:
    model, _optimizer = restore_prop_checkpoint(vocab_size, prop_checkpoint, load_optimizer=False)
    model.set_adapters_enabled(False)
    model.clear_latent_free_projectors()
    ww.set_requires_grad(prop_all_base_params(model), False)
    ww.set_requires_grad(prop_trainable_params(model), True)
    optimizer = make_prop_optimizer(model, PROP_NEW_BLOCK_LR, params=prop_trainable_params(model))

    old_teacher, _old_teacher_opt = phase.restore_phase_checkpoint(vocab_size, old_checkpoint, load_optimizer=False)
    phase.set_model_base_only(old_teacher)
    ww.set_requires_grad(old_teacher.parameters(), False)

    final_metrics = evaluate_prop_world(model, eval_tasks)
    raw_best_metrics = dict(final_metrics)
    raw_best_checkpoint = make_prop_checkpoint(model, optimizer)
    safe_best_metrics = dict(final_metrics)
    safe_best_checkpoint = raw_best_checkpoint
    print(f"[prop_attach:C_block] lateral-propagation attach for {PROP_ATTACH_STEPS} steps")

    for step in range(1, PROP_ATTACH_STEPS + 1):
        batch = new_task.sample_train_batch(step)
        compat_task = phase.pick_old_retention_task(old_tasks, step - 1, final_metrics, retention_reference)
        compat_batch = compat_task.sample_anchor_batch(500_000 + step)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        new_logits, _ = model(batch.x, batch.y)
        loss = phase.task_loss_from_logits(new_task, new_logits, batch)

        with torch.no_grad():
            teacher_logits, _ = old_teacher(compat_batch.x, compat_batch.y)
        compat_logits, _ = model(compat_batch.x, compat_batch.y)
        compat_scale = prop_compat_scale(step, PROP_ATTACH_STEPS)
        loss = (
            loss
            + compat_scale * phase.task_loss_from_logits(compat_task, compat_logits, compat_batch)
            + (compat_scale * 0.85) * lateral.distill_kl(compat_logits, teacher_logits)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prop_trainable_params(model), ww.GRAD_CLIP)
        optimizer.step()

        if ww.is_probe_step(step) or step == PROP_ATTACH_STEPS:
            set_prop_probe_mode(model)
            final_metrics = evaluate_prop_world(model, eval_tasks)
            if step % PROP_ATTACH_LOG_INTERVAL == 0 or step == PROP_ATTACH_STEPS:
                print(
                    f"[prop_attach:C_block] step={step:04d}/{PROP_ATTACH_STEPS} "
                    f"{phase.summarize_metrics(final_metrics)} "
                    f"old_gate={float(model.prop_block.old_gate().item()):.4f} "
                    f"consensus={float(model.prop_block.consensus_gate().item()):.4f}"
                )
            if (
                phase.arith_consolidation_candidate_ok(final_metrics, retention_reference)
                and phase.better_arith_candidate(final_metrics, safe_best_metrics)
            ):
                safe_best_metrics = dict(final_metrics)
                safe_best_checkpoint = make_prop_checkpoint(model, optimizer)
            if phase.better_arith_candidate(final_metrics, raw_best_metrics):
                raw_best_metrics = dict(final_metrics)
                raw_best_checkpoint = make_prop_checkpoint(model, optimizer)

    del model, _optimizer, optimizer, old_teacher, _old_teacher_opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return PropTeacherPair(
        raw=phase.StageResult(
            label="prop_expanded_C_teacher_raw",
            checkpoint=raw_best_checkpoint,
            metrics=raw_best_metrics,
            replay_count=0,
            replay_budget=0,
            base_only_verified=True,
        ),
        safe=phase.StageResult(
            label="prop_expanded_C_teacher_safe",
            checkpoint=safe_best_checkpoint,
            metrics=safe_best_metrics,
            replay_count=0,
            replay_budget=0,
            base_only_verified=True,
        ),
    )


def train_propagation_teacher(
    vocab_size: int,
    prop_checkpoint: Dict[str, object],
    old_checkpoint: Dict[str, object],
    old_tasks: Sequence[phase.TaskSpec],
    new_task: phase.TaskSpec,
    eval_tasks: Sequence[phase.TaskSpec],
    retention_reference: Dict[str, float],
) -> phase.StageResult:
    return train_propagation_teacher_candidates(
        vocab_size,
        prop_checkpoint,
        old_checkpoint,
        old_tasks,
        new_task,
        eval_tasks,
        retention_reference,
    ).raw


def main() -> None:
    expand = load_local_expand_module()

    ww.set_seed(LAB_SEEDS[0])
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 78)
    print("LAYER-EXPANSION LATERAL PROPAGATION TEST")
    print("=" * 78)
    print("Question: does true recurrent lateral propagation make the extra block learn reversal better before consolidation?")
    print(f"Device: {ww.DEVICE}")
    print(f"Seeds: {LAB_SEEDS}")
    print(f"Attach steps: {PROP_ATTACH_STEPS}")
    print(
        f"Propagation: iterations={PROP_ITERATIONS}, new_block_lr={PROP_NEW_BLOCK_LR:.2e}, "
        f"compat_scale=({PROP_COMPAT_EARLY:.2f},{PROP_COMPAT_MID:.2f},{PROP_COMPAT_LATE:.2f})"
    )

    text = ww.download_or_load_text()
    stoi, _itos = phase.build_joint_vocab(text)
    encoded = ww.encode(text, stoi)
    split = int(0.95 * len(encoded))
    train_data = encoded[:split]
    val_data = encoded[split:]
    print(f"Vocab size: {len(stoi)}")
    print(f"Train tokens: {train_data.numel():,} | Val tokens: {val_data.numel():,}")

    start = time.time()
    for seed in LAB_SEEDS:
        print("\n" + "#" * 78)
        print(f"BEGIN PROPAGATION SEED {seed}")
        print("#" * 78)
        ww.set_seed(seed)
        text_eval_positions = phase.make_text_eval_positions(len(val_data), seed)
        bracket_eval_batches = ww.make_fixed_bracket_batches(
            stoi, seed + 30_000, ww.BRACKET_EVAL_BATCHES, ww.BRACKET_EVAL_BATCH
        )
        bracket_probe_batch = ww.replay_batch_for_index(stoi, seed + 40_000, 0, batch_size=ww.BRACKET_EVAL_BATCH)
        reversal_eval_batches = phase.make_fixed_arithmetic_batches(
            stoi,
            seed + 50_000,
            ww.BRACKET_EVAL_BATCHES,
            ww.BRACKET_EVAL_BATCH,
            digits=phase.ARITH_EVAL_DIGITS,
        )

        bracket_task = phase.make_bracket_task(stoi, bracket_eval_batches, seed)
        text_task = phase.make_text_task(train_data, val_data, text_eval_positions, seed)
        reversal_task = phase.make_arithmetic_task(stoi, reversal_eval_batches, seed)

        old_anchor = ww.train_old_skill(len(stoi), stoi, seed, bracket_eval_batches, bracket_probe_batch)
        anchor_a = phase.anchor_from_old_skill(old_anchor)
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

        expanded_identity = expand.build_expanded_identity_checkpoint(len(stoi), base_ab.checkpoint)
        baseline_attach = expand.expanded_attach_new_block_teacher(
            len(stoi),
            expanded_identity,
            base_ab.checkpoint,
            [bracket_task, text_task],
            reversal_task,
            [bracket_task, text_task, reversal_task],
        )
        baseline_pair = expand.sharpen_expanded_new_block_teacher(
            len(stoi),
            baseline_attach,
            base_ab.checkpoint,
            [bracket_task, text_task],
            reversal_task,
            [bracket_task, text_task, reversal_task],
            base_ab.metrics,
        )
        baseline_teacher = expand.choose_expansion_teacher_candidate(
            baseline_pair.raw, baseline_pair.safe, base_ab.metrics
        )

        prop_identity = build_prop_identity_checkpoint(len(stoi), base_ab.checkpoint)
        prop_pair = train_propagation_teacher_candidates(
            len(stoi),
            prop_identity,
            base_ab.checkpoint,
            [bracket_task, text_task],
            reversal_task,
            [bracket_task, text_task, reversal_task],
            base_ab.metrics,
        )

        print("\n" + "=" * 78)
        print(f"SEED {seed} PROPAGATION COMPARISON")
        print("=" * 78)
        print(f"{'stage':24s} {'bracket_seq':>11s} {'text_loss':>10s} {'rev_acc':>10s} {'rev_prob':>10s}")
        for label, metrics in [
            ("base_AB_unified", base_ab.metrics),
            ("baseline_raw", baseline_pair.raw.metrics),
            ("baseline_safe", baseline_pair.safe.metrics),
            ("baseline_used", baseline_teacher.metrics),
            ("propagation_raw", prop_pair.raw.metrics),
            ("propagation_safe", prop_pair.safe.metrics),
        ]:
            print(
                f"{label:24s} "
                f"{metrics.get('bracket_seq', float('nan')):11.3f} "
                f"{metrics.get('text_loss', float('nan')):10.3f} "
                f"{metrics.get('arith_acc', float('nan')):10.3f} "
                f"{metrics.get('arith_problem_acc', float('nan')):10.3f}"
            )
        print(
            "Propagation gain over baseline: "
            f"rev_prob={prop_pair.raw.metrics.get('arith_problem_acc', 0.0) - baseline_teacher.metrics.get('arith_problem_acc', 0.0):+.3f} "
            f"rev_acc={prop_pair.raw.metrics.get('arith_acc', 0.0) - baseline_teacher.metrics.get('arith_acc', 0.0):+.3f} "
            f"bracket={prop_pair.raw.metrics.get('bracket_seq', 0.0) - baseline_teacher.metrics.get('bracket_seq', 0.0):+.3f}"
        )
        print("=" * 78)

    print("=" * 78)
    print(f"Total wall time: {phase.format_seconds(time.time() - start)}")


if __name__ == "__main__":
    main()
