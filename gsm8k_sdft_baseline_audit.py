#!/usr/bin/env python3
"""GSM8K + SDFT Baseline Comparison for Continual Learning.

This script runs a head-to-head comparison of three D-consolidation strategies:
  1. Naive SFT       — Standard fine-tuning with AdamW, no protection at all.
  2. SDFT Baseline   — Self-Distilled Fine-Tuning (Shenfeld et al., 2026 analogue).
                       The model generates its own D-task traces and trains on them
                       with standard AdamW. No geometric protection.
  3. Amoeba No-Proxy — Full geometric CL: Z-tomography + null-space gradient
                       projection + checkpoint hidden-state anchoring.

Task B = GSM8K math reasoning (the skill to PROTECT).
Task D = Proof-V2 stable sort (the NEW skill to learn).

Usage:
  python gsm8k_sdft_baseline_audit.py --device cuda --dtype bfloat16
  python gsm8k_sdft_baseline_audit.py --device cuda --dtype bfloat16 --smoke

Imports from existing qwen_continual_proof.py and qwen_cl_desiderata_audit.py
without modifying them.
"""
from __future__ import annotations

import argparse
import gc
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── imports from existing codebase ──────────────────────────────────────────────
import qwen_continual_proof as qp
import qwen_cl_desiderata_audit as audit
from standalone_latent_lora_qwen import (
    LatentLoRAConfig,
    attach_latent_lora,
    choose_dtype,
    load_causal_lm,
    load_tokenizer,
)


# ════════════════════════════════════════════════════════════════════════════════
# GSM8K Data Loading & Batching
# ════════════════════════════════════════════════════════════════════════════════

def load_gsm8k(split: str = "train", max_samples: int = 0) -> List[Dict[str, str]]:
    """Load GSM8K dataset via HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets  — required for GSM8K loading.")
    ds = load_dataset("openai/gsm8k", "main", split=split)
    examples = []
    for row in ds:
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        examples.append({"question": question, "answer": answer})
        if max_samples > 0 and len(examples) >= max_samples:
            break
    return examples


def _extract_gsm8k_final_answer(answer_text: str) -> str:
    """Extract the final numeric answer after #### from GSM8K answer strings."""
    match = re.search(r"####\s*(.+)$", answer_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return answer_text.strip().split("\n")[-1].strip()


def make_gsm8k_supervised_pair(example: Dict[str, str]) -> Tuple[str, str]:
    """Create a (prompt, target) pair for supervised GSM8K training."""
    question = example["question"]
    answer = example["answer"]
    prompt = f"Question: {question}\nAnswer: "
    target = answer
    return prompt, target


def make_gsm8k_batch_fn(
    tokenizer,
    device: str,
    cfg: qp.RuntimeConfig,
    seed: int,
    gsm8k_examples: List[Dict[str, str]],
) -> Callable[[int], Dict[str, torch.Tensor]]:
    """Create a batch function that samples random GSM8K examples."""
    def _batch(step: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(seed + 3007 * int(step))
        idxs = rng.integers(0, len(gsm8k_examples), size=cfg.batch_size)
        prompts: List[str] = []
        targets: List[str] = []
        for i in idxs:
            prompt, target = make_gsm8k_supervised_pair(gsm8k_examples[int(i)])
            prompts.append(prompt)
            targets.append(f"{target}{tokenizer.eos_token}")
        return qp._prepare_supervised_batch(tokenizer, prompts, targets, device, cfg.max_seq_len)
    return _batch


# ════════════════════════════════════════════════════════════════════════════════
# GSM8K Evaluation
# ════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_gsm8k(
    model,
    tokenizer,
    device: str,
    examples: List[Dict[str, str]],
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> Dict[str, float]:
    """Evaluate GSM8K accuracy by checking if the model produces the correct final answer."""
    model.eval()
    correct = 0
    total = 0
    token_match = 0
    token_total = 0

    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start:start + batch_size]
        prompts = [f"Question: {ex['question']}\nAnswer: " for ex in batch_examples]
        completions = qp._generate_batch_tokens(
            model, tokenizer, prompts, device, max_new_tokens=max_new_tokens,
        )
        for ex, completion in zip(batch_examples, completions):
            total += 1
            gold_answer = _extract_gsm8k_final_answer(ex["answer"])
            # Check if the gold final answer appears in the completion
            pred_answer = _extract_gsm8k_final_answer(completion)
            if gold_answer.strip() == pred_answer.strip():
                correct += 1
            # Token-level: check overlap of answer tokens
            gold_tokens = set(gold_answer.lower().split())
            pred_tokens = set(pred_answer.lower().split())
            if gold_tokens:
                token_match += len(gold_tokens & pred_tokens)
                token_total += len(gold_tokens)

    model.train()
    return {
        "gsm8k_exact": float(correct) / max(float(total), 1.0),
        "gsm8k_total": float(total),
        "gsm8k_token_overlap": float(token_match) / max(float(token_total), 1.0),
    }


@torch.no_grad()
def evaluate_gsm8k_teacher_forced(
    model,
    tokenizer,
    device: str,
    examples: List[Dict[str, str]],
    batch_size: int = 4,
    max_seq_len: int = 512,
) -> Dict[str, float]:
    """Teacher-forced perplexity and token accuracy on GSM8K."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start:start + batch_size]
        prompts = [f"Question: {ex['question']}\nAnswer: " for ex in batch_examples]
        targets = [f"{ex['answer']}{tokenizer.eos_token}" for ex in batch_examples]
        batch = qp._prepare_supervised_batch(tokenizer, prompts, targets, device, max_seq_len)
        outputs = model(**batch, use_cache=False)
        labels = batch["labels"]
        mask = labels != -100
        n_tokens = int(mask.sum().item())
        if n_tokens > 0:
            total_loss += float(outputs.loss.item()) * n_tokens
            total_tokens += n_tokens
            preds = outputs.logits.argmax(dim=-1)
            # Shift for causal LM: compare preds[..., :-1] with labels[..., 1:]
            shift_preds = preds[..., :-1]
            shift_labels = labels[..., 1:]
            shift_mask = shift_labels != -100
            correct_tokens += int((shift_preds[shift_mask] == shift_labels[shift_mask]).sum().item())

    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "gsm8k_tf_loss": avg_loss,
        "gsm8k_tf_ppl": math.exp(min(avg_loss, 20.0)),
        "gsm8k_tf_token_acc": float(correct_tokens) / max(float(total_tokens), 1.0),
    }


# ════════════════════════════════════════════════════════════════════════════════
# SDFT Consolidation (Self-Distilled Fine-Tuning Baseline)
# ════════════════════════════════════════════════════════════════════════════════

def consolidate_sdft_in_memory(
    *,
    student,
    teacher_new,
    tokenizer,
    new_task_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    steps: int,
    lr: float,
    label: str,
    cfg: qp.RuntimeConfig,
    max_new_tokens: int = 64,
) -> None:
    """SDFT baseline: the model generates its own answers to the new task,
    then trains on them with standard AdamW. No geometric protection, no
    checkpoint anchoring, no gradient projection.

    This is the direct analogue of Shenfeld et al. (2026) SDFT.
    """
    qp._freeze_model(teacher_new)
    # Unfreeze student for standard SFT
    for p in student.parameters():
        p.requires_grad = True
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    if cfg.device.startswith("cuda"):
        qp._configure_gradient_checkpointing(student, cfg.gradient_checkpointing)
    start = time.time()
    student.train()

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        # Step 1: Get a new-task batch from the teacher
        batch = new_task_batch_fn(step)
        input_ids = batch["input_ids"]

        # Step 2: Generate self-distilled traces from the STUDENT model
        # (This is the SDFT core: the model generates its own training data)
        student.eval()
        with torch.no_grad():
            # Use the student to generate completions for the prompts
            # We extract the prompt portion (where labels == -100)
            labels = batch["labels"]
            # Find first non-masked position per sequence
            prompt_texts = []
            for seq_idx in range(input_ids.shape[0]):
                mask = labels[seq_idx] == -100
                prompt_len = int(mask.sum().item())
                prompt_ids = input_ids[seq_idx, :prompt_len]
                prompt_texts.append(tokenizer.decode(prompt_ids, skip_special_tokens=False))

            # Generate traces from the student itself
            self_completions = qp._generate_batch_tokens(
                student, tokenizer, prompt_texts, cfg.device,
                max_new_tokens=max_new_tokens,
            )

        # Step 3: Create a supervised batch from self-generated traces
        sdft_targets = [f"{comp}{tokenizer.eos_token}" for comp in self_completions]
        sdft_batch = qp._prepare_supervised_batch(
            tokenizer, prompt_texts, sdft_targets, cfg.device, cfg.max_seq_len,
        )

        # Step 4: Standard SFT on self-generated data — NO protection
        student.train()
        outputs = student(**sdft_batch, use_cache=False)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0 or step == steps:
            print(
                f"[{label}] step={step:04d}/{steps} loss={float(loss.item()):.4f} "
                f"method=sdft_self_generated protection=none",
                flush=True,
            )

    print(f"[{label}] wall_time_sec={time.time() - start:.1f}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════
# Naive SFT Consolidation (No protection at all)
# ════════════════════════════════════════════════════════════════════════════════

def consolidate_naive_sft_in_memory(
    *,
    student,
    teacher_new,
    tokenizer,
    new_task_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    steps: int,
    lr: float,
    label: str,
    cfg: qp.RuntimeConfig,
) -> None:
    """Naive SFT: just train on the new task with standard backprop.
    No Z-tomography, no gradient projection, no anchoring. Pure catastrophic
    forgetting baseline.
    """
    qp._freeze_model(teacher_new)
    for p in student.parameters():
        p.requires_grad = True
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    if cfg.device.startswith("cuda"):
        qp._configure_gradient_checkpointing(student, cfg.gradient_checkpointing)
    start = time.time()
    student.train()

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = new_task_batch_fn(step)
        # Pure supervised loss from teacher labels, no protection at all
        outputs = student(**batch, use_cache=False)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0 or step == steps:
            print(
                f"[{label}] step={step:04d}/{steps} loss={float(loss.item()):.4f} "
                f"method=naive_sft protection=none",
                flush=True,
            )

    print(f"[{label}] wall_time_sec={time.time() - start:.1f}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════
# Formatting & Utilities
# ════════════════════════════════════════════════════════════════════════════════

def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "   nan"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "   nan"
    if math.isnan(value):
        return "   nan"
    return f"{value:.{digits}f}"


def line(char: str = "=") -> None:
    print(char * 96, flush=True)


def sub(title: str) -> None:
    print("-" * 96, flush=True)
    print(title, flush=True)
    print("-" * 96, flush=True)


def release(*models) -> None:
    for m in models:
        if m is not None:
            try:
                m.cpu()
            except Exception:
                pass
            del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _capture_trainable_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _restore_trainable_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    if not state:
        return
    param_map = dict(model.named_parameters())
    for name, value in state.items():
        param = param_map.get(name)
        if param is not None:
            param.data.copy_(value.to(device=param.device, dtype=param.dtype))


def _gsm8k_teacher_score(metrics: Dict[str, float]) -> Tuple[float, ...]:
    """Prefer real GSM8K acquisition over generic retention when restoring B."""
    return (
        float(metrics.get("gsm8k_tf_token_acc", 0.0)),
        -float(metrics.get("gsm8k_tf_loss", 999.0)),
        -float(metrics.get("gsm8k_tf_ppl", 1e9)),
        -float(metrics.get("wikitext_ppl", 999.0)),
    )


def parse_layer_indices(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def train_gsm8k_adapter_in_memory(
    *,
    model,
    tokenizer,
    attached,
    task_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    eval_examples: List[Dict[str, str]],
    wikitext_val: Sequence[str],
    steps: int,
    lr: float,
    label: str,
    cfg: qp.RuntimeConfig,
) -> Dict[str, float]:
    """Train the GSM8K B adapter and restore the checkpoint best on GSM8K TF.

    The generic Qwen helper restores by eval_tasks, which made GSM8K B select
    by WikiText PPL. For this baseline we need the protected B skill to be real,
    so checkpoint selection is driven by GSM8K teacher-forced token accuracy.
    """
    params = qp._trainable_params(model)
    optimizer = torch.optim.AdamW(params, lr=lr)
    start = time.time()
    model.train()
    best_score: Tuple[float, ...] | None = None
    best_step = 0
    best_state: Dict[str, torch.Tensor] | None = None
    best_metrics: Dict[str, float] | None = None
    if cfg.device.startswith("cuda"):
        qp._configure_gradient_checkpointing(model, cfg.gradient_checkpointing)

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = task_batch_fn(step)
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        qp.ESCAPE_SCHEDULE.apply_to_modules(attached, step, steps)

        if step % cfg.log_interval == 0 or step == steps:
            print(
                f"[{label}] step={step:04d}/{steps} loss={float(loss.item()):.4f} "
                f"old_teacher_batches=no",
                flush=True,
            )

        if step % cfg.eval_interval == 0 or step == steps:
            probe_metrics = qp.evaluate_retention(
                model, tokenizer, wikitext_val, cfg.device, cfg.eval_batch_size
            )
            probe_metrics.update(
                evaluate_gsm8k_teacher_forced(
                    model,
                    tokenizer,
                    cfg.device,
                    eval_examples,
                    batch_size=cfg.eval_batch_size,
                    max_seq_len=cfg.max_seq_len,
                )
            )
            score = _gsm8k_teacher_score(probe_metrics)
            if best_score is None or score > best_score:
                best_score = score
                best_step = step
                best_metrics = dict(probe_metrics)
                best_state = _capture_trainable_state(model)
                print(
                    f"[{label}] best_update step={step:04d}/{steps} "
                    f"score={tuple(round(float(v), 4) for v in score)} "
                    f"gsm8k_tf={fmt(probe_metrics.get('gsm8k_tf_token_acc'))} "
                    f"gsm8k_tf_ppl={fmt(probe_metrics.get('gsm8k_tf_ppl'))}",
                    flush=True,
                )

    if best_state is not None:
        _restore_trainable_state(model, best_state)
        print(f"[{label}] restored_best_step={best_step:04d}/{steps}", flush=True)

    metrics = qp.evaluate_retention(model, tokenizer, wikitext_val, cfg.device, cfg.eval_batch_size)
    metrics.update(
        evaluate_gsm8k(model, tokenizer, cfg.device, eval_examples, batch_size=cfg.eval_batch_size)
    )
    metrics.update(
        evaluate_gsm8k_teacher_forced(
            model,
            tokenizer,
            cfg.device,
            eval_examples,
            batch_size=cfg.eval_batch_size,
            max_seq_len=cfg.max_seq_len,
        )
    )
    if best_metrics is not None:
        metrics["teacher_best_step"] = float(best_step)
    print_comparison_row(label, metrics)
    print(
        f"{label:<30} gsm8k_tf_ppl={fmt(metrics.get('gsm8k_tf_ppl')):>7}",
        flush=True,
    )
    print(f"[{label}] wall_time_sec={time.time() - start:.1f}", flush=True)
    return metrics


def print_comparison_row(label: str, metrics: Dict[str, float]) -> None:
    print(
        f"{label:<30} "
        f"ppl={fmt(metrics.get('wikitext_ppl')):>7} "
        f"gsm8k={fmt(metrics.get('gsm8k_exact')):>7} "
        f"gsm8k_tf={fmt(metrics.get('gsm8k_tf_token_acc')):>7} "
        f"sort_tok={fmt(metrics.get('sort_token_acc')):>7} "
        f"sort_tf={fmt(metrics.get('sort_teacher_forced_token_acc')):>7} "
        f"sort_loss={fmt(metrics.get('sort_loss')):>7}",
        flush=True,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Main Audit Pipeline
# ════════════════════════════════════════════════════════════════════════════════

def run() -> None:
    parser = argparse.ArgumentParser(description="GSM8K + SDFT Baseline CL Comparison")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--smoke", action="store_true",
                        help="Fast smoke test with minimal steps/data.")
    parser.add_argument("--skip-stage-a", action="store_true",
                        help="Skip base model eval and use provided known base metrics.")
    parser.add_argument("--base-wikitext-ppl", type=float, default=17.197)
    parser.add_argument("--base-gsm8k-exact", type=float, default=0.0)
    parser.add_argument("--base-gsm8k-tf-token-acc", type=float, default=0.553)
    parser.add_argument("--base-gsm8k-tf-ppl", type=float, default=494.692)
    parser.add_argument("--skip-tomography", action="store_true",
                        help="Skip expensive GSM8K/sort tomography and use fixed Qwen layer selections.")
    parser.add_argument("--b-layer-indices", default="0,1,23,2,3,4,21,5")
    parser.add_argument("--d-layer-indices", default="0,1,2,8,3,23,10,4,5,12")

    # GSM8K settings
    parser.add_argument("--gsm8k-train-samples", type=int, default=512,
                        help="Number of GSM8K training examples to use for Task B.")
    parser.add_argument("--gsm8k-eval-samples", type=int, default=64,
                        help="Number of GSM8K eval examples for retention measurement.")
    parser.add_argument("--abort-if-weak-gsm8k-b", action=argparse.BooleanOptionalAction, default=True,
                        help="Stop before expensive D stages if GSM8K B acquisition is weak.")
    parser.add_argument("--min-b-gsm8k-tf-token-acc", type=float, default=0.62,
                        help="Absolute GSM8K teacher-forced token accuracy gate for B.")
    parser.add_argument("--min-b-gsm8k-tf-improvement", type=float, default=0.05,
                        help="Minimum GSM8K teacher-forced token accuracy improvement over base_A.")

    # Training settings
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable gradient checkpointing. Default off for Qwen-0.5B speed.")
    parser.add_argument("--b-steps", type=int, default=1200,
                        help="Steps for B adapter teacher (GSM8K).")
    parser.add_argument("--b-rank", type=int, default=48)
    parser.add_argument("--b-alpha", type=float, default=96.0)
    parser.add_argument("--b-lr", type=float, default=1e-4)
    parser.add_argument("--b-consol-steps", type=int, default=600)
    parser.add_argument("--d-steps", type=int, default=1200,
                        help="Steps for each D consolidation variant.")
    parser.add_argument("--d-rank", type=int, default=64)
    parser.add_argument("--d-alpha", type=float, default=128.0)
    parser.add_argument("--d-lr", type=float, default=1e-4)
    parser.add_argument("--d-consol-steps", type=int, default=600)
    parser.add_argument("--d-train-max-len", type=int, default=12)
    parser.add_argument("--consolidation-lr", type=float, default=1e-5)

    # Amoeba settings
    parser.add_argument("--no-proxy-old-kl-weight", type=float, default=0.7)
    parser.add_argument("--no-proxy-old-hidden-weight", type=float, default=30.0)
    parser.add_argument("--new-kl-weight", type=float, default=1.0)
    parser.add_argument("--new-hidden-weight", type=float, default=0.5)
    parser.add_argument("--projection-strength", type=float, default=1.0)

    args = parser.parse_args()

    if args.smoke:
        args.gsm8k_train_samples = 32
        args.gsm8k_eval_samples = 8
        args.b_steps = 50
        args.b_consol_steps = 30
        args.d_steps = 50
        args.d_consol_steps = 30
        args.max_seq_len = 256

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    line("=")
    print("GSM8K + SDFT BASELINE CONTINUAL LEARNING COMPARISON", flush=True)
    line("=")
    print(f"model_id={args.model_id}", flush=True)
    print(f"device={args.device} dtype={args.dtype} seed={args.seed}", flush=True)
    print(f"gsm8k_train={args.gsm8k_train_samples} gsm8k_eval={args.gsm8k_eval_samples}", flush=True)
    print(f"B_steps={args.b_steps} D_consol_steps={args.d_consol_steps}", flush=True)
    print(f"variants: naive_sft | sdft_baseline | amoeba_no_proxy", flush=True)
    print(f"skip_stage_a={args.skip_stage_a} skip_tomography={args.skip_tomography}", flush=True)
    print(f"smoke={args.smoke}", flush=True)

    # ── Load model & tokenizer ─────────────────────────────────────────────────
    tokenizer = load_tokenizer(args.model_id, trust_remote_code=True)
    base_model = load_causal_lm(
        args.model_id,
        device=args.device,
        dtype=choose_dtype(args.dtype),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── RuntimeConfig ──────────────────────────────────────────────────────────
    cfg = qp.RuntimeConfig(
        model_id=args.model_id,
        device=args.device,
        dtype=choose_dtype(args.dtype),
        local_files_only=False,
        resume=False,
        smoke=args.smoke,
        output_dir=Path("outputs/gsm8k_sdft"),
        backup_dir=None,
        seed=args.seed,
        phase_scope="gsm8k_sdft_comparison",
        task_suite="proof_v2",
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        consolidation_micro_batch_size=args.micro_batch_size,
        max_seq_len=args.max_seq_len,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        eval_interval=200,
        log_interval=100,
        wikitext_eval_samples=16,
        json_eval_samples=32,
        sort_eval_samples=24,
        reversal_eval_samples=1,
        consolidation_lr=args.consolidation_lr,
        consol_kl_weight=args.new_kl_weight,
        consol_old_kl_weight=0.75,
        consol_hidden_weight=args.new_hidden_weight,
        proof_v2_b_attach_steps=args.b_steps,
        proof_v2_b_attach_lr=args.b_lr,
        proof_v2_b_rank=args.b_rank,
        proof_v2_b_alpha=args.b_alpha,
        proof_v2_b_gate_init=-1.5,
        proof_v2_b_use_up_proj=True,
        proof_v2_b_min_layers=8,
        proof_v2_d_attach_steps=args.d_steps,
        proof_v2_d_attach_lr=args.d_lr,
        proof_v2_d_rank=args.d_rank,
        proof_v2_d_alpha=args.d_alpha,
        proof_v2_d_gate_init=-1.5,
        proof_v2_d_use_up_proj=True,
        proof_v2_d_min_layers=10,
    )
    qp._configure_gradient_checkpointing(base_model, cfg.gradient_checkpointing)

    # ── Load GSM8K data ────────────────────────────────────────────────────────
    sub("Loading GSM8K dataset")
    gsm8k_train = load_gsm8k("train", max_samples=args.gsm8k_train_samples)
    gsm8k_eval = load_gsm8k("test", max_samples=args.gsm8k_eval_samples)
    print(f"gsm8k_train_loaded={len(gsm8k_train)} gsm8k_eval_loaded={len(gsm8k_eval)}", flush=True)

    # ── Load wikitext for retention eval ───────────────────────────────────────
    wikitext_val = qp.load_wikitext_texts(
        tokenizer, split="validation",
        max_seq_len=cfg.max_seq_len, max_samples=cfg.wikitext_eval_samples,
        local_files_only=cfg.local_files_only,
    )
    wikitext_train = qp.load_wikitext_texts(
        tokenizer, split="train",
        max_seq_len=cfg.max_seq_len, max_samples=max(cfg.wikitext_eval_samples, 64),
        local_files_only=cfg.local_files_only,
    )
    eval_data: Dict[str, Any] = {"wikitext_val": wikitext_val}
    proxy_batch_fn = qp.make_wikitext_batch_fn(tokenizer, wikitext_train, cfg.device, cfg, cfg.seed + 11)
    gsm8k_batch_fn = make_gsm8k_batch_fn(tokenizer, cfg.device, cfg, cfg.seed + 101, gsm8k_train)

    # ── Stage A: Base model evaluation ─────────────────────────────────────────
    sub("Stage A: base model evaluation")
    if args.skip_stage_a:
        base_metrics = {
            "wikitext_ppl": float(args.base_wikitext_ppl),
            "gsm8k_exact": float(args.base_gsm8k_exact),
            "gsm8k_tf_token_acc": float(args.base_gsm8k_tf_token_acc),
            "gsm8k_tf_ppl": float(args.base_gsm8k_tf_ppl),
        }
        print("skipped: using provided known base_A metrics", flush=True)
    else:
        base_metrics = qp.evaluate_retention(base_model, tokenizer, wikitext_val, cfg.device, cfg.eval_batch_size)
        base_metrics.update(evaluate_gsm8k(base_model, tokenizer, cfg.device, gsm8k_eval, batch_size=cfg.eval_batch_size))
        base_metrics.update(evaluate_gsm8k_teacher_forced(
            base_model, tokenizer, cfg.device, gsm8k_eval,
            batch_size=cfg.eval_batch_size, max_seq_len=cfg.max_seq_len,
        ))
    print_comparison_row("base_A", base_metrics)
    print(
        f"{'base_A':<30} gsm8k_tf_ppl={fmt(base_metrics.get('gsm8k_tf_ppl')):>7}",
        flush=True,
    )

    # ── Stage B: Train GSM8K adapter teacher ───────────────────────────────────
    sub("Stage B: train GSM8K adapter teacher")
    base_frozen = qp._clone_model(base_model, cfg.device)
    qp._freeze_model(base_frozen)
    a_profile = qp._collect_profiles(base_frozen, "retention", proxy_batch_fn)

    if args.skip_tomography:
        b_layers = parse_layer_indices(args.b_layer_indices)
        print(f"[z_tomography:gsm8k] skipped; using fixed b_layers={b_layers}", flush=True)
    else:
        b_layers, _ = audit.select_layers(
            model=base_model,
            tokenizer=tokenizer,
            task_name="json",  # Use json layer selection heuristic
            task_batch_fn=gsm8k_batch_fn,
            protected_profiles=[a_profile],
            cfg=cfg,
        )
    qp._freeze_model(base_model)
    attached_b = attach_latent_lora(
        base_model,
        suffixes=qp._task_target_suffixes("json", cfg),
        layer_indices=set(b_layers),
        config=LatentLoRAConfig(
            rank=cfg.proof_v2_b_rank,
            alpha=cfg.proof_v2_b_alpha,
            dropout=0.0,
            projection_strength=1.0,
            gate_init=cfg.proof_v2_b_gate_init,
            freeze_base=True,
        ),
    )
    print(f"[gsm8k_lora:B] attached_modules={len(attached_b)}", flush=True)

    teacher_b_metrics = train_gsm8k_adapter_in_memory(
        model=base_model,
        tokenizer=tokenizer,
        attached=attached_b,
        task_batch_fn=gsm8k_batch_fn,
        eval_examples=gsm8k_eval,
        wikitext_val=wikitext_val,
        steps=cfg.proof_v2_b_attach_steps,
        lr=cfg.proof_v2_b_attach_lr,
        label="teacher_B_gsm8k",
        cfg=cfg,
    )
    base_b_tf = float(base_metrics.get("gsm8k_tf_token_acc", 0.0))
    teacher_b_tf = float(teacher_b_metrics.get("gsm8k_tf_token_acc", 0.0))
    teacher_b_exact = float(teacher_b_metrics.get("gsm8k_exact", 0.0))
    b_tf_delta = teacher_b_tf - base_b_tf
    b_gate_pass = (
        teacher_b_tf >= float(args.min_b_gsm8k_tf_token_acc)
        or b_tf_delta >= float(args.min_b_gsm8k_tf_improvement)
        or teacher_b_exact > 0.0
    )
    line("=")
    print("GSM8K B TEACHER ACQUISITION GATE", flush=True)
    line("=")
    print(
        f"B teacher: {'PASS' if b_gate_pass else 'FAIL'} "
        f"gsm8k={fmt(teacher_b_exact)} "
        f"tf={fmt(teacher_b_tf)} base_tf={fmt(base_b_tf)} delta={fmt(b_tf_delta, 4)} "
        f"need tf>={fmt(args.min_b_gsm8k_tf_token_acc)} "
        f"or delta>={fmt(args.min_b_gsm8k_tf_improvement)} or exact>0",
        flush=True,
    )
    if args.abort_if_weak_gsm8k_b and not b_gate_pass:
        line("=")
        print("SUMMARY TABLE", flush=True)
        line("=")
        print_comparison_row("base_A", base_metrics)
        print_comparison_row("teacher_B_gsm8k", teacher_b_metrics)
        print("stopped: GSM8K B acquisition gate failed; skipped expensive D variants.", flush=True)
        return
    teacher_b_model = base_model

    # ── Stage AB: Consolidate GSM8K into base weights ──────────────────────────
    sub("Stage AB: consolidate GSM8K into base weights")
    base_ab = qp._clone_model(base_frozen, cfg.device)
    base_ab_metrics = audit.consolidate_with_proxy_in_memory(
        student=base_ab,
        teacher_old=base_frozen,
        teacher_new=teacher_b_model,
        tokenizer=tokenizer,
        new_task_batch_fn=gsm8k_batch_fn,
        proxy_batch_fn=proxy_batch_fn,
        eval_tasks=["retention"],
        eval_data=eval_data,
        selected_layers=b_layers,
        old_profiles=[a_profile],
        project_old_gradients=True,
        projection_strength=args.projection_strength,
        steps=args.b_consol_steps,
        lr=cfg.consolidation_lr,
        label="base_AB_gsm8k",
        cfg=cfg,
    )
    base_ab_metrics.update(evaluate_gsm8k(
        base_ab, tokenizer, cfg.device, gsm8k_eval, batch_size=cfg.eval_batch_size,
    ))
    base_ab_metrics.update(evaluate_gsm8k_teacher_forced(
        base_ab, tokenizer, cfg.device, gsm8k_eval,
        batch_size=cfg.eval_batch_size, max_seq_len=cfg.max_seq_len,
    ))
    print_comparison_row("base_AB_gsm8k", base_ab_metrics)
    release(base_frozen, teacher_b_model)

    # ── Stage D: Train sort adapter teacher from base_AB ───────────────────────
    sub("Stage D: train sort adapter teacher from base_AB")
    base_ab_frozen = qp._clone_model(base_ab, cfg.device)
    qp._freeze_model(base_ab_frozen)
    a_profile_ab = qp._collect_profiles(base_ab, "retention", proxy_batch_fn)
    b_profile_ab = qp._collect_profiles(base_ab, "json", gsm8k_batch_fn)

    d_teacher_batch_fn = audit.make_audit_sort_batch_fn(
        tokenizer, cfg.device, cfg, cfg.seed + 202,
        max_train_len=args.d_train_max_len,
        schedule_total_steps=cfg.proof_v2_d_attach_steps,
    )
    d_consol_batch_fn = audit.make_audit_sort_batch_fn(
        tokenizer, cfg.device, cfg, cfg.seed + 303,
        max_train_len=args.d_train_max_len,
        schedule_total_steps=args.d_consol_steps,
    )

    if args.skip_tomography:
        d_layers = parse_layer_indices(args.d_layer_indices)
        print(f"[z_tomography:sort] skipped; using fixed d_layers={d_layers}", flush=True)
    else:
        d_layers, _ = audit.select_layers(
            model=base_ab,
            tokenizer=tokenizer,
            task_name="sort",
            task_batch_fn=d_teacher_batch_fn,
            protected_profiles=[a_profile_ab, b_profile_ab],
            cfg=cfg,
        )
    qp._freeze_model(base_ab)
    attached_d = attach_latent_lora(
        base_ab,
        suffixes=qp._task_target_suffixes("sort", cfg),
        layer_indices=set(d_layers),
        config=LatentLoRAConfig(
            rank=cfg.proof_v2_d_rank,
            alpha=cfg.proof_v2_d_alpha,
            dropout=0.0,
            projection_strength=1.0,
            gate_init=cfg.proof_v2_d_gate_init,
            freeze_base=True,
        ),
    )
    print(f"[sort_lora:D] attached_modules={len(attached_d)}", flush=True)

    teacher_d_metrics = audit.train_adapter_in_memory(
        model=base_ab,
        tokenizer=tokenizer,
        attached=attached_d,
        task_batch_fn=d_teacher_batch_fn,
        old_task_batch_fn=None,
        eval_tasks=["retention", "sort"],
        eval_data=eval_data,
        steps=cfg.proof_v2_d_attach_steps,
        lr=cfg.proof_v2_d_attach_lr,
        label="teacher_D_sort",
        cfg=cfg,
    )
    audit.add_sort_teacher_forced_metrics(teacher_d_metrics, base_ab, tokenizer, cfg)
    teacher_d_metrics.update(evaluate_gsm8k(
        base_ab, tokenizer, cfg.device, gsm8k_eval, batch_size=cfg.eval_batch_size,
    ))
    print_comparison_row("teacher_D_sort", teacher_d_metrics)
    teacher_d_model = base_ab

    # ══════════════════════════════════════════════════════════════════════════
    # HEAD-TO-HEAD COMPARISON: Three D-consolidation strategies
    # ══════════════════════════════════════════════════════════════════════════

    def full_eval(model, label: str) -> Dict[str, float]:
        """Run the full evaluation suite on a model."""
        metrics = qp.evaluate_retention(model, tokenizer, wikitext_val, cfg.device, cfg.eval_batch_size)
        metrics.update(qp.evaluate_sort(model, tokenizer, cfg.device, cfg.sort_eval_samples, cfg.eval_batch_size, cfg))
        audit.add_sort_teacher_forced_metrics(metrics, model, tokenizer, cfg)
        metrics.update(evaluate_gsm8k(model, tokenizer, cfg.device, gsm8k_eval, batch_size=cfg.eval_batch_size))
        metrics.update(evaluate_gsm8k_teacher_forced(
            model, tokenizer, cfg.device, gsm8k_eval,
            batch_size=cfg.eval_batch_size, max_seq_len=cfg.max_seq_len,
        ))
        return metrics

    # ── Variant 1: Naive SFT (catastrophic forgetting baseline) ────────────────
    sub("Variant 1: Naive SFT (no protection)")
    naive_student = qp._clone_model(base_ab_frozen, cfg.device)
    consolidate_naive_sft_in_memory(
        student=naive_student,
        teacher_new=teacher_d_model,
        tokenizer=tokenizer,
        new_task_batch_fn=d_consol_batch_fn,
        steps=args.d_consol_steps,
        lr=cfg.consolidation_lr,
        label="D_naive_sft",
        cfg=cfg,
    )
    naive_metrics = full_eval(naive_student, "D_naive_sft")
    print_comparison_row("D_naive_sft", naive_metrics)
    release(naive_student)

    # ── Variant 2: SDFT Baseline (self-distilled, no protection) ───────────────
    sub("Variant 2: SDFT Baseline (self-distilled SFT, no protection)")
    sdft_student = qp._clone_model(base_ab_frozen, cfg.device)
    consolidate_sdft_in_memory(
        student=sdft_student,
        teacher_new=teacher_d_model,
        tokenizer=tokenizer,
        new_task_batch_fn=d_consol_batch_fn,
        steps=args.d_consol_steps,
        lr=cfg.consolidation_lr,
        label="D_sdft_baseline",
        cfg=cfg,
    )
    sdft_metrics = full_eval(sdft_student, "D_sdft_baseline")
    print_comparison_row("D_sdft_baseline", sdft_metrics)
    release(sdft_student)

    # ── Variant 3: Amoeba No-Proxy (full geometric CL) ────────────────────────
    sub("Variant 3: Amoeba No-Proxy (full geometric CL)")
    amoeba_student = qp._clone_model(base_ab_frozen, cfg.device)
    amoeba_metrics_raw = audit.consolidate_no_proxy_same_batch_in_memory(
        student=amoeba_student,
        teacher_old=base_ab_frozen,
        teacher_new=teacher_d_model,
        tokenizer=tokenizer,
        new_task_batch_fn=d_consol_batch_fn,
        eval_tasks=["retention", "sort"],
        eval_data=eval_data,
        selected_layers=d_layers,
        old_profiles=[a_profile_ab, b_profile_ab],
        project_old_gradients=True,
        projection_strength=args.projection_strength,
        steps=args.d_consol_steps,
        lr=cfg.consolidation_lr,
        label="D_amoeba_no_proxy",
        cfg=cfg,
        old_task_kl_weight=args.no_proxy_old_kl_weight,
        old_task_hidden_weight=args.no_proxy_old_hidden_weight,
        new_kl_weight=args.new_kl_weight,
        new_hidden_weight=args.new_hidden_weight,
    )
    amoeba_metrics = full_eval(amoeba_student, "D_amoeba_no_proxy")
    print_comparison_row("D_amoeba_no_proxy", amoeba_metrics)
    release(amoeba_student, base_ab_frozen, teacher_d_model)

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════════
    line("=")
    print("HEAD-TO-HEAD COMPARISON TABLE", flush=True)
    line("=")
    print(
        f"{'stage':<30} {'ppl':>7} {'gsm8k':>7} {'gsm8k_tf':>9} "
        f"{'sort_tok':>9} {'sort_tf':>8} {'sort_loss':>10}",
        flush=True,
    )
    line("-")
    print_comparison_row("base_A", base_metrics)
    print_comparison_row("teacher_B_gsm8k", teacher_b_metrics)
    print_comparison_row("base_AB_gsm8k", base_ab_metrics)
    print_comparison_row("teacher_D_sort", teacher_d_metrics)
    line("-")
    print_comparison_row("D_naive_sft", naive_metrics)
    print_comparison_row("D_sdft_baseline", sdft_metrics)
    print_comparison_row("D_amoeba_no_proxy", amoeba_metrics)
    line("=")

    # ── Verdict ────────────────────────────────────────────────────────────────
    print("VERDICT", flush=True)
    line("=")
    ab_gsm = float(base_ab_metrics.get("gsm8k_exact", 0.0))
    ab_tf = float(base_ab_metrics.get("gsm8k_tf_token_acc", 0.0))

    for label, m in [("naive_sft", naive_metrics), ("sdft_baseline", sdft_metrics), ("amoeba_no_proxy", amoeba_metrics)]:
        gsm_exact = float(m.get("gsm8k_exact", 0.0))
        gsm_tf = float(m.get("gsm8k_tf_token_acc", 0.0))
        sort_tok = float(m.get("sort_token_acc", 0.0))
        sort_loss = float(m.get("sort_loss", 999.0))
        gsm_delta = gsm_exact - ab_gsm
        gsm_tf_delta = gsm_tf - ab_tf
        print(
            f"  {label:<25} "
            f"gsm8k_retention={fmt(gsm_exact)} (delta={fmt(gsm_delta, 4)}) "
            f"gsm8k_tf_retention={fmt(gsm_tf)} (delta={fmt(gsm_tf_delta, 4)}) "
            f"sort_acquired={fmt(sort_tok)} sort_loss={fmt(sort_loss)}",
            flush=True,
        )

    print(flush=True)
    print(
        "The method with the HIGHEST gsm8k retention AND non-zero sort acquisition wins.",
        flush=True,
    )


if __name__ == "__main__":
    run()
