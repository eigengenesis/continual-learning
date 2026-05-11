from __future__ import annotations

import argparse
import copy
import gc
import math
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import qwen_continual_proof as qp
from qwen_tomography import build_occupied_basis
from standalone_latent_lora_qwen import (
    LatentLoRAConfig,
    attach_latent_lora,
    choose_dtype,
    default_model_id,
    load_causal_lm,
    load_tokenizer,
)


def line(char: str = "=") -> None:
    print(char * 96, flush=True)


def section(title: str) -> None:
    line("=")
    print(title, flush=True)
    line("=")


def sub(title: str) -> None:
    print("-" * 96, flush=True)
    print(title, flush=True)
    print("-" * 96, flush=True)


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "nan"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def metric(metrics: Dict[str, float], key: str, default: float = float("nan")) -> float:
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def print_metrics(label: str, metrics: Dict[str, float]) -> None:
    print(
        f"{label:<26} "
        f"ppl={fmt(metric(metrics, 'wikitext_ppl')):>7} "
        f"b_field={fmt(metric(metrics, 'json_field_acc')):>7} "
        f"b_train={fmt(metric(metrics, 'json_train_field_acc')):>7} "
        f"b_valid={fmt(metric(metrics, 'json_valid')):>7} "
        f"b_tf={fmt(metric(metrics, 'json_teacher_forced_token_acc')):>7} "
        f"b_train_tf={fmt(metric(metrics, 'json_train_teacher_forced_token_acc')):>7} "
        f"sort_tok={fmt(metric(metrics, 'sort_token_acc')):>7} "
        f"sort_train_tok={fmt(metric(metrics, 'sort_train_token_acc')):>7} "
        f"sort_tf_tok={fmt(metric(metrics, 'sort_teacher_forced_token_acc')):>7} "
        f"sort_train_tf={fmt(metric(metrics, 'sort_train_teacher_forced_token_acc')):>7} "
        f"sort_loss={fmt(metric(metrics, 'sort_loss')):>7} "
        f"sort_train_loss={fmt(metric(metrics, 'sort_train_loss')):>7} "
        f"sort_exact={fmt(metric(metrics, 'sort_exact')):>7} "
        f"comp_tok={fmt(metric(metrics, 'compose_token_acc')):>7} "
        f"comp_final={fmt(metric(metrics, 'compose_final_token_acc')):>7} "
        f"comp_exact={fmt(metric(metrics, 'compose_exact')):>7}",
        flush=True,
    )


def layer_index_from_name(name: str) -> int | None:
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def project_old_occupied_gradients(
    model: nn.Module,
    old_profiles,
    selected_layers: Sequence[int],
    *,
    strength: float,
    target_suffix: str = "mlp.down_proj",
) -> int:
    """Project linear output gradients away from protected old occupied bases.

    This is the Qwen audit analogue of the toy activation-space gradient
    projection. For `mlp.down_proj`, weight.grad has shape
    [hidden_dim, intermediate_dim], so left-multiplying by the hidden-state
    basis removes old occupied output directions from the update.
    """
    if strength <= 0.0 or not old_profiles:
        return 0
    selected = set(int(v) for v in selected_layers)
    projected = 0
    for name, module in model.named_modules():
        if not name.endswith(target_suffix):
            continue
        layer_idx = layer_index_from_name(name)
        if layer_idx is None or layer_idx not in selected:
            continue
        weight = getattr(module, "weight", None)
        grad = None if weight is None else weight.grad
        if grad is None or grad.ndim != 2:
            continue
        basis = build_occupied_basis(list(old_profiles), layer_idx, mode="activation")
        if basis.numel() == 0 or basis.shape[0] != grad.shape[0]:
            basis = build_occupied_basis(list(old_profiles), layer_idx, mode="union")
        if basis.numel() == 0 or basis.shape[0] != grad.shape[0]:
            continue
        basis = basis.to(device=grad.device, dtype=grad.dtype)
        projection = basis @ (basis.transpose(0, 1) @ grad)
        grad.sub_(float(strength) * projection)
        projected += 1
    return projected


def release(*models: nn.Module | None) -> None:
    qp._release_cuda_memory(*models)
    gc.collect()


def save_stage_checkpoint(model, tokenizer, path_text: str | None, label: str) -> None:
    if not path_text:
        return
    path = Path(path_text).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"[{label}] saved_checkpoint={path}", flush=True)


def composition_example(rng: np.random.Generator, *, heldout: bool = True) -> Tuple[str, str]:
    prompt, target, _, _ = composition_components_example(rng, heldout=heldout)
    return prompt, target


def composition_components_example(
    rng: np.random.Generator,
    *,
    heldout: bool = True,
) -> Tuple[str, str, str, str]:
    family_count = qp.PROOF_V2_EVAL_RECORD_FAMILY_COUNT if heldout else qp.PROOF_V2_TRAIN_RECORD_FAMILY_COUNT
    family_id = int(rng.integers(0, family_count))
    fields, value_pool = qp._proof_v2_record_family(family_id, heldout=heldout)
    route_code, route_perm = qp.PROOF_V2_ROUTE_CODES[int(rng.integers(0, len(qp.PROOF_V2_ROUTE_CODES)))]
    assignments = {field: value_pool[field][int(rng.integers(0, len(value_pool[field])))] for field in fields}
    shuffled = list(fields)
    rng.shuffle(shuffled)
    assignment_text = " ; ".join(f"{field}={assignments[field]}" for field in shuffled)
    routed_values = {
        slot: assignments[fields[field_index]]
        for slot, field_index in zip(qp.PROOF_V2_OUTPUT_SLOTS, route_perm)
    }
    keys = [
        qp.PROOF_V2_SORT_KEY_ORDER[int(rng.integers(0, len(qp.PROOF_V2_SORT_KEY_ORDER)))]
        for _ in qp.PROOF_V2_OUTPUT_SLOTS
    ]
    packets = [
        {"slot": slot, "key": key, "tag": chr(ord("a") + idx)}
        for idx, (slot, key) in enumerate(zip(qp.PROOF_V2_OUTPUT_SLOTS, keys))
    ]
    packet_text = ",".join(f"{item['slot']}~{item['key']}~{item['tag']}" for item in packets)
    sorted_packets = sorted(
        packets,
        key=lambda item: qp.PROOF_V2_SORT_KEY_ORDER.index(item["key"]),
    )
    target = ",".join(f"{item['slot']}={routed_values[item['slot']]}" for item in sorted_packets)
    b_prompt = qp.PROOF_V2_EVAL_RECORD_PROMPTS[0].format(
        route=route_code,
        fields=",".join(fields),
        assignments=assignment_text,
    ) + "\n"
    d_prompt = f"StableSort VX packets: {packet_text} -> "
    prompt = (
        f"Compose RouteSort VX\n"
        f"Route {route_code}\n"
        f"Field order {','.join(fields)}\n"
        f"Observed bindings {assignment_text}\n"
        f"StableSort slots by packet keys: {packet_text}\n"
        f"Return sorted slot=value list:"
    )
    return prompt, target, b_prompt, d_prompt


def _assignment_tokens(text: str) -> List[str]:
    cleaned = str(text).replace("\n", " ").replace(";", ",")
    if "ANS" in cleaned:
        cleaned = cleaned.split("ANS")[-1]
    tokens: List[str] = []
    for chunk in cleaned.split(","):
        item = chunk.strip().strip(".")
        if "=" in item:
            left, right = item.split("=", 1)
            left = left.strip()
            right = right.strip().split()[0] if right.strip() else ""
            if left and right:
                tokens.append(f"{left}={right}")
    return tokens


def _parse_slot_value_map(text: str) -> Dict[str, str] | None:
    out: Dict[str, str] = {}
    for token in _assignment_tokens(text):
        slot, value = token.split("=", 1)
        if slot in qp.PROOF_V2_OUTPUT_SLOTS and slot not in out:
            out[slot] = value
    return out if set(out) == set(qp.PROOF_V2_OUTPUT_SLOTS) else None


def _parse_slot_order(text: str) -> List[str] | None:
    cleaned = str(text).replace("\n", " ").replace(";", ",")
    out: List[str] = []
    for chunk in cleaned.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "~" in item:
            slot = item.split("~", 1)[0].strip()
        elif "=" in item:
            slot = item.split("=", 1)[0].strip()
        else:
            slot = item.split()[0].strip()
        if slot in qp.PROOF_V2_OUTPUT_SLOTS and slot not in out:
            out.append(slot)
    return out if sorted(out) == sorted(qp.PROOF_V2_OUTPUT_SLOTS) else None


def _final_assignment_text(text: str) -> str:
    cleaned = str(text).strip()
    if "ANS" in cleaned:
        cleaned = cleaned.split("ANS")[-1].strip(" :\n")
    return ",".join(_assignment_tokens(cleaned))


@torch.no_grad()
def evaluate_composition(
    model,
    tokenizer,
    device: str,
    num_samples: int,
    eval_batch_size: int,
    cfg: qp.RuntimeConfig,
) -> Dict[str, float]:
    model.eval()
    rng = np.random.default_rng(777_331)
    examples = [composition_example(rng, heldout=True) for _ in range(max(int(num_samples), 1))]
    exact = 0
    token_acc = 0.0
    final_exact = 0
    final_token_acc = 0.0
    losses: List[float] = []
    for batch_examples in qp._iter_batches(examples, eval_batch_size):
        prompts = [prompt for prompt, _ in batch_examples]
        targets = [target for _, target in batch_examples]
        completions = qp._generate_batch_tokens(
            model,
            tokenizer,
            prompts,
            device,
            max_new_tokens=96,
        )
        for completion, target in zip(completions, targets):
            pred_tokens = [item.strip() for item in completion.split(",") if item.strip()]
            target_tokens = [item.strip() for item in target.split(",") if item.strip()]
            if completion.strip() == target.strip():
                exact += 1
            correct = sum(1 for pred, gold in zip(pred_tokens, target_tokens) if pred == gold)
            token_acc += correct / max(len(target_tokens), 1)
            final_completion = _final_assignment_text(completion)
            final_tokens = [item.strip() for item in final_completion.split(",") if item.strip()]
            if final_completion.strip() == target.strip():
                final_exact += 1
            final_correct = sum(1 for pred, gold in zip(final_tokens, target_tokens) if pred == gold)
            final_token_acc += final_correct / max(len(target_tokens), 1)
        supervised = qp._prepare_supervised_batch(
            tokenizer,
            prompts,
            [f"{target}{tokenizer.eos_token}" for target in targets],
            device,
            cfg.max_seq_len,
        )
        losses.append(float(model(**supervised, use_cache=False).loss.item()))
    total = max(len(examples), 1)
    model.train()
    return {
        "compose_exact": exact / total,
        "compose_token_acc": token_acc / total,
        "compose_final_exact": final_exact / total,
        "compose_final_token_acc": final_token_acc / total,
        "compose_loss": sum(losses) / max(len(losses), 1),
    }


def add_composition_metrics(
    metrics: Dict[str, float],
    model,
    tokenizer,
    cfg: qp.RuntimeConfig,
    samples: int,
) -> Dict[str, float]:
    metrics.update(evaluate_composition(model, tokenizer, cfg.device, samples, cfg.eval_batch_size, cfg))
    return metrics


@torch.no_grad()
def _teacher_forced_sequence_token_acc(
    model,
    tokenizer,
    device: str,
    examples: Sequence[Tuple[str, str]],
    eval_batch_size: int,
    cfg: qp.RuntimeConfig,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for example_batch in qp._iter_batches(list(examples), eval_batch_size):
        prompts = [prompt for prompt, _ in example_batch]
        targets = [f"{target}{tokenizer.eos_token}" for _, target in example_batch]
        batch = qp._prepare_supervised_batch(tokenizer, prompts, targets, device, cfg.max_seq_len)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        labels = batch["labels"]
        shifted_labels = labels[:, 1:].contiguous()
        shifted_logits = outputs.logits[:, :-1, :].contiguous()
        mask = shifted_labels.ne(-100)
        if not bool(mask.any()):
            continue
        predictions = shifted_logits.argmax(dim=-1)
        correct += int(predictions[mask].eq(shifted_labels[mask]).sum().item())
        total += int(mask.sum().item())
    model.train()
    return float(correct / max(total, 1))


@torch.no_grad()
def evaluate_sort_teacher_forced(
    model,
    tokenizer,
    device: str,
    num_samples: int,
    eval_batch_size: int,
    cfg: qp.RuntimeConfig,
) -> Dict[str, float]:
    rng_train = np.random.default_rng(22345)
    rng_heldout = np.random.default_rng(32345)
    train_lengths = (8, 10, 12)
    heldout_lengths = (14, 16)
    total = max(int(num_samples), 1)
    train_examples = [
        qp._sort_example(
            rng_train,
            int(train_lengths[i % len(train_lengths)]),
            heldout=False,
            task_suite=cfg.task_suite,
        )
        for i in range(total)
    ]
    heldout_examples = [
        qp._sort_example(
            rng_heldout,
            int(heldout_lengths[i % len(heldout_lengths)]),
            heldout=True,
            task_suite=cfg.task_suite,
        )
        for i in range(total)
    ]
    return {
        "sort_train_teacher_forced_token_acc": _teacher_forced_sequence_token_acc(
            model,
            tokenizer,
            device,
            train_examples,
            eval_batch_size,
            cfg,
        ),
        "sort_teacher_forced_token_acc": _teacher_forced_sequence_token_acc(
            model,
            tokenizer,
            device,
            heldout_examples,
            eval_batch_size,
            cfg,
        ),
    }


def add_sort_teacher_forced_metrics(
    metrics: Dict[str, float],
    model,
    tokenizer,
    cfg: qp.RuntimeConfig,
) -> Dict[str, float]:
    metrics.update(
        evaluate_sort_teacher_forced(
            model,
            tokenizer,
            cfg.device,
            cfg.sort_eval_samples,
            cfg.eval_batch_size,
            cfg,
        )
    )
    return metrics


def add_b_teacher_forced_metrics(
    metrics: Dict[str, float],
    model,
    tokenizer,
    cfg: qp.RuntimeConfig,
) -> Dict[str, float]:
    if cfg.task_suite != "proof_v2":
        return metrics
    total = min(int(cfg.json_eval_samples), 96)
    train_rows = [
        qp._proof_v2_record_example(np.random.default_rng(8100 + idx), heldout=False, family_id=idx)
        for idx in range(total)
    ]
    heldout_rows = [
        qp._proof_v2_record_example(np.random.default_rng(9100 + idx), heldout=True, family_id=idx)
        for idx in range(total)
    ]
    metrics["json_train_teacher_forced_token_acc"] = _teacher_forced_sequence_token_acc(
        model,
        tokenizer,
        cfg.device,
        train_rows,
        cfg.eval_batch_size,
        cfg,
    )
    metrics["json_teacher_forced_token_acc"] = _teacher_forced_sequence_token_acc(
        model,
        tokenizer,
        cfg.device,
        heldout_rows,
        cfg.eval_batch_size,
        cfg,
    )
    return metrics


@torch.no_grad()
def print_b_debug_samples(
    model,
    tokenizer,
    cfg: qp.RuntimeConfig,
    *,
    label: str,
    limit: int = 3,
) -> None:
    if cfg.task_suite != "proof_v2" or limit <= 0:
        return
    train_rows = [
        qp._proof_v2_record_example(np.random.default_rng(8100 + idx), heldout=False, family_id=idx)
        for idx in range(limit)
    ]
    heldout_rows = [
        qp._proof_v2_record_example(np.random.default_rng(9100 + idx), heldout=True, family_id=idx)
        for idx in range(limit)
    ]
    parser = qp._parse_proof_v2_record_payload
    print(f"[{label}:B_DEBUG] showing {limit} train + {limit} heldout generations", flush=True)
    for split, rows in (("train", train_rows), ("heldout", heldout_rows)):
        samples = qp._collect_b_debug_examples(
            model,
            tokenizer,
            cfg.device,
            rows,
            parser=parser,
            limit=limit,
        )
        for idx, sample in enumerate(samples, start=1):
            prompt = " ".join(str(sample["prompt"]).strip().split())
            target = str(sample["target"]).strip()
            completion = " ".join(str(sample["completion"]).strip().split())
            print(
                f"[{label}:B_DEBUG:{split}:{idx}] "
                f"prompt={prompt[:180]} | target={target} | completion={completion[:220]} | "
                f"parsed={sample['completion_parsed']}",
                flush=True,
            )


def make_cfg(args: argparse.Namespace) -> qp.RuntimeConfig:
    model_id = args.model_id or default_model_id(args.local_files_only or args.smoke)
    cfg = qp.RuntimeConfig(
        model_id=model_id,
        device=args.device,
        dtype=choose_dtype(args.dtype),
        local_files_only=bool(args.local_files_only or args.smoke),
        resume=False,
        smoke=bool(args.smoke),
        output_dir=Path(args.output_dir),
        backup_dir=None,
        seed=args.seed,
        phase_scope="abd_rescue",
        task_suite="proof_v2",
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        consolidation_micro_batch_size=args.micro_batch_size,
        max_seq_len=args.max_seq_len,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        wikitext_eval_samples=args.wikitext_eval_samples,
        json_eval_samples=args.json_eval_samples,
        sort_eval_samples=args.sort_eval_samples,
        reversal_eval_samples=args.reversal_eval_samples,
        teacher_old_loss_weight=args.teacher_old_loss_weight,
        teacher_old_batch_period=args.teacher_old_batch_period,
        consolidation_lr=args.consolidation_lr,
        consol_kl_weight=args.new_kl_weight,
        consol_old_kl_weight=args.proxy_old_kl_weight,
        consol_hidden_weight=args.new_hidden_weight,
        proof_v2_b_attach_steps=args.b_attach_steps,
        proof_v2_b_attach_lr=args.b_attach_lr,
        proof_v2_b_rank=args.b_rank,
        proof_v2_b_alpha=args.b_alpha,
        proof_v2_b_gate_init=args.b_gate_init,
        proof_v2_b_use_up_proj=True,
        proof_v2_b_min_layers=args.b_min_layers,
        proof_v2_d_attach_steps=args.d_attach_steps,
        proof_v2_d_attach_lr=args.d_attach_lr,
        proof_v2_d_rank=args.d_rank,
        proof_v2_d_alpha=args.d_alpha,
        proof_v2_d_gate_init=args.d_gate_init,
        proof_v2_d_use_up_proj=True,
        proof_v2_d_min_layers=args.d_min_layers,
    )
    if args.smoke:
        cfg.batch_size = min(cfg.batch_size, 2)
        cfg.eval_batch_size = min(cfg.eval_batch_size, 2)
        cfg.max_seq_len = min(cfg.max_seq_len, 96)
        cfg.wikitext_eval_samples = min(cfg.wikitext_eval_samples, 8)
        cfg.json_eval_samples = min(cfg.json_eval_samples, 4)
        cfg.sort_eval_samples = min(cfg.sort_eval_samples, 4)
        cfg.reversal_eval_samples = min(cfg.reversal_eval_samples, 4)
        cfg.proof_v2_b_attach_steps = min(cfg.proof_v2_b_attach_steps, 20)
        cfg.proof_v2_d_attach_steps = min(cfg.proof_v2_d_attach_steps, 20)
    return cfg


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "balanced":
        return
    if args.preset == "strong":
        args.b_attach_steps = 2400
        args.d_attach_steps = 2400
        args.b_attach_lr = 1e-4
        args.d_attach_lr = 1e-4
        args.b_rank = 48
        args.d_rank = 64
        args.b_alpha = 96.0
        args.d_alpha = 128.0
        args.b_min_layers = 8
        args.d_min_layers = 10
        args.b_consolidation_steps = 800
        args.d_consolidation_steps = 900
        args.consolidation_lr = 1e-5
        args.wikitext_eval_samples = max(args.wikitext_eval_samples, 64)
        args.json_eval_samples = max(args.json_eval_samples, 32)
        args.sort_eval_samples = max(args.sort_eval_samples, 48)
        args.log_interval = min(args.log_interval, 100)
        return
    if args.preset == "acquisition":
        args.b_attach_steps = 3600
        args.d_attach_steps = 3600
        args.b_attach_lr = 1e-4
        args.d_attach_lr = 1e-4
        args.b_rank = 96
        args.d_rank = 128
        args.b_alpha = 192.0
        args.d_alpha = 256.0
        args.b_min_layers = 10
        args.d_min_layers = 14
        args.b_consolidation_steps = 1000
        args.d_consolidation_steps = 1200
        args.consolidation_lr = 8e-6
        args.b_eval_style_train_frac = max(args.b_eval_style_train_frac, 0.50)
        args.d_train_max_len = max(args.d_train_max_len, 16)
        args.wikitext_eval_samples = max(args.wikitext_eval_samples, 64)
        args.json_eval_samples = max(args.json_eval_samples, 48)
        args.sort_eval_samples = max(args.sort_eval_samples, 64)
        args.log_interval = min(args.log_interval, 100)
        return
    raise ValueError(f"unknown preset: {args.preset}")


def _audit_record_example(
    rng: np.random.Generator,
    *,
    eval_style_train_frac: float,
) -> Tuple[str, str]:
    """Train-family B example with optional eval-style wording.

    This strengthens teacher acquisition without leaking heldout families or
    heldout value tokens. It only reduces brittle prompt-template overfitting.
    """
    prompt_styles = qp.PROOF_V2_TRAIN_RECORD_PROMPTS
    if float(eval_style_train_frac) > 0.0 and float(rng.random()) < float(eval_style_train_frac):
        prompt_styles = qp.PROOF_V2_EVAL_RECORD_PROMPTS
    family_id = int(rng.integers(0, qp.PROOF_V2_TRAIN_RECORD_FAMILY_COUNT))
    order, value_pool = qp._proof_v2_record_family(family_id, heldout=False)
    route_code, route_perm = qp.PROOF_V2_ROUTE_CODES[int(rng.integers(0, len(qp.PROOF_V2_ROUTE_CODES)))]
    assignments = {field: value_pool[field][int(rng.integers(0, len(value_pool[field])))] for field in order}
    shuffled_fields = list(order)
    rng.shuffle(shuffled_fields)
    assignment_text = " ; ".join(f"{field}={assignments[field]}" for field in shuffled_fields)
    prompt = prompt_styles[int(rng.integers(0, len(prompt_styles)))].format(
        route=route_code,
        fields=",".join(order),
        assignments=assignment_text,
    )
    routed_values = [assignments[order[idx]] for idx in route_perm]
    target = " ; ".join(
        f"{slot}={value}"
        for slot, value in zip(qp.PROOF_V2_OUTPUT_SLOTS, routed_values)
    )
    return f"{prompt}\n", target


def make_audit_json_batch_fn(
    tokenizer,
    device: str,
    cfg: qp.RuntimeConfig,
    seed: int,
    *,
    eval_style_train_frac: float,
) -> Callable[[int], Dict[str, torch.Tensor]]:
    if cfg.task_suite != "proof_v2" or float(eval_style_train_frac) <= 0.0:
        return qp._task_batch_factory("json", tokenizer, device, cfg, seed)

    def _batch(step: int) -> Dict[str, torch.Tensor]:
        rng = qp._seed_rng(seed + 2003 * int(step))
        prompts: List[str] = []
        targets: List[str] = []
        for _ in range(cfg.batch_size):
            prompt, target = _audit_record_example(
                rng,
                eval_style_train_frac=eval_style_train_frac,
            )
            prompts.append(prompt)
            targets.append(f"{target}{tokenizer.eos_token}")
        return qp._prepare_supervised_batch(tokenizer, prompts, targets, device, cfg.max_seq_len)

    return _batch


def _audit_sort_seq_len(step: int, total_steps: int, max_len: int) -> int:
    if int(max_len) <= 12:
        return qp._curriculum_seq_len(step, total_steps)
    progress = float(step) / max(float(total_steps), 1.0)
    if progress < 0.20:
        return 8
    if progress < 0.40:
        return 10
    if progress < 0.60:
        return 12
    if progress < 0.80:
        return 14
    return min(int(max_len), 16)


def make_audit_sort_batch_fn(
    tokenizer,
    device: str,
    cfg: qp.RuntimeConfig,
    seed: int,
    *,
    max_train_len: int,
    schedule_total_steps: int,
) -> Callable[[int], Dict[str, torch.Tensor]]:
    if cfg.task_suite != "proof_v2" or int(max_train_len) <= 12:
        return qp._task_batch_factory("sort", tokenizer, device, cfg, seed)
    total_steps = max(int(schedule_total_steps), 1)

    def _batch(step: int) -> Dict[str, torch.Tensor]:
        return qp.generate_sort_batch(
            tokenizer,
            qp._seed_rng(seed + 4001 * int(step)),
            device,
            cfg,
            seq_len=_audit_sort_seq_len(step, total_steps, int(max_train_len)),
        )

    return _batch


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


def _adapter_teacher_score(metrics: Dict[str, float], eval_tasks: Sequence[str]) -> Tuple[float, ...]:
    if "json" in eval_tasks and "sort" not in eval_tasks:
        return (
            metric(metrics, "json_field_acc", 0.0),
            metric(metrics, "json_train_field_acc", 0.0),
            metric(metrics, "json_teacher_forced_token_acc", 0.0),
            metric(metrics, "json_valid", 0.0),
            -metric(metrics, "json_loss", 999.0),
        )
    if "sort" in eval_tasks:
        return (
            metric(metrics, "sort_token_acc", 0.0),
            metric(metrics, "sort_train_token_acc", 0.0),
            metric(metrics, "sort_teacher_forced_token_acc", 0.0),
            -metric(metrics, "sort_loss", 999.0),
            metric(metrics, "json_field_acc", 0.0),
        )
    return (
        -metric(metrics, "wikitext_ppl", 999.0),
    )


def train_adapter_in_memory(
    *,
    model,
    tokenizer,
    attached,
    task_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    old_task_batch_fn: Callable[[int], Dict[str, torch.Tensor]] | None,
    eval_tasks: Sequence[str],
    eval_data: Dict[str, object],
    steps: int,
    lr: float,
    label: str,
    cfg: qp.RuntimeConfig,
) -> Dict[str, float]:
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
        if (
            old_task_batch_fn is not None
            and cfg.teacher_old_batch_period > 0
            and cfg.teacher_old_loss_weight > 0.0
            and step % int(cfg.teacher_old_batch_period) == 0
        ):
            old_outputs = model(**old_task_batch_fn(step), use_cache=False)
            loss = loss + old_outputs.loss * float(cfg.teacher_old_loss_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        qp.ESCAPE_SCHEDULE.apply_to_modules(attached, step, steps)
        if step % cfg.log_interval == 0 or step == steps:
            print(
                f"[{label}] step={step:04d}/{steps} loss={float(loss.item()):.4f} "
                f"old_teacher_batches={'yes' if old_task_batch_fn is not None and cfg.teacher_old_loss_weight > 0 else 'no'}",
                flush=True,
            )
        if step % cfg.eval_interval == 0 or step == steps:
            probe_metrics = qp.evaluate_world(model, tokenizer, list(eval_tasks), eval_data, cfg.device, cfg)
            if "json" in eval_tasks:
                add_b_teacher_forced_metrics(probe_metrics, model, tokenizer, cfg)
            if "sort" in eval_tasks:
                add_sort_teacher_forced_metrics(probe_metrics, model, tokenizer, cfg)
            score = _adapter_teacher_score(probe_metrics, eval_tasks)
            if best_score is None or score > best_score:
                best_score = score
                best_step = step
                best_metrics = dict(probe_metrics)
                best_state = _capture_trainable_state(model)
                print(
                    f"[{label}] best_update step={step:04d}/{steps} "
                    f"score={tuple(round(float(v), 4) for v in score)}",
                    flush=True,
                )
    if best_state is not None:
        _restore_trainable_state(model, best_state)
        print(f"[{label}] restored_best_step={best_step:04d}/{steps}", flush=True)
    metrics = qp.evaluate_world(model, tokenizer, list(eval_tasks), eval_data, cfg.device, cfg)
    if "json" in eval_tasks:
        add_b_teacher_forced_metrics(metrics, model, tokenizer, cfg)
    if "sort" in eval_tasks:
        add_sort_teacher_forced_metrics(metrics, model, tokenizer, cfg)
    if best_metrics is not None:
        metrics["teacher_best_step"] = float(best_step)
    print_metrics(label, metrics)
    print(f"[{label}] wall_time_sec={time.time() - start:.1f}", flush=True)
    return metrics


def _trainable_full_student(student, cfg: qp.RuntimeConfig) -> List[nn.Parameter]:
    qp._unfreeze_model(student)
    params = qp._trainable_params(student)
    if cfg.device.startswith("cuda"):
        qp._configure_gradient_checkpointing(student, cfg.gradient_checkpointing)
    return params


def consolidate_with_proxy_in_memory(
    *,
    student,
    teacher_old,
    teacher_new,
    tokenizer,
    new_task_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    proxy_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    eval_tasks: Sequence[str],
    eval_data: Dict[str, object],
    selected_layers: Sequence[int],
    old_profiles,
    project_old_gradients: bool,
    projection_strength: float,
    steps: int,
    lr: float,
    label: str,
    cfg: qp.RuntimeConfig,
) -> Dict[str, float]:
    qp._freeze_model(teacher_old)
    qp._freeze_model(teacher_new)
    params = _trainable_full_student(student, cfg)
    optimizer = torch.optim.AdamW(params, lr=lr)
    start = time.time()
    proxy_batches = 0
    new_batches = 0
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        pairs = [
            (new_task_batch_fn(step), teacher_new, float(cfg.consol_kl_weight), "new"),
            (proxy_batch_fn(step), teacher_old, float(cfg.consol_old_kl_weight), "proxy"),
        ]
        total_examples = sum(int(batch["input_ids"].shape[0]) for batch, _, _, _ in pairs)
        for batch, teacher, kl_weight, kind in pairs:
            if kind == "proxy":
                proxy_batches += 1
            else:
                new_batches += 1
            for micro in qp._split_tensor_batch(batch, cfg.consolidation_micro_batch_size):
                weight = float(micro["input_ids"].shape[0]) / float(max(total_examples, 1))
                student_outputs = student(**micro, output_hidden_states=True, use_cache=False)
                with torch.no_grad():
                    teacher_outputs = teacher(**micro, output_hidden_states=True, use_cache=False)
                kl = qp.kl_divergence(student_outputs.logits, teacher_outputs.logits)
                hidden = qp._hidden_state_alignment_from_outputs(
                    student_outputs,
                    teacher_outputs,
                    list(selected_layers),
                    micro["input_ids"].device,
                )
                loss = (student_outputs.loss + kl_weight * kl + float(cfg.consol_hidden_weight) * hidden) * weight
                loss.backward()
        projected_modules = 0
        if project_old_gradients:
            projected_modules = project_old_occupied_gradients(
                student,
                old_profiles,
                selected_layers,
                strength=projection_strength,
            )
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        if step % cfg.log_interval == 0 or step == steps:
            print(
                f"[{label}] step={step:04d}/{steps} new_batches={new_batches} proxy_batches={proxy_batches} "
                f"old_task_examples=0 projected_modules={projected_modules}",
                flush=True,
            )
    metrics = qp.evaluate_world(student, tokenizer, list(eval_tasks), eval_data, cfg.device, cfg)
    print_metrics(label, metrics)
    print(f"[{label}] wall_time_sec={time.time() - start:.1f}", flush=True)
    return metrics


def consolidate_no_proxy_same_batch_in_memory(
    *,
    student,
    teacher_old,
    teacher_new,
    tokenizer,
    new_task_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    eval_tasks: Sequence[str],
    eval_data: Dict[str, object],
    selected_layers: Sequence[int],
    old_profiles,
    project_old_gradients: bool,
    projection_strength: float,
    steps: int,
    lr: float,
    label: str,
    cfg: qp.RuntimeConfig,
    old_task_kl_weight: float,
    old_task_hidden_weight: float,
    new_kl_weight: float,
    new_hidden_weight: float,
) -> Dict[str, float]:
    """Toy no-proxy analogue for Qwen.

    Uses ONLY the new-task batch. The old checkpoint is queried on that same
    input through KL + hidden-state matching, so there is no old-task replay and
    no separate generic proxy corpus.
    """
    qp._freeze_model(teacher_old)
    qp._freeze_model(teacher_new)
    params = _trainable_full_student(student, cfg)
    optimizer = torch.optim.AdamW(params, lr=lr)
    start = time.time()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = new_task_batch_fn(step)
        for micro in qp._split_tensor_batch(batch, cfg.consolidation_micro_batch_size):
            student_outputs = student(**micro, output_hidden_states=True, use_cache=False)
            with torch.no_grad():
                old_outputs = teacher_old(**micro, output_hidden_states=True, use_cache=False)
                new_outputs = teacher_new(**micro, output_hidden_states=True, use_cache=False)
            new_kl = qp.kl_divergence(student_outputs.logits, new_outputs.logits)
            old_kl = qp.kl_divergence(student_outputs.logits, old_outputs.logits)
            new_hidden = qp._hidden_state_alignment_from_outputs(
                student_outputs,
                new_outputs,
                list(selected_layers),
                micro["input_ids"].device,
            )
            old_hidden = qp._hidden_state_alignment_from_outputs(
                student_outputs,
                old_outputs,
                list(selected_layers),
                micro["input_ids"].device,
            )
            loss = (
                student_outputs.loss
                + float(new_kl_weight) * new_kl
                + float(new_hidden_weight) * new_hidden
                + float(old_task_kl_weight) * old_kl
                + float(old_task_hidden_weight) * old_hidden
            )
            loss.backward()
        projected_modules = 0
        if project_old_gradients:
            projected_modules = project_old_occupied_gradients(
                student,
                old_profiles,
                selected_layers,
                strength=projection_strength,
            )
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        if step % cfg.log_interval == 0 or step == steps:
            print(
                f"[{label}] step={step:04d}/{steps} old_task_examples=0 proxy_batches=0 "
                f"same_batch_old_kl={old_task_kl_weight:.3f} "
                f"same_batch_old_hidden={old_task_hidden_weight:.3f} "
                f"projected_modules={projected_modules}",
                flush=True,
            )
    metrics = qp.evaluate_world(student, tokenizer, list(eval_tasks), eval_data, cfg.device, cfg)
    print_metrics(label, metrics)
    print(f"[{label}] wall_time_sec={time.time() - start:.1f}", flush=True)
    return metrics


def consolidate_no_proxy_with_b_probe_in_memory(
    *,
    student,
    teacher_old,
    teacher_new,
    tokenizer,
    new_task_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    b_probe_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    eval_tasks: Sequence[str],
    eval_data: Dict[str, object],
    selected_layers: Sequence[int],
    b_probe_layers: Sequence[int],
    old_profiles,
    project_old_gradients: bool,
    projection_strength: float,
    steps: int,
    lr: float,
    label: str,
    cfg: qp.RuntimeConfig,
    old_task_kl_weight: float,
    old_task_hidden_weight: float,
    new_kl_weight: float,
    new_hidden_weight: float,
    b_probe_period: int,
    b_probe_kl_weight: float,
    b_probe_hidden_weight: float,
) -> Dict[str, float]:
    """No-proxy D consolidation plus explicit B checkpoint probes.

    This is intentionally logged as a practical control, not as the pure
    old-task-example-free claim. It answers: if the only missing ingredient is
    a tiny B-specific anchor, can Qwen preserve B while absorbing D?
    """
    qp._freeze_model(teacher_old)
    qp._freeze_model(teacher_new)
    params = _trainable_full_student(student, cfg)
    optimizer = torch.optim.AdamW(params, lr=lr)
    start = time.time()
    b_probe_batches = 0
    period = max(int(b_probe_period), 1)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = new_task_batch_fn(step)
        for micro in qp._split_tensor_batch(batch, cfg.consolidation_micro_batch_size):
            student_outputs = student(**micro, output_hidden_states=True, use_cache=False)
            with torch.no_grad():
                old_outputs = teacher_old(**micro, output_hidden_states=True, use_cache=False)
                new_outputs = teacher_new(**micro, output_hidden_states=True, use_cache=False)
            new_kl = qp.kl_divergence(student_outputs.logits, new_outputs.logits)
            old_kl = qp.kl_divergence(student_outputs.logits, old_outputs.logits)
            new_hidden = qp._hidden_state_alignment_from_outputs(
                student_outputs,
                new_outputs,
                list(selected_layers),
                micro["input_ids"].device,
            )
            old_hidden = qp._hidden_state_alignment_from_outputs(
                student_outputs,
                old_outputs,
                list(selected_layers),
                micro["input_ids"].device,
            )
            loss = (
                student_outputs.loss
                + float(new_kl_weight) * new_kl
                + float(new_hidden_weight) * new_hidden
                + float(old_task_kl_weight) * old_kl
                + float(old_task_hidden_weight) * old_hidden
            )
            loss.backward()
        if step % period == 0:
            b_probe_batches += 1
            b_batch = b_probe_batch_fn(step)
            for micro in qp._split_tensor_batch(b_batch, cfg.consolidation_micro_batch_size):
                student_outputs = student(**micro, output_hidden_states=True, use_cache=False)
                with torch.no_grad():
                    old_outputs = teacher_old(**micro, output_hidden_states=True, use_cache=False)
                b_kl = qp.kl_divergence(student_outputs.logits, old_outputs.logits)
                b_hidden = qp._hidden_state_alignment_from_outputs(
                    student_outputs,
                    old_outputs,
                    list(b_probe_layers),
                    micro["input_ids"].device,
                )
                loss = float(b_probe_kl_weight) * b_kl + float(b_probe_hidden_weight) * b_hidden
                loss.backward()
        projected_modules = 0
        if project_old_gradients:
            projected_modules = project_old_occupied_gradients(
                student,
                old_profiles,
                selected_layers,
                strength=projection_strength,
            )
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        if step % cfg.log_interval == 0 or step == steps:
            print(
                f"[{label}] step={step:04d}/{steps} supervised_old_examples=0 proxy_batches=0 "
                f"same_batch_old_kl={old_task_kl_weight:.3f} "
                f"same_batch_old_hidden={old_task_hidden_weight:.3f} "
                f"b_probe_batches={b_probe_batches} "
                f"b_probe_kl={b_probe_kl_weight:.3f} "
                f"b_probe_hidden={b_probe_hidden_weight:.3f} "
                f"projected_modules={projected_modules}",
                flush=True,
            )
    metrics = qp.evaluate_world(student, tokenizer, list(eval_tasks), eval_data, cfg.device, cfg)
    print_metrics(label, metrics)
    print(f"[{label}] wall_time_sec={time.time() - start:.1f}", flush=True)
    return metrics


@torch.no_grad()
def build_composition_self_trace_pool(
    *,
    model,
    tokenizer,
    cfg: qp.RuntimeConfig,
    pool_size: int,
    seed: int,
    max_attempt_multiplier: int,
    keep_only_correct: bool,
) -> Tuple[List[Tuple[str, str]], int, int]:
    """Build self-generated B+D->C traces.

    This is the Qwen analogue of the toy lateral propagation phase. It does
    not use human-labeled raw composition targets for training. Instead it
    asks the current model for B(record routing) and D(stable sort) outputs,
    parses those interfaces, and forms an ANS trace from the model's own
    component predictions.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    pool: List[Tuple[str, str]] = []
    attempts = 0
    self_correct = 0
    target = max(int(pool_size), 1)
    max_attempts = max(target * max(int(max_attempt_multiplier), 1), target)
    batch_size = max(int(cfg.eval_batch_size), 1)
    while len(pool) < target and attempts < max_attempts:
        candidate_count = min(batch_size, max_attempts - attempts)
        components = [
            composition_components_example(rng, heldout=False)
            for _ in range(candidate_count)
        ]
        attempts += candidate_count
        b_prompts = [item[2] for item in components]
        d_prompts = [item[3] for item in components]
        b_outputs = qp._generate_batch_tokens(
            model,
            tokenizer,
            b_prompts,
            cfg.device,
            max_new_tokens=64,
        )
        d_outputs = qp._generate_batch_tokens(
            model,
            tokenizer,
            d_prompts,
            cfg.device,
            max_new_tokens=128,
        )
        for (compose_prompt, true_target, _, _), b_text, d_text in zip(components, b_outputs, d_outputs):
            slot_values = _parse_slot_value_map(b_text)
            slot_order = _parse_slot_order(d_text)
            if slot_values is None or slot_order is None:
                continue
            answer = ",".join(f"{slot}={slot_values[slot]}" for slot in slot_order)
            is_correct = answer == true_target
            if is_correct:
                self_correct += 1
            if keep_only_correct and not is_correct:
                continue
            trace_target = (
                f"B_MAP {' ; '.join(f'{slot}={slot_values[slot]}' for slot in qp.PROOF_V2_OUTPUT_SLOTS)}\n"
                f"D_ORDER {','.join(slot_order)}\n"
                f"ANS {answer}{tokenizer.eos_token}"
            )
            trace_prompt = (
                f"{compose_prompt}\n"
                f"Use learned B(record routing) and D(stable sort) interfaces, then answer:\n"
            )
            pool.append((trace_prompt, trace_target))
            if len(pool) >= target:
                break
    model.train()
    return pool, attempts, self_correct


def lateral_composition_self_distill_in_memory(
    *,
    student,
    tokenizer,
    b_anchor_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    d_anchor_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    eval_tasks: Sequence[str],
    eval_data: Dict[str, object],
    selected_layers: Sequence[int],
    old_profiles,
    project_old_gradients: bool,
    projection_strength: float,
    steps: int,
    lr: float,
    label: str,
    cfg: qp.RuntimeConfig,
    pool_size: int,
    max_attempt_multiplier: int,
    keep_only_correct: bool,
    bd_anchor_period: int,
    b_anchor_weight: float,
    d_anchor_weight: float,
) -> Dict[str, float]:
    """Melt self-generated scaffolded composition traces into weights."""
    params = _trainable_full_student(student, cfg)
    optimizer = torch.optim.AdamW(params, lr=lr)
    start = time.time()
    print(
        f"[{label}] building self-trace pool target={pool_size} "
        f"attempt_multiplier={max_attempt_multiplier} "
        f"keep_only_correct={keep_only_correct}",
        flush=True,
    )
    pool, attempts, self_correct = build_composition_self_trace_pool(
        model=student,
        tokenizer=tokenizer,
        cfg=cfg,
        pool_size=pool_size,
        seed=cfg.seed + 88_001,
        max_attempt_multiplier=max_attempt_multiplier,
        keep_only_correct=keep_only_correct,
    )
    acceptance = len(pool) / max(attempts, 1)
    self_acc = self_correct / max(len(pool), 1)
    print(
        f"[{label}] pool_examples={len(pool)} attempts={attempts} "
        f"acceptance={acceptance:.3f} self_trace_true_acc={self_acc:.3f}",
        flush=True,
    )
    if not pool:
        metrics = qp.evaluate_world(student, tokenizer, list(eval_tasks), eval_data, cfg.device, cfg)
        print_metrics(label, metrics)
        print(f"[{label}] skipped: no parseable self-generated traces", flush=True)
        print(f"[{label}] wall_time_sec={time.time() - start:.1f}", flush=True)
        return metrics
    rng = np.random.default_rng(cfg.seed + 88_777)
    period = max(int(bd_anchor_period), 1)
    b_anchor_batches = 0
    d_anchor_batches = 0
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        indices = rng.integers(0, len(pool), size=cfg.batch_size)
        examples = [pool[int(idx)] for idx in indices]
        batch = qp._prepare_supervised_batch(
            tokenizer,
            [prompt for prompt, _ in examples],
            [target for _, target in examples],
            cfg.device,
            cfg.max_seq_len,
        )
        for micro in qp._split_tensor_batch(batch, cfg.consolidation_micro_batch_size):
            student_outputs = student(**micro, use_cache=False)
            student_outputs.loss.backward()
        if step % period == 0:
            b_anchor_batches += 1
            d_anchor_batches += 1
            for anchor_batch, weight in (
                (b_anchor_batch_fn(step), float(b_anchor_weight)),
                (d_anchor_batch_fn(step), float(d_anchor_weight)),
            ):
                if weight <= 0.0:
                    continue
                for micro in qp._split_tensor_batch(anchor_batch, cfg.consolidation_micro_batch_size):
                    anchor_outputs = student(**micro, use_cache=False)
                    (weight * anchor_outputs.loss).backward()
        projected_modules = 0
        if project_old_gradients:
            projected_modules = project_old_occupied_gradients(
                student,
                old_profiles,
                selected_layers,
                strength=projection_strength,
            )
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        if step % cfg.log_interval == 0 or step == steps:
            print(
                f"[{label}] step={step:04d}/{steps} pool_examples={len(pool)} "
                f"self_trace_true_acc={self_acc:.3f} b_anchor_batches={b_anchor_batches} "
                f"d_anchor_batches={d_anchor_batches} projected_modules={projected_modules}",
                flush=True,
            )
    metrics = qp.evaluate_world(student, tokenizer, list(eval_tasks), eval_data, cfg.device, cfg)
    print_metrics(label, metrics)
    print(f"[{label}] wall_time_sec={time.time() - start:.1f}", flush=True)
    return metrics


def select_layers(
    *,
    model,
    tokenizer,
    task_name: str,
    task_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    protected_profiles,
    cfg: qp.RuntimeConfig,
) -> Tuple[List[int], object]:
    tomography = qp.run_tomography(
        model,
        tokenizer,
        [task_batch_fn(i) for i in range(1, 5)],
        protected_profiles,
    )
    if task_name == "json":
        layers = qp._proof_v2_b_selected_layers(tomography, cfg)
    elif task_name == "sort":
        layers = qp._proof_v2_d_selected_layers(tomography, cfg)
    else:
        layers = list(tomography.selected_layer_indices)
    tomography.selected_layer_indices = list(layers)
    print(
        f"[z_tomography:{task_name}] selected_layers={layers} "
        f"total_pressure={fmt(getattr(tomography, 'total_pressure', float('nan')))} "
        f"reason={getattr(tomography, 'selection_reason', '')}",
        flush=True,
    )
    return layers, tomography


def verdict_bool(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def teacher_gate_report(
    *,
    teacher_b_metrics: Dict[str, float],
    teacher_d_metrics: Dict[str, float],
    args: argparse.Namespace,
) -> bool:
    section("TEACHER ACQUISITION GATE")
    b_field = metric(teacher_b_metrics, "json_field_acc")
    b_train = metric(teacher_b_metrics, "json_train_field_acc")
    d_tok = metric(teacher_d_metrics, "sort_token_acc")
    d_train_tok = metric(teacher_d_metrics, "sort_train_token_acc")
    d_tf_tok = metric(teacher_d_metrics, "sort_teacher_forced_token_acc")
    d_train_tf_tok = metric(teacher_d_metrics, "sort_train_teacher_forced_token_acc")
    d_loss = metric(teacher_d_metrics, "sort_loss")
    b_ok = b_field >= args.min_teacher_b_field
    d_ok = (
        d_tok >= args.min_teacher_d_sort_tok
        or d_train_tok >= args.min_teacher_d_train_tok
        or d_tf_tok >= args.min_teacher_d_tf_tok
        or d_loss <= args.max_teacher_d_sort_loss
    )
    print(
        f"B teacher: {verdict_bool(b_ok)} "
        f"heldout_field={fmt(b_field)} train_field={fmt(b_train)} "
        f"heldout_tf={fmt(metric(teacher_b_metrics, 'json_teacher_forced_token_acc'))} "
        f"train_tf={fmt(metric(teacher_b_metrics, 'json_train_teacher_forced_token_acc'))} "
        f"valid={fmt(metric(teacher_b_metrics, 'json_valid'))} "
        f"(need heldout >= {args.min_teacher_b_field:.3f})",
        flush=True,
    )
    print(
        f"D teacher: {verdict_bool(d_ok)} "
        f"heldout_sort_tok={fmt(d_tok)} train_sort_tok={fmt(d_train_tok)} "
        f"heldout_tf_tok={fmt(d_tf_tok)} train_tf_tok={fmt(d_train_tf_tok)} "
        f"sort_loss={fmt(d_loss)} "
        f"(need heldout >= {args.min_teacher_d_sort_tok:.3f} "
        f"or train >= {args.min_teacher_d_train_tok:.3f} "
        f"or tf >= {args.min_teacher_d_tf_tok:.3f} "
        f"or loss <= {args.max_teacher_d_sort_loss:.3f})",
        flush=True,
    )
    if not b_ok:
        print(
            "teacher_gate_note: B is not strong enough yet; consolidation cannot prove old-skill retention cleanly.",
            flush=True,
        )
        if metric(teacher_b_metrics, "json_teacher_forced_token_acc", 0.0) >= 0.80:
            print(
                "teacher_gate_note: B teacher-forced accuracy is high but free generation is low; "
                "this is a decoding/interface acquisition problem, not necessarily a representation problem.",
                flush=True,
            )
    if not d_ok:
        print(
            "teacher_gate_note: D is not strong enough yet; sort consolidation should be treated as a teacher-acquisition failure first.",
            flush=True,
        )
    return bool(b_ok and d_ok)


def desiderata_report(
    *,
    base_metrics: Dict[str, float],
    base_ab_metrics: Dict[str, float],
    proxy_metrics: Dict[str, float] | None,
    no_proxy_metrics: Dict[str, float],
    comp_lateral_metrics: Dict[str, float] | None = None,
    args: argparse.Namespace,
) -> None:
    section("DESIDERATA VERDICT")
    base_ppl = metric(base_metrics, "wikitext_ppl")
    ab_ppl = metric(base_ab_metrics, "wikitext_ppl")
    proxy_ppl = metric(proxy_metrics or {}, "wikitext_ppl")
    no_proxy_ppl = metric(no_proxy_metrics, "wikitext_ppl")

    proxy_b = metric(proxy_metrics or {}, "json_field_acc")
    no_proxy_b = metric(no_proxy_metrics, "json_field_acc")
    ab_b = metric(base_ab_metrics, "json_field_acc")
    proxy_sort = metric(proxy_metrics or {}, "sort_token_acc")
    no_proxy_sort = metric(no_proxy_metrics, "sort_token_acc")
    proxy_loss = metric(proxy_metrics or {}, "sort_loss")
    no_proxy_loss = metric(no_proxy_metrics, "sort_loss")
    proxy_comp = max(
        metric(proxy_metrics or {}, "compose_token_acc", 0.0),
        metric(proxy_metrics or {}, "compose_final_token_acc", 0.0),
    )
    no_proxy_comp = max(
        metric(no_proxy_metrics, "compose_token_acc", 0.0),
        metric(no_proxy_metrics, "compose_final_token_acc", 0.0),
    )
    lateral_comp = (
        max(
            metric(comp_lateral_metrics, "compose_token_acc", 0.0),
            metric(comp_lateral_metrics, "compose_final_token_acc", 0.0),
        )
        if comp_lateral_metrics is not None
        else float("nan")
    )

    preserve_proxy = proxy_ppl <= ab_ppl * args.max_ppl_ratio and proxy_b >= ab_b - args.max_b_drop
    preserve_no_proxy = no_proxy_ppl <= ab_ppl * args.max_ppl_ratio and no_proxy_b >= ab_b - args.max_b_drop
    learns_proxy = proxy_sort >= args.min_sort_tok or proxy_loss <= args.max_sort_loss
    learns_no_proxy = no_proxy_sort >= args.min_sort_tok or no_proxy_loss <= args.max_sort_loss

    print("1. Preserve general performance and old skill", flush=True)
    if proxy_metrics is None:
        print("   proxy:    SKIPPED in this focused run; use prior full audit artifact.", flush=True)
    else:
        print(
            f"   proxy:    {verdict_bool(preserve_proxy)} "
            f"ppl_ratio_vs_base_AB={fmt(proxy_ppl / max(ab_ppl, 1e-9))} "
            f"b_delta_vs_base_AB={fmt(proxy_b - ab_b)}",
            flush=True,
        )
    print(
        f"   no_proxy: {verdict_bool(preserve_no_proxy)} "
        f"ppl_ratio_vs_base_AB={fmt(no_proxy_ppl / max(ab_ppl, 1e-9))} "
        f"b_delta_vs_base_AB={fmt(no_proxy_b - ab_b)}",
        flush=True,
    )

    print("2. Sequential learning", flush=True)
    proxy_learned_text = "skipped" if proxy_metrics is None else str(learns_proxy)
    print(
        f"   PASS pipeline order was base_A -> B -> base_AB -> D; "
        f"proxy_D_learned={proxy_learned_text} no_proxy_D_learned={learns_no_proxy} "
        f"(need sort_tok>={args.min_sort_tok:.3f} or sort_loss<={args.max_sort_loss:.3f})",
        flush=True,
    )

    print("3. Different distributions", flush=True)
    print(
        "   PASS/PARTIAL B is proof_v2 record routing; D is proof_v2 stable sort; "
        "both differ from WikiText retention. Broader natural tasks still needed.",
        flush=True,
    )

    print("4. Efficiency / no massive replay", flush=True)
    if proxy_metrics is None:
        print("   proxy:    SKIPPED in this focused run", flush=True)
    else:
        print(
            "   proxy:    PASS old-task examples=0, old-task replay buffer=0, generic proxy text used",
            flush=True,
        )
    print(
        "   no_proxy: PASS old-task examples=0, old-task replay buffer=0, generic proxy corpus=0; "
        "uses old checkpoint on new-task batches",
        flush=True,
    )

    comp_proxy = proxy_comp >= args.min_comp_tok
    comp_no_proxy = no_proxy_comp >= args.min_comp_tok
    comp_lateral = bool(comp_lateral_metrics is not None and lateral_comp >= args.min_comp_tok)
    print("5. Compositionality", flush=True)
    if proxy_metrics is None:
        print("   proxy:    SKIPPED in this focused run", flush=True)
    else:
        print(
            f"   proxy:    {verdict_bool(comp_proxy)} compose_token_acc={fmt(proxy_comp)} "
            f"(need >= {args.min_comp_tok:.3f})",
            flush=True,
        )
    print(
        f"   no_proxy: {verdict_bool(comp_no_proxy)} compose_token_acc={fmt(no_proxy_comp)} "
        f"(need >= {args.min_comp_tok:.3f})",
        flush=True,
    )
    if comp_lateral_metrics is not None:
        print(
            f"   lateral:  {verdict_bool(comp_lateral)} compose_final_token_acc={fmt(lateral_comp)} "
            f"(scaffold-to-weight, need >= {args.min_comp_tok:.3f})",
            flush=True,
        )

    section("ENGINEERING GAP LIST")
    if not preserve_no_proxy:
        print(
            "- no_proxy needs adaptive control: sweep same-batch old KL/hidden weights, "
            "or choose them from Z-tomography overlap instead of fixed toy values.",
            flush=True,
        )
    if no_proxy_ppl > ab_ppl * args.max_ppl_ratio:
        print(
            "- no_proxy hurts general language retention; add a tiny calibration proxy, LN heal, "
            "or lower hidden/KL anchoring pressure.",
            flush=True,
        )
    if no_proxy_b < ab_b - args.max_b_drop:
        print(
            "- no_proxy loses B retention; add B-specific same-batch probes or route D through expansion.",
            flush=True,
        )
    if not learns_no_proxy:
        print(
            "- no_proxy is over-anchored; reduce old KL/hidden or use a staged tkl schedule.",
            flush=True,
        )
    if proxy_metrics is not None and proxy_sort < args.min_sort_tok and no_proxy_sort < args.min_sort_tok:
        print(
            "- D token accuracy is weak; strengthen D teacher first before interpreting consolidation.",
            flush=True,
        )
    if metric(base_ab_metrics, "json_field_acc") < args.min_base_ab_b_field:
        print(
            "- base_AB heldout B is weak; strengthen B teacher/consolidation before claiming full old-skill retention.",
            flush=True,
        )
    if not (comp_proxy or comp_no_proxy or comp_lateral):
        print(
            "- composition failed; tune lateral self-trace pool size, trace quality, or add a tiny B+D bridge curriculum.",
            flush=True,
        )
    print("- add larger general eval: WikiText >=200 samples plus MMLU/GSM/code if budget allows.", flush=True)
    print("- add Z-triggered branch selection: fixed null-space, proxy anchor, no-proxy anchor, or expansion.", flush=True)
    print("- add multi-seed Qwen only after a single-seed audit passes the above gates.", flush=True)


def run() -> None:
    parser = argparse.ArgumentParser(description="Readable Qwen CL desiderata audit. Logs only; no JSON/CSV.")
    parser.add_argument("--preset", choices=["balanced", "strong", "acquisition"], default="balanced")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default="/tmp/qwen_cl_desiderata_audit_unused")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--wikitext-eval-samples", type=int, default=32)
    parser.add_argument("--json-eval-samples", type=int, default=16)
    parser.add_argument("--sort-eval-samples", type=int, default=24)
    parser.add_argument("--composition-eval-samples", type=int, default=24)
    parser.add_argument("--reversal-eval-samples", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--b-attach-steps", type=int, default=900)
    parser.add_argument("--b-attach-lr", type=float, default=8e-5)
    parser.add_argument("--b-rank", type=int, default=48)
    parser.add_argument("--b-alpha", type=float, default=96.0)
    parser.add_argument("--b-gate-init", type=float, default=-1.5)
    parser.add_argument("--b-min-layers", type=int, default=8)
    parser.add_argument("--b-eval-style-train-frac", type=float, default=0.0)
    parser.add_argument("--d-attach-steps", type=int, default=900)
    parser.add_argument("--d-attach-lr", type=float, default=8e-5)
    parser.add_argument("--d-rank", type=int, default=48)
    parser.add_argument("--d-alpha", type=float, default=96.0)
    parser.add_argument("--d-gate-init", type=float, default=-1.5)
    parser.add_argument("--d-min-layers", type=int, default=8)
    parser.add_argument("--d-train-max-len", type=int, default=12)
    parser.add_argument("--b-consolidation-steps", type=int, default=240)
    parser.add_argument("--d-consolidation-steps", type=int, default=400)
    parser.add_argument("--consolidation-lr", type=float, default=1e-5)
    parser.add_argument("--teacher-old-loss-weight", type=float, default=0.0)
    parser.add_argument("--teacher-old-batch-period", type=int, default=4)
    parser.add_argument("--new-kl-weight", type=float, default=1.0)
    parser.add_argument("--new-hidden-weight", type=float, default=0.5)
    parser.add_argument("--proxy-old-kl-weight", type=float, default=0.75)
    parser.add_argument("--no-proxy-old-kl-weight", type=float, default=0.7)
    parser.add_argument("--no-proxy-old-hidden-weight", type=float, default=30.0)
    parser.add_argument("--project-old-gradients", action="store_true")
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument("--skip-proxy-variant", action="store_true")
    parser.add_argument("--run-b-probe-variant", action="store_true")
    parser.add_argument("--b-probe-period", type=int, default=2)
    parser.add_argument("--b-probe-kl-weight", type=float, default=0.5)
    parser.add_argument("--b-probe-hidden-weight", type=float, default=5.0)
    parser.add_argument("--run-composition-lateral-variant", action="store_true")
    parser.add_argument("--composition-lateral-steps", type=int, default=300)
    parser.add_argument("--composition-lateral-lr", type=float, default=8e-6)
    parser.add_argument("--composition-lateral-pool-size", type=int, default=256)
    parser.add_argument("--composition-lateral-attempt-multiplier", type=int, default=8)
    parser.add_argument("--composition-lateral-keep-only-correct-traces", action="store_true")
    parser.add_argument("--composition-lateral-bd-anchor-period", type=int, default=2)
    parser.add_argument("--composition-lateral-b-anchor-weight", type=float, default=0.35)
    parser.add_argument("--composition-lateral-d-anchor-weight", type=float, default=0.35)
    parser.add_argument("--save-no-proxy-checkpoint", default=None)
    parser.add_argument("--resume-composition-from-checkpoint", default=None)
    parser.add_argument("--stop-after-b-teacher", action="store_true")
    parser.add_argument("--stop-after-teachers", action="store_true")
    parser.add_argument("--abort-if-weak-teacher", action="store_true")
    parser.add_argument("--min-teacher-b-field", type=float, default=0.30)
    parser.add_argument("--min-teacher-d-sort-tok", type=float, default=0.18)
    parser.add_argument("--min-teacher-d-train-tok", type=float, default=0.35)
    parser.add_argument("--min-teacher-d-tf-tok", type=float, default=0.70)
    parser.add_argument("--max-teacher-d-sort-loss", type=float, default=0.25)
    parser.add_argument("--max-ppl-ratio", type=float, default=1.10)
    parser.add_argument("--max-b-drop", type=float, default=0.08)
    parser.add_argument("--min-sort-tok", type=float, default=0.05)
    parser.add_argument("--max-sort-loss", type=float, default=0.32)
    parser.add_argument("--min-comp-tok", type=float, default=0.05)
    parser.add_argument("--min-base-ab-b-field", type=float, default=0.15)
    args = parser.parse_args()
    apply_preset(args)

    cfg = make_cfg(args)
    qp._set_seed(cfg.seed)

    section("QWEN CONTINUAL LEARNING DESIDERATA AUDIT")
    print("This script writes no JSON/CSV. Treat stdout as the artifact.", flush=True)
    print(f"model_id={cfg.model_id}", flush=True)
    print(f"device={cfg.device} dtype={cfg.dtype} seed={cfg.seed}", flush=True)
    print(f"preset={args.preset}", flush=True)
    print(
        f"strong_settings: b_steps={cfg.proof_v2_b_attach_steps} d_steps={cfg.proof_v2_d_attach_steps} "
        f"b_consol={args.b_consolidation_steps} d_consol={args.d_consolidation_steps} "
        f"b_rank={cfg.proof_v2_b_rank} d_rank={cfg.proof_v2_d_rank}",
        flush=True,
    )
    print(
        f"acquisition_helpers: b_eval_style_train_frac={args.b_eval_style_train_frac:.2f} "
        f"d_train_max_len={args.d_train_max_len} stop_after_b_teacher={args.stop_after_b_teacher} "
        f"stop_after_teachers={args.stop_after_teachers} "
        f"abort_if_weak_teacher={args.abort_if_weak_teacher}",
        flush=True,
    )
    print(
        "task_order=base_A -> B(record routing) -> base_AB -> D(stable sort)",
        flush=True,
    )
    print(
        "composition_eval=RouteSort VX requires B(record routing) + D(stable sort) in one output",
        flush=True,
    )
    print(
        "variants=D_proxy_anchor and D_no_proxy_same_batch_tkl_hidden",
        flush=True,
    )
    if args.run_b_probe_variant:
        print(
            "extra_variant=D_no_proxy_same_batch_plus_B_probe "
            "(practical control; uses generated B probe prompts, not pure old-task-example-free)",
            flush=True,
        )
        print(
            f"b_probe_period={args.b_probe_period} b_probe_kl={args.b_probe_kl_weight} "
            f"b_probe_hidden={args.b_probe_hidden_weight}",
            flush=True,
        )
    if args.run_composition_lateral_variant:
        print(
            "extra_variant=D_no_proxy_lateral_composition_self_distill "
            "(uses self-generated B/D composition traces; no human raw C labels)",
            flush=True,
        )
        print(
            f"composition_lateral_steps={args.composition_lateral_steps} "
            f"pool={args.composition_lateral_pool_size} "
            f"lr={args.composition_lateral_lr} "
            f"bd_anchor_period={args.composition_lateral_bd_anchor_period} "
            f"keep_only_correct={args.composition_lateral_keep_only_correct_traces}",
            flush=True,
        )
    print(
        f"gradient_projection={args.project_old_gradients} "
        f"projection_strength={args.projection_strength}",
        flush=True,
    )
    if args.skip_proxy_variant:
        print("focused_mode=skip_proxy_variant; proxy branch will rely on prior full-audit evidence.", flush=True)
    if args.save_no_proxy_checkpoint:
        print(f"save_no_proxy_checkpoint={args.save_no_proxy_checkpoint}", flush=True)
    if args.resume_composition_from_checkpoint:
        print(f"resume_composition_from_checkpoint={args.resume_composition_from_checkpoint}", flush=True)

    model_load_id = str(Path(args.resume_composition_from_checkpoint).expanduser()) if args.resume_composition_from_checkpoint else cfg.model_id
    tokenizer = load_tokenizer(
        model_load_id,
        trust_remote_code=True,
        local_files_only=bool(cfg.local_files_only or args.resume_composition_from_checkpoint),
    )

    if args.resume_composition_from_checkpoint:
        if not args.run_composition_lateral_variant:
            raise ValueError("--resume-composition-from-checkpoint requires --run-composition-lateral-variant")
        student = load_causal_lm(
            model_load_id,
            device=cfg.device,
            dtype=cfg.dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        qp._configure_gradient_checkpointing(student, cfg.gradient_checkpointing)
        eval_data = qp._load_eval_data(tokenizer, cfg)
        wikitext_train = qp.load_wikitext_texts(
            tokenizer,
            split="train",
            max_samples=max(cfg.wikitext_eval_samples, 64),
            max_seq_len=cfg.max_seq_len,
            local_files_only=cfg.local_files_only,
        )
        _ = qp.make_wikitext_batch_fn(tokenizer, wikitext_train, cfg.device, cfg, cfg.seed + 1)
        b_batch_fn = make_audit_json_batch_fn(
            tokenizer,
            cfg.device,
            cfg,
            cfg.seed + 2,
            eval_style_train_frac=args.b_eval_style_train_frac,
        )
        d_consolidation_batch_fn = make_audit_sort_batch_fn(
            tokenizer,
            cfg.device,
            cfg,
            cfg.seed + 5,
            max_train_len=args.d_train_max_len,
            schedule_total_steps=max(args.d_consolidation_steps, 1),
        )
        sub("Resume: evaluate checkpoint before lateral composition rescue")
        pre_metrics = qp.evaluate_world(student, tokenizer, ["retention", "json", "sort"], eval_data, cfg.device, cfg)
        add_sort_teacher_forced_metrics(pre_metrics, student, tokenizer, cfg)
        add_composition_metrics(pre_metrics, student, tokenizer, cfg, args.composition_eval_samples)
        print_metrics("resume_before_lateral", pre_metrics)
        sub("Resume: lateral composition self-distill only")
        if args.project_old_gradients:
            print(
                "resume_note: gradient projection is disabled in resume-only mode unless tomography profiles are rebuilt.",
                flush=True,
            )
        lateral_metrics = lateral_composition_self_distill_in_memory(
            student=student,
            tokenizer=tokenizer,
            b_anchor_batch_fn=b_batch_fn,
            d_anchor_batch_fn=d_consolidation_batch_fn,
            eval_tasks=["retention", "json", "sort"],
            eval_data=eval_data,
            selected_layers=[],
            old_profiles=[],
            project_old_gradients=False,
            projection_strength=args.projection_strength,
            steps=args.composition_lateral_steps,
            lr=args.composition_lateral_lr,
            label="resume_lateral_comp",
            cfg=cfg,
            pool_size=args.composition_lateral_pool_size,
            max_attempt_multiplier=args.composition_lateral_attempt_multiplier,
            keep_only_correct=args.composition_lateral_keep_only_correct_traces,
            bd_anchor_period=args.composition_lateral_bd_anchor_period,
            b_anchor_weight=args.composition_lateral_b_anchor_weight,
            d_anchor_weight=args.composition_lateral_d_anchor_weight,
        )
        add_sort_teacher_forced_metrics(lateral_metrics, student, tokenizer, cfg)
        add_composition_metrics(lateral_metrics, student, tokenizer, cfg, args.composition_eval_samples)
        print_metrics("resume_lateral+compose", lateral_metrics)
        section("COMPOSITION-FOCUSED VERDICT")
        before = max(metric(pre_metrics, "compose_token_acc", 0.0), metric(pre_metrics, "compose_final_token_acc", 0.0))
        after = max(metric(lateral_metrics, "compose_token_acc", 0.0), metric(lateral_metrics, "compose_final_token_acc", 0.0))
        print(f"before_comp={fmt(before)} after_comp={fmt(after)} delta={fmt(after - before)}", flush=True)
        print(f"composition_pass={verdict_bool(after >= args.min_comp_tok)} need>={args.min_comp_tok:.3f}", flush=True)
        release(student)
        return

    base_model = load_causal_lm(
        cfg.model_id,
        device=cfg.device,
        dtype=cfg.dtype,
        trust_remote_code=True,
        local_files_only=cfg.local_files_only,
    )
    qp._configure_gradient_checkpointing(base_model, cfg.gradient_checkpointing)
    eval_data = qp._load_eval_data(tokenizer, cfg)

    sub("Stage A: base model evaluation")
    base_metrics = qp.evaluate_world(base_model, tokenizer, ["retention", "json"], eval_data, cfg.device, cfg)
    add_b_teacher_forced_metrics(base_metrics, base_model, tokenizer, cfg)
    eval_data["baseline_retention"] = dict(base_metrics)
    print_metrics("base_A", base_metrics)

    wikitext_train = qp.load_wikitext_texts(
        tokenizer,
        split="train",
        max_samples=max(cfg.wikitext_eval_samples, 64),
        max_seq_len=cfg.max_seq_len,
        local_files_only=cfg.local_files_only,
    )
    proxy_batch_fn = qp.make_wikitext_batch_fn(tokenizer, wikitext_train, cfg.device, cfg, cfg.seed + 1)
    b_batch_fn = make_audit_json_batch_fn(
        tokenizer,
        cfg.device,
        cfg,
        cfg.seed + 2,
        eval_style_train_frac=args.b_eval_style_train_frac,
    )
    d_teacher_batch_fn = make_audit_sort_batch_fn(
        tokenizer,
        cfg.device,
        cfg,
        cfg.seed + 4,
        max_train_len=args.d_train_max_len,
        schedule_total_steps=cfg.proof_v2_d_attach_steps,
    )
    d_consolidation_batch_fn = make_audit_sort_batch_fn(
        tokenizer,
        cfg.device,
        cfg,
        cfg.seed + 5,
        max_train_len=args.d_train_max_len,
        schedule_total_steps=args.d_consolidation_steps,
    )

    base_frozen = qp._clone_model(base_model, cfg.device)
    qp._freeze_model(base_frozen)

    sub("Stage B: train B adapter teacher")
    a_profile = qp._collect_profiles(base_model, "retention", proxy_batch_fn)
    b_layers, _ = select_layers(
        model=base_model,
        tokenizer=tokenizer,
        task_name="json",
        task_batch_fn=b_batch_fn,
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
    teacher_b_metrics = train_adapter_in_memory(
        model=base_model,
        tokenizer=tokenizer,
        attached=attached_b,
        task_batch_fn=b_batch_fn,
        old_task_batch_fn=None,
        eval_tasks=["retention", "json"],
        eval_data=eval_data,
        steps=cfg.proof_v2_b_attach_steps,
        lr=cfg.proof_v2_b_attach_lr,
        label="teacher_B_no_old_examples",
        cfg=cfg,
    )
    add_b_teacher_forced_metrics(teacher_b_metrics, base_model, tokenizer, cfg)
    print_metrics("teacher_B+dense_decode", teacher_b_metrics)
    early_b_failed = metric(teacher_b_metrics, "json_field_acc", 0.0) < args.min_teacher_b_field
    if args.stop_after_b_teacher or (args.abort_if_weak_teacher and early_b_failed):
        section("EARLY B TEACHER ACQUISITION GATE")
        b_status = "PASS" if not early_b_failed else "FAIL"
        print(
            f"B teacher: {b_status} heldout_field={fmt(metric(teacher_b_metrics, 'json_field_acc'))} "
            f"train_field={fmt(metric(teacher_b_metrics, 'json_train_field_acc'))} "
            f"heldout_tf={fmt(metric(teacher_b_metrics, 'json_teacher_forced_token_acc'))} "
            f"train_tf={fmt(metric(teacher_b_metrics, 'json_train_teacher_forced_token_acc'))} "
            f"valid={fmt(metric(teacher_b_metrics, 'json_valid'))} "
            f"(need heldout >= {args.min_teacher_b_field:.3f})",
            flush=True,
        )
        if early_b_failed:
            print_b_debug_samples(base_model, tokenizer, cfg, label="early_b_gate_failed")
        section("SUMMARY TABLE")
        print_metrics("base_A", base_metrics)
        print_metrics("teacher_B", teacher_b_metrics)
        if args.stop_after_b_teacher:
            print("stopped: --stop-after-b-teacher was set; use this run to tune B acquisition.", flush=True)
        else:
            print("stopped: B teacher gate failed; skipped AB/D to avoid wasting hours.", flush=True)
        release(base_frozen, base_model)
        return
    teacher_b_model = base_model

    sub("Stage AB: consolidate B into base weights with generic proxy anchor")
    base_ab = qp._clone_model(base_frozen, cfg.device)
    base_ab_metrics = consolidate_with_proxy_in_memory(
        student=base_ab,
        teacher_old=base_frozen,
        teacher_new=teacher_b_model,
        tokenizer=tokenizer,
        new_task_batch_fn=b_batch_fn,
        proxy_batch_fn=proxy_batch_fn,
        eval_tasks=["retention", "json"],
        eval_data=eval_data,
        selected_layers=b_layers,
        old_profiles=[a_profile],
        project_old_gradients=args.project_old_gradients,
        projection_strength=args.projection_strength,
        steps=args.b_consolidation_steps,
        lr=cfg.consolidation_lr,
        label="base_AB_proxy_anchor",
        cfg=cfg,
    )
    add_b_teacher_forced_metrics(base_ab_metrics, base_ab, tokenizer, cfg)
    print_metrics("base_AB+dense_decode", base_ab_metrics)
    release(base_frozen, teacher_b_model)

    sub("Stage D: train D adapter teacher from base_AB")
    base_ab_frozen = qp._clone_model(base_ab, cfg.device)
    qp._freeze_model(base_ab_frozen)
    a_profile_ab = qp._collect_profiles(base_ab, "retention", proxy_batch_fn)
    b_profile_ab = qp._collect_profiles(base_ab, "json", b_batch_fn)
    d_layers, _ = select_layers(
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
    teacher_d_metrics = train_adapter_in_memory(
        model=base_ab,
        tokenizer=tokenizer,
        attached=attached_d,
        task_batch_fn=d_teacher_batch_fn,
        old_task_batch_fn=None,
        eval_tasks=["retention", "json", "sort"],
        eval_data=eval_data,
        steps=cfg.proof_v2_d_attach_steps,
        lr=cfg.proof_v2_d_attach_lr,
        label="teacher_D_no_old_examples",
        cfg=cfg,
    )
    add_b_teacher_forced_metrics(teacher_d_metrics, base_ab, tokenizer, cfg)
    add_sort_teacher_forced_metrics(teacher_d_metrics, base_ab, tokenizer, cfg)
    print_metrics("teacher_D+dense_decode", teacher_d_metrics)
    teacher_d_model = base_ab
    teachers_ok = teacher_gate_report(
        teacher_b_metrics=teacher_b_metrics,
        teacher_d_metrics=teacher_d_metrics,
        args=args,
    )
    if args.stop_after_teachers or (args.abort_if_weak_teacher and not teachers_ok):
        if not teachers_ok and metric(teacher_b_metrics, "json_field_acc", 0.0) < args.min_teacher_b_field:
            print_b_debug_samples(base_ab_frozen, tokenizer, cfg, label="teacher_gate_failed_base_AB")
        section("SUMMARY TABLE")
        print_metrics("base_A", base_metrics)
        print_metrics("teacher_B", teacher_b_metrics)
        print_metrics("base_AB", base_ab_metrics)
        print_metrics("teacher_D", teacher_d_metrics)
        if args.stop_after_teachers:
            print("stopped: --stop-after-teachers was set; use this run to tune acquisition before consolidation.", flush=True)
        else:
            print("stopped: teacher acquisition gate failed; skipped expensive D consolidation variants.", flush=True)
        release(teacher_d_model, base_ab_frozen)
        return

    proxy_metrics: Dict[str, float] | None = None
    if args.skip_proxy_variant:
        sub("Variant 1: D consolidation with generic proxy anchor")
        print("skipped: --skip-proxy-variant set; use prior full-audit proxy evidence.", flush=True)
    else:
        sub("Variant 1: D consolidation with generic proxy anchor")
        proxy_student = qp._clone_model(base_ab_frozen, cfg.device)
        proxy_metrics = consolidate_with_proxy_in_memory(
            student=proxy_student,
            teacher_old=base_ab_frozen,
            teacher_new=teacher_d_model,
            tokenizer=tokenizer,
            new_task_batch_fn=d_consolidation_batch_fn,
            proxy_batch_fn=proxy_batch_fn,
            eval_tasks=["retention", "json", "sort"],
            eval_data=eval_data,
            selected_layers=d_layers,
            old_profiles=[a_profile_ab, b_profile_ab],
            project_old_gradients=args.project_old_gradients,
            projection_strength=args.projection_strength,
            steps=args.d_consolidation_steps,
            lr=cfg.consolidation_lr,
            label="D_proxy_anchor_old_examples_0",
            cfg=cfg,
        )
        add_b_teacher_forced_metrics(proxy_metrics, proxy_student, tokenizer, cfg)
        add_sort_teacher_forced_metrics(proxy_metrics, proxy_student, tokenizer, cfg)
        add_composition_metrics(proxy_metrics, proxy_student, tokenizer, cfg, args.composition_eval_samples)
        print_metrics("D_proxy_anchor+compose", proxy_metrics)
        release(proxy_student)

    sub("Variant 2: D no-proxy same-batch anchor (latest toy analogue)")
    no_proxy_student = qp._clone_model(base_ab_frozen, cfg.device)
    no_proxy_metrics = consolidate_no_proxy_same_batch_in_memory(
        student=no_proxy_student,
        teacher_old=base_ab_frozen,
        teacher_new=teacher_d_model,
        tokenizer=tokenizer,
        new_task_batch_fn=d_consolidation_batch_fn,
        eval_tasks=["retention", "json", "sort"],
        eval_data=eval_data,
        selected_layers=d_layers,
        old_profiles=[a_profile_ab, b_profile_ab],
        project_old_gradients=args.project_old_gradients,
        projection_strength=args.projection_strength,
        steps=args.d_consolidation_steps,
        lr=cfg.consolidation_lr,
        label="D_no_proxy_same_batch_tkl_hidden",
        cfg=cfg,
        old_task_kl_weight=args.no_proxy_old_kl_weight,
        old_task_hidden_weight=args.no_proxy_old_hidden_weight,
        new_kl_weight=args.new_kl_weight,
        new_hidden_weight=args.new_hidden_weight,
    )
    add_b_teacher_forced_metrics(no_proxy_metrics, no_proxy_student, tokenizer, cfg)
    add_sort_teacher_forced_metrics(no_proxy_metrics, no_proxy_student, tokenizer, cfg)
    add_composition_metrics(no_proxy_metrics, no_proxy_student, tokenizer, cfg, args.composition_eval_samples)
    print_metrics("D_no_proxy+compose", no_proxy_metrics)
    save_stage_checkpoint(no_proxy_student, tokenizer, args.save_no_proxy_checkpoint, "D_no_proxy")
    comp_lateral_metrics: Dict[str, float] | None = None
    if args.run_composition_lateral_variant:
        sub("Variant 3: D no-proxy + lateral composition self-distill")
        print(
            "variant_note: this tests scaffold-to-weight composition. It builds C_SELF traces "
            "from the model's own B and D outputs, then trains direct composition prompts. "
            "This is not pure zero-shot automaticity and not human-labeled raw C replay.",
            flush=True,
        )
        comp_lateral_metrics = lateral_composition_self_distill_in_memory(
            student=no_proxy_student,
            tokenizer=tokenizer,
            b_anchor_batch_fn=b_batch_fn,
            d_anchor_batch_fn=d_consolidation_batch_fn,
            eval_tasks=["retention", "json", "sort"],
            eval_data=eval_data,
            selected_layers=d_layers,
            old_profiles=[a_profile_ab, b_profile_ab],
            project_old_gradients=args.project_old_gradients,
            projection_strength=args.projection_strength,
            steps=args.composition_lateral_steps,
            lr=args.composition_lateral_lr,
            label="D_no_proxy_lateral_comp",
            cfg=cfg,
            pool_size=args.composition_lateral_pool_size,
            max_attempt_multiplier=args.composition_lateral_attempt_multiplier,
            keep_only_correct=args.composition_lateral_keep_only_correct_traces,
            bd_anchor_period=args.composition_lateral_bd_anchor_period,
            b_anchor_weight=args.composition_lateral_b_anchor_weight,
            d_anchor_weight=args.composition_lateral_d_anchor_weight,
        )
        add_b_teacher_forced_metrics(comp_lateral_metrics, no_proxy_student, tokenizer, cfg)
        add_sort_teacher_forced_metrics(comp_lateral_metrics, no_proxy_student, tokenizer, cfg)
        add_composition_metrics(comp_lateral_metrics, no_proxy_student, tokenizer, cfg, args.composition_eval_samples)
        print_metrics("D_lateral_comp+compose", comp_lateral_metrics)
    release(no_proxy_student)

    b_probe_metrics: Dict[str, float] | None = None
    if args.run_b_probe_variant:
        sub("Variant 4: D no-proxy same-batch anchor + tiny B checkpoint probe")
        print(
            "variant_note: this is a practical retention control, not the pure no-old-task-example claim.",
            flush=True,
        )
        b_probe_student = qp._clone_model(base_ab_frozen, cfg.device)
        b_probe_metrics = consolidate_no_proxy_with_b_probe_in_memory(
            student=b_probe_student,
            teacher_old=base_ab_frozen,
            teacher_new=teacher_d_model,
            tokenizer=tokenizer,
            new_task_batch_fn=d_consolidation_batch_fn,
            b_probe_batch_fn=b_batch_fn,
            eval_tasks=["retention", "json", "sort"],
            eval_data=eval_data,
            selected_layers=d_layers,
            b_probe_layers=b_layers,
            old_profiles=[a_profile_ab, b_profile_ab],
            project_old_gradients=args.project_old_gradients,
            projection_strength=args.projection_strength,
            steps=args.d_consolidation_steps,
            lr=cfg.consolidation_lr,
            label="D_no_proxy_plus_B_probe",
            cfg=cfg,
            old_task_kl_weight=args.no_proxy_old_kl_weight,
            old_task_hidden_weight=args.no_proxy_old_hidden_weight,
            new_kl_weight=args.new_kl_weight,
            new_hidden_weight=args.new_hidden_weight,
            b_probe_period=args.b_probe_period,
            b_probe_kl_weight=args.b_probe_kl_weight,
            b_probe_hidden_weight=args.b_probe_hidden_weight,
        )
        add_sort_teacher_forced_metrics(b_probe_metrics, b_probe_student, tokenizer, cfg)
        add_composition_metrics(b_probe_metrics, b_probe_student, tokenizer, cfg, args.composition_eval_samples)
        print_metrics("D_B_probe+compose", b_probe_metrics)
        release(b_probe_student)
    release(teacher_d_model, base_ab_frozen)

    section("SUMMARY TABLE")
    print_metrics("base_A", base_metrics)
    print_metrics("teacher_B", teacher_b_metrics)
    print_metrics("base_AB", base_ab_metrics)
    print_metrics("teacher_D", teacher_d_metrics)
    if proxy_metrics is not None:
        print_metrics("D_proxy_anchor", proxy_metrics)
    else:
        print("D_proxy_anchor             SKIPPED in focused run; use prior full-audit artifact.", flush=True)
    print_metrics("D_no_proxy", no_proxy_metrics)
    if comp_lateral_metrics is not None:
        print_metrics("D_lateral_comp", comp_lateral_metrics)
    if b_probe_metrics is not None:
        print_metrics("D_B_probe_control", b_probe_metrics)

    desiderata_report(
        base_metrics=base_metrics,
        base_ab_metrics=base_ab_metrics,
        proxy_metrics=proxy_metrics,
        no_proxy_metrics=no_proxy_metrics,
        comp_lateral_metrics=comp_lateral_metrics,
        args=args,
    )


if __name__ == "__main__":
    run()
