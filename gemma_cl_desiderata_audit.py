from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

import qwen_cl_desiderata_audit as audit
import qwen_continual_proof as qp
from standalone_latent_lora_qwen import (
    LatentLoRAConfig,
    attach_latent_lora,
    choose_dtype,
    load_causal_lm,
    load_tokenizer,
)


TRANSPARENT_ROUTES: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("123", (0, 1, 2)),
    ("231", (1, 2, 0)),
    ("312", (2, 0, 1)),
    ("321", (2, 1, 0)),
    ("213", (1, 0, 2)),
    ("132", (0, 2, 1)),
)
OPAQUE_ROUTES: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("alpha", (0, 1, 2)),
    ("bravo", (1, 2, 0)),
    ("charlie", (2, 0, 1)),
    ("delta", (2, 1, 0)),
    ("echo", (1, 0, 2)),
    ("foxtrot", (0, 2, 1)),
)

TRAIN_FIELDS = (
    "red",
    "blue",
    "green",
    "gold",
    "black",
    "white",
    "north",
    "south",
    "east",
    "west",
    "sun",
    "moon",
)
HELDOUT_FIELDS = (
    "river",
    "forest",
    "stone",
    "cloud",
    "paper",
    "glass",
    "winter",
    "summer",
    "circle",
    "square",
    "orange",
    "silver",
)
TRAIN_VALUES = (
    "cat",
    "dog",
    "bird",
    "fish",
    "horse",
    "mouse",
    "lion",
    "tiger",
    "bear",
    "wolf",
    "apple",
    "bread",
)
HELDOUT_VALUES = (
    "chair",
    "table",
    "piano",
    "drum",
    "river",
    "cloud",
    "stone",
    "glass",
    "paper",
    "flame",
    "brush",
    "spoon",
)
SLOTS = ("one", "two", "three")
SORT_LABELS = TRAIN_VALUES


def line(char: str = "=") -> None:
    print(char * 96, flush=True)


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


def parse_simple_record_payload(text: str) -> Dict[str, str] | None:
    # Gemma 270M sometimes repeats the prompt. Only accept the first clean
    # one/two/three assignment for each slot.
    normalized = str(text).replace(",", ";")
    end_idx = normalized.upper().find("END")
    if end_idx >= 0:
        normalized = normalized[:end_idx]
    payload: Dict[str, str] = {}
    # Regex extraction is deliberately more tolerant than semicolon splitting:
    # Gemma often emits "one=bird =two=lion" or newline-separated assignments.
    for match in re.finditer(r"(?<![A-Za-z0-9_])(one|two|three)\s*=\s*([A-Za-z]+)", normalized, re.IGNORECASE):
        key = match.group(1).lower()
        value = match.group(2).strip(".,;:/").lower()
        if key in SLOTS and key not in payload and value:
            payload[key] = value
    return payload if payload else None


def _field_triplet(rng: np.random.Generator, heldout: bool) -> Tuple[str, str, str]:
    bank = HELDOUT_FIELDS if heldout else TRAIN_FIELDS
    start = int(rng.integers(0, len(bank)))
    return tuple(bank[(start + offset) % len(bank)] for offset in range(3))


def _route_table(opaque_routes: bool) -> Tuple[Tuple[str, Tuple[int, int, int]], ...]:
    return OPAQUE_ROUTES if opaque_routes else TRANSPARENT_ROUTES


def gemma_record_example(
    rng: np.random.Generator,
    *,
    heldout: bool,
    eval_style: bool,
    opaque_routes: bool = False,
    explicit_route_instructions: bool = True,
    heldout_values_shared: bool = True,
) -> Tuple[str, str]:
    fields = _field_triplet(rng, heldout)
    routes = _route_table(opaque_routes)
    route, perm = routes[int(rng.integers(0, len(routes)))]
    # Tiny Gemma struggles to copy never-trained output words in free
    # generation. By default, hold out field names/examples but keep the value
    # vocabulary shared so the scout tests routing acquisition first.
    value_bank = HELDOUT_VALUES if heldout and not heldout_values_shared else TRAIN_VALUES
    chosen_values = rng.choice(value_bank, size=len(fields), replace=False)
    assignments = {field: str(value) for field, value in zip(fields, chosen_values)}
    shuffled = list(fields)
    rng.shuffle(shuffled)
    assignment_text = " ; ".join(f"{field}={assignments[field]}" for field in shuffled)
    fields_text = ", ".join(f"{idx + 1}:{field}" for idx, field in enumerate(fields))
    indexed_bindings = " ; ".join(f"{idx + 1}:{field}={assignments[field]}" for idx, field in enumerate(fields))
    route_instruction = (
        f"one uses field {perm[0] + 1}; "
        f"two uses field {perm[1] + 1}; "
        f"three uses field {perm[2] + 1}"
    )
    if explicit_route_instructions and eval_style:
        templates = (
            "Apply route {route}. Rule: {route_instruction}. Records: {indexed_bindings}. Return one/two/three then END:",
            "Use map {route}. It means {route_instruction}. Items: {indexed_bindings}. Output one, two, three, then END:",
        )
    elif explicit_route_instructions:
        templates = (
            "Route {route}. Rule: {route_instruction}. Records: {indexed_bindings}. Return one, two, three, then END:",
            "Use route {route}. It means {route_instruction}. Items: {indexed_bindings}. Answer one, two, three, then END:",
        )
    elif eval_style:
        templates = (
            "Apply route {route}. Field order {fields}. Observed bindings: {assignments}. Return one/two/three then END:",
            "Use map {route}; fields {fields}; data {assignments}. Output one, two, three, then END:",
        )
    else:
        templates = (
            "Route {route}. Fields: {fields}. Payload: {assignments}. Return one, two, three, then END:",
            "Use route {route}. Given fields {fields}. Bindings: {assignments}. Answer one, two, three, then END:",
        )
    prompt = templates[int(rng.integers(0, len(templates)))].format(
        route=route,
        fields=fields_text,
        assignments=assignment_text,
        indexed_bindings=indexed_bindings,
        route_instruction=route_instruction,
    )
    routed_values = [assignments[fields[idx]] for idx in perm]
    target = " ; ".join(f"{slot}={value}" for slot, value in zip(SLOTS, routed_values)) + " ; END"
    return f"{prompt}\n", target


def make_gemma_b_batch_fn(tokenizer, device: str, cfg: qp.RuntimeConfig, seed: int) -> Callable[[int], Dict[str, torch.Tensor]]:
    opaque_routes = bool(getattr(cfg, "gemma_opaque_routes", False))
    explicit_route_instructions = bool(getattr(cfg, "gemma_explicit_route_instructions", True))
    eval_style_train_frac = float(getattr(cfg, "gemma_b_eval_style_train_frac", 0.0))
    heldout_values_shared = bool(getattr(cfg, "gemma_heldout_values_shared", True))

    def _batch(step: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(seed + 2003 * int(step))
        prompts: List[str] = []
        targets: List[str] = []
        for _ in range(cfg.batch_size):
            eval_style = bool(eval_style_train_frac > 0.0 and rng.random() < eval_style_train_frac)
            prompt, target = gemma_record_example(
                rng,
                heldout=False,
                eval_style=eval_style,
                opaque_routes=opaque_routes,
                explicit_route_instructions=explicit_route_instructions,
                heldout_values_shared=heldout_values_shared,
            )
            prompts.append(prompt)
            targets.append(f"{target}{tokenizer.eos_token}")
        return qp._prepare_supervised_batch(tokenizer, prompts, targets, device, cfg.max_seq_len)

    return _batch


def _rows(
    seed: int,
    total: int,
    *,
    heldout: bool,
    eval_style: bool,
    opaque_routes: bool = False,
    explicit_route_instructions: bool = True,
    heldout_values_shared: bool = True,
) -> List[Tuple[str, str]]:
    return [
        gemma_record_example(
            np.random.default_rng(seed + idx),
            heldout=heldout,
            eval_style=eval_style,
            opaque_routes=opaque_routes,
            explicit_route_instructions=explicit_route_instructions,
            heldout_values_shared=heldout_values_shared,
        )
        for idx in range(max(int(total), 1))
    ]


def gemma_sort_example(
    rng: np.random.Generator,
    *,
    heldout: bool,
    eval_style: bool,
    item_count: int = 3,
) -> Tuple[str, str]:
    """Gemma-friendly stable sort.

    Qwen can tolerate the proof_v2 stable-sort surface form. Gemma-270M is much
    more format-sensitive, so this keeps the same skill (ascending stable sort)
    while using short common tokens and an explicit output schema.
    """
    count = min(max(int(item_count), 2), len(SLOTS))
    labels = [str(item) for item in rng.choice(SORT_LABELS, size=count, replace=False)]
    # Keys are intentionally tiny and sometimes duplicated. Duplicates make this
    # a stable-sort task rather than a pure permutation lookup.
    keys = [int(x) for x in rng.integers(1, 4, size=count)]
    if count >= 3 and rng.random() < 0.35:
        keys[int(rng.integers(1, count))] = keys[0]
    pairs = list(zip(labels, keys))
    compact_records = " ; ".join(f"{label}={key}" for label, key in pairs)
    indexed_records = " ; ".join(
        f"{idx + 1}:{label} has key {key}" for idx, (label, key) in enumerate(pairs)
    )
    if eval_style:
        templates = (
            "Stable sort ascending. Items: {indexed_records}. Output one, two, three, then END:",
            "Order by small key first. Data: {compact_records}. Return one/two/three then END:",
        )
    else:
        templates = (
            "Sort by number, smallest first. Records: {compact_records}. Return one, two, three, then END:",
            "Stable sort by key ascending. Items: {indexed_records}. Return one, two, three, then END:",
        )
    prompt = templates[int(rng.integers(0, len(templates)))].format(
        compact_records=compact_records,
        indexed_records=indexed_records,
    )
    ordered_indices = sorted(range(count), key=lambda idx: (keys[idx], idx))
    target = " ; ".join(
        f"{slot}={labels[idx]}" for slot, idx in zip(SLOTS[:count], ordered_indices)
    ) + " ; END"
    return f"{prompt}\n", target


def make_gemma_sort_batch_fn(
    tokenizer,
    device: str,
    cfg: qp.RuntimeConfig,
    seed: int,
) -> Callable[[int], Dict[str, torch.Tensor]]:
    eval_style_train_frac = float(getattr(cfg, "gemma_sort_eval_style_train_frac", 0.0))
    item_count = int(getattr(cfg, "gemma_sort_items", 3))

    def _batch(step: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(seed + 3001 * int(step))
        prompts: List[str] = []
        targets: List[str] = []
        for _ in range(cfg.batch_size):
            eval_style = bool(eval_style_train_frac > 0.0 and rng.random() < eval_style_train_frac)
            prompt, target = gemma_sort_example(
                rng,
                heldout=False,
                eval_style=eval_style,
                item_count=item_count,
            )
            prompts.append(prompt)
            targets.append(f"{target}{tokenizer.eos_token}")
        return qp._prepare_supervised_batch(tokenizer, prompts, targets, device, cfg.max_seq_len)

    return _batch


def _sort_rows(
    seed: int,
    total: int,
    *,
    heldout: bool,
    eval_style: bool,
    item_count: int,
) -> List[Tuple[str, str]]:
    return [
        gemma_sort_example(
            np.random.default_rng(seed + idx),
            heldout=heldout,
            eval_style=eval_style,
            item_count=item_count,
        )
        for idx in range(max(int(total), 1))
    ]


@torch.no_grad()
def teacher_forced_token_acc(
    model,
    tokenizer,
    cfg: qp.RuntimeConfig,
    rows: Sequence[Tuple[str, str]],
) -> float:
    return audit._teacher_forced_sequence_token_acc(
        model,
        tokenizer,
        cfg.device,
        rows,
        cfg.eval_batch_size,
        cfg,
    )


@torch.no_grad()
def evaluate_gemma_b(model, tokenizer, cfg: qp.RuntimeConfig) -> Dict[str, float]:
    model.eval()
    total = max(int(cfg.json_eval_samples), 1)
    opaque_routes = bool(getattr(cfg, "gemma_opaque_routes", False))
    explicit_route_instructions = bool(getattr(cfg, "gemma_explicit_route_instructions", True))
    heldout_values_shared = bool(getattr(cfg, "gemma_heldout_values_shared", True))
    heldout_rows = _rows(
        9100,
        total,
        heldout=True,
        eval_style=True,
        opaque_routes=opaque_routes,
        explicit_route_instructions=explicit_route_instructions,
        heldout_values_shared=heldout_values_shared,
    )
    train_rows = _rows(
        8100,
        total,
        heldout=False,
        eval_style=False,
        opaque_routes=opaque_routes,
        explicit_route_instructions=explicit_route_instructions,
        heldout_values_shared=heldout_values_shared,
    )
    train_eval_style_rows = _rows(
        8500,
        total,
        heldout=False,
        eval_style=True,
        opaque_routes=opaque_routes,
        explicit_route_instructions=explicit_route_instructions,
        heldout_values_shared=heldout_values_shared,
    )
    heldout = qp._evaluate_b_examples(
        model,
        tokenizer,
        cfg.device,
        heldout_rows,
        cfg.eval_batch_size,
        parser=parse_simple_record_payload,
    )
    train = qp._evaluate_b_examples(
        model,
        tokenizer,
        cfg.device,
        train_rows,
        cfg.eval_batch_size,
        parser=parse_simple_record_payload,
    )
    train_eval_style = qp._evaluate_b_examples(
        model,
        tokenizer,
        cfg.device,
        train_eval_style_rows,
        cfg.eval_batch_size,
        parser=parse_simple_record_payload,
    )
    metrics = {
        "json_exact_match": heldout["exact"],
        "json_valid": heldout["valid"],
        "json_field_acc": heldout["field_acc"],
        "json_loss": heldout["loss"],
        "json_train_exact_match": train["exact"],
        "json_train_valid": train["valid"],
        "json_train_field_acc": train["field_acc"],
        "json_train_eval_style_exact_match": train_eval_style["exact"],
        "json_train_eval_style_valid": train_eval_style["valid"],
        "json_train_eval_style_field_acc": train_eval_style["field_acc"],
        "json_teacher_forced_token_acc": teacher_forced_token_acc(model, tokenizer, cfg, heldout_rows),
        "json_train_teacher_forced_token_acc": teacher_forced_token_acc(model, tokenizer, cfg, train_rows),
        "json_train_eval_style_teacher_forced_token_acc": teacher_forced_token_acc(
            model,
            tokenizer,
            cfg,
            train_eval_style_rows,
        ),
    }
    model.train()
    return metrics


@torch.no_grad()
def evaluate_gemma_sort(model, tokenizer, cfg: qp.RuntimeConfig) -> Dict[str, float]:
    model.eval()
    total = max(int(cfg.sort_eval_samples), 1)
    item_count = int(getattr(cfg, "gemma_sort_items", 3))
    heldout_rows = _sort_rows(12100, total, heldout=True, eval_style=True, item_count=item_count)
    train_rows = _sort_rows(11100, total, heldout=False, eval_style=False, item_count=item_count)
    train_eval_style_rows = _sort_rows(11500, total, heldout=False, eval_style=True, item_count=item_count)
    heldout = qp._evaluate_b_examples(
        model,
        tokenizer,
        cfg.device,
        heldout_rows,
        cfg.eval_batch_size,
        parser=parse_simple_record_payload,
    )
    train = qp._evaluate_b_examples(
        model,
        tokenizer,
        cfg.device,
        train_rows,
        cfg.eval_batch_size,
        parser=parse_simple_record_payload,
    )
    train_eval_style = qp._evaluate_b_examples(
        model,
        tokenizer,
        cfg.device,
        train_eval_style_rows,
        cfg.eval_batch_size,
        parser=parse_simple_record_payload,
    )
    metrics = {
        "sort_token_acc": heldout["field_acc"],
        "sort_train_token_acc": train["field_acc"],
        "sort_train_eval_style_token_acc": train_eval_style["field_acc"],
        "sort_teacher_forced_token_acc": teacher_forced_token_acc(model, tokenizer, cfg, heldout_rows),
        "sort_train_teacher_forced_token_acc": teacher_forced_token_acc(model, tokenizer, cfg, train_rows),
        "sort_train_eval_style_teacher_forced_token_acc": teacher_forced_token_acc(
            model,
            tokenizer,
            cfg,
            train_eval_style_rows,
        ),
        "sort_loss": heldout["loss"],
        "sort_train_loss": train["loss"],
        "sort_exact_match": heldout["exact"],
        "sort_exact": heldout["exact"],
        "sort_valid": heldout["valid"],
    }
    model.train()
    return metrics


@torch.no_grad()
def evaluate_retention(model, tokenizer, cfg: qp.RuntimeConfig) -> Dict[str, float]:
    chunks = qp.load_wikitext_texts(
        tokenizer,
        split="validation",
        max_seq_len=cfg.max_seq_len,
        max_samples=cfg.wikitext_eval_samples,
        local_files_only=cfg.local_files_only,
    )
    return qp.evaluate_retention(model, tokenizer, chunks, cfg.device, cfg.eval_batch_size)


def print_gemma_metrics(label: str, metrics: Dict[str, float]) -> None:
    print(
        f"{label:<28} "
        f"ppl={fmt(metrics.get('wikitext_ppl')):>7} "
        f"b_field={fmt(metrics.get('json_field_acc')):>7} "
        f"b_train={fmt(metrics.get('json_train_field_acc')):>7} "
        f"b_style={fmt(metrics.get('json_train_eval_style_field_acc')):>7} "
        f"b_valid={fmt(metrics.get('json_valid')):>7} "
        f"b_tf={fmt(metrics.get('json_teacher_forced_token_acc')):>7} "
        f"b_train_tf={fmt(metrics.get('json_train_teacher_forced_token_acc')):>7}",
        flush=True,
    )


def print_full_metrics(label: str, metrics: Dict[str, float]) -> None:
    audit.print_metrics(label, metrics)
    print(
        f"{label:<26} gemma_extra "
        f"b_style={fmt(metrics.get('json_train_eval_style_field_acc')):>7} "
        f"b_style_tf={fmt(metrics.get('json_train_eval_style_teacher_forced_token_acc')):>7} "
        f"sort_style={fmt(metrics.get('sort_train_eval_style_token_acc')):>7} "
        f"sort_style_tf={fmt(metrics.get('sort_train_eval_style_teacher_forced_token_acc')):>7}",
        flush=True,
    )


@torch.no_grad()
def evaluate_gemma_world(
    model,
    tokenizer,
    cfg: qp.RuntimeConfig,
    tasks: Sequence[str],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if "retention" in tasks:
        metrics.update(evaluate_retention(model, tokenizer, cfg))
    if "json" in tasks:
        metrics.update(evaluate_gemma_b(model, tokenizer, cfg))
    if "sort" in tasks:
        if bool(getattr(cfg, "gemma_friendly_sort", True)):
            metrics.update(evaluate_gemma_sort(model, tokenizer, cfg))
        else:
            metrics.update(qp.evaluate_sort(model, tokenizer, cfg.device, cfg.sort_eval_samples, cfg.eval_batch_size, cfg))
            audit.add_sort_teacher_forced_metrics(metrics, model, tokenizer, cfg)
    return metrics


def train_gemma_b_teacher(
    *,
    model,
    tokenizer,
    attached,
    batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    cfg: qp.RuntimeConfig,
    steps: int,
    lr: float,
) -> Dict[str, float]:
    params = qp._trainable_params(model)
    optimizer = torch.optim.AdamW(params, lr=lr)
    best_score: Tuple[float, ...] | None = None
    best_step = 0
    best_state = None
    start = time.time()
    if cfg.device.startswith("cuda"):
        qp._configure_gradient_checkpointing(model, cfg.gradient_checkpointing)
    model.train()
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = batch_fn(step)
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        qp.ESCAPE_SCHEDULE.apply_to_modules(attached, step, int(steps))
        if step % cfg.log_interval == 0 or step == steps:
            print(
                f"[gemma_B_no_old_examples] step={step:04d}/{steps} "
                f"loss={float(loss.item()):.4f} old_teacher_batches=no",
                flush=True,
            )
        if step % cfg.eval_interval == 0 or step == steps:
            probe = evaluate_gemma_b(model, tokenizer, cfg)
            train_field = float(probe.get("json_train_field_acc", 0.0))
            style_field = float(probe.get("json_train_eval_style_field_acc", 0.0))
            heldout_field = float(probe.get("json_field_acc", 0.0))
            score = (
                min(train_field, style_field),
                style_field,
                train_field,
                heldout_field,
                float(probe.get("json_teacher_forced_token_acc", 0.0)),
                -float(probe.get("json_loss", 99.0)),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_step = step
                best_state = audit._capture_trainable_state(model)
                print(
                    f"[gemma_B_no_old_examples] best_update step={step:04d}/{steps} "
                    f"score={tuple(round(float(v), 4) for v in score)}",
                    flush=True,
                )
    if best_state is not None:
        audit._restore_trainable_state(model, best_state)
        print(f"[gemma_B_no_old_examples] restored_best_step={best_step:04d}/{steps}", flush=True)
    metrics = evaluate_retention(model, tokenizer, cfg)
    metrics.update(evaluate_gemma_b(model, tokenizer, cfg))
    print_gemma_metrics("gemma_B_no_old_examples", metrics)
    print(f"[gemma_B_no_old_examples] wall_time_sec={time.time() - start:.1f}", flush=True)
    return metrics


def train_gemma_d_teacher(
    *,
    model,
    tokenizer,
    attached,
    batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    cfg: qp.RuntimeConfig,
    steps: int,
    lr: float,
) -> Dict[str, float]:
    params = qp._trainable_params(model)
    optimizer = torch.optim.AdamW(params, lr=lr)
    best_score: Tuple[float, ...] | None = None
    best_step = 0
    best_state = None
    start = time.time()
    if cfg.device.startswith("cuda"):
        qp._configure_gradient_checkpointing(model, cfg.gradient_checkpointing)
    model.train()
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = batch_fn(step)
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        qp.ESCAPE_SCHEDULE.apply_to_modules(attached, step, int(steps))
        if step % cfg.log_interval == 0 or step == steps:
            print(
                f"[gemma_D_no_old_examples] step={step:04d}/{steps} "
                f"loss={float(loss.item()):.4f} old_teacher_batches=no",
                flush=True,
            )
        if step % cfg.eval_interval == 0 or step == steps:
            probe = evaluate_gemma_sort(model, tokenizer, cfg)
            heldout_tok = float(probe.get("sort_token_acc", 0.0))
            train_tok = float(probe.get("sort_train_token_acc", 0.0))
            style_tok = float(probe.get("sort_train_eval_style_token_acc", 0.0))
            score = (
                min(train_tok, style_tok),
                heldout_tok,
                style_tok,
                train_tok,
                float(probe.get("sort_teacher_forced_token_acc", 0.0)),
                -float(probe.get("sort_loss", 99.0)),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_step = step
                best_state = audit._capture_trainable_state(model)
                print(
                    f"[gemma_D_no_old_examples] best_update step={step:04d}/{steps} "
                    f"score={tuple(round(float(v), 4) for v in score)}",
                    flush=True,
                )
    if best_state is not None:
        audit._restore_trainable_state(model, best_state)
        print(f"[gemma_D_no_old_examples] restored_best_step={best_step:04d}/{steps}", flush=True)
    metrics = evaluate_retention(model, tokenizer, cfg)
    metrics.update(evaluate_gemma_sort(model, tokenizer, cfg))
    print_full_metrics("gemma_D_no_old_examples", metrics)
    print(f"[gemma_D_no_old_examples] wall_time_sec={time.time() - start:.1f}", flush=True)
    return metrics


def gemma_teacher_gate(
    *,
    teacher_b_metrics: Dict[str, float],
    base_ab_metrics: Dict[str, float],
    teacher_d_metrics: Dict[str, float],
    args: argparse.Namespace,
) -> bool:
    line("=")
    print("GEMMA TEACHER ACQUISITION GATE", flush=True)
    line("=")
    b_teacher = float(teacher_b_metrics.get("json_field_acc", 0.0))
    b_ab = float(base_ab_metrics.get("json_field_acc", 0.0))
    d_tok = float(teacher_d_metrics.get("sort_token_acc", 0.0))
    d_train_tok = float(teacher_d_metrics.get("sort_train_token_acc", 0.0))
    d_tf = float(teacher_d_metrics.get("sort_teacher_forced_token_acc", 0.0))
    d_loss = float(teacher_d_metrics.get("sort_loss", 999.0))
    b_ok = b_teacher >= float(args.min_teacher_b_field) and b_ab >= float(args.min_base_ab_b_field)
    d_ok = (
        d_tok >= float(args.min_teacher_d_sort_tok)
        or d_train_tok >= float(args.min_teacher_d_train_tok)
        or d_tf >= float(args.min_teacher_d_tf_tok)
        or d_loss <= float(args.max_teacher_d_sort_loss)
    )
    print(
        f"B teacher: {'PASS' if b_teacher >= args.min_teacher_b_field else 'FAIL'} "
        f"teacher_field={fmt(b_teacher)} need>={args.min_teacher_b_field:.3f}",
        flush=True,
    )
    print(
        f"AB consolidated B: {'PASS' if b_ab >= args.min_base_ab_b_field else 'FAIL'} "
        f"base_AB_field={fmt(b_ab)} need>={args.min_base_ab_b_field:.3f}",
        flush=True,
    )
    print(
        f"D teacher: {'PASS' if d_ok else 'FAIL'} "
        f"heldout_sort_tok={fmt(d_tok)} train_sort_tok={fmt(d_train_tok)} "
        f"heldout_tf_tok={fmt(d_tf)} sort_loss={fmt(d_loss)}",
        flush=True,
    )
    return bool(b_ok and d_ok)


def gemma_preserve_and_learn(
    metrics: Dict[str, float],
    base_ab_metrics: Dict[str, float],
    args: argparse.Namespace,
) -> Tuple[bool, bool, float]:
    ab_b = float(base_ab_metrics.get("json_field_acc", 0.0))
    model_b = float(metrics.get("json_field_acc", 0.0))
    model_sort = float(metrics.get("sort_token_acc", 0.0))
    model_sort_loss = float(metrics.get("sort_loss", 999.0))
    ab_ppl = float(base_ab_metrics.get("wikitext_ppl", 999.0))
    model_ppl = float(metrics.get("wikitext_ppl", 999.0))
    preserve = model_b >= ab_b - float(args.max_b_drop) and model_ppl <= ab_ppl * float(args.max_ppl_ratio)
    learns = model_sort >= float(args.min_sort_tok) or model_sort_loss <= float(args.max_sort_loss)
    # A compact selector for rescue variants: prioritize satisfying both gates,
    # then prefer B/PPL preservation while keeping D above the learning floor.
    ppl_penalty = max(0.0, (model_ppl / max(ab_ppl, 1e-9)) - float(args.max_ppl_ratio))
    b_penalty = max(0.0, (ab_b - model_b) - float(args.max_b_drop))
    score = (
        (10.0 if preserve else 0.0)
        + (10.0 if learns else 0.0)
        + model_sort
        - model_sort_loss
        + model_b
        - 2.0 * ppl_penalty
        - 3.0 * b_penalty
    )
    return bool(preserve), bool(learns), float(score)


def print_debug(model, tokenizer, cfg: qp.RuntimeConfig, *, label: str, limit: int = 3) -> None:
    opaque_routes = bool(getattr(cfg, "gemma_opaque_routes", False))
    explicit_route_instructions = bool(getattr(cfg, "gemma_explicit_route_instructions", True))
    heldout_values_shared = bool(getattr(cfg, "gemma_heldout_values_shared", True))
    train_rows = _rows(
        8100,
        limit,
        heldout=False,
        eval_style=False,
        opaque_routes=opaque_routes,
        explicit_route_instructions=explicit_route_instructions,
        heldout_values_shared=heldout_values_shared,
    )
    train_eval_style_rows = _rows(
        8500,
        limit,
        heldout=False,
        eval_style=True,
        opaque_routes=opaque_routes,
        explicit_route_instructions=explicit_route_instructions,
        heldout_values_shared=heldout_values_shared,
    )
    heldout_rows = _rows(
        9100,
        limit,
        heldout=True,
        eval_style=True,
        opaque_routes=opaque_routes,
        explicit_route_instructions=explicit_route_instructions,
        heldout_values_shared=heldout_values_shared,
    )
    for split, rows in (("train", train_rows), ("train_eval_style", train_eval_style_rows), ("heldout", heldout_rows)):
        prompts = [prompt for prompt, _ in rows]
        completions = qp._generate_batch_tokens(model, tokenizer, prompts, cfg.device, max_new_tokens=48)
        for idx, ((prompt, target), completion) in enumerate(zip(rows, completions), start=1):
            print(
                f"[{label}:DEBUG:{split}:{idx}] "
                f"prompt={prompt.strip()} | target={target} | completion={completion} | "
                f"parsed={parse_simple_record_payload(completion)}",
                flush=True,
            )


def run_full_gemma_audit(args: argparse.Namespace, cfg: qp.RuntimeConfig, tokenizer, base_model) -> None:
    gemma_friendly_sort = bool(getattr(cfg, "gemma_friendly_sort", True))
    wikitext_val = qp.load_wikitext_texts(
        tokenizer,
        split="validation",
        max_seq_len=cfg.max_seq_len,
        max_samples=cfg.wikitext_eval_samples,
        local_files_only=cfg.local_files_only,
    )
    wikitext_train = qp.load_wikitext_texts(
        tokenizer,
        split="train",
        max_seq_len=cfg.max_seq_len,
        max_samples=max(int(cfg.wikitext_eval_samples), 64),
        local_files_only=cfg.local_files_only,
    )
    eval_data = {"wikitext_val": wikitext_val}
    proxy_batch_fn = qp.make_wikitext_batch_fn(tokenizer, wikitext_train, cfg.device, cfg, cfg.seed + 11)
    b_batch_fn = make_gemma_b_batch_fn(tokenizer, cfg.device, cfg, cfg.seed + 101)
    if gemma_friendly_sort:
        d_teacher_batch_fn = make_gemma_sort_batch_fn(tokenizer, cfg.device, cfg, cfg.seed + 202)
        d_consolidation_batch_fn = make_gemma_sort_batch_fn(tokenizer, cfg.device, cfg, cfg.seed + 303)
        d_eval_tasks = ["retention"]
    else:
        d_teacher_batch_fn = audit.make_audit_sort_batch_fn(
            tokenizer,
            cfg.device,
            cfg,
            cfg.seed + 202,
            max_train_len=args.d_train_max_len,
            schedule_total_steps=cfg.proof_v2_d_attach_steps,
        )
        d_consolidation_batch_fn = audit.make_audit_sort_batch_fn(
            tokenizer,
            cfg.device,
            cfg,
            cfg.seed + 303,
            max_train_len=args.d_train_max_len,
            schedule_total_steps=args.d_consolidation_steps,
        )
        d_eval_tasks = ["retention", "sort"]

    sub("Stage A: base model evaluation")
    base_metrics = evaluate_gemma_world(base_model, tokenizer, cfg, ["retention", "json"])
    print_full_metrics("base_A", base_metrics)

    base_frozen = qp._clone_model(base_model, cfg.device)
    qp._freeze_model(base_frozen)
    a_profile = qp._collect_profiles(base_frozen, "retention", proxy_batch_fn)

    sub("Stage B: train Gemma B adapter teacher")
    if args.all_layers:
        layer_count = len(getattr(getattr(base_model, "model", None), "layers", []))
        b_layers = list(range(layer_count)) if layer_count > 0 else None
        print(f"[gemma_layers:B] using all layers count={layer_count}", flush=True)
    else:
        b_layers, _ = audit.select_layers(
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
        layer_indices=None if b_layers is None else set(b_layers),
        config=LatentLoRAConfig(
            rank=cfg.proof_v2_b_rank,
            alpha=cfg.proof_v2_b_alpha,
            dropout=0.0,
            projection_strength=1.0,
            gate_init=cfg.proof_v2_b_gate_init,
            freeze_base=True,
        ),
    )
    print(f"[gemma_lora:B] attached_modules={len(attached_b)}", flush=True)
    teacher_b_metrics = train_gemma_b_teacher(
        model=base_model,
        tokenizer=tokenizer,
        attached=attached_b,
        batch_fn=b_batch_fn,
        cfg=cfg,
        steps=cfg.proof_v2_b_attach_steps,
        lr=cfg.proof_v2_b_attach_lr,
    )
    print_full_metrics("teacher_B", teacher_b_metrics)
    if args.stop_after_b_teacher:
        print("stopped: --stop-after-b-teacher was set.", flush=True)
        return

    if args.abort_if_weak_teacher and float(teacher_b_metrics.get("json_field_acc", 0.0)) < float(args.min_teacher_b_field):
        print("stopped: B teacher gate failed; skipped AB/D to avoid wasting hours.", flush=True)
        return

    sub("Stage AB: consolidate B into base weights with generic proxy anchor")
    base_ab = qp._clone_model(base_frozen, cfg.device)
    base_ab_metrics = audit.consolidate_with_proxy_in_memory(
        student=base_ab,
        teacher_old=base_frozen,
        teacher_new=base_model,
        tokenizer=tokenizer,
        new_task_batch_fn=b_batch_fn,
        proxy_batch_fn=proxy_batch_fn,
        eval_tasks=["retention"],
        eval_data=eval_data,
        selected_layers=[] if b_layers is None else b_layers,
        old_profiles=[a_profile],
        project_old_gradients=args.project_old_gradients,
        projection_strength=args.projection_strength,
        steps=args.b_consolidation_steps,
        lr=cfg.consolidation_lr,
        label="base_AB_proxy_anchor",
        cfg=cfg,
    )
    base_ab_metrics.update(evaluate_gemma_b(base_ab, tokenizer, cfg))
    print_full_metrics("base_AB+gemma_B", base_ab_metrics)

    qp._release_cuda_memory(base_frozen, base_model)

    sub("Stage D: train D adapter teacher from base_AB")
    base_ab_frozen = qp._clone_model(base_ab, cfg.device)
    qp._freeze_model(base_ab_frozen)
    a_profile_ab = qp._collect_profiles(base_ab, "retention", proxy_batch_fn)
    b_profile_ab = qp._collect_profiles(base_ab, "gemma_b", b_batch_fn)
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
    print(f"[gemma_lora:D] attached_modules={len(attached_d)}", flush=True)
    if gemma_friendly_sort:
        teacher_d_metrics = train_gemma_d_teacher(
            model=base_ab,
            tokenizer=tokenizer,
            attached=attached_d,
            batch_fn=d_teacher_batch_fn,
            cfg=cfg,
            steps=cfg.proof_v2_d_attach_steps,
            lr=cfg.proof_v2_d_attach_lr,
        )
    else:
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
            label="teacher_D_no_old_examples",
            cfg=cfg,
        )
    teacher_d_metrics.update(evaluate_gemma_b(base_ab, tokenizer, cfg))
    if not gemma_friendly_sort:
        audit.add_sort_teacher_forced_metrics(teacher_d_metrics, base_ab, tokenizer, cfg)
    print_full_metrics("teacher_D+gemma_B", teacher_d_metrics)

    teachers_ok = gemma_teacher_gate(
        teacher_b_metrics=teacher_b_metrics,
        base_ab_metrics=base_ab_metrics,
        teacher_d_metrics=teacher_d_metrics,
        args=args,
    )
    if args.stop_after_teachers or (args.abort_if_weak_teacher and not teachers_ok):
        line("=")
        print("SUMMARY TABLE", flush=True)
        line("=")
        print_full_metrics("base_A", base_metrics)
        print_full_metrics("teacher_B", teacher_b_metrics)
        print_full_metrics("base_AB", base_ab_metrics)
        print_full_metrics("teacher_D", teacher_d_metrics)
        if args.stop_after_teachers:
            print("stopped: --stop-after-teachers was set.", flush=True)
        else:
            print("stopped: teacher acquisition gate failed; skipped D consolidation.", flush=True)
        return

    if not args.skip_proxy_variant:
        sub("Variant 1: D consolidation with generic proxy anchor")
        proxy_student = qp._clone_model(base_ab_frozen, cfg.device)
        proxy_metrics = audit.consolidate_with_proxy_in_memory(
            student=proxy_student,
            teacher_old=base_ab_frozen,
            teacher_new=base_ab,
            tokenizer=tokenizer,
            new_task_batch_fn=d_consolidation_batch_fn,
            proxy_batch_fn=proxy_batch_fn,
            eval_tasks=d_eval_tasks,
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
        if gemma_friendly_sort:
            proxy_metrics.update(evaluate_gemma_sort(proxy_student, tokenizer, cfg))
        proxy_metrics.update(evaluate_gemma_b(proxy_student, tokenizer, cfg))
        if not gemma_friendly_sort:
            audit.add_sort_teacher_forced_metrics(proxy_metrics, proxy_student, tokenizer, cfg)
        print_full_metrics("D_proxy_anchor", proxy_metrics)
        qp._release_cuda_memory(proxy_student)

    sub("Variant 2: D no-proxy same-batch anchor")
    no_proxy_student = qp._clone_model(base_ab_frozen, cfg.device)
    no_proxy_metrics = audit.consolidate_no_proxy_same_batch_in_memory(
        student=no_proxy_student,
        teacher_old=base_ab_frozen,
        teacher_new=base_ab,
        tokenizer=tokenizer,
        new_task_batch_fn=d_consolidation_batch_fn,
        eval_tasks=d_eval_tasks,
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
    if gemma_friendly_sort:
        no_proxy_metrics.update(evaluate_gemma_sort(no_proxy_student, tokenizer, cfg))
    no_proxy_metrics.update(evaluate_gemma_b(no_proxy_student, tokenizer, cfg))
    if not gemma_friendly_sort:
        audit.add_sort_teacher_forced_metrics(no_proxy_metrics, no_proxy_student, tokenizer, cfg)
    print_full_metrics("D_no_proxy", no_proxy_metrics)

    preserve, learns_d, selected_score = gemma_preserve_and_learn(no_proxy_metrics, base_ab_metrics, args)
    selected_no_proxy_label = "D_no_proxy"
    selected_no_proxy_metrics = no_proxy_metrics
    if args.auto_anchor_rescue and (not preserve or not learns_d):
        sub("Variant 3: D no-proxy stronger-retention rescue")
        rescue_student = qp._clone_model(base_ab_frozen, cfg.device)
        rescue_lr = cfg.consolidation_lr * float(args.no_proxy_rescue_lr_scale)
        rescue_metrics = audit.consolidate_no_proxy_same_batch_in_memory(
            student=rescue_student,
            teacher_old=base_ab_frozen,
            teacher_new=base_ab,
            tokenizer=tokenizer,
            new_task_batch_fn=d_consolidation_batch_fn,
            eval_tasks=d_eval_tasks,
            eval_data=eval_data,
            selected_layers=d_layers,
            old_profiles=[a_profile_ab, b_profile_ab],
            project_old_gradients=args.project_old_gradients,
            projection_strength=args.projection_strength,
            steps=args.d_consolidation_steps,
            lr=rescue_lr,
            label="D_no_proxy_retention_rescue",
            cfg=cfg,
            old_task_kl_weight=args.no_proxy_rescue_old_kl_weight,
            old_task_hidden_weight=args.no_proxy_rescue_old_hidden_weight,
            new_kl_weight=args.no_proxy_rescue_new_kl_weight,
            new_hidden_weight=args.no_proxy_rescue_new_hidden_weight,
        )
        if gemma_friendly_sort:
            rescue_metrics.update(evaluate_gemma_sort(rescue_student, tokenizer, cfg))
        rescue_metrics.update(evaluate_gemma_b(rescue_student, tokenizer, cfg))
        if not gemma_friendly_sort:
            audit.add_sort_teacher_forced_metrics(rescue_metrics, rescue_student, tokenizer, cfg)
        print_full_metrics("D_no_proxy_rescue", rescue_metrics)
        rescue_preserve, rescue_learns, rescue_score = gemma_preserve_and_learn(
            rescue_metrics,
            base_ab_metrics,
            args,
        )
        print(
            f"[auto_anchor_rescue] primary preserve={preserve} learns={learns_d} score={selected_score:.4f}; "
            f"rescue preserve={rescue_preserve} learns={rescue_learns} score={rescue_score:.4f}",
            flush=True,
        )
        if rescue_score > selected_score:
            selected_no_proxy_label = "D_no_proxy_rescue"
            selected_no_proxy_metrics = rescue_metrics
            preserve = rescue_preserve
            learns_d = rescue_learns
            selected_score = rescue_score
        qp._release_cuda_memory(rescue_student)

    line("=")
    print("SUMMARY TABLE", flush=True)
    line("=")
    print_full_metrics("base_A", base_metrics)
    print_full_metrics("teacher_B", teacher_b_metrics)
    print_full_metrics("base_AB", base_ab_metrics)
    print_full_metrics("teacher_D", teacher_d_metrics)
    print_full_metrics("D_no_proxy", no_proxy_metrics)
    if selected_no_proxy_label != "D_no_proxy":
        print_full_metrics("D_no_proxy_selected", selected_no_proxy_metrics)

    line("=")
    print("GEMMA FULL-RUN VERDICT", flush=True)
    line("=")
    ab_b = float(base_ab_metrics.get("json_field_acc", 0.0))
    no_proxy_b = float(selected_no_proxy_metrics.get("json_field_acc", 0.0))
    no_proxy_sort = float(selected_no_proxy_metrics.get("sort_token_acc", 0.0))
    no_proxy_sort_loss = float(selected_no_proxy_metrics.get("sort_loss", 999.0))
    ab_ppl = float(base_ab_metrics.get("wikitext_ppl", 999.0))
    no_proxy_ppl = float(selected_no_proxy_metrics.get("wikitext_ppl", 999.0))
    print(
        f"selected_no_proxy_variant={selected_no_proxy_label} score={selected_score:.4f}",
        flush=True,
    )
    print(
        f"no_proxy_preserve={'PASS' if preserve else 'FAIL'} "
        f"b_delta_vs_base_AB={fmt(no_proxy_b - ab_b)} "
        f"ppl_ratio_vs_base_AB={fmt(no_proxy_ppl / max(ab_ppl, 1e-9))}",
        flush=True,
    )
    print(
        f"no_proxy_D_learning={'PASS' if learns_d else 'FAIL'} "
        f"sort_tok={fmt(no_proxy_sort)} sort_loss={fmt(no_proxy_sort_loss)}",
        flush=True,
    )
    print("efficiency: D_no_proxy uses old_task_examples=0 and proxy_batches=0.", flush=True)


def make_cfg(args: argparse.Namespace) -> qp.RuntimeConfig:
    cfg = qp.RuntimeConfig(
        model_id=args.model_id,
        device=args.device,
        dtype=choose_dtype(args.dtype),
        local_files_only=bool(args.local_files_only),
        resume=False,
        smoke=bool(args.smoke),
        output_dir=Path(args.output_dir),
        backup_dir=None,
        seed=args.seed,
        phase_scope="gemma_b_scout",
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
        reversal_eval_samples=1,
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
        cfg.json_eval_samples = min(cfg.json_eval_samples, 8)
        cfg.wikitext_eval_samples = min(cfg.wikitext_eval_samples, 8)
        cfg.proof_v2_b_attach_steps = min(cfg.proof_v2_b_attach_steps, 20)
    setattr(cfg, "gemma_opaque_routes", bool(args.opaque_routes))
    setattr(cfg, "gemma_explicit_route_instructions", not bool(args.route_code_only))
    setattr(cfg, "gemma_b_eval_style_train_frac", float(args.b_eval_style_train_frac))
    setattr(cfg, "gemma_heldout_values_shared", not bool(args.open_vocab_heldout_values))
    setattr(cfg, "gemma_friendly_sort", bool(args.gemma_friendly_sort))
    setattr(cfg, "gemma_sort_eval_style_train_frac", float(args.gemma_sort_eval_style_train_frac))
    setattr(cfg, "gemma_sort_items", int(args.gemma_sort_items))
    return cfg


def run() -> None:
    parser = argparse.ArgumentParser(description="Gemma-friendly continual-learning B acquisition scout")
    parser.add_argument("--model-id", default="google/gemma-3-270m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default="outputs/gemma_cl")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=192)
    parser.add_argument("--wikitext-eval-samples", type=int, default=16)
    parser.add_argument("--json-eval-samples", type=int, default=32)
    parser.add_argument("--sort-eval-samples", type=int, default=24)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument("--b-attach-steps", type=int, default=2400)
    parser.add_argument("--b-attach-lr", type=float, default=1e-4)
    parser.add_argument("--b-rank", type=int, default=48)
    parser.add_argument("--b-alpha", type=float, default=96.0)
    parser.add_argument("--b-gate-init", type=float, default=-1.5)
    parser.add_argument("--b-min-layers", type=int, default=8)
    parser.add_argument("--b-eval-style-train-frac", type=float, default=0.50)
    parser.add_argument("--d-attach-steps", type=int, default=2400)
    parser.add_argument("--d-attach-lr", type=float, default=1e-4)
    parser.add_argument("--d-rank", type=int, default=64)
    parser.add_argument("--d-alpha", type=float, default=128.0)
    parser.add_argument("--d-gate-init", type=float, default=-1.5)
    parser.add_argument("--d-min-layers", type=int, default=10)
    parser.add_argument("--d-train-max-len", type=int, default=16)
    parser.add_argument(
        "--gemma-friendly-sort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a Gemma-calibrated stable-sort interface for D instead of the Qwen proof_v2 sort prompt.",
    )
    parser.add_argument("--gemma-sort-eval-style-train-frac", type=float, default=0.50)
    parser.add_argument("--gemma-sort-items", type=int, default=3)
    parser.add_argument("--b-consolidation-steps", type=int, default=800)
    parser.add_argument("--d-consolidation-steps", type=int, default=900)
    parser.add_argument("--consolidation-lr", type=float, default=1e-5)
    parser.add_argument("--new-kl-weight", type=float, default=1.25)
    parser.add_argument("--new-hidden-weight", type=float, default=0.8)
    parser.add_argument("--proxy-old-kl-weight", type=float, default=0.75)
    parser.add_argument("--no-proxy-old-kl-weight", type=float, default=0.50)
    parser.add_argument("--no-proxy-old-hidden-weight", type=float, default=10.0)
    parser.add_argument(
        "--auto-anchor-rescue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If the first no-proxy Gemma run misses preservation or D learning, try a stronger-retention no-proxy variant.",
    )
    parser.add_argument("--no-proxy-rescue-old-kl-weight", type=float, default=0.75)
    parser.add_argument("--no-proxy-rescue-old-hidden-weight", type=float, default=18.0)
    parser.add_argument("--no-proxy-rescue-new-kl-weight", type=float, default=1.0)
    parser.add_argument("--no-proxy-rescue-new-hidden-weight", type=float, default=0.5)
    parser.add_argument("--no-proxy-rescue-lr-scale", type=float, default=0.75)
    parser.add_argument("--project-old-gradients", action="store_true")
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument("--skip-proxy-variant", action="store_true")
    parser.add_argument("--stop-after-b-teacher", action="store_true")
    parser.add_argument("--stop-after-teachers", action="store_true")
    parser.add_argument("--abort-if-weak-teacher", action="store_true")
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument(
        "--opaque-routes",
        action="store_true",
        help="Use alpha/bravo-style arbitrary route names instead of transparent digit routes.",
    )
    parser.add_argument(
        "--route-code-only",
        action="store_true",
        help="Harder mode: omit explicit 'one uses field N' route instructions.",
    )
    parser.add_argument(
        "--open-vocab-heldout-values",
        action="store_true",
        help="Harder mode: heldout examples use value words never seen during B training.",
    )
    parser.add_argument("--min-teacher-b-field", type=float, default=0.50)
    parser.add_argument("--min-base-ab-b-field", type=float, default=0.35)
    parser.add_argument("--min-teacher-d-sort-tok", type=float, default=0.18)
    parser.add_argument("--min-teacher-d-train-tok", type=float, default=0.35)
    parser.add_argument("--min-teacher-d-tf-tok", type=float, default=0.70)
    parser.add_argument("--max-teacher-d-sort-loss", type=float, default=0.25)
    parser.add_argument("--max-ppl-ratio", type=float, default=1.10)
    parser.add_argument("--max-b-drop", type=float, default=0.12)
    parser.add_argument("--min-sort-tok", type=float, default=0.05)
    parser.add_argument("--max-sort-loss", type=float, default=0.32)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = make_cfg(args)

    line("=")
    print("GEMMA CONTINUAL LEARNING B-ACQUISITION SCOUT", flush=True)
    line("=")
    print("This script is separate from qwen_cl_desiderata_audit.py and does not mutate the Qwen path.", flush=True)
    print("stdout is the artifact; no JSON/CSV is written.", flush=True)
    print(f"model_id={cfg.model_id}", flush=True)
    print(f"device={cfg.device} dtype={cfg.dtype} seed={cfg.seed}", flush=True)
    print(
        f"b_steps={cfg.proof_v2_b_attach_steps} b_rank={cfg.proof_v2_b_rank} "
        f"b_alpha={cfg.proof_v2_b_alpha} batch={cfg.batch_size}",
        flush=True,
    )
    if args.full_run:
        print(
            f"full_run=True d_steps={cfg.proof_v2_d_attach_steps} d_rank={cfg.proof_v2_d_rank} "
            f"d_alpha={cfg.proof_v2_d_alpha} b_consol={args.b_consolidation_steps} "
            f"d_consol={args.d_consolidation_steps}",
            flush=True,
        )
        print(
            f"variants={'D_proxy_anchor + ' if not args.skip_proxy_variant else ''}"
            "D_no_proxy_same_batch_tkl_hidden",
            flush=True,
        )
        print(
            f"gradient_projection={args.project_old_gradients} projection_strength={args.projection_strength}",
            flush=True,
        )
        print(
            f"no_proxy_anchor old_kl={args.no_proxy_old_kl_weight:.3f} "
            f"old_hidden={args.no_proxy_old_hidden_weight:.3f} "
            f"new_kl={args.new_kl_weight:.3f} new_hidden={args.new_hidden_weight:.3f}",
            flush=True,
        )
        print(
            f"auto_anchor_rescue={args.auto_anchor_rescue} "
            f"rescue_old_kl={args.no_proxy_rescue_old_kl_weight:.3f} "
            f"rescue_old_hidden={args.no_proxy_rescue_old_hidden_weight:.3f} "
            f"rescue_new_kl={args.no_proxy_rescue_new_kl_weight:.3f} "
            f"rescue_new_hidden={args.no_proxy_rescue_new_hidden_weight:.3f} "
            f"rescue_lr_scale={args.no_proxy_rescue_lr_scale:.3f}",
            flush=True,
        )
        d_task = (
            "Gemma-friendly stable sort"
            if args.gemma_friendly_sort
            else "Qwen proof_v2 stable sort"
        )
        print(
            f"D_task={d_task}; d_eval_style_train_frac={args.gemma_sort_eval_style_train_frac:.2f}; "
            f"d_items={args.gemma_sort_items}",
            flush=True,
        )
    route_mode = "opaque alpha/bravo routes" if args.opaque_routes else "transparent digit routes"
    route_instruction_mode = "route_code_only" if args.route_code_only else "explicit route instructions"
    value_mode = "open_vocab_heldout_values" if args.open_vocab_heldout_values else "shared_value_vocab"
    print(
        f"task=Gemma-friendly record routing with common tokens; route_mode={route_mode}; "
        f"instruction_mode={route_instruction_mode}; "
        f"b_eval_style_train_frac={args.b_eval_style_train_frac:.2f}; "
        f"heldout_value_mode={value_mode}; unique_values=True end_marker=END old_task_examples=0",
        flush=True,
    )

    tokenizer = load_tokenizer(cfg.model_id, trust_remote_code=True, local_files_only=cfg.local_files_only)
    model = load_causal_lm(
        cfg.model_id,
        device=cfg.device,
        dtype=cfg.dtype,
        trust_remote_code=True,
        local_files_only=cfg.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.full_run:
        run_full_gemma_audit(args, cfg, tokenizer, model)
        return

    sub("Stage A: base model evaluation")
    base_metrics = evaluate_retention(model, tokenizer, cfg)
    base_metrics.update(evaluate_gemma_b(model, tokenizer, cfg))
    print_gemma_metrics("base_A", base_metrics)

    sub("Stage B: train Gemma B adapter teacher")
    b_batch_fn = make_gemma_b_batch_fn(tokenizer, cfg.device, cfg, cfg.seed + 101)
    if args.all_layers:
        layer_count = len(getattr(getattr(model, "model", None), "layers", []))
        b_layers = list(range(layer_count)) if layer_count > 0 else None
        print(f"[gemma_layers] using all layers count={layer_count}", flush=True)
    else:
        b_layers, _ = audit.select_layers(
            model=model,
            tokenizer=tokenizer,
            task_name="json",
            task_batch_fn=b_batch_fn,
            protected_profiles=[],
            cfg=cfg,
        )
    qp._freeze_model(model)
    attached = attach_latent_lora(
        model,
        suffixes=qp._task_target_suffixes("json", cfg),
        layer_indices=None if b_layers is None else set(b_layers),
        config=LatentLoRAConfig(
            rank=cfg.proof_v2_b_rank,
            alpha=cfg.proof_v2_b_alpha,
            dropout=0.0,
            projection_strength=1.0,
            gate_init=cfg.proof_v2_b_gate_init,
            freeze_base=True,
        ),
    )
    print(f"[gemma_lora] attached_modules={len(attached)}", flush=True)
    teacher_metrics = train_gemma_b_teacher(
        model=model,
        tokenizer=tokenizer,
        attached=attached,
        batch_fn=b_batch_fn,
        cfg=cfg,
        steps=cfg.proof_v2_b_attach_steps,
        lr=cfg.proof_v2_b_attach_lr,
    )
    print_debug(model, tokenizer, cfg, label="gemma_B", limit=3)

    line("=")
    print("SUMMARY", flush=True)
    line("=")
    print_gemma_metrics("base_A", base_metrics)
    print_gemma_metrics("teacher_B", teacher_metrics)
    b_ok = float(teacher_metrics.get("json_field_acc", 0.0)) >= float(args.min_teacher_b_field)
    print(
        f"B teacher acquisition={'PASS' if b_ok else 'FAIL'} "
        f"heldout_field={fmt(teacher_metrics.get('json_field_acc'))} "
        f"train_field={fmt(teacher_metrics.get('json_train_field_acc'))} "
        f"heldout_tf={fmt(teacher_metrics.get('json_teacher_forced_token_acc'))} "
        f"need>={args.min_teacher_b_field:.3f}",
        flush=True,
    )
    if b_ok:
        print("next_step: port this Gemma-friendly B interface into a full Gemma B->D no-proxy audit.", flush=True)
    else:
        print("next_step: tune Gemma acquisition first; do not run expensive D/no-proxy stages yet.", flush=True)


if __name__ == "__main__":
    run()
