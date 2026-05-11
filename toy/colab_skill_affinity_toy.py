#!/usr/bin/env python3
"""
Toy skill-affinity probe for latent skill locality.

Question:
Do related injected skills land in similar block neighborhoods on a tiny
"pretrained-ish" model, compared with unrelated control skills?

This is intentionally small and fast. It is not a full continual-learning
benchmark. It gives a first concrete signal by:
    1. pretraining a TinyGPT base on a small text corpus,
    2. injecting a few short skills with adapter-only updates,
    3. measuring pre-injection block pressure profiles,
    4. measuring post-injection adapter-delta footprints, and
    5. checking whether related skills cluster more than unrelated ones.

Run:
    python colab_skill_affinity_toy.py

Fast smoke:
    CHAOS_SMOKE=1 python colab_skill_affinity_toy.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import colab_water_weights_benchmark as ww


ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "skill_affinity_toy_results.json"
CSV_PATH = ROOT / "skill_affinity_toy_pairwise.csv"

TASK_ALPHABET = "ABCDEFGH"
TASK_SPACE = " |\n"
TASK_NAMES = ("reverse", "rotate_left", "shift", "shift_twice")
RELATED_PAIRS = {
    ("reverse", "rotate_left"),
    ("shift", "shift_twice"),
}

ANSWER_WEIGHT = 6.0
CONTEXT_WEIGHT = 0.10

if torch.cuda.is_available():
    DEFAULT_BASE_STEPS = 180
    DEFAULT_TASK_STEPS = 180
    DEFAULT_BATCH_SIZE = min(ww.BATCH_SIZE, 16)
    DEFAULT_TEXT_EVAL_BATCHES = 6
    DEFAULT_TASK_EVAL_BATCHES = 8
    DEFAULT_PROBE_BATCHES = 6
    DEFAULT_PROJECTOR_BATCHES = 6
else:
    DEFAULT_BASE_STEPS = 70
    DEFAULT_TASK_STEPS = 90
    DEFAULT_BATCH_SIZE = min(ww.BATCH_SIZE, 8)
    DEFAULT_TEXT_EVAL_BATCHES = 4
    DEFAULT_TASK_EVAL_BATCHES = 4
    DEFAULT_PROBE_BATCHES = 4
    DEFAULT_PROJECTOR_BATCHES = 4


@dataclass
class TaskRun:
    name: str
    pressure_frontier: List[str]
    update_frontier: List[str]
    pressure_profile: Dict[str, float]
    update_profile: Dict[str, float]
    zero_shot_answer_acc: float
    zero_shot_problem_acc: float
    zero_shot_seq_acc: float
    zero_shot_loss: float
    train_answer_acc: float
    eval_answer_acc: float
    eval_problem_acc: float
    eval_seq_acc: float
    eval_loss: float
    text_loss_with_adapter: float
    text_acc_with_adapter: float


@dataclass
class InjectionOutcome:
    run: TaskRun
    delta_vector: np.ndarray


def parse_args() -> argparse.Namespace:
    smoke = os.environ.get("CHAOS_SMOKE", "0") == "1"
    parser = argparse.ArgumentParser(description="Toy skill-affinity probe")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--base-steps", type=int, default=24 if smoke else DEFAULT_BASE_STEPS)
    parser.add_argument("--task-steps", type=int, default=28 if smoke else DEFAULT_TASK_STEPS)
    parser.add_argument("--batch-size", type=int, default=4 if smoke else DEFAULT_BATCH_SIZE)
    parser.add_argument("--source-len-train", type=int, default=6 if smoke else 8)
    parser.add_argument("--source-len-eval", type=int, default=7 if smoke else 10)
    parser.add_argument("--text-eval-batches", type=int, default=2 if smoke else DEFAULT_TEXT_EVAL_BATCHES)
    parser.add_argument("--task-eval-batches", type=int, default=2 if smoke else DEFAULT_TASK_EVAL_BATCHES)
    parser.add_argument("--probe-batches", type=int, default=2 if smoke else DEFAULT_PROBE_BATCHES)
    parser.add_argument("--projector-batches", type=int, default=2 if smoke else DEFAULT_PROJECTOR_BATCHES)
    parser.add_argument("--base-lr", type=float, default=ww.BASE_LR)
    parser.add_argument("--adapter-lr", type=float, default=2e-3)
    parser.add_argument("--latent-strength", type=float, default=0.75)
    parser.add_argument("--frontier-k", type=int, default=4)
    parser.add_argument("--text-corpus-chars", type=int, default=6_000 if smoke else 80_000)
    parser.add_argument("--json-path", type=Path, default=JSON_PATH)
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    ww.set_seed(seed)


def pressure_group_keys() -> List[str]:
    keys: List[str] = []
    for block in ww.block_keys():
        keys.extend([f"{block}.attn", f"{block}.mlp"])
    return keys


def adapter_group_keys() -> List[str]:
    keys: List[str] = []
    for block in ww.block_keys():
        index = int(block[1:])
        keys.extend([f"a{index}.down", f"a{index}.up"])
    return keys


def build_joint_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text + TASK_ALPHABET + TASK_SPACE))
    stoi = {ch: idx for idx, ch in enumerate(chars)}
    itos = {idx: ch for ch, idx in stoi.items()}
    return stoi, itos


def split_text_corpus(stoi: Dict[str, int], corpus_chars: int) -> Tuple[torch.Tensor, torch.Tensor]:
    text = ww.EMBEDDED_FALLBACK_TEXT[:corpus_chars]
    encoded = ww.encode(text, stoi)
    split = int(len(encoded) * ww.TRAIN_FRACTION)
    train_data = encoded[:split]
    val_data = encoded[split:]
    return train_data, val_data


def make_text_positions(data_len: int, num_batches: int, batch_size: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    high = max(data_len - ww.BLOCK_SIZE - 1, 1)
    return torch.randint(0, high, (num_batches, batch_size), generator=gen)


def task_transform(task_name: str, source: str) -> str:
    next_map = {ch: TASK_ALPHABET[(idx + 1) % len(TASK_ALPHABET)] for idx, ch in enumerate(TASK_ALPHABET)}
    next2_map = {ch: TASK_ALPHABET[(idx + 2) % len(TASK_ALPHABET)] for idx, ch in enumerate(TASK_ALPHABET)}
    prev_map = {ch: TASK_ALPHABET[(idx - 1) % len(TASK_ALPHABET)] for idx, ch in enumerate(TASK_ALPHABET)}
    reflect_map = {ch: TASK_ALPHABET[-(idx + 1)] for idx, ch in enumerate(TASK_ALPHABET)}
    if task_name == "reverse":
        return source[::-1]
    if task_name == "rotate_left":
        return source[1:] + source[:1]
    if task_name == "rotate_right":
        return source[-1:] + source[:-1]
    if task_name == "swap_pairs":
        chars = list(source)
        out = chars[:]
        for index in range(0, len(chars) - 1, 2):
            out[index], out[index + 1] = chars[index + 1], chars[index]
        return "".join(out)
    if task_name == "block_reverse3":
        chunks = [source[index:index + 3] for index in range(0, len(source), 3)]
        return "".join(chunk[::-1] for chunk in chunks)
    if task_name == "block_rotate3":
        chunks = [source[index:index + 3] for index in range(0, len(source), 3)]
        rotated = []
        for chunk in chunks:
            if len(chunk) <= 1:
                rotated.append(chunk)
            else:
                rotated.append(chunk[1:] + chunk[:1])
        return "".join(rotated)
    if task_name == "shift":
        return "".join(next_map[ch] for ch in source)
    if task_name == "shift_twice":
        return "".join(next2_map[ch] for ch in source)
    if task_name == "unshift":
        return "".join(prev_map[ch] for ch in source)
    if task_name == "reflect":
        return "".join(reflect_map[ch] for ch in source)
    if task_name == "alt_shift":
        return "".join(next_map[ch] if index % 2 == 0 else ch for index, ch in enumerate(source))
    if task_name == "alt_shift_twice":
        return "".join(next2_map[ch] if index % 2 == 0 else ch for index, ch in enumerate(source))
    if task_name == "odd_shift":
        return "".join(next_map[ch] if index % 2 == 1 else ch for index, ch in enumerate(source))
    raise ValueError(f"Unknown task: {task_name}")


def build_task_stream(rng: np.random.Generator, task_name: str, source_len: int) -> Tuple[List[str], List[bool]]:
    token_ids = rng.integers(0, len(TASK_ALPHABET), size=source_len)
    source = "".join(TASK_ALPHABET[int(idx)] for idx in token_ids)
    target = task_transform(task_name, source)
    prompt = f"{source}|"
    answer = f"{target}\n"
    episode = list(prompt + answer)
    flags = [False] * len(prompt) + [True] * len(target) + [False]

    total_len = ww.BLOCK_SIZE + 1
    pad_len = max(total_len - len(episode), 0)
    left_pad = int(rng.integers(0, pad_len + 1)) if pad_len > 0 else 0
    right_pad = pad_len - left_pad
    chars = ([" "] * left_pad) + episode + ([" "] * right_pad)
    critical = ([False] * left_pad) + flags + ([False] * right_pad)
    return chars[:total_len], critical[:total_len]


def make_task_batch(
    stoi: Dict[str, int],
    task_name: str,
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
        chars, critical = build_task_stream(rng, task_name, source_len)
        ids = torch.tensor([stoi[ch] for ch in chars], dtype=torch.long)
        mask = torch.tensor(critical, dtype=torch.bool)
        xs.append(ids[:-1])
        ys.append(ids[1:])
        masks.append(mask[1:])
    return ww.Batch(
        torch.stack(xs).to(ww.DEVICE),
        torch.stack(ys).to(ww.DEVICE),
        torch.stack(masks).to(ww.DEVICE),
    )


def make_fixed_task_batches(
    stoi: Dict[str, int],
    task_name: str,
    seed: int,
    *,
    num_batches: int,
    batch_size: int,
    source_len: int,
) -> List[ww.Batch]:
    return [
        make_task_batch(stoi, task_name, seed, index, batch_size=batch_size, source_len=source_len)
        for index in range(num_batches)
    ]


def task_weighted_loss(logits: torch.Tensor, batch: ww.Batch) -> torch.Tensor:
    if batch.critical_mask is None:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.y.reshape(-1))
    token_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.y.reshape(-1), reduction="none").view_as(batch.y)
    weights = torch.where(
        batch.critical_mask,
        torch.full_like(token_loss, ANSWER_WEIGHT),
        torch.full_like(token_loss, CONTEXT_WEIGHT),
    )
    return (token_loss * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_task(model: ww.TinyGPT, batches: Sequence[ww.Batch]) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    token_correct = 0
    token_total = 0
    problem_correct = 0
    problem_total = 0
    seq_correct = 0
    seq_total = 0
    for batch in batches:
        logits, loss = model(batch.x, batch.y)
        losses.append(float(task_weighted_loss(logits, batch).item() if loss is not None else 0.0))
        preds = logits.argmax(dim=-1)
        assert batch.critical_mask is not None
        hits = (preds == batch.y) & batch.critical_mask
        token_correct += int(hits.sum().item())
        token_total += int(batch.critical_mask.sum().item())
        for row_hits, row_mask in zip(hits, batch.critical_mask):
            start = None
            for idx, flagged in enumerate(row_mask.tolist()):
                if flagged and start is None:
                    start = idx
                elif not flagged and start is not None:
                    segment = row_hits[start:idx]
                    problem_correct += int(bool(segment.all().item()))
                    problem_total += 1
                    start = None
            if start is not None:
                segment = row_hits[start:]
                problem_correct += int(bool(segment.all().item()))
                problem_total += 1
        seq_ok = ((preds == batch.y) | (~batch.critical_mask)).all(dim=1)
        seq_correct += int(seq_ok.sum().item())
        seq_total += int(seq_ok.numel())
    model.train()
    return {
        "loss": float(np.mean(losses)),
        "answer_acc": token_correct / max(token_total, 1),
        "problem_acc": problem_correct / max(problem_total, 1),
        "seq_acc": seq_correct / max(seq_total, 1),
    }


def train_step(
    model: ww.TinyGPT,
    optimizer: torch.optim.Optimizer,
    batch: ww.Batch,
    loss_fn=task_weighted_loss,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(batch.x, batch.y)
    loss = loss_fn(logits, batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], ww.GRAD_CLIP)
    optimizer.step()
    return float(loss.item())


def mean_group_pressure_profile(model: ww.TinyGPT, batches: Sequence[ww.Batch]) -> Dict[str, float]:
    accum = {key: 0.0 for key in pressure_group_keys()}
    for batch in batches:
        z_map, act_z = ww.probe_z(model, batch)
        for block in ww.block_keys():
            block_act = max(float(act_z.get(block, 1e-12)), 1e-12)
            for suffix in ("attn", "mlp"):
                key = f"{block}.{suffix}"
                param_value = max(float(z_map.get(key, 1e-12)), 1e-12)
                combined = math.exp(0.75 * math.log(param_value) + 0.25 * math.log(block_act))
                accum[key] += combined
    scale = 1.0 / max(len(batches), 1)
    return {key: accum[key] * scale for key in pressure_group_keys()}


def collect_text_latent_free_projectors(
    model: ww.TinyGPT,
    train_data: torch.Tensor,
    text_positions: torch.Tensor,
    blocks: Sequence[str],
) -> Dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    act_cov = {block: torch.zeros(ww.D_MODEL, ww.D_MODEL) for block in blocks}
    grad_cov = {block: torch.zeros(ww.D_MODEL, ww.D_MODEL) for block in blocks}
    counts = {block: 0 for block in blocks}
    for starts in text_positions:
        batch = ww.text_batch_from_positions(train_data, starts)
        model.zero_grad(set_to_none=True)
        _, loss, activations = model(batch.x, batch.y, return_activations=True)
        assert loss is not None
        loss.backward()
        for block in blocks:
            block_index = int(block[1:])
            activation = activations[block_index].detach().reshape(-1, ww.D_MODEL).float().cpu()
            gradient = activations[block_index].grad
            grad_flat = torch.zeros_like(activation) if gradient is None else gradient.detach().reshape(-1, ww.D_MODEL).float().cpu()
            act_cov[block] += activation.t() @ activation / max(activation.shape[0], 1)
            grad_cov[block] += grad_flat.t() @ grad_flat / max(grad_flat.shape[0], 1)
            counts[block] += 1
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    projectors = {}
    for block in blocks:
        denom = max(counts[block], 1)
        projectors[block] = ww.free_projector_from_covariances(act_cov[block] / denom, grad_cov[block] / denom)
    return projectors


def normalize_profile(profile: Dict[str, float]) -> Dict[str, float]:
    key_order = list(profile.keys())
    total = sum(max(float(v), 0.0) for v in profile.values())
    if total <= 1e-12:
        return {key: 0.0 for key in key_order}
    return {key: max(float(profile.get(key, 0.0)), 0.0) / total for key in key_order}


def profile_vector(profile: Dict[str, float]) -> np.ndarray:
    return np.asarray([float(profile.get(key, 0.0)) for key in profile.keys()], dtype=float)


def cosine_similarity(profile_a: Dict[str, float], profile_b: Dict[str, float]) -> float:
    vec_a = profile_vector(profile_a)
    vec_b = profile_vector(profile_b)
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def centered_similarity(profile_a: Dict[str, float], profile_b: Dict[str, float]) -> float:
    vec_a = profile_vector(profile_a)
    vec_b = profile_vector(profile_b)
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    vec_a = vec_a - float(vec_a.mean())
    vec_b = vec_b - float(vec_b.mean())
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def frontier_jaccard(frontier_a: Sequence[str], frontier_b: Sequence[str]) -> float:
    set_a = set(frontier_a)
    set_b = set(frontier_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return float(len(set_a & set_b) / len(union))


def adapter_delta_profile(model: ww.TinyGPT, base_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    state = model.state_dict()
    profile: Dict[str, float] = {}
    for block in ww.block_keys():
        index = int(block[1:])
        for group_name, key in (
            (f"a{index}.down", f"blocks.{index}.adapter.down.weight"),
            (f"a{index}.up", f"blocks.{index}.adapter.up.weight"),
        ):
            current = state[key].detach().float().cpu()
            reference = base_state[key].detach().float().cpu()
            profile[group_name] = float((current - reference).pow(2).sum().sqrt().item())
    return profile


def adapter_delta_vector(model: ww.TinyGPT, base_state: Dict[str, torch.Tensor]) -> np.ndarray:
    pieces: List[np.ndarray] = []
    state = model.state_dict()
    for block in ww.block_keys():
        index = int(block[1:])
        for key in (
            f"blocks.{index}.adapter.down.weight",
            f"blocks.{index}.adapter.up.weight",
        ):
            current = state[key].detach().float().cpu()
            reference = base_state[key].detach().float().cpu()
            pieces.append((current - reference).reshape(-1).numpy())
    if not pieces:
        return np.zeros(1, dtype=float)
    return np.concatenate(pieces, axis=0).astype(float, copy=False)


def top_profile_keys(profile: Dict[str, float], k: int) -> List[str]:
    return sorted(profile.keys(), key=lambda key: (profile.get(key, 0.0), key), reverse=True)[:k]


def pretrain_base_model(
    vocab_size: int,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], Dict[str, torch.Tensor]]:
    model = ww.TinyGPT(vocab_size).to(ww.DEVICE)
    model.set_adapters_enabled(False)
    ww.set_requires_grad(ww.all_adapter_params(model), False)
    optimizer = torch.optim.AdamW(
        ww.all_base_params(model),
        lr=cfg.base_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )
    text_train_positions = make_text_positions(len(train_data), cfg.base_steps, cfg.batch_size, cfg.seed + 100)
    text_eval_positions = make_text_positions(len(val_data), cfg.text_eval_batches, min(cfg.batch_size, ww.TEXT_EVAL_BATCH), cfg.seed + 200)
    for step in range(cfg.base_steps):
        batch = ww.text_batch_from_positions(train_data, text_train_positions[step])
        train_step(model, optimizer, batch, loss_fn=lambda logits, b: F.cross_entropy(logits.reshape(-1, logits.size(-1)), b.y.reshape(-1)))
    text_metrics = ww.evaluate_text(model, val_data, text_eval_positions)
    projector_positions = make_text_positions(len(train_data), cfg.projector_batches, cfg.batch_size, cfg.seed + 300)
    projectors = collect_text_latent_free_projectors(model, train_data, projector_positions, ww.block_keys())
    base_state = ww.tensor_tree_to_cpu(model.state_dict())
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return base_state, text_metrics, projectors


def restore_base_model(vocab_size: int, base_state: Dict[str, torch.Tensor]) -> ww.TinyGPT:
    model = ww.TinyGPT(vocab_size).to(ww.DEVICE)
    model.load_state_dict(base_state)
    return model


def run_task_injection(
    task_name: str,
    vocab_size: int,
    stoi: Dict[str, int],
    base_state: Dict[str, torch.Tensor],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    projectors: Dict[str, torch.Tensor],
    cfg: argparse.Namespace,
) -> InjectionOutcome:
    model = restore_base_model(vocab_size, base_state)
    pressure_probe_batches = make_fixed_task_batches(
        stoi,
        task_name,
        cfg.seed + 1_000,
        num_batches=cfg.probe_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    pressure_profile = normalize_profile(mean_group_pressure_profile(model, pressure_probe_batches))
    pressure_frontier = top_profile_keys(pressure_profile, cfg.frontier_k)

    model.set_adapters_enabled(True)
    model.set_latent_free_projectors(projectors, cfg.latent_strength)
    ww.set_requires_grad(ww.all_base_params(model), False)
    ww.set_requires_grad(ww.all_adapter_params(model), True)
    optimizer = torch.optim.AdamW(
        [param for param in ww.all_adapter_params(model) if param.requires_grad],
        lr=cfg.adapter_lr,
        betas=ww.BETAS,
        weight_decay=ww.WEIGHT_DECAY,
    )

    train_eval_batches = make_fixed_task_batches(
        stoi,
        task_name,
        cfg.seed + 2_000,
        num_batches=max(2, cfg.task_eval_batches // 2),
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_train,
    )
    eval_batches = make_fixed_task_batches(
        stoi,
        task_name,
        cfg.seed + 3_000,
        num_batches=cfg.task_eval_batches,
        batch_size=cfg.batch_size,
        source_len=cfg.source_len_eval,
    )
    zero_shot_metrics = evaluate_task(model, eval_batches)

    for step in range(cfg.task_steps):
        batch = make_task_batch(
            stoi,
            task_name,
            cfg.seed + 4_000,
            step,
            batch_size=cfg.batch_size,
            source_len=cfg.source_len_train,
        )
        train_step(model, optimizer, batch)

    train_metrics = evaluate_task(model, train_eval_batches)
    eval_metrics = evaluate_task(model, eval_batches)
    text_eval_positions = make_text_positions(len(val_data), cfg.text_eval_batches, min(cfg.batch_size, ww.TEXT_EVAL_BATCH), cfg.seed + 5_000)
    text_metrics = ww.evaluate_text(model, val_data, text_eval_positions)
    update_profile = normalize_profile(adapter_delta_profile(model, base_state))
    update_frontier = top_profile_keys(update_profile, cfg.frontier_k)
    delta_vector = adapter_delta_vector(model, base_state)

    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return InjectionOutcome(
        run=TaskRun(
            name=task_name,
            pressure_frontier=pressure_frontier,
            update_frontier=update_frontier,
            pressure_profile=pressure_profile,
            update_profile=update_profile,
            zero_shot_answer_acc=float(zero_shot_metrics["answer_acc"]),
            zero_shot_problem_acc=float(zero_shot_metrics["problem_acc"]),
            zero_shot_seq_acc=float(zero_shot_metrics["seq_acc"]),
            zero_shot_loss=float(zero_shot_metrics["loss"]),
            train_answer_acc=float(train_metrics["answer_acc"]),
            eval_answer_acc=float(eval_metrics["answer_acc"]),
            eval_problem_acc=float(eval_metrics["problem_acc"]),
            eval_seq_acc=float(eval_metrics["seq_acc"]),
            eval_loss=float(eval_metrics["loss"]),
            text_loss_with_adapter=float(text_metrics["loss"]),
            text_acc_with_adapter=float(text_metrics["acc"]),
        ),
        delta_vector=delta_vector,
    )


def pair_relation(name_a: str, name_b: str) -> str:
    ordered = tuple(sorted((name_a, name_b)))
    return "related" if ordered in RELATED_PAIRS else "control"


def save_pairwise_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_a",
                "task_b",
                "relation",
                "pressure_cosine",
                "update_cosine",
                "delta_cosine",
                "pressure_frontier_jaccard",
                "update_frontier_jaccard",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    joint_text = ww.EMBEDDED_FALLBACK_TEXT[: cfg.text_corpus_chars]
    stoi, _ = build_joint_vocab(joint_text)
    train_data, val_data = split_text_corpus(stoi, cfg.text_corpus_chars)
    vocab_size = len(stoi)

    print("=" * 78)
    print("TOY SKILL AFFINITY PROBE")
    print("=" * 78)
    print(f"device={ww.DEVICE} vocab={vocab_size} base_steps={cfg.base_steps} task_steps={cfg.task_steps}")
    print(f"tasks={', '.join(TASK_NAMES)} related_pair={RELATED_PAIRS}")

    base_state, base_text_metrics, projectors = pretrain_base_model(vocab_size, train_data, val_data, cfg)
    print(
        f"[base] text_loss={base_text_metrics['loss']:.4f} "
        f"text_acc={base_text_metrics['acc']:.3f}"
    )

    task_runs: Dict[str, TaskRun] = {}
    delta_vectors: Dict[str, np.ndarray] = {}
    for task_name in TASK_NAMES:
        outcome = run_task_injection(task_name, vocab_size, stoi, base_state, train_data, val_data, projectors, cfg)
        run = outcome.run
        task_runs[task_name] = run
        delta_vectors[task_name] = outcome.delta_vector
        print(
            f"[{task_name}] train_answer={run.train_answer_acc:.3f} "
            f"eval_answer={run.eval_answer_acc:.3f} eval_problem={run.eval_problem_acc:.3f} "
            f"pressure={'+'.join(run.pressure_frontier)} update={'+'.join(run.update_frontier)}"
        )

    pairwise_rows: List[Dict[str, object]] = []
    related_pressure: List[float] = []
    control_pressure: List[float] = []
    related_update: List[float] = []
    control_update: List[float] = []
    related_delta: List[float] = []
    control_delta: List[float] = []
    for index, name_a in enumerate(TASK_NAMES):
        for name_b in TASK_NAMES[index + 1:]:
            run_a = task_runs[name_a]
            run_b = task_runs[name_b]
            row = {
                "task_a": name_a,
                "task_b": name_b,
                "relation": pair_relation(name_a, name_b),
                "pressure_cosine": cosine_similarity(run_a.pressure_profile, run_b.pressure_profile),
                "update_cosine": cosine_similarity(run_a.update_profile, run_b.update_profile),
                "pressure_frontier_jaccard": frontier_jaccard(run_a.pressure_frontier, run_b.pressure_frontier),
                "update_frontier_jaccard": frontier_jaccard(run_a.update_frontier, run_b.update_frontier),
            }
            delta_denom = float(np.linalg.norm(delta_vectors[name_a]) * np.linalg.norm(delta_vectors[name_b]))
            row["delta_cosine"] = (
                float(np.dot(delta_vectors[name_a], delta_vectors[name_b]) / delta_denom)
                if delta_denom > 1e-12
                else 0.0
            )
            pairwise_rows.append(row)
            if row["relation"] == "related":
                related_pressure.append(float(row["pressure_cosine"]))
                related_update.append(float(row["update_cosine"]))
                related_delta.append(float(row["delta_cosine"]))
            else:
                control_pressure.append(float(row["pressure_cosine"]))
                control_update.append(float(row["update_cosine"]))
                control_delta.append(float(row["delta_cosine"]))

    pressure_gap = float(np.mean(related_pressure) - np.mean(control_pressure)) if related_pressure and control_pressure else float("nan")
    update_gap = float(np.mean(related_update) - np.mean(control_update)) if related_update and control_update else float("nan")
    delta_gap = float(np.mean(related_delta) - np.mean(control_delta)) if related_delta and control_delta else float("nan")
    signal_margin = 0.02
    mean_eval_answer = float(np.mean([run.eval_answer_acc for run in task_runs.values()])) if task_runs else float("nan")

    print("-" * 78)
    for row in pairwise_rows:
        print(
            f"{row['task_a']:>10} vs {row['task_b']:<10} "
            f"relation={row['relation']:<7} "
            f"pressure_cos={float(row['pressure_cosine']):.3f} "
            f"update_cos={float(row['update_cosine']):.3f} "
            f"delta_cos={float(row['delta_cosine']):.3f} "
            f"pressure_j={float(row['pressure_frontier_jaccard']):.3f} "
            f"update_j={float(row['update_frontier_jaccard']):.3f}"
        )
    print("-" * 78)
    print(
        f"related_vs_control pressure_cos_gap={pressure_gap:+.3f} "
        f"update_cos_gap={update_gap:+.3f} "
        f"delta_cos_gap={delta_gap:+.3f}"
    )
    if math.isfinite(pressure_gap) and math.isfinite(update_gap) and math.isfinite(delta_gap):
        if pressure_gap > signal_margin and (update_gap > signal_margin or delta_gap > signal_margin):
            print("signal: related skills cluster more than controls on both pressure and update profiles.")
        elif pressure_gap > signal_margin or update_gap > signal_margin or delta_gap > signal_margin:
            print("mixed signal: one affinity metric is positive, but not both.")
        else:
            print("no positive affinity gap yet under this toy setup.")
            if math.isfinite(mean_eval_answer) and mean_eval_answer < 0.20:
                print("diagnostic: the injected skills are still only weakly learned here, so locality may be hidden by under-training.")
            else:
                print("diagnostic: this tiny base may simply be too generic/coarse to expose clean skill neighborhoods.")

    cfg.json_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.json_path.write_text(
        json.dumps(
            {
                "config": vars(cfg),
                "base_text_metrics": base_text_metrics,
                "task_runs": {name: asdict(run) for name, run in task_runs.items()},
                "pairwise_rows": pairwise_rows,
                "summary": {
                    "pressure_cos_gap_related_minus_control": pressure_gap,
                    "update_cos_gap_related_minus_control": update_gap,
                    "delta_cos_gap_related_minus_control": delta_gap,
                    "mean_eval_answer_acc": mean_eval_answer,
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    save_pairwise_csv(cfg.csv_path, pairwise_rows)
    print(f"saved: {cfg.json_path}")
    print(f"saved: {cfg.csv_path}")


if __name__ == "__main__":
    main()
