#!/usr/bin/env python3
"""Toy probe for full automaticity from sequential training.

Question:
  Can a model learn a general composition habit first, then sequentially learn
  new skills B and D, and finally solve the unseen direct B+D composition
  without any raw B+D composition examples?

Setup:
  Meta composer pretraining:
    C(meta_route, record, meta_keys) -> ANS(sorted routed slot=value)

  Sequential new skills:
    B(new_route, record) -> MAP(slot -> value)
    D(new_keys) -> ORDER(slots)

  Heldout automaticity test:
    C(new_route, record, new_keys) -> ANS(sorted routed slot=value)

Baseline conditions never train direct C(new_route,new_keys) examples.
The lateral condition is intentionally different: it first calls the learned
B and D skills, then self-distills those generated traces into direct C prompts.
That tests whether scaffolded composition can be melted back into weights.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SLOTS = ("s0", "s1", "s2")
FIELDS = ("a", "b", "c")
ROUTES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)
META_ROUTES = (0, 1, 2, 3)
NEW_ROUTES = (4, 5)
META_KEYS = ("k0", "k1", "k2")
NEW_KEYS = ("h0", "h1", "h2")
VALUES = tuple(f"v{i}" for i in range(10))
SPECIAL = ("<pad>", "<eos>", "->", "B", "D", "C", "MAP", "ORDER", "ANS")


@dataclass(frozen=True)
class Example:
    name: str
    prompt: Tuple[str, ...]
    target: Tuple[str, ...]


class Vocab:
    def __init__(self) -> None:
        tokens = (
            list(SPECIAL)
            + list(SLOTS)
            + list(FIELDS)
            + [f"r{i}" for i in range(len(ROUTES))]
            + list(META_KEYS)
            + list(NEW_KEYS)
            + list(VALUES)
        )
        self.itos = list(dict.fromkeys(tokens))
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.eos_id = self.stoi["<eos>"]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi[token] for token in tokens]

    def decode(self, ids: Sequence[int]) -> Tuple[str, ...]:
        out: List[str] = []
        for idx in ids:
            token = self.itos[int(idx)]
            if token == "<eos>":
                break
            if token != "<pad>":
                out.append(token)
        return tuple(out)


class TinyGRULM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 192, layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=layers, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(self.embed(ids))
        return self.head(hidden)


def route_map(route_id: int, values: Sequence[str]) -> Dict[str, str]:
    return {slot: values[field_idx] for slot, field_idx in zip(SLOTS, ROUTES[int(route_id)])}


def key_order(keys: Sequence[str], key_family: Sequence[str]) -> Tuple[str, str, str]:
    return tuple(slot for _, slot in sorted(zip(keys, SLOTS), key=lambda item: key_family.index(item[0])))


def ans_tokens(slot_map: Dict[str, str], order: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = ["ANS"]
    for slot in order:
        out.extend([slot, slot_map[slot]])
    return tuple(out)


def sample_values(rng: np.random.Generator) -> Tuple[str, str, str]:
    return tuple(VALUES[int(rng.integers(0, len(VALUES)))] for _ in FIELDS)


def sample_keys(rng: np.random.Generator, family: Sequence[str]) -> Tuple[str, str, str]:
    return tuple(family[int(rng.integers(0, len(family)))] for _ in SLOTS)


def make_b(route_id: int, values: Sequence[str]) -> Example:
    mapping = route_map(route_id, values)
    return Example(
        "B",
        ("B", f"r{route_id}", "a", values[0], "b", values[1], "c", values[2], "->"),
        ("MAP", "s0", mapping["s0"], "s1", mapping["s1"], "s2", mapping["s2"]),
    )


def make_d(keys: Sequence[str], family: Sequence[str]) -> Example:
    order = key_order(keys, family)
    return Example(
        "D",
        ("D", "s0", keys[0], "s1", keys[1], "s2", keys[2], "->"),
        ("ORDER", order[0], order[1], order[2]),
    )


def map_tokens(mapping: Dict[str, str]) -> Tuple[str, ...]:
    return ("MAP", "s0", mapping["s0"], "s1", mapping["s1"], "s2", mapping["s2"])


def order_tokens(order: Sequence[str]) -> Tuple[str, ...]:
    return ("ORDER", order[0], order[1], order[2])


def make_c(
    route_id: int,
    values: Sequence[str],
    keys: Sequence[str],
    family: Sequence[str],
    *,
    scratchpad: bool = False,
) -> Example:
    mapping = route_map(route_id, values)
    order = key_order(keys, family)
    target = ans_tokens(mapping, order)
    if scratchpad:
        target = (*map_tokens(mapping), *order_tokens(order), *target)
    return Example(
        "C",
        (
            "C",
            f"r{route_id}",
            "a",
            values[0],
            "b",
            values[1],
            "c",
            values[2],
            "s0",
            keys[0],
            "s1",
            keys[1],
            "s2",
            keys[2],
            "->",
        ),
        target,
    )


def sample_example(rng: np.random.Generator, mode: str) -> Example:
    values = sample_values(rng)
    if mode == "meta_c":
        route_id = int(META_ROUTES[int(rng.integers(0, len(META_ROUTES)))])
        return make_c(route_id, values, sample_keys(rng, META_KEYS), META_KEYS)
    if mode == "meta_c_scratch":
        route_id = int(META_ROUTES[int(rng.integers(0, len(META_ROUTES)))])
        return make_c(route_id, values, sample_keys(rng, META_KEYS), META_KEYS, scratchpad=True)
    if mode == "new_b":
        route_id = int(NEW_ROUTES[int(rng.integers(0, len(NEW_ROUTES)))])
        return make_b(route_id, values)
    if mode == "new_d":
        return make_d(sample_keys(rng, NEW_KEYS), NEW_KEYS)
    if mode == "new_c":
        route_id = int(NEW_ROUTES[int(rng.integers(0, len(NEW_ROUTES)))])
        return make_c(route_id, values, sample_keys(rng, NEW_KEYS), NEW_KEYS)
    if mode == "new_c_scratch":
        route_id = int(NEW_ROUTES[int(rng.integers(0, len(NEW_ROUTES)))])
        return make_c(route_id, values, sample_keys(rng, NEW_KEYS), NEW_KEYS, scratchpad=True)
    if mode == "meta_b":
        route_id = int(META_ROUTES[int(rng.integers(0, len(META_ROUTES)))])
        return make_b(route_id, values)
    if mode == "meta_d":
        return make_d(sample_keys(rng, META_KEYS), META_KEYS)
    raise ValueError(mode)


def make_batch(vocab: Vocab, examples: Sequence[Example], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs: List[List[int]] = []
    labels: List[List[int]] = []
    for example in examples:
        prompt_ids = vocab.encode(example.prompt)
        target_ids = vocab.encode((*example.target, "<eos>"))
        full = prompt_ids + target_ids
        row_in = full[:-1]
        row_labels = full[1:]
        mask_until = max(len(prompt_ids) - 1, 0)
        row_labels = [-100] * mask_until + row_labels[mask_until:]
        inputs.append(row_in)
        labels.append(row_labels)
    max_len = max(len(row) for row in inputs)
    input_tensor = torch.full((len(inputs), max_len), vocab.pad_id, dtype=torch.long, device=device)
    label_tensor = torch.full((len(inputs), max_len), -100, dtype=torch.long, device=device)
    for i, (row_in, row_labels) in enumerate(zip(inputs, labels)):
        input_tensor[i, : len(row_in)] = torch.tensor(row_in, dtype=torch.long, device=device)
        label_tensor[i, : len(row_labels)] = torch.tensor(row_labels, dtype=torch.long, device=device)
    return input_tensor, label_tensor


def train_phase(
    model: TinyGRULM,
    vocab: Vocab,
    device: torch.device,
    *,
    label: str,
    modes: Sequence[str],
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    log_interval: int,
    trainable_token_ids: Sequence[int] | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    if trainable_token_ids is None:
        for param in model.parameters():
            param.requires_grad_(True)
        params = list(model.parameters())
    else:
        for param in model.parameters():
            param.requires_grad_(False)
        model.embed.weight.requires_grad_(True)
        params = [model.embed.weight]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    allowed_ids = None
    if trainable_token_ids is not None:
        allowed_ids = torch.tensor(list(trainable_token_ids), dtype=torch.long, device=device)
    model.train()
    start = time.time()
    for step in range(1, steps + 1):
        examples = [
            sample_example(rng, str(modes[int(rng.integers(0, len(modes)))]))
            for _ in range(batch_size)
        ]
        x, y = make_batch(vocab, examples, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if allowed_ids is not None and model.embed.weight.grad is not None:
            mask = torch.zeros_like(model.embed.weight.grad)
            mask.index_fill_(0, allowed_ids, 1.0)
            model.embed.weight.grad.mul_(mask)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % log_interval == 0 or step == steps:
            print(f"[{label}] step={step:04d}/{steps} loss={float(loss.item()):.4f}", flush=True)
    print(f"[{label}] wall_time_sec={time.time() - start:.1f}", flush=True)


@torch.no_grad()
def generate(model: TinyGRULM, vocab: Vocab, prompt: Sequence[str], device: torch.device, max_new: int = 14) -> Tuple[str, ...]:
    model.eval()
    ids = vocab.encode(prompt)
    for _ in range(max_new):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        next_id = int(torch.argmax(model(x)[0, -1]).item())
        ids.append(next_id)
        if next_id == vocab.eos_id:
            break
    return vocab.decode(ids[len(prompt) :])


def final_answer(tokens: Sequence[str]) -> Tuple[str, ...]:
    seq = tuple(tokens)
    if "ANS" not in seq:
        return seq
    idx = max(i for i, token in enumerate(seq) if token == "ANS")
    return seq[idx:]


def parse_map(tokens: Sequence[str]) -> Dict[str, str] | None:
    seq = tuple(tokens)
    if len(seq) < 7 or seq[0] != "MAP":
        return None
    out: Dict[str, str] = {}
    for idx in (1, 3, 5):
        slot, value = seq[idx], seq[idx + 1]
        if slot not in SLOTS or value not in VALUES:
            return None
        out[slot] = value
    return out if set(out) == set(SLOTS) else None


def parse_order(tokens: Sequence[str]) -> Tuple[str, str, str] | None:
    seq = tuple(tokens)
    if len(seq) < 4 or seq[0] != "ORDER":
        return None
    order = tuple(seq[1:4])
    if sorted(order) != sorted(SLOTS):
        return None
    return order  # type: ignore[return-value]


def evaluate(model: TinyGRULM, vocab: Vocab, device: torch.device, mode: str, samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    correct = 0.0
    for _ in range(samples):
        example = sample_example(rng, mode)
        predicted = generate(model, vocab, example.prompt, device)
        correct += float(predicted == example.target)
    return correct / max(samples, 1)


def evaluate_final_answer(
    model: TinyGRULM,
    vocab: Vocab,
    device: torch.device,
    mode: str,
    samples: int,
    seed: int,
    *,
    max_new: int = 28,
) -> float:
    rng = np.random.default_rng(seed)
    correct = 0.0
    for _ in range(samples):
        example = sample_example(rng, mode)
        predicted = generate(model, vocab, example.prompt, device, max_new=max_new)
        correct += float(final_answer(predicted) == final_answer(example.target))
    return correct / max(samples, 1)


def self_scaffold_examples(
    model: TinyGRULM,
    vocab: Vocab,
    device: torch.device,
    rng: np.random.Generator,
    batch_size: int,
) -> Tuple[List[Example], int]:
    examples: List[Example] = []
    attempts = 0
    while len(examples) < batch_size and attempts < batch_size * 8:
        attempts += 1
        values = sample_values(rng)
        route_id = int(NEW_ROUTES[int(rng.integers(0, len(NEW_ROUTES)))])
        keys = sample_keys(rng, NEW_KEYS)
        b_pred = generate(model, vocab, make_b(route_id, values).prompt, device, max_new=10)
        d_pred = generate(model, vocab, make_d(keys, NEW_KEYS).prompt, device, max_new=8)
        mapping = parse_map(b_pred)
        order = parse_order(d_pred)
        if mapping is None or order is None:
            continue
        c_prompt = make_c(route_id, values, keys, NEW_KEYS).prompt
        examples.append(Example("C_SELF", c_prompt, (*map_tokens(mapping), *order_tokens(order), *ans_tokens(mapping, order))))
    return examples, attempts


def train_self_scaffold_phase(
    model: TinyGRULM,
    vocab: Vocab,
    device: torch.device,
    *,
    label: str,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    log_interval: int,
    pool_size: int,
) -> None:
    rng = np.random.default_rng(seed)
    for param in model.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    start = time.time()
    pool_target = int(pool_size) if int(pool_size) > 0 else max(batch_size * 8, 256)
    print(f"[{label}] building self-scaffold pool target={pool_target}", flush=True)
    pool, total_attempts = self_scaffold_examples(model, vocab, device, rng, pool_target)
    acceptance = len(pool) / max(total_attempts, 1)
    print(
        f"[{label}] pool_examples={len(pool)} attempts={total_attempts} acceptance={acceptance:.3f}",
        flush=True,
    )
    if not pool:
        print(f"[{label}] skipped: no parseable self-scaffold traces", flush=True)
        print(f"[{label}] wall_time_sec={time.time() - start:.1f}", flush=True)
        return
    model.train()  # restore training mode since generate() called eval()
    for step in range(1, steps + 1):
        indices = rng.integers(0, len(pool), size=batch_size)
        examples = [pool[int(idx)] for idx in indices]
        x, y = make_batch(vocab, examples, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % log_interval == 0 or step == steps:
            print(
                f"[{label}] step={step:04d}/{steps} loss={float(loss.item()):.4f} "
                f"pool_examples={len(pool)} acceptance={acceptance:.3f}",
                flush=True,
            )
    print(f"[{label}] wall_time_sec={time.time() - start:.1f}", flush=True)


def run_condition(
    *,
    label: str,
    vocab: Vocab,
    device: torch.device,
    seed: int,
    meta_steps: int,
    b_steps: int,
    d_steps: int,
    batch_size: int,
    lr: float,
    eval_samples: int,
    meta_modes: Sequence[str],
    embedding_only_new_skills: bool = False,
    embedding_lr_scale: float = 10.0,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    model = TinyGRULM(len(vocab.itos)).to(device)
    if meta_steps > 0:
        train_phase(
            model,
            vocab,
            device,
            label=f"{label}:meta",
            modes=meta_modes,
            seed=seed + 1,
            steps=meta_steps,
            batch_size=batch_size,
            lr=lr,
            log_interval=max(meta_steps // 4, 1),
        )
    trainable_ids = None
    if embedding_only_new_skills:
        trainable_ids = [
            vocab.stoi[f"r{route_id}"]
            for route_id in NEW_ROUTES
        ] + [vocab.stoi[key] for key in NEW_KEYS]
    skill_lr = float(lr) * (float(embedding_lr_scale) if embedding_only_new_skills else 1.0)
    train_phase(
        model,
        vocab,
        device,
        label=f"{label}:B",
        modes=("new_b",),
        seed=seed + 2,
        steps=b_steps,
        batch_size=batch_size,
        lr=skill_lr,
        log_interval=max(b_steps // 3, 1),
        trainable_token_ids=trainable_ids,
    )
    train_phase(
        model,
        vocab,
        device,
        label=f"{label}:D",
        modes=("new_d",),
        seed=seed + 3,
        steps=d_steps,
        batch_size=batch_size,
        lr=skill_lr,
        log_interval=max(d_steps // 3, 1),
        trainable_token_ids=trainable_ids,
    )
    metrics = {
        "new_B": evaluate(model, vocab, device, "new_b", eval_samples, seed + 10),
        "new_D": evaluate(model, vocab, device, "new_d", eval_samples, seed + 11),
        "meta_C": evaluate(model, vocab, device, "meta_c", eval_samples, seed + 12),
        "new_C_direct": evaluate(model, vocab, device, "new_c", eval_samples, seed + 13),
        "meta_C_final": evaluate_final_answer(model, vocab, device, "meta_c_scratch", eval_samples, seed + 14),
        "new_C_final": evaluate_final_answer(model, vocab, device, "new_c_scratch", eval_samples, seed + 15),
    }
    print(
        f"{label:<22} "
        f"B={metrics['new_B']:.3f} D={metrics['new_D']:.3f} "
        f"metaC={metrics['meta_C']:.3f} newC_direct={metrics['new_C_direct']:.3f} "
        f"metaC_final={metrics['meta_C_final']:.3f} newC_final={metrics['new_C_final']:.3f}",
        flush=True,
    )
    return metrics


def run_lateral_condition(
    *,
    label: str,
    vocab: Vocab,
    device: torch.device,
    seed: int,
    meta_steps: int,
    b_steps: int,
    d_steps: int,
    lateral_steps: int,
    batch_size: int,
    lr: float,
    lateral_lr: float,
    eval_samples: int,
    meta_modes: Sequence[str],
    lateral_pool_size: int,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    model = TinyGRULM(len(vocab.itos)).to(device)
    train_phase(
        model,
        vocab,
        device,
        label=f"{label}:meta",
        modes=meta_modes,
        seed=seed + 1,
        steps=meta_steps,
        batch_size=batch_size,
        lr=lr,
        log_interval=max(meta_steps // 4, 1),
    )
    train_phase(
        model,
        vocab,
        device,
        label=f"{label}:B",
        modes=("new_b",),
        seed=seed + 2,
        steps=b_steps,
        batch_size=batch_size,
        lr=lr,
        log_interval=max(b_steps // 3, 1),
    )
    train_phase(
        model,
        vocab,
        device,
        label=f"{label}:D_with_B_anchor",
        modes=("new_d", "new_b"),
        seed=seed + 3,
        steps=d_steps,
        batch_size=batch_size,
        lr=lr,
        log_interval=max(d_steps // 3, 1),
    )
    train_self_scaffold_phase(
        model,
        vocab,
        device,
        label=f"{label}:lateral_self_scaffold",
        seed=seed + 4,
        steps=lateral_steps,
        batch_size=batch_size,
        lr=lateral_lr,
        log_interval=max(lateral_steps // 3, 1),
        pool_size=lateral_pool_size,
    )
    metrics = {
        "new_B": evaluate(model, vocab, device, "new_b", eval_samples, seed + 10),
        "new_D": evaluate(model, vocab, device, "new_d", eval_samples, seed + 11),
        "meta_C": evaluate(model, vocab, device, "meta_c", eval_samples, seed + 12),
        "new_C_direct": evaluate(model, vocab, device, "new_c", eval_samples, seed + 13),
        "meta_C_final": evaluate_final_answer(model, vocab, device, "meta_c_scratch", eval_samples, seed + 14),
        "new_C_final": evaluate_final_answer(model, vocab, device, "new_c_scratch", eval_samples, seed + 15),
    }
    print(
        f"{label:<22} "
        f"B={metrics['new_B']:.3f} D={metrics['new_D']:.3f} "
        f"metaC={metrics['meta_C']:.3f} newC_direct={metrics['new_C_direct']:.3f} "
        f"metaC_final={metrics['meta_C_final']:.3f} newC_final={metrics['new_C_final']:.3f}",
        flush=True,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy automatic composition from sequential training; logs only.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--meta-steps", type=int, default=1800)
    parser.add_argument("--b-steps", type=int, default=900)
    parser.add_argument("--d-steps", type=int, default=900)
    parser.add_argument("--lateral-steps", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lateral-lr", type=float, default=8e-4)
    parser.add_argument("--lateral-pool-size", type=int, default=0)
    parser.add_argument("--embedding-lr-scale", type=float, default=10.0)
    parser.add_argument("--meta-scratchpad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast:
        args.meta_steps = min(args.meta_steps, 700)
        args.b_steps = min(args.b_steps, 400)
        args.d_steps = min(args.d_steps, 400)
        args.lateral_steps = min(args.lateral_steps, 300)
        args.lateral_pool_size = min(args.lateral_pool_size, 512) if args.lateral_pool_size else 512
        args.batch_size = min(args.batch_size, 96)
        args.eval_samples = min(args.eval_samples, 128)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    vocab = Vocab()

    print("=" * 96, flush=True)
    print("TOY META-COMPOSER AUTOMATICITY PROBE", flush=True)
    print("=" * 96, flush=True)
    print(
        f"device={device} seed={args.seed} meta_steps={args.meta_steps} "
        f"b_steps={args.b_steps} d_steps={args.d_steps} batch={args.batch_size} eval={args.eval_samples}",
        flush=True,
    )
    print(
        "heldout_test: baseline conditions never train C(new_route,new_keys). "
        "The lateral condition uses self-generated C_SELF traces from learned B and D, not human C labels.",
        flush=True,
    )
    print(f"meta_scratchpad={args.meta_scratchpad}", flush=True)
    print(
        "metric_note: newC_direct requires exact no-scratchpad ANS; "
        "newC_final allows scratchpad/interface tokens and checks the final ANS.",
        flush=True,
    )
    print("-" * 96, flush=True)

    no_meta = run_condition(
        label="no_meta_then_BD",
        vocab=vocab,
        device=device,
        seed=args.seed,
        meta_steps=0,
        b_steps=args.b_steps,
        d_steps=args.d_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_samples=args.eval_samples,
        meta_modes=(),
    )
    meta = run_condition(
        label="metaC_then_BD",
        vocab=vocab,
        device=device,
        seed=args.seed + 100,
        meta_steps=args.meta_steps,
        b_steps=args.b_steps,
        d_steps=args.d_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_samples=args.eval_samples,
        meta_modes=("meta_c_scratch" if args.meta_scratchpad else "meta_c",),
    )
    meta_tools = run_condition(
        label="metaBCD_then_BD",
        vocab=vocab,
        device=device,
        seed=args.seed + 200,
        meta_steps=args.meta_steps,
        b_steps=args.b_steps,
        d_steps=args.d_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_samples=args.eval_samples,
        meta_modes=("meta_b", "meta_d", "meta_c_scratch" if args.meta_scratchpad else "meta_c"),
    )
    meta_embed = run_condition(
        label="metaBCD_embedBD",
        vocab=vocab,
        device=device,
        seed=args.seed + 300,
        meta_steps=args.meta_steps,
        b_steps=args.b_steps,
        d_steps=args.d_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_samples=args.eval_samples,
        meta_modes=("meta_b", "meta_d", "meta_c_scratch" if args.meta_scratchpad else "meta_c"),
        embedding_only_new_skills=True,
        embedding_lr_scale=args.embedding_lr_scale,
    )
    lateral = run_lateral_condition(
        label="metaBCD_lateral",
        vocab=vocab,
        device=device,
        seed=args.seed + 400,
        meta_steps=args.meta_steps,
        b_steps=args.b_steps,
        d_steps=args.d_steps,
        lateral_steps=args.lateral_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lateral_lr=args.lateral_lr,
        eval_samples=args.eval_samples,
        meta_modes=("meta_b", "meta_d", "meta_c_scratch" if args.meta_scratchpad else "meta_c"),
        lateral_pool_size=args.lateral_pool_size,
    )

    print("=" * 96, flush=True)
    print("INTERPRETATION", flush=True)
    print("=" * 96, flush=True)
    print(
        f"no_meta newC={no_meta['new_C_direct']:.3f}; "
        f"metaC newC={meta['new_C_direct']:.3f}; "
        f"metaBCD newC={meta_tools['new_C_direct']:.3f}; "
        f"metaEmbed newC={meta_embed['new_C_direct']:.3f}; "
        f"lateral newC={lateral['new_C_direct']:.3f}",
        flush=True,
    )
    print(
        f"final-answer automaticity: no_meta={no_meta['new_C_final']:.3f}; "
        f"metaC={meta['new_C_final']:.3f}; metaBCD={meta_tools['new_C_final']:.3f}; "
        f"metaEmbed={meta_embed['new_C_final']:.3f}; lateral={lateral['new_C_final']:.3f}",
        flush=True,
    )
    best = max(meta["new_C_final"], meta_tools["new_C_final"], meta_embed["new_C_final"], lateral["new_C_final"])
    if best > no_meta["new_C_final"] + 0.50:
        print(
            "signal: lateral self-distillation can melt scaffolded composition into direct final-answer behavior.",
            flush=True,
        )
    elif best > no_meta["new_C_final"] + 0.10:
        print(
            "mixed signal: meta-composition helps, but automaticity is not solved cleanly yet.",
            flush=True,
        )
    else:
        print(
            "negative signal: this architecture did not internalize automatic composition from meta-training alone.",
            flush=True,
        )
    print(
        "paper-safe wording: this is scaffold-to-weight composition, not pure zero-shot automaticity.",
        flush=True,
    )


if __name__ == "__main__":
    main()
