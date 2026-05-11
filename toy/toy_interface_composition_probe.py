#!/usr/bin/env python3
"""Toy probe for zero-shot compositionality via shared interfaces.

This is intentionally small and log-only. It tests whether two separately
learned skills compose better when both skills are trained around typed
intermediate states.

Tasks:
  B route:     record + route -> MAP(slot -> value)
  D sort:      slot keys -> ORDER(slots)
  F decoder:   MAP + ORDER -> ANS(sorted slot=value)
  C compose:   record + route + slot keys -> ANS(sorted slot=value)

Important: the interface condition trains B, D, and F, but trains ZERO raw C
examples unless --bridge-examples is set. Scaffolding C is done by calling the
same model three times: B(record), D(keys), F(map, order).
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

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
KEYS = ("k0", "k1", "k2")
VALUES = tuple(f"v{i}" for i in range(10))
SPECIAL = ("<pad>", "<eos>", "->", "B", "D", "F", "C", "MAP", "ORDER", "ANS")


@dataclass(frozen=True)
class Example:
    task: str
    prompt: Tuple[str, ...]
    target: Tuple[str, ...]


class Vocab:
    def __init__(self) -> None:
        tokens = list(SPECIAL) + list(SLOTS) + list(FIELDS) + [f"r{i}" for i in range(len(ROUTES))] + list(KEYS) + list(VALUES)
        self.itos = list(dict.fromkeys(tokens))
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.eos_id = self.stoi["<eos>"]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi[token] for token in tokens]

    def decode(self, ids: Sequence[int]) -> List[str]:
        out: List[str] = []
        for idx in ids:
            token = self.itos[int(idx)]
            if token == "<eos>":
                break
            if token != "<pad>":
                out.append(token)
        return out


class TinyGRULM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 160, layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=layers, batch_first=True)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(self.embed(input_ids))
        return self.lm_head(hidden)


def route_values(route_id: int, field_values: Sequence[str]) -> Dict[str, str]:
    perm = ROUTES[int(route_id)]
    return {slot: field_values[field_idx] for slot, field_idx in zip(SLOTS, perm)}


def sorted_slots(slot_keys: Sequence[str]) -> Tuple[str, str, str]:
    return tuple(slot for _, slot in sorted(zip(slot_keys, SLOTS), key=lambda pair: KEYS.index(pair[0])))


def answer_tokens(slot_map: Dict[str, str], order: Sequence[str]) -> Tuple[str, ...]:
    tokens: List[str] = ["ANS"]
    for slot in order:
        tokens.extend([slot, slot_map[slot]])
    return tuple(tokens)


def sample_world(rng: np.random.Generator) -> Tuple[int, Tuple[str, str, str], Tuple[str, str, str]]:
    route_id = int(rng.integers(0, len(ROUTES)))
    field_values = tuple(VALUES[int(rng.integers(0, len(VALUES)))] for _ in FIELDS)
    slot_keys = tuple(KEYS[int(rng.integers(0, len(KEYS)))] for _ in SLOTS)
    return route_id, field_values, slot_keys


def make_b_example(route_id: int, field_values: Sequence[str]) -> Example:
    slot_map = route_values(route_id, field_values)
    prompt = ("B", f"r{route_id}", "a", field_values[0], "b", field_values[1], "c", field_values[2], "->")
    target = ("MAP", "s0", slot_map["s0"], "s1", slot_map["s1"], "s2", slot_map["s2"])
    return Example("B", prompt, target)


def make_d_example(slot_keys: Sequence[str]) -> Example:
    order = sorted_slots(slot_keys)
    prompt = ("D", "s0", slot_keys[0], "s1", slot_keys[1], "s2", slot_keys[2], "->")
    target = ("ORDER", order[0], order[1], order[2])
    return Example("D", prompt, target)


def make_f_example(slot_map: Dict[str, str], order: Sequence[str]) -> Example:
    prompt = (
        "F",
        "MAP",
        "s0",
        slot_map["s0"],
        "s1",
        slot_map["s1"],
        "s2",
        slot_map["s2"],
        "ORDER",
        order[0],
        order[1],
        order[2],
        "->",
    )
    return Example("F", prompt, answer_tokens(slot_map, order))


def make_c_example(route_id: int, field_values: Sequence[str], slot_keys: Sequence[str]) -> Example:
    slot_map = route_values(route_id, field_values)
    order = sorted_slots(slot_keys)
    prompt = (
        "C",
        f"r{route_id}",
        "a",
        field_values[0],
        "b",
        field_values[1],
        "c",
        field_values[2],
        "s0",
        slot_keys[0],
        "s1",
        slot_keys[1],
        "s2",
        slot_keys[2],
        "->",
    )
    return Example("C", prompt, answer_tokens(slot_map, order))


def make_random_example(rng: np.random.Generator, tasks: Sequence[str]) -> Example:
    route_id, field_values, slot_keys = sample_world(rng)
    slot_map = route_values(route_id, field_values)
    order = sorted_slots(slot_keys)
    task = str(tasks[int(rng.integers(0, len(tasks)))])
    if task == "B":
        return make_b_example(route_id, field_values)
    if task == "D":
        return make_d_example(slot_keys)
    if task == "F":
        return make_f_example(slot_map, order)
    if task == "C":
        return make_c_example(route_id, field_values, slot_keys)
    raise ValueError(f"unknown task: {task}")


def make_batch(
    vocab: Vocab,
    examples: Sequence[Example],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded: List[List[int]] = []
    label_rows: List[List[int]] = []
    for example in examples:
        prompt_ids = vocab.encode(example.prompt)
        target_ids = vocab.encode((*example.target, "<eos>"))
        full = prompt_ids + target_ids
        input_ids = full[:-1]
        labels = full[1:]
        mask_until = max(len(prompt_ids) - 1, 0)
        labels = [-100] * mask_until + labels[mask_until:]
        encoded.append(input_ids)
        label_rows.append(labels)
    max_len = max(len(row) for row in encoded)
    input_tensor = torch.full((len(encoded), max_len), vocab.pad_id, dtype=torch.long, device=device)
    label_tensor = torch.full((len(encoded), max_len), -100, dtype=torch.long, device=device)
    for row_idx, (input_ids, labels) in enumerate(zip(encoded, label_rows)):
        input_tensor[row_idx, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long, device=device)
        label_tensor[row_idx, : len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)
    return input_tensor, label_tensor


def train_model(
    *,
    label: str,
    vocab: Vocab,
    device: torch.device,
    seed: int,
    tasks: Sequence[str],
    steps: int,
    batch_size: int,
    lr: float,
    bridge_examples: Sequence[Example] = (),
    log_interval: int = 200,
) -> TinyGRULM:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyGRULM(len(vocab.itos)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    start = time.time()
    bridge = list(bridge_examples)
    for step in range(1, steps + 1):
        examples = [make_random_example(rng, tasks) for _ in range(batch_size)]
        if bridge and step % 4 == 0:
            examples[0] = bridge[int(rng.integers(0, len(bridge)))]
        input_ids, labels = make_batch(vocab, examples, device)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % log_interval == 0 or step == steps:
            print(f"[{label}] step={step:04d}/{steps} loss={float(loss.item()):.4f}", flush=True)
    print(f"[{label}] wall_time_sec={time.time() - start:.1f}", flush=True)
    return model


@torch.no_grad()
def generate(
    model: TinyGRULM,
    vocab: Vocab,
    prompt: Sequence[str],
    device: torch.device,
    max_new_tokens: int = 16,
) -> Tuple[str, ...]:
    model.eval()
    ids = vocab.encode(prompt)
    for _ in range(max_new_tokens):
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(input_tensor)
        next_id = int(torch.argmax(logits[0, -1]).item())
        ids.append(next_id)
        if next_id == vocab.eos_id:
            break
    return tuple(vocab.decode(ids[len(prompt) :]))


def exact(predicted: Sequence[str], target: Sequence[str]) -> float:
    return float(tuple(predicted) == tuple(target))


def evaluate_task(
    model: TinyGRULM,
    vocab: Vocab,
    device: torch.device,
    task: str,
    samples: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    correct = 0.0
    for _ in range(samples):
        route_id, field_values, slot_keys = sample_world(rng)
        if task == "B":
            example = make_b_example(route_id, field_values)
        elif task == "D":
            example = make_d_example(slot_keys)
        elif task == "F":
            example = make_f_example(route_values(route_id, field_values), sorted_slots(slot_keys))
        elif task == "C":
            example = make_c_example(route_id, field_values, slot_keys)
        else:
            raise ValueError(task)
        predicted = generate(model, vocab, example.prompt, device, max_new_tokens=16)
        correct += exact(predicted, example.target)
    return correct / max(samples, 1)


def parse_map(tokens: Sequence[str]) -> Dict[str, str] | None:
    if len(tokens) != 7 or tokens[0] != "MAP":
        return None
    out: Dict[str, str] = {}
    for idx in (1, 3, 5):
        slot, value = tokens[idx], tokens[idx + 1]
        if slot not in SLOTS or value not in VALUES:
            return None
        out[slot] = value
    return out if set(out) == set(SLOTS) else None


def parse_order(tokens: Sequence[str]) -> Tuple[str, str, str] | None:
    if len(tokens) != 4 or tokens[0] != "ORDER":
        return None
    order = tuple(tokens[1:4])
    if sorted(order) != sorted(SLOTS):
        return None
    return order  # type: ignore[return-value]


def scaffold_compose_once(
    model: TinyGRULM,
    vocab: Vocab,
    device: torch.device,
    route_id: int,
    field_values: Sequence[str],
    slot_keys: Sequence[str],
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    b_example = make_b_example(route_id, field_values)
    d_example = make_d_example(slot_keys)
    map_tokens = generate(model, vocab, b_example.prompt, device, max_new_tokens=12)
    order_tokens = generate(model, vocab, d_example.prompt, device, max_new_tokens=8)
    slot_map = parse_map(map_tokens)
    order = parse_order(order_tokens)
    if slot_map is None or order is None:
        return map_tokens, order_tokens, ()
    f_example = make_f_example(slot_map, order)
    answer = generate(model, vocab, f_example.prompt, device, max_new_tokens=12)
    return map_tokens, order_tokens, answer


def evaluate_scaffold_composition(
    model: TinyGRULM,
    vocab: Vocab,
    device: torch.device,
    samples: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    b_ok = 0.0
    d_ok = 0.0
    f_ok = 0.0
    c_ok = 0.0
    for _ in range(samples):
        route_id, field_values, slot_keys = sample_world(rng)
        gold_map = make_b_example(route_id, field_values).target
        gold_order = make_d_example(slot_keys).target
        gold_answer = make_c_example(route_id, field_values, slot_keys).target
        map_tokens, order_tokens, answer = scaffold_compose_once(model, vocab, device, route_id, field_values, slot_keys)
        b_ok += exact(map_tokens, gold_map)
        d_ok += exact(order_tokens, gold_order)
        f_ok += float(bool(answer))
        c_ok += exact(answer, gold_answer)
    denom = max(samples, 1)
    return b_ok / denom, d_ok / denom, f_ok / denom, c_ok / denom


def make_bridge_set(seed: int, count: int) -> List[Example]:
    rng = np.random.default_rng(seed)
    return [make_random_example(rng, ("C",)) for _ in range(max(int(count), 0))]


def run_condition(
    *,
    label: str,
    vocab: Vocab,
    device: torch.device,
    seed: int,
    tasks: Sequence[str],
    steps: int,
    batch_size: int,
    lr: float,
    eval_samples: int,
    bridge_examples: Sequence[Example] = (),
) -> Dict[str, float]:
    model = train_model(
        label=label,
        vocab=vocab,
        device=device,
        seed=seed,
        tasks=tasks,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        bridge_examples=bridge_examples,
    )
    metrics = {
        "B_exact": evaluate_task(model, vocab, device, "B", eval_samples, seed + 11),
        "D_exact": evaluate_task(model, vocab, device, "D", eval_samples, seed + 12),
        "F_exact": evaluate_task(model, vocab, device, "F", eval_samples, seed + 13),
        "C_direct_exact": evaluate_task(model, vocab, device, "C", eval_samples, seed + 14),
    }
    b_step, d_step, f_step, c_scaffold = evaluate_scaffold_composition(model, vocab, device, eval_samples, seed + 15)
    metrics.update(
        {
            "scaffold_B_exact": b_step,
            "scaffold_D_exact": d_step,
            "scaffold_F_parse": f_step,
            "C_scaffold_exact": c_scaffold,
        }
    )
    print(
        f"{label:<24} "
        f"B={metrics['B_exact']:.3f} D={metrics['D_exact']:.3f} F={metrics['F_exact']:.3f} "
        f"C_direct={metrics['C_direct_exact']:.3f} "
        f"C_scaffold={metrics['C_scaffold_exact']:.3f} "
        f"stepB={metrics['scaffold_B_exact']:.3f} stepD={metrics['scaffold_D_exact']:.3f}",
        flush=True,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy shared-interface composition probe; logs only.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--bridge-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--bridge-examples", type=int, default=16)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast:
        args.steps = min(args.steps, 700)
        args.batch_size = min(args.batch_size, 64)
        args.eval_samples = min(args.eval_samples, 128)
    bridge_steps = int(args.bridge_steps or max(args.steps, 1))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    vocab = Vocab()

    print("=" * 96, flush=True)
    print("TOY SHARED-INTERFACE COMPOSITION PROBE", flush=True)
    print("=" * 96, flush=True)
    print(
        f"device={device} seed={args.seed} steps={args.steps} batch={args.batch_size} "
        f"eval_samples={args.eval_samples}",
        flush=True,
    )
    print(
        "claim_test: no raw C composition examples are used in interface_bdf; "
        "composition is attempted by B(record)->MAP, D(keys)->ORDER, F(MAP,ORDER)->ANS.",
        flush=True,
    )
    print("-" * 96, flush=True)

    rows: Dict[str, Dict[str, float]] = {}
    rows["isolated_bd"] = run_condition(
        label="isolated_bd",
        vocab=vocab,
        device=device,
        seed=args.seed,
        tasks=("B", "D"),
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_samples=args.eval_samples,
    )
    rows["interface_bdf"] = run_condition(
        label="interface_bdf",
        vocab=vocab,
        device=device,
        seed=args.seed + 100,
        tasks=("B", "D", "F"),
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_samples=args.eval_samples,
    )
    bridge = make_bridge_set(args.seed + 200, args.bridge_examples)
    rows[f"interface_bridge{args.bridge_examples}"] = run_condition(
        label=f"interface_bridge{args.bridge_examples}",
        vocab=vocab,
        device=device,
        seed=args.seed + 200,
        tasks=("B", "D", "F"),
        steps=bridge_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_samples=args.eval_samples,
        bridge_examples=bridge,
    )

    print("=" * 96, flush=True)
    print("INTERPRETATION", flush=True)
    print("=" * 96, flush=True)
    isolated = rows["isolated_bd"]["C_scaffold_exact"]
    interface = rows["interface_bdf"]["C_scaffold_exact"]
    bridge_score = rows[f"interface_bridge{args.bridge_examples}"]["C_direct_exact"]
    print(
        f"isolated_bd scaffold composition={isolated:.3f}; "
        f"interface_bdf scaffold composition={interface:.3f}; "
        f"bridge direct composition={bridge_score:.3f}",
        flush=True,
    )
    if interface > isolated + 0.50:
        print(
            "signal: typed interfaces unlock composition without raw C training, but via scaffolded interface calls.",
            flush=True,
        )
    else:
        print(
            "signal: interface composition is not clean yet; first check B/D/F exact scores before interpreting.",
            flush=True,
        )
    if bridge_score > rows["interface_bdf"]["C_direct_exact"] + 0.50:
        print(
            "signal: a tiny bridge teaches one-shot direct composition syntax, separate from the interface mechanism.",
            flush=True,
        )
    print(
        "paper-safe wording: this tests efficient compositional binding, not spontaneous single-prompt composition.",
        flush=True,
    )


if __name__ == "__main__":
    main()
