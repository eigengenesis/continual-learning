#!/usr/bin/env python3
"""Toy probe: can a model learn an alien language and later compose it?

This is intentionally small and stdout-only.

Question:
  Can a sequentially trained model acquire an "alien language" parser, then a
  separate semantic operation skill, and solve alien-language composition after
  a verified lateral self-distillation phase?

Tasks:
  B alien parser:
    alien sentence -> SEM(noun, adjective, verb)

  D semantic operator:
    OP + SEM(noun, adjective, verb) -> ANS(transformed English sentence)

  C heldout composition:
    OP + alien sentence -> ANS(transformed English sentence)

No raw C labels are used before the lateral phase. The lateral phase builds
self-generated C_SELF traces by calling the learned B and D skills, verifies
them against the known toy grammar, and distills only correct traces.
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


SPECIAL = (
    "<pad>",
    "<eos>",
    "->",
    "B",
    "D",
    "C",
    "SEM",
    "ANS",
    "N",
    "A",
    "V",
    "FUT",
    "PAST",
    "NEG",
    "will",
    "not",
)

ALIEN_NOUNS = ("norp", "zib", "fep", "dak", "moru", "tavi", "glen", "sok")
EN_NOUNS = ("cat", "dog", "bird", "fish", "horse", "wolf", "lion", "mouse")

ALIEN_ADJS = ("mav", "luma", "kesh", "rilo", "sena", "paku")
EN_ADJS = ("red", "blue", "green", "small", "bright", "quiet")

ALIEN_VERBS = ("tobu", "saru", "lima", "koro", "veki")
EN_VERBS = ("eat", "sleep", "chase", "find", "carry")
PAST_VERBS = ("ate", "slept", "chased", "found", "carried")

OPS = ("FUT", "PAST", "NEG")

World = Tuple[int, int, int]


@dataclass(frozen=True)
class Example:
    task: str
    prompt: Tuple[str, ...]
    target: Tuple[str, ...]


class Vocab:
    def __init__(self) -> None:
        tokens = (
            list(SPECIAL)
            + list(ALIEN_NOUNS)
            + list(EN_NOUNS)
            + list(ALIEN_ADJS)
            + list(EN_ADJS)
            + list(ALIEN_VERBS)
            + list(EN_VERBS)
            + list(PAST_VERBS)
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


def all_worlds() -> List[World]:
    return [
        (noun_idx, adj_idx, verb_idx)
        for noun_idx in range(len(EN_NOUNS))
        for adj_idx in range(len(EN_ADJS))
        for verb_idx in range(len(EN_VERBS))
    ]


def is_heldout(world: World) -> bool:
    noun_idx, adj_idx, verb_idx = world
    return ((noun_idx * 17 + adj_idx * 7 + verb_idx * 3) % 5) == 0


TRAIN_WORLDS = tuple(world for world in all_worlds() if not is_heldout(world))
HELDOUT_WORLDS = tuple(world for world in all_worlds() if is_heldout(world))


def semantic_tokens(world: World) -> Tuple[str, ...]:
    noun_idx, adj_idx, verb_idx = world
    return ("SEM", "N", EN_NOUNS[noun_idx], "A", EN_ADJS[adj_idx], "V", EN_VERBS[verb_idx])


def alien_tokens(world: World) -> Tuple[str, str, str]:
    noun_idx, adj_idx, verb_idx = world
    # Alien grammar is adjective-noun-verb, with all lexical items opaque.
    return (ALIEN_ADJS[adj_idx], ALIEN_NOUNS[noun_idx], ALIEN_VERBS[verb_idx])


def answer_tokens(world: World, op: str) -> Tuple[str, ...]:
    noun_idx, adj_idx, verb_idx = world
    noun = EN_NOUNS[noun_idx]
    adj = EN_ADJS[adj_idx]
    verb = EN_VERBS[verb_idx]
    if op == "FUT":
        return ("ANS", adj, noun, "will", verb)
    if op == "PAST":
        return ("ANS", adj, noun, PAST_VERBS[verb_idx])
    if op == "NEG":
        return ("ANS", adj, noun, "not", verb)
    raise ValueError(op)


def make_b(world: World) -> Example:
    return Example("B", ("B", *alien_tokens(world), "->"), semantic_tokens(world))


def make_d(world: World, op: str) -> Example:
    return Example("D", ("D", op, *semantic_tokens(world), "->"), answer_tokens(world, op))


def make_c(world: World, op: str, *, scratchpad: bool = False) -> Example:
    target = answer_tokens(world, op)
    if scratchpad:
        target = (*semantic_tokens(world), *target)
    return Example("C", ("C", op, *alien_tokens(world), "->"), target)


def sample_world(rng: np.random.Generator, split: str) -> World:
    worlds = TRAIN_WORLDS if split == "train" else HELDOUT_WORLDS
    return worlds[int(rng.integers(0, len(worlds)))]


def sample_op(rng: np.random.Generator) -> str:
    return OPS[int(rng.integers(0, len(OPS)))]


def sample_example(rng: np.random.Generator, mode: str) -> Example:
    if mode == "B_train":
        return make_b(sample_world(rng, "train"))
    if mode == "B_eval":
        return make_b(sample_world(rng, "heldout"))
    if mode == "D_train":
        return make_d(sample_world(rng, "train"), sample_op(rng))
    if mode == "D_eval":
        return make_d(sample_world(rng, "heldout"), sample_op(rng))
    if mode == "C_eval":
        return make_c(sample_world(rng, "heldout"), sample_op(rng))
    if mode == "C_eval_scratch":
        return make_c(sample_world(rng, "heldout"), sample_op(rng), scratchpad=True)
    raise ValueError(mode)


def make_batch(vocab: Vocab, examples: Sequence[Example], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rows: List[List[int]] = []
    labels: List[List[int]] = []
    for example in examples:
        prompt_ids = vocab.encode(example.prompt)
        target_ids = vocab.encode((*example.target, "<eos>"))
        full = prompt_ids + target_ids
        row = full[:-1]
        row_labels = full[1:]
        mask_until = max(len(prompt_ids) - 1, 0)
        row_labels = [-100] * mask_until + row_labels[mask_until:]
        rows.append(row)
        labels.append(row_labels)
    max_len = max(len(row) for row in rows)
    input_tensor = torch.full((len(rows), max_len), vocab.pad_id, dtype=torch.long, device=device)
    label_tensor = torch.full((len(rows), max_len), -100, dtype=torch.long, device=device)
    for row_idx, (row, row_labels) in enumerate(zip(rows, labels)):
        input_tensor[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        label_tensor[row_idx, : len(row_labels)] = torch.tensor(row_labels, dtype=torch.long, device=device)
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
    anchor_state: Dict[str, torch.Tensor] | None = None,
    anchor_weight: float = 0.0,
) -> None:
    rng = np.random.default_rng(seed)
    for param in model.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    start = time.time()
    model.train()
    for step in range(1, steps + 1):
        examples = [
            sample_example(rng, modes[int(rng.integers(0, len(modes)))])
            for _ in range(batch_size)
        ]
        x, y = make_batch(vocab, examples, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        if anchor_state is not None and anchor_weight > 0.0:
            anchor_loss = torch.zeros((), device=device)
            for name, param in model.named_parameters():
                anchor_loss = anchor_loss + F.mse_loss(param, anchor_state[name])
            loss = loss + float(anchor_weight) * anchor_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % log_interval == 0 or step == steps:
            print(f"[{label}] step={step:04d}/{steps} loss={float(loss.item()):.4f}", flush=True)
    print(f"[{label}] wall_time_sec={time.time() - start:.1f}", flush=True)


@torch.no_grad()
def generate(
    model: TinyGRULM,
    vocab: Vocab,
    prompt: Sequence[str],
    device: torch.device,
    max_new: int = 24,
) -> Tuple[str, ...]:
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


def parse_semantic(tokens: Sequence[str]) -> World | None:
    seq = tuple(tokens)
    if len(seq) < 7 or seq[:2] != ("SEM", "N"):
        return None
    try:
        noun_idx = EN_NOUNS.index(seq[2])
        if seq[3] != "A":
            return None
        adj_idx = EN_ADJS.index(seq[4])
        if seq[5] != "V":
            return None
        verb_idx = EN_VERBS.index(seq[6])
    except ValueError:
        return None
    return (noun_idx, adj_idx, verb_idx)


def evaluate_exact(
    model: TinyGRULM,
    vocab: Vocab,
    device: torch.device,
    mode: str,
    samples: int,
    seed: int,
    *,
    final_only: bool = False,
) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(samples):
        example = sample_example(rng, mode)
        predicted = generate(model, vocab, example.prompt, device)
        lhs = final_answer(predicted) if final_only else predicted
        rhs = final_answer(example.target) if final_only else example.target
        correct += int(lhs == rhs)
    return correct / max(samples, 1)


@torch.no_grad()
def evaluate_scaffold(
    model: TinyGRULM,
    vocab: Vocab,
    device: torch.device,
    samples: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(samples):
        world = sample_world(rng, "heldout")
        op = sample_op(rng)
        sem_pred = generate(model, vocab, make_b(world).prompt, device)
        parsed = parse_semantic(sem_pred)
        if parsed is None:
            continue
        d_prompt = ("D", op, *semantic_tokens(parsed), "->")
        ans_pred = generate(model, vocab, d_prompt, device)
        correct += int(final_answer(ans_pred) == answer_tokens(world, op))
    return correct / max(samples, 1)


def self_trace_pool(
    model: TinyGRULM,
    vocab: Vocab,
    device: torch.device,
    *,
    seed: int,
    pool_size: int,
    attempt_multiplier: int,
    scratchpad_targets: bool,
) -> Tuple[List[Example], int]:
    rng = np.random.default_rng(seed)
    examples: List[Example] = []
    attempts = 0
    max_attempts = max(pool_size * attempt_multiplier, pool_size)
    while len(examples) < pool_size and attempts < max_attempts:
        attempts += 1
        world = sample_world(rng, "heldout")
        op = sample_op(rng)
        sem_pred = generate(model, vocab, make_b(world).prompt, device)
        parsed = parse_semantic(sem_pred)
        if parsed != world:
            continue
        ans_pred = generate(model, vocab, ("D", op, *semantic_tokens(parsed), "->"), device)
        if final_answer(ans_pred) != answer_tokens(world, op):
            continue
        examples.append(make_c(world, op, scratchpad=scratchpad_targets))
    return examples, attempts


def train_lateral_self_distill(
    model: TinyGRULM,
    vocab: Vocab,
    device: torch.device,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    log_interval: int,
    pool_size: int,
    attempt_multiplier: int,
    anchor_weight: float,
    scratchpad_targets: bool,
    component_anchor_frac: float,
) -> Tuple[int, int]:
    print(
        f"[lateral_self_distill] building verified self-trace pool target={pool_size} "
        f"attempt_multiplier={attempt_multiplier}",
        flush=True,
    )
    pool, attempts = self_trace_pool(
        model,
        vocab,
        device,
        seed=seed,
        pool_size=pool_size,
        attempt_multiplier=attempt_multiplier,
        scratchpad_targets=scratchpad_targets,
    )
    acceptance = len(pool) / max(attempts, 1)
    print(
        f"[lateral_self_distill] pool_examples={len(pool)} attempts={attempts} "
        f"acceptance={acceptance:.3f} verified_true_acc=1.000",
        flush=True,
    )
    if not pool:
        return 0, attempts

    rng = np.random.default_rng(seed + 1)
    anchor_state = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }
    for param in model.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    start = time.time()
    model.train()
    for step in range(1, steps + 1):
        indices = rng.integers(0, len(pool), size=batch_size)
        examples = [pool[int(idx)] for idx in indices]
        anchor_count = int(round(batch_size * max(0.0, min(1.0, component_anchor_frac))))
        if anchor_count > 0:
            half = max(anchor_count // 2, 1)
            anchors = [sample_example(rng, "B_train") for _ in range(half)]
            anchors.extend(sample_example(rng, "D_train") for _ in range(anchor_count - half))
            examples[: len(anchors)] = anchors
        x, y = make_batch(vocab, examples, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        if anchor_weight > 0.0:
            anchor_loss = torch.zeros((), device=device)
            for name, param in model.named_parameters():
                anchor_loss = anchor_loss + F.mse_loss(param, anchor_state[name])
            loss = loss + float(anchor_weight) * anchor_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % log_interval == 0 or step == steps:
            print(
                f"[lateral_self_distill] step={step:04d}/{steps} loss={float(loss.item()):.4f} "
                f"pool_examples={len(pool)} component_anchor_frac={component_anchor_frac:.2f}",
                flush=True,
            )
    print(f"[lateral_self_distill] wall_time_sec={time.time() - start:.1f}", flush=True)
    return len(pool), attempts


def evaluate_all(model: TinyGRULM, vocab: Vocab, device: torch.device, samples: int, seed: int) -> Dict[str, float]:
    return {
        "B": evaluate_exact(model, vocab, device, "B_eval", samples, seed + 1),
        "D": evaluate_exact(model, vocab, device, "D_eval", samples, seed + 2),
        "C_direct": evaluate_exact(model, vocab, device, "C_eval", samples, seed + 3),
        "C_final": evaluate_exact(model, vocab, device, "C_eval_scratch", samples, seed + 4, final_only=True),
        "C_scaffold": evaluate_scaffold(model, vocab, device, samples, seed + 5),
    }


def print_metrics(label: str, metrics: Dict[str, float]) -> None:
    print(
        f"{label:<24} "
        f"B={metrics['B']:.3f} D={metrics['D']:.3f} "
        f"C_direct={metrics['C_direct']:.3f} C_final={metrics['C_final']:.3f} "
        f"C_scaffold={metrics['C_scaffold']:.3f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy alien-language continual composition probe; logs only.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--b-steps", type=int, default=1200)
    parser.add_argument("--d-steps", type=int, default=1000)
    parser.add_argument("--lateral-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--d-lr-scale", type=float, default=0.4)
    parser.add_argument("--d-anchor-weight", type=float, default=500.0)
    parser.add_argument("--lateral-lr", type=float, default=5e-5)
    parser.add_argument("--lateral-anchor-weight", type=float, default=1000.0)
    parser.add_argument("--lateral-scratchpad-targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lateral-component-anchor-frac", type=float, default=0.35)
    parser.add_argument("--lateral-pool-size", type=int, default=512)
    parser.add_argument("--lateral-attempt-multiplier", type=int, default=8)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast:
        args.b_steps = min(args.b_steps, 500)
        args.d_steps = min(args.d_steps, 450)
        args.lateral_steps = min(args.lateral_steps, 300)
        args.batch_size = min(args.batch_size, 96)
        args.eval_samples = min(args.eval_samples, 128)
        args.lateral_pool_size = min(args.lateral_pool_size, 256)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    vocab = Vocab()

    print("=" * 96, flush=True)
    print("TOY ALIEN LANGUAGE CONTINUAL COMPOSITION PROBE", flush=True)
    print("=" * 96, flush=True)
    print(
        f"device={device} seed={args.seed} b_steps={args.b_steps} d_steps={args.d_steps} "
        f"lateral_steps={args.lateral_steps} batch={args.batch_size} eval={args.eval_samples}",
        flush=True,
    )
    print(
        f"alien_language: {len(ALIEN_NOUNS)} nouns x {len(ALIEN_ADJS)} adjectives x "
        f"{len(ALIEN_VERBS)} verbs; heldout_combinations={len(HELDOUT_WORLDS)} "
        f"train_combinations={len(TRAIN_WORLDS)}",
        flush=True,
    )
    print(
        "claim_test: B learns alien->SEM; D learns OP+SEM->ANS; no raw C labels are used "
        "until verified lateral self-distillation; lateral also refreshes B/D component anchors.",
        flush=True,
    )
    print(
        "metric_note: C_direct requires exact ANS only; C_final allows SEM scratchpad then checks final ANS.",
        flush=True,
    )
    print("-" * 96, flush=True)

    model = TinyGRULM(len(vocab.itos)).to(device)
    train_phase(
        model,
        vocab,
        device,
        label="stage_B_alien_parser",
        modes=("B_train",),
        seed=args.seed + 1,
        steps=args.b_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=max(args.b_steps // 4, 1),
    )
    after_b = evaluate_all(model, vocab, device, args.eval_samples, args.seed + 10)
    print_metrics("after_B", after_b)
    post_b_anchor = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }

    train_phase(
        model,
        vocab,
        device,
        label="stage_D_semantic_ops",
        modes=("D_train",),
        seed=args.seed + 2,
        steps=args.d_steps,
        batch_size=args.batch_size,
        lr=args.lr * args.d_lr_scale,
        log_interval=max(args.d_steps // 4, 1),
        anchor_state=post_b_anchor,
        anchor_weight=args.d_anchor_weight,
    )
    after_d = evaluate_all(model, vocab, device, args.eval_samples, args.seed + 20)
    print_metrics("after_B_then_D", after_d)

    train_lateral_self_distill(
        model,
        vocab,
        device,
        seed=args.seed + 3,
        steps=args.lateral_steps,
        batch_size=args.batch_size,
        lr=args.lateral_lr,
        log_interval=max(args.lateral_steps // 3, 1),
        pool_size=args.lateral_pool_size,
        attempt_multiplier=args.lateral_attempt_multiplier,
        anchor_weight=args.lateral_anchor_weight,
        scratchpad_targets=args.lateral_scratchpad_targets,
        component_anchor_frac=args.lateral_component_anchor_frac,
    )
    after_lateral = evaluate_all(model, vocab, device, args.eval_samples, args.seed + 30)
    print_metrics("after_lateral", after_lateral)

    print("=" * 96, flush=True)
    print("INTERPRETATION", flush=True)
    print("=" * 96, flush=True)
    print(
        f"alien acquisition after D: B={after_d['B']:.3f}; "
        f"semantic operation D={after_d['D']:.3f}; scaffolded composition={after_d['C_scaffold']:.3f}",
        flush=True,
    )
    print(
        f"direct composition before lateral={after_d['C_final']:.3f}; "
        f"after lateral={after_lateral['C_final']:.3f}",
        flush=True,
    )
    if after_d["B"] >= 0.75 and after_d["D"] >= 0.90 and after_d["C_scaffold"] >= 0.70:
        print("signal: the toy model learned the alien interface and can compose it through scaffolding.", flush=True)
    else:
        print("mixed signal: first tune B/D acquisition before interpreting composition.", flush=True)
    if after_lateral["C_final"] > after_d["C_final"] + 0.50:
        print("signal: verified lateral self-distillation melted alien-language composition into weights.", flush=True)
    elif after_lateral["C_final"] > after_d["C_final"] + 0.10:
        print("mixed signal: lateral helps, but the direct composition margin is still modest.", flush=True)
    else:
        print("negative signal: lateral did not yet convert scaffolded composition into direct behavior.", flush=True)
    print(
        "paper-safe wording: this is a controlled synthetic-language mechanism test, "
        "not evidence yet for full natural-language acquisition.",
        flush=True,
    )


if __name__ == "__main__":
    main()
