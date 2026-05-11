#!/usr/bin/env python3
"""Accelerated grokking / gradient-shaping audit.

Claim under test:

    Generalization is a geometric event in weight space. If we shape gradients
    toward algorithmically structured directions, a model should reach heldout
    generalization in fewer optimizer steps than the same model trained with
    ordinary SGD/AdamW.

This audit intentionally uses a tiny, reproducible modular-arithmetic grokking
setup because it gives a clean falsification target:

    same initialization + same train split + same optimizer budget
    -> shaped gradients must hit heldout accuracy earlier than baseline

Artifacts:
  accelerated_grokking_curves.csv
  accelerated_grokking_summary.csv
  accelerated_grokking_verdict.json
  accelerated_grokking_config.json

The "fourier_shape" branches are positive controls for the user's thesis:
they do not use heldout labels or extra examples, but they do inject the known
symmetry shape of modular arithmetic into the gradient flow. The SVD/Z branches
are model-generic gradient geometry controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SPECIAL_PLUS = "plus"
SPECIAL_EQ = "eq"
SPECIAL_BOS = "bos"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_seeds(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def format_step(step: int | None) -> str:
    return "never" if step is None else str(int(step))


def make_modular_dataset(
    prime: int,
    op: str,
    train_fraction: float,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    examples: List[Tuple[int, int, int]] = []
    for a in range(prime):
        for b in range(prime):
            if op == "add":
                y = (a + b) % prime
            elif op == "sub":
                y = (a - b) % prime
            elif op == "mul":
                y = (a * b) % prime
            else:
                raise ValueError(f"unknown op: {op}")
            examples.append((a, b, y))

    indices = np.arange(len(examples))
    rng.shuffle(indices)
    train_size = max(1, min(len(indices) - 1, int(round(float(train_fraction) * len(indices)))))
    train_idx = set(int(i) for i in indices[:train_size])

    plus = prime
    eq = prime + 1
    bos = prime + 2
    train_x: List[List[int]] = []
    train_y: List[int] = []
    test_x: List[List[int]] = []
    test_y: List[int] = []

    for idx, (a, b, y) in enumerate(examples):
        row = [bos, a, plus, b, eq]
        if idx in train_idx:
            train_x.append(row)
            train_y.append(y)
        else:
            test_x.append(row)
            test_y.append(y)

    return (
        torch.tensor(train_x, dtype=torch.long, device=device),
        torch.tensor(train_y, dtype=torch.long, device=device),
        torch.tensor(test_x, dtype=torch.long, device=device),
        torch.tensor(test_y, dtype=torch.long, device=device),
    )


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGrokTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        prime: int,
        seq_len: int,
        d_model: int,
        layers: int,
        heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.prime = int(prime)
        self.seq_len = int(seq_len)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(seq_len, d_model))
        self.blocks = nn.ModuleList([TinyBlock(d_model, heads, d_ff, dropout) for _ in range(layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, prime)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(tokens) + self.position_embedding[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x[:, -1, :])
        return self.classifier(x)


@dataclass
class EvalMetrics:
    train_acc: float
    test_acc: float
    train_loss: float
    test_loss: float


@dataclass
class BranchConfig:
    name: str
    svd_rank: int = 0
    random_low_rank: bool = False
    fourier_shape: bool = False
    equivariant_shape: bool = False
    discover_shape: bool = False
    blind_discover_shape: bool = False
    invent_shape: bool = False
    z_guided: bool = False
    spectral_lambda: float = 0.0


@dataclass(frozen=True)
class ShapeCandidate:
    name: str
    slot: str
    output_coeff: int
    score: float = 1.0
    support: int = 0
    input_from: int = -1
    input_to: int = -1
    input_delta: int = -1
    output_delta: int = -1
    output_mapping: Tuple[int, ...] | None = None


def batch_sample(x: torch.Tensor, y: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, x.shape[0], (batch_size,), device=x.device)
    return x[idx], y[idx]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    batch_size: int,
) -> EvalMetrics:
    model.eval()

    def eval_split(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        correct = 0
        loss_sum = 0.0
        count = 0
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction="sum")
            pred = logits.argmax(dim=-1)
            correct += int((pred == yb).sum().item())
            loss_sum += float(loss.item())
            count += int(yb.numel())
        return correct / max(count, 1), loss_sum / max(count, 1)

    train_acc, train_loss = eval_split(train_x, train_y)
    test_acc, test_loss = eval_split(test_x, test_y)
    model.train()
    return EvalMetrics(train_acc=train_acc, test_acc=test_acc, train_loss=train_loss, test_loss=test_loss)


def stable_rank_penalty(model: nn.Module) -> torch.Tensor:
    penalty: torch.Tensor | None = None
    for name, param in model.named_parameters():
        if param.ndim != 2 or "embedding" in name:
            continue
        matrix = param.float()
        fro_sq = matrix.pow(2).sum()
        # One singular-value call per matrix is fine for this tiny audit.
        spectral = torch.linalg.matrix_norm(matrix, ord=2).pow(2).clamp_min(1e-8)
        value = fro_sq / spectral
        penalty = value if penalty is None else penalty + value
    if penalty is None:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    return penalty


def shift_logits(logits: torch.Tensor, delta: torch.Tensor, prime: int) -> torch.Tensor:
    """Shift class distributions by per-example modular deltas.

    If target y maps to y + delta, then the shifted distribution at class c is
    the original distribution at c - delta.
    """
    batch, classes = logits.shape
    if classes != prime:
        raise ValueError(f"expected {prime} classes, got {classes}")
    idx = torch.arange(prime, device=logits.device)[None, :].expand(batch, -1)
    src = (idx - delta[:, None]) % prime
    return logits.gather(1, src)


def make_shifted_inputs(x: torch.Tensor, delta: torch.Tensor, prime: int, which: str) -> torch.Tensor:
    shifted = x.clone()
    if which == "a":
        shifted[:, 1] = (shifted[:, 1] + delta) % prime
    elif which == "b":
        shifted[:, 3] = (shifted[:, 3] + delta) % prime
    else:
        raise ValueError(which)
    return shifted


def manual_shape_candidates(op: str) -> List[ShapeCandidate]:
    if op == "add":
        return [
            ShapeCandidate(name="manual_a_shift_out_+1", slot="a", output_coeff=1),
            ShapeCandidate(name="manual_b_shift_out_+1", slot="b", output_coeff=1),
        ]
    if op == "sub":
        return [
            ShapeCandidate(name="manual_a_shift_out_+1", slot="a", output_coeff=1),
            ShapeCandidate(name="manual_b_shift_out_-1", slot="b", output_coeff=-1),
        ]
    return []


def discover_shape_candidates(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    prime: int,
    coeff_radius: int,
    top_k: int,
    min_score: float,
    min_support: int,
    max_deltas: int,
    seed: int,
) -> List[ShapeCandidate]:
    """Mine modular transport candidates from training labels only.

    The controller is not told that addition means output_coeff=+1. It tries a
    broad family:

        shift slot a or b by d; predict output should shift by c*d

    and keeps candidates that are empirically true on observed training pairs.
    """
    x_cpu = train_x.detach().cpu()
    y_cpu = train_y.detach().cpu()
    label_grid = torch.full((prime, prime), -1, dtype=torch.long)
    for row, target in zip(x_cpu, y_cpu):
        label_grid[int(row[1]), int(row[3])] = int(target)

    rng = random.Random(seed)
    all_deltas = list(range(1, prime))
    if 0 < int(max_deltas) < len(all_deltas):
        deltas = sorted(rng.sample(all_deltas, int(max_deltas)))
    else:
        deltas = all_deltas

    candidates: List[ShapeCandidate] = []
    coeffs = [coeff for coeff in range(-int(coeff_radius), int(coeff_radius) + 1) if coeff != 0]
    for slot in ("a", "b"):
        for coeff in coeffs:
            correct = 0
            support = 0
            for row, target in zip(x_cpu, y_cpu):
                a = int(row[1])
                b = int(row[3])
                y = int(target)
                for delta in deltas:
                    if slot == "a":
                        a2 = (a + delta) % prime
                        b2 = b
                    else:
                        a2 = a
                        b2 = (b + delta) % prime
                    y2 = int(label_grid[a2, b2].item())
                    if y2 < 0:
                        continue
                    support += 1
                    expected = (y + coeff * delta) % prime
                    correct += int(y2 == expected)
            score = correct / max(support, 1)
            if support >= int(min_support) and score >= float(min_score):
                candidates.append(
                    ShapeCandidate(
                        name=f"discover_{slot}_shift_out_{coeff:+d}",
                        slot=slot,
                        output_coeff=int(coeff),
                        score=float(score),
                        support=int(support),
                    )
                )

    candidates.sort(key=lambda item: (float(item.score), int(item.support)), reverse=True)
    return candidates[: max(0, int(top_k))]


def discover_blind_shape_candidates(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    prime: int,
    top_k: int,
    min_score: float,
    min_support: int,
) -> List[ShapeCandidate]:
    """Mine input-output transports without a modular-shift DSL.

    This is the stricter "invent the candidate family" toy proxy. It does not
    enumerate output coefficients. It asks a generic relation question:

        if input slot value u changes to v while the other coordinate is held
        fixed, is there a consistent output relabeling y -> pi(y)?

    The discovered pi may be partial because the train split is sparse. The
    consistency loss only transports the output probability mass on observed
    domain labels, so it does not need us to complete the permutation by hand.
    """
    x_cpu = train_x.detach().cpu()
    y_cpu = train_y.detach().cpu()
    label_grid = torch.full((prime, prime), -1, dtype=torch.long)
    for row, target in zip(x_cpu, y_cpu):
        label_grid[int(row[1]), int(row[3])] = int(target)

    candidates: List[ShapeCandidate] = []
    for slot in ("a", "b"):
        slot_pos = 1 if slot == "a" else 3
        other_pos = 3 if slot == "a" else 1
        for source_value in range(prime):
            for target_value in range(prime):
                if source_value == target_value:
                    continue
                mapping = [-1 for _ in range(prime)]
                support = 0
                conflicts = 0
                for row, target in zip(x_cpu, y_cpu):
                    if int(row[slot_pos]) != source_value:
                        continue
                    other_value = int(row[other_pos])
                    if slot == "a":
                        y2 = int(label_grid[target_value, other_value].item())
                    else:
                        y2 = int(label_grid[other_value, target_value].item())
                    if y2 < 0:
                        continue
                    y1 = int(target)
                    support += 1
                    if mapping[y1] < 0:
                        mapping[y1] = y2
                    elif mapping[y1] != y2:
                        conflicts += 1

                if support < int(min_support):
                    continue
                score = 1.0 - (conflicts / max(support, 1))
                observed_domain = sum(1 for item in mapping if item >= 0)
                # Require a nontrivial output transport, not identity on the
                # observed domain.
                non_identity = sum(1 for idx, item in enumerate(mapping) if item >= 0 and item != idx)
                if score < float(min_score) or observed_domain <= 0 or non_identity <= 0:
                    continue
                candidates.append(
                    ShapeCandidate(
                        name=f"blind_{slot}_{source_value}_to_{target_value}_map{observed_domain}",
                        slot=slot,
                        output_coeff=0,
                        score=float(score),
                        support=int(support),
                        input_from=int(source_value),
                        input_to=int(target_value),
                        output_mapping=tuple(int(item) for item in mapping),
                    )
                )

    candidates.sort(
        key=lambda item: (
            float(item.score),
            int(item.support),
            sum(1 for value in (item.output_mapping or ()) if value >= 0),
        ),
        reverse=True,
    )
    return candidates[: max(0, int(top_k))]


def discover_invented_shape_candidates(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    prime: int,
    top_k: int,
    min_support: int,
) -> List[ShapeCandidate]:
    """Induce reusable transport families from local blind maps.

    This is the strongest "from less structure" toy rung:

    1. Mine local slice transports u -> v from the observed relation.
    2. Infer the output relabeling each local transport induces.
    3. Cluster those local transports by input movement and output movement.
    4. Promote high-support clusters to global consistency laws.

    Unlike `discover_shape_candidates`, this does not try coefficients c in
    output=c*d. It rediscovers the input/output movement pairs from table
    regularities.
    """
    x_cpu = train_x.detach().cpu()
    y_cpu = train_y.detach().cpu()
    label_grid = torch.full((prime, prime), -1, dtype=torch.long)
    for row, target in zip(x_cpu, y_cpu):
        label_grid[int(row[1]), int(row[3])] = int(target)

    cluster_support: Dict[Tuple[str, int, int], int] = {}
    pair_count: Dict[Tuple[str, int, int], int] = {}
    for slot in ("a", "b"):
        slot_pos = 1 if slot == "a" else 3
        other_pos = 3 if slot == "a" else 1
        for source_value in range(prime):
            for target_value in range(prime):
                if source_value == target_value:
                    continue
                support = 0
                output_deltas: List[int] = []
                for row, target in zip(x_cpu, y_cpu):
                    if int(row[slot_pos]) != source_value:
                        continue
                    other_value = int(row[other_pos])
                    if slot == "a":
                        y2 = int(label_grid[target_value, other_value].item())
                    else:
                        y2 = int(label_grid[other_value, target_value].item())
                    if y2 < 0:
                        continue
                    y1 = int(target)
                    support += 1
                    output_deltas.append((y2 - y1) % prime)

                if support <= 0:
                    continue
                # A clean local transport has the same output movement for all
                # observed examples in the slice.
                if len(set(output_deltas)) != 1:
                    continue
                input_delta = (target_value - source_value) % prime
                output_delta = output_deltas[0]
                if input_delta == 0 or output_delta == 0:
                    continue
                key = (slot, int(input_delta), int(output_delta))
                cluster_support[key] = cluster_support.get(key, 0) + int(support)
                pair_count[key] = pair_count.get(key, 0) + 1

    candidates: List[ShapeCandidate] = []
    for (slot, input_delta, output_delta), support in cluster_support.items():
        if support < int(min_support):
            continue
        pairs = int(pair_count.get((slot, input_delta, output_delta), 0))
        candidates.append(
            ShapeCandidate(
                name=f"invent_{slot}_delta_{input_delta}_out_{output_delta}",
                slot=slot,
                output_coeff=0,
                score=1.0,
                support=int(support),
                input_delta=int(input_delta),
                output_delta=int(output_delta),
                # A complete cyclic output transport induced from the local maps.
                output_mapping=tuple(int((value + output_delta) % prime) for value in range(prime)),
                input_from=-1,
                input_to=-1,
            )
        )
        # Store pair count by slightly inflating support ordering through score
        # fields in the CSV/JSON name; no extra dataclass field needed.
        candidates[-1] = ShapeCandidate(
            **{
                **asdict(candidates[-1]),
                "score": 1.0 + min(pairs, prime) * 1e-6,
            }
        )

    candidates.sort(key=lambda item: (int(item.support), float(item.score)), reverse=True)
    return candidates[: max(0, int(top_k))]


def transport_logits_with_mapping(logits: torch.Tensor, mapping: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    map_tensor = torch.tensor(mapping, device=logits.device, dtype=torch.long)
    known = map_tensor >= 0
    if int(known.sum().item()) == 0:
        return logits.new_zeros(logits.shape), logits.new_zeros((logits.shape[0],), dtype=torch.bool)

    base_probs = F.softmax(logits, dim=-1)
    known_idx = torch.nonzero(known, as_tuple=False).squeeze(1)
    target_idx = map_tensor[known_idx]
    transported = logits.new_zeros(logits.shape)
    transported[:, target_idx] += base_probs[:, known_idx]
    mass = transported.sum(dim=-1)
    valid = mass > 1e-7
    transported = transported / mass.clamp_min(1e-7)[:, None]
    return transported, valid


def equivariance_shape_loss(
    model: TinyGrokTransformer,
    x: torch.Tensor,
    logits: torch.Tensor,
    prime: int,
    samples: int,
    temperature: float,
    candidates: Sequence[ShapeCandidate],
) -> Tuple[torch.Tensor, int]:
    """Unlabeled modular-transport gradient shaping.

    This does not reveal heldout labels. It says: if the current model believes
    f(a,b), then f(T(a,b)) should be the same belief transported by R. Manual
    branches supply R directly; discovery branches mine R from the train set.
    """
    if not candidates or samples <= 0:
        return logits.new_zeros(()), 0

    losses: List[torch.Tensor] = []
    queries = 0
    base_logits = logits.detach() / float(temperature)
    for _ in range(int(samples)):
        delta = torch.randint(1, prime, (x.shape[0],), device=x.device)
        for candidate in candidates:
            if candidate.input_delta >= 0 and candidate.output_delta >= 0:
                fixed_delta = torch.full((x.shape[0],), int(candidate.input_delta), device=x.device, dtype=torch.long)
                x_shift = make_shifted_inputs(x, fixed_delta, prime, candidate.slot)
                shifted_logits = model(x_shift) / float(temperature)
                class_delta = torch.full((x.shape[0],), int(candidate.output_delta), device=x.device, dtype=torch.long)
                target_logits = shift_logits(base_logits, class_delta, prime)
                target_probs = F.softmax(target_logits, dim=-1)
                log_probs = F.log_softmax(shifted_logits, dim=-1)
                losses.append(F.kl_div(log_probs, target_probs, reduction="batchmean") * (temperature ** 2))
                queries += int(x.shape[0])
            elif candidate.output_mapping is not None and candidate.input_from >= 0 and candidate.input_to >= 0:
                slot_pos = 1 if candidate.slot == "a" else 3
                mask = x[:, slot_pos] == int(candidate.input_from)
                if int(mask.sum().item()) == 0:
                    continue
                x_shift = x[mask].clone()
                x_shift[:, slot_pos] = int(candidate.input_to)
                shifted_logits = model(x_shift) / float(temperature)
                target_probs, valid = transport_logits_with_mapping(base_logits[mask], candidate.output_mapping)
                if int(valid.sum().item()) == 0:
                    continue
                log_probs = F.log_softmax(shifted_logits[valid], dim=-1)
                target = target_probs[valid]
                kl = (target * ((target + 1e-8).log() - log_probs)).sum(dim=-1).mean()
                losses.append(kl * (temperature ** 2))
                queries += int(x_shift.shape[0])
            else:
                class_delta = (int(candidate.output_coeff) * delta) % prime
                x_shift = make_shifted_inputs(x, delta, prime, candidate.slot)
                shifted_logits = model(x_shift) / float(temperature)
                target_logits = shift_logits(base_logits, class_delta, prime)
                target_probs = F.softmax(target_logits, dim=-1)
                log_probs = F.log_softmax(shifted_logits, dim=-1)
                losses.append(F.kl_div(log_probs, target_probs, reduction="batchmean") * (temperature ** 2))
                queries += int(x.shape[0])
    if not losses:
        return logits.new_zeros(()), 0
    return torch.stack(losses).mean(), queries


def make_fourier_basis(prime: int, modes: int, device: torch.device) -> torch.Tensor:
    # QR is not implemented for MPS in some PyTorch builds. Build this tiny
    # fixed basis on CPU, then move it to the training device.
    xs = torch.arange(prime, device=torch.device("cpu"), dtype=torch.float32)
    cols = [torch.ones_like(xs)]
    for k in range(1, int(modes) + 1):
        angle = 2.0 * math.pi * float(k) * xs / float(prime)
        cols.append(torch.cos(angle))
        cols.append(torch.sin(angle))
    basis = torch.stack(cols, dim=1)
    q, _ = torch.linalg.qr(basis, mode="reduced")
    return q.to(device)


@torch.no_grad()
def project_rows_to_basis(grad: torch.Tensor, basis: torch.Tensor, rows: int) -> None:
    if grad is None or grad.ndim == 0:
        return
    original_dtype = grad.dtype
    g = grad[:rows].float()
    b = basis.to(device=grad.device, dtype=g.dtype)
    projected = b @ (b.transpose(0, 1) @ g)
    grad[:rows].copy_(projected.to(dtype=original_dtype))


@torch.no_grad()
def apply_fourier_gradient_shape(model: TinyGrokTransformer, basis: torch.Tensor) -> int:
    shaped = 0
    if model.token_embedding.weight.grad is not None:
        project_rows_to_basis(model.token_embedding.weight.grad, basis, model.prime)
        shaped += 1
    if model.classifier.weight.grad is not None:
        project_rows_to_basis(model.classifier.weight.grad, basis, model.prime)
        shaped += 1
    if model.classifier.bias is not None and model.classifier.bias.grad is not None:
        g = model.classifier.bias.grad[: model.prime].float().unsqueeze(1)
        b = basis.to(device=g.device, dtype=g.dtype)
        projected = (b @ (b.transpose(0, 1) @ g)).squeeze(1)
        model.classifier.bias.grad[: model.prime].copy_(projected.to(dtype=model.classifier.bias.grad.dtype))
        shaped += 1
    return shaped


def block_for_name(name: str) -> str:
    if name.startswith("blocks."):
        parts = name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            return f"block_{parts[1]}"
    if name.startswith("token_embedding") or name.startswith("position_embedding"):
        return "embedding"
    if name.startswith("classifier") or name.startswith("final_ln"):
        return "head"
    return "other"


@torch.no_grad()
def gradient_block_pressure(model: nn.Module, loss_value: float) -> Dict[str, float]:
    stats: Dict[str, List[float]] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        block = block_for_name(name)
        grad_norm = float(param.grad.detach().float().norm().item())
        weight_norm = float(param.detach().float().norm().item())
        if block not in stats:
            stats[block] = [0.0, 0.0]
        stats[block][0] += grad_norm * grad_norm
        stats[block][1] += weight_norm * weight_norm
    pressure: Dict[str, float] = {}
    for block, (grad_sq, weight_sq) in stats.items():
        grad_norm = math.sqrt(max(grad_sq, 0.0))
        weight_norm = math.sqrt(max(weight_sq, 1e-12))
        # We log impedance-style Z and use it as a pressure sensor. Higher
        # means high loss per available gradient/scale: the block is constrained.
        pressure[block] = abs(float(loss_value)) / (grad_norm * weight_norm + 1e-12)
    return pressure


@torch.no_grad()
def apply_z_guided_scaling(
    model: nn.Module,
    pressure: Dict[str, float],
    top_k: int,
    boost: float,
    suppress: float,
) -> Tuple[str, float]:
    if not pressure:
        return "", 0.0
    ranked = sorted(pressure.items(), key=lambda item: float(item[1]), reverse=True)
    selected = {name for name, _ in ranked[: max(1, int(top_k))]}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        block = block_for_name(name)
        param.grad.mul_(float(boost) if block in selected else float(suppress))
    return ",".join(sorted(selected)), float(ranked[0][1])


@torch.no_grad()
def svd_project_gradient(grad: torch.Tensor, rank: int) -> bool:
    if grad is None or grad.ndim != 2 or rank <= 0:
        return False
    rows, cols = int(grad.shape[0]), int(grad.shape[1])
    if min(rows, cols) <= rank:
        return False
    # CPU fallback keeps the script portable across MPS/CUDA/CPU. The audit
    # matrices are small enough that this is fine; CUDA runs can remove the
    # transfer cost by using devices with full linalg coverage.
    g = grad.detach().float().cpu()
    try:
        u, s, vh = torch.linalg.svd(g, full_matrices=False)
    except RuntimeError:
        return False
    shaped = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
    grad.copy_(shaped.to(device=grad.device, dtype=grad.dtype))
    return True


@torch.no_grad()
def random_low_rank_project_gradient(grad: torch.Tensor, rank: int) -> bool:
    if grad is None or grad.ndim != 2 or rank <= 0:
        return False
    rows, cols = int(grad.shape[0]), int(grad.shape[1])
    if min(rows, cols) <= rank:
        return False
    g = grad.detach().float().cpu()
    u = torch.randn(rows, rank, device=g.device, dtype=g.dtype)
    v = torch.randn(cols, rank, device=g.device, dtype=g.dtype)
    u, _ = torch.linalg.qr(u, mode="reduced")
    v, _ = torch.linalg.qr(v, mode="reduced")
    shaped = u @ (u.transpose(0, 1) @ g @ v) @ v.transpose(0, 1)
    grad.copy_(shaped.to(device=grad.device, dtype=grad.dtype))
    return True


@torch.no_grad()
def apply_low_rank_gradient_shape(
    model: nn.Module,
    rank: int,
    random_projection: bool,
    skip_embeddings: bool,
) -> int:
    shaped = 0
    for name, param in model.named_parameters():
        if param.grad is None or param.grad.ndim != 2:
            continue
        if skip_embeddings and "embedding" in name:
            continue
        if random_projection:
            did_shape = random_low_rank_project_gradient(param.grad, rank)
        else:
            did_shape = svd_project_gradient(param.grad, rank)
        shaped += int(did_shape)
    return shaped


def branch_configs(args: argparse.Namespace) -> List[BranchConfig]:
    requested = [item.strip() for item in str(args.branches).split(",") if item.strip()]
    library = {
        "baseline": BranchConfig("baseline"),
        "random_lowrank": BranchConfig("random_lowrank", svd_rank=args.svd_rank, random_low_rank=True),
        "svd_shape": BranchConfig("svd_shape", svd_rank=args.svd_rank),
        "fourier_shape": BranchConfig("fourier_shape", fourier_shape=True),
        "equivariant_shape": BranchConfig("equivariant_shape", equivariant_shape=True),
        "discover_shape": BranchConfig("discover_shape", discover_shape=True),
        "blind_discover_shape": BranchConfig("blind_discover_shape", blind_discover_shape=True),
        "invent_shape": BranchConfig("invent_shape", invent_shape=True),
        "z_svd_shape": BranchConfig("z_svd_shape", svd_rank=args.svd_rank, z_guided=True),
        "z_fourier_svd": BranchConfig(
            "z_fourier_svd",
            svd_rank=args.svd_rank,
            fourier_shape=True,
            z_guided=True,
            spectral_lambda=args.spectral_lambda,
        ),
        "z_equivariant_svd": BranchConfig(
            "z_equivariant_svd",
            svd_rank=args.svd_rank,
            equivariant_shape=True,
            z_guided=True,
            spectral_lambda=args.spectral_lambda,
        ),
        "z_discover_svd": BranchConfig(
            "z_discover_svd",
            svd_rank=args.svd_rank,
            discover_shape=True,
            z_guided=True,
            spectral_lambda=args.spectral_lambda,
        ),
        "z_blind_discover_svd": BranchConfig(
            "z_blind_discover_svd",
            svd_rank=args.svd_rank,
            blind_discover_shape=True,
            z_guided=True,
            spectral_lambda=args.spectral_lambda,
        ),
        "z_invent_svd": BranchConfig(
            "z_invent_svd",
            svd_rank=args.svd_rank,
            invent_shape=True,
            z_guided=True,
            spectral_lambda=args.spectral_lambda,
        ),
    }
    unknown = [name for name in requested if name not in library]
    if unknown:
        raise ValueError(f"unknown branches: {unknown}; available={sorted(library)}")
    return [library[name] for name in requested]


def train_branch(
    args: argparse.Namespace,
    seed: int,
    cfg: BranchConfig,
    init_state: Dict[str, torch.Tensor],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    fourier_basis: torch.Tensor,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    model = TinyGrokTransformer(
        vocab_size=args.prime + 3,
        prime=args.prime,
        seq_len=5,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(deepcopy(init_state))
    model.train()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(0.9, 0.98),
        weight_decay=float(args.weight_decay),
    )

    shape_candidates: List[ShapeCandidate] = []
    if cfg.equivariant_shape:
        shape_candidates = manual_shape_candidates(str(args.op))
    if cfg.discover_shape:
        shape_candidates = discover_shape_candidates(
            train_x=train_x,
            train_y=train_y,
            prime=int(args.prime),
            coeff_radius=int(args.discover_coeff_radius),
            top_k=int(args.discover_top_k),
            min_score=float(args.discover_min_score),
            min_support=int(args.discover_min_support),
            max_deltas=int(args.discover_max_deltas),
            seed=seed + 910_003,
        )
        if not shape_candidates:
            print(f"[{cfg.name}] discovered_shapes=NONE", flush=True)
        else:
            shape_text = "; ".join(
                f"{item.name}(score={item.score:.3f},support={item.support})" for item in shape_candidates
            )
            print(f"[{cfg.name}] discovered_shapes={shape_text}", flush=True)
    if cfg.blind_discover_shape:
        shape_candidates = discover_blind_shape_candidates(
            train_x=train_x,
            train_y=train_y,
            prime=int(args.prime),
            top_k=int(args.blind_top_k),
            min_score=float(args.blind_min_score),
            min_support=int(args.blind_min_support),
        )
        if not shape_candidates:
            print(f"[{cfg.name}] blind_shapes=NONE", flush=True)
        else:
            shape_text = "; ".join(
                f"{item.name}(score={item.score:.3f},support={item.support})" for item in shape_candidates
            )
            print(f"[{cfg.name}] blind_shapes={shape_text}", flush=True)
    if cfg.invent_shape:
        shape_candidates = discover_invented_shape_candidates(
            train_x=train_x,
            train_y=train_y,
            prime=int(args.prime),
            top_k=int(args.invent_top_k),
            min_support=int(args.invent_min_support),
        )
        if not shape_candidates:
            print(f"[{cfg.name}] invented_shapes=NONE", flush=True)
        else:
            shape_text = "; ".join(
                f"{item.name}(score={item.score:.3f},support={item.support})" for item in shape_candidates
            )
            print(f"[{cfg.name}] invented_shapes={shape_text}", flush=True)

    rows: List[Dict[str, object]] = []
    first_train_step: int | None = None
    first_test_step: int | None = None
    first_grok_step: int | None = None
    best_test_acc = 0.0
    best_train_acc = 0.0
    last_selected_blocks = ""
    last_top_z = 0.0
    shaped_matrix_count = 0
    fourier_shape_count = 0
    unlabeled_symmetry_queries = 0
    start_time = time.time()

    for step in range(1, int(args.steps) + 1):
        xb, yb = batch_sample(train_x, train_y, int(args.batch_size))
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        if cfg.equivariant_shape or cfg.discover_shape or cfg.blind_discover_shape or cfg.invent_shape:
            eq_loss, eq_queries = equivariance_shape_loss(
                model,
                xb,
                logits,
                prime=int(args.prime),
                samples=int(args.equivariance_samples),
                temperature=float(args.equivariance_temperature),
                candidates=shape_candidates,
            )
            loss = loss + float(args.equivariance_weight) * eq_loss
            unlabeled_symmetry_queries += int(eq_queries)
        if cfg.spectral_lambda > 0.0:
            loss = loss + float(cfg.spectral_lambda) * stable_rank_penalty(model)
        loss.backward()

        loss_value = float(loss.detach().item())
        pressure = gradient_block_pressure(model, loss_value)
        if cfg.z_guided:
            last_selected_blocks, last_top_z = apply_z_guided_scaling(
                model,
                pressure,
                top_k=int(args.z_top_k),
                boost=float(args.z_boost),
                suppress=float(args.z_suppress),
            )

        if cfg.fourier_shape:
            fourier_shape_count += apply_fourier_gradient_shape(model, fourier_basis)

        if cfg.svd_rank > 0:
            shaped_matrix_count += apply_low_rank_gradient_shape(
                model,
                rank=int(cfg.svd_rank),
                random_projection=bool(cfg.random_low_rank),
                skip_embeddings=bool(args.svd_skip_embeddings),
            )

        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        opt.step()

        should_eval = step == 1 or step % int(args.eval_interval) == 0 or step == int(args.steps)
        if not should_eval:
            continue

        metrics = evaluate(model, train_x, train_y, test_x, test_y, int(args.eval_batch_size))
        best_test_acc = max(best_test_acc, metrics.test_acc)
        best_train_acc = max(best_train_acc, metrics.train_acc)
        if first_train_step is None and metrics.train_acc >= float(args.train_threshold):
            first_train_step = step
        if first_test_step is None and metrics.test_acc >= float(args.test_threshold):
            first_test_step = step
        if (
            first_grok_step is None
            and metrics.train_acc >= float(args.train_threshold)
            and metrics.test_acc >= float(args.test_threshold)
        ):
            first_grok_step = step

        row = {
            "seed": seed,
            "branch": cfg.name,
            "step": step,
            "train_acc": metrics.train_acc,
            "test_acc": metrics.test_acc,
            "train_loss": metrics.train_loss,
            "test_loss": metrics.test_loss,
            "loss": loss_value,
            "selected_blocks": last_selected_blocks,
            "top_z": last_top_z,
            "shaped_matrix_count": shaped_matrix_count,
            "fourier_shape_count": fourier_shape_count,
            "unlabeled_symmetry_queries": unlabeled_symmetry_queries,
            "shape_candidates": "|".join(item.name for item in shape_candidates),
            "elapsed_sec": time.time() - start_time,
        }
        rows.append(row)
        if step == 1 or step % int(args.log_interval) == 0 or first_grok_step == step or step == int(args.steps):
            print(
                f"[{cfg.name}] step={step:05d}/{int(args.steps)} "
                f"train={metrics.train_acc:.3f} test={metrics.test_acc:.3f} "
                f"loss={metrics.train_loss:.3f}/{metrics.test_loss:.3f} "
                f"grok={format_step(first_grok_step)} z={last_top_z:.3e}",
                flush=True,
            )

        if bool(args.stop_on_grok) and first_grok_step is not None:
            break

    final_metrics = evaluate(model, train_x, train_y, test_x, test_y, int(args.eval_batch_size))
    summary = {
        "seed": seed,
        "branch": cfg.name,
        "first_train_step": first_train_step,
        "first_test_step": first_test_step,
        "first_grok_step": first_grok_step,
        "best_train_acc": best_train_acc,
        "best_test_acc": best_test_acc,
        "final_train_acc": final_metrics.train_acc,
        "final_test_acc": final_metrics.test_acc,
        "final_train_loss": final_metrics.train_loss,
        "final_test_loss": final_metrics.test_loss,
        "shaped_matrix_count": shaped_matrix_count,
        "fourier_shape_count": fourier_shape_count,
        "unlabeled_symmetry_queries": unlabeled_symmetry_queries,
        "uses_svd": int(cfg.svd_rank > 0),
        "uses_random_lowrank": int(cfg.random_low_rank),
        "uses_fourier_shape": int(cfg.fourier_shape),
        "uses_equivariant_shape": int(cfg.equivariant_shape),
        "uses_discover_shape": int(cfg.discover_shape),
        "uses_blind_discover_shape": int(cfg.blind_discover_shape),
        "uses_invent_shape": int(cfg.invent_shape),
        "uses_z_guided": int(cfg.z_guided),
        "spectral_lambda": cfg.spectral_lambda,
        "shape_candidates": "|".join(item.name for item in shape_candidates),
        "shape_candidate_scores": json.dumps([asdict(item) for item in shape_candidates], sort_keys=True),
        "wall_time_sec": time.time() - start_time,
    }
    print(
        f"[summary] seed={seed} branch={cfg.name} "
        f"grok={format_step(first_grok_step)} best_test={best_test_acc:.3f} "
        f"final_test={final_metrics.test_acc:.3f}",
        flush=True,
    )
    return summary, rows


def run_seed(args: argparse.Namespace, seed: int, device: torch.device) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    print("=" * 96)
    print(f"ACCELERATED GROKKING AUDIT seed={seed}")
    print("=" * 96)
    set_seed(seed)

    train_x, train_y, test_x, test_y = make_modular_dataset(
        prime=int(args.prime),
        op=str(args.op),
        train_fraction=float(args.train_fraction),
        seed=seed,
        device=device,
    )
    print(
        f"task=({args.op} mod {args.prime}) train={train_x.shape[0]} "
        f"test={test_x.shape[0]} train_fraction={args.train_fraction}",
        flush=True,
    )

    init_model = TinyGrokTransformer(
        vocab_size=args.prime + 3,
        prime=args.prime,
        seq_len=5,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)
    init_state = deepcopy(init_model.state_dict())
    fourier_basis = make_fourier_basis(int(args.prime), int(args.fourier_modes), device)

    summaries: List[Dict[str, object]] = []
    curves: List[Dict[str, object]] = []
    for cfg in branch_configs(args):
        print("-" * 96)
        print(f"Branch: {cfg.name}")
        print("-" * 96)
        summary, rows = train_branch(
            args,
            seed,
            cfg,
            init_state,
            train_x,
            train_y,
            test_x,
            test_y,
            fourier_basis,
            device,
        )
        summaries.append(summary)
        curves.extend(rows)
    return summaries, curves


def summarize_verdict(args: argparse.Namespace, summaries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_branch: Dict[str, List[Dict[str, object]]] = {}
    for row in summaries:
        by_branch.setdefault(str(row["branch"]), []).append(dict(row))

    def grok_steps(branch: str) -> List[int | None]:
        return [None if row.get("first_grok_step") in ("", None) else int(row["first_grok_step"]) for row in by_branch.get(branch, [])]

    def mean_reached(branch: str) -> float:
        vals = grok_steps(branch)
        return float(np.mean([v is not None for v in vals])) if vals else 0.0

    def mean_step(branch: str) -> float | None:
        vals = [v for v in grok_steps(branch) if v is not None]
        if not vals:
            return None
        return float(np.mean(vals))

    def mean_best(branch: str) -> float:
        vals = [float(row.get("best_test_acc", 0.0)) for row in by_branch.get(branch, [])]
        return float(np.mean(vals)) if vals else 0.0

    baseline_step = mean_step("baseline")
    baseline_reached = mean_reached("baseline")
    baseline_best = mean_best("baseline")
    random_step = mean_step("random_lowrank")
    random_reached = mean_reached("random_lowrank")
    random_best = mean_best("random_lowrank")

    shaped_candidates = [
        branch
        for branch in by_branch
        if branch not in {"baseline", "random_lowrank"}
    ]
    candidate_stats: List[Dict[str, object]] = []
    for branch in shaped_candidates:
        step = mean_step(branch)
        reached = mean_reached(branch)
        best = mean_best(branch)
        if baseline_step is None and step is not None:
            speedup = float("inf")
        elif baseline_step is not None and step is not None:
            speedup = float(baseline_step / max(step, 1.0))
        else:
            speedup = 0.0
        candidate_stats.append(
            {
                "branch": branch,
                "mean_grok_step": step,
                "reached_fraction": reached,
                "mean_best_test_acc": best,
                "speedup_vs_baseline": speedup,
            }
        )

    candidate_stats.sort(
        key=lambda row: (
            float(row["reached_fraction"]),
            float(row["speedup_vs_baseline"]),
            float(row["mean_best_test_acc"]),
        ),
        reverse=True,
    )
    best = candidate_stats[0] if candidate_stats else {}
    best_step = best.get("mean_grok_step")
    best_reached = float(best.get("reached_fraction", 0.0) or 0.0)
    best_speedup = float(best.get("speedup_vs_baseline", 0.0) or 0.0)

    random_control_ok = True
    if random_step is not None and best_step is not None:
        random_control_ok = float(best_step) <= float(random_step) / max(float(args.min_speedup_vs_random), 1e-9)
    elif random_reached > 0 and best_reached <= random_reached:
        random_control_ok = False

    if baseline_step is None and best_step is not None:
        acceleration_pass = True
    else:
        acceleration_pass = bool(best_step is not None and best_speedup >= float(args.min_speedup))

    pass_status = bool(best_reached >= 1.0 and acceleration_pass and random_control_ok)
    status = "PASS" if pass_status else "FAIL"
    if best_reached > 0 and acceleration_pass and not random_control_ok:
        status = "PARTIAL_RANDOM_CONTROL"
    elif best_reached > 0 and not acceleration_pass:
        status = "PARTIAL_REACHED_NO_SPEEDUP"

    verdict = {
        "status": status,
        "passed": pass_status,
        "claim": "gradient shaping accelerates heldout generalization under matched data/init/optimizer budget",
        "task": f"{args.op}_mod_{args.prime}",
        "train_fraction": float(args.train_fraction),
        "train_threshold": float(args.train_threshold),
        "test_threshold": float(args.test_threshold),
        "baseline_reached_fraction": baseline_reached,
        "baseline_mean_grok_step": baseline_step,
        "baseline_mean_best_test_acc": baseline_best,
        "random_lowrank_reached_fraction": random_reached,
        "random_lowrank_mean_grok_step": random_step,
        "random_lowrank_mean_best_test_acc": random_best,
        "best_shaped_branch": best.get("branch", ""),
        "best_shaped_reached_fraction": best_reached,
        "best_shaped_mean_grok_step": best_step,
        "best_shaped_mean_best_test_acc": best.get("mean_best_test_acc", 0.0),
        "best_speedup_vs_baseline": best_speedup,
        "random_control_ok": random_control_ok,
        "teacher_labels_used": 0,
        "heldout_labels_used_for_training": 0,
        "extra_labeled_examples_used": 0,
        "unlabeled_symmetry_queries_used": int(sum(int(row.get("unlabeled_symmetry_queries", 0)) for row in summaries)),
        "candidate_stats": candidate_stats,
    }
    if pass_status:
        verdict["interpretation"] = (
            "PASS: a shaped-gradient branch reached heldout generalization faster than "
            "the matched baseline, without heldout labels or extra labeled examples."
        )
    elif status == "PARTIAL_RANDOM_CONTROL":
        verdict["interpretation"] = (
            "PARTIAL: a shaped branch accelerated vs baseline, but the random low-rank "
            "control was too competitive."
        )
    elif status == "PARTIAL_REACHED_NO_SPEEDUP":
        verdict["interpretation"] = (
            "PARTIAL: a shaped branch generalized, but not faster than the baseline under "
            "the requested speedup gate."
        )
    else:
        verdict["interpretation"] = (
            "FAIL: no shaped branch cleared the heldout-generalization acceleration gate."
        )
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="1337")
    parser.add_argument("--output-dir", default="outputs/accelerated_grokking")

    parser.add_argument("--prime", type=int, default=47)
    parser.add_argument("--op", choices=["add", "sub", "mul"], default="add")
    parser.add_argument("--train-fraction", type=float, default=0.35)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--stop-on-grok", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--branches",
        default="baseline,random_lowrank,svd_shape,fourier_shape,equivariant_shape,discover_shape,blind_discover_shape,invent_shape,z_equivariant_svd,z_discover_svd,z_blind_discover_svd,z_invent_svd",
        help="Comma-separated branch names. Available: baseline, random_lowrank, svd_shape, fourier_shape, equivariant_shape, discover_shape, blind_discover_shape, invent_shape, z_svd_shape, z_fourier_svd, z_equivariant_svd, z_discover_svd, z_blind_discover_svd, z_invent_svd.",
    )
    parser.add_argument("--svd-rank", type=int, default=8)
    parser.add_argument("--svd-skip-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fourier-modes", type=int, default=6)
    parser.add_argument("--z-top-k", type=int, default=2)
    parser.add_argument("--z-boost", type=float, default=1.5)
    parser.add_argument("--z-suppress", type=float, default=0.75)
    parser.add_argument("--spectral-lambda", type=float, default=0.0)
    parser.add_argument("--equivariance-weight", type=float, default=2.0)
    parser.add_argument("--equivariance-samples", type=int, default=1)
    parser.add_argument("--equivariance-temperature", type=float, default=1.0)
    parser.add_argument("--discover-coeff-radius", type=int, default=3)
    parser.add_argument("--discover-top-k", type=int, default=2)
    parser.add_argument("--discover-min-score", type=float, default=0.98)
    parser.add_argument("--discover-min-support", type=int, default=32)
    parser.add_argument("--discover-max-deltas", type=int, default=0)
    parser.add_argument("--blind-top-k", type=int, default=32)
    parser.add_argument("--blind-min-score", type=float, default=1.0)
    parser.add_argument("--blind-min-support", type=int, default=4)
    parser.add_argument("--invent-top-k", type=int, default=4)
    parser.add_argument("--invent-min-support", type=int, default=64)

    parser.add_argument("--train-threshold", type=float, default=0.99)
    parser.add_argument("--test-threshold", type=float, default=0.95)
    parser.add_argument("--min-speedup", type=float, default=1.5)
    parser.add_argument("--min-speedup-vs-random", type=float, default=1.1)
    parser.add_argument("--smoke", action="store_true", help="Fast compile/sanity run; not meant to prove the claim.")
    args = parser.parse_args()

    if args.smoke:
        args.prime = 17
        args.steps = 40
        args.eval_interval = 10
        args.log_interval = 10
        args.d_model = 48
        args.d_ff = 128
        args.layers = 1
        args.heads = 4
        args.batch_size = 64
        args.eval_batch_size = 512
        args.branches = "baseline,svd_shape,invent_shape"
        args.blind_min_support = 2
        args.invent_min_support = 8
        args.output_dir = str(Path(args.output_dir) / "smoke")

    device = resolve_device(str(args.device))
    seeds = parse_seeds(str(args.seeds))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("ACCELERATED GROKKING / GRADIENT SHAPING AUDIT")
    print("=" * 96)
    print(f"device={device} seeds={seeds}")
    print(f"task=({args.op} mod {args.prime}) train_fraction={args.train_fraction}")
    print(f"branches={args.branches}")
    print("claim: shaped gradients should reduce steps-to-heldout-generalization.")
    print("teacher_labels_used=0 heldout_labels_used_for_training=0 extra_labeled_examples_used=0")

    config_path = output_dir / "accelerated_grokking_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump({**vars(args), "device_resolved": str(device)}, handle, indent=2, sort_keys=True)

    all_summaries: List[Dict[str, object]] = []
    all_curves: List[Dict[str, object]] = []
    start = time.time()
    for seed in seeds:
        summaries, curves = run_seed(args, seed, device)
        all_summaries.extend(summaries)
        all_curves.extend(curves)

    verdict = summarize_verdict(args, all_summaries)
    verdict["wall_time_sec"] = time.time() - start

    write_csv(output_dir / "accelerated_grokking_summary.csv", all_summaries)
    write_csv(output_dir / "accelerated_grokking_curves.csv", all_curves)
    with (output_dir / "accelerated_grokking_verdict.json").open("w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2, sort_keys=True)

    print("=" * 96)
    print("ACCELERATED GROKKING VERDICT")
    print("=" * 96)
    for row in all_summaries:
        print(
            f"{row['branch']:<18} seed={row['seed']} "
            f"grok={format_step(row.get('first_grok_step')):<8} "
            f"best_test={float(row['best_test_acc']):.3f} "
            f"final_test={float(row['final_test_acc']):.3f}",
            flush=True,
        )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"artifacts={output_dir}")


if __name__ == "__main__":
    main()
