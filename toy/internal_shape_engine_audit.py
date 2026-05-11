#!/usr/bin/env python3
"""Internal shape-engine audit.

Level-4 toy target:

    A neural learner observes support examples, induces reusable input-output
    transports internally, and applies those transports to answer novel queries
    in one forward pass. No external relation miner runs at test time.

This is not "abstractions from metaphysical nothing." The model is given a
finite-symbol task interface and meta-training over many tasks. The question is
whether the learned model can internalize the pattern-discovery/apply loop.

Task family:

    y = c0 + c1*a + c2*b mod p

At meta-test, coefficient triples are held out. The model gets support examples
from a new task and must answer heldout query points. The internal_shape branch
learns to transport support labels through query/support deltas; query_only and
context_mlp are matched controls.

Artifacts:
  internal_shape_engine_curves.csv
  internal_shape_engine_summary.csv
  internal_shape_engine_verdict.json
  internal_shape_engine_config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


@dataclass(frozen=True)
class TaskCoeff:
    c0: int
    c1: int
    c2: int


def make_task_splits(prime: int, train_fraction: float, seed: int) -> Tuple[List[TaskCoeff], List[TaskCoeff]]:
    coeffs = [
        TaskCoeff(c0=c0, c1=c1, c2=c2)
        for c0 in range(prime)
        for c1 in range(1, prime)
        for c2 in range(1, prime)
    ]
    rng = random.Random(seed)
    rng.shuffle(coeffs)
    train_count = max(1, min(len(coeffs) - 1, int(round(float(train_fraction) * len(coeffs)))))
    return coeffs[:train_count], coeffs[train_count:]


def sample_episode(
    coeffs: Sequence[TaskCoeff],
    prime: int,
    task_batch: int,
    support_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    task_ids = torch.randint(0, len(coeffs), (task_batch,))
    c0 = torch.tensor([coeffs[int(i)].c0 for i in task_ids], dtype=torch.long, device=device)
    c1 = torch.tensor([coeffs[int(i)].c1 for i in task_ids], dtype=torch.long, device=device)
    c2 = torch.tensor([coeffs[int(i)].c2 for i in task_ids], dtype=torch.long, device=device)

    support_a = torch.randint(0, prime, (task_batch, support_size), device=device)
    support_b = torch.randint(0, prime, (task_batch, support_size), device=device)
    support_y = (c0[:, None] + c1[:, None] * support_a + c2[:, None] * support_b) % prime

    query_a = torch.randint(0, prime, (task_batch,), device=device)
    query_b = torch.randint(0, prime, (task_batch,), device=device)
    query_y = (c0 + c1 * query_a + c2 * query_b) % prime
    return {
        "support_a": support_a,
        "support_b": support_b,
        "support_y": support_y,
        "query_a": query_a,
        "query_b": query_b,
        "query_y": query_y,
        "c0": c0,
        "c1": c1,
        "c2": c2,
    }


class QueryOnly(nn.Module):
    def __init__(self, prime: int, d_model: int) -> None:
        super().__init__()
        self.prime = int(prime)
        self.value_emb = nn.Embedding(prime, d_model)
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, prime),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        qa = self.value_emb(batch["query_a"])
        qb = self.value_emb(batch["query_b"])
        return self.net(torch.cat([qa, qb], dim=-1))


class ContextMLP(nn.Module):
    def __init__(self, prime: int, d_model: int) -> None:
        super().__init__()
        self.prime = int(prime)
        self.value_emb = nn.Embedding(prime, d_model)
        self.pair_net = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.out = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, prime),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        sa = self.value_emb(batch["support_a"])
        sb = self.value_emb(batch["support_b"])
        sy = self.value_emb(batch["support_y"])
        support = self.pair_net(torch.cat([sa, sb, sy], dim=-1)).mean(dim=1)
        qa = self.value_emb(batch["query_a"])
        qb = self.value_emb(batch["query_b"])
        return self.out(torch.cat([support, qa, qb], dim=-1))


class InternalShapeEngine(nn.Module):
    """Neural transport voter.

    For each support example, infer an output movement from query-support input
    deltas, then transport the support label by that movement. The final answer
    is the aggregate vote across support examples. The transport function is
    learned; no Python relation miner is invoked in forward().
    """

    def __init__(self, prime: int, d_model: int) -> None:
        super().__init__()
        self.prime = int(prime)
        self.value_emb = nn.Embedding(prime, d_model)
        self.delta_emb = nn.Embedding(prime, d_model)
        self.support_net = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.relation_net = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.context_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.transport_net = nn.Sequential(
            nn.Linear(6 * d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, prime),
        )
        self.direct_head = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, prime),
        )
        self.direct_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        p = self.prime
        sa = self.value_emb(batch["support_a"])
        sb = self.value_emb(batch["support_b"])
        sy = self.value_emb(batch["support_y"])
        support = self.support_net(torch.cat([sa, sb, sy], dim=-1))

        # Internal pattern observation: compare support examples to support
        # examples and encode the observed relation deltas. This is the neural
        # version of "notice transports"; no external miner or candidate list is
        # called at inference time.
        da_rel = (batch["support_a"][:, :, None] - batch["support_a"][:, None, :]) % p
        db_rel = (batch["support_b"][:, :, None] - batch["support_b"][:, None, :]) % p
        dy_rel = (batch["support_y"][:, :, None] - batch["support_y"][:, None, :]) % p
        rel = self.relation_net(
            torch.cat(
                [
                    self.delta_emb(da_rel),
                    self.delta_emb(db_rel),
                    self.delta_emb(dy_rel),
                ],
                dim=-1,
            )
        ).mean(dim=(1, 2))
        context = self.context_net(torch.cat([support.mean(dim=1), rel], dim=-1))

        query_a = batch["query_a"][:, None]
        query_b = batch["query_b"][:, None]
        delta_a = (query_a - batch["support_a"]) % p
        delta_b = (query_b - batch["support_b"]) % p
        da = self.delta_emb(delta_a)
        db = self.delta_emb(delta_b)
        qa = self.value_emb(batch["query_a"])[:, None, :].expand_as(support)
        qb = self.value_emb(batch["query_b"])[:, None, :].expand_as(support)
        ctx = context[:, None, :].expand_as(support)

        features = torch.cat([support, ctx, da, db, qa, qb], dim=-1)
        dy_logits = self.transport_net(features)

        classes = torch.arange(p, device=dy_logits.device)
        answer_index = (classes[None, None, :] - batch["support_y"][:, :, None]) % p
        transported_logits = dy_logits.gather(-1, answer_index)
        vote_logits = torch.logsumexp(transported_logits, dim=1) - math.log(batch["support_y"].shape[1])

        direct = self.direct_head(torch.cat([context, self.value_emb(batch["query_a"]), self.value_emb(batch["query_b"])], dim=-1))
        gate = torch.sigmoid(self.direct_gate)
        return vote_logits * (1.0 - gate) + direct * gate


class InternalTransportTable(nn.Module):
    """One-forward internal relation table.

    This branch does not learn a relation miner. It performs the pattern
    observation as an internal tensor operation: support-support deltas induce a
    table (delta_a, delta_b) -> delta_y, and query-support deltas look up that
    transport. It is the cleanest toy analogue of "observe patterns and apply
    them internally" without an external Python miner at test time.
    """

    def __init__(self, prime: int) -> None:
        super().__init__()
        self.prime = int(prime)
        self.logit_scale = nn.Parameter(torch.tensor(6.0))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        p = self.prime
        support_a = batch["support_a"]
        support_b = batch["support_b"]
        support_y = batch["support_y"]
        rel_da = (support_a[:, :, None] - support_a[:, None, :]) % p
        rel_db = (support_b[:, :, None] - support_b[:, None, :]) % p
        rel_dy = (support_y[:, :, None] - support_y[:, None, :]) % p

        query_da = (batch["query_a"][:, None] - support_a) % p
        query_db = (batch["query_b"][:, None] - support_b) % p

        mask = (rel_da[:, None, :, :] == query_da[:, :, None, None]) & (
            rel_db[:, None, :, :] == query_db[:, :, None, None]
        )
        dy_onehot = F.one_hot(rel_dy, p).float()
        counts = (mask[..., None].float() * dy_onehot[:, None, :, :, :]).sum(dim=(2, 3))
        # If a specific delta was not observed, fall back to a weak uniform
        # transport rather than crashing. With enough support this rarely fires.
        dy_logits = torch.log(counts + 1e-4) * self.logit_scale.clamp(0.1, 20.0)

        classes = torch.arange(p, device=dy_logits.device)
        answer_index = (classes[None, None, :] - support_y[:, :, None]) % p
        transported_logits = dy_logits.gather(-1, answer_index)
        return torch.logsumexp(transported_logits, dim=1) - math.log(support_y.shape[1])


def make_model(branch: str, prime: int, d_model: int) -> nn.Module:
    if branch == "query_only":
        return QueryOnly(prime, d_model)
    if branch == "context_mlp":
        return ContextMLP(prime, d_model)
    if branch == "internal_shape":
        return InternalShapeEngine(prime, d_model)
    if branch == "internal_table_shape":
        return InternalTransportTable(prime)
    raise ValueError(f"unknown branch: {branch}")


def branch_kind(branch: str) -> str:
    if branch == "internal_shape":
        return "learned_internal_shape_engine"
    if branch == "internal_table_shape":
        return "oracle_programmatic_internal_transport"
    if branch in {"query_only", "context_mlp"}:
        return "learned_control"
    return "unknown"


@torch.no_grad()
def evaluate(
    model: nn.Module,
    coeffs: Sequence[TaskCoeff],
    prime: int,
    task_batch: int,
    support_size: int,
    batches: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for _ in range(int(batches)):
        batch = sample_episode(coeffs, prime, task_batch, support_size, device)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch["query_y"], reduction="sum")
        pred = logits.argmax(dim=-1)
        correct += int((pred == batch["query_y"]).sum().item())
        total += int(batch["query_y"].numel())
        loss_sum += float(loss.item())
    model.train()
    return correct / max(total, 1), loss_sum / max(total, 1)


def train_branch(
    args: argparse.Namespace,
    seed: int,
    branch: str,
    train_coeffs: Sequence[TaskCoeff],
    test_coeffs: Sequence[TaskCoeff],
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    set_seed(seed + hash(branch) % 1_000_000)
    model = make_model(branch, int(args.prime), int(args.d_model)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    rows: List[Dict[str, object]] = []
    first_pass_step: int | None = None
    best_test_acc = 0.0
    start = time.time()
    for step in range(1, int(args.steps) + 1):
        batch = sample_episode(train_coeffs, int(args.prime), int(args.task_batch), int(args.support_size), device)
        opt.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch["query_y"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        opt.step()

        if step == 1 or step % int(args.eval_interval) == 0 or step == int(args.steps):
            train_acc, train_loss = evaluate(
                model,
                train_coeffs,
                int(args.prime),
                int(args.eval_task_batch),
                int(args.support_size),
                int(args.eval_batches),
                device,
            )
            test_acc, test_loss = evaluate(
                model,
                test_coeffs,
                int(args.prime),
                int(args.eval_task_batch),
                int(args.support_size),
                int(args.eval_batches),
                device,
            )
            best_test_acc = max(best_test_acc, test_acc)
            if first_pass_step is None and test_acc >= float(args.test_threshold):
                first_pass_step = step
            rows.append(
                {
                    "seed": seed,
                    "branch": branch,
                    "step": step,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "loss": float(loss.item()),
                    "elapsed_sec": time.time() - start,
                }
            )
            print(
                f"[{branch}] step={step:05d}/{int(args.steps)} "
                f"train={train_acc:.3f} test={test_acc:.3f} "
                f"loss={train_loss:.3f}/{test_loss:.3f} "
                f"pass={'never' if first_pass_step is None else first_pass_step}",
                flush=True,
            )

    final_train_acc, final_train_loss = evaluate(
        model,
        train_coeffs,
        int(args.prime),
        int(args.eval_task_batch),
        int(args.support_size),
        int(args.eval_batches),
        device,
    )
    final_test_acc, final_test_loss = evaluate(
        model,
        test_coeffs,
        int(args.prime),
        int(args.eval_task_batch),
        int(args.support_size),
        int(args.eval_batches),
        device,
    )
    summary = {
        "seed": seed,
        "branch": branch,
        "branch_kind": branch_kind(branch),
        "allowed_to_pass_learned_claim": int(branch == "internal_shape"),
        "first_pass_step": first_pass_step,
        "best_test_acc": best_test_acc,
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
        "final_train_loss": final_train_loss,
        "final_test_loss": final_test_loss,
        "wall_time_sec": time.time() - start,
    }
    print(
        f"[summary] seed={seed} branch={branch} "
        f"pass={'never' if first_pass_step is None else first_pass_step} "
        f"best_test={best_test_acc:.3f} final_test={final_test_acc:.3f}",
        flush=True,
    )
    return summary, rows


def run_seed(args: argparse.Namespace, seed: int, device: torch.device) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    print("=" * 96)
    print(f"INTERNAL SHAPE ENGINE AUDIT seed={seed}")
    print("=" * 96)
    train_coeffs, test_coeffs = make_task_splits(int(args.prime), float(args.meta_train_fraction), seed)
    print(
        f"task_family=affine_mod_{args.prime} train_tasks={len(train_coeffs)} "
        f"heldout_tasks={len(test_coeffs)} support={args.support_size}",
        flush=True,
    )
    summaries: List[Dict[str, object]] = []
    curves: List[Dict[str, object]] = []
    for branch in [item.strip() for item in str(args.branches).split(",") if item.strip()]:
        print("-" * 96)
        print(f"Branch: {branch}")
        print("-" * 96)
        summary, rows = train_branch(args, seed, branch, train_coeffs, test_coeffs, device)
        summaries.append(summary)
        curves.extend(rows)
    return summaries, curves


def verdict_from(args: argparse.Namespace, summaries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_branch: Dict[str, List[Dict[str, object]]] = {}
    for row in summaries:
        by_branch.setdefault(str(row["branch"]), []).append(dict(row))

    def mean_best(branch: str) -> float:
        vals = [float(row.get("best_test_acc", 0.0)) for row in by_branch.get(branch, [])]
        return float(np.mean(vals)) if vals else 0.0

    def mean_pass(branch: str) -> float | None:
        vals = [row.get("first_pass_step") for row in by_branch.get(branch, []) if row.get("first_pass_step") is not None]
        return float(np.mean([int(v) for v in vals])) if vals else None

    learned_internal_best = mean_best("internal_shape")
    oracle_table_best = mean_best("internal_table_shape")
    query_best = mean_best("query_only")
    context_best = mean_best("context_mlp")
    learned_internal_step = mean_pass("internal_shape")
    oracle_table_step = mean_pass("internal_table_shape")
    pass_status = bool(
        learned_internal_best >= float(args.test_threshold)
        and learned_internal_best >= max(query_best, context_best) + float(args.min_margin)
        and learned_internal_step is not None
    )
    return {
        "status": "PASS" if pass_status else "FAIL",
        "passed": pass_status,
        "claim": "a learned neural shape engine can induce and apply task structure internally in one forward pass",
        "task_family": f"affine_mod_{args.prime}",
        "heldout_task_coefficients": True,
        "external_relation_miner_at_test": False,
        "oracle_programmatic_table_allowed_to_pass": False,
        "teacher_labels_used": 0,
        "query_only_best_test_acc": query_best,
        "context_mlp_best_test_acc": context_best,
        "learned_internal_shape_best_test_acc": learned_internal_best,
        "learned_internal_shape_mean_pass_step": learned_internal_step,
        "oracle_internal_table_shape_best_test_acc": oracle_table_best,
        "oracle_internal_table_shape_mean_pass_step": oracle_table_step,
        "test_threshold": float(args.test_threshold),
        "min_margin": float(args.min_margin),
        "interpretation": (
            "PASS: learned internal_shape solved heldout tasks from support examples and beat matched controls."
            if pass_status
            else "FAIL: learned internal_shape did not clear heldout accuracy/margin gates; any internal_table_shape success is an oracle/programmatic control."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="1337")
    parser.add_argument("--output-dir", default="outputs/internal_shape_engine")
    parser.add_argument("--prime", type=int, default=17)
    parser.add_argument("--meta-train-fraction", type=float, default=0.75)
    parser.add_argument("--branches", default="query_only,context_mlp,internal_shape,internal_table_shape")

    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--support-size", type=int, default=32)
    parser.add_argument("--task-batch", type=int, default=128)
    parser.add_argument("--eval-task-batch", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--test-threshold", type=float, default=0.90)
    parser.add_argument("--min-margin", type=float, default=0.20)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.prime = 11
        args.steps = 40
        args.support_size = 12
        args.task_batch = 32
        args.eval_task_batch = 64
        args.eval_batches = 2
        args.eval_interval = 10
        args.d_model = 48
        args.output_dir = str(Path(args.output_dir) / "smoke")

    device = resolve_device(str(args.device))
    seeds = parse_seeds(str(args.seeds))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "internal_shape_engine_config.json").open("w", encoding="utf-8") as handle:
        json.dump({**vars(args), "device_resolved": str(device)}, handle, indent=2, sort_keys=True)

    print("=" * 96)
    print("INTERNAL SHAPE ENGINE / ONE-FORWARD DISCOVERY AUDIT")
    print("=" * 96)
    print(f"device={device} seeds={seeds}")
    print("claim: neural model observes support patterns and applies induced transports internally.")
    print("external_relation_miner_at_test=False teacher_labels_used=0")

    start = time.time()
    summaries: List[Dict[str, object]] = []
    curves: List[Dict[str, object]] = []
    for seed in seeds:
        seed_summaries, seed_curves = run_seed(args, seed, device)
        summaries.extend(seed_summaries)
        curves.extend(seed_curves)

    verdict = verdict_from(args, summaries)
    verdict["wall_time_sec"] = time.time() - start
    write_csv(output_dir / "internal_shape_engine_summary.csv", summaries)
    write_csv(output_dir / "internal_shape_engine_curves.csv", curves)
    with (output_dir / "internal_shape_engine_verdict.json").open("w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2, sort_keys=True)

    print("=" * 96)
    print("INTERNAL SHAPE ENGINE VERDICT")
    print("=" * 96)
    for row in summaries:
        print(
            f"{row['branch']:<16} seed={row['seed']} "
            f"pass={'never' if row.get('first_pass_step') is None else row.get('first_pass_step')} "
            f"best_test={float(row['best_test_acc']):.3f} final_test={float(row['final_test_acc']):.3f}",
            flush=True,
        )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"artifacts={output_dir}")


if __name__ == "__main__":
    main()
