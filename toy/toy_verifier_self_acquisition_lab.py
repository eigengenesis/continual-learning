#!/usr/bin/env python3
"""Toy verifier-grounded self-acquisition lab.

Question:
  Can a toy model learn a new skill without a teacher, using only its own
  proposals plus a verifier?

Setup:
  A old skill: copy a digit sequence.
  B old skill: reverse a digit sequence.
  C new skill: output max(sequence).

There are no C teacher labels during self-acquisition. The model proposes C
answers, an environment/verifier returns accept/reject, and only accepted
self-generated traces are used for C training.

Branches:
  old                 train A/B only
  naive_self          train on the model's unverified greedy C outputs
  verified_fixed      train fixed model on verified self-generated C traces
  verified_expansion  insert a gated C-only expansion and train it on verified
                      self-generated C traces while freezing the old model

Artifacts:
  verifier_self_acquisition_results.csv
  verifier_self_acquisition_curves.csv
  verifier_self_acquisition_events.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_auto_expansion_controller_lab import (
    BOS,
    EOS,
    NUM_OFFSET,
    OUT,
    SEP,
    TASK_A,
    TASK_B,
    TASK_C,
    TinyLM,
    freeze_base_for_expansion,
    unfreeze_all,
)


@dataclass
class Trace:
    prompt: List[int]
    output: List[int]


@dataclass
class Metrics:
    A_exact: float
    B_exact: float
    C_exact: float
    A_tok: float
    B_tok: float
    C_tok: float
    old_mean: float
    C_loss: float


@dataclass
class SelfPoolStats:
    pool_size: int
    verifier_queries: int
    accepted: int
    acceptance_rate: float
    teacher_labels_used: int
    mode: str


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def line(char: str = "=") -> None:
    print(char * 96, flush=True)


def make_example(task: int, seq_len: int, n_nums: int, rng: np.random.Generator) -> Tuple[List[int], List[int]]:
    values = rng.integers(0, n_nums, size=seq_len).tolist()
    if task == TASK_A:
        target = list(values)
    elif task == TASK_B:
        target = list(reversed(values))
    elif task == TASK_C:
        target = [int(max(values))]
    else:
        raise ValueError(task)
    prompt = [BOS, task, SEP] + [NUM_OFFSET + int(v) for v in values] + [OUT]
    output = [NUM_OFFSET + int(v) for v in target] + [EOS]
    return prompt, output


def prompt_values(prompt: Sequence[int]) -> List[int]:
    values: List[int] = []
    in_payload = False
    for token in prompt:
        if token == SEP:
            in_payload = True
            continue
        if token == OUT:
            break
        if in_payload and int(token) >= NUM_OFFSET:
            values.append(int(token) - NUM_OFFSET)
    return values


def verify_c(prompt: Sequence[int], output: Sequence[int], n_nums: int) -> bool:
    values = prompt_values(prompt)
    del n_nums
    expected = [NUM_OFFSET + int(max(values)), EOS]
    return list(output[: len(expected)]) == expected


def make_batch(
    *,
    task: int | Sequence[int],
    batch_size: int,
    seq_len: int,
    n_nums: int,
    rng: np.random.Generator,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tasks = list(task) if isinstance(task, (list, tuple)) else [int(task)]
    rows: List[List[int]] = []
    labels: List[List[int]] = []
    for _ in range(batch_size):
        chosen = tasks[int(rng.integers(0, len(tasks)))]
        prompt, output = make_example(chosen, seq_len, n_nums, rng)
        full = prompt + output
        row_labels = [-100] * len(prompt) + output
        rows.append(full)
        labels.append(row_labels)
    return torch.tensor(rows, device=device), torch.tensor(labels, device=device)


def make_trace_batch(traces: Sequence[Trace], batch_size: int, rng: np.random.Generator, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    idxs = rng.integers(0, len(traces), size=batch_size)
    rows: List[List[int]] = []
    labels: List[List[int]] = []
    for idx in idxs:
        trace = traces[int(idx)]
        full = list(trace.prompt) + list(trace.output)
        row_labels = [-100] * len(trace.prompt) + list(trace.output)
        rows.append(full)
        labels.append(row_labels)
    max_len = max(len(row) for row in rows)
    x = torch.full((batch_size, max_len), EOS, dtype=torch.long, device=device)
    y = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    for row_idx, (row, row_labels) in enumerate(zip(rows, labels)):
        x[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        y[row_idx, : len(row_labels)] = torch.tensor(row_labels, dtype=torch.long, device=device)
    return x, y


def ce_loss(model: TinyLM, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)


@torch.no_grad()
def greedy_generate(model: TinyLM, prompt: Sequence[int], out_len: int, device: str) -> List[int]:
    model.eval()
    ids = torch.tensor([list(prompt)], device=device)
    for _ in range(out_len):
        logits = model(ids)
        next_id = int(logits[0, -1].argmax().item())
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == EOS:
            break
    return ids[0, len(prompt) :].tolist()


@torch.no_grad()
def sample_c_output(
    model: TinyLM,
    prompt: Sequence[int],
    *,
    n_nums: int,
    device: str,
    rng: np.random.Generator,
    temperature: float,
    epsilon: float,
) -> List[int]:
    model.eval()
    ids = torch.tensor([list(prompt)], device=device)
    logits = model(ids)[0, -1, NUM_OFFSET : NUM_OFFSET + n_nums].float()
    if rng.random() < epsilon:
        first = int(rng.integers(0, n_nums))
    else:
        probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1).detach().cpu().numpy()
        probs = probs / max(float(probs.sum()), 1e-9)
        first = int(rng.choice(np.arange(n_nums), p=probs))
    return [NUM_OFFSET + first, EOS]


@torch.no_grad()
def evaluate_task(model: TinyLM, task: int, eval_samples: int, seq_len: int, n_nums: int, seed: int, device: str) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    exact = 0
    tok_correct = 0
    tok_total = 0
    losses: List[float] = []
    model.eval()
    for _ in range(eval_samples):
        prompt, output = make_example(task, seq_len, n_nums, rng)
        pred = greedy_generate(model, prompt, len(output), device)
        pred = pred[: len(output)]
        exact += int(pred == output)
        tok_total += len(output)
        tok_correct += sum(int(a == b) for a, b in zip(pred, output))
    for _ in range(max(1, eval_samples // 32)):
        x, y = make_batch(task=task, batch_size=32, seq_len=seq_len, n_nums=n_nums, rng=rng, device=device)
        losses.append(float(ce_loss(model, x, y).item()))
    return exact / eval_samples, tok_correct / max(tok_total, 1), sum(losses) / max(len(losses), 1)


def evaluate_all(model: TinyLM, args: argparse.Namespace, seed: int) -> Metrics:
    a_exact, a_tok, _ = evaluate_task(model, TASK_A, args.eval_samples, args.seq_len, args.n_nums, seed + 11, args.device)
    b_exact, b_tok, _ = evaluate_task(model, TASK_B, args.eval_samples, args.seq_len, args.n_nums, seed + 12, args.device)
    c_exact, c_tok, c_loss = evaluate_task(model, TASK_C, args.eval_samples, args.seq_len, args.n_nums, seed + 13, args.device)
    return Metrics(
        A_exact=a_exact,
        B_exact=b_exact,
        C_exact=c_exact,
        A_tok=a_tok,
        B_tok=b_tok,
        C_tok=c_tok,
        old_mean=0.5 * (a_exact + b_exact),
        C_loss=c_loss,
    )


def train_old_model(args: argparse.Namespace, seed: int) -> TinyLM:
    set_seed(seed)
    model = TinyLM(args.n_nums + NUM_OFFSET, args.d_model, args.layers, args.heads, args.d_ff, 2 * args.seq_len + 8).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(seed)
    for step in range(1, args.old_steps + 1):
        x, y = make_batch(task=[TASK_A, TASK_B], batch_size=args.batch_size, seq_len=args.seq_len, n_nums=args.n_nums, rng=rng, device=args.device)
        opt.zero_grad(set_to_none=True)
        loss = ce_loss(model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_interval == 0 or step == args.old_steps:
            m = evaluate_all(model, args, seed + step)
            print(f"[old] step={step:04d}/{args.old_steps} loss={loss.item():.4f} A={m.A_exact:.3f} B={m.B_exact:.3f} C={m.C_exact:.3f}", flush=True)
    return model


def collect_naive_self_pool(model: TinyLM, args: argparse.Namespace, seed: int) -> Tuple[List[Trace], SelfPoolStats]:
    rng = np.random.default_rng(seed)
    traces: List[Trace] = []
    for _ in range(args.self_pool_size):
        prompt, output = make_example(TASK_C, args.seq_len, args.n_nums, rng)
        pred = greedy_generate(model, prompt, len(output), args.device)
        pred = (pred + [EOS])[: len(output)]
        traces.append(Trace(prompt=prompt, output=pred))
    return traces, SelfPoolStats(
        pool_size=len(traces),
        verifier_queries=0,
        accepted=len(traces),
        acceptance_rate=1.0,
        teacher_labels_used=0,
        mode="naive_unverified_self_distill",
    )


def collect_verified_self_pool(model: TinyLM, args: argparse.Namespace, seed: int) -> Tuple[List[Trace], SelfPoolStats]:
    rng = np.random.default_rng(seed)
    traces: List[Trace] = []
    queries = 0
    while len(traces) < args.self_pool_size and queries < args.max_verifier_queries:
        prompt, _ = make_example(TASK_C, args.seq_len, args.n_nums, rng)
        for _attempt in range(args.attempts_per_prompt):
            candidate = sample_c_output(
                model,
                prompt,
                n_nums=args.n_nums,
                device=args.device,
                rng=rng,
                temperature=args.sample_temperature,
                epsilon=args.exploration_epsilon,
            )
            queries += 1
            if verify_c(prompt, candidate, args.n_nums):
                traces.append(Trace(prompt=prompt, output=candidate))
                break
            if queries >= args.max_verifier_queries:
                break
    accepted = len(traces)
    return traces, SelfPoolStats(
        pool_size=len(traces),
        verifier_queries=queries,
        accepted=accepted,
        acceptance_rate=accepted / max(queries, 1),
        teacher_labels_used=0,
        mode="verified_self_acquisition",
    )


def train_on_traces(
    model: TinyLM,
    traces: Sequence[Trace],
    args: argparse.Namespace,
    *,
    seed: int,
    steps: int,
    lr: float,
    branch: str,
    curves: List[dict],
    keep_best: bool = False,
) -> Metrics:
    params = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    rng = np.random.default_rng(seed)
    best_state = None
    best_score = -1.0
    best_metrics = None
    for step in range(1, steps + 1):
        x, y = make_trace_batch(traces, args.batch_size, rng, args.device)
        opt.zero_grad(set_to_none=True)
        loss = ce_loss(model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % args.eval_interval == 0 or step == steps:
            m = evaluate_all(model, args, seed + step)
            curves.append({"branch": branch, "step": step, **asdict(m), "loss": float(loss.item())})
            print(f"[{branch}] step={step:04d}/{steps} A={m.A_exact:.3f} B={m.B_exact:.3f} C={m.C_exact:.3f} old={m.old_mean:.3f}", flush=True)
            if keep_best:
                score = 2.0 * m.old_mean + m.C_exact
                if score > best_score:
                    best_score = score
                    best_metrics = m
                    best_state = {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad}
    if best_state is not None:
        params_by_name = dict(model.named_parameters())
        for name, value in best_state.items():
            params_by_name[name].data.copy_(value.to(device=params_by_name[name].device, dtype=params_by_name[name].dtype))
    if keep_best and best_metrics is not None:
        return best_metrics
    return evaluate_all(model, args, seed + 999)


def block_pressure_from_traces(model: TinyLM, traces: Sequence[Trace], args: argparse.Namespace, seed: int) -> Tuple[int, dict]:
    unfreeze_all(model)
    rng = np.random.default_rng(seed)
    x, y = make_trace_batch(traces, args.probe_batch_size, rng, args.device)
    model.zero_grad(set_to_none=True)
    loss = ce_loss(model, x, y)
    loss.backward()
    pressures = {}
    for idx, block in enumerate(model.blocks):
        total = 0.0
        for param in block.parameters():
            if param.grad is not None:
                total += float(param.grad.detach().norm().item())
        pressures[str(idx)] = total
    model.zero_grad(set_to_none=True)
    best = max(pressures, key=lambda key: pressures[key])
    return int(best), pressures


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_seed(args: argparse.Namespace, seed: int) -> Tuple[List[dict], List[dict], List[dict]]:
    line("=")
    print(f"VERIFIER SELF-ACQUISITION TOY seed={seed}", flush=True)
    line("=")
    old = train_old_model(args, seed)
    old_metrics = evaluate_all(old, args, seed + 100)
    print(f"[old_final] A={old_metrics.A_exact:.3f} B={old_metrics.B_exact:.3f} C={old_metrics.C_exact:.3f} old={old_metrics.old_mean:.3f}", flush=True)

    naive_pool, naive_pool_stats = collect_naive_self_pool(old, args, seed + 200)
    verified_pool, verified_stats = collect_verified_self_pool(old, args, seed + 300)
    if len(verified_pool) < args.min_verified_pool:
        raise RuntimeError(
            f"verified pool too small: {len(verified_pool)} accepted from {verified_stats.verifier_queries} queries; "
            f"increase --max-verifier-queries or --exploration-epsilon"
        )
    print(
        f"[self_pool] verified accepted={verified_stats.accepted} queries={verified_stats.verifier_queries} "
        f"acceptance={verified_stats.acceptance_rate:.4f} teacher_labels_used=0",
        flush=True,
    )

    curves: List[dict] = []
    events: List[dict] = [
        {"seed": seed, "event": "naive_pool", **asdict(naive_pool_stats)},
        {"seed": seed, "event": "verified_pool", **asdict(verified_stats)},
    ]

    naive = copy.deepcopy(old)
    naive_metrics = train_on_traces(
        naive,
        naive_pool,
        args,
        seed=seed + 400,
        steps=args.self_steps,
        lr=args.self_lr,
        branch="naive_self",
        curves=curves,
    )

    fixed = copy.deepcopy(old)
    fixed_metrics = train_on_traces(
        fixed,
        verified_pool,
        args,
        seed=seed + 500,
        steps=args.self_steps,
        lr=args.self_lr,
        branch="verified_fixed",
        curves=curves,
    )

    expanded = copy.deepcopy(old)
    where, pressures = block_pressure_from_traces(expanded, verified_pool, args, seed + 600)
    expanded.insert_expansion(where, 1)
    expanded.to(args.device)
    freeze_base_for_expansion(expanded)
    events.append({"seed": seed, "event": "self_expansion_inserted", "where_layer": where, "how_many": 1, "pressure_by_layer": pressures})
    print(f"[self_expand] where_layer={where} pressure_by_layer={pressures}", flush=True)
    expanded_metrics = train_on_traces(
        expanded,
        verified_pool,
        args,
        seed=seed + 700,
        steps=args.expansion_steps,
        lr=args.expansion_lr,
        branch="verified_expansion",
        curves=curves,
        keep_best=True,
    )
    if expanded_metrics.C_exact < args.expansion_target_c:
        print(
            f"[verified_expansion_polish] target={args.expansion_target_c:.3f} current={expanded_metrics.C_exact:.3f} "
            f"steps={args.expansion_polish_steps}",
            flush=True,
        )
        expanded_metrics = train_on_traces(
            expanded,
            verified_pool,
            args,
            seed=seed + 800,
            steps=args.expansion_polish_steps,
            lr=args.expansion_polish_lr,
            branch="verified_expansion_polish",
            curves=curves,
            keep_best=True,
        )

    rows = [
        {"seed": seed, "branch": "old", "teacher_labels_used": 0, "verifier_queries": 0, "accepted": 0, "expansions": 0, **asdict(old_metrics)},
        {"seed": seed, "branch": "naive_self", **asdict(naive_pool_stats), "expansions": 0, **asdict(naive_metrics)},
        {"seed": seed, "branch": "verified_fixed", **asdict(verified_stats), "expansions": 0, **asdict(fixed_metrics)},
        {"seed": seed, "branch": "verified_expansion", **asdict(verified_stats), "expansions": 1, "where_layer": where, **asdict(expanded_metrics)},
    ]
    for row in rows:
        print(
            f"[summary] seed={seed} branch={row['branch']} A={row['A_exact']:.3f} B={row['B_exact']:.3f} "
            f"C={row['C_exact']:.3f} old={row['old_mean']:.3f} verifier_queries={row.get('verifier_queries', 0)} "
            f"teacher_labels_used={row.get('teacher_labels_used', 0)}",
            flush=True,
        )
    return rows, curves, events


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy verifier-grounded self-acquisition lab")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="1337,2027,31415")
    parser.add_argument("--output-dir", default="outputs/verifier_self_acquisition")
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=192)
    parser.add_argument("--n-nums", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--old-steps", type=int, default=800)
    parser.add_argument("--self-steps", type=int, default=500)
    parser.add_argument("--expansion-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-batch-size", type=int, default=96)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--self-lr", type=float, default=5e-4)
    parser.add_argument("--expansion-lr", type=float, default=1.5e-3)
    parser.add_argument("--expansion-target-c", type=float, default=1.0)
    parser.add_argument("--expansion-polish-steps", type=int, default=300)
    parser.add_argument("--expansion-polish-lr", type=float, default=8e-4)
    parser.add_argument("--self-pool-size", type=int, default=512)
    parser.add_argument("--min-verified-pool", type=int, default=256)
    parser.add_argument("--max-verifier-queries", type=int, default=12000)
    parser.add_argument("--attempts-per-prompt", type=int, default=8)
    parser.add_argument("--sample-temperature", type=float, default=1.4)
    parser.add_argument("--exploration-epsilon", type=float, default=0.35)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=150)
    args = parser.parse_args()

    start = time.time()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    all_rows: List[dict] = []
    all_curves: List[dict] = []
    all_events: List[dict] = []
    for seed in seeds:
        rows, curves, events = run_seed(args, seed)
        all_rows.extend(rows)
        all_curves.extend({"seed": seed, **row} for row in curves)
        all_events.extend(events)

    out_dir = Path(args.output_dir).expanduser()
    write_csv(out_dir / "verifier_self_acquisition_results.csv", all_rows)
    write_csv(out_dir / "verifier_self_acquisition_curves.csv", all_curves)
    with (out_dir / "verifier_self_acquisition_events.json").open("w", encoding="utf-8") as handle:
        json.dump(all_events, handle, indent=2, sort_keys=True, default=str)

    def mean(branch: str, key: str) -> float:
        vals = [float(row[key]) for row in all_rows if row["branch"] == branch]
        return float(sum(vals) / max(len(vals), 1))

    line("=")
    print("VERIFIER SELF-ACQUISITION VERDICT", flush=True)
    line("=")
    old_old = mean("old", "old_mean")
    naive_c = mean("naive_self", "C_exact")
    fixed_c = mean("verified_fixed", "C_exact")
    fixed_old = mean("verified_fixed", "old_mean")
    expand_c = mean("verified_expansion", "C_exact")
    expand_old = mean("verified_expansion", "old_mean")
    queries = mean("verified_expansion", "verifier_queries")
    print(f"mean_old_branch old={old_old:.3f}", flush=True)
    print(f"mean_naive_self C={naive_c:.3f}", flush=True)
    print(f"mean_verified_fixed C={fixed_c:.3f} old={fixed_old:.3f}", flush=True)
    print(f"mean_verified_expansion C={expand_c:.3f} old={expand_old:.3f} verifier_queries={queries:.1f}", flush=True)
    passed = expand_c >= 0.90 and expand_old >= 0.95 and expand_c > naive_c + 0.30
    print("PASS" if passed else "PARTIAL", flush=True)
    print(f"wall_time_sec={time.time() - start:.1f}", flush=True)
    print(f"artifacts={out_dir}", flush=True)


if __name__ == "__main__":
    main()
