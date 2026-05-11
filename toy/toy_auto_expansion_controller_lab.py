#!/usr/bin/env python3
"""Toy automatic expansion controller lab.

Question:
  Can a toy continual learner decide when to expand, where to insert capacity,
  and how much capacity to add, instead of using a hand-picked expansion branch?

This is not a universal proof. It is a controlled mechanism test for an
automatic expansion policy:

  when: trigger on old-skill retention failure or new-task stall/interference
  where: choose the block with largest old/new gradient conflict pressure
  how much: insert 1 or 2 gated blocks based on trigger severity

Artifacts:
  auto_expansion_toy_results.csv
  auto_expansion_toy_curves.csv
  auto_expansion_toy_events.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PAD, BOS, SEP, OUT, EOS = 0, 1, 2, 3, 4
TASK_A, TASK_B, TASK_C = 5, 6, 7
NUM_OFFSET = 8


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Metrics:
    A_exact: float
    B_exact: float
    C_exact: float
    A_tok: float
    B_tok: float
    C_tok: float
    old_mean: float
    loss_c: float


@dataclass
class ExpansionEvent:
    step: int
    where_layer: int
    how_many_blocks: int
    accepted: bool
    selected_candidate: str
    severity: float
    old_drop: float
    progress: float
    conflict: float
    trigger_reason: str
    pressure_by_layer: Dict[str, float]
    candidates: List[Dict[str, float | int | str]]


@dataclass
class CandidateResult:
    name: str
    where_layer: int
    how_many_blocks: int
    growth: bool
    metrics: Metrics


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn
        x = x + self.ff(self.ln2(x))
        return x


class AutoExpansionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, bias_init: float = -6.0) -> None:
        super().__init__()
        self.block = TinyBlock(d_model, n_heads, d_ff)
        self.router = nn.Linear(d_model, 1)
        nn.init.zeros_(self.router.weight)
        nn.init.constant_(self.router.bias, bias_init)

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        # Route from the first token, which carries the task identity in this toy.
        return torch.sigmoid(self.router(x[:, 0, :])).view(-1, 1, 1)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        delta = self.block(x, attn_mask) - x
        task_mask = (task_ids == TASK_C).to(device=x.device, dtype=x.dtype).view(-1, 1, 1)
        return x + task_mask * self.gate(x).to(dtype=x.dtype) * delta


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, max_len: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)
        self.max_len = int(max_len)
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TinyBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, length = idx.shape
        pos = torch.arange(length, device=idx.device).unsqueeze(0)
        x = self.token(idx) + self.pos(pos)
        mask = self.causal_mask(length, idx.device)
        task_ids = idx[:, 1] if length > 1 else torch.full((bsz,), -1, device=idx.device, dtype=idx.dtype)
        for block in self.blocks:
            if isinstance(block, AutoExpansionBlock):
                x = block(x, mask, task_ids)
            else:
                x = block(x, mask)
        return self.head(self.ln(x))

    def insert_expansion(self, after_layer: int, how_many: int) -> None:
        layers = list(self.blocks)
        insert_at = min(max(int(after_layer) + 1, 0), len(layers))
        new_blocks = [AutoExpansionBlock(self.d_model, self.n_heads, self.d_ff) for _ in range(int(how_many))]
        self.blocks = nn.ModuleList(layers[:insert_at] + new_blocks + layers[insert_at:])


def freeze_base_for_expansion(model: TinyLM) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for block in model.blocks:
        if isinstance(block, AutoExpansionBlock):
            for param in block.parameters():
                param.requires_grad = True


def unfreeze_all(model: TinyLM) -> None:
    for param in model.parameters():
        param.requires_grad = True


def make_example(task: int, seq_len: int, n_nums: int, rng: np.random.Generator) -> Tuple[List[int], List[int]]:
    values = rng.integers(0, n_nums, size=seq_len).tolist()
    if task == TASK_A:
        target = list(values)
    elif task == TASK_B:
        target = list(reversed(values))
    elif task == TASK_C:
        target = sorted(values)
    else:
        raise ValueError(task)
    prompt = [BOS, task, SEP] + [NUM_OFFSET + v for v in values] + [OUT]
    output = [NUM_OFFSET + v for v in target] + [EOS]
    return prompt, output


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
    prompts: List[List[int]] = []
    targets: List[List[int]] = []
    for _ in range(batch_size):
        chosen = tasks[int(rng.integers(0, len(tasks)))]
        prompt, output = make_example(chosen, seq_len, n_nums, rng)
        full = prompt + output
        labels = [-100] * len(prompt) + output
        prompts.append(full)
        targets.append(labels)
    return torch.tensor(prompts, device=device), torch.tensor(targets, device=device)


def ce_loss(model: TinyLM, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)


@torch.no_grad()
def generate(model: TinyLM, prompt: List[int], out_len: int, device: str) -> List[int]:
    model.eval()
    ids = torch.tensor([prompt], device=device)
    for _ in range(out_len):
        logits = model(ids)
        next_id = int(logits[0, -1].argmax().item())
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == EOS:
            break
    return ids[0, len(prompt) :].tolist()


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
        pred = generate(model, prompt, len(output), device)
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
        old_mean=(a_exact + b_exact) / 2.0,
        loss_c=c_loss,
    )


def block_param_groups(model: TinyLM) -> List[List[nn.Parameter]]:
    return [[p for p in block.parameters() if p.requires_grad] for block in model.blocks]


def flat_grads(params: Iterable[nn.Parameter]) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for param in params:
        if param.grad is not None:
            parts.append(param.grad.detach().flatten().float().cpu())
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


def grad_pressure(model: TinyLM, args: argparse.Namespace, seed: int) -> Tuple[int, float, Dict[str, float]]:
    rng_old = np.random.default_rng(seed + 1001)
    rng_new = np.random.default_rng(seed + 2001)
    old_x, old_y = make_batch(task=[TASK_A, TASK_B], batch_size=args.probe_batch_size, seq_len=args.seq_len, n_nums=args.n_nums, rng=rng_old, device=args.device)
    new_x, new_y = make_batch(task=TASK_C, batch_size=args.probe_batch_size, seq_len=args.seq_len, n_nums=args.n_nums, rng=rng_new, device=args.device)
    groups = block_param_groups(model)

    model.zero_grad(set_to_none=True)
    ce_loss(model, new_x, new_y).backward()
    new_grads = [flat_grads(group) for group in groups]

    model.zero_grad(set_to_none=True)
    ce_loss(model, old_x, old_y).backward()
    old_grads = [flat_grads(group) for group in groups]
    model.zero_grad(set_to_none=True)

    pressures: Dict[str, float] = {}
    for idx, (new_g, old_g) in enumerate(zip(new_grads, old_grads)):
        n_norm = float(new_g.norm().item())
        o_norm = float(old_g.norm().item())
        denom = max(n_norm * o_norm, 1e-9)
        cos = float(torch.dot(new_g, old_g).item() / denom) if new_g.numel() == old_g.numel() else 0.0
        conflict = max(0.0, -cos)
        pressure = n_norm * (1.0 + conflict) * (0.5 + min(o_norm, 5.0))
        if isinstance(model.blocks[idx], AutoExpansionBlock):
            pressure *= 0.25
        pressures[str(idx)] = pressure
    where = int(max(pressures, key=lambda key: pressures[key]))
    return where, float(pressures[str(where)]), pressures


def train_old_model(args: argparse.Namespace, seed: int) -> TinyLM:
    rng = np.random.default_rng(seed)
    vocab_size = NUM_OFFSET + args.n_nums
    max_len = 3 + args.seq_len + 1 + args.seq_len + 1
    model = TinyLM(vocab_size, args.d_model, args.layers, args.heads, args.d_ff, max_len).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for step in range(1, args.old_steps + 1):
        x, y = make_batch(task=[TASK_A, TASK_B], batch_size=args.batch_size, seq_len=args.seq_len, n_nums=args.n_nums, rng=rng, device=args.device)
        loss = ce_loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_interval == 0 or step == args.old_steps:
            m = evaluate_all(model, args, seed + 3000 + step)
            print(f"[old] step={step:04d}/{args.old_steps} loss={float(loss.item()):.4f} A={m.A_exact:.3f} B={m.B_exact:.3f} C={m.C_exact:.3f}", flush=True)
    return model


def train_fixed_branch(base: TinyLM, args: argparse.Namespace, seed: int, curves: List[Dict[str, float]]) -> Metrics:
    model = copy.deepcopy(base).to(args.device)
    unfreeze_all(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(seed + 4000)
    for step in range(1, args.new_steps + 1):
        x, y = make_batch(task=TASK_C, batch_size=args.batch_size, seq_len=args.seq_len, n_nums=args.n_nums, rng=rng, device=args.device)
        loss = ce_loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_interval == 0 or step == args.new_steps:
            m = evaluate_all(model, args, seed + 5000 + step)
            curves.append({"branch": "fixed", "step": step, **asdict(m)})
            print(f"[fixed] step={step:04d}/{args.new_steps} A={m.A_exact:.3f} B={m.B_exact:.3f} C={m.C_exact:.3f} old={m.old_mean:.3f}", flush=True)
    return evaluate_all(model, args, seed + 5999)


def train_c_branch(
    model: TinyLM,
    args: argparse.Namespace,
    seed: int,
    steps: int,
    *,
    expansion_only: bool,
) -> Tuple[TinyLM, Metrics]:
    if expansion_only:
        freeze_base_for_expansion(model)
        lr = args.expansion_lr
    else:
        unfreeze_all(model)
        lr = args.lr
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("candidate has no trainable parameters")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    for _ in range(int(steps)):
        x, y = make_batch(task=TASK_C, batch_size=args.batch_size, seq_len=args.seq_len, n_nums=args.n_nums, rng=rng, device=args.device)
        loss = ce_loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    return model, evaluate_all(model, args, seed + 999)


def candidate_score(candidate: CandidateResult, args: argparse.Namespace) -> float:
    return (
        float(candidate.metrics.C_exact)
        + float(args.governor_old_weight) * float(candidate.metrics.old_mean)
        - float(args.governor_block_penalty) * float(candidate.how_many_blocks)
    )


def run_growth_governor(
    *,
    base: TinyLM,
    args: argparse.Namespace,
    seed: int,
    step: int,
    where: int,
    severity_how_many: int,
    pressures: Dict[str, float],
) -> Tuple[TinyLM, CandidateResult, List[CandidateResult]]:
    sorted_layers = [
        int(layer)
        for layer, _pressure in sorted(pressures.items(), key=lambda item: float(item[1]), reverse=True)
    ][: max(1, int(args.governor_top_k_layers))]
    block_counts = sorted(set([1, int(severity_how_many)]))
    block_counts = [min(max(1, value), int(args.max_new_blocks)) for value in block_counts]

    candidates: List[Tuple[str, TinyLM, bool, int, int]] = []
    no_growth = copy.deepcopy(base).to(args.device)
    candidates.append(("no_growth", no_growth, False, -1, 0))
    for layer in sorted_layers:
        for count in block_counts:
            grown = copy.deepcopy(base).to(args.device)
            grown.insert_expansion(layer, count)
            candidates.append((f"expand_l{layer}_x{count}", grown, True, layer, count))

    results: List[CandidateResult] = []
    models: Dict[str, TinyLM] = {}
    for idx, (name, model, growth, layer, count) in enumerate(candidates):
        trained, metrics = train_c_branch(
            model,
            args,
            seed + 20_000 + step * 31 + idx * 997,
            args.governor_probe_steps,
            expansion_only=growth,
        )
        result = CandidateResult(
            name=name,
            where_layer=layer,
            how_many_blocks=count,
            growth=growth,
            metrics=metrics,
        )
        results.append(result)
        models[name] = trained
        print(
            f"[governor_candidate] {name:<14} score={candidate_score(result, args):.3f} "
            f"A={metrics.A_exact:.3f} B={metrics.B_exact:.3f} C={metrics.C_exact:.3f} old={metrics.old_mean:.3f}",
            flush=True,
        )

    no_growth_result = next(item for item in results if item.name == "no_growth")
    growth_results = [item for item in results if item.growth]
    feasible_growth = [
        item
        for item in growth_results
        if item.metrics.old_mean >= args.accept_old_floor
        and candidate_score(item, args) >= candidate_score(no_growth_result, args) - args.accept_score_slack
    ]
    if feasible_growth:
        selected = max(feasible_growth, key=lambda item: candidate_score(item, args))
    else:
        selected = no_growth_result
    # Drop unselected models promptly. The selected model continues training.
    selected_model = models[selected.name]
    for name, model in list(models.items()):
        if name != selected.name:
            del model
    print(
        f"[governor_select] selected={selected.name} accepted_growth={selected.growth} "
        f"score={candidate_score(selected, args):.3f}",
        flush=True,
    )
    return selected_model, selected, results


def train_auto_branch(base: TinyLM, args: argparse.Namespace, seed: int, curves: List[Dict[str, float]]) -> Tuple[Metrics, List[ExpansionEvent]]:
    model = copy.deepcopy(base).to(args.device)
    unfreeze_all(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(seed + 6000)
    reference = evaluate_all(model, args, seed + 6100)
    events: List[ExpansionEvent] = []
    prev_c = 0.0
    expanded = False
    for step in range(1, args.new_steps + 1):
        x, y = make_batch(task=TASK_C, batch_size=args.batch_size, seq_len=args.seq_len, n_nums=args.n_nums, rng=rng, device=args.device)
        loss = ce_loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % args.eval_interval == 0 or step == args.new_steps:
            m = evaluate_all(model, args, seed + 7000 + step)
            curves.append({"branch": "auto", "step": step, **asdict(m)})
            progress = m.C_exact - prev_c
            prev_c = m.C_exact
            old_drop = max(0.0, reference.old_mean - m.old_mean)
            print(f"[auto] step={step:04d}/{args.new_steps} A={m.A_exact:.3f} B={m.B_exact:.3f} C={m.C_exact:.3f} old={m.old_mean:.3f} progress={progress:+.3f}", flush=True)

            retention_failure = old_drop >= args.old_drop_trigger
            learning_stall = m.C_exact < args.new_target_exact and progress <= args.stall_trigger
            should_probe = step >= args.min_expand_step and not expanded and (retention_failure or learning_stall)
            if should_probe:
                where, conflict, pressures = grad_pressure(model, args, seed + step)
                severity = old_drop + max(0.0, args.stall_trigger - progress) + min(conflict / max(args.pressure_scale, 1e-9), 2.0)
                how_many = 2 if severity >= args.two_block_severity else 1
                how_many = min(how_many, args.max_new_blocks)
                trigger_reason = "retention_failure" if retention_failure else "learning_stall"
                print(
                    f"[auto_expand] step={step} where_layer={where} how_many={how_many} "
                    f"severity={severity:.3f} old_drop={old_drop:.3f} progress={progress:.3f} "
                    f"conflict={conflict:.3f} trigger={trigger_reason}",
                    flush=True,
                )
                # Growth governor: branch from the protected checkpoint, train
                # no-growth and expansion candidates briefly, then continue only
                # the branch that improves the retention/new-skill Pareto score.
                del model
                model, selected, candidates = run_growth_governor(
                    base=base,
                    args=args,
                    seed=seed,
                    step=step,
                    where=where,
                    severity_how_many=how_many,
                    pressures=pressures,
                )
                if selected.growth:
                    freeze_base_for_expansion(model)
                    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.expansion_lr, weight_decay=0.01)
                else:
                    unfreeze_all(model)
                    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
                events.append(
                    ExpansionEvent(
                        step=step,
                        where_layer=selected.where_layer,
                        how_many_blocks=selected.how_many_blocks,
                        accepted=bool(selected.growth),
                        selected_candidate=selected.name,
                        severity=severity,
                        old_drop=old_drop,
                        progress=progress,
                        conflict=conflict,
                        trigger_reason=trigger_reason,
                        pressure_by_layer=pressures,
                        candidates=[
                            {
                                "name": item.name,
                                "where_layer": item.where_layer,
                                "how_many_blocks": item.how_many_blocks,
                                "growth": int(item.growth),
                                "score": candidate_score(item, args),
                                **asdict(item.metrics),
                            }
                            for item in candidates
                        ],
                    )
                )
                expanded = True
    return evaluate_all(model, args, seed + 7999), events


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


def run_seed(args: argparse.Namespace, seed: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    print("=" * 96)
    print(f"AUTO EXPANSION TOY seed={seed}")
    print("=" * 96)
    set_seed(seed)
    curves: List[Dict[str, object]] = []
    old = train_old_model(args, seed)
    old_metrics = evaluate_all(old, args, seed + 333)
    print(f"[old_final] A={old_metrics.A_exact:.3f} B={old_metrics.B_exact:.3f} C={old_metrics.C_exact:.3f} old={old_metrics.old_mean:.3f}", flush=True)

    fixed = train_fixed_branch(old, args, seed, curves)
    auto, events = train_auto_branch(old, args, seed, curves)
    rows = [
        {"seed": seed, "branch": "old", "expansions": 0, **asdict(old_metrics)},
        {"seed": seed, "branch": "fixed", "expansions": 0, **asdict(fixed)},
        {
            "seed": seed,
            "branch": "auto",
            "expansions": sum(event.how_many_blocks for event in events),
            "growth_accepted": int(any(event.accepted for event in events)),
            "selected_candidate": events[0].selected_candidate if events else "none",
            "first_expand_step": events[0].step if events else -1,
            "first_expand_where": events[0].where_layer if events else -1,
            **asdict(auto),
        },
    ]
    event_rows = [{"seed": seed, **asdict(event)} for event in events]
    for row in rows:
        print(
            f"[summary] seed={seed} branch={row['branch']} expansions={row.get('expansions', 0)} "
            f"A={row['A_exact']:.3f} B={row['B_exact']:.3f} C={row['C_exact']:.3f} old={row['old_mean']:.3f}",
            flush=True,
        )
    return rows, curves, event_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy automatic expansion controller lab")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="1337,2027,31415")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=192)
    parser.add_argument("--n-nums", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--old-steps", type=int, default=900)
    parser.add_argument("--new-steps", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-batch-size", type=int, default=96)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--expansion-lr", type=float, default=1.5e-3)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=150)
    parser.add_argument("--min-expand-step", type=int, default=100)
    parser.add_argument("--new-target-exact", type=float, default=0.82)
    parser.add_argument("--old-drop-trigger", type=float, default=0.08)
    parser.add_argument("--stall-trigger", type=float, default=0.08)
    parser.add_argument("--pressure-scale", type=float, default=10.0)
    parser.add_argument("--two-block-severity", type=float, default=1.20)
    parser.add_argument("--max-new-blocks", type=int, default=2)
    parser.add_argument("--governor-probe-steps", type=int, default=80)
    parser.add_argument("--governor-top-k-layers", type=int, default=2)
    parser.add_argument("--governor-old-weight", type=float, default=0.75)
    parser.add_argument("--governor-block-penalty", type=float, default=0.02)
    parser.add_argument("--accept-old-floor", type=float, default=0.85)
    parser.add_argument("--accept-score-slack", type=float, default=0.05)
    args = parser.parse_args()

    if args.fast:
        args.seeds = args.seeds.split(",")[0]
        args.d_model = min(args.d_model, 64)
        args.d_ff = min(args.d_ff, 128)
        args.old_steps = min(args.old_steps, 250)
        args.new_steps = min(args.new_steps, 300)
        args.batch_size = min(args.batch_size, 64)
        args.probe_batch_size = min(args.probe_batch_size, 48)
        args.eval_samples = min(args.eval_samples, 48)
        args.eval_interval = min(args.eval_interval, 75)
        args.log_interval = min(args.log_interval, 100)
        args.min_expand_step = min(args.min_expand_step, 75)
        args.governor_probe_steps = min(args.governor_probe_steps, 55)

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, object]] = []
    all_curves: List[Dict[str, object]] = []
    all_events: List[Dict[str, object]] = []
    start = time.time()
    for seed_text in args.seeds.split(","):
        seed = int(seed_text.strip())
        rows, curves, events = run_seed(args, seed)
        all_rows.extend(rows)
        all_curves.extend({"seed": seed, **row} for row in curves)
        all_events.extend(events)
    write_csv(out_dir / "auto_expansion_toy_results.csv", all_rows)
    write_csv(out_dir / "auto_expansion_toy_curves.csv", all_curves)
    (out_dir / "auto_expansion_toy_events.json").write_text(json.dumps(all_events, indent=2, sort_keys=True), encoding="utf-8")

    auto_rows = [row for row in all_rows if row["branch"] == "auto"]
    fixed_rows = [row for row in all_rows if row["branch"] == "fixed"]
    if auto_rows and fixed_rows:
        auto_c = float(np.mean([float(row["C_exact"]) for row in auto_rows]))
        fixed_c = float(np.mean([float(row["C_exact"]) for row in fixed_rows]))
        auto_old = float(np.mean([float(row["old_mean"]) for row in auto_rows]))
        fixed_old = float(np.mean([float(row["old_mean"]) for row in fixed_rows]))
        expansions = int(sum(int(row.get("expansions", 0)) for row in auto_rows))
        accepted = int(sum(int(row.get("growth_accepted", 0)) for row in auto_rows))
        print("=" * 96)
        print("AUTO EXPANSION VERDICT")
        print("=" * 96)
        print(f"mean_fixed C={fixed_c:.3f} old={fixed_old:.3f}", flush=True)
        print(f"mean_auto  C={auto_c:.3f} old={auto_old:.3f} expansions={expansions} accepted={accepted}", flush=True)
        pass_auto = accepted > 0 and expansions > 0 and auto_c >= fixed_c - 0.02 and auto_old >= max(0.85, fixed_old + 0.50)
        print("PASS" if pass_auto else "PARTIAL", flush=True)
    print(f"wall_time_sec={time.time() - start:.1f}", flush=True)
    print(f"artifacts={out_dir}", flush=True)


if __name__ == "__main__":
    main()
