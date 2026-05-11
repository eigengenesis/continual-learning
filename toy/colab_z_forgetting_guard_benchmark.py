#!/usr/bin/env python3
"""
Z-guided continual-learning forgetting guard benchmark.

Run:
    python colab_z_forgetting_guard_benchmark.py

Hidden smoke mode:
    CHAOS_SMOKE=1 python colab_z_forgetting_guard_benchmark.py

Core question:
Does task-conditioned old-skill Z shock predict catastrophic forgetting early,
and can Z-triggered replay/shielding preserve the old skill better than
compute-matched non-Z controls?
"""

from __future__ import annotations

import copy
import csv
import math
import os
import random
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEEDS = [1337, 2027, 31415]
TEXT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".colab_cache"
CACHE_PATH = CACHE_DIR / "tinyshakespeare.txt"
CSV_PATH = ROOT / "z_forgetting_guard_results.csv"

EMBEDDED_FALLBACK_TEXT = """
ROMEO:
But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.

JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.

HAMLET:
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles.

MACBETH:
Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day,
To the last syllable of recorded time.
""".strip() * 500

OPENERS = "([{<"
CLOSERS = ")]}>"
CLOSE_FOR_OPEN = dict(zip(OPENERS, CLOSERS))
BRACKET_CHARS = OPENERS + CLOSERS + "=\n "

TRAIN_FRACTION = 0.95
BASE_LR = 3e-4
WEIGHT_DECAY = 0.08
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0

OLD_READY_SEQ = 0.90
FORGET_DROP = 0.15
Z_WARNING_MAX_DROP = 0.03
Z_SHOCK_THRESHOLD = 0.75
Z_WARNING_PERSISTENCE = 2
Z_FRONTIER_K = 2
LOSS_TRIGGER_REL = 0.10
ACCURACY_TRIGGER_DROP = 0.05
REPLAY_BUDGET_FRACTION = 0.12
REPLAY_BURST_OFFSETS = [1, 5, 9]
SHIELD_LR_MULTIPLIER = 0.25
SHIELD_COOLDOWN_STEPS = 50
FORGETTING_PREDICTION_WINDOW = 150
BOOTSTRAP_SAMPLES = 5000
BRACKET_MIN_DEPTH = 5
BRACKET_MAX_DEPTH = 18

BRANCH_NAMES = [
    "none",
    "periodic_replay_10pct",
    "random_matched_replay",
    "loss_triggered_replay",
    "accuracy_triggered_replay",
    "z_replay",
    "z_shield",
    "z_replay_shield",
]


if torch.cuda.is_available():
    DEVICE = "cuda"
    PHASE_A_MAX_STEPS = 1200
    PHASE_B_STEPS = 1200
    PHASE_A_EVAL_INTERVAL = 50
    BRANCH_LOG_INTERVAL = 200
    BLOCK_SIZE = 160
    BATCH_SIZE = 32
    D_MODEL = 256
    N_HEAD = 8
    N_LAYER = 6
    DROPOUT = 0.10
    TEXT_EVAL_BATCHES = 12
    TEXT_EVAL_BATCH = 24
    BRACKET_EVAL_BATCHES = 12
    BRACKET_EVAL_BATCH = 64
    PROBE_BATCH = 24
else:
    DEVICE = "cpu"
    PHASE_A_MAX_STEPS = 220
    PHASE_B_STEPS = 220
    PHASE_A_EVAL_INTERVAL = 25
    BRANCH_LOG_INTERVAL = 50
    BLOCK_SIZE = 96
    BATCH_SIZE = 12
    D_MODEL = 96
    N_HEAD = 4
    N_LAYER = 3
    DROPOUT = 0.10
    TEXT_EVAL_BATCHES = 4
    TEXT_EVAL_BATCH = 12
    BRACKET_EVAL_BATCHES = 4
    BRACKET_EVAL_BATCH = 32
    PROBE_BATCH = 12


SMOKE = os.environ.get("CHAOS_SMOKE", "0") == "1"
if SMOKE:
    SEEDS = [1337]
    PHASE_A_MAX_STEPS = min(PHASE_A_MAX_STEPS, 24)
    PHASE_B_STEPS = min(PHASE_B_STEPS, 36)
    PHASE_A_EVAL_INTERVAL = min(PHASE_A_EVAL_INTERVAL, 6)
    BRANCH_LOG_INTERVAL = min(BRANCH_LOG_INTERVAL, 12)
    BLOCK_SIZE = min(BLOCK_SIZE, 64)
    BATCH_SIZE = min(BATCH_SIZE, 6)
    D_MODEL = min(D_MODEL, 96)
    N_HEAD = 4
    N_LAYER = min(N_LAYER, 3)
    TEXT_EVAL_BATCHES = min(TEXT_EVAL_BATCHES, 2)
    TEXT_EVAL_BATCH = min(TEXT_EVAL_BATCH, 6)
    BRACKET_EVAL_BATCHES = min(BRACKET_EVAL_BATCHES, 2)
    BRACKET_EVAL_BATCH = min(BRACKET_EVAL_BATCH, 8)
    PROBE_BATCH = min(PROBE_BATCH, 6)
    BRACKET_MIN_DEPTH = 2
    BRACKET_MAX_DEPTH = 6
    Z_SHOCK_THRESHOLD = 0.0
    Z_WARNING_PERSISTENCE = 1
    Z_WARNING_MAX_DROP = 1.0
    SHIELD_COOLDOWN_STEPS = min(SHIELD_COOLDOWN_STEPS, 10)


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    critical_mask: torch.Tensor | None = None


@dataclass
class AnchorInfo:
    step: int
    old_loss: float
    old_close_acc: float
    old_seq_acc: float
    old_z: Dict[str, float]
    old_act_z: Dict[str, float]
    block_anchor_z: Dict[str, float]
    old_frontier: List[str]
    checkpoint: Dict[str, object]
    reached_ready: bool


@dataclass
class BranchRow:
    step: int
    old_loss: float
    old_close_acc: float
    old_seq_acc: float
    text_loss: float
    text_acc: float
    z_shock: float
    loss_shock: float
    acc_drop: float
    z_warning: bool
    replay_count: int
    shield_active: bool


@dataclass
class BranchResult:
    name: str
    rows: List[BranchRow]
    old_seq_auc: float
    old_close_auc: float
    text_loss_auc: float
    text_acc_auc: float
    forgetting_area: float
    forgetting_step: float
    first_z_warning_step: float
    z_lead_time: float
    z_precision_150: float
    auroc_z: float
    auroc_loss: float
    auroc_accuracy: float
    final_old_seq: float
    final_old_close: float
    final_text_loss: float
    final_text_acc: float
    replay_count: int
    replay_budget: int
    replay_steps: List[int]
    shield_steps: int
    verified_replay_cap: bool
    verified_shield_groups: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {sec:02d}s"
    return f"{minutes:02d}m {sec:02d}s"


def format_step(value: float) -> str:
    if not math.isfinite(value):
        return "miss"
    return str(int(value))


def format_signed(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:+.{digits}f}"


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    if not math.isfinite(mean):
        return "nan +/- nan"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def download_or_load_text() -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists():
        print(f"Loaded cached corpus: {CACHE_PATH}")
        return CACHE_PATH.read_text(encoding="utf-8")
    try:
        print(f"Downloading corpus from {TEXT_URL}")
        with urllib.request.urlopen(TEXT_URL, timeout=30) as response:
            text = response.read().decode("utf-8")
        CACHE_PATH.write_text(text, encoding="utf-8")
        print(f"Saved corpus cache to: {CACHE_PATH}")
        return text
    except Exception as exc:
        print(f"Download failed: {exc}")
        print("Falling back to embedded local corpus.")
        CACHE_PATH.write_text(EMBEDDED_FALLBACK_TEXT, encoding="utf-8")
        return EMBEDDED_FALLBACK_TEXT


def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text + BRACKET_CHARS))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def make_text_positions(train_len: int, val_len: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 100_000)
    train_positions = torch.randint(
        0,
        train_len - BLOCK_SIZE - 1,
        (PHASE_B_STEPS, BATCH_SIZE),
        generator=gen,
    )
    eval_positions = torch.randint(
        0,
        val_len - BLOCK_SIZE - 1,
        (TEXT_EVAL_BATCHES, TEXT_EVAL_BATCH),
        generator=gen,
    )
    return train_positions, eval_positions


def text_batch_from_positions(data: torch.Tensor, starts: torch.Tensor) -> Batch:
    x = torch.stack([data[int(s): int(s) + BLOCK_SIZE] for s in starts], dim=0)
    y = torch.stack([data[int(s) + 1: int(s) + BLOCK_SIZE + 1] for s in starts], dim=0)
    return Batch(x.to(DEVICE), y.to(DEVICE), None)


def build_bracket_stream(
    rng: random.Random,
    min_depth: int = BRACKET_MIN_DEPTH,
    max_depth: int = BRACKET_MAX_DEPTH,
) -> Tuple[List[str], List[bool]]:
    chars: List[str] = []
    critical: List[bool] = []
    while len(chars) < BLOCK_SIZE + 1:
        depth = rng.randint(min_depth, max_depth)
        opens = [rng.choice(OPENERS) for _ in range(depth)]
        closes = [CLOSE_FOR_OPEN[ch] for ch in reversed(opens)]
        episode = opens + ["="] + closes + ["\n"]
        flags = [False] * len(opens) + [False] + [True] * len(closes) + [False]
        chars.extend(episode)
        critical.extend(flags)
    return chars[: BLOCK_SIZE + 1], critical[: BLOCK_SIZE + 1]


def make_bracket_batch(rng: random.Random, stoi: Dict[str, int], batch_size: int) -> Batch:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for _ in range(batch_size):
        chars, critical = build_bracket_stream(rng)
        ids = torch.tensor([stoi[ch] for ch in chars], dtype=torch.long)
        crit = torch.tensor(critical, dtype=torch.bool)
        xs.append(ids[:-1])
        ys.append(ids[1:])
        masks.append(crit[1:])
    return Batch(
        torch.stack(xs).to(DEVICE),
        torch.stack(ys).to(DEVICE),
        torch.stack(masks).to(DEVICE),
    )


def make_fixed_bracket_batches(stoi: Dict[str, int], seed: int, num_batches: int, batch_size: int) -> List[Batch]:
    rng = random.Random(seed)
    return [make_bracket_batch(rng, stoi, batch_size) for _ in range(num_batches)]


def bracket_train_batch_for_step(stoi: Dict[str, int], seed: int, step: int, batch_size: int = BATCH_SIZE) -> Batch:
    rng = random.Random(seed * 1_000_003 + step)
    return make_bracket_batch(rng, stoi, batch_size)


def replay_batch_for_index(stoi: Dict[str, int], seed: int, replay_index: int, batch_size: int = BATCH_SIZE) -> Batch:
    rng = random.Random(seed * 2_000_003 + replay_index)
    return make_bracket_batch(rng, stoi, batch_size)


class CausalSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        head_dim = channels // N_HEAD
        q = q.view(batch, seq_len, N_HEAD, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, N_HEAD, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, N_HEAD, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=DROPOUT if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.fc2 = nn.Linear(4 * D_MODEL, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_activations: bool = False,
    ):
        _, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device)
        h = self.token_embedding(x) + self.position_embedding(pos)[None, :, :]
        activations: List[torch.Tensor] = []
        for block in self.blocks:
            h = block(h)
            if return_activations:
                h.retain_grad()
                activations.append(h)
        logits = self.head(self.ln_f(h))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if return_activations:
            return logits, loss, activations
        return logits, loss


def block_module_keys() -> List[str]:
    return [f"b{i}" for i in range(N_LAYER)]


def parameter_groups(model: TinyGPT) -> Dict[str, List[nn.Parameter]]:
    groups: Dict[str, List[nn.Parameter]] = {
        "embed": list(model.token_embedding.parameters()) + list(model.position_embedding.parameters()),
        "head": list(model.ln_f.parameters()) + list(model.head.parameters()),
    }
    for i, block in enumerate(model.blocks):
        groups[f"b{i}"] = list(block.parameters())
        groups[f"b{i}.attn"] = list(block.attn.parameters())
        groups[f"b{i}.mlp"] = list(block.mlp.parameters())
        groups[f"b{i}.norm"] = list(block.ln1.parameters()) + list(block.ln2.parameters())
    return groups


def make_optimizer(model: TinyGPT) -> torch.optim.Optimizer:
    groups = [{"params": list(model.token_embedding.parameters()) + list(model.position_embedding.parameters())
               + list(model.ln_f.parameters()) + list(model.head.parameters()), "lr": BASE_LR, "name": "other"}]
    for i, block in enumerate(model.blocks):
        groups.append({"params": list(block.parameters()), "lr": BASE_LR, "name": f"b{i}"})
    return torch.optim.AdamW(groups, lr=BASE_LR, betas=BETAS, weight_decay=WEIGHT_DECAY)


def set_optimizer_lrs(optimizer: torch.optim.Optimizer, shield_modules: Sequence[str]) -> None:
    shield = set(shield_modules)
    for group in optimizer.param_groups:
        name = str(group.get("name", "other"))
        group["lr"] = BASE_LR * SHIELD_LR_MULTIPLIER if name in shield else BASE_LR


def verify_shield_groups(optimizer: torch.optim.Optimizer, shield_modules: Sequence[str]) -> bool:
    shield = set(shield_modules)
    for group in optimizer.param_groups:
        name = str(group.get("name", "other"))
        expected = BASE_LR * SHIELD_LR_MULTIPLIER if name in shield else BASE_LR
        if abs(float(group["lr"]) - expected) > 1e-12:
            return False
    return True


def norm_pair(params: Iterable[nn.Parameter]) -> Tuple[float, float]:
    grad_sq = 0.0
    weight_sq = 0.0
    for p in params:
        weight_sq += p.detach().float().norm(2).item() ** 2
        if p.grad is not None:
            grad_sq += p.grad.detach().float().norm(2).item() ** 2
    return grad_sq ** 0.5, weight_sq ** 0.5


def z_from_norms(loss_value: float, grad_norm: float, weight_norm: float) -> float:
    if grad_norm <= 1e-12 or weight_norm <= 1e-12:
        return 1_000.0
    z = abs(float(loss_value)) / (grad_norm * weight_norm + 1e-12)
    if not math.isfinite(z):
        return 1_000.0
    return min(z, 1_000.0)


def measure_z_map(model: TinyGPT, loss_value: float) -> Dict[str, float]:
    groups = parameter_groups(model)
    grad_norm, weight_norm = norm_pair(model.parameters())
    out = {"global": z_from_norms(loss_value, grad_norm, weight_norm)}
    for name, params in groups.items():
        g, w = norm_pair(params)
        out[name] = z_from_norms(loss_value, g, w)
    return out


def block_param_raw_map(z_map: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for block in block_module_keys():
        keys = [block, f"{block}.attn", f"{block}.mlp", f"{block}.norm"]
        logs = [math.log(max(z_map.get(key, 0.0), 1e-12)) for key in keys]
        out[block] = float(math.exp(float(np.median(np.asarray(logs, dtype=float)))))
    return out


def block_combined_z_map(z_map: Dict[str, float], act_z: Dict[str, float]) -> Dict[str, float]:
    param = block_param_raw_map(z_map)
    combined = {}
    for block in block_module_keys():
        combined[block] = math.exp(
            0.75 * math.log(max(param.get(block, 1e-12), 1e-12))
            + 0.25 * math.log(max(act_z.get(block, 1e-12), 1e-12))
        )
    return combined


def top_blocks(scores: Dict[str, float], k: int = Z_FRONTIER_K) -> List[str]:
    return sorted(block_module_keys(), key=lambda block: (scores.get(block, float("-inf")), block), reverse=True)[:k]


@torch.no_grad()
def evaluate_text(model: TinyGPT, val_data: torch.Tensor, eval_positions: torch.Tensor) -> Dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    for starts in eval_positions:
        batch = text_batch_from_positions(val_data, starts)
        logits, loss = model(batch.x, batch.y)
        losses.append(float(loss.item()))
        preds = logits.argmax(dim=-1)
        correct += int((preds == batch.y).sum().item())
        total += int(batch.y.numel())
    model.train()
    return {"loss": float(np.mean(losses)), "acc": correct / max(total, 1)}


@torch.no_grad()
def evaluate_bracket(model: TinyGPT, batches: List[Batch]) -> Dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    close_correct = 0
    close_total = 0
    seq_correct = 0
    seq_total = 0
    for batch in batches:
        logits, loss = model(batch.x, batch.y)
        losses.append(float(loss.item()))
        preds = logits.argmax(dim=-1)
        correct += int((preds == batch.y).sum().item())
        total += int(batch.y.numel())
        assert batch.critical_mask is not None
        close_hits = (preds == batch.y) & batch.critical_mask
        close_correct += int(close_hits.sum().item())
        close_total += int(batch.critical_mask.sum().item())
        seq_ok = ((preds == batch.y) | (~batch.critical_mask)).all(dim=1)
        seq_correct += int(seq_ok.sum().item())
        seq_total += int(seq_ok.numel())
    model.train()
    return {
        "loss": float(np.mean(losses)),
        "acc": correct / max(total, 1),
        "close_acc": close_correct / max(close_total, 1),
        "seq_acc": seq_correct / max(seq_total, 1),
    }


def probe_z(model: TinyGPT, batch: Batch) -> Tuple[Dict[str, float], Dict[str, float]]:
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    _, loss, activations = model(batch.x, batch.y, return_activations=True)
    loss.backward()
    loss_value = float(loss.item())
    z_map = measure_z_map(model, loss_value)
    act_z = {}
    for index, activation in enumerate(activations):
        grad_norm = 0.0 if activation.grad is None else activation.grad.detach().float().norm(2).item()
        activation_norm = activation.detach().float().norm(2).item()
        act_z[f"b{index}"] = z_from_norms(loss_value, grad_norm, activation_norm)
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return z_map, act_z


def train_one_step(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    shield_modules: Sequence[str] = (),
) -> float:
    model.train()
    set_optimizer_lrs(optimizer, shield_modules)
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(batch.x, batch.y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    return float(loss.item())


def tensor_tree_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: tensor_tree_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensor_tree_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(tensor_tree_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def make_checkpoint(model: TinyGPT, optimizer: torch.optim.Optimizer) -> Dict[str, object]:
    return {
        "model": tensor_tree_to_cpu(model.state_dict()),
        "optimizer": tensor_tree_to_cpu(optimizer.state_dict()),
    }


def restore_model_from_checkpoint(vocab_size: int, checkpoint: Dict[str, object]) -> Tuple[TinyGPT, torch.optim.Optimizer]:
    model = TinyGPT(vocab_size).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer = make_optimizer(model)
    optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer_to_device(optimizer, DEVICE)
    return model, optimizer


def checkpoint_signature(checkpoint: Dict[str, object]) -> float:
    total = 0.0
    state = checkpoint["model"]
    assert isinstance(state, dict)
    for value in state.values():
        if torch.is_tensor(value):
            total += float(value.float().sum().item())
    return total


def is_probe_step(step: int) -> bool:
    if step == 0 or step == PHASE_B_STEPS:
        return True
    if step <= 400:
        return step % 25 == 0
    return step % 50 == 0


def auc_rows(rows: List[BranchRow], attr: str) -> float:
    if len(rows) < 2:
        return 0.0
    area = 0.0
    for left, right in zip(rows, rows[1:]):
        area += 0.5 * (getattr(left, attr) + getattr(right, attr)) * (right.step - left.step)
    width = max(rows[-1].step - rows[0].step, 1)
    return area / width


def forgetting_area(rows: List[BranchRow], anchor_seq: float) -> float:
    if len(rows) < 2:
        return 0.0
    area = 0.0
    for left, right in zip(rows, rows[1:]):
        y0 = max(anchor_seq - left.old_seq_acc, 0.0)
        y1 = max(anchor_seq - right.old_seq_acc, 0.0)
        area += 0.5 * (y0 + y1) * (right.step - left.step)
    width = max(rows[-1].step - rows[0].step, 1)
    return area / width


def first_forgetting_step(rows: List[BranchRow], anchor_seq: float) -> float:
    threshold = anchor_seq - FORGET_DROP
    for row in rows:
        if row.old_seq_acc <= threshold:
            return float(row.step)
    return float("nan")


def first_z_warning_step(rows: List[BranchRow]) -> float:
    for row in rows:
        if row.z_warning:
            return float(row.step)
    return float("nan")


def z_warning_precision(rows: List[BranchRow], forgetting_step: float) -> float:
    warnings = [row for row in rows if row.z_warning]
    if not warnings:
        return float("nan")
    hits = 0
    for row in warnings:
        if math.isfinite(forgetting_step) and 0 < forgetting_step - row.step <= FORGETTING_PREDICTION_WINDOW:
            hits += 1
    return hits / len(warnings)


def auroc(scores: List[float], labels: List[int]) -> float:
    pairs = [(s, y) for s, y in zip(scores, labels) if math.isfinite(s)]
    positives = [s for s, y in pairs if y == 1]
    negatives = [s for s, y in pairs if y == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def future_forgetting_labels(rows: List[BranchRow], forgetting_step: float) -> List[int]:
    labels = []
    for row in rows:
        labels.append(
            int(math.isfinite(forgetting_step) and 0 < forgetting_step - row.step <= FORGETTING_PREDICTION_WINDOW)
        )
    return labels


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    clean = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if clean.size == 0:
        return float("nan"), float("nan")
    return float(clean.mean()), float(clean.std(ddof=0))


def median_value(values: Sequence[float]) -> float:
    clean = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if clean.size == 0:
        return float("nan")
    return float(np.median(clean))


def bootstrap_ci(values: Sequence[float], samples: int = BOOTSTRAP_SAMPLES) -> Tuple[float, float]:
    clean = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(12345)
    means = []
    for _ in range(samples):
        sample = rng.choice(clean, size=clean.size, replace=True)
        means.append(float(sample.mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def format_ci(lo: float, hi: float, digits: int = 3) -> str:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return "[nan, nan]"
    return f"[{lo:+.{digits}f}, {hi:+.{digits}f}]"


def schedule_replay_burst(current_step: int, schedule: set[int], max_replays: int) -> None:
    for offset in REPLAY_BURST_OFFSETS:
        if len(schedule) >= max_replays:
            break
        future = current_step + offset
        if 1 <= future <= PHASE_B_STEPS:
            schedule.add(future)


def make_random_replay_schedule(seed: int, count: int) -> set[int]:
    if count <= 0:
        return set()
    rng = random.Random(seed + 555_555)
    count = min(count, PHASE_B_STEPS)
    return set(rng.sample(range(1, PHASE_B_STEPS + 1), count))


def branch_uses_replay(branch_name: str) -> bool:
    return branch_name in {
        "periodic_replay_10pct",
        "random_matched_replay",
        "loss_triggered_replay",
        "accuracy_triggered_replay",
        "z_replay",
        "z_replay_shield",
    }


def branch_uses_shield(branch_name: str) -> bool:
    return branch_name in {"z_shield", "z_replay_shield"}


def summarize_branch(name: str, rows: List[BranchRow], anchor_seq: float, replay_count: int, replay_budget: int,
                     replay_steps: List[int], shield_steps: int, verified_shield_groups: bool) -> BranchResult:
    old_seq_auc = auc_rows(rows, "old_seq_acc")
    old_close_auc = auc_rows(rows, "old_close_acc")
    text_loss_auc = auc_rows(rows, "text_loss")
    text_acc_auc = auc_rows(rows, "text_acc")
    f_step = first_forgetting_step(rows, anchor_seq)
    z_step = first_z_warning_step(rows)
    z_lead = f_step - z_step if math.isfinite(f_step) and math.isfinite(z_step) else float("nan")
    labels = future_forgetting_labels(rows, f_step)
    final = rows[-1]
    return BranchResult(
        name=name,
        rows=rows,
        old_seq_auc=old_seq_auc,
        old_close_auc=old_close_auc,
        text_loss_auc=text_loss_auc,
        text_acc_auc=text_acc_auc,
        forgetting_area=forgetting_area(rows, anchor_seq),
        forgetting_step=f_step,
        first_z_warning_step=z_step,
        z_lead_time=z_lead,
        z_precision_150=z_warning_precision(rows, f_step),
        auroc_z=auroc([row.z_shock for row in rows], labels),
        auroc_loss=auroc([row.loss_shock for row in rows], labels),
        auroc_accuracy=auroc([row.acc_drop for row in rows], labels),
        final_old_seq=final.old_seq_acc,
        final_old_close=final.old_close_acc,
        final_text_loss=final.text_loss,
        final_text_acc=final.text_acc,
        replay_count=replay_count,
        replay_budget=replay_budget,
        replay_steps=replay_steps,
        shield_steps=shield_steps,
        verified_replay_cap=replay_count <= replay_budget,
        verified_shield_groups=verified_shield_groups,
    )


def train_old_skill(
    vocab_size: int,
    stoi: Dict[str, int],
    seed: int,
    bracket_eval_batches: List[Batch],
    bracket_probe_batch: Batch,
) -> AnchorInfo:
    model = TinyGPT(vocab_size).to(DEVICE)
    optimizer = make_optimizer(model)
    start = time.time()

    initial = evaluate_bracket(model, bracket_eval_batches)
    print(
        f"[seed {seed}] old step=0000 close={initial['close_acc']:.3f} "
        f"seq={initial['seq_acc']:.3f} loss={initial['loss']:.4f}"
    )
    last_metrics = initial
    reached_ready = False
    step = 0
    for step in range(1, PHASE_A_MAX_STEPS + 1):
        batch = bracket_train_batch_for_step(stoi, seed, step)
        loss = train_one_step(model, optimizer, batch)
        if step % PHASE_A_EVAL_INTERVAL == 0 or step == PHASE_A_MAX_STEPS:
            last_metrics = evaluate_bracket(model, bracket_eval_batches)
            print(
                f"[seed {seed}] old step={step:04d}/{PHASE_A_MAX_STEPS} "
                f"train_loss={loss:.4f} close={last_metrics['close_acc']:.3f} "
                f"seq={last_metrics['seq_acc']:.3f}"
            )
            if last_metrics["seq_acc"] >= OLD_READY_SEQ:
                reached_ready = True
                break

    old_z, old_act_z = probe_z(model, bracket_probe_batch)
    block_anchor_z = block_combined_z_map(old_z, old_act_z)
    old_frontier = top_blocks(block_anchor_z)
    checkpoint = make_checkpoint(model, optimizer)
    print(
        f"[seed {seed}] old anchor step={step} close={last_metrics['close_acc']:.3f} "
        f"seq={last_metrics['seq_acc']:.3f} frontier={'+'.join(old_frontier)} "
        f"ready={reached_ready} time={format_seconds(time.time() - start)}"
    )
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return AnchorInfo(
        step=step,
        old_loss=last_metrics["loss"],
        old_close_acc=last_metrics["close_acc"],
        old_seq_acc=last_metrics["seq_acc"],
        old_z=old_z,
        old_act_z=old_act_z,
        block_anchor_z=block_anchor_z,
        old_frontier=old_frontier,
        checkpoint=checkpoint,
        reached_ready=reached_ready,
    )


def probe_branch_row(
    model: TinyGPT,
    val_data: torch.Tensor,
    text_eval_positions: torch.Tensor,
    bracket_eval_batches: List[Batch],
    bracket_probe_batch: Batch,
    anchor: AnchorInfo,
    step: int,
    z_history: List[bool],
    replay_count: int,
    shield_active: bool,
) -> BranchRow:
    old_metrics = evaluate_bracket(model, bracket_eval_batches)
    text_metrics = evaluate_text(model, val_data, text_eval_positions)
    old_z, old_act_z = probe_z(model, bracket_probe_batch)
    current_block_z = block_combined_z_map(old_z, old_act_z)
    shocks = []
    for block in anchor.old_frontier:
        current = max(current_block_z.get(block, 1e-12), 1e-12)
        base = max(anchor.block_anchor_z.get(block, 1e-12), 1e-12)
        shocks.append(abs(math.log2(current / base)))
    z_shock = float(np.mean(shocks)) if shocks else 0.0
    acc_drop = max(anchor.old_seq_acc - old_metrics["seq_acc"], 0.0)
    loss_shock = max((old_metrics["loss"] - anchor.old_loss) / max(abs(anchor.old_loss), 1e-12), 0.0)
    z_candidate = z_shock >= Z_SHOCK_THRESHOLD and acc_drop <= Z_WARNING_MAX_DROP
    z_history.append(z_candidate)
    z_warning = (
        len(z_history) >= Z_WARNING_PERSISTENCE
        and all(z_history[-Z_WARNING_PERSISTENCE:])
    )
    return BranchRow(
        step=step,
        old_loss=old_metrics["loss"],
        old_close_acc=old_metrics["close_acc"],
        old_seq_acc=old_metrics["seq_acc"],
        text_loss=text_metrics["loss"],
        text_acc=text_metrics["acc"],
        z_shock=z_shock,
        loss_shock=loss_shock,
        acc_drop=acc_drop,
        z_warning=z_warning,
        replay_count=replay_count,
        shield_active=shield_active,
    )


def run_branch(
    branch_name: str,
    anchor: AnchorInfo,
    vocab_size: int,
    stoi: Dict[str, int],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    text_train_positions: torch.Tensor,
    text_eval_positions: torch.Tensor,
    bracket_eval_batches: List[Batch],
    bracket_probe_batch: Batch,
    seed: int,
    fixed_replay_schedule: set[int] | None = None,
) -> BranchResult:
    model, optimizer = restore_model_from_checkpoint(vocab_size, anchor.checkpoint)
    rows: List[BranchRow] = []
    z_history: List[bool] = []
    dynamic_replay_schedule: set[int] = set()
    fixed_replay_schedule = fixed_replay_schedule or set()
    replay_budget = int(PHASE_B_STEPS * REPLAY_BUDGET_FRACTION)
    replay_count = 0
    replay_steps: List[int] = []
    shield_until = -1
    shield_steps = 0
    verified_shield_groups = True

    print(f"[branch:{branch_name}] start")

    row0 = probe_branch_row(
        model,
        val_data,
        text_eval_positions,
        bracket_eval_batches,
        bracket_probe_batch,
        anchor,
        0,
        z_history,
        replay_count,
        False,
    )
    rows.append(row0)
    if branch_name in {"z_replay", "z_replay_shield"} and row0.z_warning:
        schedule_replay_burst(0, dynamic_replay_schedule, replay_budget)
    if branch_uses_shield(branch_name) and row0.z_warning:
        shield_until = max(shield_until, SHIELD_COOLDOWN_STEPS)

    for step in range(1, PHASE_B_STEPS + 1):
        replay_this_step = False
        if branch_name == "periodic_replay_10pct" and step % 10 == 0 and replay_count < replay_budget:
            replay_this_step = True
        elif branch_name == "random_matched_replay" and step in fixed_replay_schedule and replay_count < replay_budget:
            replay_this_step = True
        elif step in dynamic_replay_schedule and replay_count < replay_budget:
            replay_this_step = True

        if replay_this_step:
            batch = replay_batch_for_index(stoi, seed, replay_count)
            replay_count += 1
            replay_steps.append(step)
        else:
            batch = text_batch_from_positions(train_data, text_train_positions[step - 1])

        shield_active = branch_uses_shield(branch_name) and step <= shield_until
        shield_modules = anchor.old_frontier if shield_active else []
        if shield_active:
            shield_steps += 1
        train_one_step(model, optimizer, batch, shield_modules)
        if shield_active:
            verified_shield_groups = verified_shield_groups and verify_shield_groups(optimizer, anchor.old_frontier)

        if is_probe_step(step):
            row = probe_branch_row(
                model,
                val_data,
                text_eval_positions,
                bracket_eval_batches,
                bracket_probe_batch,
                anchor,
                step,
                z_history,
                replay_count,
                shield_active,
            )
            rows.append(row)
            if branch_name in {"z_replay", "z_replay_shield"} and row.z_warning:
                schedule_replay_burst(step, dynamic_replay_schedule, replay_budget)
            if branch_name == "loss_triggered_replay" and row.loss_shock >= LOSS_TRIGGER_REL:
                schedule_replay_burst(step, dynamic_replay_schedule, replay_budget)
            if branch_name == "accuracy_triggered_replay" and row.acc_drop >= ACCURACY_TRIGGER_DROP:
                schedule_replay_burst(step, dynamic_replay_schedule, replay_budget)
            if branch_uses_shield(branch_name) and row.z_warning:
                shield_until = max(shield_until, step + SHIELD_COOLDOWN_STEPS)
            if step % BRANCH_LOG_INTERVAL == 0 or step == PHASE_B_STEPS:
                print(
                    f"[branch:{branch_name}] step={step:04d}/{PHASE_B_STEPS} "
                    f"old_seq={row.old_seq_acc:.3f} text_loss={row.text_loss:.3f} "
                    f"z_shock={row.z_shock:.2f} replay={replay_count}/{replay_budget} "
                    f"shield={'on' if shield_active else 'off'}"
                )

    set_optimizer_lrs(optimizer, [])
    result = summarize_branch(
        branch_name,
        rows,
        anchor.old_seq_acc,
        replay_count,
        replay_budget,
        replay_steps,
        shield_steps,
        verified_shield_groups,
    )
    print(
        f"[branch:{branch_name}] old_seq_auc={result.old_seq_auc:.3f} "
        f"final_old_seq={result.final_old_seq:.3f} text_loss_auc={result.text_loss_auc:.3f} "
        f"forget_step={format_step(result.forgetting_step)} z_warn={format_step(result.first_z_warning_step)} "
        f"lead={format_step(result.z_lead_time)} replay={result.replay_count}/{result.replay_budget}"
    )
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def branch_to_csv_row(seed: int, anchor: AnchorInfo, result: BranchResult) -> Dict[str, object]:
    return {
        "seed": seed,
        "branch": result.name,
        "old_ready": int(anchor.reached_ready),
        "anchor_step": anchor.step,
        "anchor_old_seq": anchor.old_seq_acc,
        "anchor_old_close": anchor.old_close_acc,
        "old_frontier": "+".join(anchor.old_frontier),
        "old_seq_auc": result.old_seq_auc,
        "old_close_auc": result.old_close_auc,
        "final_old_seq": result.final_old_seq,
        "final_old_close": result.final_old_close,
        "forgetting_area": result.forgetting_area,
        "forgetting_step": result.forgetting_step,
        "text_loss_auc": result.text_loss_auc,
        "text_acc_auc": result.text_acc_auc,
        "final_text_loss": result.final_text_loss,
        "final_text_acc": result.final_text_acc,
        "first_z_warning_step": result.first_z_warning_step,
        "z_lead_time": result.z_lead_time,
        "z_precision_150": result.z_precision_150,
        "auroc_z": result.auroc_z,
        "auroc_loss": result.auroc_loss,
        "auroc_accuracy": result.auroc_accuracy,
        "replay_count": result.replay_count,
        "replay_budget": result.replay_budget,
        "shield_steps": result.shield_steps,
        "verified_replay_cap": int(result.verified_replay_cap),
        "verified_shield_groups": int(result.verified_shield_groups),
    }


def print_seed_summary(seed: int, anchor: AnchorInfo, results: Dict[str, BranchResult]) -> None:
    print("\n" + "=" * 78)
    print(f"SEED {seed} FORGETTING-GUARD RESULT")
    print("=" * 78)
    print(
        f"Anchor: step={anchor.step} ready={anchor.reached_ready} "
        f"old_close={anchor.old_close_acc:.3f} old_seq={anchor.old_seq_acc:.3f} "
        f"frontier={'+'.join(anchor.old_frontier)}"
    )
    print(
        f"{'branch':26s} {'old_auc':>7s} {'final_old':>9s} {'forget':>6s} "
        f"{'z_warn':>6s} {'lead':>6s} {'text_loss':>9s} {'replay':>9s} {'shield':>6s}"
    )
    for name in BRANCH_NAMES:
        result = results[name]
        print(
            f"{name:26s} {result.old_seq_auc:7.3f} {result.final_old_seq:9.3f} "
            f"{format_step(result.forgetting_step):>6s} {format_step(result.first_z_warning_step):>6s} "
            f"{format_step(result.z_lead_time):>6s} {result.final_text_loss:9.3f} "
            f"{result.replay_count:3d}/{result.replay_budget:<5d} {result.shield_steps:6d}"
        )
    none = results["none"]
    periodic = results["periodic_replay_10pct"]
    random_matched = results["random_matched_replay"]
    z_replay = results["z_replay"]
    z_shield = results["z_shield"]
    z_combo = results["z_replay_shield"]
    gap = periodic.old_seq_auc - none.old_seq_auc
    recovered = (z_combo.old_seq_auc - none.old_seq_auc) / max(gap, 1e-12) if gap > 0 else float("nan")
    print(
        f"Z replay vs random old_seq_auc delta: {z_replay.old_seq_auc - random_matched.old_seq_auc:+.3f}"
    )
    print(
        f"Z shield vs none old_seq_auc delta: {z_shield.old_seq_auc - none.old_seq_auc:+.3f}"
    )
    print(
        f"Z replay+shield vs random old_seq_auc delta: {z_combo.old_seq_auc - random_matched.old_seq_auc:+.3f}"
    )
    print(f"Z replay+shield recovered {recovered * 100:.1f}% of none->periodic retention gap")
    print(
        f"Replay matching check: z_replay={z_replay.replay_count}, "
        f"random_matched={random_matched.replay_count}"
    )
    print("=" * 78)


def summarize_all(csv_rows: List[Dict[str, object]]) -> None:
    by_branch: Dict[str, List[Dict[str, object]]] = {name: [] for name in BRANCH_NAMES}
    for row in csv_rows:
        by_branch[str(row["branch"])].append(row)
    seeds = sorted({int(row["seed"]) for row in csv_rows})
    ready = [row for row in by_branch["none"] if int(row["old_ready"]) == 1]
    none_forgets = [
        row for row in by_branch["none"]
        if math.isfinite(float(row["forgetting_step"]))
    ]
    z_leads = [float(row["z_lead_time"]) for row in by_branch["z_replay"] if math.isfinite(float(row["z_lead_time"]))]
    z_before = [lead for lead in z_leads if lead > 0]

    print("\n" + "=" * 78)
    print("FORGETTING-GUARD SUMMARY ACROSS SEEDS")
    print("=" * 78)
    print(f"Seeds run: {len(seeds)}")
    print(f"Old bracket seq>=0.90 reached before text phase: {len(ready)}/{len(seeds)}")
    print(f"None branch forgetting observed: {len(none_forgets)}/{len(seeds)}")
    print(f"Z warning before forgetting in z_replay: {len(z_before)}/{len(seeds)}")
    print(f"Median Z lead time in z_replay: {format_step(median_value(z_leads))} steps")
    print("\nMean old_seq_auc by branch:")
    for name in BRANCH_NAMES:
        mean, std = mean_std([float(row["old_seq_auc"]) for row in by_branch[name]])
        print(f"  {name:26s} {format_mean_std(mean, std)}")

    def branch_values(branch: str, key: str) -> List[float]:
        return [float(row[key]) for row in by_branch[branch]]

    z_combo_gain_random = [
        z - r for z, r in zip(branch_values("z_replay_shield", "old_seq_auc"),
                             branch_values("random_matched_replay", "old_seq_auc"))
    ]
    z_replay_gain_random = [
        z - r for z, r in zip(branch_values("z_replay", "old_seq_auc"),
                             branch_values("random_matched_replay", "old_seq_auc"))
    ]
    z_shield_gain_none = [
        z - n for z, n in zip(branch_values("z_shield", "old_seq_auc"),
                             branch_values("none", "old_seq_auc"))
    ]
    text_loss_slowdown = [
        (z - r) / max(abs(r), 1e-12)
        for z, r in zip(branch_values("z_replay_shield", "final_text_loss"),
                        branch_values("random_matched_replay", "final_text_loss"))
    ]
    combo_mean, combo_std = mean_std(z_combo_gain_random)
    combo_ci = bootstrap_ci(z_combo_gain_random)
    replay_mean, replay_std = mean_std(z_replay_gain_random)
    shield_mean, shield_std = mean_std(z_shield_gain_none)
    slowdown_mean, slowdown_std = mean_std(text_loss_slowdown)
    print("\nPrimary Z-vs-control effects:")
    print(f"  z_replay_shield old_seq_auc gain over random_matched: {format_mean_std(combo_mean, combo_std)}")
    print(f"  bootstrap 95% CI: {format_ci(*combo_ci)}")
    print(f"  z_replay old_seq_auc gain over random_matched: {format_mean_std(replay_mean, replay_std)}")
    print(f"  z_shield old_seq_auc gain over none: {format_mean_std(shield_mean, shield_std)}")
    print(f"  z_replay_shield final_text_loss slowdown vs random: {format_mean_std(slowdown_mean, slowdown_std)}")

    auroc_z_mean, auroc_z_std = mean_std(branch_values("z_replay", "auroc_z"))
    auroc_loss_mean, auroc_loss_std = mean_std(branch_values("z_replay", "auroc_loss"))
    auroc_acc_mean, auroc_acc_std = mean_std(branch_values("z_replay", "auroc_accuracy"))
    print("\nEarly-warning AUROC on z_replay probes:")
    print(f"  Z shock:       {format_mean_std(auroc_z_mean, auroc_z_std)}")
    print(f"  loss shock:    {format_mean_std(auroc_loss_mean, auroc_loss_std)}")
    print(f"  accuracy drop: {format_mean_std(auroc_acc_mean, auroc_acc_std)}")

    if len(ready) >= 6 and len(none_forgets) >= 6 and len(z_before) >= 6 and combo_ci[0] > 0 and slowdown_mean <= 0.05:
        interpretation = "strong pass: Z warning and Z-guided protection beat matched controls."
    elif len(z_before) >= max(1, len(seeds) // 2) and combo_mean > 0:
        interpretation = "mixed but promising: positive Z-guided retention signal, not yet decisive."
    else:
        interpretation = "fail/diagnostic: Z-guided forgetting guard did not beat controls cleanly."
    print(f"\nInterpretation: {interpretation}")
    print(f"CSV saved to: {CSV_PATH}")
    print("=" * 78)


def write_csv(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_seed(
    seed: int,
    vocab_size: int,
    stoi: Dict[str, int],
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> List[Dict[str, object]]:
    print("\n" + "#" * 78)
    print(f"BEGIN FORGETTING-GUARD SEED {seed}")
    print("#" * 78)
    set_seed(seed)
    text_train_positions, text_eval_positions = make_text_positions(len(train_data), len(val_data), seed)
    bracket_eval_batches = make_fixed_bracket_batches(
        stoi, seed + 30_000, BRACKET_EVAL_BATCHES, BRACKET_EVAL_BATCH
    )
    bracket_probe_batch = make_fixed_bracket_batches(stoi, seed + 40_000, 1, PROBE_BATCH)[0]
    anchor = train_old_skill(vocab_size, stoi, seed, bracket_eval_batches, bracket_probe_batch)
    checkpoint_sig = checkpoint_signature(anchor.checkpoint)

    results: Dict[str, BranchResult] = {}
    run_order = [
        "none",
        "periodic_replay_10pct",
        "z_replay",
        "random_matched_replay",
        "loss_triggered_replay",
        "accuracy_triggered_replay",
        "z_shield",
        "z_replay_shield",
    ]
    random_schedule: set[int] | None = None
    for branch_name in run_order:
        fixed_schedule = random_schedule if branch_name == "random_matched_replay" else None
        result = run_branch(
            branch_name,
            anchor,
            vocab_size,
            stoi,
            train_data,
            val_data,
            text_train_positions,
            text_eval_positions,
            bracket_eval_batches,
            bracket_probe_batch,
            seed,
            fixed_schedule,
        )
        results[branch_name] = result
        if branch_name == "z_replay":
            random_schedule = make_random_replay_schedule(seed, result.replay_count)
        if SMOKE:
            after_sig = checkpoint_signature(anchor.checkpoint)
            assert abs(after_sig - checkpoint_sig) < 1e-5, "branch mutated shared checkpoint"

    if SMOKE:
        assert set(results) == set(BRANCH_NAMES), "not all branches executed"
        assert results["z_replay"].replay_count > 0, "smoke expected z_replay schedule"
        assert results["random_matched_replay"].replay_count == results["z_replay"].replay_count, (
            "random replay did not match z_replay count"
        )
        assert results["z_shield"].shield_steps > 0 or results["z_replay_shield"].shield_steps > 0, (
            "smoke expected shield to activate"
        )
        assert results["z_shield"].verified_shield_groups and results["z_replay_shield"].verified_shield_groups, (
            "shield LR verification failed"
        )
        assert all(result.verified_replay_cap for result in results.values()), "replay cap exceeded"

    print_seed_summary(seed, anchor, results)
    return [branch_to_csv_row(seed, anchor, results[name]) for name in BRANCH_NAMES]


def main() -> None:
    set_seed(SEEDS[0])
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 78)
    print("Z-GUIDED CONTINUAL LEARNING: FORGETTING GUARD BENCHMARK")
    print("=" * 78)
    print("Question: does old-skill Z shock predict and prevent catastrophic forgetting?")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Phase A: bracket until seq>={OLD_READY_SEQ:.2f} or {PHASE_A_MAX_STEPS} steps")
    print(f"Phase B: text training for {PHASE_B_STEPS} steps")
    print(f"Model: d={D_MODEL}, layers={N_LAYER}, heads={N_HEAD}, block={BLOCK_SIZE}")
    print(f"Bracket task: {len(OPENERS)} bracket types, depth=[{BRACKET_MIN_DEPTH}, {BRACKET_MAX_DEPTH}]")
    print(
        f"Z warning: shock>={Z_SHOCK_THRESHOLD:.2f}, persistence={Z_WARNING_PERSISTENCE}, "
        f"old_seq_drop<={Z_WARNING_MAX_DROP:.2f}"
    )
    print(
        f"Replay cap={REPLAY_BUDGET_FRACTION:.0%}, burst_offsets={REPLAY_BURST_OFFSETS}, "
        f"shield_lr={SHIELD_LR_MULTIPLIER:.2f}, shield_cooldown={SHIELD_COOLDOWN_STEPS}"
    )
    print(f"Branches: {BRANCH_NAMES}")

    text = download_or_load_text()
    stoi, _ = build_vocab(text)
    encoded = encode(text, stoi)
    split = int(len(encoded) * TRAIN_FRACTION)
    train_data = encoded[:split]
    val_data = encoded[split:]
    print(f"Vocab size: {len(stoi)}")
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")
    print("All branches replay identical text batches by absolute step and old replay batches by replay index.")

    all_rows: List[Dict[str, object]] = []
    start = time.time()
    for seed in SEEDS:
        all_rows.extend(run_seed(seed, len(stoi), stoi, train_data, val_data))
        write_csv(all_rows)
    summarize_all(all_rows)
    write_csv(all_rows)
    print(f"Total wall time: {format_seconds(time.time() - start)}")


if __name__ == "__main__":
    main()
