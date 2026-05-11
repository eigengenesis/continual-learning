#!/usr/bin/env python3
"""
Water Weights continual-learning benchmark.

Run:
    python colab_water_weights_benchmark.py

Hidden smoke mode:
    CHAOS_SMOKE=1 python colab_water_weights_benchmark.py

Core question:
Can old-skill gradient anchors, solid-base latent adapters, and latent
free-space routing plus prophylactic Z-guided viscosity, old-data reminiscence,
and bounded replay preserve an old bracket-stack skill while learning Tiny
Shakespeare text?
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
CSV_PATH = ROOT / "water_weights_results.csv"

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
ADAPTER_LR_MULT = 4.0
REMINISCENCE_LR = 3e-5
WEIGHT_DECAY = 0.08
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0

OLD_READY_SEQ = 0.90
FORGET_DROP = 0.15
Z_FRONTIER_K = 2
STATIC_VISCOSITY = 0.05
INITIAL_Z_VISCOSITY = 0.05
REPLAY_BUDGET_FRACTION = 0.12
GRAD_ANCHOR_BATCHES = 24
GRAD_ANCHOR_RANK = 4
INITIAL_GRAD_PROJECTION = 0.75
LATENT_ANCHOR_BATCHES = 12
LATENT_ACT_RANK = 32
LATENT_GRAD_RANK = 16
INITIAL_LATENT_PROJECTION = 0.75
BRACKET_MIN_DEPTH = 5
BRACKET_MAX_DEPTH = 18
BOOTSTRAP_SAMPLES = 5000

BRANCH_NAMES = [
    "none",
    "water_weights_gradient_full",
    "water_weights_latent_free",
    "water_weights_latent_gradient_full",
    "water_weights_latent_strong",
    "water_weights_latent_adapter_only",
]


if torch.cuda.is_available():
    DEVICE = "cuda"
    PHASE_A_MAX_STEPS = 1200
    PHASE_B_STEPS = 1200
    REMINISCENCE_STEPS = 60
    PHASE_A_EVAL_INTERVAL = 50
    BRANCH_LOG_INTERVAL = 200
    BLOCK_SIZE = 160
    BATCH_SIZE = 32
    D_MODEL = 256
    N_HEAD = 8
    N_LAYER = 6
    ADAPTER_RANK = 32
    DROPOUT = 0.10
    TEXT_EVAL_BATCHES = 12
    TEXT_EVAL_BATCH = 24
    BRACKET_EVAL_BATCHES = 12
    BRACKET_EVAL_BATCH = 64
    PROBE_BATCH = 24
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    PHASE_A_MAX_STEPS = 800
    PHASE_B_STEPS = 600
    REMINISCENCE_STEPS = 40
    PHASE_A_EVAL_INTERVAL = 40
    BRANCH_LOG_INTERVAL = 100
    BLOCK_SIZE = 128
    BATCH_SIZE = 16
    D_MODEL = 192
    N_HEAD = 6
    N_LAYER = 4
    ADAPTER_RANK = 24
    DROPOUT = 0.10
    TEXT_EVAL_BATCHES = 6
    TEXT_EVAL_BATCH = 16
    BRACKET_EVAL_BATCHES = 6
    BRACKET_EVAL_BATCH = 32
    PROBE_BATCH = 16
    GRAD_ANCHOR_BATCHES = 12
    GRAD_ANCHOR_RANK = 4
    LATENT_ANCHOR_BATCHES = 8
    LATENT_ACT_RANK = 24
    LATENT_GRAD_RANK = 12
else:
    DEVICE = "cpu"
    PHASE_A_MAX_STEPS = 220
    PHASE_B_STEPS = 220
    REMINISCENCE_STEPS = 20
    PHASE_A_EVAL_INTERVAL = 25
    BRANCH_LOG_INTERVAL = 50
    BLOCK_SIZE = 96
    BATCH_SIZE = 12
    D_MODEL = 96
    N_HEAD = 4
    N_LAYER = 3
    ADAPTER_RANK = 16
    DROPOUT = 0.10
    TEXT_EVAL_BATCHES = 4
    TEXT_EVAL_BATCH = 12
    BRACKET_EVAL_BATCHES = 4
    BRACKET_EVAL_BATCH = 32
    PROBE_BATCH = 12
    GRAD_ANCHOR_BATCHES = 8
    GRAD_ANCHOR_RANK = 3
    LATENT_ANCHOR_BATCHES = 6
    LATENT_ACT_RANK = 12
    LATENT_GRAD_RANK = 8


SMOKE = os.environ.get("CHAOS_SMOKE", "0") == "1"
if SMOKE:
    SEEDS = [1337]
    PHASE_A_MAX_STEPS = min(PHASE_A_MAX_STEPS, 30)
    PHASE_B_STEPS = min(PHASE_B_STEPS, 44)
    REMINISCENCE_STEPS = min(REMINISCENCE_STEPS, 6)
    PHASE_A_EVAL_INTERVAL = min(PHASE_A_EVAL_INTERVAL, 6)
    BRANCH_LOG_INTERVAL = min(BRANCH_LOG_INTERVAL, 12)
    BLOCK_SIZE = min(BLOCK_SIZE, 64)
    BATCH_SIZE = min(BATCH_SIZE, 6)
    D_MODEL = min(D_MODEL, 96)
    N_HEAD = 4
    N_LAYER = min(N_LAYER, 3)
    ADAPTER_RANK = min(ADAPTER_RANK, 12)
    TEXT_EVAL_BATCHES = min(TEXT_EVAL_BATCHES, 2)
    TEXT_EVAL_BATCH = min(TEXT_EVAL_BATCH, 6)
    BRACKET_EVAL_BATCHES = min(BRACKET_EVAL_BATCHES, 2)
    BRACKET_EVAL_BATCH = min(BRACKET_EVAL_BATCH, 8)
    PROBE_BATCH = min(PROBE_BATCH, 6)
    BRACKET_MIN_DEPTH = 2
    BRACKET_MAX_DEPTH = 6
    GRAD_ANCHOR_BATCHES = min(GRAD_ANCHOR_BATCHES, 3)
    GRAD_ANCHOR_RANK = min(GRAD_ANCHOR_RANK, 2)
    LATENT_ANCHOR_BATCHES = min(LATENT_ANCHOR_BATCHES, 2)
    LATENT_ACT_RANK = min(LATENT_ACT_RANK, 4)
    LATENT_GRAD_RANK = min(LATENT_GRAD_RANK, 3)


EARLY_PROBE_STEPS = {0, 1, 2, 5, 10, 15, 20, 25}


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
    grad_basis: Dict[str, torch.Tensor]
    latent_free_projectors: Dict[str, torch.Tensor]
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
    viscosity: float
    replay_count: int
    adapter_enabled: bool
    grad_projection: float
    latent_projection: float


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
    final_old_seq: float
    final_old_close: float
    final_text_loss: float
    final_text_acc: float
    replay_count: int
    replay_budget: int
    viscosity_steps: int
    grad_projection_steps: int
    latent_projection_steps: int
    reminiscence_steps: int
    adapter_enabled: bool
    verified_replay_cap: bool


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


def build_bracket_stream(rng: random.Random) -> Tuple[List[str], List[bool]]:
    chars: List[str] = []
    critical: List[bool] = []
    while len(chars) < BLOCK_SIZE + 1:
        depth = rng.randint(BRACKET_MIN_DEPTH, BRACKET_MAX_DEPTH)
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


class Adapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down = nn.Linear(D_MODEL, ADAPTER_RANK, bias=False)
        self.up = nn.Linear(ADAPTER_RANK, D_MODEL, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(x)))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = MLP()
        self.adapter = Adapter()
        self.adapter_enabled = True
        self.latent_free_projector: torch.Tensor | None = None
        self.latent_projection_strength = 0.0

    def base_parameters(self) -> List[nn.Parameter]:
        return (
            list(self.ln1.parameters())
            + list(self.attn.parameters())
            + list(self.ln2.parameters())
            + list(self.mlp.parameters())
        )

    def adapter_parameters(self) -> List[nn.Parameter]:
        return list(self.adapter.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        if self.adapter_enabled:
            delta = self.adapter(x)
            if self.latent_free_projector is not None and self.latent_projection_strength > 0.0:
                projector = self.latent_free_projector.to(device=delta.device, dtype=delta.dtype)
                free_delta = torch.matmul(delta, projector)
                strength = min(max(float(self.latent_projection_strength), 0.0), 1.0)
                delta = strength * free_delta + (1.0 - strength) * delta
            x = x + delta
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def set_adapters_enabled(self, enabled: bool) -> None:
        for block in self.blocks:
            block.adapter_enabled = enabled

    def set_latent_free_projectors(self, projectors: Dict[str, torch.Tensor], strength: float) -> None:
        for index, block in enumerate(self.blocks):
            block_name = f"b{index}"
            block.latent_free_projector = projectors.get(block_name)
            block.latent_projection_strength = strength if block.latent_free_projector is not None else 0.0

    def clear_latent_free_projectors(self) -> None:
        for block in self.blocks:
            block.latent_free_projector = None
            block.latent_projection_strength = 0.0

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


def block_keys() -> List[str]:
    return [f"b{i}" for i in range(N_LAYER)]


def adapter_keys() -> List[str]:
    return [f"a{i}" for i in range(N_LAYER)]


def base_block_params(model: TinyGPT, block_name: str) -> List[nn.Parameter]:
    index = int(block_name[1:])
    return model.blocks[index].base_parameters()


def adapter_params(model: TinyGPT, adapter_name: str) -> List[nn.Parameter]:
    index = int(adapter_name[1:])
    return model.blocks[index].adapter_parameters()


def all_base_params(model: TinyGPT) -> List[nn.Parameter]:
    params: List[nn.Parameter] = (
        list(model.token_embedding.parameters())
        + list(model.position_embedding.parameters())
        + list(model.ln_f.parameters())
        + list(model.head.parameters())
    )
    for block in block_keys():
        params += base_block_params(model, block)
    return params


def all_adapter_params(model: TinyGPT) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for name in adapter_keys():
        params += adapter_params(model, name)
    return params


def parameter_groups(model: TinyGPT) -> Dict[str, List[nn.Parameter]]:
    groups: Dict[str, List[nn.Parameter]] = {
        "embed": list(model.token_embedding.parameters()) + list(model.position_embedding.parameters()),
        "head": list(model.ln_f.parameters()) + list(model.head.parameters()),
    }
    for i, block in enumerate(model.blocks):
        groups[f"b{i}"] = block.base_parameters()
        groups[f"b{i}.attn"] = list(block.attn.parameters())
        groups[f"b{i}.mlp"] = list(block.mlp.parameters())
        groups[f"b{i}.norm"] = list(block.ln1.parameters()) + list(block.ln2.parameters())
        groups[f"a{i}"] = block.adapter_parameters()
    return groups


def make_optimizer(model: TinyGPT, base_lr: float = BASE_LR) -> torch.optim.Optimizer:
    groups = [
        {
            "params": list(model.token_embedding.parameters()) + list(model.position_embedding.parameters()),
            "lr": base_lr,
            "name": "embed",
        },
        {"params": list(model.ln_f.parameters()) + list(model.head.parameters()), "lr": base_lr, "name": "head"},
    ]
    for i, block in enumerate(model.blocks):
        groups.append({"params": block.base_parameters(), "lr": base_lr, "name": f"b{i}"})
        groups.append({"params": block.adapter_parameters(), "lr": base_lr * ADAPTER_LR_MULT, "name": f"a{i}"})
    return torch.optim.AdamW(groups, lr=base_lr, betas=BETAS, weight_decay=WEIGHT_DECAY)


def set_optimizer_lrs(optimizer: torch.optim.Optimizer, base_lr: float, adapter_fluid: bool) -> None:
    for group in optimizer.param_groups:
        name = str(group.get("name", ""))
        if name.startswith("a") and adapter_fluid:
            group["lr"] = base_lr * ADAPTER_LR_MULT
        else:
            group["lr"] = base_lr


def set_requires_grad(params: Iterable[nn.Parameter], enabled: bool) -> None:
    for param in params:
        param.requires_grad = enabled


def configure_branch_trainability(model: TinyGPT, branch_name: str, frontier: Sequence[str]) -> None:
    for param in model.parameters():
        param.requires_grad = True
    set_requires_grad(all_adapter_params(model), False)
    model.set_adapters_enabled(False)
    model.clear_latent_free_projectors()

    if branch_name in {
        "water_weights_gradient_full",
        "water_weights_latent_free",
        "water_weights_latent_gradient_full",
        "water_weights_latent_strong",
        "water_weights_latent_adapter_only",
    }:
        set_requires_grad(all_adapter_params(model), True)
        model.set_adapters_enabled(True)

    if branch_name == "water_weights_latent_adapter_only":
        set_requires_grad(all_base_params(model), False)


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
    for block in block_keys():
        keys = [block, f"{block}.attn", f"{block}.mlp", f"{block}.norm"]
        logs = [math.log(max(z_map.get(key, 0.0), 1e-12)) for key in keys]
        out[block] = float(math.exp(float(np.median(np.asarray(logs, dtype=float)))))
    return out


def block_combined_z_map(z_map: Dict[str, float], act_z: Dict[str, float]) -> Dict[str, float]:
    param = block_param_raw_map(z_map)
    combined = {}
    for block in block_keys():
        combined[block] = math.exp(
            0.75 * math.log(max(param.get(block, 1e-12), 1e-12))
            + 0.25 * math.log(max(act_z.get(block, 1e-12), 1e-12))
        )
    return combined


def top_blocks(scores: Dict[str, float], k: int = Z_FRONTIER_K) -> List[str]:
    return sorted(block_keys(), key=lambda block: (scores.get(block, float("-inf")), block), reverse=True)[:k]


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


def multiply_grads(params: Iterable[nn.Parameter], multiplier: float) -> None:
    for param in params:
        if param.grad is not None:
            param.grad.mul_(multiplier)


def flatten_grads(params: Sequence[nn.Parameter]) -> torch.Tensor:
    pieces = []
    for param in params:
        if param.grad is None:
            pieces.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            pieces.append(param.grad.reshape(-1))
    return torch.cat(pieces)


def assign_flat_grads(params: Sequence[nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for param in params:
        width = param.numel()
        if param.grad is not None:
            param.grad.copy_(flat[offset: offset + width].view_as(param))
        offset += width


def project_gradient_away(flat: torch.Tensor, basis: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0.0 or basis.numel() == 0:
        return flat
    basis = basis.to(device=flat.device, dtype=flat.dtype)
    coeff = torch.mv(basis, flat)
    projected = torch.mv(basis.t(), coeff)
    return flat - min(max(strength, 0.0), 1.0) * projected


def project_block_gradients(
    model: TinyGPT,
    basis_by_block: Dict[str, torch.Tensor],
    projection_strength: float,
) -> None:
    if projection_strength <= 0.0:
        return
    for block, basis in basis_by_block.items():
        params = base_block_params(model, block)
        flat = flatten_grads(params)
        safe = project_gradient_away(flat, basis, projection_strength)
        assign_flat_grads(params, safe)


def train_one_step(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    grad_multipliers: Dict[str, float] | None = None,
    gradient_basis: Dict[str, torch.Tensor] | None = None,
    projection_strength: float = 0.0,
    base_lr: float = BASE_LR,
    adapter_fluid: bool = False,
) -> float:
    model.train()
    set_optimizer_lrs(optimizer, base_lr, adapter_fluid)
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(batch.x, batch.y)
    loss.backward()
    if gradient_basis:
        project_block_gradients(model, gradient_basis, projection_strength)
    for block, multiplier in (grad_multipliers or {}).items():
        multiply_grads(base_block_params(model, block), multiplier)
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


def restore_model_from_checkpoint(
    vocab_size: int,
    checkpoint: Dict[str, object],
    load_optimizer: bool = True,
) -> Tuple[TinyGPT, torch.optim.Optimizer]:
    model = TinyGPT(vocab_size).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer = make_optimizer(model)
    if load_optimizer:
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
    if step in EARLY_PROBE_STEPS or step == PHASE_B_STEPS:
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


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    clean = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if clean.size == 0:
        return float("nan"), float("nan")
    return float(clean.mean()), float(clean.std(ddof=0))


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


def viscosity_from_shock(shock: float) -> float:
    if shock < 0.5:
        return 0.10
    if shock < 1.5:
        return 0.03
    if shock < 3.0:
        return 0.01
    return 0.005


def projection_from_shock(shock: float) -> float:
    if shock < 0.5:
        return 0.25
    if shock < 1.5:
        return 0.50
    if shock < 3.0:
        return 0.80
    return 1.00


def latent_projection_from_shock(shock: float) -> float:
    if shock < 0.5:
        return 0.35
    if shock < 1.5:
        return 0.65
    if shock < 3.0:
        return 0.85
    return 1.00


def make_low_rank_basis(rows: List[torch.Tensor], rank: int) -> torch.Tensor:
    if not rows:
        return torch.empty(0, 0)
    normalized = []
    for row in rows:
        row = row.float().cpu()
        norm = row.norm().clamp_min(1e-12)
        normalized.append(row / norm)
    matrix = torch.stack(normalized, dim=0)
    if matrix.shape[0] == 1:
        return matrix[:1].contiguous()
    gram = matrix @ matrix.t()
    eigvals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(eigvals, descending=True)
    basis_rows = []
    for idx in order[: min(rank, matrix.shape[0])]:
        value = eigvals[idx].clamp_min(1e-12).sqrt()
        vector = eigvecs[:, idx]
        basis = (vector @ matrix) / value
        basis = basis / basis.norm().clamp_min(1e-12)
        basis_rows.append(basis)
    if not basis_rows:
        return torch.empty(0, matrix.shape[1])
    return torch.stack(basis_rows, dim=0).contiguous()


def top_cov_basis(cov: torch.Tensor, rank: int) -> torch.Tensor:
    if rank <= 0:
        return torch.empty(cov.shape[0], 0)
    cov = 0.5 * (cov.float().cpu() + cov.float().cpu().t())
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    max_eval = float(eigvals[order[0]].item()) if order.numel() > 0 else 0.0
    cutoff = max(max_eval * 1e-6, 1e-12)
    keep = [idx for idx in order[: min(rank, cov.shape[0])] if float(eigvals[idx].item()) > cutoff]
    if not keep:
        return torch.empty(cov.shape[0], 0)
    keep = torch.tensor(keep, dtype=torch.long)
    return eigvecs[:, keep].contiguous()


def orthonormalize_columns(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    parts = [part.float().cpu() for part in parts if part.numel() > 0]
    if not parts:
        return torch.empty(D_MODEL, 0)
    merged = torch.cat(parts, dim=1)
    q, r = torch.linalg.qr(merged, mode="reduced")
    keep = torch.diag(r).abs() > 1e-6
    if keep.any():
        q = q[:, keep]
    return q.contiguous()


def free_projector_from_covariances(act_cov: torch.Tensor, grad_cov: torch.Tensor) -> torch.Tensor:
    act_basis = top_cov_basis(act_cov, LATENT_ACT_RANK)
    grad_basis = top_cov_basis(grad_cov, LATENT_GRAD_RANK)
    occupied = orthonormalize_columns([act_basis, grad_basis])
    eye = torch.eye(act_cov.shape[0])
    if occupied.numel() == 0:
        return eye
    projector = eye - occupied @ occupied.t()
    return projector.contiguous()


def collect_gradient_basis(
    model: TinyGPT,
    stoi: Dict[str, int],
    seed: int,
    blocks: Sequence[str],
) -> Dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    rows: Dict[str, List[torch.Tensor]] = {block: [] for block in blocks}
    for index in range(GRAD_ANCHOR_BATCHES):
        batch = replay_batch_for_index(stoi, seed + 7_000_000, index, batch_size=BATCH_SIZE)
        model.zero_grad(set_to_none=True)
        _, loss = model(batch.x, batch.y)
        loss.backward()
        for block in blocks:
            rows[block].append(flatten_grads(base_block_params(model, block)).detach().cpu())
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return {block: make_low_rank_basis(block_rows, GRAD_ANCHOR_RANK) for block, block_rows in rows.items()}


def collect_latent_free_projectors(
    model: TinyGPT,
    stoi: Dict[str, int],
    seed: int,
    blocks: Sequence[str],
) -> Dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    act_cov = {block: torch.zeros(D_MODEL, D_MODEL) for block in blocks}
    grad_cov = {block: torch.zeros(D_MODEL, D_MODEL) for block in blocks}
    counts = {block: 0 for block in blocks}

    for index in range(LATENT_ANCHOR_BATCHES):
        batch = replay_batch_for_index(stoi, seed + 8_000_000, index, batch_size=BATCH_SIZE)
        model.zero_grad(set_to_none=True)
        _, loss, activations = model(batch.x, batch.y, return_activations=True)
        loss.backward()
        for block in blocks:
            block_index = int(block[1:])
            activation = activations[block_index].detach().reshape(-1, D_MODEL).float().cpu()
            gradient = activations[block_index].grad
            if gradient is None:
                gradient_flat = torch.zeros_like(activation)
            else:
                gradient_flat = gradient.detach().reshape(-1, D_MODEL).float().cpu()
            act_cov[block] += activation.t() @ activation / max(activation.shape[0], 1)
            grad_cov[block] += gradient_flat.t() @ gradient_flat / max(gradient_flat.shape[0], 1)
            counts[block] += 1

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    projectors = {}
    for block in blocks:
        denom = max(counts[block], 1)
        projectors[block] = free_projector_from_covariances(act_cov[block] / denom, grad_cov[block] / denom)
    return projectors


def train_old_skill(
    vocab_size: int,
    stoi: Dict[str, int],
    seed: int,
    bracket_eval_batches: List[Batch],
    bracket_probe_batch: Batch,
) -> AnchorInfo:
    model = TinyGPT(vocab_size).to(DEVICE)
    model.set_adapters_enabled(False)
    set_requires_grad(all_adapter_params(model), False)
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
    print(
        f"[seed {seed}] collecting old-gradient anchor basis for { '+'.join(old_frontier) } "
        f"using {GRAD_ANCHOR_BATCHES} batches, rank={GRAD_ANCHOR_RANK}"
    )
    grad_basis = collect_gradient_basis(model, stoi, seed, old_frontier)
    print(
        f"[seed {seed}] collecting latent occupied-space basis for all blocks "
        f"using {LATENT_ANCHOR_BATCHES} batches, act_rank={LATENT_ACT_RANK}, grad_rank={LATENT_GRAD_RANK}"
    )
    latent_free_projectors = collect_latent_free_projectors(model, stoi, seed, block_keys())
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
        grad_basis=grad_basis,
        latent_free_projectors=latent_free_projectors,
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
    viscosity: float,
    replay_count: int,
    adapter_enabled: bool,
    grad_projection: float,
    latent_projection: float,
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
    return BranchRow(
        step=step,
        old_loss=old_metrics["loss"],
        old_close_acc=old_metrics["close_acc"],
        old_seq_acc=old_metrics["seq_acc"],
        text_loss=text_metrics["loss"],
        text_acc=text_metrics["acc"],
        z_shock=z_shock,
        viscosity=viscosity,
        replay_count=replay_count,
        adapter_enabled=adapter_enabled,
        grad_projection=grad_projection,
        latent_projection=latent_projection,
    )


def branch_uses_adapters(branch_name: str) -> bool:
    return branch_name in {
        "water_weights_gradient_full",
        "water_weights_latent_free",
        "water_weights_latent_gradient_full",
        "water_weights_latent_strong",
        "water_weights_latent_adapter_only",
    }


def branch_uses_z_viscosity(branch_name: str) -> bool:
    return branch_name in {
        "water_weights_gradient_full",
        "water_weights_latent_free",
        "water_weights_latent_gradient_full",
        "water_weights_latent_strong",
    }


def branch_uses_static_viscosity(branch_name: str) -> bool:
    return False


def branch_uses_gradient_anchor(branch_name: str) -> bool:
    return branch_name in {
        "water_weights_gradient_full",
        "water_weights_latent_gradient_full",
        "water_weights_latent_strong",
    }


def branch_uses_z_projection(branch_name: str) -> bool:
    return branch_name in {
        "water_weights_gradient_full",
        "water_weights_latent_gradient_full",
    }


def branch_uses_latent_free(branch_name: str) -> bool:
    return branch_name in {
        "water_weights_latent_free",
        "water_weights_latent_gradient_full",
        "water_weights_latent_strong",
        "water_weights_latent_adapter_only",
    }


def branch_uses_z_latent_projection(branch_name: str) -> bool:
    return branch_name in {
        "water_weights_latent_free",
        "water_weights_latent_gradient_full",
        "water_weights_latent_adapter_only",
    }


def should_replay(branch_name: str, step: int, replay_count: int, replay_budget: int) -> bool:
    if replay_count >= replay_budget:
        return False
    if branch_name != "none":
        if step <= 60 and step % 4 == 1:
            return True
        return step % 10 == 0
    return False


def run_reminiscence(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    stoi: Dict[str, int],
    seed: int,
) -> None:
    model.set_adapters_enabled(False)
    set_requires_grad(all_adapter_params(model), False)
    for step in range(1, REMINISCENCE_STEPS + 1):
        batch = bracket_train_batch_for_step(stoi, seed + 9_000_000, step)
        train_one_step(model, optimizer, batch, base_lr=REMINISCENCE_LR, adapter_fluid=False)


def branch_grad_multipliers(branch_name: str, frontier: Sequence[str], current_z_viscosity: float) -> Dict[str, float]:
    if branch_uses_z_viscosity(branch_name):
        return {block: current_z_viscosity for block in frontier}
    if branch_uses_static_viscosity(branch_name):
        return {block: STATIC_VISCOSITY for block in frontier}
    return {}


def projection_strength_for_branch(branch_name: str, current_projection: float) -> float:
    if branch_name == "water_weights_latent_strong":
        return 1.0
    if branch_uses_z_projection(branch_name):
        return current_projection
    if branch_uses_gradient_anchor(branch_name):
        return 1.0
    return 0.0


def latent_projection_strength_for_branch(branch_name: str, current_projection: float) -> float:
    if branch_name == "water_weights_latent_strong":
        return 1.0
    if branch_uses_z_latent_projection(branch_name):
        return current_projection
    if branch_uses_latent_free(branch_name):
        return 1.0
    return 0.0


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
) -> BranchResult:
    load_optimizer = branch_name == "none"
    model, optimizer = restore_model_from_checkpoint(vocab_size, anchor.checkpoint, load_optimizer=load_optimizer)
    replay_budget = int(PHASE_B_STEPS * REPLAY_BUDGET_FRACTION)
    replay_count = 0
    rows: List[BranchRow] = []
    current_z_viscosity = INITIAL_Z_VISCOSITY
    current_projection = INITIAL_GRAD_PROJECTION
    current_latent_projection = INITIAL_LATENT_PROJECTION
    viscosity_steps = 0
    grad_projection_steps = 0
    latent_projection_steps = 0
    reminiscence_steps = 0
    adapter_enabled = branch_uses_adapters(branch_name)
    gradient_basis = (
        {block: basis.to(DEVICE) for block, basis in anchor.grad_basis.items()}
        if branch_uses_gradient_anchor(branch_name)
        else {}
    )
    latent_projectors = (
        {block: projector.to(DEVICE) for block, projector in anchor.latent_free_projectors.items()}
        if branch_uses_latent_free(branch_name)
        else {}
    )

    print(f"[branch:{branch_name}] start")
    if branch_name != "none":
        print(
            f"[branch:{branch_name}] momentum reset + old-data reminiscence "
            f"for {REMINISCENCE_STEPS} steps at lr={REMINISCENCE_LR:.1e}"
        )
        run_reminiscence(model, optimizer, stoi, seed)
        reminiscence_steps = REMINISCENCE_STEPS

    configure_branch_trainability(model, branch_name, anchor.old_frontier)
    if branch_name == "water_weights_latent_adapter_only":
        assert not any(param.requires_grad for param in all_base_params(model)), "adapter-only branch left base trainable"
        assert any(param.requires_grad for param in all_adapter_params(model)), "adapter-only branch did not enable adapters"
    if latent_projectors:
        model.set_latent_free_projectors(
            latent_projectors,
            latent_projection_strength_for_branch(branch_name, current_latent_projection),
        )

    row0 = probe_branch_row(
        model,
        val_data,
        text_eval_positions,
        bracket_eval_batches,
        bracket_probe_batch,
        anchor,
        0,
        current_z_viscosity if branch_uses_z_viscosity(branch_name) else 1.0,
        replay_count,
        adapter_enabled,
        projection_strength_for_branch(branch_name, current_projection),
        latent_projection_strength_for_branch(branch_name, current_latent_projection),
    )
    rows.append(row0)

    for step in range(1, PHASE_B_STEPS + 1):
        replay_this_step = should_replay(branch_name, step, replay_count, replay_budget)
        if replay_this_step:
            batch = replay_batch_for_index(stoi, seed, replay_count)
            replay_count += 1
        else:
            batch = text_batch_from_positions(train_data, text_train_positions[step - 1])

        grad_multipliers = branch_grad_multipliers(branch_name, anchor.old_frontier, current_z_viscosity)
        projection_strength = projection_strength_for_branch(branch_name, current_projection)
        latent_projection_strength = latent_projection_strength_for_branch(branch_name, current_latent_projection)
        if latent_projectors:
            model.set_latent_free_projectors(latent_projectors, latent_projection_strength)
        else:
            model.clear_latent_free_projectors()
        if grad_multipliers:
            viscosity_steps += 1
        if projection_strength > 0.0:
            grad_projection_steps += 1
        if latent_projection_strength > 0.0:
            latent_projection_steps += 1
        train_one_step(
            model,
            optimizer,
            batch,
            grad_multipliers=grad_multipliers,
            gradient_basis=gradient_basis,
            projection_strength=projection_strength,
            base_lr=BASE_LR,
            adapter_fluid=adapter_enabled,
        )

        if is_probe_step(step):
            shown_viscosity = current_z_viscosity if branch_uses_z_viscosity(branch_name) else (
                STATIC_VISCOSITY if branch_uses_static_viscosity(branch_name) else 1.0
            )
            shown_projection = projection_strength_for_branch(branch_name, current_projection)
            shown_latent_projection = latent_projection_strength_for_branch(branch_name, current_latent_projection)
            if latent_projectors:
                model.set_latent_free_projectors(latent_projectors, shown_latent_projection)
            row = probe_branch_row(
                model,
                val_data,
                text_eval_positions,
                bracket_eval_batches,
                bracket_probe_batch,
                anchor,
                step,
                shown_viscosity,
                replay_count,
                adapter_enabled,
                shown_projection,
                shown_latent_projection,
            )
            rows.append(row)
            if branch_uses_z_viscosity(branch_name):
                current_z_viscosity = viscosity_from_shock(row.z_shock)
            if branch_uses_z_projection(branch_name):
                current_projection = projection_from_shock(row.z_shock)
            if branch_uses_z_latent_projection(branch_name):
                current_latent_projection = latent_projection_from_shock(row.z_shock)
            if step % BRANCH_LOG_INTERVAL == 0 or step == PHASE_B_STEPS:
                print(
                    f"[branch:{branch_name}] step={step:04d}/{PHASE_B_STEPS} "
                    f"old_seq={row.old_seq_acc:.3f} text_loss={row.text_loss:.3f} "
                    f"z_shock={row.z_shock:.2f} viscosity={shown_viscosity:.3f} "
                    f"grad_proj={shown_projection:.2f} latent_proj={shown_latent_projection:.2f} "
                    f"replay={replay_count}/{replay_budget} "
                    f"adapters={'on' if adapter_enabled else 'off'}"
                )

    result = summarize_branch(
        branch_name,
        rows,
        anchor.old_seq_acc,
        replay_count,
        replay_budget,
        viscosity_steps,
        grad_projection_steps,
        latent_projection_steps,
        reminiscence_steps,
        adapter_enabled,
    )
    print(
        f"[branch:{branch_name}] old_seq_auc={result.old_seq_auc:.3f} "
        f"final_old_seq={result.final_old_seq:.3f} text_loss_auc={result.text_loss_auc:.3f} "
        f"forget_step={format_step(result.forgetting_step)} replay={result.replay_count}/{result.replay_budget} "
        f"visc_steps={result.viscosity_steps} grad_proj_steps={result.grad_projection_steps} "
        f"latent_proj_steps={result.latent_projection_steps}"
    )
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def summarize_branch(
    name: str,
    rows: List[BranchRow],
    anchor_seq: float,
    replay_count: int,
    replay_budget: int,
    viscosity_steps: int,
    grad_projection_steps: int,
    latent_projection_steps: int,
    reminiscence_steps: int,
    adapter_enabled: bool,
) -> BranchResult:
    final = rows[-1]
    return BranchResult(
        name=name,
        rows=rows,
        old_seq_auc=auc_rows(rows, "old_seq_acc"),
        old_close_auc=auc_rows(rows, "old_close_acc"),
        text_loss_auc=auc_rows(rows, "text_loss"),
        text_acc_auc=auc_rows(rows, "text_acc"),
        forgetting_area=forgetting_area(rows, anchor_seq),
        forgetting_step=first_forgetting_step(rows, anchor_seq),
        final_old_seq=final.old_seq_acc,
        final_old_close=final.old_close_acc,
        final_text_loss=final.text_loss,
        final_text_acc=final.text_acc,
        replay_count=replay_count,
        replay_budget=replay_budget,
        viscosity_steps=viscosity_steps,
        grad_projection_steps=grad_projection_steps,
        latent_projection_steps=latent_projection_steps,
        reminiscence_steps=reminiscence_steps,
        adapter_enabled=adapter_enabled,
        verified_replay_cap=replay_count <= replay_budget,
    )


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
        "replay_count": result.replay_count,
        "replay_budget": result.replay_budget,
        "viscosity_steps": result.viscosity_steps,
        "grad_projection_steps": result.grad_projection_steps,
        "latent_projection_steps": result.latent_projection_steps,
        "reminiscence_steps": result.reminiscence_steps,
        "adapter_enabled": int(result.adapter_enabled),
        "verified_replay_cap": int(result.verified_replay_cap),
    }


def print_seed_summary(seed: int, anchor: AnchorInfo, results: Dict[str, BranchResult]) -> None:
    print("\n" + "=" * 78)
    print(f"SEED {seed} WATER-WEIGHTS RESULT")
    print("=" * 78)
    print(
        f"Anchor: step={anchor.step} ready={anchor.reached_ready} "
        f"old_close={anchor.old_close_acc:.3f} old_seq={anchor.old_seq_acc:.3f} "
        f"frontier={'+'.join(anchor.old_frontier)}"
    )
    print(
        f"{'branch':36s} {'old_auc':>7s} {'final_old':>9s} {'forget':>6s} "
        f"{'text_loss':>9s} {'replay':>9s} {'visc':>6s} {'grad':>6s} "
        f"{'latent':>7s} {'rem':>5s} {'adapt':>5s}"
    )
    for name in BRANCH_NAMES:
        result = results[name]
        print(
            f"{name:36s} {result.old_seq_auc:7.3f} {result.final_old_seq:9.3f} "
            f"{format_step(result.forgetting_step):>6s} {result.final_text_loss:9.3f} "
            f"{result.replay_count:3d}/{result.replay_budget:<5d} {result.viscosity_steps:6d} "
            f"{result.grad_projection_steps:6d} {result.latent_projection_steps:7d} "
            f"{result.reminiscence_steps:5d} "
            f"{'yes' if result.adapter_enabled else 'no':>5s}"
        )
    none = results["none"]
    water_grad = results["water_weights_gradient_full"]
    latent_free = results["water_weights_latent_free"]
    latent_grad = results["water_weights_latent_gradient_full"]
    latent_strong = results["water_weights_latent_strong"]
    latent_adapter = results["water_weights_latent_adapter_only"]
    best_latent = max(
        latent_free.old_seq_auc,
        latent_grad.old_seq_auc,
        latent_strong.old_seq_auc,
        latent_adapter.old_seq_auc,
    )
    print(f"Water+gradient gain over none: {water_grad.old_seq_auc - none.old_seq_auc:+.3f}")
    print(f"Latent-free gain over Water+gradient: {latent_free.old_seq_auc - water_grad.old_seq_auc:+.3f}")
    print(f"Latent+gradient gain over Water+gradient: {latent_grad.old_seq_auc - water_grad.old_seq_auc:+.3f}")
    print(f"Latent strong gain over latent+gradient: {latent_strong.old_seq_auc - latent_grad.old_seq_auc:+.3f}")
    print(f"Latent adapter-only gain over Water+gradient: {latent_adapter.old_seq_auc - water_grad.old_seq_auc:+.3f}")
    print(f"Best latent branch gain over Water+gradient: {best_latent - water_grad.old_seq_auc:+.3f}")
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

    print("\n" + "=" * 78)
    print("WATER-WEIGHTS SUMMARY ACROSS SEEDS")
    print("=" * 78)
    print(f"Seeds run: {len(seeds)}")
    print(f"Old bracket seq>=0.90 reached before text phase: {len(ready)}/{len(seeds)}")
    print(f"None branch forgetting observed: {len(none_forgets)}/{len(seeds)}")
    print("\nMean old_seq_auc by branch:")
    for name in BRANCH_NAMES:
        mean, std = mean_std([float(row["old_seq_auc"]) for row in by_branch[name]])
        print(f"  {name:26s} {format_mean_std(mean, std)}")
    print("\nMean final text loss by branch:")
    for name in BRANCH_NAMES:
        mean, std = mean_std([float(row["final_text_loss"]) for row in by_branch[name]])
        print(f"  {name:26s} {format_mean_std(mean, std)}")

    def branch_values(branch: str, key: str) -> List[float]:
        return [float(row[key]) for row in by_branch[branch]]

    water_grad_auc = branch_values("water_weights_gradient_full", "old_seq_auc")
    latent_free_auc = branch_values("water_weights_latent_free", "old_seq_auc")
    latent_grad_auc = branch_values("water_weights_latent_gradient_full", "old_seq_auc")
    latent_strong_auc = branch_values("water_weights_latent_strong", "old_seq_auc")
    latent_adapter_auc = branch_values("water_weights_latent_adapter_only", "old_seq_auc")
    none_auc = branch_values("none", "old_seq_auc")
    water_grad_loss = branch_values("water_weights_gradient_full", "final_text_loss")
    latent_free_loss = branch_values("water_weights_latent_free", "final_text_loss")
    latent_grad_loss = branch_values("water_weights_latent_gradient_full", "final_text_loss")
    latent_strong_loss = branch_values("water_weights_latent_strong", "final_text_loss")
    latent_adapter_loss = branch_values("water_weights_latent_adapter_only", "final_text_loss")
    none_loss = branch_values("none", "final_text_loss")
    gains_grad_none = [wg - n for wg, n in zip(water_grad_auc, none_auc)]
    gains_latent_free_grad = [lf - wg for lf, wg in zip(latent_free_auc, water_grad_auc)]
    gains_latent_grad_grad = [lg - wg for lg, wg in zip(latent_grad_auc, water_grad_auc)]
    gains_latent_strong_grad = [ls - wg for ls, wg in zip(latent_strong_auc, water_grad_auc)]
    gains_latent_adapter_grad = [la - wg for la, wg in zip(latent_adapter_auc, water_grad_auc)]
    gains_latent_strong_latent_grad = [ls - lg for ls, lg in zip(latent_strong_auc, latent_grad_auc)]
    gains_latent_grad_none = [lg - n for lg, n in zip(latent_grad_auc, none_auc)]
    gains_latent_strong_none = [ls - n for ls, n in zip(latent_strong_auc, none_auc)]
    gains_latent_adapter_none = [la - n for la, n in zip(latent_adapter_auc, none_auc)]
    text_slowdown_grad = [(w - n) / max(abs(n), 1e-12) for w, n in zip(water_grad_loss, none_loss)]
    text_slowdown_latent_free = [(w - n) / max(abs(n), 1e-12) for w, n in zip(latent_free_loss, none_loss)]
    text_slowdown_latent_grad = [(w - n) / max(abs(n), 1e-12) for w, n in zip(latent_grad_loss, none_loss)]
    text_slowdown_latent_strong = [(w - n) / max(abs(n), 1e-12) for w, n in zip(latent_strong_loss, none_loss)]
    text_slowdown_latent_adapter = [(w - n) / max(abs(n), 1e-12) for w, n in zip(latent_adapter_loss, none_loss)]
    best_latent_gains = []
    best_latent_names: Dict[str, int] = {
        "water_weights_latent_free": 0,
        "water_weights_latent_gradient_full": 0,
        "water_weights_latent_strong": 0,
        "water_weights_latent_adapter_only": 0,
    }
    for index in range(len(seeds)):
        latent_candidates = {
            name: float(by_branch[name][index]["old_seq_auc"])
            for name in {
                "water_weights_latent_free",
                "water_weights_latent_gradient_full",
                "water_weights_latent_strong",
                "water_weights_latent_adapter_only",
            }
        }
        best_name = max(latent_candidates, key=latent_candidates.get)
        best_latent_names[best_name] += 1
        best_latent_gains.append(
            latent_candidates[best_name] - float(by_branch["water_weights_gradient_full"][index]["old_seq_auc"])
        )

    best_latent_mean, best_latent_std = mean_std(best_latent_gains)
    slowdown_grad_mean, slowdown_grad_std = mean_std(text_slowdown_grad)
    slowdown_latent_free_mean, slowdown_latent_free_std = mean_std(text_slowdown_latent_free)
    slowdown_latent_grad_mean, slowdown_latent_grad_std = mean_std(text_slowdown_latent_grad)
    slowdown_latent_strong_mean, slowdown_latent_strong_std = mean_std(text_slowdown_latent_strong)
    slowdown_latent_adapter_mean, slowdown_latent_adapter_std = mean_std(text_slowdown_latent_adapter)
    print("\nPrimary latent-free Water effects:")
    print(f"  Water+gradient gain over none old_seq_auc: {format_mean_std(*mean_std(gains_grad_none))}")
    print(f"  Latent-free gain over Water+gradient: {format_mean_std(*mean_std(gains_latent_free_grad))}")
    print(f"  Latent+gradient gain over Water+gradient: {format_mean_std(*mean_std(gains_latent_grad_grad))}")
    print(f"  Latent strong gain over Water+gradient: {format_mean_std(*mean_std(gains_latent_strong_grad))}")
    print(f"  Latent adapter-only gain over Water+gradient: {format_mean_std(*mean_std(gains_latent_adapter_grad))}")
    print(f"  Latent strong gain over latent+gradient: {format_mean_std(*mean_std(gains_latent_strong_latent_grad))}")
    print(f"  Latent+gradient gain over none: {format_mean_std(*mean_std(gains_latent_grad_none))}")
    print(f"  Latent strong gain over none: {format_mean_std(*mean_std(gains_latent_strong_none))}")
    print(f"  Latent adapter-only gain over none: {format_mean_std(*mean_std(gains_latent_adapter_none))}")
    print(f"  Latent+gradient bootstrap 95% CI vs Water+gradient: {format_ci(*bootstrap_ci(gains_latent_grad_grad))}")
    print(f"  Latent adapter-only bootstrap 95% CI vs Water+gradient: {format_ci(*bootstrap_ci(gains_latent_adapter_grad))}")
    print(f"  Best latent branch gain over Water+gradient: {format_mean_std(best_latent_mean, best_latent_std)}")
    print(f"  Best latent branch counts: {best_latent_names}")
    print(f"  Water+gradient final text-loss slowdown vs none: {format_mean_std(slowdown_grad_mean, slowdown_grad_std)}")
    print(f"  Latent-free final text-loss slowdown vs none: {format_mean_std(slowdown_latent_free_mean, slowdown_latent_free_std)}")
    print(f"  Latent+gradient final text-loss slowdown vs none: {format_mean_std(slowdown_latent_grad_mean, slowdown_latent_grad_std)}")
    print(f"  Latent strong final text-loss slowdown vs none: {format_mean_std(slowdown_latent_strong_mean, slowdown_latent_strong_std)}")
    print(f"  Latent adapter-only final text-loss slowdown vs none: {format_mean_std(slowdown_latent_adapter_mean, slowdown_latent_adapter_std)}")

    if (
        len(ready) == len(seeds)
        and len(none_forgets) == len(seeds)
        and mean_std(gains_latent_adapter_grad)[0] > 0
        and slowdown_latent_adapter_mean <= 0.15
    ):
        interpretation = "strong pass: solid-base latent adapters improve retention over gradient anchoring with acceptable text slowdown."
    elif best_latent_mean > 0 and min(
        slowdown_latent_free_mean,
        slowdown_latent_grad_mean,
        slowdown_latent_strong_mean,
        slowdown_latent_adapter_mean,
    ) <= 0.15:
        interpretation = "mixed but promising: at least one latent-free Water branch improves retention, but branch choice remains open."
    else:
        interpretation = "diagnostic/fail: latent-free routing did not beat gradient anchoring cleanly under this setup."
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
    print(f"BEGIN WATER-WEIGHTS SEED {seed}")
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
    for branch_name in BRANCH_NAMES:
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
        )
        results[branch_name] = result
        if SMOKE:
            after_sig = checkpoint_signature(anchor.checkpoint)
            assert abs(after_sig - checkpoint_sig) < 1e-5, "branch mutated shared checkpoint"

    if SMOKE:
        assert set(results) == set(BRANCH_NAMES), "not all branches executed"
        assert results["none"].replay_count == 0, "none branch replayed"
        assert results["water_weights_gradient_full"].replay_count > 0, "smoke expected water-gradient replay"
        assert results["water_weights_latent_gradient_full"].replay_count > 0, "smoke expected latent-gradient replay"
        assert results["water_weights_latent_strong"].replay_count > 0, "smoke expected strong replay"
        assert results["water_weights_latent_adapter_only"].replay_count > 0, "smoke expected adapter-only replay"
        assert results["water_weights_gradient_full"].reminiscence_steps == REMINISCENCE_STEPS, (
            "reminiscence did not run"
        )
        assert results["water_weights_latent_strong"].viscosity_steps > 0, "strong viscosity did not engage"
        assert results["water_weights_gradient_full"].grad_projection_steps > 0, (
            "water-gradient projection did not engage"
        )
        assert results["water_weights_latent_strong"].grad_projection_steps > 0, (
            "strong projection did not engage"
        )
        assert results["water_weights_latent_free"].latent_projection_steps > 0, (
            "latent-free projection did not engage"
        )
        assert results["water_weights_latent_gradient_full"].latent_projection_steps > 0, (
            "latent-gradient projection did not engage"
        )
        assert results["water_weights_latent_adapter_only"].latent_projection_steps > 0, (
            "adapter-only latent projection did not engage"
        )
        assert results["water_weights_latent_adapter_only"].grad_projection_steps == 0, (
            "adapter-only branch should not project frozen base gradients"
        )
        assert results["water_weights_latent_adapter_only"].viscosity_steps == 0, (
            "adapter-only branch should keep the base solid rather than viscous"
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
    print("WATER WEIGHTS: PROPHYLACTIC CONTINUAL LEARNING BENCHMARK")
    print("=" * 78)
    print("Question: can solid-base latent adapters plus Water Weights prevent catastrophic forgetting?")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Phase A: bracket until seq>={OLD_READY_SEQ:.2f} or {PHASE_A_MAX_STEPS} steps")
    print(f"Phase B: text training for {PHASE_B_STEPS} steps")
    print(f"Reminiscence: {REMINISCENCE_STEPS} old-task steps at lr={REMINISCENCE_LR:.1e}")
    print(f"Model: d={D_MODEL}, layers={N_LAYER}, heads={N_HEAD}, block={BLOCK_SIZE}, adapter_rank={ADAPTER_RANK}")
    print(f"Bracket task: {len(OPENERS)} bracket types, depth=[{BRACKET_MIN_DEPTH}, {BRACKET_MAX_DEPTH}]")
    print(
        f"Viscosity: static={STATIC_VISCOSITY:.3f}, initial_z={INITIAL_Z_VISCOSITY:.3f}, "
        f"adapter_lr_mult={ADAPTER_LR_MULT:.1f}, replay_cap={REPLAY_BUDGET_FRACTION:.0%}"
    )
    print(f"Gradient anchor: batches={GRAD_ANCHOR_BATCHES}, rank={GRAD_ANCHOR_RANK}, initial_projection={INITIAL_GRAD_PROJECTION:.2f}")
    print(
        f"Latent free-space anchor: batches={LATENT_ANCHOR_BATCHES}, "
        f"act_rank={LATENT_ACT_RANK}, grad_rank={LATENT_GRAD_RANK}, "
        f"initial_projection={INITIAL_LATENT_PROJECTION:.2f}"
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
    print("Primary metrics: old_seq_auc, final_old_seq, and final text loss.")

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
