from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from qwen_tomography import (
    DEFAULT_TARGET_SUFFIX,
    LayerSaturation,
    SATURATION_CONSECUTIVE_REQUIRED,
    SaturationReport,
    TaskProfile,
    TomographyResult,
    collect_task_profile,
    compute_saturation_report,
    run_tomography,
    should_expand,
    write_tomography_csv,
)
from standalone_latent_lora_qwen import (
    EscapeSchedule,
    LatentLoRAConfig,
    LatentLoRALinear,
    attach_latent_lora,
    choose_dtype,
    default_model_id,
    detach_latent_lora,
    load_causal_lm,
    load_tokenizer,
)


MODEL_ID = "Qwen/Qwen2.5-0.5B"
LOCAL_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_SUFFIX = ["mlp.down_proj"]

ATTACH_STEPS = 800
CONSOLIDATION_STEPS = 400
POLISH_STEPS = 120
ATTACH_LR = 2e-5
CONSOLIDATION_LR = 1e-5
POLISH_LR = 5e-6
BATCH_SIZE = 4
MAX_SEQ_LEN = 256
GRAD_CLIP = 1.0
GRADIENT_CHECKPOINTING = True
EVAL_BATCH_SIZE = 8

CONSOL_KL_WEIGHT = 1.0
CONSOL_OLD_KL_WEIGHT = 0.75
CONSOL_HIDDEN_WEIGHT = 0.5
CONSOL_OLD_BATCH_PERIOD = 3
CONSOL_VISCOSITY = 0.03
CONSOL_AMOEBA_ENABLED = False
CONSOL_AMOEBA_GENTLE_FRAC = 0.30
CONSOL_AMOEBA_POLISH_FRAC = 0.10
CONSOL_AMOEBA_GENTLE_OLD_SCALE = 1.0
CONSOL_AMOEBA_GENTLE_NEW_SCALE = 1.0
CONSOL_AMOEBA_POLISH_LR_SCALE = 0.20

EXPANSION_GATE_INIT = 0.0025
EXPANSION_GATE_FLOORS = (0.015, 0.030, 0.050)
EXPANSION_NEW_BLOCK_LR_SCALE = 3.0

BEST_D_PPL_DEGRADATION = 0.15
BALANCED_D_MARGIN = 0.05
PROOF_V2_B_ATTACH_STEPS = 2400
PROOF_V2_B_ATTACH_LR = 8e-5
PROOF_V2_B_RANK = 48
PROOF_V2_B_ALPHA = 96.0
PROOF_V2_B_GATE_INIT = -1.5
PROOF_V2_B_USE_UP_PROJ = True
PROOF_V2_B_MIN_LAYERS = 8
PROOF_V2_D_ATTACH_STEPS = 1200
PROOF_V2_D_ATTACH_LR = 8e-5
PROOF_V2_D_RANK = 48
PROOF_V2_D_ALPHA = 96.0
PROOF_V2_D_GATE_INIT = -1.5
PROOF_V2_D_USE_UP_PROJ = True
PROOF_V2_D_MIN_LAYERS = 8

WIKITEXT_EVAL_SAMPLES = 200
JSON_EVAL_SAMPLES = 200
REVERSAL_EVAL_SAMPLES = 200
SORT_EVAL_SAMPLES = 200

PROOF_SEEDS = [1337, 2027, 31415]
PHASE_VERSION = 10
TASK_SUITES = ("legacy", "proof_v2")
TEACHER_OLD_LOSS_WEIGHT = 0.10
TEACHER_OLD_BATCH_PERIOD = 4

ESCAPE_SCHEDULE = EscapeSchedule(
    levels=(1.0, 0.65, 0.35, 0.15),
    step_fractions=(0.0, 0.21, 0.50, 0.75),
)

FALLBACK_WIKITEXT = [
    "The town slept beneath a sheet of winter fog while the station clock ticked into dawn.",
    "Scientific progress depends on observation, prediction, and the discipline to test both.",
    "A library is not only a building of books, but a memory system for civilization.",
    "The speaker paused, adjusted the microphone, and began again with unusual calm.",
]

JSON_TRAIN_TEMPLATES: List[Dict[str, str]] = [
    {
        "prompt": "Create a JSON user profile with fields: name, email, age",
        "target": '{"name": "Alice", "email": "alice@example.com", "age": 30}',
    },
    {
        "prompt": "Create a JSON product with fields: id, name, price, in_stock",
        "target": '{"id": 1, "name": "Widget", "price": 9.99, "in_stock": true}',
    },
    {
        "prompt": "Create a JSON book with fields: title, author, pages, hardcover",
        "target": '{"title": "Signals", "author": "Mina", "pages": 240, "hardcover": false}',
    },
    {
        "prompt": "Create a JSON city record with fields: city, country, population",
        "target": '{"city": "Pune", "country": "India", "population": 7120000}',
    },
    {
        "prompt": "Create a JSON order with fields: order_id, status, total",
        "target": '{"order_id": 42, "status": "shipped", "total": 118.5}',
    },
    {
        "prompt": "Create a JSON sensor reading with fields: sensor, value, unit",
        "target": '{"sensor": "thermometer", "value": 21.5, "unit": "C"}',
    },
    {
        "prompt": "Create a JSON account with fields: username, active, followers",
        "target": '{"username": "orbit", "active": true, "followers": 1280}',
    },
    {
        "prompt": "Create a JSON event with fields: title, day, attendees",
        "target": '{"title": "Demo Day", "day": "Friday", "attendees": 75}',
    },
    {
        "prompt": "Create a JSON weather report with fields: city, temperature_c, condition",
        "target": '{"city": "Jaipur", "temperature_c": 34, "condition": "clear"}',
    },
    {
        "prompt": "Create a JSON invoice with fields: invoice_id, due_date, paid",
        "target": '{"invoice_id": 1007, "due_date": "2026-05-10", "paid": false}',
    },
    {
        "prompt": "Create a JSON movie record with fields: title, year, rating",
        "target": '{"title": "North Star", "year": 2021, "rating": 8.1}',
    },
    {
        "prompt": "Create a JSON employee record with fields: employee_id, department, remote",
        "target": '{"employee_id": 88, "department": "Research", "remote": true}',
    },
    {
        "prompt": "Create a JSON shipment with fields: tracking_id, carrier, delivered",
        "target": '{"tracking_id": "ZX-55", "carrier": "Dart", "delivered": false}',
    },
    {
        "prompt": "Create a JSON recipe with fields: name, servings, vegetarian",
        "target": '{"name": "Lentil Soup", "servings": 4, "vegetarian": true}',
    },
    {
        "prompt": "Create a JSON classroom with fields: course, room, enrolled",
        "target": '{"course": "Physics", "room": "B12", "enrolled": 36}',
    },
    {
        "prompt": "Create a JSON stock quote with fields: symbol, price, currency",
        "target": '{"symbol": "ACME", "price": 47.25, "currency": "USD"}',
    },
]

JSON_EVAL_TEMPLATES: List[Dict[str, str]] = [
    {
        "prompt": "Create a JSON support ticket with fields: ticket_id, priority, resolved",
        "target": '{"ticket_id": 731, "priority": "high", "resolved": false}',
    },
    {
        "prompt": "Create a JSON travel leg with fields: origin, destination, duration_min",
        "target": '{"origin": "DEL", "destination": "BLR", "duration_min": 155}',
    },
    {
        "prompt": "Create a JSON conference talk with fields: speaker, topic, minutes",
        "target": '{"speaker": "Riya", "topic": "Systems", "minutes": 30}',
    },
    {
        "prompt": "Create a JSON fitness log with fields: exercise, reps, completed",
        "target": '{"exercise": "pushup", "reps": 20, "completed": true}',
    },
]

PROOF_V2_TRAIN_RECORD_GROUPS: List[Tuple[Tuple[str, ...], Dict[str, Tuple[str, ...]]]] = [
    (
        ("A7X", "K2Q", "M1P"),
        {
            "A7X": ("ALX", "BEX", "CYN", "DOR"),
            "K2Q": ("17", "23", "41", "58"),
            "M1P": ("ON", "OFF", "IDLE", "STBY"),
        },
    ),
    (
        ("Q9R", "B4S", "L2T"),
        {
            "Q9R": ("RIV", "NEX", "SUL", "TAR"),
            "B4S": ("05", "11", "29", "44"),
            "L2T": ("RED", "BLUE", "GREEN", "GOLD"),
        },
    ),
    (
        ("C5U", "G6V", "H3W"),
        {
            "C5U": ("ION", "VEX", "UMA", "KAI"),
            "G6V": ("09", "33", "72", "84"),
            "H3W": ("LOW", "MID", "HIGH", "PEAK"),
        },
    ),
]

PROOF_V2_EVAL_RECORD_GROUPS: List[Tuple[Tuple[str, ...], Dict[str, Tuple[str, ...]]]] = [
    (
        ("T4N", "P1R", "R8K"),
        {
            "T4N": ("JAX", "LUM", "QOR", "VYN"),
            "P1R": ("12", "37", "64", "91"),
            "R8K": ("UP", "DOWN", "HOLD", "GLIDE"),
        },
    ),
    (
        ("N5M", "R4D", "ZXQ"),
        {
            "N5M": ("OME", "RHO", "SIG", "TAU"),
            "R4D": ("03", "27", "55", "88"),
            "ZXQ": ("COOL", "WARM", "HOT", "MILD"),
        },
    ),
]

PROOF_V2_ROUTE_CODES: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("RAZ", (1, 2, 0)),
    ("KIV", (2, 0, 1)),
    ("MOP", (0, 2, 1)),
    ("TEX", (2, 1, 0)),
)
PROOF_V2_OUTPUT_SLOTS: Tuple[str, str, str] = ("LU1", "KE2", "ZO3")
PROOF_V2_TRAIN_RECORD_FAMILY_COUNT = 256
PROOF_V2_EVAL_RECORD_FAMILY_COUNT = 128
PROOF_V2_RECORD_VALUE_POOL_SIZE = 8
PROOF_V2_NONCE_ALPHABET = "BCDFGHJKLMNPQRSTVWXYZ23456789"

PROOF_V2_TRAIN_RECORD_PROMPTS: Tuple[str, ...] = (
    "Route {route}\nFields: {fields}\nPayload: {assignments}\nEmit LU1,KE2,ZO3:",
)
PROOF_V2_EVAL_RECORD_PROMPTS: Tuple[str, ...] = (
    "Apply map {route}\nField order {fields}\nObserved bindings {assignments}\nReturn LU1/KE2/ZO3:",
)

PROOF_V2_TRAIN_SEQUENCE_SYMBOLS: Tuple[str, ...] = (
    "A7X", "K2Q", "M1P", "Q9R", "B4S", "L2T", "C5U", "G6V", "H3W", "N5M", "R4D", "D8Y"
)
PROOF_V2_EVAL_SEQUENCE_SYMBOLS: Tuple[str, ...] = (
    "T4N", "P1R", "R8K", "ZXQ", "HK5", "JD3", "PN6", "LM2", "VV9", "QF7", "RX4", "UM8"
)
PROOF_V2_SORT_KEY_ORDER: Tuple[str, ...] = ("VO", "KE", "ZI", "TU", "XA")


@dataclass
class RuntimeConfig:
    model_id: str
    device: str
    dtype: torch.dtype
    local_files_only: bool
    resume: bool
    smoke: bool
    output_dir: Path
    backup_dir: Path | None
    seed: int
    phase_scope: str
    task_suite: str = "proof_v2"
    update_task: str | None = None
    attach_steps: int = ATTACH_STEPS
    consolidation_steps: int = CONSOLIDATION_STEPS
    polish_steps: int = POLISH_STEPS
    attach_lr: float = ATTACH_LR
    consolidation_lr: float = CONSOLIDATION_LR
    polish_lr: float = POLISH_LR
    consol_kl_weight: float = CONSOL_KL_WEIGHT
    consol_old_kl_weight: float = CONSOL_OLD_KL_WEIGHT
    consol_hidden_weight: float = CONSOL_HIDDEN_WEIGHT
    consol_old_batch_period: int = CONSOL_OLD_BATCH_PERIOD
    consol_amoeba_enabled: bool = CONSOL_AMOEBA_ENABLED
    consol_amoeba_gentle_frac: float = CONSOL_AMOEBA_GENTLE_FRAC
    consol_amoeba_polish_frac: float = CONSOL_AMOEBA_POLISH_FRAC
    consol_amoeba_gentle_old_scale: float = CONSOL_AMOEBA_GENTLE_OLD_SCALE
    consol_amoeba_gentle_new_scale: float = CONSOL_AMOEBA_GENTLE_NEW_SCALE
    consol_amoeba_polish_lr_scale: float = CONSOL_AMOEBA_POLISH_LR_SCALE
    teacher_old_loss_weight: float = TEACHER_OLD_LOSS_WEIGHT
    teacher_old_batch_period: int = TEACHER_OLD_BATCH_PERIOD
    batch_size: int = BATCH_SIZE
    eval_batch_size: int = EVAL_BATCH_SIZE
    consolidation_micro_batch_size: int = 0
    max_seq_len: int = MAX_SEQ_LEN
    grad_clip: float = GRAD_CLIP
    gradient_checkpointing: bool = GRADIENT_CHECKPOINTING
    run_controls: bool = True
    wikitext_eval_samples: int = WIKITEXT_EVAL_SAMPLES
    json_eval_samples: int = JSON_EVAL_SAMPLES
    reversal_eval_samples: int = REVERSAL_EVAL_SAMPLES
    sort_eval_samples: int = SORT_EVAL_SAMPLES
    eval_interval: int = 100
    log_interval: int = 25
    require_real_frontier: bool = True
    allow_ab_gate_bypass: bool = False
    proof_v2_b_attach_steps: int = PROOF_V2_B_ATTACH_STEPS
    proof_v2_b_attach_lr: float = PROOF_V2_B_ATTACH_LR
    proof_v2_b_rank: int = PROOF_V2_B_RANK
    proof_v2_b_alpha: float = PROOF_V2_B_ALPHA
    proof_v2_b_gate_init: float = PROOF_V2_B_GATE_INIT
    proof_v2_b_use_up_proj: bool = PROOF_V2_B_USE_UP_PROJ
    proof_v2_b_min_layers: int = PROOF_V2_B_MIN_LAYERS
    proof_v2_d_attach_steps: int = PROOF_V2_D_ATTACH_STEPS
    proof_v2_d_attach_lr: float = PROOF_V2_D_ATTACH_LR
    proof_v2_d_rank: int = PROOF_V2_D_RANK
    proof_v2_d_alpha: float = PROOF_V2_D_ALPHA
    proof_v2_d_gate_init: float = PROOF_V2_D_GATE_INIT
    proof_v2_d_use_up_proj: bool = PROOF_V2_D_USE_UP_PROJ
    proof_v2_d_min_layers: int = PROOF_V2_D_MIN_LAYERS
    adapter_config: LatentLoRAConfig = field(
        default_factory=lambda: LatentLoRAConfig(
            rank=16,
            alpha=32.0,
            dropout=0.0,
            projection_strength=1.0,
            gate_init=-6.0,
            freeze_base=True,
        )
    )


@dataclass
class PhaseResult:
    label: str
    metrics: Dict[str, float]
    checkpoint_path: Path | None
    saturation: SaturationReport | None
    wall_time: float
    step: int = 0
    tracked: Dict[str, "PhaseResult"] | None = None
    saturation_history: List[SaturationReport] = field(default_factory=list)
    gate_trajectory: List[Tuple[int, float]] = field(default_factory=list)
    post_expansion_saturation: SaturationReport | None = None
    expected_steps: int = 0
    completed: bool = False
    phase_version: int = PHASE_VERSION
    task_suite: str = "proof_v2"
    update_task: str | None = None


@dataclass
class ProofResult:
    seed: int
    base_a: PhaseResult
    teacher_b: PhaseResult
    base_ab: PhaseResult
    teacher_c: PhaseResult | None
    base_abc: PhaseResult | None
    tomography_d: TomographyResult
    fixed_teacher_d: PhaseResult
    fixed_frontier: PhaseResult
    fixed_final: PhaseResult
    saturation_history: List[SaturationReport]
    expanded_teacher_d: PhaseResult
    expanded_best_d: PhaseResult
    expanded_balanced: PhaseResult | None
    expanded_headline: PhaseResult
    controls: Dict[str, PhaseResult]


def _placeholder_phase(label: str) -> PhaseResult:
    return PhaseResult(label=label, metrics={}, checkpoint_path=None, saturation=None, wall_time=0.0, step=0)


class FrontierTracker:
    def __init__(
        self,
        new_key: str,
        retention_key: str,
        baseline_ppl: float,
        min_new_score: float = 0.10,
        dense_new_key: str | None = None,
        min_dense_score: float = 0.20,
    ) -> None:
        self.new_key = new_key
        self.retention_key = retention_key
        self.baseline_ppl = baseline_ppl
        self.min_new_score = min_new_score
        self.dense_new_key = dense_new_key
        self.min_dense_score = min_dense_score

    def _task_score(self, metrics: Dict[str, float]) -> tuple[float, bool]:
        exact_score = float(metrics.get(self.new_key, 0.0))
        dense_score = 0.0 if self.dense_new_key is None else float(metrics.get(self.dense_new_key, 0.0))
        viability = exact_score >= self.min_new_score or dense_score >= self.min_dense_score
        return max(exact_score, 0.5 * dense_score), viability

    def score(self, metrics: Dict[str, float]) -> float:
        new_score, viable = self._task_score(metrics)
        if not viable:
            return -1e9
        ppl = float(metrics.get(self.retention_key, float("inf")))
        retention = self.baseline_ppl / max(ppl, 1e-6)
        return min(new_score, retention) + 0.25 * new_score + 0.25 * retention

    def better(self, candidate: Dict[str, float], current: Dict[str, float] | None) -> bool:
        if current is None:
            return self.score(candidate) > -1e8
        return self.score(candidate) > self.score(current)


def _higher_metric(key: str) -> Callable[[Dict[str, float], Dict[str, float] | None], bool]:
    def _cmp(candidate: Dict[str, float], current: Dict[str, float] | None) -> bool:
        if current is None:
            return True
        return float(candidate.get(key, float("-inf"))) > float(current.get(key, float("-inf")))
    return _cmp


def _b_metric_cmp(candidate: Dict[str, float], current: Dict[str, float] | None) -> bool:
    def _score(metrics: Dict[str, float]) -> tuple[float, float, float, float]:
        return (
            float(metrics.get("json_train_field_acc", 0.0)),
            float(metrics.get("json_field_acc", 0.0)),
            float(metrics.get("json_train_valid", 0.0)),
            -float(metrics.get("json_loss", float("inf"))),
        )

    if current is None:
        return True
    return _score(candidate) > _score(current)


def _d_metric_cmp(candidate: Dict[str, float], current: Dict[str, float] | None) -> bool:
    def _score(metrics: Dict[str, float]) -> tuple[float, float, float, float, float]:
        return (
            float(metrics.get("sort_train_exact", 0.0)),
            float(metrics.get("sort_train_token_acc", 0.0)),
            float(metrics.get("sort_token_acc", 0.0)),
            -float(metrics.get("sort_train_loss", metrics.get("sort_loss", float("inf")))),
            -float(metrics.get("sort_loss", float("inf"))),
        )

    if current is None:
        return True
    return _score(candidate) > _score(current)


def _lower_metric(key: str) -> Callable[[Dict[str, float], Dict[str, float] | None], bool]:
    def _cmp(candidate: Dict[str, float], current: Dict[str, float] | None) -> bool:
        if current is None:
            return True
        return float(candidate.get(key, float("inf"))) < float(current.get(key, float("inf")))
    return _cmp


def _best_d_cmp(fixed_frontier: PhaseResult, new_key: str) -> Callable[[Dict[str, float], Dict[str, float] | None], bool]:
    frontier_ppl = float(fixed_frontier.metrics.get("wikitext_ppl", float("inf")))
    max_ppl = frontier_ppl * (1.0 + BEST_D_PPL_DEGRADATION)

    def _cmp(candidate: Dict[str, float], current: Dict[str, float] | None) -> bool:
        candidate_valid = float(candidate.get("wikitext_ppl", float("inf"))) <= max_ppl
        current_valid = current is not None and float(current.get("wikitext_ppl", float("inf"))) <= max_ppl
        if current is None:
            return True
        if candidate_valid and not current_valid:
            return True
        if not candidate_valid and current_valid:
            return False
        cand_new = float(candidate.get(new_key, float("-inf")))
        curr_new = float(current.get(new_key, float("-inf")))
        if cand_new != curr_new:
            return cand_new > curr_new
        cand_dense = float(candidate.get("sort_token_acc", float("-inf")))
        curr_dense = float(current.get("sort_token_acc", float("-inf")))
        if cand_dense != curr_dense:
            return cand_dense > curr_dense
        return float(candidate.get("wikitext_ppl", float("inf"))) < float(current.get("wikitext_ppl", float("inf")))

    return _cmp


def _balanced_cmp(fixed_frontier: PhaseResult, new_key: str) -> Callable[[Dict[str, float], Dict[str, float] | None], bool]:
    min_new = float(fixed_frontier.metrics.get(new_key, 0.0)) + BALANCED_D_MARGIN
    frontier_ppl = float(fixed_frontier.metrics.get("wikitext_ppl", float("inf")))

    def _cmp(candidate: Dict[str, float], current: Dict[str, float] | None) -> bool:
        cand_ok = (
            float(candidate.get(new_key, 0.0)) >= min_new
            and float(candidate.get("wikitext_ppl", float("inf"))) < frontier_ppl
        )
        curr_ok = current is not None and (
            float(current.get(new_key, 0.0)) >= min_new
            and float(current.get("wikitext_ppl", float("inf"))) < frontier_ppl
        )
        if current is None:
            return cand_ok
        if cand_ok and not curr_ok:
            return True
        if not cand_ok:
            return False
        curr_ppl = float(current.get("wikitext_ppl", float("inf")))
        cand_ppl = float(candidate.get("wikitext_ppl", float("inf")))
        if cand_ppl != curr_ppl:
            return cand_ppl < curr_ppl
        cand_dense = float(candidate.get("sort_token_acc", float("-inf")))
        curr_dense = float(current.get("sort_token_acc", float("-inf")))
        if cand_dense != curr_dense:
            return cand_dense > curr_dense
        return float(candidate.get(new_key, 0.0)) > float(current.get(new_key, 0.0))

    return _cmp


def _headline_choice(
    fixed_frontier: PhaseResult,
    best_d: PhaseResult,
    balanced: PhaseResult | None,
    new_key: str,
) -> PhaseResult:
    if balanced is not None:
        if (
            float(balanced.metrics.get("wikitext_ppl", float("inf")))
            < float(fixed_frontier.metrics.get("wikitext_ppl", float("inf")))
            and float(balanced.metrics.get(new_key, 0.0)) > float(fixed_frontier.metrics.get(new_key, 0.0))
        ):
            return balanced
    return best_d


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _clone_model(model: nn.Module, device: str) -> nn.Module:
    clone = copy.deepcopy(model)
    return clone.to(device)


def _freeze_model(model: nn.Module) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def _trainable_params(model: nn.Module) -> List[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def _save_checkpoint(model: nn.Module, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def _load_checkpoint(model: nn.Module, path: Path, device: str) -> nn.Module:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model.to(device)


def _materialize_phase_model(template: nn.Module, phase: PhaseResult, device: str) -> nn.Module:
    materialized = _clone_model(template, device)
    if phase.checkpoint_path is not None:
        _load_checkpoint(materialized, phase.checkpoint_path, device)
    return materialized


def _resolve_tracked_phase(
    primary: PhaseResult,
    tracked_key: str,
    *,
    fallback_label: str,
    fallback_metric: str,
) -> PhaseResult:
    if primary.tracked and tracked_key in primary.tracked:
        return primary.tracked[tracked_key]
    fallback = copy.deepcopy(primary)
    fallback.label = fallback_label
    fallback.metrics = dict(primary.metrics)
    fallback.metrics[fallback_metric] = 1.0
    return fallback


def _is_tuned_sort_phase_label(label: str) -> bool:
    if label in {
        "fixed_teacher_D",
        "fixed_ABCD",
        "expanded_teacher_D",
        "expanded_best_D",
        "expanded_balanced",
        "control_gate_null",
    }:
        return True
    return label.startswith("control_") and "teacher" in label


def _mark_sort_phase(phase: PhaseResult, phase_dir: Path, cfg: RuntimeConfig) -> None:
    if not _is_tuned_sort_phase_label(phase.label):
        return
    phase.metrics["sort_phase_tuned"] = 4.0
    phase.metrics["sort_phase_teacher_old_weight"] = float(cfg.teacher_old_loss_weight)
    phase.metrics["sort_phase_teacher_old_period"] = float(cfg.teacher_old_batch_period)
    phase.metrics["sort_phase_consol_old_period"] = float(cfg.consol_old_batch_period)
    phase.metrics["sort_phase_consol_amoeba_enabled"] = 1.0 if cfg.consol_amoeba_enabled else 0.0
    phase.metrics["sort_phase_consol_amoeba_gentle_frac"] = float(cfg.consol_amoeba_gentle_frac)
    phase.metrics["sort_phase_consol_amoeba_polish_frac"] = float(cfg.consol_amoeba_polish_frac)
    phase.metrics["sort_phase_consol_amoeba_gentle_old_scale"] = float(cfg.consol_amoeba_gentle_old_scale)
    phase.metrics["sort_phase_consol_amoeba_gentle_new_scale"] = float(cfg.consol_amoeba_gentle_new_scale)
    phase.metrics["sort_phase_consol_amoeba_polish_lr_scale"] = float(cfg.consol_amoeba_polish_lr_scale)
    _save_phase_result(phase_dir, phase, cfg)


def _prepare_supervised_batch(
    tokenizer,
    prompts: Sequence[str],
    targets: Sequence[str],
    device: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    rows: List[torch.Tensor] = []
    label_rows: List[torch.Tensor] = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    for prompt, target in zip(prompts, targets):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        if not target_ids and tokenizer.eos_token_id is not None:
            target_ids = [int(tokenizer.eos_token_id)]

        if len(target_ids) >= max_length:
            input_ids = target_ids[:max_length]
            labels = list(input_ids)
        else:
            prompt_budget = max(max_length - len(target_ids), 0)
            prompt_tail = prompt_ids[-prompt_budget:] if prompt_budget else []
            input_ids = prompt_tail + target_ids
            labels = [-100] * len(prompt_tail) + list(target_ids)

        rows.append(torch.tensor(input_ids, dtype=torch.long))
        label_rows.append(torch.tensor(labels, dtype=torch.long))

    batch_len = max((row.numel() for row in rows), default=0)
    input_batch = torch.full((len(rows), batch_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), batch_len), dtype=torch.long)
    labels_batch = torch.full((len(rows), batch_len), -100, dtype=torch.long)
    for row_idx, (input_ids, labels) in enumerate(zip(rows, label_rows)):
        length = int(input_ids.numel())
        input_batch[row_idx, :length] = input_ids
        attention_mask[row_idx, :length] = 1
        labels_batch[row_idx, :length] = labels
    return {
        "input_ids": input_batch.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels_batch.to(device),
    }


def _prepare_language_model_batch(
    tokenizer,
    texts: Sequence[str],
    device: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "labels": labels.to(device),
    }


def _chunk_texts(tokenizer, texts: Sequence[str], max_length: int, max_samples: int) -> List[torch.Tensor]:
    chunks: List[torch.Tensor] = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        if ids.numel() < 4:
            continue
        for start in range(0, ids.size(0), max_length):
            chunk = ids[start : start + max_length]
            if chunk.numel() < 4:
                continue
            chunks.append(chunk)
            if len(chunks) >= max_samples:
                return chunks
    return chunks[:max_samples]


def load_wikitext_texts(
    tokenizer,
    *,
    split: str,
    max_samples: int,
    max_seq_len: int,
    local_files_only: bool,
) -> List[torch.Tensor]:
    texts: List[str] = []
    if load_dataset is not None and not local_files_only:
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            texts = [row["text"] for row in dataset if row.get("text", "").strip()]
        except Exception:
            texts = []
    if not texts:
        texts = FALLBACK_WIKITEXT
    return _chunk_texts(tokenizer, texts, max_seq_len, max_samples)


@torch.no_grad()
def evaluate_retention(
    model,
    tokenizer,
    eval_chunks: List[torch.Tensor],
    device: str,
    eval_batch_size: int,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for chunk_batch in _iter_batches(eval_chunks, eval_batch_size):
        max_len = max(chunk.numel() for chunk in chunk_batch)
        input_ids = torch.full((len(chunk_batch), max_len), int(pad_token_id), device=device, dtype=torch.long)
        attention_mask = torch.zeros((len(chunk_batch), max_len), device=device, dtype=torch.long)
        labels = torch.full((len(chunk_batch), max_len), -100, device=device, dtype=torch.long)
        for row_idx, chunk in enumerate(chunk_batch):
            length = chunk.numel()
            input_ids[row_idx, :length] = chunk.to(device)
            attention_mask[row_idx, :length] = 1
            labels[row_idx, :length] = chunk.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        losses.append(float(outputs.loss.item()))
    mean_loss = float(sum(losses) / max(len(losses), 1))
    ppl = float(math.exp(min(mean_loss, 20.0)))
    model.train()
    return {"wikitext_loss": mean_loss, "wikitext_ppl": ppl}


def _sample_templates(rng: np.random.Generator, batch_size: int) -> List[Dict[str, str]]:
    idxs = rng.integers(0, len(JSON_TRAIN_TEMPLATES), size=batch_size)
    return [JSON_TRAIN_TEMPLATES[int(i)] for i in idxs]


def _proof_v2_nonce_token(index: int, prefix: str) -> str:
    alphabet = PROOF_V2_NONCE_ALPHABET
    base = len(alphabet)
    value = int(index)
    chars: List[str] = []
    for _ in range(4):
        chars.append(alphabet[value % base])
        value //= base
    return prefix + "".join(reversed(chars))


def _proof_v2_record_family(family_id: int, *, heldout: bool) -> Tuple[Tuple[str, ...], Dict[str, Tuple[str, ...]]]:
    split_offset = 50_000 if heldout else 0
    global_family_id = split_offset + int(family_id)
    fields = tuple(
        _proof_v2_nonce_token(global_family_id * 5 + field_idx, "F")
        for field_idx in range(3)
    )
    value_pool: Dict[str, Tuple[str, ...]] = {}
    for field_idx, field in enumerate(fields):
        values = tuple(
            _proof_v2_nonce_token(
                global_family_id * 37 + field_idx * PROOF_V2_RECORD_VALUE_POOL_SIZE + value_idx,
                "V",
            )
            for value_idx in range(PROOF_V2_RECORD_VALUE_POOL_SIZE)
        )
        value_pool[field] = values
    return fields, value_pool


def _proof_v2_record_example(
    rng: np.random.Generator,
    *,
    heldout: bool,
    family_id: int | None = None,
) -> Tuple[str, str]:
    prompt_styles = PROOF_V2_EVAL_RECORD_PROMPTS if heldout else PROOF_V2_TRAIN_RECORD_PROMPTS
    family_count = PROOF_V2_EVAL_RECORD_FAMILY_COUNT if heldout else PROOF_V2_TRAIN_RECORD_FAMILY_COUNT
    chosen_family = int(rng.integers(0, family_count)) if family_id is None else int(family_id) % family_count
    order, value_pool = _proof_v2_record_family(chosen_family, heldout=heldout)
    route_code, route_perm = PROOF_V2_ROUTE_CODES[int(rng.integers(0, len(PROOF_V2_ROUTE_CODES)))]
    assignments = {field: value_pool[field][int(rng.integers(0, len(value_pool[field])))] for field in order}
    shuffled_fields = list(order)
    rng.shuffle(shuffled_fields)
    assignment_text = " ; ".join(f"{field}={assignments[field]}" for field in shuffled_fields)
    prompt = prompt_styles[int(rng.integers(0, len(prompt_styles)))].format(
        route=route_code,
        fields=",".join(order),
        assignments=assignment_text,
    )
    routed_values = [assignments[order[idx]] for idx in route_perm]
    target = " ; ".join(
        f"{slot}={value}"
        for slot, value in zip(PROOF_V2_OUTPUT_SLOTS, routed_values)
    )
    return f"{prompt}\n", target


def _proof_v2_sequence_symbols(heldout: bool) -> Tuple[str, ...]:
    return PROOF_V2_EVAL_SEQUENCE_SYMBOLS if heldout else PROOF_V2_TRAIN_SEQUENCE_SYMBOLS


def _proof_v2_reversal_example(
    rng: np.random.Generator,
    seq_len: int,
    *,
    heldout: bool,
) -> Tuple[str, str]:
    symbols = _proof_v2_sequence_symbols(heldout)
    values = [str(symbols[int(rng.integers(0, len(symbols)))]) for _ in range(seq_len)]
    prompt = f"Braid VX chain: {','.join(values)} -> "
    target_values: List[str] = []
    for idx in range(0, len(values), 2):
        if idx + 1 < len(values):
            target_values.extend([values[idx + 1], values[idx]])
        else:
            target_values.append(values[idx])
    target = ",".join(target_values)
    return prompt, target


def _proof_v2_sort_item(symbol: str, key_token: str, tag: str) -> str:
    return f"{symbol}~{key_token}~{tag}"


def _proof_v2_sort_key(token: str) -> int:
    parts = [part.strip() for part in token.split("~")]
    if len(parts) < 3:
        return len(PROOF_V2_SORT_KEY_ORDER)
    try:
        return PROOF_V2_SORT_KEY_ORDER.index(parts[1])
    except ValueError:
        return len(PROOF_V2_SORT_KEY_ORDER)


def _proof_v2_sort_example(
    rng: np.random.Generator,
    seq_len: int,
    *,
    heldout: bool,
) -> Tuple[str, str]:
    symbols = _proof_v2_sequence_symbols(heldout)
    items: List[str] = []
    for idx in range(seq_len):
        symbol = str(symbols[int(rng.integers(0, len(symbols)))])
        key = str(PROOF_V2_SORT_KEY_ORDER[int(rng.integers(0, len(PROOF_V2_SORT_KEY_ORDER)))])
        tag = chr(ord("a") + (idx % 26))
        items.append(_proof_v2_sort_item(symbol, key, tag))
    prompt = f"StableSort VX packets: {','.join(items)} -> "
    target = ",".join(sorted(items, key=_proof_v2_sort_key))
    return prompt, target


def generate_json_batch(tokenizer, rng: np.random.Generator, device: str, cfg: RuntimeConfig) -> Dict[str, torch.Tensor]:
    if cfg.task_suite == "proof_v2":
        prompts: List[str] = []
        targets: List[str] = []
        for _ in range(cfg.batch_size):
            prompt, target = _proof_v2_record_example(rng, heldout=False)
            prompts.append(prompt)
            targets.append(f"{target}{tokenizer.eos_token}")
        return _prepare_supervised_batch(tokenizer, prompts, targets, device, cfg.max_seq_len)
    rows = _sample_templates(rng, cfg.batch_size)
    prompts = [f"### Input: {row['prompt']}\n### Output: " for row in rows]
    targets = [f"{row['target']}{tokenizer.eos_token}" for row in rows]
    return _prepare_supervised_batch(tokenizer, prompts, targets, device, cfg.max_seq_len)


def _sequence_to_text(values: Sequence[int]) -> str:
    return ",".join(str(v) for v in values)


def generate_reversal_batch(
    tokenizer,
    rng: np.random.Generator,
    device: str,
    cfg: RuntimeConfig,
    seq_len: int = 8,
) -> Dict[str, torch.Tensor]:
    if cfg.task_suite == "proof_v2":
        prompts: List[str] = []
        targets: List[str] = []
        for _ in range(cfg.batch_size):
            prompt, target = _proof_v2_reversal_example(rng, seq_len, heldout=False)
            prompts.append(prompt)
            targets.append(f"{target}{tokenizer.eos_token}")
        return _prepare_supervised_batch(tokenizer, prompts, targets, device, cfg.max_seq_len)
    prompts: List[str] = []
    targets: List[str] = []
    for _ in range(cfg.batch_size):
        values = rng.integers(0, 10, size=seq_len).tolist()
        prompt = f"Reverse: {_sequence_to_text(values)} -> "
        target = f"{_sequence_to_text(list(reversed(values)))}{tokenizer.eos_token}"
        prompts.append(prompt)
        targets.append(target)
    return _prepare_supervised_batch(tokenizer, prompts, targets, device, cfg.max_seq_len)


def generate_sort_batch(
    tokenizer,
    rng: np.random.Generator,
    device: str,
    cfg: RuntimeConfig,
    seq_len: int = 8,
) -> Dict[str, torch.Tensor]:
    if cfg.task_suite == "proof_v2":
        prompts: List[str] = []
        targets: List[str] = []
        for _ in range(cfg.batch_size):
            prompt, target = _proof_v2_sort_example(rng, seq_len, heldout=False)
            prompts.append(prompt)
            targets.append(f"{target}{tokenizer.eos_token}")
        return _prepare_supervised_batch(tokenizer, prompts, targets, device, cfg.max_seq_len)
    prompts: List[str] = []
    targets: List[str] = []
    for _ in range(cfg.batch_size):
        values = rng.integers(0, 100, size=seq_len).tolist()
        prompt = f"Sort: {_sequence_to_text(values)} -> "
        target = f"{_sequence_to_text(sorted(values))}{tokenizer.eos_token}"
        prompts.append(prompt)
        targets.append(target)
    return _prepare_supervised_batch(tokenizer, prompts, targets, device, cfg.max_seq_len)


def _curriculum_seq_len(step: int, total_steps: int) -> int:
    progress = float(step) / max(float(total_steps), 1.0)
    if progress < 0.30:
        return 8
    if progress < 0.65:
        return 10
    return 12


def _iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    size = max(int(batch_size), 1)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _compose_task_batch_fns(batch_fns: Sequence[Callable[[int], Dict[str, torch.Tensor]]]) -> Callable[[int], Dict[str, torch.Tensor]]:
    factories = [fn for fn in batch_fns if fn is not None]
    if not factories:
        raise ValueError("Need at least one batch factory to compose.")
    if len(factories) == 1:
        return factories[0]

    def _batch(step: int) -> Dict[str, torch.Tensor]:
        index = (max(int(step), 1) - 1) % len(factories)
        return factories[index](step)

    return _batch


def _split_tensor_batch(batch: Dict[str, torch.Tensor], micro_batch_size: int) -> List[Dict[str, torch.Tensor]]:
    if micro_batch_size <= 0:
        return [batch]
    first_tensor = next(iter(batch.values()))
    total = int(first_tensor.shape[0])
    if total <= micro_batch_size:
        return [batch]
    parts: List[Dict[str, torch.Tensor]] = []
    for start in range(0, total, micro_batch_size):
        end = min(start + micro_batch_size, total)
        parts.append({key: value[start:end] for key, value in batch.items()})
    return parts


def make_wikitext_batch_fn(
    tokenizer,
    chunks: List[torch.Tensor],
    device: str,
    cfg: RuntimeConfig,
    seed: int,
) -> Callable[[int], Dict[str, torch.Tensor]]:
    def _batch(step: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(seed + 1009 * int(step))
        idxs = rng.integers(0, len(chunks), size=cfg.batch_size)
        texts = [tokenizer.decode(chunks[int(i)], skip_special_tokens=False) for i in idxs]
        return _prepare_language_model_batch(tokenizer, texts, device, cfg.max_seq_len)

    return _batch


def _generate_tokens(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 64) -> str:
    return _generate_batch_tokens(model, tokenizer, [prompt], device, max_new_tokens=max_new_tokens)[0]


def _generate_batch_tokens(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: str,
    max_new_tokens: int = 64,
) -> List[str]:
    if not prompts:
        return []
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
    finally:
        tokenizer.padding_side = original_padding_side
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    # With left padding, attention_mask.sum() is the unpadded prompt length, not
    # the column where generation begins. Slice at the padded input width so
    # decoded completions do not include the tail of shorter prompts.
    prompt_width = int(inputs["input_ids"].shape[1])
    completions: List[str] = []
    for row_idx in range(len(prompts)):
        new_tokens = outputs[row_idx, prompt_width:]
        completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return completions


def _canonical_json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _parse_json_payload(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _parse_proof_v2_record_payload(text: str) -> Dict[str, str] | None:
    normalized = text.strip()
    if normalized.startswith("ZXREC|"):
        normalized = normalized[len("ZXREC|") :]
        segments = normalized.split("|")
    else:
        lowered = normalized.lower()
        cut_markers = [
            "\nanswer with ordered key=value pairs",
            "\nreturn canonical assignments",
            "\ncanonical order ",
            "\nassignments:",
            "\napply map ",
            "\nfield order ",
            "\nobserved bindings ",
            "\nemit lu1,ke2,zo3",
            "\nreturn lu1/ke2/zo3",
        ]
        for marker in cut_markers:
            idx = lowered.find(marker)
            if idx > 0:
                normalized = normalized[:idx]
                lowered = normalized.lower()
        normalized = normalized.replace("\r", "\n")
        segments = []
        for line in normalized.split("\n"):
            pieces = [piece.strip() for piece in line.split(";")]
            segments.extend(piece for piece in pieces if piece)
    payload: Dict[str, str] = {}
    for segment in segments:
        if not segment:
            continue
        if "=" not in segment:
            return None
        key, value = segment.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            return None
        if " " in value and "=" in value:
            value = value.split(" ", 1)[0].strip()
        payload[key] = value
    return payload if payload else None


def _json_field_accuracy(predicted: Any, target: Any) -> float:
    if not isinstance(target, dict):
        return float(predicted == target)
    if not isinstance(predicted, dict):
        return 0.0
    keys = list(target.keys())
    if not keys:
        return 1.0
    correct = 0
    for key in keys:
        if key in predicted and predicted[key] == target[key]:
            correct += 1
    return float(correct / len(keys))


def _evaluate_b_examples(
    model,
    tokenizer,
    device: str,
    rows: Sequence[Tuple[str, str]],
    eval_batch_size: int,
    *,
    parser: Callable[[str], Any | None],
) -> Dict[str, float]:
    exact = 0
    valid = 0
    field_acc = 0.0
    losses: List[float] = []
    for row_batch in _iter_batches(list(rows), eval_batch_size):
        prompts = [prompt for prompt, _ in row_batch]
        targets = [target for _, target in row_batch]
        completions = _generate_batch_tokens(model, tokenizer, prompts, device)
        for completion, target in zip(completions, targets):
            predicted = parser(completion)
            target_payload = parser(target)
            if predicted is not None:
                valid += 1
            if predicted is not None and target_payload is not None:
                if predicted == target_payload:
                    exact += 1
                field_acc += _json_field_accuracy(predicted, target_payload)
        batch = _prepare_supervised_batch(
            tokenizer,
            prompts,
            [f"{target}{tokenizer.eos_token}" for target in targets],
            device,
            256,
        )
        losses.append(float(model(**batch, use_cache=False).loss.item()))
    total = max(len(rows), 1)
    return {
        "exact": float(exact / total),
        "valid": float(valid / total),
        "field_acc": float(field_acc / total),
        "loss": float(sum(losses) / max(len(losses), 1)),
    }


@torch.no_grad()
def _collect_b_debug_examples(
    model,
    tokenizer,
    device: str,
    rows: Sequence[Tuple[str, str]],
    *,
    parser: Callable[[str], Any | None],
    limit: int = 6,
) -> List[Dict[str, Any]]:
    sample_rows = list(rows)[: max(int(limit), 0)]
    prompts = [prompt for prompt, _ in sample_rows]
    targets = [target for _, target in sample_rows]
    completions = _generate_batch_tokens(model, tokenizer, prompts, device)
    samples: List[Dict[str, Any]] = []
    for prompt, target, completion in zip(prompts, targets, completions):
        samples.append(
            {
                "prompt": prompt,
                "target": target,
                "completion": completion,
                "target_parsed": parser(target),
                "completion_parsed": parser(completion),
            }
        )
    return samples


@torch.no_grad()
def _write_b_debug_samples(
    model,
    tokenizer,
    device: str,
    phase_dir: Path,
    cfg: RuntimeConfig,
    *,
    limit: int = 6,
) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    if cfg.task_suite == "proof_v2":
        train_rows = [
            _proof_v2_record_example(np.random.default_rng(8100 + idx), heldout=False, family_id=idx)
            for idx in range(limit)
        ]
        heldout_rows = [
            _proof_v2_record_example(np.random.default_rng(9100 + idx), heldout=True, family_id=idx)
            for idx in range(limit)
        ]
        parser = _parse_proof_v2_record_payload
    else:
        train_rows = [
            (f"### Input: {row['prompt']}\n### Output: ", row["target"])
            for row in JSON_TRAIN_TEMPLATES[:limit]
        ]
        heldout_rows = [
            (f"### Input: {row['prompt']}\n### Output: ", row["target"])
            for row in JSON_EVAL_TEMPLATES[:limit]
        ]
        parser = _parse_json_payload
    payload = {
        "task_suite": cfg.task_suite,
        "phase_version": PHASE_VERSION,
        "train_samples": _collect_b_debug_examples(model, tokenizer, device, train_rows, parser=parser, limit=limit),
        "heldout_samples": _collect_b_debug_examples(model, tokenizer, device, heldout_rows, parser=parser, limit=limit),
    }
    path = phase_dir / "b_debug_samples.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    _sync_path_to_backup(path, cfg.output_dir, cfg.backup_dir)


@torch.no_grad()
def evaluate_json(
    model,
    tokenizer,
    device: str,
    num_samples: int,
    eval_batch_size: int,
    cfg: RuntimeConfig,
) -> Dict[str, float]:
    model.eval()
    if cfg.task_suite == "proof_v2":
        heldout_rows = [
            _proof_v2_record_example(np.random.default_rng(9100 + idx), heldout=True, family_id=idx)
            for idx in range(min(int(num_samples), 96))
        ]
        train_rows = [
            _proof_v2_record_example(np.random.default_rng(8100 + idx), heldout=False, family_id=idx)
            for idx in range(min(int(num_samples), 96))
        ]
        heldout_metrics = _evaluate_b_examples(
            model,
            tokenizer,
            device,
            heldout_rows,
            eval_batch_size,
            parser=_parse_proof_v2_record_payload,
        )
        train_metrics = _evaluate_b_examples(
            model,
            tokenizer,
            device,
            train_rows,
            eval_batch_size,
            parser=_parse_proof_v2_record_payload,
        )
        model.train()
        return {
            "json_exact_match": heldout_metrics["exact"],
            "json_valid": heldout_metrics["valid"],
            "json_field_acc": heldout_metrics["field_acc"],
            "json_loss": heldout_metrics["loss"],
            "json_train_exact_match": train_metrics["exact"],
            "json_train_valid": train_metrics["valid"],
            "json_train_field_acc": train_metrics["field_acc"],
            "b_exact_match": heldout_metrics["exact"],
            "b_valid": heldout_metrics["valid"],
            "b_field_acc": heldout_metrics["field_acc"],
            "b_loss": heldout_metrics["loss"],
            "b_train_exact_match": train_metrics["exact"],
            "b_train_valid": train_metrics["valid"],
            "b_train_field_acc": train_metrics["field_acc"],
            "b_zero_shot_guard_pass": 0.0,
        }
    heldout_rows = [
        (f"### Input: {row['prompt']}\n### Output: ", row["target"])
        for row in JSON_EVAL_TEMPLATES[: min(num_samples, len(JSON_EVAL_TEMPLATES))]
    ]
    train_rows = [
        (f"### Input: {row['prompt']}\n### Output: ", row["target"])
        for row in JSON_TRAIN_TEMPLATES[: min(num_samples, len(JSON_TRAIN_TEMPLATES))]
    ]
    heldout_metrics = _evaluate_b_examples(
        model,
        tokenizer,
        device,
        heldout_rows,
        eval_batch_size,
        parser=_parse_json_payload,
    )
    train_metrics = _evaluate_b_examples(
        model,
        tokenizer,
        device,
        train_rows,
        eval_batch_size,
        parser=_parse_json_payload,
    )
    model.train()
    return {
        "json_exact_match": heldout_metrics["exact"],
        "json_valid": heldout_metrics["valid"],
        "json_field_acc": heldout_metrics["field_acc"],
        "json_loss": heldout_metrics["loss"],
        "json_train_exact_match": train_metrics["exact"],
        "json_train_valid": train_metrics["valid"],
        "json_train_field_acc": train_metrics["field_acc"],
        "b_exact_match": heldout_metrics["exact"],
        "b_valid": heldout_metrics["valid"],
        "b_field_acc": heldout_metrics["field_acc"],
        "b_loss": heldout_metrics["loss"],
        "b_train_exact_match": train_metrics["exact"],
        "b_train_valid": train_metrics["valid"],
        "b_train_field_acc": train_metrics["field_acc"],
        "b_zero_shot_guard_pass": 0.0,
    }


def _evaluate_exact_sequence_task(
    model,
    tokenizer,
    device: str,
    num_samples: int,
    generator: Callable[[np.random.Generator, int], Tuple[str, str]],
    metric_prefix: str,
    eval_batch_size: int,
    eval_lengths: Sequence[int],
) -> Dict[str, float]:
    model.eval()
    rng = np.random.default_rng(12345)
    exact = 0
    token_acc = 0.0
    losses: List[float] = []
    lengths = list(eval_lengths) or [8]
    examples = [generator(rng, int(lengths[i % len(lengths)])) for i in range(num_samples)]
    for example_batch in _iter_batches(examples, eval_batch_size):
        prompts = [prompt for prompt, _ in example_batch]
        targets = [target for _, target in example_batch]
        target_token_budget = 48
        for target in targets:
            target_ids = tokenizer(target, return_tensors="pt", truncation=False)["input_ids"][0]
            target_token_budget = max(target_token_budget, int(target_ids.numel()) + 8)
        completions = _generate_batch_tokens(
            model,
            tokenizer,
            prompts,
            device,
            max_new_tokens=min(target_token_budget, 192),
        )
        for completion, target in zip(completions, targets):
            completion_tokens = completion.split(",") if completion else []
            target_tokens = target.split(",")
            if completion == target:
                exact += 1
            if target_tokens:
                correct = 0
                for pred, gold in zip(completion_tokens, target_tokens):
                    if pred.strip() == gold.strip():
                        correct += 1
                token_acc += correct / len(target_tokens)
        batch = _prepare_supervised_batch(
            tokenizer,
            prompts,
            [f"{target}{tokenizer.eos_token}" for target in targets],
            device,
            256,
        )
        losses.append(float(model(**batch, use_cache=False).loss.item()))
    model.train()
    total = max(num_samples, 1)
    return {
        f"{metric_prefix}_exact": float(exact / total),
        f"{metric_prefix}_token_acc": float(token_acc / total),
        f"{metric_prefix}_loss": float(sum(losses) / max(len(losses), 1)),
    }


@torch.no_grad()
def _evaluate_exact_sequence_task_dual(
    model,
    tokenizer,
    device: str,
    num_samples: int,
    train_generator: Callable[[np.random.Generator, int], Tuple[str, str]],
    heldout_generator: Callable[[np.random.Generator, int], Tuple[str, str]],
    metric_prefix: str,
    eval_batch_size: int,
    train_lengths: Sequence[int],
    heldout_lengths: Sequence[int],
) -> Dict[str, float]:
    heldout_metrics = _evaluate_exact_sequence_task(
        model,
        tokenizer,
        device,
        num_samples,
        heldout_generator,
        metric_prefix,
        eval_batch_size,
        heldout_lengths,
    )
    train_metrics = _evaluate_exact_sequence_task(
        model,
        tokenizer,
        device,
        num_samples,
        train_generator,
        f"{metric_prefix}_train",
        eval_batch_size,
        train_lengths,
    )
    return {**heldout_metrics, **train_metrics}


def _reversal_example(rng: np.random.Generator, seq_len: int, *, heldout: bool = False, task_suite: str = "legacy") -> Tuple[str, str]:
    if task_suite == "proof_v2":
        return _proof_v2_reversal_example(rng, seq_len, heldout=heldout)
    values = rng.integers(0, 10, size=seq_len).tolist()
    prompt = f"Reverse: {_sequence_to_text(values)} -> "
    target = _sequence_to_text(list(reversed(values)))
    return prompt, target


def _sort_example(rng: np.random.Generator, seq_len: int, *, heldout: bool = False, task_suite: str = "legacy") -> Tuple[str, str]:
    if task_suite == "proof_v2":
        return _proof_v2_sort_example(rng, seq_len, heldout=heldout)
    values = rng.integers(0, 100, size=seq_len).tolist()
    prompt = f"Sort: {_sequence_to_text(values)} -> "
    target = _sequence_to_text(sorted(values))
    return prompt, target


@torch.no_grad()
def evaluate_reversal(model, tokenizer, device: str, num_samples: int, eval_batch_size: int, cfg: RuntimeConfig) -> Dict[str, float]:
    return _evaluate_exact_sequence_task(
        model,
        tokenizer,
        device,
        num_samples,
        lambda rng, seq_len: _reversal_example(rng, seq_len, heldout=True, task_suite=cfg.task_suite),
        "reversal",
        eval_batch_size,
        eval_lengths=(14, 16),
    )


@torch.no_grad()
def evaluate_sort(model, tokenizer, device: str, num_samples: int, eval_batch_size: int, cfg: RuntimeConfig) -> Dict[str, float]:
    return _evaluate_exact_sequence_task_dual(
        model,
        tokenizer,
        device,
        num_samples,
        lambda rng, seq_len: _sort_example(rng, seq_len, heldout=False, task_suite=cfg.task_suite),
        lambda rng, seq_len: _sort_example(rng, seq_len, heldout=True, task_suite=cfg.task_suite),
        "sort",
        eval_batch_size,
        train_lengths=(8, 10, 12),
        heldout_lengths=(14, 16),
    )


def evaluate_world(
    model,
    tokenizer,
    tasks: List[str],
    eval_data: Dict[str, Any],
    device: str,
    cfg: RuntimeConfig,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if "retention" in tasks:
        metrics.update(evaluate_retention(model, tokenizer, eval_data["wikitext_val"], device, cfg.eval_batch_size))
    if "json" in tasks:
        metrics.update(evaluate_json(model, tokenizer, device, cfg.json_eval_samples, cfg.eval_batch_size, cfg))
    if "reversal" in tasks:
        metrics.update(evaluate_reversal(model, tokenizer, device, cfg.reversal_eval_samples, cfg.eval_batch_size, cfg))
    if "sort" in tasks:
        metrics.update(evaluate_sort(model, tokenizer, device, cfg.sort_eval_samples, cfg.eval_batch_size, cfg))
    return metrics


def kl_divergence(student_logits, teacher_logits, temperature: float = 2.0) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kl * (temperature ** 2)


def hidden_state_alignment(
    student_model,
    teacher_model,
    batch: Dict[str, torch.Tensor],
    layer_indices: List[int],
) -> torch.Tensor:
    student_outputs = student_model(**batch, output_hidden_states=True, use_cache=False)
    with torch.no_grad():
        teacher_outputs = teacher_model(**batch, output_hidden_states=True, use_cache=False)
    losses: List[torch.Tensor] = []
    for layer_index in layer_indices:
        idx = int(layer_index) + 1
        losses.append(F.mse_loss(student_outputs.hidden_states[idx], teacher_outputs.hidden_states[idx]))
    if not losses:
        return torch.tensor(0.0, device=batch["input_ids"].device)
    return torch.stack(losses).mean()


def _hidden_state_alignment_from_outputs(student_outputs, teacher_outputs, layer_indices: List[int], device: str) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for layer_index in layer_indices:
        idx = int(layer_index) + 1
        losses.append(F.mse_loss(student_outputs.hidden_states[idx], teacher_outputs.hidden_states[idx]))
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def _make_optimizer(params: List[nn.Parameter], lr: float) -> torch.optim.Optimizer:
    if not params:
        raise ValueError("No trainable parameters available for optimization.")
    return torch.optim.AdamW(params, lr=lr)


def _configure_gradient_checkpointing(model, enabled: bool) -> None:
    if hasattr(model, "gradient_checkpointing_enable") and hasattr(model, "gradient_checkpointing_disable"):
        if enabled:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()


def _release_cuda_memory(*models: nn.Module | None) -> None:
    for model in models:
        if model is None:
            continue
        if isinstance(model, nn.Module):
            try:
                model.to("cpu")
            except Exception:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def _run_tracker_update(
    tracker_results: Dict[str, PhaseResult],
    tracker_cmps: Dict[str, Callable[[Dict[str, float], Dict[str, float] | None], bool]],
    metrics: Dict[str, float],
    model: nn.Module,
    ckpt_dir: Path,
    label: str,
    step: int,
    wall_time: float,
    saturation: SaturationReport | None,
    cfg: RuntimeConfig | None = None,
) -> None:
    for name, cmp_fn in tracker_cmps.items():
        current_metrics = tracker_results[name].metrics if name in tracker_results else None
        if cmp_fn(metrics, current_metrics):
            ckpt_path = _save_checkpoint(model, ckpt_dir / f"{label}_{name}.pt")
            if cfg is not None:
                _sync_path_to_backup(ckpt_path, cfg.output_dir, cfg.backup_dir)
            tracker_results[name] = PhaseResult(
                label=f"{label}_{name}",
                metrics=dict(metrics),
                checkpoint_path=ckpt_path,
                saturation=saturation,
                wall_time=wall_time,
                step=step,
                completed=False,
                phase_version=PHASE_VERSION,
                task_suite="proof_v2" if cfg is None else cfg.task_suite,
                update_task=None if cfg is None else cfg.update_task,
            )


def train_adapter_teacher(
    model,
    tokenizer,
    attached: List[Tuple[str, LatentLoRALinear]],
    task_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    old_task_batch_fn: Callable[[int], Dict[str, torch.Tensor]] | None,
    escape_schedule: EscapeSchedule,
    eval_tasks: List[str],
    eval_data: Dict[str, Any],
    steps: int,
    lr: float,
    label: str,
    output_dir: Path,
    cfg: RuntimeConfig,
    tracker_cmps: Dict[str, Callable[[Dict[str, float], Dict[str, float] | None], bool]],
    saturation_old_profiles: List[TaskProfile] | None = None,
    saturation_layer_indices: List[int] | None = None,
    saturation_probe_batch_fn: Callable[[int], Dict[str, torch.Tensor]] | None = None,
) -> PhaseResult:
    params = _trainable_params(model)
    optimizer = _make_optimizer(params, lr)
    tracker_results: Dict[str, PhaseResult] = {}
    start = time.time()
    saturation_history: List[SaturationReport] = []
    gate_trajectory: List[Tuple[int, float]] = []
    progress_key = "sort_train_token_acc" if "sort" in eval_tasks else "reversal_exact" if "reversal" in eval_tasks else "json_train_field_acc"
    prev_progress_score: float | None = None
    resume_state = _load_latest_state(
        output_dir,
        label=label,
        model=model,
        optimizer=optimizer,
        expected_steps=steps,
        device=cfg.device,
        task_suite=cfg.task_suite,
        update_task=cfg.update_task,
    )
    start_step = 1
    if resume_state is not None:
        start_step = int(resume_state["step"]) + 1
        tracker_results = {
            name: _deserialize_phase(value)
            for name, value in (resume_state.get("tracker_results") or {}).items()
        }
        saturation_history = [
            saturation
            for saturation in (
                _deserialize_saturation(item) for item in resume_state.get("saturation_history", [])
            )
            if saturation is not None
        ]
        gate_trajectory = [(int(step), float(value)) for step, value in resume_state.get("gate_trajectory", [])]
        prev_progress_score = resume_state.get("prev_progress_score")
    model.train()
    if cfg.device.startswith("cuda"):
        _configure_gradient_checkpointing(model, cfg.gradient_checkpointing)

    iterator: Iterable[int] = range(start_step, steps + 1)
    if tqdm is not None:
        iterator = tqdm(iterator, desc=label, leave=False)
    for step in iterator:
        _set_gated_layer_steps(model, step)
        optimizer.zero_grad(set_to_none=True)
        batch = task_batch_fn(step)
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        teacher_old_period = int(cfg.teacher_old_batch_period)
        teacher_old_weight = float(cfg.teacher_old_loss_weight)
        if (
            old_task_batch_fn is not None
            and teacher_old_period > 0
            and teacher_old_weight > 0.0
            and step % teacher_old_period == 0
        ):
            old_batch = old_task_batch_fn(step)
            old_outputs = model(**old_batch, use_cache=False)
            loss = loss + old_outputs.loss * teacher_old_weight
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        escape_schedule.apply_to_modules(attached, step, steps)

        if step % cfg.eval_interval == 0 or step == steps:
            metrics = evaluate_world(model, tokenizer, eval_tasks, eval_data, cfg.device, cfg)
            saturation = None
            current_progress_score = float(metrics.get(progress_key, 0.0))
            task_progress_delta = 1.0 if prev_progress_score is None else current_progress_score - prev_progress_score
            prev_progress_score = current_progress_score
            if saturation_old_profiles and saturation_layer_indices and saturation_probe_batch_fn is not None:
                probe_batches = [saturation_probe_batch_fn(step + offset) for offset in range(min(4, cfg.batch_size))]
                retention_delta = (
                    metrics.get("wikitext_ppl", 0.0) / max(eval_data["baseline_retention"]["wikitext_ppl"], 1e-6) - 1.0
                )
                saturation = compute_saturation_report(
                    model,
                    tokenizer,
                    probe_batches,
                    saturation_old_profiles,
                    saturation_layer_indices,
                    step,
                    label,
                    saturation_history,
                    retention_delta=retention_delta,
                    task_progress_delta=task_progress_delta,
                )
                saturation_history.append(saturation)
            gate_value = _extract_gate_value(model)
            if gate_value is not None:
                gate_trajectory.append((step, gate_value))
            _run_tracker_update(
                tracker_results,
                tracker_cmps,
                metrics,
                model,
                output_dir,
                label,
                step,
                time.time() - start,
                saturation,
                cfg,
            )
            _save_latest_state(
                output_dir,
                label=label,
                model=model,
                optimizer=optimizer,
                step=step,
                expected_steps=steps,
                tracker_results=tracker_results,
                saturation_history=saturation_history,
                gate_trajectory=gate_trajectory,
                prev_progress_score=prev_progress_score,
                cfg=cfg,
            )
    primary = tracker_results.get("primary")
    if primary is None:
        metrics = evaluate_world(model, tokenizer, eval_tasks, eval_data, cfg.device, cfg)
        ckpt = _save_checkpoint(model, output_dir / f"{label}_primary.pt")
        _sync_path_to_backup(ckpt, cfg.output_dir, cfg.backup_dir)
        primary = PhaseResult(
            label=label,
            metrics=metrics,
            checkpoint_path=ckpt,
            saturation=None,
            wall_time=time.time() - start,
            step=steps,
            expected_steps=steps,
            completed=True,
            phase_version=PHASE_VERSION,
            task_suite=cfg.task_suite,
            update_task=cfg.update_task,
        )
    primary.tracked = {key: value for key, value in tracker_results.items() if key != "primary"}
    primary.saturation_history = list(saturation_history)
    primary.gate_trajectory = list(gate_trajectory)
    primary.expected_steps = steps
    primary.step = max(primary.step, steps)
    primary.completed = True
    primary.phase_version = PHASE_VERSION
    primary.task_suite = cfg.task_suite
    primary.update_task = cfg.update_task
    for tracked in primary.tracked.values():
        tracked.saturation_history = list(saturation_history)
        tracked.gate_trajectory = list(gate_trajectory)
        tracked.expected_steps = steps
        tracked.phase_version = PHASE_VERSION
        tracked.task_suite = cfg.task_suite
        tracked.update_task = cfg.update_task
    _save_phase_result(output_dir, primary, cfg)
    _clear_latest_state(output_dir, cfg)
    return primary


def dual_teacher_consolidation(
    student,
    teacher_old,
    teacher_new,
    tokenizer,
    new_task_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    old_task_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    eval_tasks: List[str],
    eval_data: Dict[str, Any],
    selected_layers: List[int],
    steps: int,
    lr: float,
    label: str,
    output_dir: Path,
    cfg: RuntimeConfig,
    tracker_cmps: Dict[str, Callable[[Dict[str, float], Dict[str, float] | None], bool]],
) -> PhaseResult:
    _freeze_model(teacher_old)
    _freeze_model(teacher_new)
    _unfreeze_model(student)
    params = _trainable_params(student)
    optimizer = _make_optimizer(params, lr)
    tracker_results: Dict[str, PhaseResult] = {}
    start = time.time()
    gate_trajectory: List[Tuple[int, float]] = []
    resume_state = _load_latest_state(
        output_dir,
        label=label,
        model=student,
        optimizer=optimizer,
        expected_steps=steps,
        device=cfg.device,
        task_suite=cfg.task_suite,
        update_task=cfg.update_task,
    )
    start_step = 1
    if resume_state is not None:
        start_step = int(resume_state["step"]) + 1
        tracker_results = {
            name: _deserialize_phase(value)
            for name, value in (resume_state.get("tracker_results") or {}).items()
        }
        gate_trajectory = [(int(step), float(value)) for step, value in resume_state.get("gate_trajectory", [])]
    student.train()
    if cfg.device.startswith("cuda"):
        _configure_gradient_checkpointing(student, cfg.gradient_checkpointing)

    def _set_optimizer_lr_scale(scale: float) -> None:
        for group in optimizer.param_groups:
            group["lr"] = lr * float(scale)

    def _amoeba_phase(step_index: int) -> tuple[float, float, float]:
        gentle_frac = min(max(float(cfg.consol_amoeba_gentle_frac), 0.0), 1.0)
        polish_frac = min(max(float(cfg.consol_amoeba_polish_frac), 0.0), 1.0)
        gentle_steps = min(int(round(steps * gentle_frac)), steps)
        polish_steps = min(int(round(steps * polish_frac)), max(steps - gentle_steps, 0))
        moderate_cutoff = max(steps - polish_steps, gentle_steps)
        if step_index <= gentle_steps:
            return (
                max(float(cfg.consol_amoeba_gentle_old_scale), 0.0),
                max(float(cfg.consol_amoeba_gentle_new_scale), 0.0),
                1.0,
            )
        if step_index > moderate_cutoff:
            return (1.0, 1.0, max(float(cfg.consol_amoeba_polish_lr_scale), 0.0))
        return (1.0, 1.0, 1.0)

    iterator: Iterable[int] = range(start_step, steps + 1)
    if tqdm is not None:
        iterator = tqdm(iterator, desc=label, leave=False)
    for step in iterator:
        _set_gated_layer_steps(student, step)
        optimizer.zero_grad(set_to_none=True)
        if cfg.consol_amoeba_enabled:
            old_scale, new_scale, lr_scale = _amoeba_phase(step)
            _set_optimizer_lr_scale(lr_scale)
            old_batch = old_task_batch_fn(step)
            new_batch = new_task_batch_fn(step)
            total_examples = max(
                int(old_batch["input_ids"].shape[0]) + int(new_batch["input_ids"].shape[0]),
                1,
            )
            paired = (
                (old_batch, teacher_old, float(cfg.consol_old_kl_weight) * old_scale),
                (new_batch, teacher_new, float(cfg.consol_kl_weight) * new_scale),
            )
            for batch, teacher, kl_weight in paired:
                micro_batches = _split_tensor_batch(batch, cfg.consolidation_micro_batch_size)
                batch_examples = max(int(batch["input_ids"].shape[0]), 1)
                for micro_batch in micro_batches:
                    weight = float(micro_batch["input_ids"].shape[0]) / float(total_examples)
                    student_outputs = student(**micro_batch, output_hidden_states=True, use_cache=False)
                    with torch.no_grad():
                        teacher_outputs = teacher(**micro_batch, output_hidden_states=True, use_cache=False)
                    ce_loss = student_outputs.loss
                    kl_loss = kl_divergence(student_outputs.logits, teacher_outputs.logits)
                    hidden_loss = _hidden_state_alignment_from_outputs(
                        student_outputs,
                        teacher_outputs,
                        selected_layers,
                        micro_batch["input_ids"].device,
                    )
                    loss = (ce_loss + kl_weight * kl_loss + float(cfg.consol_hidden_weight) * hidden_loss) * weight
                    loss.backward()
            old_step = True
        else:
            _set_optimizer_lr_scale(1.0)
            old_period = int(cfg.consol_old_batch_period)
            old_step = old_period > 0 and (step % old_period) == 1
            batch = old_task_batch_fn(step) if old_step else new_task_batch_fn(step)
            teacher = teacher_old if old_step else teacher_new
            micro_batches = _split_tensor_batch(batch, cfg.consolidation_micro_batch_size)
            total_examples = max(int(batch["input_ids"].shape[0]), 1)
            kl_weight = float(cfg.consol_old_kl_weight) if old_step else float(cfg.consol_kl_weight)
            for micro_batch in micro_batches:
                weight = float(micro_batch["input_ids"].shape[0]) / float(total_examples)
                student_outputs = student(**micro_batch, output_hidden_states=True, use_cache=False)
                with torch.no_grad():
                    teacher_outputs = teacher(**micro_batch, output_hidden_states=True, use_cache=False)
                ce_loss = student_outputs.loss
                kl_loss = kl_divergence(student_outputs.logits, teacher_outputs.logits)
                hidden_loss = _hidden_state_alignment_from_outputs(
                    student_outputs,
                    teacher_outputs,
                    selected_layers,
                    micro_batch["input_ids"].device,
                )
                loss = (ce_loss + kl_weight * kl_loss + float(cfg.consol_hidden_weight) * hidden_loss) * weight
                loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == steps:
            metrics = evaluate_world(student, tokenizer, eval_tasks, eval_data, cfg.device, cfg)
            gate_value = _extract_gate_value(student)
            if gate_value is not None:
                gate_trajectory.append((step, gate_value))
            _run_tracker_update(
                tracker_results,
                tracker_cmps,
                metrics,
                student,
                output_dir,
                label,
                step,
                time.time() - start,
                None,
                cfg,
            )
            _save_latest_state(
                output_dir,
                label=label,
                model=student,
                optimizer=optimizer,
                step=step,
                expected_steps=steps,
                tracker_results=tracker_results,
                saturation_history=[],
                gate_trajectory=gate_trajectory,
                prev_progress_score=None,
                cfg=cfg,
            )

    primary = tracker_results.get("primary")
    if primary is None:
        metrics = evaluate_world(student, tokenizer, eval_tasks, eval_data, cfg.device, cfg)
        ckpt = _save_checkpoint(student, output_dir / f"{label}_primary.pt")
        _sync_path_to_backup(ckpt, cfg.output_dir, cfg.backup_dir)
        primary = PhaseResult(
            label=label,
            metrics=metrics,
            checkpoint_path=ckpt,
            saturation=None,
            wall_time=time.time() - start,
            step=steps,
            expected_steps=steps,
            completed=True,
            phase_version=PHASE_VERSION,
            task_suite=cfg.task_suite,
            update_task=cfg.update_task,
        )
    primary.tracked = {key: value for key, value in tracker_results.items() if key != "primary"}
    primary.gate_trajectory = list(gate_trajectory)
    primary.expected_steps = steps
    primary.step = max(primary.step, steps)
    primary.completed = True
    primary.phase_version = PHASE_VERSION
    primary.task_suite = cfg.task_suite
    primary.update_task = cfg.update_task
    for tracked in primary.tracked.values():
        tracked.gate_trajectory = list(gate_trajectory)
        tracked.expected_steps = steps
        tracked.phase_version = PHASE_VERSION
        tracked.task_suite = cfg.task_suite
        tracked.update_task = cfg.update_task
    _save_phase_result(output_dir, primary, cfg)
    _clear_latest_state(output_dir, cfg)
    return primary


def base_only_polish(
    model,
    tokenizer,
    new_task_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    old_task_batch_fn: Callable[[int], Dict[str, torch.Tensor]],
    eval_tasks: List[str],
    eval_data: Dict[str, Any],
    steps: int,
    lr: float,
    label: str,
    output_dir: Path,
    cfg: RuntimeConfig,
    tracker_cmps: Dict[str, Callable[[Dict[str, float], Dict[str, float] | None], bool]],
) -> PhaseResult:
    _unfreeze_model(model)
    params = _trainable_params(model)
    optimizer = _make_optimizer(params, lr)
    tracker_results: Dict[str, PhaseResult] = {}
    start = time.time()
    resume_state = _load_latest_state(
        output_dir,
        label=label,
        model=model,
        optimizer=optimizer,
        expected_steps=steps,
        device=cfg.device,
        task_suite=cfg.task_suite,
        update_task=cfg.update_task,
    )
    start_step = 1
    if resume_state is not None:
        start_step = int(resume_state["step"]) + 1
        tracker_results = {
            name: _deserialize_phase(value)
            for name, value in (resume_state.get("tracker_results") or {}).items()
        }
    if cfg.device.startswith("cuda"):
        _configure_gradient_checkpointing(model, cfg.gradient_checkpointing)
    iterator: Iterable[int] = range(start_step, steps + 1)
    if tqdm is not None:
        iterator = tqdm(iterator, desc=label, leave=False)
    for step in iterator:
        optimizer.zero_grad(set_to_none=True)
        batch = old_task_batch_fn(step) if step % 2 == 0 else new_task_batch_fn(step)
        loss = model(**batch, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        if step % cfg.eval_interval == 0 or step == steps:
            metrics = evaluate_world(model, tokenizer, eval_tasks, eval_data, cfg.device, cfg)
            _run_tracker_update(
                tracker_results,
                tracker_cmps,
                metrics,
                model,
                output_dir,
                label,
                step,
                time.time() - start,
                None,
                cfg,
            )
            _save_latest_state(
                output_dir,
                label=label,
                model=model,
                optimizer=optimizer,
                step=step,
                expected_steps=steps,
                tracker_results=tracker_results,
                saturation_history=[],
                gate_trajectory=[],
                prev_progress_score=None,
                cfg=cfg,
            )
    primary = tracker_results.get("primary")
    if primary is None:
        metrics = evaluate_world(model, tokenizer, eval_tasks, eval_data, cfg.device, cfg)
        ckpt = _save_checkpoint(model, output_dir / f"{label}_primary.pt")
        _sync_path_to_backup(ckpt, cfg.output_dir, cfg.backup_dir)
        primary = PhaseResult(
            label=label,
            metrics=metrics,
            checkpoint_path=ckpt,
            saturation=None,
            wall_time=time.time() - start,
            step=steps,
            expected_steps=steps,
            completed=True,
            phase_version=PHASE_VERSION,
            task_suite=cfg.task_suite,
            update_task=cfg.update_task,
        )
    primary.tracked = {key: value for key, value in tracker_results.items() if key != "primary"}
    primary.expected_steps = steps
    primary.step = max(primary.step, steps)
    primary.completed = True
    primary.phase_version = PHASE_VERSION
    primary.task_suite = cfg.task_suite
    primary.update_task = cfg.update_task
    for tracked in primary.tracked.values():
        tracked.expected_steps = steps
        tracked.phase_version = PHASE_VERSION
        tracked.task_suite = cfg.task_suite
        tracked.update_task = cfg.update_task
    _save_phase_result(output_dir, primary, cfg)
    _clear_latest_state(output_dir, cfg)
    return primary


class GatedQwenLayer(nn.Module):
    def __init__(self, source_layer: nn.Module, gate_init: float = EXPANSION_GATE_INIT, gate_floors: Tuple[float, float, float] = EXPANSION_GATE_FLOORS) -> None:
        super().__init__()
        self.layer = copy.deepcopy(source_layer)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.gate_floors = gate_floors
        self._step = 0
        self.freeze_gate = False

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            modules = self.__dict__.get("_modules", {})
            wrapped = modules.get("layer")
            if wrapped is not None and hasattr(wrapped, name):
                return getattr(wrapped, name)
            raise exc

    def set_step(self, step: int) -> None:
        self._step = int(step)

    @property
    def gate_value(self) -> torch.Tensor:
        if self._step < 400:
            floor = self.gate_floors[0]
        elif self._step < 800:
            floor = self.gate_floors[1]
        else:
            floor = self.gate_floors[2]
        raw = self.gate.detach() if self.freeze_gate else self.gate
        clamped = torch.clamp(raw, min=floor)
        if self.freeze_gate:
            return clamped
        return raw + (clamped - raw).detach()

    def forward(self, hidden_states, **kwargs):
        identity = hidden_states.clone()
        layer_out = self.layer(hidden_states, **kwargs)
        if isinstance(layer_out, tuple):
            residual_out = layer_out[0]
        else:
            residual_out = layer_out
        contribution = residual_out - identity
        gated = identity + self.gate_value * contribution
        if isinstance(layer_out, tuple):
            return (gated,) + layer_out[1:]
        return gated


def _reindex_decoder_layers(model: nn.Module) -> None:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return
    for idx, layer in enumerate(layers):
        target = layer.layer if isinstance(layer, GatedQwenLayer) else layer
        if hasattr(target, "layer_idx"):
            target.layer_idx = idx
        self_attn = getattr(target, "self_attn", None)
        if self_attn is not None and hasattr(self_attn, "layer_idx"):
            self_attn.layer_idx = idx


def insert_expansion_layer(
    model,
    insert_after: int,
    gate_init: float = EXPANSION_GATE_INIT,
    gate_floors: Tuple[float, float, float] = EXPANSION_GATE_FLOORS,
) -> Tuple[nn.Module, GatedQwenLayer]:
    source = model.model.layers[insert_after]
    gated_layer = GatedQwenLayer(source, gate_init=gate_init, gate_floors=gate_floors)
    layers = list(model.model.layers)
    layers.insert(insert_after + 1, gated_layer)
    model.model.layers = nn.ModuleList(layers)
    _reindex_decoder_layers(model)
    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = len(model.model.layers)
    return model, gated_layer


def _iter_gated_layers(model: nn.Module) -> Iterable[GatedQwenLayer]:
    for module in model.modules():
        if isinstance(module, GatedQwenLayer):
            yield module


def _set_gated_layer_steps(model: nn.Module, step: int) -> None:
    for module in _iter_gated_layers(model):
        module.set_step(step)


def _extract_gate_value(model: nn.Module) -> float | None:
    for module in _iter_gated_layers(model):
        return float(module.gate_value.item())
    return None


def _seed_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _phase_output_dir(output_dir: Path, label: str) -> Path:
    path = output_dir / label
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fsync_parent(path: Path) -> None:
    try:
        parent_fd = os.open(str(path.parent), os.O_RDONLY)
    except Exception:
        return
    try:
        os.fsync(parent_fd)
    except Exception:
        pass
    finally:
        os.close(parent_fd)


def _atomic_copy2(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temp_path)
    try:
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
    except Exception:
        pass
    os.replace(temp_path, destination)
    _fsync_parent(destination)


def _sync_path_to_backup(path: Path, output_root: Path, backup_root: Path | None) -> None:
    if backup_root is None or not path.exists():
        return
    try:
        relative = path.resolve().relative_to(output_root.resolve())
    except Exception:
        return
    destination = backup_root / relative
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if path.is_dir():
                shutil.copytree(path, destination, dirs_exist_ok=True)
                _fsync_parent(destination)
            else:
                _atomic_copy2(path, destination)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1.0 + attempt)
    if last_error is not None:
        raise last_error


def _restore_output_from_backup(cfg: RuntimeConfig) -> None:
    if not cfg.resume or cfg.backup_dir is None or not cfg.backup_dir.exists():
        return
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cfg.backup_dir, cfg.output_dir, dirs_exist_ok=True)


def _phase_latest_state_path(phase_dir: Path) -> Path:
    return phase_dir / "latest_state.pt"


def _capture_rng_state() -> Dict[str, Any]:
    return {
        "torch": torch.random.get_rng_state(),
        "numpy": np.random.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(payload: Dict[str, Any] | None) -> None:
    if not payload:
        return
    if payload.get("torch") is not None:
        torch.random.set_rng_state(payload["torch"])
    if payload.get("numpy") is not None:
        np.random.set_state(payload["numpy"])
    if torch.cuda.is_available() and payload.get("cuda") is not None:
        torch.cuda.set_rng_state_all(payload["cuda"])


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _save_latest_state(
    phase_dir: Path,
    *,
    label: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    expected_steps: int,
    tracker_results: Dict[str, PhaseResult],
    saturation_history: List[SaturationReport],
    gate_trajectory: List[Tuple[int, float]],
    prev_progress_score: float | None,
    cfg: RuntimeConfig | None = None,
) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    latest_state = {
        "phase_version": PHASE_VERSION,
        "label": label,
        "task_suite": None if cfg is None else cfg.task_suite,
        "update_task": None if cfg is None else cfg.update_task,
        "step": int(step),
        "expected_steps": int(expected_steps),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "tracker_results": {name: _serialize_phase(value) for name, value in tracker_results.items()},
        "saturation_history": [_serialize_saturation(item) for item in saturation_history],
        "gate_trajectory": list(gate_trajectory),
        "prev_progress_score": prev_progress_score,
        "rng_state": _capture_rng_state(),
    }
    latest_path = _phase_latest_state_path(phase_dir)
    torch.save(latest_state, latest_path)
    if cfg is not None:
        _sync_path_to_backup(latest_path, cfg.output_dir, cfg.backup_dir)


def _load_latest_state(
    phase_dir: Path,
    *,
    label: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_steps: int,
    device: str,
    task_suite: str,
    update_task: str | None,
) -> Dict[str, Any] | None:
    latest_path = _phase_latest_state_path(phase_dir)
    if not latest_path.exists():
        return None
    payload = torch.load(latest_path, map_location=device)
    if int(payload.get("phase_version", -1)) != PHASE_VERSION:
        return None
    if str(payload.get("label")) != label:
        return None
    if payload.get("task_suite") != task_suite:
        return None
    if payload.get("update_task") != update_task:
        return None
    if int(payload.get("expected_steps", -1)) != int(expected_steps):
        return None
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    _optimizer_to_device(optimizer, device)
    _restore_rng_state(payload.get("rng_state"))
    return payload


def _clear_latest_state(phase_dir: Path, cfg: RuntimeConfig | None = None) -> None:
    latest_path = _phase_latest_state_path(phase_dir)
    if latest_path.exists():
        latest_path.unlink()
    if cfg is not None and cfg.backup_dir is not None:
        backup_latest = cfg.backup_dir / latest_path.relative_to(cfg.output_dir)
        if backup_latest.exists():
            backup_latest.unlink()


def _task_batch_factory(task_name: str, tokenizer, device: str, cfg: RuntimeConfig, seed: int) -> Callable[[int], Dict[str, torch.Tensor]]:
    total_steps = max(cfg.attach_steps, cfg.consolidation_steps, cfg.polish_steps, 1)
    if cfg.task_suite == "proof_v2" and task_name == "json":
        total_steps = max(total_steps, int(cfg.proof_v2_b_attach_steps))
    if cfg.task_suite == "proof_v2" and task_name == "sort":
        total_steps = max(total_steps, int(cfg.proof_v2_d_attach_steps))
    if task_name == "json":
        return lambda step: generate_json_batch(tokenizer, _seed_rng(seed + 2003 * int(step)), device, cfg)
    if task_name == "reversal":
        return lambda step: generate_reversal_batch(
            tokenizer,
            _seed_rng(seed + 3001 * int(step)),
            device,
            cfg,
            seq_len=_curriculum_seq_len(step, total_steps),
        )
    if task_name == "sort":
        return lambda step: generate_sort_batch(
            tokenizer,
            _seed_rng(seed + 4001 * int(step)),
            device,
            cfg,
            seq_len=_curriculum_seq_len(step, total_steps),
        )
    raise ValueError(f"unsupported task: {task_name}")


def _task_attach_steps(task_name: str, cfg: RuntimeConfig) -> int:
    if cfg.task_suite == "proof_v2" and task_name == "json":
        return max(int(cfg.attach_steps), int(cfg.proof_v2_b_attach_steps))
    if cfg.task_suite == "proof_v2" and task_name == "sort":
        return max(int(cfg.attach_steps), int(cfg.proof_v2_d_attach_steps))
    return int(cfg.attach_steps)


def _task_attach_lr(task_name: str, cfg: RuntimeConfig) -> float:
    if cfg.task_suite == "proof_v2" and task_name == "json":
        return max(float(cfg.attach_lr), float(cfg.proof_v2_b_attach_lr))
    if cfg.task_suite == "proof_v2" and task_name == "sort":
        return max(float(cfg.attach_lr), float(cfg.proof_v2_d_attach_lr))
    return float(cfg.attach_lr)


def _task_target_suffixes(task_name: str, cfg: RuntimeConfig) -> List[str]:
    suffixes = list(TARGET_SUFFIX)
    if cfg.task_suite == "proof_v2" and task_name == "json" and cfg.proof_v2_b_use_up_proj:
        suffixes.append("mlp.up_proj")
    if cfg.task_suite == "proof_v2" and task_name == "sort" and cfg.proof_v2_d_use_up_proj:
        suffixes.append("mlp.up_proj")
    return suffixes


def _task_adapter_config(task_name: str, cfg: RuntimeConfig) -> LatentLoRAConfig:
    if cfg.task_suite == "proof_v2" and task_name == "json":
        return LatentLoRAConfig(
            rank=int(cfg.proof_v2_b_rank),
            alpha=float(cfg.proof_v2_b_alpha),
            dropout=cfg.adapter_config.dropout,
            projection_strength=cfg.adapter_config.projection_strength,
            gate_init=float(cfg.proof_v2_b_gate_init),
            freeze_base=True,
        )
    if cfg.task_suite == "proof_v2" and task_name == "sort":
        return LatentLoRAConfig(
            rank=int(cfg.proof_v2_d_rank),
            alpha=float(cfg.proof_v2_d_alpha),
            dropout=cfg.adapter_config.dropout,
            projection_strength=cfg.adapter_config.projection_strength,
            gate_init=float(cfg.proof_v2_d_gate_init),
            freeze_base=True,
        )
    return copy.deepcopy(cfg.adapter_config)


def _proof_v2_selected_layers(tomography: TomographyResult, cfg: RuntimeConfig, *, minimum: int, task_label: str) -> List[int]:
    selected = list(int(item) for item in tomography.selected_layer_indices)
    if cfg.task_suite != "proof_v2":
        return selected
    min_layers = max(int(minimum), 1)
    if len(selected) >= min_layers:
        return selected
    extras: List[int] = []
    chosen = set(selected)
    for item in tomography.layer_saturations:
        layer_index = int(item.layer_index)
        if layer_index in chosen:
            continue
        extras.append(layer_index)
        chosen.add(layer_index)
        if len(selected) + len(extras) >= min_layers:
            break
    if extras:
        tomography.selection_reason = (
            f"{tomography.selection_reason}; proof_v2 {task_label} floor expanded selection to {len(selected) + len(extras)} layers"
        )
    return selected + extras


def _proof_v2_b_selected_layers(tomography: TomographyResult, cfg: RuntimeConfig) -> List[int]:
    return _proof_v2_selected_layers(tomography, cfg, minimum=cfg.proof_v2_b_min_layers, task_label="B")


def _proof_v2_d_selected_layers(tomography: TomographyResult, cfg: RuntimeConfig) -> List[int]:
    return _proof_v2_selected_layers(tomography, cfg, minimum=cfg.proof_v2_d_min_layers, task_label="D")


def _protected_profiles(
    old_profiles: List[TaskProfile],
    *,
    target_task_name: str,
    cfg: RuntimeConfig,
) -> List[TaskProfile]:
    if not cfg.update_task or cfg.update_task != target_task_name:
        return list(old_profiles)
    return [profile for profile in old_profiles if profile.task_name != target_task_name]


def _collect_profiles(
    model,
    task_name: str,
    batch_fn: Callable[[int], Dict[str, torch.Tensor]],
) -> TaskProfile:
    batches = [batch_fn(i) for i in range(1, 1 + 4)]
    return collect_task_profile(model, batches, task_name=task_name, stage_label=task_name)


def _serialize_saturation(result: SaturationReport) -> Dict[str, Any]:
    return {
        "step": result.step,
        "phase": result.phase,
        "model_mean_saturation": result.model_mean_saturation,
        "model_mean_occupied_overlap": result.model_mean_occupied_overlap,
        "model_mean_free_rank_fraction": result.model_mean_free_rank_fraction,
        "trigger_eligible": result.trigger_eligible,
        "expansion_trigger": result.expansion_trigger,
        "trigger_reason": result.trigger_reason,
        "consecutive_trigger_count": result.consecutive_trigger_count,
        "layer_saturations": [asdict(layer) for layer in result.layer_saturations],
    }


def _serialize_phase(result: PhaseResult) -> Dict[str, Any]:
    return {
        "label": result.label,
        "metrics": result.metrics,
        "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else None,
        "wall_time": result.wall_time,
        "step": result.step,
        "expected_steps": result.expected_steps,
        "completed": result.completed,
        "phase_version": result.phase_version,
        "task_suite": result.task_suite,
        "update_task": result.update_task,
        "gate_trajectory": list(result.gate_trajectory),
        "saturation_history": [_serialize_saturation(item) for item in result.saturation_history],
        "saturation": None if result.saturation is None else _serialize_saturation(result.saturation),
        "post_expansion_saturation": None if result.post_expansion_saturation is None else _serialize_saturation(result.post_expansion_saturation),
        "tracked": None if result.tracked is None else {
            name: _serialize_phase(value) for name, value in result.tracked.items()
        },
    }


def _deserialize_saturation(data: Dict[str, Any] | None) -> SaturationReport | None:
    if data is None:
        return None
    return SaturationReport(
        step=int(data["step"]),
        phase=str(data["phase"]),
        layer_saturations=[LayerSaturation(**item) for item in data.get("layer_saturations", [])],
        model_mean_saturation=float(data["model_mean_saturation"]),
        model_mean_occupied_overlap=float(data["model_mean_occupied_overlap"]),
        model_mean_free_rank_fraction=float(data["model_mean_free_rank_fraction"]),
        trigger_eligible=bool(data.get("trigger_eligible", data.get("expansion_trigger", False))),
        expansion_trigger=bool(data["expansion_trigger"]),
        trigger_reason=str(data.get("trigger_reason", "")),
        consecutive_trigger_count=int(data.get("consecutive_trigger_count", 0)),
    )


def _deserialize_phase(data: Dict[str, Any]) -> PhaseResult:
    tracked_payload = data.get("tracked") or {}
    tracked = {name: _deserialize_phase(value) for name, value in tracked_payload.items()} or None
    return PhaseResult(
        label=str(data["label"]),
        metrics=dict(data.get("metrics", {})),
        checkpoint_path=None if data.get("checkpoint_path") is None else Path(str(data["checkpoint_path"])),
        saturation=_deserialize_saturation(data.get("saturation")),
        wall_time=float(data.get("wall_time", 0.0)),
        step=int(data.get("step", 0)),
        expected_steps=int(data.get("expected_steps", 0)),
        completed=bool(data.get("completed", False)),
        phase_version=int(data.get("phase_version", 0)),
        task_suite=str(data.get("task_suite", "legacy")),
        update_task=None if data.get("update_task") is None else str(data.get("update_task")),
        tracked=tracked,
        saturation_history=[
            saturation
            for saturation in (
                _deserialize_saturation(item) for item in data.get("saturation_history", [])
            )
            if saturation is not None
        ],
        gate_trajectory=[(int(step), float(value)) for step, value in data.get("gate_trajectory", [])],
        post_expansion_saturation=_deserialize_saturation(data.get("post_expansion_saturation")),
    )


def _phase_checkpoint_map(phase_dir: Path, label: str) -> Dict[str, Path]:
    prefix = f"{label}_"
    checkpoints: Dict[str, Path] = {}
    if not phase_dir.exists():
        return checkpoints
    for path in phase_dir.glob(f"{label}_*.pt"):
        stem = path.stem
        if not stem.startswith(prefix):
            continue
        checkpoints[stem[len(prefix) :]] = path
    return checkpoints


def _save_phase_result(phase_dir: Path, phase: PhaseResult, cfg: RuntimeConfig | None = None) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    result_path = phase_dir / "phase_result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(_serialize_phase(phase), handle, indent=2)
    if cfg is not None:
        _sync_path_to_backup(phase_dir, cfg.output_dir, cfg.backup_dir)


def _resume_phase_from_artifacts(
    phase_dir: Path,
    label: str,
    template_model: nn.Module,
    tokenizer,
    eval_tasks: List[str],
    eval_data: Dict[str, Any],
    cfg: RuntimeConfig,
    expected_steps: int | None = None,
) -> tuple[PhaseResult | None, nn.Module | None]:
    if not cfg.resume or not phase_dir.exists():
        return None, None
    result_path = phase_dir / "phase_result.json"
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as handle:
            phase = _deserialize_phase(json.load(handle))
        phase_is_complete = (
            phase.phase_version == PHASE_VERSION
            and phase.task_suite == cfg.task_suite
            and phase.update_task == cfg.update_task
            and phase.completed
            and phase.step >= phase.expected_steps
        )
        if expected_steps is not None and int(phase.step) < int(expected_steps):
            phase_is_complete = False
        if (
            phase_is_complete
            and cfg.task_suite == "proof_v2"
            and _is_tuned_sort_phase_label(label)
            and (
                float(phase.metrics.get("sort_phase_tuned", 0.0)) < 4.0
                or abs(float(phase.metrics.get("sort_phase_teacher_old_weight", TEACHER_OLD_LOSS_WEIGHT)) - float(cfg.teacher_old_loss_weight)) > 1e-9
                or int(round(float(phase.metrics.get("sort_phase_teacher_old_period", TEACHER_OLD_BATCH_PERIOD)))) != int(cfg.teacher_old_batch_period)
                or int(round(float(phase.metrics.get("sort_phase_consol_old_period", CONSOL_OLD_BATCH_PERIOD)))) != int(cfg.consol_old_batch_period)
                or int(round(float(phase.metrics.get("sort_phase_consol_amoeba_enabled", 0.0)))) != int(cfg.consol_amoeba_enabled)
                or abs(float(phase.metrics.get("sort_phase_consol_amoeba_gentle_frac", CONSOL_AMOEBA_GENTLE_FRAC)) - float(cfg.consol_amoeba_gentle_frac)) > 1e-9
                or abs(float(phase.metrics.get("sort_phase_consol_amoeba_polish_frac", CONSOL_AMOEBA_POLISH_FRAC)) - float(cfg.consol_amoeba_polish_frac)) > 1e-9
                or abs(float(phase.metrics.get("sort_phase_consol_amoeba_gentle_old_scale", CONSOL_AMOEBA_GENTLE_OLD_SCALE)) - float(cfg.consol_amoeba_gentle_old_scale)) > 1e-9
                or abs(float(phase.metrics.get("sort_phase_consol_amoeba_gentle_new_scale", CONSOL_AMOEBA_GENTLE_NEW_SCALE)) - float(cfg.consol_amoeba_gentle_new_scale)) > 1e-9
                or abs(float(phase.metrics.get("sort_phase_consol_amoeba_polish_lr_scale", CONSOL_AMOEBA_POLISH_LR_SCALE)) - float(cfg.consol_amoeba_polish_lr_scale)) > 1e-9
            )
        ):
            phase_is_complete = False
        if not phase_is_complete:
            return None, None
        materialized = None
        if phase.checkpoint_path is not None and phase.checkpoint_path.exists():
            materialized = _materialize_phase_model(template_model, phase, cfg.device)
        return phase, materialized
    return None, None


def _save_seed_result(path: Path, result: ProofResult, cfg: RuntimeConfig | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": result.seed,
        "base_a": _serialize_phase(result.base_a),
        "teacher_b": _serialize_phase(result.teacher_b),
        "base_ab": _serialize_phase(result.base_ab),
        "teacher_c": None if result.teacher_c is None else _serialize_phase(result.teacher_c),
        "base_abc": None if result.base_abc is None else _serialize_phase(result.base_abc),
        "tomography_d": {
            "selected_layer_indices": result.tomography_d.selected_layer_indices,
            "selection_reason": result.tomography_d.selection_reason,
            "total_pressure": result.tomography_d.total_pressure,
            "layer_saturations": [asdict(item) for item in result.tomography_d.layer_saturations],
        },
        "fixed_teacher_d": _serialize_phase(result.fixed_teacher_d),
        "fixed_frontier": _serialize_phase(result.fixed_frontier),
        "fixed_final": _serialize_phase(result.fixed_final),
        "saturation_history": [_serialize_saturation(item) for item in result.saturation_history],
        "expanded_teacher_d": _serialize_phase(result.expanded_teacher_d),
        "expanded_best_d": _serialize_phase(result.expanded_best_d),
        "expanded_balanced": None if result.expanded_balanced is None else _serialize_phase(result.expanded_balanced),
        "expanded_headline": _serialize_phase(result.expanded_headline),
        "controls": {name: _serialize_phase(value) for name, value in result.controls.items()},
        "evidence": _qwen_evidence_tables(result),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    if cfg is not None:
        _sync_path_to_backup(path, cfg.output_dir, cfg.backup_dir)


def _save_summary_csv(path: Path, results: List[ProofResult], cfg: RuntimeConfig | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "stage",
                "task_suite",
                "wikitext_ppl",
                "b_exact_match",
                "b_field_acc",
                "b_train_field_acc",
                "reversal_exact",
                "sort_train_exact",
                "sort_train_token_acc",
                "sort_token_acc",
                "sort_loss",
                "sort_exact",
            ]
        )
        for result in results:
            rows = [
                ("base_A", result.base_a),
                ("teacher_B", result.teacher_b),
                ("base_AB", result.base_ab),
                ("fixed_teacher_D", result.fixed_teacher_d),
                ("fixed_mixed_frontier", result.fixed_frontier),
                ("fixed_final", result.fixed_final),
                ("expanded_teacher_D", result.expanded_teacher_d),
                ("expanded_best_D", result.expanded_best_d),
                ("expanded_headline", result.expanded_headline),
            ]
            if result.teacher_c is not None:
                rows.append(("teacher_C", result.teacher_c))
            if result.base_abc is not None:
                rows.append(("base_ABC", result.base_abc))
            if result.expanded_balanced is not None:
                rows.append(("expanded_balanced", result.expanded_balanced))
            for stage, phase in rows:
                writer.writerow(
                    [
                        result.seed,
                        stage,
                        phase.task_suite,
                        phase.metrics.get("wikitext_ppl", ""),
                        phase.metrics.get("json_exact_match", ""),
                        phase.metrics.get("json_field_acc", ""),
                        phase.metrics.get("json_train_field_acc", ""),
                        phase.metrics.get("reversal_exact", ""),
                        phase.metrics.get("sort_train_exact", ""),
                        phase.metrics.get("sort_train_token_acc", ""),
                        phase.metrics.get("sort_token_acc", ""),
                        phase.metrics.get("sort_loss", ""),
                        phase.metrics.get("sort_exact", ""),
                    ]
                )
    if cfg is not None:
        _sync_path_to_backup(path, cfg.output_dir, cfg.backup_dir)


def _metric_float(phase: PhaseResult, key: str, default: float = float("nan")) -> float:
    try:
        return float(phase.metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def _qwen_evidence_tables(result: ProofResult) -> Dict[str, Any]:
    base = result.base_ab
    fixed = result.fixed_final
    expanded = result.expanded_headline
    base_b_field = _metric_float(base, "json_field_acc")
    base_b_train = _metric_float(base, "json_train_field_acc")
    base_ppl = _metric_float(base, "wikitext_ppl")
    fixed_sort = _metric_float(fixed, "sort_token_acc", 0.0)

    rows = []
    for stage, phase in (
        ("base_AB", base),
        ("fixed_final", fixed),
        ("expanded_headline", expanded),
    ):
        ppl = _metric_float(phase, "wikitext_ppl")
        b_field = _metric_float(phase, "json_field_acc")
        b_train = _metric_float(phase, "json_train_field_acc")
        sort_tok = _metric_float(phase, "sort_token_acc")
        row = {
            "stage": stage,
            "wikitext_ppl": ppl,
            "b_field": b_field,
            "b_train": b_train,
            "sort_train_token_acc": _metric_float(phase, "sort_train_token_acc"),
            "sort_token_acc": sort_tok,
            "sort_loss": _metric_float(phase, "sort_loss"),
            "sort_exact": _metric_float(phase, "sort_exact"),
            "b_field_delta_vs_base_ab": b_field - base_b_field,
            "b_train_delta_vs_base_ab": b_train - base_b_train,
            "ppl_delta_vs_base_ab": ppl - base_ppl,
            "ppl_ratio_vs_base_ab": ppl / base_ppl if base_ppl and not math.isnan(base_ppl) else float("nan"),
            "sort_token_delta_vs_fixed": sort_tok - fixed_sort if stage != "base_AB" else float("nan"),
        }
        rows.append(row)

    expanded_row = next(row for row in rows if row["stage"] == "expanded_headline")
    fixed_row = next(row for row in rows if row["stage"] == "fixed_final")
    acceptance = {
        "expanded_beats_fixed_sort_token_acc": expanded_row["sort_token_acc"] > fixed_row["sort_token_acc"],
        "expanded_beats_fixed_sort_loss": expanded_row["sort_loss"] < fixed_row["sort_loss"],
        "expanded_preserves_b_field_vs_base_ab": expanded_row["b_field"] >= base_b_field - 0.05,
        "expanded_preserves_wikitext_vs_base_ab": expanded_row["ppl_ratio_vs_base_ab"] <= 1.10,
        "expanded_dense_pareto_win": (
            expanded_row["sort_token_acc"] > fixed_row["sort_token_acc"]
            and expanded_row["wikitext_ppl"] <= fixed_row["wikitext_ppl"]
        ),
    }
    return {
        "qwen_pareto_table": rows,
        "acceptance": acceptance,
        "forgetting_definition": (
            "For old accuracy metrics, forgetting is score_after - score_base_AB. "
            "For WikiText, forgetting is ppl_after / ppl_base_AB. Catastrophic forgetting "
            "would mean old-skill collapse toward base_A or material WikiText degradation."
        ),
    }


def _summary_table(results: List[ProofResult]) -> str:
    lines = []
    lines.append("==============================================================================")
    lines.append("QWEN 0.5B CONTINUAL LEARNING PROOF")
    lines.append("==============================================================================")
    lines.append(
        f"{'stage':<24} {'wikitext_ppl':>12} {'b_field':>12} {'b_train':>12} {'rev_exact':>12} {'sort_train_tok':>14} {'sort_tok':>12} {'sort_loss':>12} {'sort_exact':>12}"
    )
    exemplar = results[0]
    rows = [
        ("base_A", exemplar.base_a),
        ("teacher_B", exemplar.teacher_b),
        ("base_AB", exemplar.base_ab),
    ]
    if exemplar.teacher_c is not None:
        rows.append(("teacher_C", exemplar.teacher_c))
    if exemplar.base_abc is not None:
        rows.append(("base_ABC", exemplar.base_abc))
    rows.extend(
        [
            ("fixed_teacher_D", exemplar.fixed_teacher_d),
            ("fixed_mixed_frontier", exemplar.fixed_frontier),
            ("fixed_final", exemplar.fixed_final),
            ("expanded_teacher_D", exemplar.expanded_teacher_d),
            ("expanded_best_D", exemplar.expanded_best_d),
        ]
    )
    if exemplar.expanded_balanced is not None:
        rows.append(("expanded_balanced", exemplar.expanded_balanced))
    rows.append(("expanded_headline", exemplar.expanded_headline))
    for stage, phase in rows:
        lines.append(
            f"{stage:<24} "
            f"{phase.metrics.get('wikitext_ppl', float('nan')):>12.3f} "
            f"{phase.metrics.get('json_field_acc', float('nan')):>12.3f} "
            f"{phase.metrics.get('json_train_field_acc', float('nan')):>12.3f} "
            f"{phase.metrics.get('reversal_exact', float('nan')):>12.3f} "
            f"{phase.metrics.get('sort_train_token_acc', float('nan')):>14.3f} "
            f"{phase.metrics.get('sort_token_acc', float('nan')):>12.3f} "
            f"{phase.metrics.get('sort_loss', float('nan')):>12.3f} "
            f"{phase.metrics.get('sort_exact', float('nan')):>12.3f}"
        )
    if all(math.isnan(float(phase.metrics.get("reversal_exact", float("nan")))) for _, phase in rows):
        lines.append("note: rev_exact=nan means task C/reversal was not run in this phase scope.")
    max_sort_exact = max(float(phase.metrics.get("sort_exact", float("nan"))) for _, phase in rows)
    max_sort_tok = max(float(phase.metrics.get("sort_token_acc", 0.0)) for _, phase in rows)
    if (math.isnan(max_sort_exact) or max_sort_exact <= 0.0) and max_sort_tok > 0.0:
        lines.append(
            "note: sort_exact is a strict full-sequence held-out exact metric; inspect "
            "sort_train_token_acc, sort_token_acc, and sort_loss for dense D progress."
        )
    evidence = _qwen_evidence_tables(exemplar)
    lines.append("------------------------------------------------------------------------------")
    lines.append("QWEN FORGETTING / EXPANSION DELTAS")
    lines.append(
        f"{'stage':<20} {'b_field_delta':>14} {'b_train_delta':>14} {'ppl_ratio':>10} {'sort_tok':>10} {'sort_vs_fixed':>14}"
    )
    for row in evidence["qwen_pareto_table"]:
        lines.append(
            f"{row['stage']:<20} "
            f"{row['b_field_delta_vs_base_ab']:>14.3f} "
            f"{row['b_train_delta_vs_base_ab']:>14.3f} "
            f"{row['ppl_ratio_vs_base_ab']:>10.3f} "
            f"{row['sort_token_acc']:>10.3f} "
            f"{row['sort_token_delta_vs_fixed']:>14.3f}"
        )
    acceptance = evidence["acceptance"]
    lines.append(
        "acceptance: "
        f"expanded_dense_pareto_win={acceptance['expanded_dense_pareto_win']} "
        f"expanded_preserves_b={acceptance['expanded_preserves_b_field_vs_base_ab']} "
        f"expanded_preserves_wikitext={acceptance['expanded_preserves_wikitext_vs_base_ab']}"
    )
    lines.append("==============================================================================")
    return "\n".join(lines)


def _load_eval_data(tokenizer, cfg: RuntimeConfig) -> Dict[str, Any]:
    wikitext_val = load_wikitext_texts(
        tokenizer,
        split="validation",
        max_samples=cfg.wikitext_eval_samples,
        max_seq_len=cfg.max_seq_len,
        local_files_only=cfg.local_files_only,
    )
    return {"wikitext_val": wikitext_val}


def _annotate_b_lift(metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> None:
    zero_exact = float(baseline_metrics.get("json_exact_match", 0.0))
    zero_field = float(baseline_metrics.get("json_field_acc", 0.0))
    zero_train = float(baseline_metrics.get("json_train_field_acc", 0.0))
    metrics["b_zero_shot_exact"] = zero_exact
    metrics["b_zero_shot_field"] = zero_field
    metrics["b_zero_shot_train"] = zero_train
    metrics["b_heldout_lift"] = float(metrics.get("json_field_acc", 0.0)) - zero_field
    metrics["b_train_lift"] = float(metrics.get("json_train_field_acc", 0.0)) - zero_train
    metrics["b_exact_lift"] = float(metrics.get("json_exact_match", 0.0)) - zero_exact


def _ab_success_gate(
    baseline_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
    cfg: RuntimeConfig,
) -> tuple[bool, str]:
    zero_exact = float(baseline_metrics.get("json_exact_match", 0.0))
    zero_field = float(baseline_metrics.get("json_field_acc", 0.0))
    zero_train = float(baseline_metrics.get("json_train_field_acc", 0.0))
    train_field = float(current_metrics.get("json_train_field_acc", 0.0))
    train_valid = float(current_metrics.get("json_train_valid", current_metrics.get("json_valid", 0.0)))
    heldout_field = float(current_metrics.get("json_field_acc", 0.0))
    train_lift = train_field - zero_train
    heldout_lift = heldout_field - zero_field
    if cfg.task_suite == "proof_v2":
        contamination_ok = zero_exact <= 0.10 and zero_field <= 0.15
        proficiency_ok = train_field >= 0.70 and train_valid >= 0.80 and heldout_field >= 0.45
        lift_ok = train_lift >= 0.35 and heldout_lift >= 0.25
        ok = contamination_ok and proficiency_ok and lift_ok
        reason = (
            "proof_v2 B gate requires low zero-shot contamination and real post-training lift "
            f"(need zero_exact<=0.10, zero_field<=0.15, train_field>=0.70, train_valid>=0.80, "
            f"heldout_field>=0.45, train_lift>=0.35, heldout_lift>=0.25; "
            f"got zero_exact={zero_exact:.3f}, zero_field={zero_field:.3f}, "
            f"train_field={train_field:.3f}, train_valid={train_valid:.3f}, heldout_field={heldout_field:.3f}, "
            f"train_lift={train_lift:.3f}, heldout_lift={heldout_lift:.3f})"
        )
        return ok, reason
    ok = train_field >= 0.60 and train_valid >= 0.60
    reason = (
        f"legacy B gate requires train_field>=0.60 and train_valid>=0.60 "
        f"(got train_field={train_field:.3f}, train_valid={train_valid:.3f})"
    )
    return ok, reason


def _resolve_runtime(args) -> RuntimeConfig:
    smoke = bool(args.smoke or os.environ.get("QWEN_SMOKE"))
    model_id = args.model_id or default_model_id(args.local_files_only or smoke)
    cfg = RuntimeConfig(
        model_id=model_id,
        device=args.device,
        dtype=choose_dtype(args.dtype),
        local_files_only=bool(args.local_files_only or smoke),
        resume=bool(args.resume),
        smoke=smoke,
        output_dir=Path(args.output_dir),
        backup_dir=None if args.backup_dir is None else Path(args.backup_dir),
        seed=args.seed,
        phase_scope=args.phase_scope,
        task_suite=args.task_suite,
        update_task=args.update_task,
        consolidation_steps=args.consolidation_steps,
        consolidation_lr=args.consolidation_lr,
        consol_kl_weight=args.consol_kl_weight,
        consol_old_kl_weight=args.consol_old_kl_weight,
        consol_hidden_weight=args.consol_hidden_weight,
        consol_old_batch_period=args.consol_old_batch_period,
        consol_amoeba_enabled=bool(args.consol_amoeba),
        consol_amoeba_gentle_frac=args.consol_amoeba_gentle_frac,
        consol_amoeba_polish_frac=args.consol_amoeba_polish_frac,
        consol_amoeba_gentle_old_scale=args.consol_amoeba_gentle_old_scale,
        consol_amoeba_gentle_new_scale=args.consol_amoeba_gentle_new_scale,
        consol_amoeba_polish_lr_scale=args.consol_amoeba_polish_lr_scale,
        teacher_old_loss_weight=args.teacher_old_loss_weight,
        teacher_old_batch_period=args.teacher_old_batch_period,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        consolidation_micro_batch_size=args.consolidation_micro_batch_size,
        gradient_checkpointing=not bool(args.no_gradient_checkpointing),
        run_controls=not bool(args.skip_controls),
        eval_interval=args.eval_interval,
        require_real_frontier=not bool(args.allow_frontier_fallback),
        allow_ab_gate_bypass=bool(args.allow_ab_gate_bypass),
        wikitext_eval_samples=args.wikitext_eval_samples,
        json_eval_samples=args.json_eval_samples,
        reversal_eval_samples=args.reversal_eval_samples,
        sort_eval_samples=args.sort_eval_samples,
        proof_v2_b_attach_steps=args.proof_v2_b_attach_steps,
        proof_v2_b_attach_lr=args.proof_v2_b_attach_lr,
        proof_v2_b_rank=args.proof_v2_b_rank,
        proof_v2_b_alpha=args.proof_v2_b_alpha,
        proof_v2_b_gate_init=args.proof_v2_b_gate_init,
        proof_v2_b_use_up_proj=bool(args.proof_v2_b_use_up_proj),
        proof_v2_b_min_layers=args.proof_v2_b_min_layers,
        proof_v2_d_attach_steps=args.proof_v2_d_attach_steps,
        proof_v2_d_attach_lr=args.proof_v2_d_attach_lr,
        proof_v2_d_rank=args.proof_v2_d_rank,
        proof_v2_d_alpha=args.proof_v2_d_alpha,
        proof_v2_d_gate_init=args.proof_v2_d_gate_init,
        proof_v2_d_use_up_proj=bool(args.proof_v2_d_use_up_proj),
        proof_v2_d_min_layers=args.proof_v2_d_min_layers,
    )
    if smoke:
        cfg.attach_steps = 20
        cfg.consolidation_steps = 10
        cfg.polish_steps = 5
        cfg.batch_size = 2
        cfg.eval_batch_size = 2
        cfg.consolidation_micro_batch_size = min(cfg.batch_size, 2)
        cfg.max_seq_len = 64
        cfg.wikitext_eval_samples = 8
        cfg.json_eval_samples = 4
        cfg.reversal_eval_samples = 4
        cfg.sort_eval_samples = 4
        cfg.eval_interval = 5
        cfg.log_interval = 5
        cfg.proof_v2_b_attach_steps = min(cfg.proof_v2_b_attach_steps, 20)
        cfg.proof_v2_d_attach_steps = min(cfg.proof_v2_d_attach_steps, 20)
    return cfg


def _run_control_fixed_size(
    control_name: str,
    base_old_model,
    teacher_new_model,
    tokenizer,
    old_task_batch_fn,
    new_task_batch_fn,
    eval_tasks,
    eval_data,
    selected_layers,
    cfg: RuntimeConfig,
    output_dir: Path,
    fixed_frontier: PhaseResult,
) -> PhaseResult:
    student = _clone_model(base_old_model, cfg.device)
    tracker = {"primary": _best_d_cmp(fixed_frontier, "sort_token_acc")}
    label = f"control_{control_name}"
    phase_dir = _phase_output_dir(output_dir, label)
    resumed, _ = _resume_phase_from_artifacts(
        phase_dir,
        label,
        student,
        tokenizer,
        eval_tasks,
        eval_data,
        cfg,
        expected_steps=cfg.consolidation_steps,
    )
    if resumed is not None:
        _release_cuda_memory(student)
        return resumed
    result = dual_teacher_consolidation(
        student,
        base_old_model,
        teacher_new_model,
        tokenizer,
        new_task_batch_fn,
        old_task_batch_fn,
        eval_tasks,
        eval_data,
        selected_layers,
        cfg.consolidation_steps,
        cfg.consolidation_lr,
        label,
        phase_dir,
        cfg,
        tracker,
    )
    _mark_sort_phase(result, phase_dir, cfg)
    _release_cuda_memory(student)
    return result


def run_proof_seed(cfg: RuntimeConfig) -> ProofResult:
    _set_seed(cfg.seed)
    seed_dir = _phase_output_dir(cfg.output_dir, f"seed_{cfg.seed}")
    tokenizer = load_tokenizer(
        cfg.model_id,
        trust_remote_code=True,
        local_files_only=cfg.local_files_only,
    )
    model = load_causal_lm(
        cfg.model_id,
        device=cfg.device,
        dtype=cfg.dtype,
        trust_remote_code=True,
        local_files_only=cfg.local_files_only,
    )
    eval_data = _load_eval_data(tokenizer, cfg)
    frozen_base_model = _clone_model(model, cfg.device)
    _freeze_model(frozen_base_model)

    base_a_metrics = evaluate_world(model, tokenizer, ["retention", "json"], eval_data, cfg.device, cfg)
    eval_data["baseline_retention"] = dict(base_a_metrics)
    base_a_metrics["b_zero_shot_guard_pass"] = 1.0 if (
        float(base_a_metrics.get("json_exact_match", 0.0)) <= 0.10
        and float(base_a_metrics.get("json_field_acc", 0.0)) <= 0.15
    ) else 0.0
    _write_b_debug_samples(model, tokenizer, cfg.device, _phase_output_dir(seed_dir, "base_A"), cfg)
    base_a = PhaseResult(
        "base_A",
        base_a_metrics,
        None,
        None,
        0.0,
        step=0,
        expected_steps=0,
        completed=True,
        phase_version=PHASE_VERSION,
        task_suite=cfg.task_suite,
        update_task=cfg.update_task,
    )

    wikitext_train = load_wikitext_texts(
        tokenizer,
        split="train",
        max_samples=max(cfg.wikitext_eval_samples, 64),
        max_seq_len=cfg.max_seq_len,
        local_files_only=cfg.local_files_only,
    )
    old_task_batch_fn = make_wikitext_batch_fn(tokenizer, wikitext_train, cfg.device, cfg, cfg.seed + 1)

    # Task B
    b_batch_fn = _task_batch_factory("json", tokenizer, cfg.device, cfg, cfg.seed + 2)
    a_profile = _collect_profiles(model, "retention", old_task_batch_fn)
    tomography_b = run_tomography(
        model,
        tokenizer,
        [b_batch_fn(i) for i in range(1, 1 + 4)],
        _protected_profiles([a_profile], target_task_name="json", cfg=cfg),
    )
    b_selected_layers = _proof_v2_b_selected_layers(tomography_b, cfg)
    tomography_b.selected_layer_indices = list(b_selected_layers)
    _freeze_model(model)
    attached_b = attach_latent_lora(
        model,
        suffixes=_task_target_suffixes("json", cfg),
        layer_indices=set(b_selected_layers),
        config=_task_adapter_config("json", cfg),
    )
    teacher_b_dir = _phase_output_dir(seed_dir, "teacher_B")
    teacher_b, teacher_b_model = _resume_phase_from_artifacts(
        teacher_b_dir,
        "teacher_B",
        model,
        tokenizer,
        ["retention", "json"],
        eval_data,
        cfg,
        expected_steps=_task_attach_steps("json", cfg),
    )
    if teacher_b is None:
        teacher_b = train_adapter_teacher(
            model,
            tokenizer,
            attached_b,
            b_batch_fn,
            old_task_batch_fn,
            ESCAPE_SCHEDULE,
            ["retention", "json"],
            eval_data,
            _task_attach_steps("json", cfg),
            _task_attach_lr("json", cfg),
            "teacher_B",
            teacher_b_dir,
            cfg,
            {"primary": _b_metric_cmp},
        )
        teacher_b_model = _materialize_phase_model(model, teacher_b, cfg.device)
    elif teacher_b_model is None:
        teacher_b_model = _materialize_phase_model(model, teacher_b, cfg.device)
    _annotate_b_lift(teacher_b.metrics, base_a.metrics)
    teacher_b.metrics["b_zero_shot_guard_pass"] = base_a.metrics["b_zero_shot_guard_pass"]
    _save_phase_result(teacher_b_dir, teacher_b, cfg)
    _write_b_debug_samples(teacher_b_model, tokenizer, cfg.device, teacher_b_dir, cfg)

    if cfg.phase_scope == "b_teacher":
        empty_tomography = TomographyResult(layer_saturations=[], selected_layer_indices=[], selection_reason="B-only scope", total_pressure=0.0)
        result = ProofResult(
            seed=cfg.seed,
            base_a=base_a,
            teacher_b=teacher_b,
            base_ab=_placeholder_phase("base_AB"),
            teacher_c=None,
            base_abc=None,
            tomography_d=empty_tomography,
            fixed_teacher_d=_placeholder_phase("fixed_teacher_D"),
            fixed_frontier=_placeholder_phase("fixed_mixed_frontier"),
            fixed_final=_placeholder_phase("fixed_final"),
            saturation_history=[],
            expanded_teacher_d=_placeholder_phase("expanded_teacher_D"),
            expanded_best_d=_placeholder_phase("expanded_best_D"),
            expanded_balanced=None,
            expanded_headline=_placeholder_phase("expanded_headline"),
            controls={},
        )
        _save_seed_result(seed_dir / "seed_result.json", result, cfg)
        _release_cuda_memory(model, teacher_b_model, frozen_base_model)
        return result

    # Consolidate AB
    base_ab_student = _clone_model(frozen_base_model, cfg.device)
    base_ab_dir = _phase_output_dir(seed_dir, "base_AB")
    base_ab, base_ab_model = _resume_phase_from_artifacts(
        base_ab_dir,
        "base_AB",
        base_ab_student,
        tokenizer,
        ["retention", "json"],
        eval_data,
        cfg,
        expected_steps=cfg.consolidation_steps,
    )
    if base_ab is None:
        base_ab = dual_teacher_consolidation(
            base_ab_student,
            frozen_base_model,
            teacher_b_model,
            tokenizer,
            b_batch_fn,
            old_task_batch_fn,
            ["retention", "json"],
            eval_data,
            b_selected_layers,
            cfg.consolidation_steps,
            cfg.consolidation_lr,
            "base_AB",
            base_ab_dir,
            cfg,
            {"primary": _b_metric_cmp},
        )
        base_ab_model = _clone_model(base_ab_student, cfg.device)
        _load_checkpoint(base_ab_model, base_ab.checkpoint_path, cfg.device)
    elif base_ab_model is None:
        base_ab_model = _materialize_phase_model(base_ab_student, base_ab, cfg.device)
    _annotate_b_lift(base_ab.metrics, base_a.metrics)
    base_ab.metrics["b_zero_shot_guard_pass"] = base_a.metrics["b_zero_shot_guard_pass"]
    _save_phase_result(base_ab_dir, base_ab, cfg)
    _write_b_debug_samples(base_ab_model, tokenizer, cfg.device, base_ab_dir, cfg)

    ab_ok, ab_reason = _ab_success_gate(base_a.metrics, base_ab.metrics, cfg)
    base_ab.metrics["ab_success"] = 1.0 if ab_ok else 0.0
    base_ab.metrics["ab_success_threshold_met"] = 1.0 if ab_ok else 0.0

    _release_cuda_memory(model, teacher_b_model, frozen_base_model, base_ab_student)
    model = None
    teacher_b_model = None
    frozen_base_model = None
    base_ab_student = None
    if cfg.phase_scope == "ab":
        empty_tomography = TomographyResult(layer_saturations=[], selected_layer_indices=[], selection_reason="AB-only scope", total_pressure=0.0)
        result = ProofResult(
            seed=cfg.seed,
            base_a=base_a,
            teacher_b=teacher_b,
            base_ab=base_ab,
            teacher_c=None,
            base_abc=None,
            tomography_d=empty_tomography,
            fixed_teacher_d=_placeholder_phase("fixed_teacher_D"),
            fixed_frontier=_placeholder_phase("fixed_mixed_frontier"),
            fixed_final=_placeholder_phase("fixed_final"),
            saturation_history=[],
            expanded_teacher_d=_placeholder_phase("expanded_teacher_D"),
            expanded_best_d=_placeholder_phase("expanded_best_D"),
            expanded_balanced=None,
            expanded_headline=_placeholder_phase("expanded_headline"),
            controls={},
        )
        _save_seed_result(seed_dir / "seed_result.json", result, cfg)
        return result

    if not cfg.smoke and not ab_ok and not cfg.allow_ab_gate_bypass:
        raise RuntimeError(
            "Refusing to continue to D because A->B did not clear the B-task success gate. "
            + ab_reason
        )

    teacher_c = None
    base_abc = None
    old_profiles = [a_profile, _collect_profiles(base_ab_model, "json", b_batch_fn)]
    active_old_model = base_ab_model
    active_old_batch_fn = _compose_task_batch_fns([old_task_batch_fn, b_batch_fn])
    active_eval_tasks = ["retention", "json"]

    if cfg.phase_scope == "full_abcd":
        c_batch_fn = _task_batch_factory("reversal", tokenizer, cfg.device, cfg, cfg.seed + 3)
        tomography_c = run_tomography(
            active_old_model,
            tokenizer,
            [c_batch_fn(i) for i in range(1, 1 + 4)],
            _protected_profiles(old_profiles, target_task_name="reversal", cfg=cfg),
        )
        _freeze_model(active_old_model)
        attached_c = attach_latent_lora(
            active_old_model,
            suffixes=TARGET_SUFFIX,
            layer_indices=set(tomography_c.selected_layer_indices),
            config=cfg.adapter_config,
        )
        teacher_c_dir = _phase_output_dir(seed_dir, "teacher_C")
        teacher_c, teacher_c_model = _resume_phase_from_artifacts(
            teacher_c_dir,
            "teacher_C",
            active_old_model,
            tokenizer,
            ["retention", "json", "reversal"],
            eval_data,
            cfg,
            expected_steps=cfg.attach_steps,
        )
        if teacher_c is None:
            teacher_c = train_adapter_teacher(
                active_old_model,
                tokenizer,
                attached_c,
                c_batch_fn,
                active_old_batch_fn,
                ESCAPE_SCHEDULE,
                ["retention", "json", "reversal"],
                eval_data,
                cfg.attach_steps,
                cfg.attach_lr,
                "teacher_C",
                teacher_c_dir,
                cfg,
                {"primary": _higher_metric("reversal_exact")},
            )
            teacher_c_model = _materialize_phase_model(active_old_model, teacher_c, cfg.device)
        elif teacher_c_model is None:
            teacher_c_model = _materialize_phase_model(active_old_model, teacher_c, cfg.device)
        base_abc_student = _clone_model(base_ab_model, cfg.device)
        base_abc_dir = _phase_output_dir(seed_dir, "base_ABC")
        base_abc, next_active_old_model = _resume_phase_from_artifacts(
            base_abc_dir,
            "base_ABC",
            base_abc_student,
            tokenizer,
            ["retention", "json", "reversal"],
            eval_data,
            cfg,
            expected_steps=cfg.consolidation_steps,
        )
        if base_abc is None:
            base_abc = dual_teacher_consolidation(
                base_abc_student,
                base_ab_model,
                teacher_c_model,
                tokenizer,
                c_batch_fn,
                active_old_batch_fn,
                ["retention", "json", "reversal"],
                eval_data,
                tomography_c.selected_layer_indices,
                cfg.consolidation_steps,
                cfg.consolidation_lr,
                "base_ABC",
                base_abc_dir,
                cfg,
                {"primary": _higher_metric("reversal_exact")},
            )
            next_active_old_model = _clone_model(base_abc_student, cfg.device)
            _load_checkpoint(next_active_old_model, base_abc.checkpoint_path, cfg.device)
        elif next_active_old_model is None:
            next_active_old_model = _materialize_phase_model(base_abc_student, base_abc, cfg.device)
        _release_cuda_memory(active_old_model, teacher_c_model, base_abc_student)
        active_old_model = next_active_old_model
        active_old_batch_fn = _compose_task_batch_fns([old_task_batch_fn, b_batch_fn, c_batch_fn])
        active_eval_tasks = ["retention", "json", "reversal"]
        old_profiles.append(_collect_profiles(active_old_model, "reversal", c_batch_fn))

    # Task D fixed-size
    d_batch_fn = _task_batch_factory("sort", tokenizer, cfg.device, cfg, cfg.seed + 4)
    tomography_d = run_tomography(
        active_old_model,
        tokenizer,
        [d_batch_fn(i) for i in range(1, 1 + 4)],
        _protected_profiles(old_profiles, target_task_name="sort", cfg=cfg),
    )
    d_selected_layers = _proof_v2_d_selected_layers(tomography_d, cfg)
    tomography_d.selected_layer_indices = list(d_selected_layers)
    tomography_path = seed_dir / "layer_tomography.csv"
    write_tomography_csv(tomography_path, [tomography_d])
    _sync_path_to_backup(tomography_path, cfg.output_dir, cfg.backup_dir)

    fixed_teacher_model = _clone_model(active_old_model, cfg.device)
    _freeze_model(fixed_teacher_model)
    attached_d = attach_latent_lora(
        fixed_teacher_model,
        suffixes=_task_target_suffixes("sort", cfg),
        layer_indices=set(tomography_d.selected_layer_indices),
        config=_task_adapter_config("sort", cfg),
    )
    fixed_teacher_dir = _phase_output_dir(seed_dir, "fixed_teacher_D")
    fixed_teacher_d, fixed_teacher_best_model = _resume_phase_from_artifacts(
        fixed_teacher_dir,
        "fixed_teacher_D",
        fixed_teacher_model,
        tokenizer,
        active_eval_tasks + ["sort"],
        eval_data,
        cfg,
        expected_steps=_task_attach_steps("sort", cfg),
    )
    if fixed_teacher_d is None:
        fixed_teacher_d = train_adapter_teacher(
            fixed_teacher_model,
            tokenizer,
            attached_d,
            d_batch_fn,
            active_old_batch_fn,
            ESCAPE_SCHEDULE,
            active_eval_tasks + ["sort"],
            eval_data,
            _task_attach_steps("sort", cfg),
            _task_attach_lr("sort", cfg),
            "fixed_teacher_D",
            fixed_teacher_dir,
            cfg,
            {"primary": _d_metric_cmp},
            saturation_old_profiles=_protected_profiles(old_profiles, target_task_name="sort", cfg=cfg),
            saturation_layer_indices=tomography_d.selected_layer_indices,
            saturation_probe_batch_fn=d_batch_fn,
        )
        fixed_teacher_best_model = _materialize_phase_model(fixed_teacher_model, fixed_teacher_d, cfg.device)
    elif fixed_teacher_best_model is None:
        fixed_teacher_best_model = _materialize_phase_model(fixed_teacher_model, fixed_teacher_d, cfg.device)
    _mark_sort_phase(fixed_teacher_d, fixed_teacher_dir, cfg)
    _release_cuda_memory(fixed_teacher_model)
    fixed_teacher_model = None

    if cfg.phase_scope == "d_teacher":
        result = ProofResult(
            seed=cfg.seed,
            base_a=base_a,
            teacher_b=teacher_b,
            base_ab=base_ab,
            teacher_c=teacher_c,
            base_abc=base_abc,
            tomography_d=tomography_d,
            fixed_teacher_d=fixed_teacher_d,
            fixed_frontier=_placeholder_phase("fixed_mixed_frontier"),
            fixed_final=_placeholder_phase("fixed_final"),
            saturation_history=list(fixed_teacher_d.saturation_history),
            expanded_teacher_d=_placeholder_phase("expanded_teacher_D"),
            expanded_best_d=_placeholder_phase("expanded_best_D"),
            expanded_balanced=None,
            expanded_headline=_placeholder_phase("expanded_headline"),
            controls={},
        )
        _save_seed_result(seed_dir / "seed_result.json", result, cfg)
        _release_cuda_memory(fixed_teacher_best_model, active_old_model, frozen_base_model, base_ab_model)
        return result

    fixed_frontier_tracker = FrontierTracker(
        "sort_exact",
        "wikitext_ppl",
        evaluate_world(active_old_model, tokenizer, ["retention"], eval_data, cfg.device, cfg)["wikitext_ppl"],
        dense_new_key="sort_token_acc",
        min_dense_score=0.20,
    )
    fixed_student = _clone_model(active_old_model, cfg.device)
    fixed_abcd_dir = _phase_output_dir(seed_dir, "fixed_ABCD")
    fixed_run, fixed_student_model = _resume_phase_from_artifacts(
        fixed_abcd_dir,
        "fixed_ABCD",
        fixed_student,
        tokenizer,
        active_eval_tasks + ["sort"],
        eval_data,
        cfg,
        expected_steps=cfg.consolidation_steps,
    )
    if fixed_run is None:
        fixed_run = dual_teacher_consolidation(
            fixed_student,
            active_old_model,
            fixed_teacher_best_model,
            tokenizer,
            d_batch_fn,
            active_old_batch_fn,
            active_eval_tasks + ["sort"],
            eval_data,
            tomography_d.selected_layer_indices,
            cfg.consolidation_steps,
            cfg.consolidation_lr,
            "fixed_ABCD",
            fixed_abcd_dir,
            cfg,
            {
                "primary": _d_metric_cmp,
                "frontier": fixed_frontier_tracker.better,
            },
        )
        fixed_student_model = None
    if fixed_run.tracked and "frontier" in fixed_run.tracked:
        fixed_frontier = fixed_run.tracked["frontier"]
    else:
        fixed_frontier = _resolve_tracked_phase(
            fixed_run,
            "frontier",
            fallback_label="fixed_mixed_frontier_fallback",
            fallback_metric="frontier_fallback",
        )
        if cfg.require_real_frontier and cfg.phase_scope == "abd_fixed":
            raise RuntimeError(
                "fixed-size D did not produce a real tracked frontier; refusing to continue with fallback. "
                "Rerun with --allow-frontier-fallback only for debugging."
            )
        fixed_run.metrics["frontier_fallback_used"] = 1.0
        fixed_frontier.metrics["frontier_fallback_used"] = 1.0
    fixed_final = fixed_run
    _mark_sort_phase(fixed_run, fixed_abcd_dir, cfg)
    _release_cuda_memory(fixed_teacher_best_model, fixed_student if fixed_student_model is None else fixed_student_model)
    fixed_teacher_best_model = None
    fixed_student = None

    saturation_history = list(fixed_teacher_d.saturation_history)
    expand_now, _reason = should_expand(saturation_history)

    if cfg.phase_scope == "abd_fixed":
        result = ProofResult(
            seed=cfg.seed,
            base_a=base_a,
            teacher_b=teacher_b,
            base_ab=base_ab,
            teacher_c=teacher_c,
            base_abc=base_abc,
            tomography_d=tomography_d,
            fixed_teacher_d=fixed_teacher_d,
            fixed_frontier=fixed_frontier,
            fixed_final=fixed_final,
            saturation_history=saturation_history,
            expanded_teacher_d=_placeholder_phase("expanded_teacher_D"),
            expanded_best_d=_placeholder_phase("expanded_best_D"),
            expanded_balanced=None,
            expanded_headline=_placeholder_phase("expanded_headline"),
            controls={},
        )
        _save_seed_result(seed_dir / "seed_result.json", result, cfg)
        return result

    # Expanded path
    expanded_teacher_model = _clone_model(active_old_model, cfg.device)
    gated_layer_ref = None
    if expand_now or cfg.phase_scope in {"abd_rescue", "full_abcd", "controls"}:
        insert_after = tomography_d.selected_layer_indices[0]
        expanded_teacher_model, gated_layer_ref = insert_expansion_layer(expanded_teacher_model, insert_after)
        _freeze_model(expanded_teacher_model)
        for idx, layer in enumerate(expanded_teacher_model.model.layers):
            trainable = isinstance(layer, GatedQwenLayer)
            for param in layer.parameters():
                param.requires_grad = trainable
        attached_expanded_d = attach_latent_lora(
            expanded_teacher_model,
            suffixes=_task_target_suffixes("sort", cfg),
            layer_indices=set(tomography_d.selected_layer_indices),
            config=_task_adapter_config("sort", cfg),
        )
        best_d_cmp = _best_d_cmp(fixed_frontier, "sort_token_acc")
        balanced_cmp = _balanced_cmp(fixed_frontier, "sort_token_acc")
        expanded_teacher_dir = _phase_output_dir(seed_dir, "expanded_teacher_D")
        expanded_teacher_d, expanded_teacher_primary_model = _resume_phase_from_artifacts(
            expanded_teacher_dir,
            "expanded_teacher_D",
            expanded_teacher_model,
            tokenizer,
            active_eval_tasks + ["sort"],
            eval_data,
            cfg,
            expected_steps=_task_attach_steps("sort", cfg),
        )
        if expanded_teacher_d is None:
            expanded_teacher_d = train_adapter_teacher(
                expanded_teacher_model,
                tokenizer,
                attached_expanded_d,
                d_batch_fn,
                active_old_batch_fn,
                ESCAPE_SCHEDULE,
                active_eval_tasks + ["sort"],
                eval_data,
                _task_attach_steps("sort", cfg),
                _task_attach_lr("sort", cfg),
                "expanded_teacher_D",
                expanded_teacher_dir,
                cfg,
                {
                    "primary": _d_metric_cmp,
                    "best_d": best_d_cmp,
                    "balanced": balanced_cmp,
                },
                saturation_old_profiles=_protected_profiles(old_profiles, target_task_name="sort", cfg=cfg),
                saturation_layer_indices=tomography_d.selected_layer_indices,
                saturation_probe_batch_fn=d_batch_fn,
            )
            expanded_teacher_primary_model = _materialize_phase_model(expanded_teacher_model, expanded_teacher_d, cfg.device)
        elif expanded_teacher_primary_model is None:
            expanded_teacher_primary_model = _materialize_phase_model(expanded_teacher_model, expanded_teacher_d, cfg.device)
        expanded_probe_batches = [d_batch_fn(i) for i in range(1, 1 + min(4, cfg.batch_size))]
        expanded_teacher_d.post_expansion_saturation = compute_saturation_report(
            expanded_teacher_primary_model,
            tokenizer,
            expanded_probe_batches,
            _protected_profiles(old_profiles, target_task_name="sort", cfg=cfg),
            list(range(len(expanded_teacher_primary_model.model.layers))),
            _task_attach_steps("sort", cfg),
            "expanded_teacher_post_expansion",
            [],
            retention_delta=(
                expanded_teacher_d.metrics.get("wikitext_ppl", 0.0)
                / max(eval_data["baseline_retention"]["wikitext_ppl"], 1e-6)
                - 1.0
            ),
            task_progress_delta=0.0,
        )
        expanded_teacher_best = expanded_teacher_d.tracked.get("best_d", expanded_teacher_d) if expanded_teacher_d.tracked else expanded_teacher_d
        expanded_teacher_balanced = expanded_teacher_d.tracked.get("balanced") if expanded_teacher_d.tracked else None
        _mark_sort_phase(expanded_teacher_d, expanded_teacher_dir, cfg)
    else:
        expanded_teacher_d = fixed_teacher_d
        expanded_teacher_best = fixed_teacher_d
        expanded_teacher_balanced = None

    # best-D branch
    expanded_best_student = _clone_model(active_old_model, cfg.device)
    if gated_layer_ref is not None:
        expanded_best_student, _best_gate = insert_expansion_layer(expanded_best_student, tomography_d.selected_layer_indices[0])
    teacher_best_model = _materialize_phase_model(expanded_teacher_model, expanded_teacher_best, cfg.device)
    expanded_best_dir = _phase_output_dir(seed_dir, "expanded_best_D")
    expanded_best_d, expanded_best_model = _resume_phase_from_artifacts(
        expanded_best_dir,
        "expanded_best_D",
        expanded_best_student,
        tokenizer,
        active_eval_tasks + ["sort"],
        eval_data,
        cfg,
        expected_steps=cfg.consolidation_steps,
    )
    if expanded_best_d is None:
        expanded_best_d = dual_teacher_consolidation(
            expanded_best_student,
            active_old_model,
            teacher_best_model,
            tokenizer,
            d_batch_fn,
            active_old_batch_fn,
            active_eval_tasks + ["sort"],
            eval_data,
            tomography_d.selected_layer_indices,
            cfg.consolidation_steps,
            cfg.consolidation_lr,
            "expanded_best_D",
            expanded_best_dir,
            cfg,
            {"primary": best_d_cmp},
        )
        expanded_best_model = None
    _mark_sort_phase(expanded_best_d, expanded_best_dir, cfg)

    expanded_balanced = None
    if expanded_teacher_balanced is not None:
        expanded_balanced_student = _clone_model(active_old_model, cfg.device)
        expanded_balanced_student, _balanced_gate = insert_expansion_layer(expanded_balanced_student, tomography_d.selected_layer_indices[0])
        teacher_balanced_model = _materialize_phase_model(expanded_teacher_model, expanded_teacher_balanced, cfg.device)
        expanded_balanced_dir = _phase_output_dir(seed_dir, "expanded_balanced")
        expanded_balanced, expanded_balanced_model = _resume_phase_from_artifacts(
            expanded_balanced_dir,
            "expanded_balanced",
            expanded_balanced_student,
            tokenizer,
            active_eval_tasks + ["sort"],
            eval_data,
            cfg,
            expected_steps=cfg.consolidation_steps,
        )
        if expanded_balanced is None:
            expanded_balanced = dual_teacher_consolidation(
                expanded_balanced_student,
                active_old_model,
                teacher_balanced_model,
                tokenizer,
                d_batch_fn,
                active_old_batch_fn,
                active_eval_tasks + ["sort"],
                eval_data,
                tomography_d.selected_layer_indices,
                cfg.consolidation_steps,
                cfg.consolidation_lr,
                "expanded_balanced",
                expanded_balanced_dir,
                cfg,
                {"primary": balanced_cmp},
            )
            expanded_balanced_model = None
        _mark_sort_phase(expanded_balanced, expanded_balanced_dir, cfg)
        _release_cuda_memory(teacher_balanced_model, expanded_balanced_student, expanded_balanced_model)
    expanded_headline = _headline_choice(fixed_frontier, expanded_best_d, expanded_balanced, "sort_token_acc")
    if gated_layer_ref is not None:
        gate_value = float(gated_layer_ref.gate_value.item())
        gate_floor = gated_layer_ref.gate_floors[2]
        insert_after_metric = float(insert_after)
        expanded_teacher_d.metrics["expansion_gate"] = gate_value
        expanded_teacher_d.metrics["expansion_gate_floor"] = gate_floor
        expanded_teacher_d.metrics["expansion_gate_delta"] = gate_value - gate_floor
        expanded_teacher_d.metrics["expansion_insert_after"] = insert_after_metric
        expanded_best_d.metrics["expansion_gate"] = gate_value
        expanded_best_d.metrics["expansion_gate_floor"] = gate_floor
        expanded_best_d.metrics["expansion_gate_delta"] = gate_value - gate_floor
        expanded_best_d.metrics["expansion_insert_after"] = insert_after_metric
        if expanded_balanced is not None:
            expanded_balanced.metrics["expansion_gate"] = gate_value
            expanded_balanced.metrics["expansion_gate_floor"] = gate_floor
            expanded_balanced.metrics["expansion_gate_delta"] = gate_value - gate_floor
            expanded_balanced.metrics["expansion_insert_after"] = insert_after_metric
        expanded_headline.metrics["expansion_gate"] = gate_value
        expanded_headline.metrics["expansion_gate_floor"] = gate_floor
        expanded_headline.metrics["expansion_gate_delta"] = gate_value - gate_floor
        expanded_headline.metrics["expansion_insert_after"] = insert_after_metric
    _release_cuda_memory(expanded_teacher_primary_model, teacher_best_model, expanded_teacher_model, expanded_best_student, expanded_best_model)

    controls: Dict[str, PhaseResult] = {}
    if cfg.run_controls and cfg.phase_scope in {"abd_rescue", "full_abcd", "controls"}:
        standard_cfg = copy.deepcopy(cfg)
        standard_cfg.adapter_config = LatentLoRAConfig(
            rank=cfg.adapter_config.rank,
            alpha=cfg.adapter_config.alpha,
            dropout=cfg.adapter_config.dropout,
            projection_strength=0.0,
            gate_init=cfg.adapter_config.gate_init,
            freeze_base=True,
        )
        standard_teacher_model = _clone_model(active_old_model, cfg.device)
        _freeze_model(standard_teacher_model)
        attached_standard = attach_latent_lora(
            standard_teacher_model,
            suffixes=_task_target_suffixes("sort", standard_cfg),
            layer_indices=set(tomography_d.selected_layer_indices),
            config=_task_adapter_config("sort", standard_cfg),
        )
        standard_teacher_dir = _phase_output_dir(seed_dir, "control_standard_lora_teacher")
        standard_teacher, standard_teacher_model_loaded = _resume_phase_from_artifacts(
            standard_teacher_dir,
            "control_standard_lora_teacher",
            standard_teacher_model,
            tokenizer,
            active_eval_tasks + ["sort"],
            eval_data,
            standard_cfg,
            expected_steps=_task_attach_steps("sort", standard_cfg),
        )
        if standard_teacher is None:
            standard_teacher = train_adapter_teacher(
                standard_teacher_model,
                tokenizer,
                attached_standard,
                d_batch_fn,
                active_old_batch_fn,
                ESCAPE_SCHEDULE,
                active_eval_tasks + ["sort"],
                eval_data,
                _task_attach_steps("sort", standard_cfg),
                _task_attach_lr("sort", standard_cfg),
                "control_standard_lora_teacher",
                standard_teacher_dir,
                standard_cfg,
                {"primary": _d_metric_cmp},
            )
            standard_teacher_model_loaded = _materialize_phase_model(standard_teacher_model, standard_teacher, cfg.device)
        elif standard_teacher_model_loaded is None:
            standard_teacher_model_loaded = _materialize_phase_model(standard_teacher_model, standard_teacher, cfg.device)
        _mark_sort_phase(standard_teacher, standard_teacher_dir, cfg)
        controls["standard_lora"] = _run_control_fixed_size(
            "standard_lora",
            active_old_model,
            standard_teacher_model_loaded,
            tokenizer,
            active_old_batch_fn,
            d_batch_fn,
            active_eval_tasks + ["sort"],
            eval_data,
            tomography_d.selected_layer_indices,
            cfg,
            seed_dir,
            fixed_frontier,
        )
        _release_cuda_memory(standard_teacher_model, standard_teacher_model_loaded)
        high_rank_cfg = copy.deepcopy(cfg)
        high_rank_cfg.adapter_config = LatentLoRAConfig(
            rank=64,
            alpha=128.0,
            dropout=0.0,
            projection_strength=1.0,
            gate_init=-6.0,
            freeze_base=True,
        )
        high_rank_teacher = _clone_model(active_old_model, cfg.device)
        _freeze_model(high_rank_teacher)
        attached_high_rank = attach_latent_lora(
            high_rank_teacher,
            suffixes=_task_target_suffixes("sort", high_rank_cfg),
            layer_indices=set(tomography_d.selected_layer_indices),
            config=high_rank_cfg.adapter_config,
        )
        high_rank_teacher_dir = _phase_output_dir(seed_dir, "control_high_rank_teacher")
        high_rank_teacher_phase, high_rank_teacher_loaded = _resume_phase_from_artifacts(
            high_rank_teacher_dir,
            "control_high_rank_teacher",
            high_rank_teacher,
            tokenizer,
            active_eval_tasks + ["sort"],
            eval_data,
            high_rank_cfg,
            expected_steps=_task_attach_steps("sort", high_rank_cfg),
        )
        if high_rank_teacher_phase is None:
            high_rank_teacher_phase = train_adapter_teacher(
                high_rank_teacher,
                tokenizer,
                attached_high_rank,
                d_batch_fn,
                active_old_batch_fn,
                ESCAPE_SCHEDULE,
                active_eval_tasks + ["sort"],
                eval_data,
                _task_attach_steps("sort", high_rank_cfg),
                _task_attach_lr("sort", high_rank_cfg),
                "control_high_rank_teacher",
                high_rank_teacher_dir,
                high_rank_cfg,
                {"primary": _d_metric_cmp},
            )
            high_rank_teacher_loaded = _materialize_phase_model(high_rank_teacher, high_rank_teacher_phase, cfg.device)
        elif high_rank_teacher_loaded is None:
            high_rank_teacher_loaded = _materialize_phase_model(high_rank_teacher, high_rank_teacher_phase, cfg.device)
        _mark_sort_phase(high_rank_teacher_phase, high_rank_teacher_dir, cfg)
        controls["high_rank"] = _run_control_fixed_size(
            "high_rank",
            active_old_model,
            high_rank_teacher_loaded,
            tokenizer,
            active_old_batch_fn,
            d_batch_fn,
            active_eval_tasks + ["sort"],
            eval_data,
            tomography_d.selected_layer_indices,
            cfg,
            seed_dir,
            fixed_frontier,
        )
        _release_cuda_memory(high_rank_teacher, high_rank_teacher_loaded)
        gate_null_model = _clone_model(active_old_model, cfg.device)
        gate_null_model, gate_null_layer = insert_expansion_layer(gate_null_model, tomography_d.selected_layer_indices[0], gate_init=0.001, gate_floors=(0.001, 0.001, 0.001))
        gate_null_layer.freeze_gate = True
        _freeze_model(gate_null_model)
        for idx, layer in enumerate(gate_null_model.model.layers):
            trainable = isinstance(layer, GatedQwenLayer)
            for param in layer.parameters():
                param.requires_grad = trainable
        attached_gate_null = attach_latent_lora(
            gate_null_model,
            suffixes=_task_target_suffixes("sort", cfg),
            layer_indices=set(tomography_d.selected_layer_indices),
            config=_task_adapter_config("sort", cfg),
        )
        gate_null_teacher_dir = _phase_output_dir(seed_dir, "control_gate_null_teacher")
        gate_null_teacher, gate_null_teacher_model = _resume_phase_from_artifacts(
            gate_null_teacher_dir,
            "control_gate_null_teacher",
            gate_null_model,
            tokenizer,
            active_eval_tasks + ["sort"],
            eval_data,
            cfg,
            expected_steps=_task_attach_steps("sort", cfg),
        )
        if gate_null_teacher is None:
            gate_null_teacher = train_adapter_teacher(
                gate_null_model,
                tokenizer,
                attached_gate_null,
                d_batch_fn,
                active_old_batch_fn,
                ESCAPE_SCHEDULE,
                active_eval_tasks + ["sort"],
                eval_data,
                _task_attach_steps("sort", cfg),
                _task_attach_lr("sort", cfg),
                "control_gate_null_teacher",
                gate_null_teacher_dir,
                cfg,
                {"primary": _d_metric_cmp},
            )
            gate_null_teacher_model = _materialize_phase_model(gate_null_model, gate_null_teacher, cfg.device)
        elif gate_null_teacher_model is None:
            gate_null_teacher_model = _materialize_phase_model(gate_null_model, gate_null_teacher, cfg.device)
        _mark_sort_phase(gate_null_teacher, gate_null_teacher_dir, cfg)
        gate_null_student = _clone_model(active_old_model, cfg.device)
        gate_null_student, gate_null_layer_student = insert_expansion_layer(gate_null_student, tomography_d.selected_layer_indices[0], gate_init=0.001, gate_floors=(0.001, 0.001, 0.001))
        gate_null_layer_student.freeze_gate = True
        controls["gate_null"] = dual_teacher_consolidation(
            gate_null_student,
            active_old_model,
            gate_null_teacher_model,
            tokenizer,
            d_batch_fn,
            active_old_batch_fn,
            active_eval_tasks + ["sort"],
            eval_data,
            tomography_d.selected_layer_indices,
            cfg.consolidation_steps,
            cfg.consolidation_lr,
            "control_gate_null",
            _phase_output_dir(seed_dir, "control_gate_null"),
            cfg,
            {"primary": _d_metric_cmp},
        )
        _mark_sort_phase(controls["gate_null"], _phase_output_dir(seed_dir, "control_gate_null"), cfg)
        _release_cuda_memory(gate_null_model, gate_null_teacher_model, gate_null_student)

    result = ProofResult(
        seed=cfg.seed,
        base_a=base_a,
        teacher_b=teacher_b,
        base_ab=base_ab,
        teacher_c=teacher_c,
        base_abc=base_abc,
        tomography_d=tomography_d,
        fixed_teacher_d=fixed_teacher_d,
        fixed_frontier=fixed_frontier,
        fixed_final=fixed_final,
        saturation_history=saturation_history,
        expanded_teacher_d=expanded_teacher_d,
        expanded_best_d=expanded_best_d,
        expanded_balanced=expanded_balanced,
        expanded_headline=expanded_headline,
        controls=controls,
    )
    _save_seed_result(seed_dir / "seed_result.json", result, cfg)
    return result


def run_full_proof(cfg: RuntimeConfig) -> List[ProofResult]:
    _restore_output_from_backup(cfg)
    seeds = [cfg.seed]
    if getattr(cfg, "all_seeds", False):
        seeds = list(PROOF_SEEDS)
    results: List[ProofResult] = []
    for seed in seeds:
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg.seed = seed
        result = run_proof_seed(seed_cfg)
        results.append(result)
    _save_summary_csv(cfg.output_dir / "summary.csv", results, cfg)
    print(_summary_table(results))
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen continual-learning proof runner")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default="/Users/swapnil/Desktop/chaos/results/qwen_continual_proof")
    parser.add_argument("--backup-dir", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--task-suite", choices=list(TASK_SUITES), default="proof_v2")
    parser.add_argument("--update-task", choices=["retention", "json", "reversal", "sort"], default=None)
    parser.add_argument("--wikitext-eval-samples", type=int, default=WIKITEXT_EVAL_SAMPLES)
    parser.add_argument("--json-eval-samples", type=int, default=JSON_EVAL_SAMPLES)
    parser.add_argument("--reversal-eval-samples", type=int, default=REVERSAL_EVAL_SAMPLES)
    parser.add_argument("--sort-eval-samples", type=int, default=SORT_EVAL_SAMPLES)
    parser.add_argument("--proof-v2-b-attach-steps", type=int, default=PROOF_V2_B_ATTACH_STEPS)
    parser.add_argument("--proof-v2-b-attach-lr", type=float, default=PROOF_V2_B_ATTACH_LR)
    parser.add_argument("--proof-v2-b-rank", type=int, default=PROOF_V2_B_RANK)
    parser.add_argument("--proof-v2-b-alpha", type=float, default=PROOF_V2_B_ALPHA)
    parser.add_argument("--proof-v2-b-gate-init", type=float, default=PROOF_V2_B_GATE_INIT)
    parser.add_argument("--proof-v2-b-min-layers", type=int, default=PROOF_V2_B_MIN_LAYERS)
    parser.set_defaults(proof_v2_b_use_up_proj=PROOF_V2_B_USE_UP_PROJ)
    parser.add_argument("--proof-v2-b-use-up-proj", dest="proof_v2_b_use_up_proj", action="store_true")
    parser.add_argument("--proof-v2-b-no-up-proj", dest="proof_v2_b_use_up_proj", action="store_false")
    parser.add_argument("--proof-v2-d-attach-steps", type=int, default=PROOF_V2_D_ATTACH_STEPS)
    parser.add_argument("--proof-v2-d-attach-lr", type=float, default=PROOF_V2_D_ATTACH_LR)
    parser.add_argument("--proof-v2-d-rank", type=int, default=PROOF_V2_D_RANK)
    parser.add_argument("--proof-v2-d-alpha", type=float, default=PROOF_V2_D_ALPHA)
    parser.add_argument("--proof-v2-d-gate-init", type=float, default=PROOF_V2_D_GATE_INIT)
    parser.add_argument("--proof-v2-d-min-layers", type=int, default=PROOF_V2_D_MIN_LAYERS)
    parser.set_defaults(proof_v2_d_use_up_proj=PROOF_V2_D_USE_UP_PROJ)
    parser.add_argument("--proof-v2-d-use-up-proj", dest="proof_v2_d_use_up_proj", action="store_true")
    parser.add_argument("--proof-v2-d-no-up-proj", dest="proof_v2_d_use_up_proj", action="store_false")
    parser.add_argument("--consolidation-steps", type=int, default=CONSOLIDATION_STEPS)
    parser.add_argument("--consolidation-lr", type=float, default=CONSOLIDATION_LR)
    parser.add_argument("--consol-kl-weight", type=float, default=CONSOL_KL_WEIGHT)
    parser.add_argument("--consol-old-kl-weight", type=float, default=CONSOL_OLD_KL_WEIGHT)
    parser.add_argument("--consol-hidden-weight", type=float, default=CONSOL_HIDDEN_WEIGHT)
    parser.add_argument("--consol-old-batch-period", type=int, default=CONSOL_OLD_BATCH_PERIOD)
    parser.add_argument("--consol-amoeba", action="store_true")
    parser.add_argument("--consol-amoeba-gentle-frac", type=float, default=CONSOL_AMOEBA_GENTLE_FRAC)
    parser.add_argument("--consol-amoeba-polish-frac", type=float, default=CONSOL_AMOEBA_POLISH_FRAC)
    parser.add_argument("--consol-amoeba-gentle-old-scale", type=float, default=CONSOL_AMOEBA_GENTLE_OLD_SCALE)
    parser.add_argument("--consol-amoeba-gentle-new-scale", type=float, default=CONSOL_AMOEBA_GENTLE_NEW_SCALE)
    parser.add_argument("--consol-amoeba-polish-lr-scale", type=float, default=CONSOL_AMOEBA_POLISH_LR_SCALE)
    parser.add_argument("--teacher-old-loss-weight", type=float, default=TEACHER_OLD_LOSS_WEIGHT)
    parser.add_argument("--teacher-old-batch-period", type=int, default=TEACHER_OLD_BATCH_PERIOD)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--consolidation-micro-batch-size", type=int, default=2)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--allow-frontier-fallback", action="store_true")
    parser.add_argument("--allow-ab-gate-bypass", action="store_true")
    parser.add_argument(
        "--phase-scope",
        default="abd_rescue",
        choices=["b_teacher", "d_teacher", "ab", "abd_fixed", "abd_rescue", "full_abcd", "controls"],
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = _resolve_runtime(args)
    cfg.all_seeds = bool(args.all_seeds)
    results = run_full_proof(cfg)
    if cfg.phase_scope not in {"ab", "b_teacher", "d_teacher"}:
        try:
            from qwen_proof_plots import generate_all_plots
        except Exception:
            return
        generate_all_plots(results, cfg.output_dir / "plots")
        _sync_path_to_backup(cfg.output_dir / "plots", cfg.output_dir, cfg.backup_dir)


if __name__ == "__main__":
    main()
