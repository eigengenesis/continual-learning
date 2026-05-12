from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


TOMOGRAPHY_BATCHES = 8
TOMOGRAPHY_ACT_RANK = 32
TOMOGRAPHY_GRAD_RANK = 16
DEFAULT_TARGET_SUFFIX = "mlp.down_proj"
DEFAULT_TOP_LAYERS = 6
MIN_PRESSURE_FRACTION_FOR_6 = 0.50
MAX_PRESSURE_FRACTION_FOR_4 = 0.60

SATURATION_OCCUPIED_THRESHOLD = 0.70
SATURATION_FREE_RANK_THRESHOLD = 0.15
SATURATION_CONSECUTIVE_REQUIRED = 3

SATURATION_WEIGHT_OCCUPIED = 0.45
SATURATION_WEIGHT_ACTIVATION = 0.35
SATURATION_WEIGHT_RANK_PRESSURE = 0.20

_EPS = 1e-12
_SAMPLE_LIMIT = 512


@dataclass
class LayerProfile:
    layer_index: int
    layer_name: str
    activation_basis: torch.Tensor
    gradient_basis: torch.Tensor
    effective_act_rank: int
    effective_grad_rank: int
    explained_variance: float


@dataclass
class TaskProfile:
    task_name: str
    stage_label: str
    layer_profiles: Dict[int, LayerProfile]


@dataclass
class LayerSaturation:
    layer_index: int
    layer_name: str
    hidden_dim: int
    gradient_norm: float
    activation_shift: float
    occupied_overlap: float
    free_overlap: float
    free_rank_estimate: int
    activation_overlap: float
    rank_pressure: float
    saturation_score: float
    learning_pressure: float


@dataclass
class UnitScore:
    layer_index: int
    layer_name: str
    module_name: str
    module_suffix: str
    unit_type: str
    unit_index: int
    unit_start: int
    unit_end: int
    unit_count: int
    domain_pressure: float
    language_occupancy: float
    free_domain_score: float


@dataclass
class SaturationReport:
    step: int
    phase: str
    layer_saturations: List[LayerSaturation]
    model_mean_saturation: float
    model_mean_occupied_overlap: float
    model_mean_free_rank_fraction: float
    trigger_eligible: bool
    expansion_trigger: bool
    trigger_reason: str
    consecutive_trigger_count: int


@dataclass
class TomographyResult:
    layer_saturations: List[LayerSaturation]
    selected_layer_indices: List[int]
    selection_reason: str
    total_pressure: float


def _layer_index_from_name(name: str) -> int:
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    raise ValueError(f"could not parse layer index from module name: {name}")


def _matching_modules(
    model: nn.Module,
    target_suffix: str,
    selected_layers: Sequence[int] | None = None,
) -> List[Tuple[int, str, nn.Module]]:
    selected = None if selected_layers is None else set(int(v) for v in selected_layers)
    out: List[Tuple[int, str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name.endswith(target_suffix):
            continue
        if not isinstance(module, nn.Module):
            continue
        layer_index = _layer_index_from_name(name)
        if selected is not None and layer_index not in selected:
            continue
        out.append((layer_index, name, module))
    out.sort(key=lambda item: item[0])
    return out


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _flatten_hidden(x: torch.Tensor, limit: int = _SAMPLE_LIMIT) -> torch.Tensor:
    if x.ndim == 2:
        flat = x
    else:
        flat = x.reshape(-1, x.shape[-1])
    if flat.size(0) > limit:
        idx = torch.linspace(0, flat.size(0) - 1, steps=limit, device=flat.device).round().long()
        flat = flat.index_select(0, idx)
    return flat.detach().float().cpu()


def _collect_layer_samples(
    model: nn.Module,
    task_batches: List[Dict[str, torch.Tensor]],
    target_suffix: str = DEFAULT_TARGET_SUFFIX,
    selected_layers: Sequence[int] | None = None,
) -> Dict[int, Dict[str, object]]:
    modules = _matching_modules(model, target_suffix, selected_layers)
    if not modules:
        raise ValueError(f"no modules found for suffix {target_suffix}")

    device = next(model.parameters()).device
    buffers: Dict[int, Dict[str, object]] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []
    original_requires_grad: Dict[nn.Parameter, bool] = {}

    for layer_index, name, module in modules:
        buffers[layer_index] = {
            "layer_name": name,
            "activations": [],
            "gradients": [],
        }
        for param in module.parameters():
            if param not in original_requires_grad:
                original_requires_grad[param] = bool(param.requires_grad)
                param.requires_grad_(True)

        def _hook(_module, _inputs, output, *, layer_idx=layer_index):
            if isinstance(output, tuple):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return output
            buffers[layer_idx]["activations"].append(_flatten_hidden(output))
            if output.requires_grad:
                output.register_hook(
                    lambda grad, li=layer_idx: buffers[li]["gradients"].append(_flatten_hidden(grad))
                )
            return output

        handles.append(module.register_forward_hook(_hook))

    was_training = model.training
    model.train()
    try:
        with torch.enable_grad():
            for batch in task_batches:
                batch = _move_batch_to_device(batch, device)
                model.zero_grad(set_to_none=True)
                outputs = model(**batch, use_cache=False)
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = batch["input_ids"][..., 1:].contiguous()
                    loss = nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                loss.backward()
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        for param, flag in original_requires_grad.items():
            param.requires_grad_(flag)
        model.train(was_training)

    return buffers


def _compute_basis(samples: List[torch.Tensor], rank: int) -> Tuple[torch.Tensor, int, float]:
    if not samples:
        return torch.empty(0, 0), 0, 0.0
    matrix = torch.cat(samples, dim=0)
    if matrix.ndim != 2 or matrix.numel() == 0:
        return torch.empty(0, 0), 0, 0.0
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    try:
        _u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError:
        return torch.empty(matrix.size(1), 0), 0, 0.0
    if s.numel() == 0:
        return torch.empty(matrix.size(1), 0), 0, 0.0
    keep = min(rank, vh.size(0))
    basis = vh[:keep].transpose(0, 1).contiguous()
    max_sv = float(s.max().item())
    effective_rank = int((s[:keep] > max_sv * 0.01).sum().item()) if max_sv > 0 else 0
    explained = float((s[:keep] ** 2).sum().item() / ((s ** 2).sum().item() + _EPS))
    return basis, effective_rank, explained


def collect_task_profile(
    model: nn.Module,
    task_batches: List[Dict[str, torch.Tensor]],
    target_suffix: str = DEFAULT_TARGET_SUFFIX,
    act_rank: int = TOMOGRAPHY_ACT_RANK,
    grad_rank: int = TOMOGRAPHY_GRAD_RANK,
    task_name: str = "",
    stage_label: str = "",
) -> TaskProfile:
    raw = _collect_layer_samples(model, task_batches, target_suffix)
    profiles: Dict[int, LayerProfile] = {}
    for layer_index, payload in raw.items():
        activation_basis, effective_act_rank, explained = _compute_basis(
            payload["activations"], act_rank
        )
        gradient_basis, effective_grad_rank, _grad_explained = _compute_basis(
            payload["gradients"], grad_rank
        )
        layer_name = str(payload["layer_name"])
        hidden_dim = 0
        if activation_basis.numel() > 0:
            hidden_dim = activation_basis.size(0)
        elif gradient_basis.numel() > 0:
            hidden_dim = gradient_basis.size(0)
        profiles[layer_index] = LayerProfile(
            layer_index=layer_index,
            layer_name=layer_name,
            activation_basis=activation_basis if activation_basis.numel() > 0 else torch.empty(hidden_dim, 0),
            gradient_basis=gradient_basis if gradient_basis.numel() > 0 else torch.empty(hidden_dim, 0),
            effective_act_rank=effective_act_rank,
            effective_grad_rank=effective_grad_rank,
            explained_variance=explained,
        )
    return TaskProfile(task_name=task_name, stage_label=stage_label, layer_profiles=profiles)


def build_occupied_basis(
    old_profiles: List[TaskProfile],
    layer_index: int,
    mode: str = "union",
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for profile in old_profiles:
        layer_profile = profile.layer_profiles.get(layer_index)
        if layer_profile is None:
            continue
        if mode == "activation":
            if layer_profile.activation_basis.numel() > 0:
                chunks.append(layer_profile.activation_basis)
            continue
        if mode == "gradient":
            if layer_profile.gradient_basis.numel() > 0:
                chunks.append(layer_profile.gradient_basis)
            continue
        if layer_profile.activation_basis.numel() > 0:
            chunks.append(layer_profile.activation_basis)
        if layer_profile.gradient_basis.numel() > 0:
            chunks.append(layer_profile.gradient_basis)
    if not chunks:
        return torch.empty(0, 0)
    union = torch.cat(chunks, dim=1)
    q, _ = torch.linalg.qr(union, mode="reduced")
    return q


def _mean_projection_overlap(samples: torch.Tensor, basis: torch.Tensor) -> float:
    if samples.numel() == 0 or basis.numel() == 0:
        return 0.0
    proj = torch.matmul(samples, basis)
    proj_energy = (proj ** 2).sum(dim=-1)
    total_energy = (samples ** 2).sum(dim=-1).clamp_min(_EPS)
    return float((proj_energy / total_energy).mean().item())


def _unit_occupancy(vectors: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Protected-basis energy fraction for matrix columns."""
    if vectors.numel() == 0:
        return torch.empty(0)
    if basis.numel() == 0 or basis.shape[0] != vectors.shape[0]:
        return torch.zeros(vectors.shape[1], dtype=vectors.dtype)
    projected = basis.transpose(0, 1) @ vectors
    projected_energy = projected.square().sum(dim=0)
    total_energy = vectors.square().sum(dim=0).clamp_min(_EPS)
    return projected_energy / total_energy


def _unit_score(domain_pressure: float, language_occupancy: float, eps: float) -> float:
    return float(domain_pressure) / max(float(language_occupancy), float(eps))


def collect_fine_grained_unit_scores(
    model: nn.Module,
    task_batches: List[Dict[str, torch.Tensor]],
    old_profiles: List[TaskProfile],
    *,
    target_suffix: str = DEFAULT_TARGET_SUFFIX,
    unit_type: str = "mlp_column",
    num_batches: int = TOMOGRAPHY_BATCHES,
    num_heads: int | None = None,
    occupancy_mode: str = "activation",
    score_eps: float = 1e-3,
) -> List[UnitScore]:
    """Score MLP columns or attention heads by language-free domain pressure.

    For the default `mlp_column` mode, each `mlp.down_proj` column is one MLP
    channel's residual contribution. Domain pressure is the column gradient
    norm on the domain batches. Language occupancy is the fraction of that
    gradient in the protected language basis. The resulting score is:

        domain_pressure / (language_occupancy + eps)

    `attention_head` mode groups `o_proj` columns into head slices for logging.
    """
    if unit_type not in {"mlp_column", "attention_head"}:
        raise ValueError(f"unsupported unit_type: {unit_type}")

    modules = _matching_modules(model, target_suffix)
    if not modules:
        raise ValueError(f"no modules found for suffix {target_suffix}")

    inferred_heads = num_heads
    if inferred_heads is None:
        inferred_heads = int(getattr(getattr(model, "config", None), "num_attention_heads", 0) or 0)

    device = next(model.parameters()).device
    original_requires_grad: Dict[nn.Parameter, bool] = {}
    accum: Dict[Tuple[int, str], Dict[str, object]] = {}
    for layer_index, name, module in modules:
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            continue
        original_requires_grad[weight] = bool(weight.requires_grad)
        weight.requires_grad_(True)
        unit_count = int(weight.shape[1])
        if unit_type == "attention_head":
            if inferred_heads <= 0 or int(weight.shape[1]) % inferred_heads != 0:
                continue
            unit_count = inferred_heads
        accum[(layer_index, name)] = {
            "module": module,
            "domain": torch.zeros(unit_count, dtype=torch.float64),
            "occupancy": torch.zeros(unit_count, dtype=torch.float64),
            "count": 0,
            "unit_count": unit_count,
        }

    was_training = model.training
    model.train()
    try:
        with torch.enable_grad():
            for batch in task_batches[:num_batches]:
                batch = _move_batch_to_device(batch, device)
                model.zero_grad(set_to_none=True)
                outputs = model(**batch, use_cache=False)
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = batch["input_ids"][..., 1:].contiguous()
                    loss = nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                loss.backward()

                for layer_index, name, module in modules:
                    payload = accum.get((layer_index, name))
                    if payload is None:
                        continue
                    weight = getattr(module, "weight", None)
                    grad = None if weight is None else weight.grad
                    if grad is None or grad.ndim != 2:
                        continue
                    grad_cpu = grad.detach().float().cpu()
                    basis = build_occupied_basis(old_profiles, layer_index, mode=occupancy_mode)
                    if basis.numel() == 0 or basis.shape[0] != grad_cpu.shape[0]:
                        basis = build_occupied_basis(old_profiles, layer_index, mode="union")
                    basis = basis.detach().float().cpu()

                    if unit_type == "mlp_column":
                        pressure = grad_cpu.norm(dim=0).double()
                        occupancy = _unit_occupancy(grad_cpu, basis).double()
                    else:
                        head_count = int(payload["unit_count"])
                        head_dim = grad_cpu.shape[1] // max(head_count, 1)
                        pressures: List[float] = []
                        occupancies: List[float] = []
                        for head_idx in range(head_count):
                            start = head_idx * head_dim
                            end = start + head_dim
                            block = grad_cpu[:, start:end]
                            pressures.append(float(block.norm().item()))
                            if basis.numel() == 0 or basis.shape[0] != block.shape[0]:
                                occupancies.append(0.0)
                            else:
                                projected = basis.transpose(0, 1) @ block
                                occupancies.append(
                                    float(projected.square().sum().item() / max(float(block.square().sum().item()), _EPS))
                                )
                        pressure = torch.tensor(pressures, dtype=torch.float64)
                        occupancy = torch.tensor(occupancies, dtype=torch.float64)

                    payload["domain"] = payload["domain"] + pressure
                    payload["occupancy"] = payload["occupancy"] + occupancy
                    payload["count"] = int(payload["count"]) + 1
    finally:
        model.zero_grad(set_to_none=True)
        for param, flag in original_requires_grad.items():
            param.requires_grad_(flag)
        model.train(was_training)

    scores: List[UnitScore] = []
    for layer_key, payload in accum.items():
        layer_index, name = layer_key
        count = max(int(payload["count"]), 1)
        domain = (payload["domain"] / count).float()
        occupancy = (payload["occupancy"] / count).float().clamp(min=0.0, max=1.0)
        unit_count = int(payload["unit_count"])
        span_width = 1
        if unit_type == "attention_head":
            module = payload["module"]
            weight = getattr(module, "weight", None)
            span_width = int(weight.shape[1]) // max(unit_count, 1) if weight is not None else 1
        for unit_index in range(unit_count):
            start = unit_index * span_width
            end = start + span_width
            pressure = float(domain[unit_index].item())
            occ = float(occupancy[unit_index].item())
            scores.append(
                UnitScore(
                    layer_index=int(layer_index),
                    layer_name=str(name),
                    module_name=str(name),
                    module_suffix=str(target_suffix),
                    unit_type=unit_type,
                    unit_index=int(unit_index),
                    unit_start=int(start),
                    unit_end=int(end),
                    unit_count=int(unit_count),
                    domain_pressure=pressure,
                    language_occupancy=occ,
                    free_domain_score=_unit_score(pressure, occ, score_eps),
                )
            )
    scores.sort(key=lambda item: item.free_domain_score, reverse=True)
    return scores


def select_top_unit_indices(
    scores: Sequence[UnitScore],
    selected_layers: Sequence[int],
    *,
    budget_fraction: float,
    min_units_per_layer: int = 1,
    max_units_per_layer: int | None = None,
) -> Dict[int, List[int]]:
    selected = {int(layer) for layer in selected_layers}
    by_layer: Dict[int, List[UnitScore]] = {}
    unit_counts: Dict[int, int] = {}
    for score in scores:
        layer_index = int(score.layer_index)
        if layer_index not in selected:
            continue
        by_layer.setdefault(layer_index, []).append(score)
        unit_counts[layer_index] = max(unit_counts.get(layer_index, 0), int(score.unit_count))

    out: Dict[int, List[int]] = {}
    for layer_index, layer_scores in by_layer.items():
        ordered = sorted(layer_scores, key=lambda item: item.free_domain_score, reverse=True)
        count = max(unit_counts.get(layer_index, len(ordered)), 1)
        keep = max(int(min_units_per_layer), int(math.ceil(count * float(budget_fraction))))
        if max_units_per_layer is not None and max_units_per_layer > 0:
            keep = min(keep, int(max_units_per_layer))
        keep = min(keep, len(ordered))
        out[layer_index] = sorted(int(item.unit_index) for item in ordered[:keep])
    return out


def compute_layer_saturation(
    layer_index: int,
    layer_name: str,
    occupied_basis: torch.Tensor,
    new_gradient_samples: torch.Tensor,
    new_activation_samples: torch.Tensor,
    hidden_dim: int,
) -> LayerSaturation:
    if occupied_basis.numel() == 0:
        occupied_basis = torch.empty(hidden_dim, 0)
    occupied_overlap = _mean_projection_overlap(new_gradient_samples, occupied_basis)
    free_overlap = 1.0 - occupied_overlap
    free_rank_estimate = max(hidden_dim - occupied_basis.shape[1], 0)
    activation_overlap = _mean_projection_overlap(new_activation_samples, occupied_basis)
    rank_pressure = 1.0 - max(min(free_rank_estimate / max(hidden_dim, 1), 1.0), 0.0)
    saturation_score = (
        SATURATION_WEIGHT_OCCUPIED * occupied_overlap
        + SATURATION_WEIGHT_ACTIVATION * activation_overlap
        + SATURATION_WEIGHT_RANK_PRESSURE * rank_pressure
    )
    gradient_norm = float(new_gradient_samples.norm(dim=-1).mean().item()) if new_gradient_samples.numel() else 0.0
    activation_shift = float(new_activation_samples.norm(dim=-1).mean().item()) if new_activation_samples.numel() else 0.0
    return LayerSaturation(
        layer_index=layer_index,
        layer_name=layer_name,
        hidden_dim=hidden_dim,
        gradient_norm=gradient_norm,
        activation_shift=activation_shift,
        occupied_overlap=occupied_overlap,
        free_overlap=free_overlap,
        free_rank_estimate=free_rank_estimate,
        activation_overlap=activation_overlap,
        rank_pressure=rank_pressure,
        saturation_score=saturation_score,
        learning_pressure=0.0,
    )


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo + _EPS:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _finalize_learning_pressure(items: List[LayerSaturation]) -> List[LayerSaturation]:
    grad_scores = _minmax([item.gradient_norm for item in items])
    act_scores = _minmax([item.activation_shift for item in items])
    free_scores = _minmax([1.0 - item.rank_pressure for item in items])
    out: List[LayerSaturation] = []
    for item, grad_score, act_score, free_score in zip(items, grad_scores, act_scores, free_scores):
        out.append(
            LayerSaturation(
                **{
                    **asdict(item),
                    "learning_pressure": 0.50 * grad_score + 0.30 * act_score + 0.20 * free_score,
                }
            )
        )
    return out


def run_tomography(
    model: nn.Module,
    tokenizer,
    task_batches: List[Dict[str, torch.Tensor]],
    old_profiles: List[TaskProfile],
    target_suffix: str = DEFAULT_TARGET_SUFFIX,
    num_batches: int = TOMOGRAPHY_BATCHES,
) -> TomographyResult:
    del tokenizer  # not needed, retained for interface symmetry
    raw = _collect_layer_samples(model, task_batches[:num_batches], target_suffix)
    layer_items: List[LayerSaturation] = []
    for layer_index, payload in raw.items():
        activations = payload["activations"]
        gradients = payload["gradients"]
        activation_samples = torch.cat(activations, dim=0) if activations else torch.empty(0, 0)
        gradient_samples = torch.cat(gradients, dim=0) if gradients else torch.empty(0, 0)
        hidden_dim = 0
        if activation_samples.numel() > 0:
            hidden_dim = activation_samples.shape[-1]
        elif gradient_samples.numel() > 0:
            hidden_dim = gradient_samples.shape[-1]
        occupied_basis = build_occupied_basis(old_profiles, layer_index)
        if occupied_basis.numel() == 0:
            occupied_basis = torch.empty(hidden_dim, 0)
        layer_items.append(
            compute_layer_saturation(
                layer_index=layer_index,
                layer_name=str(payload["layer_name"]),
                occupied_basis=occupied_basis,
                new_gradient_samples=gradient_samples,
                new_activation_samples=activation_samples,
                hidden_dim=hidden_dim,
            )
        )

    layer_items = _finalize_learning_pressure(layer_items)
    layer_items.sort(key=lambda item: item.learning_pressure, reverse=True)
    total_pressure = float(sum(item.learning_pressure for item in layer_items))
    top6_fraction = (
        sum(item.learning_pressure for item in layer_items[:6]) / max(total_pressure, _EPS)
        if layer_items
        else 0.0
    )
    top4_fraction = (
        sum(item.learning_pressure for item in layer_items[:4]) / max(total_pressure, _EPS)
        if layer_items
        else 0.0
    )
    if layer_items and top4_fraction > MAX_PRESSURE_FRACTION_FOR_4:
        selected = [item.layer_index for item in layer_items[:4]]
        reason = f"top 4 layers explain {top4_fraction:.3f} of total pressure"
    elif layer_items and top6_fraction < MIN_PRESSURE_FRACTION_FOR_6 and len(layer_items) >= 8:
        selected = [item.layer_index for item in layer_items[:8]]
        reason = f"top 6 layers explain only {top6_fraction:.3f}; expanding to 8"
    else:
        selected = [item.layer_index for item in layer_items[: min(DEFAULT_TOP_LAYERS, len(layer_items))]]
        reason = "default top-6 pressure selection"
    return TomographyResult(
        layer_saturations=layer_items,
        selected_layer_indices=selected,
        selection_reason=reason,
        total_pressure=total_pressure,
    )


def compute_saturation_report(
    model: nn.Module,
    tokenizer,
    task_batches: List[Dict[str, torch.Tensor]],
    old_profiles: List[TaskProfile],
    selected_layers: List[int],
    step: int,
    phase: str,
    trigger_history: List[SaturationReport],
    retention_delta: float = 0.0,
    task_progress_delta: float = 0.0,
) -> SaturationReport:
    del tokenizer
    raw = _collect_layer_samples(model, task_batches[:TOMOGRAPHY_BATCHES], DEFAULT_TARGET_SUFFIX, selected_layers)
    items: List[LayerSaturation] = []
    for layer_index, payload in raw.items():
        activations = payload["activations"]
        gradients = payload["gradients"]
        activation_samples = torch.cat(activations, dim=0) if activations else torch.empty(0, 0)
        gradient_samples = torch.cat(gradients, dim=0) if gradients else torch.empty(0, 0)
        hidden_dim = activation_samples.shape[-1] if activation_samples.numel() else gradient_samples.shape[-1]
        occupied_basis = build_occupied_basis(old_profiles, layer_index)
        if occupied_basis.numel() == 0:
            occupied_basis = torch.empty(hidden_dim, 0)
        items.append(
            compute_layer_saturation(
                layer_index,
                str(payload["layer_name"]),
                occupied_basis,
                gradient_samples,
                activation_samples,
                hidden_dim,
            )
        )
    items = _finalize_learning_pressure(items)
    mean_saturation = float(sum(item.saturation_score for item in items) / max(len(items), 1))
    mean_occupied = float(sum(item.occupied_overlap for item in items) / max(len(items), 1))
    mean_free_rank_fraction = float(
        sum(item.free_rank_estimate / max(item.hidden_dim, 1) for item in items)
        / max(len(items), 1)
    ) if items else 1.0
    trigger_eligible = (
        mean_occupied >= SATURATION_OCCUPIED_THRESHOLD
        and mean_free_rank_fraction <= SATURATION_FREE_RANK_THRESHOLD
        and (retention_delta > 0.10 or task_progress_delta < 0.05)
    )
    consecutive = 1 if trigger_eligible else 0
    if trigger_eligible:
        for previous in reversed(trigger_history):
            if previous.trigger_eligible:
                consecutive += 1
            else:
                break
    expansion_trigger = trigger_eligible and consecutive >= SATURATION_CONSECUTIVE_REQUIRED
    reason = ""
    if trigger_eligible:
        reason = (
            f"occupied={mean_occupied:.3f}, free_rank_fraction={mean_free_rank_fraction:.3f}, "
            f"retention_delta={retention_delta:.3f}, task_progress_delta={task_progress_delta:.3f}"
        )
    return SaturationReport(
        step=step,
        phase=phase,
        layer_saturations=items,
        model_mean_saturation=mean_saturation,
        model_mean_occupied_overlap=mean_occupied,
        model_mean_free_rank_fraction=mean_free_rank_fraction,
        trigger_eligible=trigger_eligible,
        expansion_trigger=expansion_trigger,
        trigger_reason=reason,
        consecutive_trigger_count=consecutive if trigger_eligible else 0,
    )


def should_expand(history: List[SaturationReport]) -> Tuple[bool, str]:
    if len(history) < SATURATION_CONSECUTIVE_REQUIRED:
        return False, ""
    window = history[-SATURATION_CONSECUTIVE_REQUIRED:]
    if all(item.trigger_eligible for item in window):
        return True, window[-1].trigger_reason
    return False, ""


def write_tomography_csv(
    path: Path,
    results: Iterable[TomographyResult | SaturationReport],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "phase",
                "layer_index",
                "layer_name",
                "gradient_norm",
                "activation_shift",
                "occupied_overlap",
                "free_overlap",
                "free_rank_estimate",
                "activation_overlap",
                "rank_pressure",
                "saturation_score",
                "learning_pressure",
                "selected",
                "trigger_eligible",
                "trigger",
            ]
        )
        for item in results:
            if isinstance(item, TomographyResult):
                selected = set(item.selected_layer_indices)
                for layer in item.layer_saturations:
                    writer.writerow(
                        [
                            "",
                            "tomography",
                            layer.layer_index,
                            layer.layer_name,
                            layer.gradient_norm,
                            layer.activation_shift,
                            layer.occupied_overlap,
                            layer.free_overlap,
                            layer.free_rank_estimate,
                            layer.activation_overlap,
                            layer.rank_pressure,
                            layer.saturation_score,
                            layer.learning_pressure,
                            int(layer.layer_index in selected),
                            0,
                            0,
                        ]
                    )
            else:
                for layer in item.layer_saturations:
                    writer.writerow(
                        [
                            item.step,
                            item.phase,
                            layer.layer_index,
                            layer.layer_name,
                            layer.gradient_norm,
                            layer.activation_shift,
                            layer.occupied_overlap,
                            layer.free_overlap,
                            layer.free_rank_estimate,
                            layer.activation_overlap,
                            layer.rank_pressure,
                            layer.saturation_score,
                            layer.learning_pressure,
                            1,
                            int(item.trigger_eligible),
                            int(item.expansion_trigger),
                        ]
                    )


def write_unit_scores_csv(path: Path, scores: Iterable[UnitScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "layer_index",
        "layer_name",
        "module_name",
        "module_suffix",
        "unit_type",
        "unit_index",
        "unit_start",
        "unit_end",
        "unit_count",
        "domain_pressure",
        "language_occupancy",
        "free_domain_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for score in scores:
            writer.writerow(asdict(score))
