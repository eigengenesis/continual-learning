from __future__ import annotations

import csv
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
