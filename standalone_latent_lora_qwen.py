from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None


def _require_transformers() -> None:
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError("transformers is required for Qwen latent LoRA scripts") from _IMPORT_ERROR


@dataclass
class LatentLoRAConfig:
    rank: int = 16
    alpha: float | None = None
    dropout: float = 0.0
    projection_strength: float = 1.0
    gate_init: float = -6.0
    freeze_base: bool = True


@dataclass
class EscapeSchedule:
    """Projection-strength schedule for adapter training."""

    levels: Tuple[float, ...] = (1.0, 0.65, 0.35, 0.15)
    step_fractions: Tuple[float, ...] = (0.0, 0.21, 0.50, 0.75)

    def get_strength(self, step: int, total_steps: int) -> float:
        progress = float(step) / float(max(total_steps, 1))
        strength = self.levels[0]
        for i, frac in enumerate(self.step_fractions):
            if progress >= frac:
                strength = self.levels[min(i, len(self.levels) - 1)]
        return strength

    def apply_to_modules(
        self,
        attached: List[Tuple[str, "LatentLoRALinear"]],
        step: int,
        total_steps: int,
    ) -> float:
        strength = self.get_strength(step, total_steps)
        for _, module in attached:
            module.set_projection_strength(strength)
        return strength


class LatentLoRALinear(nn.Module):
    """Standalone LoRA wrapper with soft null-space projection in output space."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int = 8,
        alpha: float | None = None,
        dropout: float = 0.0,
        projection_strength: float = 1.0,
        gate_init: float = -6.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LatentLoRALinear expects an nn.Linear base layer")
        if rank < 0:
            raise ValueError("rank must be >= 0")

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = int(rank)
        self.lora_alpha = float(alpha if alpha is not None else max(rank, 1))
        self.scaling = self.lora_alpha / float(max(self.rank, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.adapter_enabled = True
        self.gate_enabled = True
        self.freeze_base = freeze_base
        base_device = self.base_layer.weight.device
        base_dtype = self.base_layer.weight.dtype

        if freeze_base:
            for param in self.base_layer.parameters():
                param.requires_grad = False

        if self.rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty(self.rank, self.in_features, device=base_device, dtype=base_dtype)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(self.out_features, self.rank, device=base_device, dtype=base_dtype)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32, device=base_device))
        self.register_buffer(
            "occupied_basis",
            torch.empty(self.out_features, 0, device=base_device, dtype=base_dtype),
            persistent=True,
        )
        self.register_buffer(
            "projection_strength",
            torch.tensor(float(projection_strength), dtype=torch.float32, device=base_device),
            persistent=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        if not self.adapter_enabled or self.rank == 0:
            return base

        delta = self.get_projected_delta(x)
        if self.gate_enabled:
            delta = delta * self.gate_value().to(delta.dtype)
        return base + delta

    def gate_value(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def get_delta(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            return torch.zeros(
                *x.shape[:-1],
                self.out_features,
                device=x.device,
                dtype=self.base_layer.weight.dtype,
            )
        delta = F.linear(self.dropout(x), self.lora_A)
        delta = F.linear(delta, self.lora_B) * self.scaling
        return delta

    def get_projected_delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.project_delta(self.get_delta(x))

    def get_delta_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            return torch.zeros(x.shape[:-1], device=x.device, dtype=self.base_layer.weight.dtype)
        return self.get_delta(x).norm(dim=-1)

    def get_projected_delta_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            return torch.zeros(x.shape[:-1], device=x.device, dtype=self.base_layer.weight.dtype)
        return self.get_projected_delta(x).norm(dim=-1)

    def set_projection_basis(self, basis: torch.Tensor | None) -> None:
        if basis is None or basis.numel() == 0:
            self.occupied_basis = torch.empty(
                self.out_features,
                0,
                device=self.base_layer.weight.device,
                dtype=self.base_layer.weight.dtype,
            )
            return

        basis = torch.as_tensor(basis, dtype=self.base_layer.weight.dtype, device=self.base_layer.weight.device)
        if basis.ndim != 2:
            raise ValueError("projection basis must be 2D")
        if basis.shape[0] != self.out_features and basis.shape[1] == self.out_features:
            basis = basis.transpose(0, 1)
        if basis.shape[0] != self.out_features:
            raise ValueError(
                f"projection basis must have out_features={self.out_features} rows, got {tuple(basis.shape)}"
            )

        q, _ = torch.linalg.qr(basis, mode="reduced")
        self.occupied_basis = q

    def set_projection_strength(self, value: float) -> None:
        with torch.no_grad():
            self.projection_strength.fill_(float(value))

    def set_gate_logit(self, value: float) -> None:
        with torch.no_grad():
            self.gate_logit.fill_(float(value))

    def freeze_adapter(self) -> None:
        if self.lora_A is not None:
            self.lora_A.requires_grad = False
        if self.lora_B is not None:
            self.lora_B.requires_grad = False
        self.gate_logit.requires_grad = False

    def unfreeze_adapter(self) -> None:
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True
        self.gate_logit.requires_grad = True

    def disable_and_passthrough(self) -> None:
        self.adapter_enabled = False

    def enable(self) -> None:
        self.adapter_enabled = True

    def project_delta(self, delta: torch.Tensor) -> torch.Tensor:
        if self.occupied_basis.numel() == 0:
            return delta
        strength = float(self.projection_strength.item())
        if strength <= 0.0:
            return delta

        basis = self.occupied_basis.to(device=delta.device, dtype=delta.dtype)
        occupied = torch.matmul(torch.matmul(delta, basis), basis.transpose(0, 1))
        return delta - occupied * delta.new_tensor(strength)

    def merge_into_base(self) -> None:
        if self.rank == 0:
            return
        with torch.no_grad():
            gate = self.gate_value().item() if self.gate_enabled else 1.0
            merged = (self.lora_B @ self.lora_A) * self.scaling * gate
            if self.occupied_basis.numel() > 0:
                basis = self.occupied_basis.to(dtype=merged.dtype, device=merged.device)
                strength = float(self.projection_strength.item())
                if strength > 0.0:
                    occupied = basis @ (basis.transpose(0, 1) @ merged)
                    merged = merged - occupied * strength
            self.base_layer.weight.add_(merged.to(dtype=self.base_layer.weight.dtype, device=self.base_layer.weight.device))

        if self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
        if self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.set_gate_logit(-6.0)


def iter_named_linears(model: nn.Module) -> Iterable[Tuple[str, nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, LatentLoRALinear):
            yield name, module


def resolve_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parent = root
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def module_name_matches(name: str, suffixes: Sequence[str], layer_indices: set[int] | None = None) -> bool:
    if not any(name == suffix or name.endswith(f".{suffix}") for suffix in suffixes):
        return False
    if layer_indices is None:
        return True
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1]) in layer_indices
            except ValueError:
                return False
    return False


def attach_latent_lora(
    model: nn.Module,
    *,
    suffixes: Sequence[str],
    layer_indices: set[int] | None,
    config: LatentLoRAConfig,
) -> List[Tuple[str, LatentLoRALinear]]:
    attached: List[Tuple[str, LatentLoRALinear]] = []
    for name, module in list(iter_named_linears(model)):
        if not module_name_matches(name, suffixes, layer_indices):
            continue
        wrapped = LatentLoRALinear(
            module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            projection_strength=config.projection_strength,
            gate_init=config.gate_init,
            freeze_base=config.freeze_base,
        )
        wrapped = wrapped.to(device=module.weight.device)
        parent, child_name = resolve_parent_module(model, name)
        setattr(parent, child_name, wrapped)
        attached.append((name, wrapped))
    return attached


def detach_latent_lora(
    model: nn.Module,
    attached: List[Tuple[str, LatentLoRALinear]],
    merge: bool = False,
) -> None:
    for name, wrapper in attached:
        if merge:
            wrapper.merge_into_base()
        parent, child_name = resolve_parent_module(model, name)
        setattr(parent, child_name, wrapper.base_layer)


def parse_layer_indices(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    values = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.add(int(token))
    return values or None


def count_trainable_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def load_basis_map(path: str | None) -> Dict[str, torch.Tensor]:
    if not path:
        return {}
    basis_path = Path(path)
    if not basis_path.exists():
        raise FileNotFoundError(f"basis file not found: {basis_path}")

    if basis_path.suffix == ".pt":
        payload = torch.load(basis_path, map_location="cpu")
    elif basis_path.suffix == ".pth":
        payload = torch.load(basis_path, map_location="cpu")
    else:
        raise ValueError("basis file must be a .pt or .pth file containing {module_name: basis_tensor}")

    if not isinstance(payload, dict):
        raise TypeError("basis file must contain a dict mapping module names to tensors")
    return payload


def apply_basis_map(attached: List[Tuple[str, LatentLoRALinear]], basis_map: Dict[str, torch.Tensor]) -> int:
    applied = 0
    for name, module in attached:
        basis = basis_map.get(name)
        if basis is None:
            continue
        module.set_projection_basis(basis)
        applied += 1
    return applied


def choose_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if key not in table:
        raise ValueError(f"unsupported dtype: {name}")
    return table[key]


def default_model_id(local_files_only: bool) -> str:
    if local_files_only:
        return "Qwen/Qwen2.5-0.5B-Instruct"
    return "Qwen/Qwen2.5-0.5B"


def load_tokenizer(
    model_id: str,
    *,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
):
    _require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_id: str,
    *,
    device: str,
    dtype: torch.dtype,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
):
    _require_transformers()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    return model.to(device)


def smoke_forward(model, tokenizer, prompt: str, device: str) -> None:
    batch = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**batch)
    print(f"smoke logits shape: {tuple(outputs.logits.shape)}")
    top_ids = outputs.logits[0, -1].topk(k=5).indices.tolist()
    print("top next tokens:", tokenizer.convert_ids_to_tokens(top_ids))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Qwen LatentLoRA prototype")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--suffix", action="append", default=["mlp.down_proj"], help="Module suffix to wrap")
    parser.add_argument("--layers", default=None, help="Comma-separated decoder layer indices to wrap")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument("--gate-init", type=float, default=-6.0)
    parser.add_argument("--basis-file", default=None, help="Optional .pt/.pth dict {module_name: basis_tensor}")
    parser.add_argument("--prompt", default="The answer is")
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--local-files-only", action="store_true", help="Only load from local Hugging Face cache")
    parser.add_argument("--no-freeze-base", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a single forward pass after attachment")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _require_transformers()

    dtype = choose_dtype(args.dtype)
    model_id = args.model_id or default_model_id(args.local_files_only)
    layer_indices = parse_layer_indices(args.layers)
    config = LatentLoRAConfig(
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        projection_strength=args.projection_strength,
        gate_init=args.gate_init,
        freeze_base=not args.no_freeze_base,
    )

    print(f"loading tokenizer: {model_id}")
    tokenizer = load_tokenizer(
        model_id,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    print(f"loading model: {model_id}")
    model = load_causal_lm(
        model_id,
        device=args.device,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    attached = attach_latent_lora(
        model,
        suffixes=args.suffix,
        layer_indices=layer_indices,
        config=config,
    )
    if not attached:
        raise RuntimeError("no matching linear modules were wrapped")

    basis_map = load_basis_map(args.basis_file)
    applied = apply_basis_map(attached, basis_map)

    print(f"attached latent LoRA modules: {len(attached)}")
    for name, module in attached:
        basis_rank = int(module.occupied_basis.shape[1])
        print(
            f"  {name}: rank={module.rank} gate={module.gate_value().item():.4f} "
            f"proj_strength={module.projection_strength.item():.2f} basis_rank={basis_rank}"
        )
    if basis_map:
        print(f"applied projection bases: {applied}/{len(attached)}")

    print(f"trainable params: {count_trainable_params(model):,}")

    if args.smoke:
        smoke_forward(model, tokenizer, args.prompt, args.device)


if __name__ == "__main__":
    main()
