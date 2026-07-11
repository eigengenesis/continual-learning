from typing import Dict, List, Tuple

import torch.nn as nn


try:  # Optional peft dependency
    from peft.tuners.lora import LoraLayer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LoraLayer = None


def is_lora_layer(module: nn.Module) -> bool:
    if LoraLayer is not None and isinstance(module, LoraLayer):
        return True
    return hasattr(module, "lora_A") and hasattr(module, "lora_B")


def get_lora_scale(module: nn.Module):
    if hasattr(module, "scaling"):
        scaling = getattr(module, "scaling")
        if isinstance(scaling, dict):
            if "default" in scaling:
                return scaling["default"]
            for value in scaling.values():
                return value
            return None
        return scaling
    if hasattr(module, "lora_alpha") and hasattr(module, "r"):
        r = getattr(module, "r") or 1
        return getattr(module, "lora_alpha") / float(r)
    return None


def set_lora_scale(module: nn.Module, value: float) -> bool:
    if hasattr(module, "scaling"):
        scaling = getattr(module, "scaling")
        if isinstance(scaling, dict):
            for key in list(scaling.keys()):
                scaling[key] = value
            setattr(module, "scaling", scaling)
        else:
            setattr(module, "scaling", value)
        return True
    if hasattr(module, "lora_alpha") and hasattr(module, "r"):
        r = getattr(module, "r") or 1
        setattr(module, "lora_alpha", float(value) * float(r))
        return True
    return False


def find_lora_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    layers: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if is_lora_layer(module):
            layers.append((name, module))
    return layers


def group_lora_layers(lora_layers: List[Tuple[str, nn.Module]], scheme: str = "attn_mlp") -> Dict[str, List[Tuple[str, nn.Module]]]:
    groups: Dict[str, List[Tuple[str, nn.Module]]] = {"attn": [], "mlp": [], "other": []}
    if scheme != "attn_mlp":
        groups["other"] = list(lora_layers)
        return groups

    for name, module in lora_layers:
        lname = name.lower()
        if any(tok in lname for tok in ("attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj")):
            groups["attn"].append((name, module))
        elif any(tok in lname for tok in ("mlp", "ffn", "fc")):
            groups["mlp"].append((name, module))
        else:
            groups["other"].append((name, module))

    return groups
