from typing import Any, Callable, Dict, List, Optional, Tuple

import torch.nn as nn

from .batch_thermostat import BatchThermostat
from .lora import find_lora_layers, group_lora_layers, get_lora_scale, set_lora_scale


def find_logit_head(model: nn.Module) -> Optional[nn.Module]:
    candidates = [
        "lm_head",
        "classifier",
        "score",
        "action_net",
        "policy",
        "logits",
    ]
    for name, module in model.named_modules():
        if any(name.endswith(c) for c in candidates):
            return module
    return None


class Actuator:
    name = "actuator"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def update(self, stats: Dict[str, Any], step_idx: int) -> None:
        pass


class LearningRateActuator(Actuator):
    name = "lr"

    def __init__(self, optimizer, bounds: Tuple[float, float], enabled: bool = True):
        super().__init__(enabled=enabled)
        self.optimizer = optimizer
        self.bounds = bounds

    def update(self, stats: Dict[str, Any], step_idx: int) -> None:
        if not self.enabled:
            return
        multiplier = stats.get("lr_multiplier")
        if multiplier is None:
            return
        for pg in self.optimizer.param_groups:
            new_lr = pg["lr"] * float(multiplier)
            new_lr = max(min(new_lr, self.bounds[1]), self.bounds[0])
            pg["lr"] = new_lr


class LoRAAlphaActuator(Actuator):
    name = "lora_alpha"

    def __init__(
        self,
        model: nn.Module,
        bounds: Tuple[float, float] = (0.5, 2.0),
        update_every: int = 10,
        group_scheme: str = "attn_mlp",
        enabled: bool = True,
        up_multiplier: float = 1.08,
        down_multiplier: float = 0.92,
        drift: float = 0.02,
    ):
        super().__init__(enabled=enabled)
        self.model = model
        self.bounds = bounds
        self.update_every = update_every
        self.group_scheme = group_scheme
        self.up_multiplier = up_multiplier
        self.down_multiplier = down_multiplier
        self.drift = drift

        self._initialized = False
        self._group_scales: Dict[str, float] = {}
        self._group_layers: Dict[str, List[Tuple[str, nn.Module]]] = {}
        self._base_scales: Dict[int, float] = {}

    def _initialize(self) -> None:
        lora_layers = find_lora_layers(self.model)
        if not lora_layers:
            self.enabled = False
            return
        self._group_layers = group_lora_layers(lora_layers, scheme=self.group_scheme)
        self._group_scales = {k: 1.0 for k in self._group_layers.keys()}
        for _, module in lora_layers:
            base = get_lora_scale(module)
            if base is not None:
                self._base_scales[id(module)] = float(base)
        self._initialized = True

    def _apply_scales(self) -> None:
        for group, layers in self._group_layers.items():
            scale = self._group_scales.get(group, 1.0)
            scale = max(self.bounds[0], min(scale, self.bounds[1]))
            self._group_scales[group] = scale
            for _, module in layers:
                base = self._base_scales.get(id(module))
                if base is None:
                    base = get_lora_scale(module) or 1.0
                    self._base_scales[id(module)] = float(base)
                set_lora_scale(module, base * scale)

    def update(self, stats: Dict[str, Any], step_idx: int) -> None:
        if not self.enabled:
            return
        if not self._initialized:
            self._initialize()
            if not self.enabled:
                return
        if self.update_every > 1 and step_idx % self.update_every != 0:
            return

        action = stats.get("action", "none")
        if action in ("thaw", "pre_thaw", "supercooled"):
            multiplier = self.up_multiplier
        elif action in ("cool",):
            multiplier = self.down_multiplier
        else:
            multiplier = 1.0

        for group in self._group_scales:
            if multiplier != 1.0:
                self._group_scales[group] *= multiplier
            else:
                self._group_scales[group] = (1.0 - self.drift) * self._group_scales[group] + self.drift * 1.0

        self._apply_scales()


class BatchActuator(Actuator):
    name = "batch"

    def __init__(
        self,
        batch_thermostat: BatchThermostat,
        on_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.batch_thermostat = batch_thermostat
        self.on_update = on_update
        self.last: Optional[Dict[str, Any]] = None

    def update(self, stats: Dict[str, Any], step_idx: int) -> None:
        if not self.enabled:
            return
        z = stats.get("z")
        if z is None:
            return
        self.last = self.batch_thermostat.update(z)
        if self.on_update is not None:
            self.on_update(self.last)


class WeightDecayActuator(Actuator):
    name = "weight_decay"

    def __init__(self, optimizer, bounds: Tuple[float, float] = (1e-6, 1e-2), enabled: bool = True):
        super().__init__(enabled=enabled)
        self.optimizer = optimizer
        self.bounds = bounds

    def update(self, stats: Dict[str, Any], step_idx: int) -> None:
        if not self.enabled:
            return
        multiplier = stats.get("lr_multiplier")
        if multiplier is None:
            return
        for pg in self.optimizer.param_groups:
            if "weight_decay" not in pg:
                continue
            new_wd = pg["weight_decay"] * float(multiplier)
            new_wd = max(min(new_wd, self.bounds[1]), self.bounds[0])
            pg["weight_decay"] = new_wd


class EntropyCoefActuator(Actuator):
    name = "entropy_coef"

    def __init__(self, model: Any, bounds: Tuple[float, float] = (0.0, 0.1), enabled: bool = True):
        super().__init__(enabled=enabled)
        self.model = model
        self.bounds = bounds

    def update(self, stats: Dict[str, Any], step_idx: int) -> None:
        if not self.enabled:
            return
        if not hasattr(self.model, "ent_coef"):
            return
        multiplier = stats.get("lr_multiplier")
        if multiplier is None:
            return
        try:
            current = float(self.model.ent_coef)
        except Exception:
            return
        new_val = current * float(multiplier)
        new_val = max(min(new_val, self.bounds[1]), self.bounds[0])
        self.model.ent_coef = new_val


class DPOBetaActuator(Actuator):
    name = "dpo_beta"

    def __init__(self, trainer: Any, bounds: Tuple[float, float] = (0.01, 1.0), enabled: bool = True):
        super().__init__(enabled=enabled)
        self.trainer = trainer
        self.bounds = bounds

    def _set_beta(self, value: float) -> bool:
        if hasattr(self.trainer, "beta"):
            self.trainer.beta = value
            return True
        if hasattr(self.trainer, "config") and hasattr(self.trainer.config, "beta"):
            self.trainer.config.beta = value
            return True
        if hasattr(self.trainer, "args") and hasattr(self.trainer.args, "beta"):
            self.trainer.args.beta = value
            return True
        return False

    def update(self, stats: Dict[str, Any], step_idx: int) -> None:
        if not self.enabled:
            return
        multiplier = stats.get("lr_multiplier")
        if multiplier is None:
            return
        current = None
        for attr in ("beta",):
            if hasattr(self.trainer, attr):
                try:
                    current = float(getattr(self.trainer, attr))
                    break
                except Exception:
                    current = None
        if current is None:
            current = 0.1
        new_val = current * float(multiplier)
        new_val = max(min(new_val, self.bounds[1]), self.bounds[0])
        self._set_beta(new_val)


class DropoutActuator(Actuator):
    name = "dropout"

    def __init__(self, model: nn.Module, bounds: Tuple[float, float] = (0.0, 0.5), enabled: bool = True):
        super().__init__(enabled=enabled)
        self.model = model
        self.bounds = bounds
        self._base: Dict[int, float] = {}
        self._initialized = False

    def _initialize(self) -> None:
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                self._base[id(module)] = float(module.p)
        self._initialized = True

    def update(self, stats: Dict[str, Any], step_idx: int) -> None:
        if not self.enabled:
            return
        if not self._initialized:
            self._initialize()
        action = stats.get("action", "none")
        if action in ("thaw", "pre_thaw", "supercooled"):
            multiplier = 1.05
        elif action in ("cool",):
            multiplier = 0.95
        else:
            multiplier = 1.0

        for module in self.model.modules():
            if not isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                continue
            base = self._base.get(id(module), float(module.p))
            new_p = base * float(multiplier)
            new_p = max(min(new_p, self.bounds[1]), self.bounds[0])
            module.p = new_p


class LogitTemperatureActuator(Actuator):
    name = "logit_temperature"

    def __init__(
        self,
        model: nn.Module,
        module_path: Optional[str] = None,
        selector: Optional[Callable[[nn.Module], Optional[nn.Module]]] = None,
        bounds: Tuple[float, float] = (0.5, 2.0),
        update_every: int = 10,
        enabled: bool = True,
        up_multiplier: float = 1.05,
        down_multiplier: float = 0.95,
        drift: float = 0.02,
    ):
        super().__init__(enabled=enabled)
        self.model = model
        self.module_path = module_path
        self.selector = selector
        self.bounds = bounds
        self.update_every = update_every
        self.up_multiplier = up_multiplier
        self.down_multiplier = down_multiplier
        self.drift = drift

        self.temperature = 1.0
        self._hook_handle = None
        self._module = None
        self._attach()

    def _resolve_path(self, path: str) -> Optional[nn.Module]:
        current: Any = self.model
        for part in path.split("."):
            if not hasattr(current, part):
                return None
            current = getattr(current, part)
        return current if isinstance(current, nn.Module) else None

    def _hook(self, module, inputs, output):
        t = max(self.bounds[0], min(self.temperature, self.bounds[1]))
        if isinstance(output, tuple):
            if not output:
                return output
            first = output[0]
            if hasattr(first, "dtype"):
                scaled = first / t
                return (scaled,) + output[1:]
            return output
        if isinstance(output, dict):
            if "logits" in output and hasattr(output["logits"], "dtype"):
                output = dict(output)
                output["logits"] = output["logits"] / t
            return output
        if hasattr(output, "dtype"):
            return output / t
        return output

    def _attach(self) -> None:
        if not self.enabled:
            return
        module = None
        if self.selector is not None:
            module = self.selector(self.model)
        elif self.module_path is not None:
            module = self._resolve_path(self.module_path)
        else:
            module = find_logit_head(self.model)

        if module is None:
            self.enabled = False
            return
        self._module = module
        self._hook_handle = module.register_forward_hook(self._hook)

    def update(self, stats: Dict[str, Any], step_idx: int) -> None:
        if not self.enabled:
            return
        if self.update_every > 1 and step_idx % self.update_every != 0:
            return
        action = stats.get("action", "none")
        if action in ("thaw", "pre_thaw", "supercooled"):
            self.temperature *= self.up_multiplier
        elif action in ("cool",):
            self.temperature *= self.down_multiplier
        else:
            self.temperature = (1.0 - self.drift) * self.temperature + self.drift * 1.0

        self.temperature = max(self.bounds[0], min(self.temperature, self.bounds[1]))

    def close(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
