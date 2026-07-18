import torch
from typing import Optional, Dict, Any, Tuple, List
from .impedance import Z0Monitor
from .thermostat import Z0Thermostat
from .controller import Z0Controller
from .actuators import (
    Actuator,
    LearningRateActuator,
    LoRAAlphaActuator,
    BatchActuator,
    WeightDecayActuator,
    DropoutActuator,
    LogitTemperatureActuator,
)

class Z0Optimizer:
    """
    Drop-in wrapper for PyTorch optimizers that adds Z0 Thermostat control.

    Usage:
        base_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt = Z0Optimizer(base_opt, target_z=1.0)

        # In loop:
        loss.backward()
        opt.step(loss.item()) # Just pass the loss!
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 target_z: float = 1.0,
                 lr_bounds: Tuple[float, float] = (1e-6, 1e-3),
                 auto_target: bool = False,
                 config: Optional[Dict[str, Any]] = None,
                 model: Optional[torch.nn.Module] = None,
                 controls: Optional[Dict[str, bool]] = None,
                 actuators: Optional[List[Actuator]] = None,
                 observe_only: bool = False):

        self.optimizer = optimizer
        self.step_count = 0
        self.observe_only = observe_only

        default_controls = {
            "lr": True,
            "lora": True,
            "logits": False,
            "batch": False,
            "weight_decay": False,
            "entropy_coef": False,
            "dpo_beta": False,
            "dropout": False,
        }
        if controls is None:
            controls = default_controls
        else:
            merged = default_controls.copy()
            merged.update(controls)
            controls = merged
        self.controls = controls

        monitor = Z0Monitor()
        self._monitor = monitor
        thermostat = Z0Thermostat(
            optimizer,
            target_z=target_z,
            lr_bounds=lr_bounds,
            thresholds=config,
            auto_target=auto_target,
            apply_lr=False,
        )

        actuator_list: List[Actuator] = list(actuators or [])
        existing = {a.name for a in actuator_list}

        if controls.get("lr", False) and "lr" not in existing:
            actuator_list.append(LearningRateActuator(optimizer, bounds=lr_bounds))
        if model is not None and controls.get("lora", False) and "lora_alpha" not in existing:
            actuator_list.append(LoRAAlphaActuator(model))
        if model is not None and controls.get("logits", False) and "logit_temperature" not in existing:
            actuator_list.append(LogitTemperatureActuator(model))
        if model is not None and controls.get("dropout", False) and "dropout" not in existing:
            actuator_list.append(DropoutActuator(model))
        if controls.get("weight_decay", False) and "weight_decay" not in existing:
            actuator_list.append(WeightDecayActuator(optimizer))
        # Entropy/DPO actuators are opt-in and should be passed explicitly with the correct object.
        # BatchActuator is opt-in and should be passed explicitly with a BatchThermostat.
        # If provided via actuators list, it will be used as-is.

        if observe_only:
            for actuator in actuator_list:
                actuator.enabled = False

        self.controller = Z0Controller(
            monitor=monitor,
            thermostat=thermostat,
            actuators=actuator_list,
            observe_only=observe_only,
        )
        self._monitor = self.controller.monitor
        self.thermostat = self.controller.thermostat

    def step(self, loss: float, entropy: Optional[float] = None, closure=None, score: Optional[float] = None):
        """
        Performs a single optimization step with Thermodynamic adjustments.

        Args:
            loss (float): The current step's loss (required for Z calculation).
            entropy (float, optional): The current policy entropy (required for Supercooled detection).
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
            score (float, optional): External performance metric (e.g. reward) for Economic Guardrail.
        """
        self.step_count += 1

        # 1. Inspect Gradients (Kinetic Energy)
        total_grad_norm = 0.0

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2

        grad_norm = total_grad_norm ** 0.5

        # 2. Controller Step (monitor + thermostat + actuators)
        stats = self.controller.step(
            loss=loss,
            step_idx=self.step_count,
            grad_norm=grad_norm,
            entropy=entropy,
            score=score,
        )

        # 4. Optimizer Step
        self.optimizer.step(closure)

        return stats

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def monitor(self) -> Z0Monitor:
        return self._monitor

    @monitor.setter
    def monitor(self, value: Z0Monitor) -> None:
        self._monitor = value
        if hasattr(self, "controller"):
            self.controller.monitor = value

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'monitor': self.monitor.history,
            'thermostat': self.thermostat.__dict__ # Hacky, but sufficient for now
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        # Simplified loading for now

    def __getattr__(self, name):
        # Forward unknown attributes to the base optimizer (e.g. param_groups)
        return getattr(self.optimizer, name)


# Back-compat alias
EquilibriumOptimizer = Z0Optimizer
