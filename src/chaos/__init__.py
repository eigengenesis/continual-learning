from importlib.metadata import version, PackageNotFoundError

from .optimizers import Z0Optimizer, EquilibriumOptimizer
from .impedance import Z0Monitor
from .thermostat import Z0Thermostat
from .controller import Z0Controller
from .actuators import (
    Actuator,
    LearningRateActuator,
    LoRAAlphaActuator,
    BatchActuator,
    WeightDecayActuator,
    EntropyCoefActuator,
    DPOBetaActuator,
    DropoutActuator,
    LogitTemperatureActuator,
    find_logit_head,
)
from .lora import find_lora_layers, group_lora_layers
from .structural import (
    TemperatureControlledNorm,
    StructuralThermostat,
    inject_thermodynamics,
)
from .batch_thermostat import BatchThermostat
from .presets import pre, sft, dpo, rl

try:
    __version__ = version("chaos")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Z0Optimizer",
    "EquilibriumOptimizer",
    "Z0Monitor",
    "Z0Thermostat",
    "Z0Controller",
    "TemperatureControlledNorm",
    "StructuralThermostat",
    "inject_thermodynamics",
    "BatchThermostat",
    "Actuator",
    "LearningRateActuator",
    "LoRAAlphaActuator",
    "BatchActuator",
    "WeightDecayActuator",
    "EntropyCoefActuator",
    "DPOBetaActuator",
    "DropoutActuator",
    "LogitTemperatureActuator",
    "find_logit_head",
    "find_lora_layers",
    "group_lora_layers",
    "sft",
    "pre",
    "dpo",
    "rl",
    "__version__",
]
