"""Deprecated compatibility shim. Use `chaos` instead."""
import importlib
import sys
import warnings

warnings.warn(
    "equilibrium_rl is deprecated; use chaos instead",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export chaos symbols
from chaos import *  # noqa: F401,F403

# Alias submodules for backwards compatibility
for _mod in [
    "optimizers",
    "impedance",
    "thermostat",
    "structural",
    "batch_thermostat",
    "presets",
    "actuators",
    "controller",
    "lora",
]:
    sys.modules[f"equilibrium_rl.{_mod}"] = importlib.import_module(f"chaos.{_mod}")
