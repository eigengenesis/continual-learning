import torch
import numpy as np
from typing import Optional, Dict

class Z0Monitor:
    """
    The 'Sensors' of the Z0 thermostat.
    Measures Thermodynamic Impedance (Z) = Loss / GradNorm.

    Z is the ratio of Potential Energy (Loss) to Kinetic Energy (Gradient).
    Scale differences across architectures and tasks are handled by the
    calibration step, which estimates a reference baseline during the first
    few steps and computes a relative scale factor for downstream consumers.
    """
    def __init__(self, window: int = 20, smoothing: float = 0.15):
        self.window = window
        self.smoothing = smoothing

        self.history = []
        self.scale_factor = None
        self.reference_z = None
        self._calibration_buffer = []
        self._calibration_steps = 10
        self.z_ema = None
        self.last_raw_z = 1.0
        self.last_relative_z = None

    def measure(self, loss: float, model: Optional[torch.nn.Module] = None, grad_norm: Optional[float] = None, **kwargs) -> float:
        """
        Calculates Z-value for the current step.
        Z = |Loss| / ||GradNorm||

        Ideally called *after* backward() and *before* optimizer.step().

        The ``weight_norm`` keyword is accepted for API compatibility but
        is no longer used in the computation.
        """
        # 1. Compute Gradient Norm (Kinetic Energy) if not provided
        if grad_norm is None:
            if model is None:
                grad_norm = 0.0
            else:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm = total_norm ** 0.5
        if not np.isfinite(grad_norm):
            grad_norm = 0.0

        # 2. Compute Raw Impedance Z = Potential / Kinetic
        loss_value = float(loss)
        if not np.isfinite(loss_value):
            fallback = self.get_current_z()
            return float(fallback if np.isfinite(fallback) else 1.0)

        if grad_norm < 1e-9:
            # Zero gradients = Infinite Impedance (Frozen)
            # We cap at a high value to represent "Ice"
            raw_z = 10.0
        else:
            raw_z = abs(loss_value) / (grad_norm + 1e-9)
            if not np.isfinite(raw_z):
                fallback = self.get_current_z()
                raw_z = float(fallback if np.isfinite(fallback) else 1.0)

        # 3. Passive Reference Calibration
        # We estimate a reference scale during the first few steps so
        # downstream controllers can work with task-relative Z values.
        self.last_raw_z = raw_z
        self._maybe_calibrate_reference(raw_z)
        if self.scale_factor is not None:
            self.last_relative_z = min(raw_z * self.scale_factor, 20.0)
        else:
            self.last_relative_z = None

        self._add_to_history(raw_z)
        return raw_z

    def _maybe_calibrate_reference(self, raw_z: float) -> None:
        if self.scale_factor is not None or not np.isfinite(raw_z):
            return
        self._calibration_buffer.append(raw_z)
        if len(self._calibration_buffer) < self._calibration_steps:
            return

        avg_raw = float(np.mean(self._calibration_buffer))
        if not np.isfinite(avg_raw) or avg_raw <= 1e-9:
            self.reference_z = 1.0
            self.scale_factor = 1.0
        else:
            self.reference_z = avg_raw
            self.scale_factor = 1.0 / (avg_raw + 1e-9)
        print(
            f"⚖️ Z0 reference estimated: baseline={self.reference_z:.4f}, "
            f"relative_scale={self.scale_factor:.2f}"
        )

    def _add_to_history(self, z: float):
        self.history.append(z)
        if len(self.history) > self.window:
            self.history.pop(0)

        # EMA Smoothing
        if self.z_ema is None:
            self.z_ema = z
        else:
            self.z_ema = self.smoothing * z + (1 - self.smoothing) * self.z_ema

    def get_avg_z(self) -> float:
        if not self.history: return 1.0
        return np.mean(self.history)

    def get_current_z(self) -> float:
        return self.z_ema if self.z_ema is not None else 1.0

    def get_relative_z(self) -> float:
        if self.last_relative_z is not None:
            return self.last_relative_z
        return 1.0
