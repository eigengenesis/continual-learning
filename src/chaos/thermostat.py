
import torch
import numpy as np
from typing import Dict, Optional, Tuple, List, Any

class Z0Thermostat:
    """
    Thermodynamic Hyperparameter Controller.
    Adjusts Learning Rate based on Thermodynamic Impedance (Z).

    States:
    - LIQUID (Z ~ target_z): Target regime. PID control maintains this.
    - ICE (Z >> target_z): Frozen weights. Signals "Thaw" (LR Boost).
    - GAS (Z << target_z): Chaotic gradients. Signals "Cool" (LR Decay).
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 target_z: float = 1.0,
                 lr_bounds: Tuple[float, float] = (1e-6, 1e-3),
                 thresholds: Optional[Dict[str, float]] = None,
                 use_autotuner: bool = True,
                 auto_target: bool = False,
                 calibration_steps: int = 50,
                 apply_lr: bool = True):

        self.optimizer = optimizer
        self.target_z = target_z
        self.lr_bounds = lr_bounds
        self.apply_lr = apply_lr
        self.last_multiplier = 1.0

        # Auto-Targeting State
        self.auto_target = auto_target
        self.calibration_steps = calibration_steps
        self.calibration_buffer = []
        self._is_calibrated = not auto_target
        self.calibration_complete_step = None

        # Default Thresholds (Scaled relative to Target Z)
        # Ratio: Ice ~ 3x Target, Gas ~ 0.5x Target
        if thresholds is None:
            self.thresholds = self._thresholds_for_target(target_z)
        else:
            self.thresholds = thresholds

        # State Tracking
        self.z_history = []
        self.window = 20
        self.z_ema = None

        # Cooldowns
        self.last_adjustment_step = 0
        self.cooldown = 30

        # PID Controller
        self.kp = 0.02
        self.ki = 0.001
        self.kd = 0.005
        self.pid_integral = 0.0
        self.pid_max_integral = 10.0
        self.last_error = 0.0

        # PID Autotuner
        self.use_autotuner = use_autotuner
        self.gains_bounds = {
            "kp": (0.005, 0.05),
            "ki": (0.0001, 0.01),
            "kd": (0.001, 0.02)
        }
        self.error_history = []
        self.adaptation_cooldown = 0

        # Performance/Economic Guardrail State
        self.score_ema = None
        self.last_score_ema = None

    def step(self, current_z: float, step_idx: int, entropy: Optional[float] = None, score: Optional[float] = None) -> Dict[str, Any]:
        """
        Advances the thermostat state machine one step.
        """
        if not np.isfinite(current_z):
            return self._finalize(
                {
                "state": "invalid_z",
                "action": "skip_nonfinite_z",
                "z": float("nan"),
                "lr_multiplier": 1.0,
                "calibration_step": self.calibration_complete_step,
                },
                float("nan"),
            )

        self.last_multiplier = 1.0
        z = self._update_stats(current_z)
        z_momentum = self._get_momentum()

        # Performance/Economic Guardrail
        is_improving = False
        if score is not None:
            is_improving = self._check_improvement(score)

        adjustments = {"state": "liquid", "action": "none", "z": z}

        # === CALIBRATION PHASE ===
        if not self._is_calibrated:
            self.calibration_buffer.append(current_z)
            adjustments["state"] = "calibrating"

            if len(self.calibration_buffer) >= self.calibration_steps:
                # Set Target Z to the median of observed values (robust to outliers)
                avg_z = np.median(self.calibration_buffer)
                self.target_z = avg_z

                # Update Thresholds based on new Target
                self.thresholds = self._thresholds_for_target(self.target_z)

                self._is_calibrated = True
                if self.calibration_complete_step is None:
                    self.calibration_complete_step = step_idx
                adjustments["action"] = "calibration_complete"
                print(f"⚖️ THERMOSTAT CALIBRATED: Target Z = {self.target_z:.2f}")

            return self._finalize(adjustments, z)

        # Cooldown Check (PID executes, but major state changes wait)
        if step_idx - self.last_adjustment_step < self.cooldown:
            self._apply_pid(z)
            adjustments["action"] = "pid_maintain"
            return self._finalize(adjustments, z)

        # === STATE MACHINE ===

        # 1. ICE (Frozen)
        if z > self.thresholds["ice"]:
            if is_improving:
                adjustments.update({"state": "ice", "action": "veto_thaw"})
                # Do NOT adjust LR
            else:
                self._adjust_lr(1.15) # Boost 15%
                self.pid_integral = 0.0
                self.last_adjustment_step = step_idx
                adjustments.update({"state": "ice", "action": "thaw"})

        # 2. PRE-ICE (Warning)
        elif z > self.thresholds["pre_ice"] and z_momentum > 0.1:
            if is_improving:
                adjustments.update({"state": "pre_ice", "action": "veto_pre_thaw"})
            else:
                self._adjust_lr(1.08) # Nudge 8%
                adjustments.update({"state": "pre_ice", "action": "pre_thaw"})

        # 3. GAS (Chaotic or Supercooled)
        elif z < self.thresholds["gas"]:
            # SUPERCOOLED CHECK: High Entropy + Low Z = Stuck in Chaos (Needs Heat)
            if entropy is not None and entropy > 0.5:
                 if is_improving:
                     adjustments.update({"state": "supercooled", "action": "veto_thaw"})
                 else:
                     self._adjust_lr(1.1)
                     self.pid_integral = 0.0
                     self.last_adjustment_step = step_idx
                     adjustments.update({"state": "supercooled", "action": "thaw"})
            else:
                # Standard Plasma/Gas -> Cool down
                # Note: We usually allow cooling even if improving to prevent explosion,
                # but if it's improving rapidly, maybe we maintain?
                # For now, SAFETY FIRST: Always allow cooling if chaotic.
                self._adjust_lr(0.9) # Cool 10%
                self.pid_integral = 0.0
                self.last_adjustment_step = step_idx
                adjustments.update({"state": "gas", "action": "cool"})

        # 4. LIQUID (Optimal)
        else:
            self._apply_pid(z)
            adjustments["state"] = "liquid"

        return self._finalize(adjustments, z)

    def _check_improvement(self, score: float) -> bool:
        """
        Returns True if the score is improving relative to EMA.
        Using EMA to smooth out noisy rewards.
        """
        decay = 0.1
        if self.score_ema is None:
            self.score_ema = score
        else:
            self.score_ema = (1 - decay) * self.score_ema + decay * score

        is_improving = False
        if self.last_score_ema is not None:
             # If slope is positive
             if self.score_ema > self.last_score_ema + 1e-6: # Slight epsilon
                 is_improving = True

        self.last_score_ema = self.score_ema
        return is_improving

    def _thresholds_for_target(self, target_z: float) -> Dict[str, float]:
        return {
            "ice": target_z * 3.0,
            "pre_ice": target_z * 2.4,
            "gas": target_z * 0.5,
        }

    def _finalize(self, adjustments: Dict[str, Any], z: float) -> Dict[str, Any]:
        target = float(self.target_z)
        adjustments["z"] = z
        adjustments["target_z"] = target
        adjustments["z_ratio"] = z / max(abs(target), 1e-9) if np.isfinite(z) else float("nan")
        adjustments["lr_multiplier"] = self.last_multiplier
        adjustments["calibration_step"] = self.calibration_complete_step
        return adjustments

    def _update_stats(self, z: float) -> float:
        self.z_history.append(z)
        if len(self.z_history) > self.window: self.z_history.pop(0)

        if self.z_ema is None:
            self.z_ema = z
        else:
            self.z_ema = 0.15 * z + 0.85 * self.z_ema
        return self.z_ema

    def _get_momentum(self) -> float:
        if len(self.z_history) < 10: return 0.0
        return np.mean(self.z_history[-5:]) - np.mean(self.z_history[-10:-5])

    def _apply_pid(self, z: float):
        error = z - self.target_z
        self.error_history.append(error)
        if len(self.error_history) > 50: self.error_history.pop(0)

        # Autotune
        if self.use_autotuner:
            self._autotune()

        # PID Calc
        self.pid_integral = max(-self.pid_max_integral, min(self.pid_max_integral, self.pid_integral + error))
        derivative = error - self.last_error
        self.last_error = error

        control = (self.kp * error) + (self.ki * self.pid_integral) + (self.kd * derivative)

        # Apply small adjustment (Liquid State logic)
        multiplier = max(0.9, min(1.1, 1.0 + control))
        self._adjust_lr(multiplier)

    def _adjust_lr(self, multiplier: float):
        self.last_multiplier = multiplier
        if not self.apply_lr:
            return
        for pg in self.optimizer.param_groups:
            new_lr = pg['lr'] * multiplier
            new_lr = max(min(new_lr, self.lr_bounds[1]), self.lr_bounds[0])
            pg['lr'] = new_lr

    def _autotune(self):
        """Level 2 Supervisor: Adjusts PID gains based on error dynamics."""
        if self.adaptation_cooldown > 0:
            self.adaptation_cooldown -= 1
            return

        if len(self.error_history) < 20: return

        self.adaptation_cooldown = 50
        recent = self.error_history[-20:]

        # Logic: Detect Oscillation vs Stagnation
        sign_flips = sum(1 for i in range(1, len(recent)) if recent[i] * recent[i-1] < 0)
        avg_error = np.mean(np.abs(recent))

        if sign_flips > 8: # Oscillating
            self.kd *= 1.15
            self.kp *= 0.9
        elif avg_error > 0.5 and sign_flips < 3: # Stuck
            self.ki *= 1.05

        # Clamp gains to safe bounds
        self.kp = max(self.gains_bounds["kp"][0], min(self.kp, self.gains_bounds["kp"][1]))
        self.ki = max(self.gains_bounds["ki"][0], min(self.ki, self.gains_bounds["ki"][1]))
        self.kd = max(self.gains_bounds["kd"][0], min(self.kd, self.gains_bounds["kd"][1]))
