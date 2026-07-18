
import numpy as np
from typing import Dict, Any

class BatchThermostat:
    """
    Thermodynamic Batch Size Controller.

    Theory:
    Batch Size (B) is inversely proportional to Temperature (T).
    - Small B = High Variance = High T (Good for breaking Ice)
    - Large B = Low Variance = Low T (Good for stabilizing Gas)

    This controller outputs `accumulation_steps` to adjust the effective batch size.
    Effective Batch Size = Base Batch Size * Accumulation Steps
    """
    def __init__(self,
                 base_batch_size: int,
                 min_steps: int = 1,
                 max_steps: int = 16,
                 target_z: float = 1.0,
                 window: int = 10):

        self.base_batch_size = base_batch_size
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.target_z = target_z

        # State
        self.current_accumulation_steps = 1
        self.z_history = []
        self.window = window
        self.cooldown = 0

    def update(self, z_current: float) -> Dict[str, Any]:
        """
        Updates the internal state and returns the new accumulation steps.
        Call this every Optimizer Step (not every batch).
        """
        self.z_history.append(z_current)
        if len(self.z_history) > self.window:
            self.z_history.pop(0)

        avg_z = np.mean(self.z_history)

        if self.cooldown > 0:
            self.cooldown -= 1
            return {
                "accumulation_steps": self.current_accumulation_steps,
                "action": "cooldown",
                "avg_z": avg_z
            }

        action = "maintain"

        # LOGIC:
        # ICE (Z > 3.0) -> Need Heat -> High T -> Reduce Batch Size
        if avg_z > 3.0:
            if self.current_accumulation_steps > self.min_steps:
                self.current_accumulation_steps = max(self.min_steps, self.current_accumulation_steps // 2)
                action = "decrease_batch (heat_up)"
                self.cooldown = 5

        # GAS (Z < 0.5) -> Need Cold -> Low T -> Increase Batch Size
        elif avg_z < 0.5:
             if self.current_accumulation_steps < self.max_steps:
                self.current_accumulation_steps = min(self.max_steps, self.current_accumulation_steps * 2)
                action = "increase_batch (cool_down)"
                self.cooldown = 5

        return {
            "accumulation_steps": self.current_accumulation_steps,
            "effective_batch_size": self.base_batch_size * self.current_accumulation_steps,
            "action": action,
            "avg_z": avg_z
        }
