from typing import Any, Dict, Iterable, List, Optional

from .impedance import Z0Monitor
from .thermostat import Z0Thermostat
from .actuators import Actuator


class Z0Controller:
    """
    Orchestrates Z0 monitoring, thermostat policy, and actuator updates.
    """

    def __init__(
        self,
        monitor: Optional[Z0Monitor] = None,
        thermostat: Optional[Z0Thermostat] = None,
        actuators: Optional[Iterable[Actuator]] = None,
        observe_only: bool = False,
    ):
        self.monitor = monitor or Z0Monitor()
        if thermostat is None:
            raise ValueError("Z0Controller requires a thermostat instance")
        self.thermostat = thermostat
        self.actuators: List[Actuator] = list(actuators or [])
        self.observe_only = observe_only

    def step(
        self,
        loss: float,
        step_idx: int,
        grad_norm: Optional[float] = None,
        entropy: Optional[float] = None,
        score: Optional[float] = None,
    ) -> Dict[str, Any]:
        z_raw = self.monitor.measure(loss, model=None, grad_norm=grad_norm)
        stats = self.thermostat.step(z_raw, step_idx, entropy=entropy, score=score)
        stats["z_raw"] = z_raw
        stats["z_relative"] = self.monitor.get_relative_z()
        stats["z_reference"] = self.monitor.reference_z
        stats["lr_multiplier"] = getattr(self.thermostat, "last_multiplier", 1.0)

        if not self.observe_only:
            for actuator in self.actuators:
                actuator.update(stats, step_idx)

        return stats
