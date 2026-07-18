import torch
import torch.nn as nn

from chaos.impedance import Z0Monitor
from chaos.optimizers import Z0Optimizer
from chaos.thermostat import Z0Thermostat


def test_lr_adjusts_for_ice_and_gas():
    model = nn.Linear(10, 1)
    data = torch.randn(8, 10)
    target = torch.randn(8, 1)
    criterion = nn.MSELoss()

    base_opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    opt = Z0Optimizer(base_opt, target_z=1.0, lr_bounds=(1e-6, 1.0))
    opt.thermostat.cooldown = 0
    opt.monitor._calibration_steps = 1
    opt.monitor.scale_factor = 1.0

    # ICE: shrink gradients to increase Z
    base_opt.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(1e-4)
    lr_before = base_opt.param_groups[0]["lr"]
    opt.step(loss.item())
    lr_after = base_opt.param_groups[0]["lr"]
    assert lr_after >= lr_before

    # GAS: inflate gradients to decrease Z.
    # EMA smoothing can keep Z high for a step or two, so run a few steps.
    opt.thermostat.z_history = []
    opt.thermostat.z_ema = None
    lr_before = base_opt.param_groups[0]["lr"]
    for _ in range(5):
        base_opt.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(1e4)
        opt.step(loss.item(), entropy=0.0)
    lr_after = base_opt.param_groups[0]["lr"]
    assert lr_after <= lr_before


def test_auto_target_calibrates_high_impedance():
    param = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=1e-3)
    thermostat = Z0Thermostat(optimizer, auto_target=True, calibration_steps=5)

    for step in range(5):
        thermostat.step(current_z=10.0, step_idx=step)

    assert thermostat._is_calibrated is True
    assert abs(thermostat.target_z - 10.0) < 1e-3


def test_monitor_returns_raw_z_after_reference_estimation():
    monitor = Z0Monitor()
    monitor._calibration_steps = 2

    zs = [monitor.measure(loss=1.0, grad_norm=0.1) for _ in range(3)]

    assert all(abs(z - 10.0) < 1e-6 for z in zs)
    assert abs(monitor.reference_z - 10.0) < 1e-6
    assert abs(monitor.scale_factor - 0.1) < 1e-6
    assert abs(monitor.get_current_z() - 10.0) < 1e-6


def test_fixed_target_survives_monitor_reference_estimation():
    optimizer = torch.optim.SGD([nn.Parameter(torch.tensor(1.0))], lr=1e-3)
    thermostat = Z0Thermostat(optimizer, target_z=10.0, apply_lr=False)
    thermostat.cooldown = 0

    monitor = Z0Monitor()
    monitor._calibration_steps = 2

    stats = None
    for step in range(3):
        z = monitor.measure(loss=1.0, grad_norm=0.1)
        stats = thermostat.step(current_z=z, step_idx=step)

    assert stats is not None
    assert abs(stats["z"] - 10.0) < 1e-6
    assert abs(stats["target_z"] - 10.0) < 1e-6
    assert abs(stats["z_ratio"] - 1.0) < 1e-6
    assert stats["state"] == "liquid"
