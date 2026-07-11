
import torch
import torch.nn as nn
from chaos.optimizers import Z0Optimizer
from chaos.impedance import Z0Monitor


def test_z_formula_is_loss_over_gradnorm():
    """Verify Z = L / ||grad|| without weight_norm."""
    print("🧪 Testing Z = L / ||grad||...")

    monitor = Z0Monitor()
    monitor._calibration_steps = 100  # prevent calibration from firing

    # Known values
    loss = 2.0
    grad_norm = 0.5
    expected_z = 2.0 / 0.5  # = 4.0

    z = monitor.measure(loss=loss, grad_norm=grad_norm)

    print(f"   Loss = {loss}, GradNorm = {grad_norm}")
    print(f"   Expected Z = {expected_z}")
    print(f"   Actual Z   = {z}")

    assert abs(z - expected_z) < 1e-6, f"Z formula incorrect: got {z}, expected {expected_z}"
    print("✅ Z = L / ||grad|| confirmed.")


def test_weight_norm_kwarg_accepted_but_ignored():
    """Verify that passing weight_norm doesn't crash (back-compat) but doesn't affect Z."""
    print("🧪 Testing weight_norm kwarg is accepted but ignored...")

    monitor = Z0Monitor()
    monitor._calibration_steps = 100

    z_without = monitor.measure(loss=2.0, grad_norm=0.5)

    monitor2 = Z0Monitor()
    monitor2._calibration_steps = 100
    z_with = monitor2.measure(loss=2.0, grad_norm=0.5, weight_norm=999.0)

    assert abs(z_without - z_with) < 1e-6, "weight_norm should not affect Z!"
    print("✅ weight_norm accepted but ignored.")


def test_optimizer_computes_z_without_weight_norm():
    """Verify the full optimizer pipeline works without weight_norm."""
    print("🧪 Testing optimizer pipeline...")

    model = nn.Linear(10, 1)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt = Z0Optimizer(base_opt)

    data = torch.randn(1, 10)
    loss = model(data).mean()
    loss.backward()

    stats = opt.step(loss.item())
    assert "z_raw" in stats
    assert stats["z_raw"] > 0
    print(f"   Z_raw = {stats['z_raw']:.4f}")
    print("✅ Optimizer pipeline works without weight_norm.")


if __name__ == "__main__":
    test_z_formula_is_loss_over_gradnorm()
    test_weight_norm_kwarg_accepted_but_ignored()
    test_optimizer_computes_z_without_weight_norm()
