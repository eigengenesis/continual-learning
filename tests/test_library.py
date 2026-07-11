import torch
import torch.nn as nn
from chaos.optimizers import Z0Optimizer, EquilibriumOptimizer

def test_integration():
    print("🧪 Testing Z0 Integration...")

    # 1. Setup Dummy Model/Problem
    model = nn.Linear(10, 1)
    data = torch.randn(5, 10)
    target = torch.randn(5, 1)

    # 2. Setup Optimizer (AdamW - Target Z=1.0)
    base_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt = Z0Optimizer(base_opt, target_z=1.0, lr_bounds=(1e-6, 1.0))
    opt.monitor._calibration_steps = 2 # Speed up for test
    opt.thermostat.cooldown = 0 # React instantly

    print(f"✅ Initialized Optimizer (Target Z={opt.thermostat.target_z})")
    print(f"   Thresholds: {opt.thermostat.thresholds}")

    # 3. Simulate Training Loop
    criterion = nn.MSELoss()

    # Phase 1: Normal Calibration
    print("\n⚖️ Phase 1: Calibration (Normal Dynamics)...")
    for step in range(5):
        opt.optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        opt.step(loss.item())

    # Phase 2: Artificial Freeze
    print("\n❄️ Phase 2: Simulating ICE State (High LR Reqd)...")
    for step in range(5):
        opt.optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Artificially scale down gradients to trigger High Z
        for p in model.parameters():
            p.grad *= 0.01

        # Step
        loss_val = loss.item()
        stats = opt.step(loss_val)

        print(f"Step {step}: Loss={loss_val:.4f} | Mean Z={stats['z']:.2f} | Action={stats['action'].upper()} | State={stats['state']}")

    # 4. Confirm Thaw Happened
    current_lr = opt.optimizer.param_groups[0]['lr']
    print(f"\n📈 Final LR: {current_lr:.6f} (Started at 0.001)")

    assert current_lr > 0.001, "Thermostat did not boost LR in ICE simulation."
    print("✅ SUCCESS: Thermostat detected Ice and boosted LR!")

def test_muon_scaling():
    print("\n🧪 Testing Muon Scaling (Target Z=0.5)...")
    model = nn.Linear(10, 1) # Local instance
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01) # Dummy base
    opt = Z0Optimizer(base_opt, target_z=0.5)

    t = opt.thermostat.thresholds
    print(f"   Target Z=0.5 -> Ice Threshold should be ~1.5. Actual: {t['ice']}")

    assert t['ice'] == 1.5, f"Scaling incorrect! Got {t['ice']}"
    print("✅ Threshold scaling working correctly.")


def test_equilibrium_alias():
    assert EquilibriumOptimizer is Z0Optimizer

if __name__ == "__main__":
    model = nn.Linear(10, 1) # Shared instance for muon test
    test_integration()
    test_muon_scaling()
