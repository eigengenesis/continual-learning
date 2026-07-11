import torch
import torch.nn as nn
from chaos.structural import TemperatureControlledNorm, StructuralThermostat
import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

def test_structural_thermodynamics(tmp_path):
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Setup Model (The "Heater")
    dim = 64
    batch_size = 16
    norm_layer = TemperatureControlledNorm(dim, initial_temp=1.0)

    # 2. Setup Controller (The "Thermostat")
    thermostat = StructuralThermostat([norm_layer], base_temp=1.0)

    # 3. Simulation Loop
    entropy_history = []
    temp_history = []
    z_history = []

    # Generate static "features" (random logits)
    torch.manual_seed(42)
    features = torch.randn(batch_size, dim)

    print("\nPhase 1: Normal Operation (Z=1.0)")
    for _ in range(5):
        # Simulate Z=1.0 (Liquid)
        z = 1.0
        t = thermostat.step(z_current=z)
        output = norm_layer(features)
        entropy = norm_layer.get_entropy(output)

        entropy_history.append(entropy.item())
        temp_history.append(t)
        z_history.append(z)
        print(f"   Z={z:.1f} | Temp={t:.2f} | Entropy={entropy.item():.4f}")

    print("\nPhase 2: ICE AGE (Z=5.0) -> Expect Melting")
    for _ in range(5):
        # Simulate Z=5.0 (Frozen)
        z = 5.0
        t = thermostat.step(z_current=z) # Should boost T
        output = norm_layer(features)
        entropy = norm_layer.get_entropy(output)

        entropy_history.append(entropy.item())
        temp_history.append(t)
        z_history.append(z)
        print(f"   Z={z:.1f} | Temp={t:.2f} | Entropy={entropy.item():.4f}")

    print("\nPhase 3: PLASMA (Z=0.1) -> Expect Freezing")
    for _ in range(5):
        # Simulate Z=0.1 (Chaotic)
        z = 0.1
        t = thermostat.step(z_current=z) # Should lower T
        output = norm_layer(features)
        entropy = norm_layer.get_entropy(output)

        entropy_history.append(entropy.item())
        temp_history.append(t)
        z_history.append(z)
        print(f"   Z={z:.1f} | Temp={t:.2f} | Entropy={entropy.item():.4f}")

    # Validation
    final_entropy = entropy_history[-1]
    peak_entropy = max(entropy_history)

    if entropy_history[7] > entropy_history[2]: # Ice phase should have higher entropy
        print("\n✅ SUCCESS: Ice Phase melted (Entropy Increased as Expected)")
    else:
        print("\n❌ FAIL: Ice Phase did not heat up.")

    if entropy_history[-1] < entropy_history[2]: # Gas phase should have lower entropy
        print("✅ SUCCESS: Gas Phase froze (Entropy Decreased as Expected)")
    else:
        print("❌ FAIL: Gas Phase did not cool down.")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(z_history, label="Impedance (Z)", linestyle='--')
    plt.plot(temp_history, label="Temperature (T)")
    plt.plot(entropy_history, label="System Entropy")
    plt.title("Thermodynamic Phase Transitions")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = tmp_path / "structural_test_results.png"
    plt.savefig(out_path)
    print(f"\n📊 Graph saved to '{out_path}'")

if __name__ == "__main__":
    test_structural_thermodynamics()
