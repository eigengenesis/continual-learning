import torch
import torch.nn as nn
import pytest

transformers = pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2Model

from chaos.structural import inject_thermodynamics, TemperatureControlledNorm

def test_injection():
    print("🧪 TESTING: Structural Injection on GPT-2")

    # 1. Instantiate a standard HF Model
    config = GPT2Config(n_layer=2, n_embd=128, n_head=4)
    model = GPT2Model(config)

    print("\n[Before Injection]")
    print(model.h[0].ln_1) # Should be LayerNorm

    # 2. Inject Thermodynamics
    print("\n💉 Injecting...")
    thermostat = inject_thermodynamics(model, target_types=(nn.LayerNorm,))

    print("\n[After Injection]")
    print(model.h[0].ln_1) # Should be TemperatureControlledNorm

    # 3. Functional Test
    inputs = torch.randn(1, 10, 128)

    # Force Phase: ICE
    print("\n❄️ Setting Temp=0.5 (Ice)...")
    thermostat.step(z_current=0.1) # Gas -> Lowers T
    out_ice = model(inputs_embeds=inputs).last_hidden_state

    # Force Phase: GAS
    print("\n🔥 Setting Temp=2.0 (Gas)...")
    thermostat.step(z_current=5.0) # Ice -> Raises T
    out_gas = model(inputs_embeds=inputs).last_hidden_state

    # Check Difference
    diff = (out_ice - out_gas).abs().mean().item()
    print(f"\nDifference between Phases: {diff:.6f}")

    if isinstance(model.h[0].ln_1, TemperatureControlledNorm):
        print("✅ SUCCESS: Layer replacement verified.")
    else:
        print("❌ FAIL: Layer replacement failed.")

if __name__ == "__main__":
    test_injection()
