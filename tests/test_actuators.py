import torch
import torch.nn as nn

from chaos import Z0Optimizer
from chaos.lora import find_lora_layers, group_lora_layers
from chaos.actuators import LoRAAlphaActuator, LogitTemperatureActuator


class DummyLoRALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Linear(4, 4, bias=False)
        self.lora_B = nn.Linear(4, 4, bias=False)
        self.scaling = 1.0


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = DummyLoRALayer()
        self.mlp = DummyLoRALayer()
        self.other = DummyLoRALayer()

    def forward(self, x):
        return x


class DummyLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.lm_head(x)


def test_lora_grouping():
    model = DummyModel()
    layers = find_lora_layers(model)
    groups = group_lora_layers(layers, scheme="attn_mlp")
    assert len(groups["attn"]) == 1
    assert len(groups["mlp"]) == 1
    assert len(groups["other"]) == 1


def test_lora_actuator_updates():
    model = DummyModel()
    actuator = LoRAAlphaActuator(model, update_every=1)
    stats = {"action": "thaw", "z": 5.0}
    actuator.update(stats, step_idx=1)
    assert model.attn.scaling != 1.0 or model.mlp.scaling != 1.0


def test_observe_only_mode():
    model = DummyModel()
    base_opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    opt = Z0Optimizer(base_opt, observe_only=True, model=model)

    # Create a fake loss to generate gradients
    loss = (model.attn.lora_A.weight ** 2).sum()
    loss.backward()

    lr_before = base_opt.param_groups[0]["lr"]
    scaling_before = model.attn.scaling

    opt.step(loss.item())

    lr_after = base_opt.param_groups[0]["lr"]
    scaling_after = model.attn.scaling

    assert lr_after == lr_before
    assert scaling_after == scaling_before


def test_disable_lora_control():
    model = DummyModel()
    base_opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    opt = Z0Optimizer(base_opt, model=model, controls={"lora": False})

    loss = (model.attn.lora_A.weight ** 2).sum()
    loss.backward()

    scaling_before = model.attn.scaling
    opt.step(loss.item())
    scaling_after = model.attn.scaling

    assert scaling_after == scaling_before


def test_logit_temperature_actuator():
    model = DummyLogitModel()
    actuator = LogitTemperatureActuator(model, update_every=1)
    x = torch.randn(2, 4)

    out_before = model(x)
    actuator.update({"action": "thaw"}, step_idx=1)
    out_after = model(x)

    assert not torch.allclose(out_before, out_after)


def test_monitor_setter_updates_controller():
    model = DummyModel()
    base_opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    opt = Z0Optimizer(base_opt, model=model)

    class DummyMonitor:
        def measure(self, loss, model=None, grad_norm=None, **kwargs):
            return 1.0

    dummy = DummyMonitor()
    opt.monitor = dummy
    assert opt.controller.monitor is dummy
