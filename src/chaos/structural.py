import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureControlledNorm(nn.Module):
    """
    Drop-in replacement for LayerNorm with an externally controlled \"temperature\".
    - Applies standard LayerNorm for stability.
    - Scales the normalized activations by 1/temperature to widen/narrow the distribution.
      High T → smaller scale (more exploration), Low T → larger scale (sharper/colder).
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, initial_temp=1.0, learned_temp=False):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        if learned_temp:
            self.temperature = nn.Parameter(torch.tensor(float(initial_temp)))
        else:
            self.register_buffer('temperature', torch.tensor(float(initial_temp)))

    def set_temperature(self, t: float):
        with torch.no_grad():
            self.temperature.fill_(float(t))

    def get_entropy(self, x):
        """
        Proxy entropy: softmax over feature dimension of the normalized tensor.
        Useful for diagnostics; not used in the forward pass.
        """
        # Flatten last dimension for entropy estimation
        x_flat = x.view(x.size(0), -1)
        probs = F.softmax(x_flat, dim=1)
        return -(probs * torch.log(probs + 1e-9)).sum(dim=1).mean()

    def forward(self, x):
        y = self.layernorm(x)
        # Temperature scales variance; safeguard against zero/neg values
        t = torch.clamp(self.temperature, min=1e-3)
        return y / t

class StructuralThermostat:
    """
    Couples Z-Monitor to Structural Temperature.

    Logic:
    If Z is High (Ice) -> Increase T (Melt structure to allow movement)
    If Z is Low (Gas) -> Decrease T (Freeze structure to force decisions)
    """
    def __init__(self, model_layers, base_temp=1.0):
        self.layers = model_layers # List of TemperatureControlledNorm layers
        self.base_temp = base_temp

    def step(self, z_current):
        # Simple negative feedback loop
        # Z high (frozen) -> Need heat -> T > 1.0
        # Z low (chaotic) -> Need cooling -> T < 1.0

        # Transfer function: T = Base * log(Z + 1) ? or T = Base * Z?
        # Let's keep it robust:
        if z_current > 3.0:
            target_t = self.base_temp * 1.5 # Melt
        elif z_current < 0.5:
            target_t = self.base_temp * 0.8 # Freeze
        else:
            target_t = self.base_temp # Maintain

        for layer in self.layers:
            layer.set_temperature(target_t)

        return target_t

def inject_thermodynamics(model: nn.Module,
                          target_types=(nn.LayerNorm,),
                          verbose: bool = True) -> StructuralThermostat:
    """
    Automatically converts a standard PyTorch model into a Thermodynamic Model.

    Arg:
        model: The PyTorch model (e.g., HuggingFace Transformer)
        target_types: Tuple of layer classes to replace (default: nn.LayerNorm)

    Returns:
        StructuralThermostat: The controller for the injected layers.
    """
    controlled_layers = []

    # Recursive replacement function
    def replace_layers(module, prefix=""):
        for name, child in module.named_children():
            if isinstance(child, target_types):
                normalized_shape = getattr(child, "normalized_shape", None)
                if normalized_shape is None and hasattr(child, "weight") and child.weight is not None:
                    normalized_shape = child.weight.shape
                if normalized_shape is None:
                    continue
                eps = getattr(child, "eps", 1e-5)
                elementwise_affine = getattr(child, "elementwise_affine", True)

                new_layer = TemperatureControlledNorm(
                    normalized_shape,
                    eps=eps,
                    elementwise_affine=elementwise_affine,
                    initial_temp=1.0,
                    learned_temp=False,
                )

                if elementwise_affine and hasattr(child, "weight") and child.weight is not None:
                    with torch.no_grad():
                        new_layer.layernorm.weight.copy_(child.weight)
                if elementwise_affine and hasattr(child, "bias") and child.bias is not None:
                    with torch.no_grad():
                        new_layer.layernorm.bias.copy_(child.bias)

                setattr(module, name, new_layer)
                controlled_layers.append(new_layer)
                if verbose:
                    print(f"🔥 Injected Thermodynamics: {prefix}.{name}")
            else:
                # Recurse
                replace_layers(child, prefix=f"{prefix}.{name}" if prefix else name)

    replace_layers(model)

    if not controlled_layers:
        print("⚠️ Warning: No layers matched the target types. Structural control inactive.")

    return StructuralThermostat(controlled_layers, base_temp=1.0)
