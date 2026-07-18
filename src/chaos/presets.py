from typing import Dict, Any, Tuple


def pre() -> Dict[str, Dict[str, Any]]:
    """
    Preset for pretraining workloads.
    """
    return {
        "optimizer": {
            "target_z": 1.0,
            "auto_target": True,
            "lr_bounds": (1e-5, 2e-3),
        },
        "controls": {
            "lr": True,
            "lora": False,
        },
    }


def sft() -> Dict[str, Dict[str, Any]]:
    """
    Preset for supervised fine-tuning (SFT).
    Uses a higher target impedance with tight LR bounds and conservative batch scaling.
    """
    return {
        "optimizer": {
            "target_z": 10.0,
            "auto_target": False,
            "lr_bounds": (1e-6, 5e-5),
        },
        "controls": {
            "lr": True,
            "lora": True,
        },
        "batch": {
            "min_steps": 1,
            "max_steps": 4,
        },
    }


def rl() -> Dict[str, Dict[str, Any]]:
    """
    Preset for RL workloads with noisy gradients.
    Uses auto-targeting and wider LR bounds with more batch scaling headroom.
    """
    return {
        "optimizer": {
            "auto_target": True,
            "lr_bounds": (1e-5, 1e-3),
        },
        "controls": {
            "lr": True,
            "lora": False,
        },
        "batch": {
            "min_steps": 1,
            "max_steps": 16,
        },
    }


def dpo() -> Dict[str, Dict[str, Any]]:
    """
    Preset for DPO workloads.
    """
    return {
        "optimizer": {
            "target_z": 8.0,
            "auto_target": False,
            "lr_bounds": (1e-6, 5e-5),
        },
        "controls": {
            "lr": True,
            "lora": True,
        },
    }
