from __future__ import annotations

import platform
from typing import Optional, Tuple


def _parse_macos_version() -> Optional[Tuple[int, int]]:
    version = platform.mac_ver()[0]
    if not version:
        return None
    parts = version.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return None
    return major, minor


def patch_torch_mps_is_macos_or_newer(torch_module) -> None:
    """
    Backfill torch.backends.mps.is_macos_or_newer for older torch releases.

    Transformers/TRL call this helper with (major, minor) arguments. Older torch
    exposes only is_macos13_or_newer(minor=0), so we provide a compatible shim.
    """
    if not hasattr(torch_module, "backends") or not hasattr(torch_module.backends, "mps"):
        return
    mps_backend = torch_module.backends.mps
    if hasattr(mps_backend, "is_macos_or_newer"):
        return

    def _is_macos_or_newer(major: int = 0, minor: int = 0) -> bool:
        if hasattr(mps_backend, "is_macos13_or_newer"):
            if major < 13:
                return True
            if major == 13:
                try:
                    return bool(mps_backend.is_macos13_or_newer(minor))
                except TypeError:
                    return bool(mps_backend.is_macos13_or_newer())

        current = _parse_macos_version()
        if current is None:
            return False
        return current >= (major, minor)

    mps_backend.is_macos_or_newer = _is_macos_or_newer  # type: ignore[attr-defined]
