from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ._io import atomic_write_json, safe_name, sha256_file, sha256_json


PROFILE_STATUSES = {"protected", "released", "retired"}


@dataclass(frozen=True)
class ProfileRecord:
    profile_id: str
    capability: str
    dependencies: Tuple[str, ...]
    selected_layers: Tuple[int, ...]
    checkpoint_version: int
    checkpoint_hash: str
    creation_event: str
    status: str = "protected"
    scope: Mapping[str, str] = field(default_factory=dict)
    tensor_path: str = ""
    tensor_hash: str = ""
    tensor_schema: str = "parameter_delta_basis_v1"
    metrics: Mapping[str, float] = field(default_factory=dict)
    canary_dataset: str = ""
    canary_path: str = ""

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or not self.capability.strip():
            raise ValueError("profile_id and capability must be non-empty")
        if self.status not in PROFILE_STATUSES:
            raise ValueError(f"unsupported profile status={self.status}")
        if self.checkpoint_version < 0:
            raise ValueError("profile checkpoint version must be non-negative")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError(f"profile {self.profile_id} has duplicate dependencies")
        if len(self.selected_layers) != len(set(self.selected_layers)):
            raise ValueError(f"profile {self.profile_id} has duplicate selected layers")
        if any(layer < 0 for layer in self.selected_layers):
            raise ValueError(f"profile {self.profile_id} has a negative layer index")
        if any(not math.isfinite(float(value)) for value in self.metrics.values()):
            raise ValueError(f"profile {self.profile_id} has non-finite metrics")


@dataclass
class ProfileRegistry:
    records: Dict[str, ProfileRecord] = field(default_factory=dict)
    schema_version: int = 1

    def clone(self) -> "ProfileRegistry":
        return ProfileRegistry(records=copy.deepcopy(self.records), schema_version=self.schema_version)

    def register(self, profile: ProfileRecord, *, replace_retired: bool = False) -> None:
        existing = self.records.get(profile.profile_id)
        if existing is not None and not (replace_retired and existing.status == "retired"):
            raise ValueError(f"profile already exists: {profile.profile_id}")
        self.records[profile.profile_id] = profile

    def active(self) -> List[ProfileRecord]:
        return [record for record in self.records.values() if record.status == "protected"]

    def dependency_closure(self, roots: Sequence[str]) -> List[str]:
        unknown = sorted(set(roots) - set(self.records))
        if unknown:
            raise KeyError(f"unknown profile IDs in release request={unknown}")
        closure = set(roots)
        changed = True
        while changed:
            changed = False
            for profile_id, record in self.records.items():
                if profile_id in closure or record.status == "retired":
                    continue
                if closure.intersection(record.dependencies):
                    closure.add(profile_id)
                    changed = True
        return sorted(closure)

    def release_closure(self, roots: Sequence[str]) -> List[str]:
        closure = self.dependency_closure(roots)
        for profile_id in closure:
            record = self.records[profile_id]
            if record.status == "protected":
                self.records[profile_id] = ProfileRecord(**{**asdict(record), "status": "released"})
        return closure

    def retire(self, profile_ids: Sequence[str]) -> None:
        for profile_id in profile_ids:
            if profile_id not in self.records:
                raise KeyError(f"cannot retire unknown profile={profile_id}")
            record = self.records[profile_id]
            self.records[profile_id] = ProfileRecord(**{**asdict(record), "status": "retired"})

    def protected_excluding(self, profile_ids: Sequence[str]) -> List[ProfileRecord]:
        excluded = set(profile_ids)
        return [record for record in self.active() if record.profile_id not in excluded]

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "profiles": {key: asdict(value) for key, value in sorted(self.records.items())},
        }
        if include_fingerprint:
            payload["fingerprint"] = sha256_json(payload)
        return payload

    def save(self, path: Path) -> None:
        atomic_write_json(path, self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProfileRegistry":
        raw = dict(payload)
        expected = raw.pop("fingerprint", "")
        registry = cls(
            schema_version=int(raw.get("schema_version", 1)),
            records={
                key: ProfileRecord(
                    dependencies=tuple(value.get("dependencies", ())),
                    selected_layers=tuple(value.get("selected_layers", ())),
                    **{k: v for k, v in value.items() if k not in {"dependencies", "selected_layers"}},
                )
                for key, value in raw.get("profiles", {}).items()
            },
        )
        if expected and expected != registry.fingerprint:
            raise ValueError(f"profile registry fingerprint mismatch expected={expected} actual={registry.fingerprint}")
        return registry

    @classmethod
    def load(cls, path: Path) -> "ProfileRegistry":
        import json

        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class ProfileTensorStore:
    """Content-addressed CPU safetensor storage for geometry profile tensors."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, profile_id: str, tensors: Mapping[str, Any]) -> Tuple[str, str]:
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise ImportError("ProfileTensorStore requires torch and safetensors") from exc
        if not tensors:
            raise ValueError("cannot save an empty profile tensor set")
        normalized: Dict[str, Any] = {}
        for key, value in tensors.items():
            tensor = torch.as_tensor(value).detach().to(device="cpu").contiguous()
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"profile tensor {key} contains NaN/Inf")
            normalized[str(key)] = tensor
        provisional = self.root / f".{safe_name(profile_id)}.tmp.safetensors"
        save_file(normalized, str(provisional))
        digest = sha256_file(provisional)
        destination = self.root / f"{digest}.safetensors"
        if destination.exists():
            provisional.unlink()
        else:
            provisional.replace(destination)
        return str(destination.relative_to(self.root.parent)), digest

    def load(self, relative_path: str, expected_hash: str = "") -> Dict[str, Any]:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover
            raise ImportError("ProfileTensorStore requires safetensors") from exc
        path = self.root.parent / relative_path
        if expected_hash and sha256_file(path) != expected_hash:
            raise RuntimeError(f"profile tensor checksum mismatch: {path}")
        return load_file(str(path), device="cpu")
