from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from ._io import atomic_write_json, sha256_json
from .verifiers import prohibited_keys


@dataclass(frozen=True)
class AcquisitionArtifact:
    event_key: str
    route: str
    base_policy_version: int
    base_policy_hash: str
    temporary_policy_version: str
    adapter_path: str
    sample_ledger_path: str
    example_ids: Tuple[str, ...]
    candidate_layers: Tuple[int, ...]
    metrics: Mapping[str, float]
    access_log_hash: str = ""
    rng_state_path: str = ""
    contains_optimizer_targets: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.route in {"reward", "hybrid"} and prohibited_keys(self.metadata):
            raise ValueError("reward acquisition artifact metadata leaks target-like fields")

    @property
    def fingerprint(self) -> str:
        return sha256_json(asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint
        return payload

    def save(self, path: Path) -> None:
        atomic_write_json(path, self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AcquisitionArtifact":
        raw = dict(payload)
        expected = raw.pop("fingerprint", "")
        artifact = cls(
            example_ids=tuple(raw.pop("example_ids", ())),
            candidate_layers=tuple(raw.pop("candidate_layers", ())),
            **raw,
        )
        if expected and expected != artifact.fingerprint:
            raise ValueError("acquisition artifact fingerprint mismatch")
        return artifact


@dataclass(frozen=True)
class CandidateArtifact:
    event_key: str
    model_path: str
    registry_path: str
    metrics_path: str
    source_policy_version: int
    selected_layers: Tuple[int, ...]
    profile_ids: Tuple[str, ...]
    numerical_stable: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return sha256_json(asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateArtifact":
        raw = dict(payload)
        expected = raw.pop("fingerprint", "")
        artifact = cls(
            selected_layers=tuple(raw.pop("selected_layers", ())),
            profile_ids=tuple(raw.pop("profile_ids", ())),
            **raw,
        )
        if expected and expected != artifact.fingerprint:
            raise ValueError("candidate artifact fingerprint mismatch")
        return artifact
