from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ._io import sha256_json


EVENT_KINDS = {"demonstration", "reward", "revision", "hybrid", "evaluation"}
TARGET_VISIBILITIES = {"optimizer", "verifier_only", "none"}
GATE_OPERATORS = {"ge", "le", "delta_ge", "drop_le", "finite", "true"}
PRIVATE_METADATA_KEYS = {
    "answer",
    "correct_answer",
    "correct_output",
    "expected",
    "gold",
    "ground_truth",
    "label",
    "reference_answer",
    "solution",
    "stale_output",
    "target",
}


def public_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): public_metadata(child)
            for key, child in value.items()
            if str(key).strip().lower() not in PRIVATE_METADATA_KEYS
        }
    if isinstance(value, tuple):
        return tuple(public_metadata(child) for child in value)
    if isinstance(value, list):
        return [public_metadata(child) for child in value]
    return value


def metadata_contains_exact_value(value: Any, expected: str) -> bool:
    normalized = str(expected).strip().casefold()
    if not normalized:
        return False
    if isinstance(value, Mapping):
        return any(metadata_contains_exact_value(child, expected) for child in value.values())
    if isinstance(value, (tuple, list)):
        return any(metadata_contains_exact_value(child, expected) for child in value)
    return str(value).strip().casefold() == normalized


@dataclass(frozen=True)
class PublicExample:
    example_id: str
    prompt: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExampleRecord:
    example_id: str
    prompt: str
    target: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("example_id must be non-empty")
        if not self.prompt.strip():
            raise ValueError(f"example {self.example_id}: prompt must be non-empty")

    def public(self) -> PublicExample:
        return PublicExample(self.example_id, self.prompt, public_metadata(self.metadata))


@dataclass(frozen=True)
class DatasetRef:
    dataset_id: str
    split: str
    records: Tuple[ExampleRecord, ...]
    source_uri: str = ""
    source_checksum: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.split.strip():
            raise ValueError("dataset_id and split must be non-empty")
        ids = [record.example_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"dataset {self.dataset_id}/{self.split} has duplicate example IDs")
        if not records_are_nonempty(self.records):
            raise ValueError(f"dataset {self.dataset_id}/{self.split} must contain records")

    @property
    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


def records_are_nonempty(records: Sequence[ExampleRecord]) -> bool:
    return len(records) > 0


@dataclass(frozen=True)
class TargetRef:
    field: str = "target"
    visibility: str = "optimizer"

    def __post_init__(self) -> None:
        if self.visibility not in TARGET_VISIBILITIES:
            raise ValueError(f"unsupported target visibility={self.visibility}")


@dataclass(frozen=True)
class VerifierSpec:
    name: str
    config: Mapping[str, Any] = field(default_factory=dict)
    import_path: str = ""
    code_hash: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("verifier name must be non-empty")
        if self.import_path and not self.code_hash:
            raise ValueError("imported verifier plugins require an explicit code_hash")

    @property
    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class ContextSpec:
    name: str
    config: Mapping[str, Any] = field(default_factory=dict)
    import_path: str = ""
    code_hash: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("context provider name must be non-empty")
        if self.import_path and not self.code_hash:
            raise ValueError("imported context providers require an explicit code_hash")

    @property
    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class GateRule:
    gate_id: str
    category: str
    metric: str
    operator: str
    threshold: float = 0.0
    required: bool = True

    def __post_init__(self) -> None:
        if self.operator not in GATE_OPERATORS:
            raise ValueError(f"unsupported gate operator={self.operator}")
        if not self.gate_id.strip() or not self.metric.strip():
            raise ValueError("gate_id and metric must be non-empty")


@dataclass(frozen=True)
class GateBundle:
    rules: Tuple[GateRule, ...] = ()
    require_numerical_stability: bool = True
    require_access_audit: bool = True
    require_budget: bool = True
    require_retention: bool = True
    require_general_capability: bool = True
    require_staleness_on_revision: bool = True


@dataclass(frozen=True)
class AcquisitionBudget:
    max_optimizer_steps: int = 120
    max_rollouts: int = 4096
    max_tokens: int = 1_000_000
    max_wall_seconds: float = 43_200.0
    group_size: int = 4
    batch_size: int = 1

    def __post_init__(self) -> None:
        values = (
            self.max_optimizer_steps,
            self.max_rollouts,
            self.max_tokens,
            self.group_size,
            self.batch_size,
        )
        if any(int(value) <= 0 for value in values) or float(self.max_wall_seconds) <= 0:
            raise ValueError("all acquisition budget values must be positive")


@dataclass(frozen=True)
class LearningEvent:
    event_id: str
    revision: int
    kind: str
    examples: DatasetRef
    eval_examples: Optional[DatasetRef] = None
    targets: Optional[TargetRef] = None
    verifier: Optional[VerifierSpec] = None
    privileged_context: Optional[ContextSpec] = None
    dependencies: Tuple[str, ...] = ()
    supersedes: Tuple[str, ...] = ()
    gates: GateBundle = field(default_factory=GateBundle)
    budget: AcquisitionBudget = field(default_factory=AcquisitionBudget)
    seed: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must be non-empty")
        if self.revision < 0:
            raise ValueError("event revision must be >= 0")
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unsupported event kind={self.kind}")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError("event dependencies must be unique")
        if len(self.supersedes) != len(set(self.supersedes)):
            raise ValueError("event supersedes entries must be unique")
        if self.kind == "demonstration" and self.targets is None:
            raise ValueError("demonstration events require targets")
        if self.kind == "reward" and self.verifier is None:
            raise ValueError("reward events require a verifier")
        if self.kind == "hybrid" and (self.targets is None or self.verifier is None):
            raise ValueError("hybrid events require targets and a verifier")
        if self.kind == "revision" and self.targets is None and self.verifier is None:
            raise ValueError("revision events require demonstrations, reward, or both")
        if self.kind == "revision" and not self.supersedes:
            raise ValueError("revision events must explicitly identify superseded profiles")
        if self.targets is not None and self.targets.visibility == "optimizer":
            missing = [record.example_id for record in self.examples.records if record.target is None]
            if missing:
                raise ValueError(f"optimizer-visible targets missing for examples={missing[:5]}")
        if (
            self.kind != "evaluation"
            and self.targets is not None
            and self.targets.visibility == "verifier_only"
        ):
            leaked = [
                record.example_id
                for record in self.examples.records
                if record.target is not None
                and metadata_contains_exact_value(record.public().metadata, record.target)
            ]
            if leaked:
                raise ValueError(
                    "verifier-only target value appears in public metadata for "
                    f"examples={leaked[:5]}"
                )

    @property
    def event_key(self) -> str:
        return f"{self.event_id}@{self.revision}"

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict(include_fingerprint=False))

    def public_examples(self) -> Tuple[PublicExample, ...]:
        return tuple(record.public() for record in self.examples.records)

    def to_dict(self, *, include_fingerprint: bool = True) -> Dict[str, Any]:
        payload = asdict(self)
        if include_fingerprint:
            payload["fingerprint"] = sha256_json(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LearningEvent":
        raw = dict(payload)
        expected = raw.pop("fingerprint", "")
        examples = dataset_from_dict(raw.pop("examples"))
        raw_eval = raw.pop("eval_examples", None)
        targets = raw.pop("targets", None)
        verifier = raw.pop("verifier", None)
        context = raw.pop("privileged_context", None)
        gates = raw.pop("gates", {})
        budget = raw.pop("budget", {})
        event = cls(
            examples=examples,
            eval_examples=dataset_from_dict(raw_eval) if raw_eval else None,
            targets=TargetRef(**targets) if targets else None,
            verifier=VerifierSpec(**verifier) if verifier else None,
            privileged_context=ContextSpec(**context) if context else None,
            dependencies=tuple(raw.pop("dependencies", ())),
            supersedes=tuple(raw.pop("supersedes", ())),
            gates=GateBundle(
                rules=tuple(GateRule(**item) for item in gates.get("rules", ())),
                require_numerical_stability=bool(gates.get("require_numerical_stability", True)),
                require_access_audit=bool(gates.get("require_access_audit", True)),
                require_budget=bool(gates.get("require_budget", True)),
                require_retention=bool(gates.get("require_retention", True)),
                require_general_capability=bool(gates.get("require_general_capability", True)),
                require_staleness_on_revision=bool(gates.get("require_staleness_on_revision", True)),
            ),
            budget=AcquisitionBudget(**budget),
            **raw,
        )
        if expected and expected != event.fingerprint:
            raise ValueError(
                f"event fingerprint mismatch for {event.event_key}: expected={expected} actual={event.fingerprint}"
            )
        return event


def dataset_from_dict(payload: Mapping[str, Any]) -> DatasetRef:
    raw = dict(payload)
    records = tuple(ExampleRecord(**item) for item in raw.pop("records"))
    return DatasetRef(records=records, **raw)
