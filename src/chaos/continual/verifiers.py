from __future__ import annotations

import asyncio
import importlib
import inspect
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Protocol, Set

from ._io import sha256_file
from .events import ExampleRecord, VerifierSpec


PROHIBITED_DIAGNOSTIC_KEYS = {
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


@dataclass(frozen=True)
class TrajectoryView:
    prompt: str
    completion: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardResult:
    reward: float
    success: bool
    valid: bool
    stale: bool = False
    components: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.reward)):
            raise ValueError("verifier reward must be finite")
        if any(not math.isfinite(float(value)) for value in self.components.values()):
            raise ValueError("verifier reward components must be finite")
        leaked = prohibited_keys(self.diagnostics)
        if leaked:
            raise ValueError(f"verifier diagnostics contain prohibited target-like keys={sorted(leaked)}")

    def public_dict(self) -> Dict[str, Any]:
        return asdict(self)


def prohibited_keys(value: Any) -> Set[str]:
    found: Set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().strip()
            if normalized in PROHIBITED_DIAGNOSTIC_KEYS:
                found.add(normalized)
            found.update(prohibited_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(prohibited_keys(child))
    return found


class Verifier(Protocol):
    async def score(self, example: ExampleRecord, trajectory: TrajectoryView) -> RewardResult:
        ...


VerifierFactory = Callable[[VerifierSpec], Verifier]
_VERIFIERS: Dict[str, VerifierFactory] = {}


def register_verifier(name: str) -> Callable[[VerifierFactory], VerifierFactory]:
    def decorator(factory: VerifierFactory) -> VerifierFactory:
        if not name.strip():
            raise ValueError("verifier registry name must be non-empty")
        if name in _VERIFIERS and _VERIFIERS[name] is not factory:
            raise ValueError(f"verifier already registered: {name}")
        _VERIFIERS[name] = factory
        return factory

    return decorator


def build_verifier(spec: VerifierSpec) -> Verifier:
    if spec.import_path:
        module_name, separator, attribute = spec.import_path.partition(":")
        if not separator:
            module_name, separator, attribute = spec.import_path.rpartition(".")
        if not module_name or not attribute:
            raise ValueError(f"invalid verifier import path={spec.import_path!r}")
        factory = getattr(importlib.import_module(module_name), attribute)
        source = inspect.getsourcefile(factory if inspect.isclass(factory) or inspect.isfunction(factory) else type(factory))
        if source is None or sha256_file(Path(source)) != spec.code_hash:
            raise RuntimeError(f"verifier code hash mismatch for {spec.import_path}")
        instance = factory(spec) if callable(factory) else factory
        if not hasattr(instance, "score"):
            raise TypeError(f"verifier plugin {spec.import_path} does not expose score()")
        return instance
    try:
        return _VERIFIERS[spec.name](spec)
    except KeyError as exc:
        raise KeyError(f"unknown verifier={spec.name}; registered={sorted(_VERIFIERS)}") from exc


class ExactMatchVerifier:
    def __init__(self, *, case_sensitive: bool = False, strip: bool = True) -> None:
        self.case_sensitive = bool(case_sensitive)
        self.strip = bool(strip)

    def normalize(self, value: str) -> str:
        normalized = str(value)
        if self.strip:
            normalized = normalized.strip()
        if not self.case_sensitive:
            normalized = normalized.casefold()
        return normalized

    async def score(self, example: ExampleRecord, trajectory: TrajectoryView) -> RewardResult:
        if example.target is None:
            raise ValueError(f"exact_match verifier has no private target for {example.example_id}")
        prediction = self.normalize(trajectory.completion)
        expected = self.normalize(example.target)
        correct = prediction == expected
        valid = bool(prediction)
        return RewardResult(
            reward=1.0 if correct else (0.0 if valid else -0.25),
            success=correct,
            valid=valid,
            components={"correctness": 1.0 if correct else 0.0, "format": 1.0 if valid else 0.0},
            diagnostics={"prediction_length": len(prediction)},
        )


@register_verifier("exact_match")
def _exact_match_factory(spec: VerifierSpec) -> Verifier:
    return ExactMatchVerifier(**dict(spec.config))


class RevisionExactVerifier(ExactMatchVerifier):
    async def score(self, example: ExampleRecord, trajectory: TrajectoryView) -> RewardResult:
        base = await super().score(example, trajectory)
        stale_output = str(example.metadata.get("stale_output", "")).strip()
        stale = bool(stale_output) and self.normalize(trajectory.completion) == self.normalize(stale_output)
        reward = -0.5 if stale else base.reward
        return RewardResult(
            reward=reward,
            success=base.success,
            valid=base.valid,
            stale=stale,
            components={**dict(base.components), "stale": 1.0 if stale else 0.0},
            diagnostics=base.diagnostics,
        )


@register_verifier("revision_exact")
def _revision_exact_factory(spec: VerifierSpec) -> Verifier:
    return RevisionExactVerifier(**dict(spec.config))


class RegexVerifier:
    def __init__(self, pattern: str, success_reward: float = 1.0, failure_reward: float = 0.0) -> None:
        self.pattern = re.compile(pattern)
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)

    async def score(self, example: ExampleRecord, trajectory: TrajectoryView) -> RewardResult:
        del example
        match = self.pattern.fullmatch(trajectory.completion.strip())
        return RewardResult(
            reward=self.success_reward if match else self.failure_reward,
            success=bool(match),
            valid=bool(match),
            components={"regex_match": 1.0 if match else 0.0},
        )


@register_verifier("regex")
def _regex_factory(spec: VerifierSpec) -> Verifier:
    return RegexVerifier(**dict(spec.config))


def score_sync(verifier: Verifier, example: ExampleRecord, trajectory: TrajectoryView) -> RewardResult:
    result = verifier.score(example, trajectory)
    if not inspect.isawaitable(result):
        return result  # type: ignore[return-value]
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)
    raise RuntimeError("score_sync cannot run inside an active event loop; await verifier.score() instead")
