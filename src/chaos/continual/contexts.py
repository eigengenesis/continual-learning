from __future__ import annotations

import importlib
import inspect
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence

from ._io import sha256_file, sha256_json
from .events import ContextSpec, PublicExample
from .verifiers import prohibited_keys


@dataclass(frozen=True)
class ContextResult:
    prompt: str
    mode: str
    quality: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if prohibited_keys(self.metadata):
            raise ValueError("privileged-context metadata contains target-like fields")

    @property
    def fingerprint(self) -> str:
        return sha256_json(asdict(self))


class ContextProvider(Protocol):
    def build(self, example: PublicExample, mode: str, policy: Any) -> ContextResult:
        ...


ContextFactory = Callable[[ContextSpec], ContextProvider]
_CONTEXTS: Dict[str, ContextFactory] = {}


def register_context(name: str) -> Callable[[ContextFactory], ContextFactory]:
    def decorator(factory: ContextFactory) -> ContextFactory:
        if name in _CONTEXTS and _CONTEXTS[name] is not factory:
            raise ValueError(f"context provider already registered: {name}")
        _CONTEXTS[name] = factory
        return factory

    return decorator


def build_context_provider(spec: ContextSpec) -> ContextProvider:
    if spec.import_path:
        module_name, separator, attribute = spec.import_path.partition(":")
        if not separator:
            module_name, separator, attribute = spec.import_path.rpartition(".")
        if not module_name or not attribute:
            raise ValueError(f"invalid context import path={spec.import_path!r}")
        factory = getattr(importlib.import_module(module_name), attribute)
        source = inspect.getsourcefile(factory if inspect.isclass(factory) or inspect.isfunction(factory) else type(factory))
        if source is None or sha256_file(Path(source)) != spec.code_hash:
            raise RuntimeError(f"context provider code hash mismatch for {spec.import_path}")
        provider = factory(spec)
        if not hasattr(provider, "build"):
            raise TypeError(f"context provider {spec.import_path} does not expose build()")
        return provider
    try:
        return _CONTEXTS[spec.name](spec)
    except KeyError as exc:
        raise KeyError(f"unknown context provider={spec.name}; registered={sorted(_CONTEXTS)}") from exc


class NoContextProvider:
    def build(self, example: PublicExample, mode: str, policy: Any) -> ContextResult:
        del policy
        return ContextResult(example.prompt, "none" if mode == "none" else mode)


@register_context("none")
def _none_factory(spec: ContextSpec) -> ContextProvider:
    del spec
    return NoContextProvider()


class TemplateContextProvider:
    """Formats only public example metadata; private targets are never available here."""

    def __init__(self, templates: Mapping[str, str]) -> None:
        self.templates = dict(templates)

    def build(self, example: PublicExample, mode: str, policy: Any) -> ContextResult:
        del policy
        if mode == "none":
            return ContextResult(example.prompt, mode)
        template = self.templates.get(mode)
        if template is None:
            raise KeyError(f"no privileged-context template for mode={mode}")
        public = {key: value for key, value in example.metadata.items() if str(key).lower() not in {"target", "answer", "gold"}}
        context = template.format(prompt=example.prompt, **public)
        return ContextResult(context, mode, metadata={"template": mode})


@register_context("template")
def _template_factory(spec: ContextSpec) -> ContextProvider:
    templates = spec.config.get("templates", {})
    if not isinstance(templates, Mapping):
        raise TypeError("template context config requires a templates mapping")
    return TemplateContextProvider(templates)


def deterministic_context_modes(total: int, mixture: Mapping[str, float], salt: str) -> Sequence[str]:
    if total <= 0:
        return ()
    allowed = {"full", "compressed", "none"}
    unknown = set(mixture) - allowed
    if unknown:
        raise ValueError(f"unsupported context modes={sorted(unknown)}")
    weights = {key: max(0.0, float(mixture.get(key, 0.0))) for key in allowed}
    if sum(weights.values()) <= 0:
        return ("none",) * total
    normalized = {key: value / sum(weights.values()) for key, value in weights.items()}
    raw = {key: normalized[key] * total for key in allowed}
    counts = {key: int(raw[key]) for key in allowed}
    remaining = total - sum(counts.values())
    ranked = sorted(allowed, key=lambda key: (-(raw[key] - counts[key]), key))
    for key in ranked[:remaining]:
        counts[key] += 1
    values = [(key, index) for key in sorted(allowed) for index in range(counts[key])]
    # Stable hash ordering gives exact counts without depending on process-randomized hash().
    ordered = sorted(values, key=lambda value: sha256_json({"salt": salt, "mode": value[0], "i": value[1]}))
    return tuple(value[0] for value in ordered)
