from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from ._io import sha256_json
from .events import LearningEvent


@dataclass(frozen=True)
class RoutingDecision:
    event_key: str
    acquisition: str
    requires_release: bool
    has_demonstrations: bool
    has_reward: bool
    rationale: str

    @property
    def fingerprint(self) -> str:
        return sha256_json(asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint
        return payload


class LearningSignalRouter:
    def route(self, event: LearningEvent) -> RoutingDecision:
        has_demo = event.targets is not None and event.targets.visibility == "optimizer"
        has_reward = event.verifier is not None
        if event.kind == "evaluation":
            acquisition = "evaluation"
        elif has_demo and has_reward:
            acquisition = "hybrid"
        elif has_demo:
            acquisition = "demonstration"
        elif has_reward:
            acquisition = "reward"
        else:
            raise ValueError(
                f"event {event.event_key} has neither optimizer-visible demonstrations nor a verifier"
            )
        requires_release = event.kind == "revision" or bool(event.supersedes)
        if requires_release and not event.supersedes:
            raise ValueError(f"revision event {event.event_key} does not declare superseded profiles")
        rationale = (
            f"targets={has_demo}, verifier={has_reward}, revision={requires_release}; route={acquisition}"
        )
        return RoutingDecision(
            event_key=event.event_key,
            acquisition=acquisition,
            requires_release=requires_release,
            has_demonstrations=has_demo,
            has_reward=has_reward,
            rationale=rationale,
        )
