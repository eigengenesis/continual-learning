from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Sequence, Set

from ._io import append_jsonl, sha256_file
from .events import LearningEvent


class EventDataAccessAudit:
    def __init__(self, path: Path) -> None:
        self.path = path

    def log(
        self,
        *,
        event: LearningEvent,
        row_ids: Sequence[str],
        purpose: str,
        split: str = "train",
        step: int = 0,
        committed_event_keys: Optional[Set[str]] = None,
    ) -> None:
        allowed = {record.example_id for record in event.examples.records}
        observed = {str(value) for value in row_ids}
        if purpose == "update":
            if not observed:
                raise RuntimeError(f"empty update row IDs for event={event.event_key}")
            prohibited = sorted(observed - allowed)
            if prohibited:
                raise RuntimeError(
                    f"event={event.event_key} accessed training rows outside its frozen dataset: {prohibited[:5]}"
                )
            replayed = self._historical_replays(event, observed, committed_event_keys)
            if replayed:
                raise RuntimeError(
                    f"event={event.event_key} attempted historical training-row replay: {replayed[:5]}"
                )
        dataset_identity = event.examples.source_uri or event.examples.dataset_id
        append_jsonl(
            self.path,
            {
                "time": time.time(),
                "event_key": event.event_key,
                "event_fingerprint": event.fingerprint,
                "dataset_id": event.examples.dataset_id,
                "dataset_identity": dataset_identity,
                "split": split,
                "purpose": purpose,
                "row_ids": sorted(observed),
                "step": int(step),
            },
        )

    def assert_update_allowed(
        self,
        event: LearningEvent,
        committed_event_keys: Optional[Set[str]] = None,
    ) -> None:
        row_ids = {record.example_id for record in event.examples.records}
        replayed = self._historical_replays(event, row_ids, committed_event_keys)
        if replayed:
            raise RuntimeError(
                f"event={event.event_key} includes historical training rows before acquisition: {replayed[:5]}"
            )

    def validate_event(
        self,
        event: LearningEvent,
        committed_event_keys: Optional[Set[str]] = None,
    ) -> List[str]:
        if not self.path.exists():
            return []
        allowed = {record.example_id for record in event.examples.records}
        dataset_identity = event.examples.source_uri or event.examples.dataset_id
        violations: List[str] = []
        parsed = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parsed.append((line_number, json.loads(line)))
        for line_number, row in parsed:
            if row.get("event_key") != event.event_key or row.get("purpose") != "update":
                continue
            unknown = sorted(set(row.get("row_ids", ())) - allowed)
            if unknown:
                violations.append(f"line={line_number} prohibited_rows={unknown[:5]}")
            for prior_number, prior in parsed:
                prior_identity = prior.get("dataset_identity") or prior.get("dataset_id")
                if (
                    prior.get("purpose") == "update"
                    and prior.get("event_key") != event.event_key
                    and (
                        committed_event_keys is None
                        or prior.get("event_key") in committed_event_keys
                    )
                    and prior_identity == dataset_identity
                ):
                    replayed = sorted(set(row.get("row_ids", ())) & set(prior.get("row_ids", ())))
                    if replayed:
                        violations.append(
                            f"line={line_number} historical_replay_from={prior_number} rows={replayed[:5]}"
                        )
        return violations

    def _historical_replays(
        self,
        event: LearningEvent,
        observed: Set[str],
        committed_event_keys: Optional[Set[str]] = None,
    ) -> List[str]:
        if not self.path.exists():
            return []
        dataset_identity = event.examples.source_uri or event.examples.dataset_id
        replayed = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prior_identity = row.get("dataset_identity") or row.get("dataset_id")
            if (
                row.get("purpose") == "update"
                and row.get("event_key") != event.event_key
                and (committed_event_keys is None or row.get("event_key") in committed_event_keys)
                and prior_identity == dataset_identity
            ):
                replayed.update(observed & set(row.get("row_ids", ())))
        return sorted(replayed)

    @property
    def fingerprint(self) -> str:
        return sha256_file(self.path) if self.path.exists() else ""
