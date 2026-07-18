from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._io import atomic_write_json, safe_name
from .events import LearningEvent


@dataclass(frozen=True)
class EventLease:
    event: LearningEvent
    path: Path
    leased_at: float


class DirectoryEventSource:
    """Durable single-consumer event queue using atomic directory renames."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.inbox = root / "inbox"
        self.leased = root / "leased"
        self.committed = root / "committed"
        self.rejected = root / "rejected"
        for path in (self.inbox, self.leased, self.committed, self.rejected):
            path.mkdir(parents=True, exist_ok=True)

    def submit(self, event: LearningEvent) -> Path:
        suffix = f"{safe_name(event.event_key)}-{event.fingerprint[:12]}"
        for directory in (self.committed, self.inbox, self.leased):
            if any(directory.glob(f"*-{suffix}.json")):
                raise FileExistsError(f"event is already queued or completed: {event.event_key}")
        sequence = time.time_ns()
        stem = f"{sequence:020d}-{suffix}"
        path = self.inbox / f"{stem}.json"
        atomic_write_json(path, event.to_dict())
        return path

    def peek(self) -> Optional[LearningEvent]:
        paths = sorted(self.inbox.glob("*.json"))
        if not paths:
            return None
        return LearningEvent.from_dict(json.loads(paths[0].read_text(encoding="utf-8")))

    def lease(self) -> Optional[EventLease]:
        for source in sorted(self.inbox.glob("*.json")):
            destination = self.leased / source.name
            try:
                os.replace(source, destination)
            except FileNotFoundError:
                continue
            event = LearningEvent.from_dict(json.loads(destination.read_text(encoding="utf-8")))
            return EventLease(event, destination, time.time())
        return None

    def ack(self, lease: EventLease, result: Dict[str, Any]) -> Path:
        destination = self.committed / lease.path.name
        atomic_write_json(lease.path.with_suffix(".result.json"), result)
        os.replace(lease.path, destination)
        result_path = self.committed / f"{lease.path.stem}.result.json"
        os.replace(self.leased / f"{lease.path.stem}.result.json", result_path)
        return destination

    def nack(self, lease: EventLease, reason: str, *, requeue: bool = False) -> Path:
        if requeue:
            destination = self.inbox / lease.path.name
            atomic_write_json(self.root / "last_requeue_error.json", {"reason": reason, "time": time.time()})
        else:
            destination = self.rejected / lease.path.name
            atomic_write_json(
                self.rejected / f"{lease.path.stem}.error.json",
                {"reason": reason, "time": time.time()},
            )
        os.replace(lease.path, destination)
        return destination

    def recover_leases(self, *, older_than_seconds: float = 3600.0) -> List[Path]:
        recovered: List[Path] = []
        cutoff = time.time() - float(older_than_seconds)
        for path in sorted(self.leased.glob("*.json")):
            if path.name.endswith(".result.json") or path.name.endswith(".error.json"):
                continue
            if path.stat().st_mtime > cutoff:
                continue
            destination = self.inbox / path.name
            os.replace(path, destination)
            recovered.append(destination)
        return recovered

    def checkpoint(self) -> Dict[str, int]:
        return {
            "inbox": len(list(self.inbox.glob("*.json"))),
            "leased": len([path for path in self.leased.glob("*.json") if ".result." not in path.name]),
            "committed": len([path for path in self.committed.glob("*.json") if ".result." not in path.name]),
            "rejected": len([path for path in self.rejected.glob("*.json") if ".error." not in path.name]),
        }
