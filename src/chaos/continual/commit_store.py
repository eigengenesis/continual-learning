from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._io import (
    append_jsonl,
    atomic_write_json,
    copy_path,
    file_manifest,
    fsync_directory,
    safe_name,
    sha256_file,
    sha256_json,
    verify_file_manifest,
)
from .artifacts import CandidateArtifact
from .events import LearningEvent
from .evaluator import EvaluationReport
from .profiles import ProfileRegistry
from .router import RoutingDecision


TERMINAL_TRANSACTION_STATES = {"committed", "rejected"}


@dataclass(frozen=True)
class CurrentVersion:
    version: int
    model_path: str
    registry_path: str
    source_event: str
    source_event_fingerprint: str
    commit_hash: str
    committed_at: float


@dataclass
class TransactionHandle:
    root: Path
    attempt_id: str
    event_key: str
    event_fingerprint: str
    state_path: Path

    @property
    def candidate_model_dir(self) -> Path:
        return self.root / "candidate" / "model"

    @property
    def acquisition_dir(self) -> Path:
        return self.root / "acquisition"

    def state(self) -> Dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def set_phase(self, phase: str, **metadata: Any) -> None:
        payload = self.state()
        if payload.get("status") in TERMINAL_TRANSACTION_STATES:
            raise RuntimeError(f"cannot change terminal transaction {self.attempt_id}")
        payload.update({"phase": phase, "updated_at": time.time(), **metadata})
        atomic_write_json(self.state_path, payload)

    def write_json(self, relative_path: str, payload: Any) -> Path:
        path = self.root / relative_path
        atomic_write_json(path, payload)
        return path


class TransactionStore:
    """Atomic version store for a single evolving checkpoint and profile registry."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.versions = root / "versions"
        self.transactions = root / "transactions"
        self.journal_path = root / "journal" / "events.jsonl"
        self.current_path = root / "current.json"
        self.lock_path = root / ".writer.lock"
        for path in (self.versions, self.transactions, self.journal_path.parent):
            path.mkdir(parents=True, exist_ok=True)
        self.recover()

    def initialize(
        self,
        *,
        model_path: Path,
        registry: Optional[ProfileRegistry] = None,
        model_hash: str = "",
    ) -> CurrentVersion:
        if self.current_path.exists():
            return self.current()
        version_dir = self.versions / "v000000"
        version_dir.mkdir(parents=True, exist_ok=False)
        registry = registry or ProfileRegistry()
        registry_path = version_dir / "registry.json"
        registry.save(registry_path)
        source_hash = model_hash or self._path_identity(model_path)
        atomic_write_json(
            version_dir / "source_model.json",
            {"external_path": str(model_path), "identity": source_hash},
        )
        atomic_write_json(version_dir / "acceptance.json", {"bootstrap": True})
        atomic_write_json(version_dir / "COMMITTED", {"version": 0, "time": time.time()})
        checksums = file_manifest(version_dir, exclude_names=("checksums.json",))
        atomic_write_json(version_dir / "checksums.json", checksums)
        commit_hash = sha256_json({"version": 0, "checksums": checksums, "model": source_hash})
        current = CurrentVersion(
            version=0,
            model_path=str(model_path),
            registry_path=str(registry_path),
            source_event="bootstrap",
            source_event_fingerprint="",
            commit_hash=commit_hash,
            committed_at=time.time(),
        )
        atomic_write_json(self.current_path, asdict(current))
        append_jsonl(self.journal_path, {"status": "committed", **asdict(current)})
        return current

    def current(self) -> CurrentVersion:
        if not self.current_path.exists():
            raise FileNotFoundError(f"continual store is not initialized: {self.current_path}")
        payload = json.loads(self.current_path.read_text(encoding="utf-8"))
        current = CurrentVersion(**payload)
        version_dir = self.versions / f"v{current.version:06d}"
        if not (version_dir / "COMMITTED").exists():
            raise RuntimeError(f"current pointer references an incomplete version: {version_dir}")
        checksums = json.loads((version_dir / "checksums.json").read_text(encoding="utf-8"))
        verify_file_manifest(version_dir, checksums)
        return current

    def registry(self) -> ProfileRegistry:
        return ProfileRegistry.load(Path(self.current().registry_path))

    def begin(self, event: LearningEvent, route: RoutingDecision) -> TransactionHandle:
        existing = self.event_status(event.event_key)
        if existing == "committed":
            raise FileExistsError(f"event already committed: {event.event_key}")
        event_dir = self.transactions / f"{safe_name(event.event_key)}-{event.fingerprint[:12]}"
        event_dir.mkdir(parents=True, exist_ok=True)
        attempts = sorted(path for path in event_dir.glob("attempt-*" ) if path.is_dir())
        attempt_number = len(attempts) + 1
        attempt_id = f"attempt-{attempt_number:04d}"
        root = event_dir / attempt_id
        root.mkdir(parents=True, exist_ok=False)
        state_path = root / "transaction.json"
        atomic_write_json(root / "event.json", event.to_dict())
        atomic_write_json(root / "route.json", route.to_dict())
        atomic_write_json(
            state_path,
            {
                "attempt_id": attempt_id,
                "event_key": event.event_key,
                "event_fingerprint": event.fingerprint,
                "base_version": self.current().version,
                "phase": "validated",
                "status": "running",
                "created_at": time.time(),
                "updated_at": time.time(),
            },
        )
        return TransactionHandle(root, attempt_id, event.event_key, event.fingerprint, state_path)

    def resumable(self, event: LearningEvent) -> Optional[TransactionHandle]:
        event_dir = self.transactions / f"{safe_name(event.event_key)}-{event.fingerprint[:12]}"
        if not event_dir.exists():
            return None
        for root in sorted(event_dir.glob("attempt-*"), reverse=True):
            state_path = root / "transaction.json"
            if not state_path.exists():
                continue
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") == "running":
                return TransactionHandle(
                    root,
                    str(state["attempt_id"]),
                    event.event_key,
                    event.fingerprint,
                    state_path,
                )
        return None

    def commit(
        self,
        transaction: TransactionHandle,
        *,
        event: LearningEvent,
        route: RoutingDecision,
        candidate: CandidateArtifact,
        registry: ProfileRegistry,
        report: EvaluationReport,
    ) -> CurrentVersion:
        if not report.passed:
            raise ValueError("cannot commit a candidate that failed mandatory gates")
        state = transaction.state()
        current = self.current()
        if int(state["base_version"]) != current.version:
            raise RuntimeError(
                f"transaction base version={state['base_version']} is stale; current={current.version}"
            )
        transaction.set_phase("commit_preparing")
        transaction.write_json("evaluation.json", report.to_dict())
        transaction.write_json("candidate/registry.json", registry.to_dict())
        transaction.write_json("candidate/artifact.json", candidate.to_dict())
        transaction.write_json("COMMIT_READY", {"event_key": event.event_key, "time": time.time()})
        checksums = file_manifest(
            transaction.root,
            exclude_names=("checksums.json", "transaction.json"),
        )
        transaction.write_json("checksums.json", checksums)
        verify_file_manifest(
            transaction.root,
            checksums,
            exclude_names=("checksums.json", "transaction.json"),
        )

        next_version = current.version + 1
        final_dir = self.versions / f"v{next_version:06d}"
        temp_dir = self.versions / f".v{next_version:06d}.{uuid.uuid4().hex}.tmp"
        if final_dir.exists():
            raise FileExistsError(f"version already exists: {final_dir}")
        temp_dir.mkdir(parents=True)
        source_model = Path(candidate.model_path)
        reuse_model = bool(candidate.metadata.get("reuse_current_model", False))
        if not source_model.exists():
            raise FileNotFoundError(f"candidate model is missing: {source_model}")
        if reuse_model:
            atomic_write_json(
                temp_dir / "model_reference.json",
                {"path": str(source_model), "identity": self._path_identity(source_model)},
            )
            published_model_path = str(source_model)
        else:
            copy_path(source_model, temp_dir / "model")
            published_model_path = str(final_dir / "model")
        registry.save(temp_dir / "registry.json")
        atomic_write_json(temp_dir / "event.json", event.to_dict())
        atomic_write_json(temp_dir / "route.json", route.to_dict())
        atomic_write_json(temp_dir / "acceptance.json", report.to_dict())
        atomic_write_json(temp_dir / "candidate.json", candidate.to_dict())
        atomic_write_json(temp_dir / "COMMITTED", {"version": next_version, "time": time.time()})
        version_checksums = file_manifest(temp_dir, exclude_names=("checksums.json",))
        atomic_write_json(temp_dir / "checksums.json", version_checksums)
        fsync_directory(temp_dir)
        os.replace(temp_dir, final_dir)
        fsync_directory(self.versions)
        commit_hash = sha256_json(
            {
                "version": next_version,
                "event": event.fingerprint,
                "registry": registry.fingerprint,
                "checksums": version_checksums,
            }
        )
        pointer = CurrentVersion(
            version=next_version,
            model_path=published_model_path,
            registry_path=str(final_dir / "registry.json"),
            source_event=event.event_key,
            source_event_fingerprint=event.fingerprint,
            commit_hash=commit_hash,
            committed_at=time.time(),
        )
        # The current pointer is the publication boundary and is always written last.
        atomic_write_json(self.current_path, asdict(pointer))
        append_jsonl(
            self.journal_path,
            {
                "status": "committed",
                "event_key": event.event_key,
                "event_fingerprint": event.fingerprint,
                "attempt_id": transaction.attempt_id,
                **asdict(pointer),
            },
        )
        terminal = transaction.state()
        terminal.update(
            {
                "phase": "committed",
                "status": "committed",
                "committed_version": next_version,
                "commit_hash": commit_hash,
                "updated_at": time.time(),
            }
        )
        atomic_write_json(transaction.state_path, terminal)
        self._finalize_transaction_artifacts(transaction.root)
        self.prune_versions(keep=2)
        return pointer

    def reject(
        self,
        transaction: TransactionHandle,
        *,
        event: LearningEvent,
        reason: str,
        report: Optional[EvaluationReport] = None,
    ) -> None:
        if report is not None:
            transaction.write_json("evaluation.json", report.to_dict())
        payload = transaction.state()
        payload.update(
            {
                "phase": "rejected",
                "status": "rejected",
                "reason": reason,
                "updated_at": time.time(),
            }
        )
        atomic_write_json(transaction.state_path, payload)
        self._finalize_transaction_artifacts(transaction.root)
        append_jsonl(
            self.journal_path,
            {
                "status": "rejected",
                "event_key": event.event_key,
                "event_fingerprint": event.fingerprint,
                "attempt_id": transaction.attempt_id,
                "reason": reason,
                "current_version": self.current().version,
                "time": time.time(),
            },
        )

    def event_status(self, event_key: str) -> str:
        status = ""
        for row in self.journal():
            if row.get("event_key") == event_key or row.get("source_event") == event_key:
                status = str(row.get("status", ""))
        return status

    def journal(self) -> List[Dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        return [json.loads(line) for line in self.journal_path.read_text(encoding="utf-8").splitlines() if line]

    def recover(self) -> None:
        if not self.current_path.exists():
            return
        current = CurrentVersion(**json.loads(self.current_path.read_text(encoding="utf-8")))
        version_dir = self.versions / f"v{current.version:06d}"
        if not (version_dir / "COMMITTED").exists():
            raise RuntimeError(f"published current version is incomplete: {version_dir}")
        known = any(
            row.get("status") == "committed" and int(row.get("version", -1)) == current.version
            for row in self.journal()
        )
        if not known:
            append_jsonl(self.journal_path, {"status": "committed", "recovered": True, **asdict(current)})
        for path in self.versions.glob(".v*.tmp"):
            shutil.rmtree(path)
        for path in self.versions.glob("v[0-9]*"):
            try:
                version = int(path.name[1:])
            except ValueError:
                continue
            if version > current.version:
                shutil.rmtree(path)
        matching = []
        for state_path in self.transactions.glob("**/transaction.json"):
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                state.get("status") == "running"
                and state.get("event_key") == current.source_event
                and state.get("event_fingerprint") == current.source_event_fingerprint
                and int(state.get("base_version", -1)) == current.version - 1
                and (state_path.parent / "COMMIT_READY").exists()
            ):
                matching.append(state_path)
        if matching:
            state_path = sorted(matching)[-1]
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state.update(
                {
                    "phase": "committed",
                    "status": "committed",
                    "committed_version": current.version,
                    "commit_hash": current.commit_hash,
                    "recovered": True,
                    "updated_at": time.time(),
                }
            )
            atomic_write_json(state_path, state)
            self._finalize_transaction_artifacts(state_path.parent)
        self.prune_versions(keep=2)

    def prune_versions(self, keep: int = 2) -> None:
        pointer = self.current()
        current = pointer.version
        candidates = sorted(
            (path for path in self.versions.glob("v[0-9]*") if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        retained = {f"v{current:06d}", "v000000"}
        retained.update(path.name for path in candidates[: max(1, int(keep))])
        current_model = Path(pointer.model_path).resolve()
        for path in candidates:
            model_path = (path / "model").resolve()
            if current_model == model_path:
                retained.add(path.name)
        for path in candidates:
            if path.name not in retained:
                shutil.rmtree(path)

    def _finalize_transaction_artifacts(self, root: Path) -> None:
        removable = [root / "candidate" / "model", root / "consolidation_resume"]
        acquisition = root / "acquisition"
        if acquisition.exists():
            removable.extend(path for path in acquisition.rglob("*") if path.name in {"resume_policy", "temporary_policy"})
        for path in sorted(removable, key=lambda item: len(item.parts), reverse=True):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        atomic_write_json(root / "weights_pruned.json", {"pruned": True, "time": time.time()})
        checksums = file_manifest(root, exclude_names=("checksums.json", "transaction.json"))
        atomic_write_json(root / "checksums.json", checksums)

    def _path_identity(self, path: Path) -> str:
        if path.is_file():
            return sha256_file(path)
        if path.is_dir():
            return sha256_json(file_manifest(path))
        return sha256_json({"external": str(path)})
