from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, (canonical_json(payload) + "\n").encode("utf-8"))


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(path.parent)


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def file_manifest(root: Path, *, exclude_names: Sequence[str] = ()) -> Dict[str, str]:
    excluded = set(exclude_names)
    result: Dict[str, str] = {}
    if not root.exists():
        return result
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in excluded:
            continue
        result[str(path.relative_to(root))] = sha256_file(path)
    return result


def verify_file_manifest(
    root: Path,
    expected: Mapping[str, str],
    *,
    exclude_names: Sequence[str] = ("checksums.json",),
) -> None:
    actual = file_manifest(root, exclude_names=exclude_names)
    if dict(expected) != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(key for key in set(expected) & set(actual) if expected[key] != actual[key])
        raise RuntimeError(
            f"artifact checksum mismatch missing={missing[:5]} extra={extra[:5]} changed={changed[:5]}"
        )


def copy_path(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def all_finite(values: Iterable[Any]) -> bool:
    for value in values:
        try:
            if not math.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed"
