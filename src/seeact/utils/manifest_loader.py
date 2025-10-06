from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional


def _normalize_domain(value: str) -> str:
    value = (value or "").strip().lower()
    if value.startswith("http://") or value.startswith("https://"):
        from urllib.parse import urlparse

        parsed = urlparse(value)
        host = parsed.hostname or ""
    else:
        host = value
    if host.startswith("www."):
        host = host[4:]
    return host


def _candidate_filenames(domain: str) -> Iterable[str]:
    base = _normalize_domain(domain)
    if not base:
        return []
    parts = base.split(".")
    candidates = [base]
    if len(parts) > 2:
        candidates.append(".".join(parts[-2:]))
    # Preserve order but remove duplicates
    seen: set[str] = set()
    for cand in candidates:
        if cand and cand not in seen:
            seen.add(cand)
            yield cand


@dataclass(frozen=True)
class ManifestRecord:
    domain: str
    path: Path
    data: Dict[str, object]

    @property
    def selectors(self) -> Dict[str, object]:
        raw = self.data.get("selectors") if isinstance(self.data, dict) else None
        return raw or {}


def _manifest_path(manifest_dir: Path, candidate: str) -> Path:
    return manifest_dir / f"{candidate}.json"


def _resolve_dir(manifest_dir: str | os.PathLike[str] | None) -> Path:
    path = Path(manifest_dir) if manifest_dir else Path()
    if not path.is_absolute():
        path = path.expanduser().resolve()
    return path


@lru_cache(maxsize=64)
def load_manifest(domain: str, manifest_dir: str | os.PathLike[str]) -> Optional[ManifestRecord]:
    root = _resolve_dir(manifest_dir)
    for candidate in _candidate_filenames(domain):
        path = _manifest_path(root, candidate)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            return ManifestRecord(domain=candidate, path=path, data=data)
    return None


def require_manifest_dir(manifest_dir: str | os.PathLike[str]) -> Path:
    """Return the resolved manifest directory or raise a helpful error."""

    path = _resolve_dir(manifest_dir)
    if not path.exists():
        raise FileNotFoundError(f"Manifest directory not found: {path}")
    if not any(path.glob("*.json")):
        raise FileNotFoundError(f"Manifest directory is empty: {path}")
    return path


__all__ = ["ManifestRecord", "load_manifest", "require_manifest_dir"]
