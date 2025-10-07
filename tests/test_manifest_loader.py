import json
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from seeact.utils.manifest_loader import load_manifest, require_manifest_dir


def test_load_manifest(tmp_path):
    domain = "example.com"
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / f"{domain}.json"
    manifest_path.write_text(json.dumps({"selectors": {"search": {"input": "input[name=q]"}}, "scraped_at": "2024-01-01"}), encoding="utf-8")

    load_manifest.cache_clear()  # type: ignore[attr-defined]
    manifest = load_manifest(domain, manifest_dir)
    assert manifest is not None
    assert manifest.domain == domain
    assert manifest.selectors["search"]["input"] == "input[name=q]"

    # Cached path used on second load
    manifest2 = load_manifest(domain, manifest_dir)
    assert manifest2 is manifest


def test_load_manifest_missing(tmp_path):
    load_manifest.cache_clear()  # type: ignore[attr-defined]
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest = load_manifest("missing.com", manifest_dir)
    assert manifest is None


def test_require_manifest_dir(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "example.com.json").write_text("{}", encoding="utf-8")
    resolved = require_manifest_dir(manifest_dir)
    assert resolved == manifest_dir.resolve()
