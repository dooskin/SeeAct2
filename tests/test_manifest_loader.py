import json
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from seeact.manifest.loader import load_manifest, clear_cache, DEFAULT_CACHE_DIR


def test_load_manifest(tmp_path):
    clear_cache()
    domain = "example.com"
    cache_dir = tmp_path / "manifests"
    cache_dir.mkdir()
    manifest_path = cache_dir / f"{domain}.json"
    manifest_path.write_text(json.dumps({"selectors": {"search": {"input": "input[name=q]"}}, "scraped_at": "2024-01-01"}), encoding="utf-8")

    manifest = load_manifest(domain, cache_dir=cache_dir)
    assert manifest is not None
    assert manifest.domain == domain
    assert manifest.data["selectors"]["search"]["input"] == "input[name=q]"
    assert manifest.scraped_at == "2024-01-01"

    # Cached path used on second load
    manifest2 = load_manifest(domain, cache_dir=cache_dir)
    assert manifest2 is manifest


def test_load_manifest_missing(tmp_path):
    clear_cache()
    cache_dir = tmp_path / "manifests"
    manifest = load_manifest("missing.com", cache_dir=cache_dir)
    assert manifest is None
