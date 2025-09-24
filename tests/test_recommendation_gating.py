import sys
from pathlib import Path


def _ensure_pkg():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_pkg()

from seeact.recommendations.gating import gate_recommendations


class DummyManifest:
    def __init__(self, selectors):
        self.data = {"selectors": selectors}


def test_gate_recommendations_blocks_missing_capabilities():
    manifest = DummyManifest({
        "pdp": {"add_to_cart": "button.add"},
        "cart": {"checkout": "button.checkout"},
    })
    recs = [
        {"id": "size_smoothing", "capabilities": ["variant"], "title": "Smooth size selector"},
        {"id": "checkout_stream", "capabilities": ["checkout"], "title": "Checkout flow"},
    ]
    allowed, blocked = gate_recommendations(recs, manifest)
    assert len(allowed) == 1
    assert allowed[0]["id"] == "checkout_stream"
    assert len(blocked) == 1
    assert blocked[0]["missing_capabilities"] == ["variant"]


def test_gate_recommendations_no_manifest_blocks_all():
    recs = [{"id": "search_opt", "capabilities": ["search"], "title": "Search module"}]
    allowed, blocked = gate_recommendations(recs, None)
    assert allowed == []
    assert blocked[0]["status"] == "blocked_manifest_capability"
