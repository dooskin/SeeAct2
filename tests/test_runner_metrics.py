import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from seeact.runner import _summarize_step_metrics


def test_summarize_step_metrics():
    metrics = [
        {"scan_ms": 5, "llm_ms": 10, "macro_used": False, "num_candidates": 3},
        {"scan_ms": 7, "llm_ms": 0, "macro_used": True, "num_candidates": 1},
    ]
    summary = _summarize_step_metrics(metrics)
    assert summary["steps"] == 2
    assert summary["total_scan_ms"] == 12
    assert summary["total_llm_ms"] == 10
    assert summary["macro_steps"] == 1
    assert summary["per_step"] == metrics


def test_summarize_step_metrics_empty():
    assert _summarize_step_metrics(None) is None
    assert _summarize_step_metrics([]) is None
