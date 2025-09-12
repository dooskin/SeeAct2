import json
import sys
from pathlib import Path

import pytest


@pytest.mark.smoke
def test_runner_smoke(monkeypatch, tmp_path: Path):
    # Create a tiny tasks file
    tasks = [
        {"task_id": "t1", "confirmed_task": "Open homepage", "website": "https://example.com/"},
        {"task_id": "t2", "confirmed_task": "Open homepage", "website": "https://example.com/"},
    ]
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(tasks), encoding="utf-8")

    # Create a minimal config TOML
    cfg = """
[basic]
save_file_dir = "../online_results"
default_task = "Demo task"
default_website = "https://example.com/"

[openai]
rate_limit = -1
model = "gpt-4o"
temperature = 0

[playwright]
save_video = false
tracing = false
locale = "en-US"
geolocation.longitude=0
geolocation.latitude=0
viewport.width = 640
viewport.height = 480
trace.screenshots = true
trace.snapshots = true
trace.sources = false

[runtime]
provider = "local"

[runner]
concurrency = 2
max_retries = 0
task_timeout_sec = 5
metrics_dir = "../runs"
"""
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(cfg, encoding="utf-8")

    # Monkeypatch SeeActAgent with a dummy that completes immediately
    class DummyAgent:
        def __init__(self, *a, **k):
            self.config = {"agent": {"max_auto_op": 1}}
            self.complete_flag = False

        async def start(self, *a, **k):
            return None

        def change_task(self, *a, **k):
            return None

        async def predict(self):
            # Complete on first predict
            self.complete_flag = True
            return None

        async def perform_action(self, *a, **k):
            return None

        async def stop(self):
            return None

    pkg_root = Path(__file__).resolve().parents[1] / "src"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    import importlib
    agent_mod = importlib.import_module("seeact.agent")
    monkeypatch.setattr(agent_mod, "SeeActAgent", DummyAgent, raising=True)

    # Import runner and execute
    from src import runner as runner_mod  # type: ignore
    # Run synchronously
    runner_mod.main(["-c", str(cfg_path), "--tasks", str(tasks_path), "--concurrency", "2", "--metrics-dir", str(tmp_path / "runs")])

    # Verify metrics JSONL created
    run_dirs = list((tmp_path / "runs").glob("run_*") )
    assert run_dirs, "No run directory created"
    metrics_files = [p / "metrics.jsonl" for p in run_dirs]
    assert any(m.exists() and m.stat().st_size > 0 for m in metrics_files)
