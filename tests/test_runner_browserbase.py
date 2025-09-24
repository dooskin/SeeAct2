import json
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from seeact import runner


@pytest.mark.smoke
def test_browserbase_session_reuse(monkeypatch, tmp_path):
    # Create tasks file
    tasks = [
        {"task_id": "t1", "confirmed_task": "Task 1", "website": "https://example.com/"},
        {"task_id": "t2", "confirmed_task": "Task 2", "website": "https://example.com/"},
    ]
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(tasks), encoding="utf-8")

    # Minimal config
    cfg_text = f"""
[basic]
save_file_dir = "{tmp_path}"
default_task = "Demo task"
default_website = "https://example.com/"

[openai]
model = "gpt-4o"
rate_limit = -1
temperature = 0

[runtime]
provider = "browserbase"
project_id = "proj"

[runner]
concurrency = 1
max_retries = 0
task_timeout_sec = 5
metrics_dir = "{tmp_path / 'runs'}"
"""
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    # Monkeypatch Browserbase client
    create_calls = []
    close_calls = []

    def fake_resolve(project_id, api_key):
        return project_id or "proj", "bb-key"

    def fake_create_session(project_id, api_key, **kwargs):
        create_calls.append((project_id, api_key, kwargs))
        return "wss://stub", "session-1"

    def fake_close_session(session_id, api_key, **kwargs):
        close_calls.append((session_id, api_key, kwargs))

    monkeypatch.setattr("seeact.runtime.browserbase_client.resolve_credentials", fake_resolve, raising=True)
    monkeypatch.setattr("seeact.runtime.browserbase_client.create_session", fake_create_session, raising=True)
    monkeypatch.setattr("seeact.runtime.browserbase_client.close_session", fake_close_session, raising=True)

    # Dummy agent to avoid real browser work
    class DummyAgent:
        def __init__(self, config):
            self.config = config
            self.complete_flag = False
            self._step_metrics = [{"scan_ms": 1, "llm_ms": 2, "macro_used": False, "num_candidates": 1}]

        async def start(self, website=None):
            assert self.config.get("runtime", {}).get("cdp_url") == "wss://stub"

        def change_task(self, *a, **k):
            return None

        async def predict(self):
            self.complete_flag = True
            return None

        async def perform_action(self, *a, **k):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr("seeact.agent.SeeActAgent", DummyAgent, raising=True)

    # Run runner
    runner.main([
        "-c", str(cfg_path),
        "--tasks", str(tasks_path),
        "--metrics-dir", str(tmp_path / "runs"),
    ])

    assert len(create_calls) == 1
    assert len(close_calls) == 1

    # Metrics file should include step_metrics summary
    run_dirs = list((tmp_path / "runs").glob("run_*"))
    assert run_dirs
    metrics_files = [p / "metrics.jsonl" for p in run_dirs]
    contents = "\n".join(m.read_text(encoding="utf-8") for m in metrics_files)
    assert "step_metrics" in contents
