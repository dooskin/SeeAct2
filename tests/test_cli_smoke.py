import sys
from pathlib import Path

import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_cli_demo_import_and_minimal_run(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    import importlib

    cli_mod = importlib.import_module("seeact.seeact")

    class DummyAgent:
        def __init__(self, *a, **k):
            self.complete_flag = False
            self.config = {"agent": {"max_auto_op": 1}}
            self.final_result = {"status": "ok"}

        async def start(self, *a, **k):
            return None

        def change_task(self, *a, **k):
            return None

        async def predict(self):
            if self.complete_flag:
                return None
            self.complete_flag = True
            return {"action": "TERMINATE", "element": None, "value": None}

        async def perform_action(self, *a, **k):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(cli_mod, "SeeActAgent", DummyAgent, raising=True)

    cfg = {
        "basic": {
            "save_file_dir": str(tmp_path),
            "default_task": "Open homepage",
            "default_website": "https://example.com/",
        },
        "agent": {"max_auto_op": 1},
    }

    tasks = [{"task_id": "t1", "confirmed_task": "Open homepage", "website": "https://example.com/"}]

    results = await cli_mod.run_with_config(cfg, tasks, max_steps=1)
    assert len(results) == 1
    assert results[0].success is True
