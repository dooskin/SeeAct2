import sys
from pathlib import Path

import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_shopping_flow_smoke(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    import importlib
    cli_mod = importlib.import_module("seeact.seeact")

    class DummyAgent:
        def __init__(self, *a, **k):
            self.config = {"agent": {"max_auto_op": 10}}
            self.complete_flag = False
            self.final_result = None
            self._predictions = [
                {"action": "TYPE", "element": {"description": "Search"}, "value": "running shoes"},
                {"action": "PRESS ENTER", "element": None, "value": None},
                {"action": "CLICK", "element": {"description": "Product tile"}, "value": None},
                {"action": "TERMINATE", "element": None, "value": None},
            ]

        async def start(self, *a, **k):
            return None

        def change_task(self, *a, **k):
            return None

        async def predict(self):
            if not self._predictions:
                self.complete_flag = True
                return None
            return self._predictions.pop(0)

        async def perform_action(self, *, action_name=None, **_):
            if action_name == "TERMINATE":
                self.complete_flag = True
                self.final_result = {"status": "ok"}
            return None

        async def stop(self):
            return None

    monkeypatch.setattr("seeact.seeact.SeeActAgent", DummyAgent, raising=True)

    cfg = {
        "basic": {
            "save_file_dir": str(tmp_path),
            "default_task": "Find running shoes",
            "default_website": "https://example.com/",
        },
    }
    tasks = [
        {
            "task_id": "shopping",
            "confirmed_task": cfg["basic"]["default_task"],
            "website": cfg["basic"]["default_website"],
        }
    ]

    results = await cli_mod.run_with_config(cfg, tasks, max_steps=10)
    assert len(results) == 1
    assert results[0].success is True
    assert results[0].steps == 4
    assert results[0].result_payload == {"status": "ok"}
