import os
import sys
import types
import pytest
import asyncio
from pathlib import Path


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_cli_demo_import_and_minimal_run(monkeypatch, tmp_path):
    # Ensure package path is importable
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Stub aioconsole ainput to immediately select defaults
    import importlib
    cli_mod = importlib.import_module("seeact.seeact")
    async def _ainput(prompt: str = "") -> str:
        return ""
    monkeypatch.setattr(cli_mod, "ainput", _ainput, raising=True)

    # Provide a minimal config derived from demo, but save outputs to tmp
    base_dir = (Path(cli_mod.__file__).resolve().parent)
    cfg = {
        "basic": {
            "is_demo": True,
            "save_file_dir": str(tmp_path),
            "default_task": "Open homepage",
            "default_website": "https://example.com/",
        },
        "experiment": {
            "task_file_path": str(tmp_path / "tasks.json"),
            "overwrite": True,
            "top_k": 50,
            "fixed_choice_batch_size": 10,
            "dynamic_choice_batch_size": 0,
            "max_continuous_no_op": 1,
            "max_op": 2,
            "highlight": False,
            "monitor": False,
            "dev_mode": False,
        },
        "openai": {"rate_limit": -1, "model": "gpt-4o", "temperature": 0},
        "oss_model": {},
        "playwright": {
            "save_video": False,
            "tracing": False,
            "locale": "en-US",
            "geolocation": {"longitude": 0, "latitude": 0},
            "viewport": {"width": 640, "height": 480},
            "trace": {"screenshots": True, "snapshots": True, "sources": False},
        },
        "runtime": {"provider": "cdp", "cdp_url": "wss://example"},
    }

    # Stub Playwright async context and connect_over_cdp
    class DummyPage:
        url = "about:blank"
        async def goto(self, *a, **k):
            return None
        async def title(self):
            return "Dummy"

    class DummyContext:
        def __init__(self):
            self.pages = []
        def on(self, *a, **k):
            return None
        async def new_page(self):
            p = DummyPage()
            self.pages.append(p)
            return p
        class tracing:
            @staticmethod
            async def start(*a, **k):
                return None
            @staticmethod
            async def start_chunk(*a, **k):
                return None
            @staticmethod
            async def stop_chunk(*a, **k):
                return None
        async def close(self):
            return None

    class DummyBrowser:
        def __init__(self):
            self.contexts = [DummyContext()]

    class DummyChromium:
        async def connect_over_cdp(self, url, headers=None):
            return DummyBrowser()
        async def launch(self, *a, **k):
            return DummyBrowser()

    class DummyAP:
        def __init__(self):
            self.chromium = DummyChromium()
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False

    dummy_pa = types.SimpleNamespace(async_playwright=lambda: DummyAP())
    sys.modules["playwright"] = types.SimpleNamespace(async_api=dummy_pa)
    sys.modules["playwright.async_api"] = dummy_pa

    # Avoid real LLM calls by setting a dummy OPENAI_API_KEY and relying on elements=[] fast path
    os.environ.setdefault("OPENAI_API_KEY", "sk-test")

    # Monkeypatch element discovery to return empty (fast exit)
    async def _no_elements(page):
        return []
    monkeypatch.setattr(cli_mod, "get_interactive_elements_with_playwright", _no_elements, raising=True)

    # Run the CLI main with our synthetic config; should not raise
    await cli_mod.main(cfg, str(base_dir))
