import sys
import types
import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_runtime_local_calls_local_launch(monkeypatch, tmp_path):
    # Stub playwright.async_api before importing agent module
    dummy_pa = types.SimpleNamespace()

    class DummyPage:
        def on(self, *a, **k):
            pass

        async def evaluate(self, *a, **k):
            return None

        async def goto(self, *a, **k):
            return None

    class DummyContext:
        def __init__(self):
            self._page = DummyPage()
            self._page_handler = None

        def on(self, event, handler):
            if event == "page":
                self._page_handler = handler

        async def new_page(self):
            if self._page_handler:
                await self._page_handler(self._page)
            return self._page

    class DummyBrowser:
        def __init__(self):
            self.contexts = []

    class DummyAP:
        def __init__(self):
            self.chromium = types.SimpleNamespace()

        async def start(self):
            return self

    dummy_pa.async_playwright = lambda: DummyAP()
    dummy_pa.Locator = object
    sys.modules.setdefault("playwright", types.SimpleNamespace(async_api=dummy_pa))
    sys.modules.setdefault("playwright.async_api", dummy_pa)

    # Import agent after stubbing
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    pkg_root = repo_root / "seeact_package"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    import importlib
    agent_module = importlib.import_module("seeact.agent")  # type: ignore

    # Monkeypatch local launch to assert it is called
    local_called = {"flag": False}

    async def fake_normal_launch_async(playwright, headless=False, args=None):
        local_called["flag"] = True
        return DummyBrowser()

    async def fake_normal_new_context_async(browser, **kwargs):
        return DummyContext()

    monkeypatch.setattr(agent_module, "normal_launch_async", fake_normal_launch_async, raising=True)
    monkeypatch.setattr(agent_module, "normal_new_context_async", fake_normal_new_context_async, raising=True)

    agent = agent_module.SeeActAgent(save_file_dir=str(tmp_path), model="gpt-4o")
    # No runtime specified -> defaults to local path
    await agent.start(headless=True, args=[])

    assert local_called["flag"], "Expected local normal_launch_async to be called for local provider"
