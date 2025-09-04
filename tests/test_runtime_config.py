import os
import sys
import types
import asyncio
import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_runtime_cdp_env_expansion_and_connect(monkeypatch, tmp_path):
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
            # Emit 'page' event to set agent.page
            if self._page_handler:
                await self._page_handler(self._page)
            return self._page

    class DummyBrowser:
        def __init__(self):
            self.contexts = [DummyContext()]

    class DummyChromium:
        def __init__(self):
            self.calls = []

        async def connect_over_cdp(self, url, headers=None):
            self.calls.append((url, headers or {}))
            return DummyBrowser()

    class DummyAP:
        def __init__(self):
            self.chromium = DummyChromium()

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
    from seeact.agent import SeeActAgent  # type: ignore

    # Set env for expansion
    os.environ["BROWSERBASE_CDP_URL"] = "wss://example.browserbase/ws"
    os.environ["BROWSERBASE_API_KEY"] = "secret-token"

    agent = SeeActAgent(save_file_dir=str(tmp_path), model="gpt-4o")
    agent.config["runtime"] = {
        "provider": "cdp",
        "cdp_url": "${BROWSERBASE_CDP_URL}",
        "headers": {"Authorization": "Bearer ${BROWSERBASE_API_KEY}"},
    }

    # Start should call connect_over_cdp with expanded env values
    await agent.start()

    # Validate that our dummy chromium recorded the connection with expanded values
    chromium = agent.playwright.chromium  # type: ignore[attr-defined]
    assert chromium.calls, "Expected connect_over_cdp to be called"
    url, headers = chromium.calls[0]
    assert url == os.environ["BROWSERBASE_CDP_URL"]
    assert headers.get("Authorization") == f"Bearer {os.environ['BROWSERBASE_API_KEY']}"
