import sys
import types
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
pkg_root = repo_root / "src"
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from seeact.demo_utils.browser_helper import normal_launch_async


@pytest.mark.asyncio
async def test_normal_launch_respects_headless(monkeypatch):
    class DummyChromium:
        def __init__(self):
            self.calls = []

        async def launch(self, **kwargs):
            self.calls.append(kwargs)
            return "browser"

    dummy_playwright = types.SimpleNamespace(chromium=DummyChromium())

    browser = await normal_launch_async(dummy_playwright, headless=True, args=["--foo"])

    assert browser == "browser"
    assert dummy_playwright.chromium.calls
    call = dummy_playwright.chromium.calls[0]
    assert call["headless"] is True
    assert call["args"] == ["--foo"]


@pytest.mark.asyncio
async def test_normal_launch_defaults_headed():
    class DummyChromium:
        def __init__(self):
            self.calls = []

        async def launch(self, **kwargs):
            self.calls.append(kwargs)
            return "browser"

    dummy_playwright = types.SimpleNamespace(chromium=DummyChromium())

    await normal_launch_async(dummy_playwright)
    call = dummy_playwright.chromium.calls[0]
    assert call["headless"] is False
