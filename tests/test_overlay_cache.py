import sys
from pathlib import Path
import pytest

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from types import SimpleNamespace

from seeact.demo_utils.browser_helper import auto_dismiss_overlays, _clear_overlay_cache


class StubElement:
    def __init__(self, visible=True):
        self.visible = visible
        self.clicks = 0

    async def is_visible(self, timeout=None):
        return self.visible

    async def click(self, timeout=None):
        self.clicks += 1

    def locator(self, selector):
        return StubLocator([])


class StubLocator:
    def __init__(self, elements):
        self.elements = elements

    async def count(self):
        return len(self.elements)

    def nth(self, index):
        return self.elements[index]


class StubPage:
    def __init__(self, url, locator_map):
        self.url = url
        self.locator_map = locator_map
        self.requested = []

    def locator(self, selector):
        self.requested.append(selector)
        return self.locator_map.get(selector, StubLocator([]))

    async def wait_for_load_state(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_overlay_cache_reuses_selector():
    _clear_overlay_cache()
    element = StubElement()
    selector = 'button:has-text("Close")'
    page = StubPage("https://example.com", {selector: StubLocator([element])})

    clicks = await auto_dismiss_overlays(page, max_clicks=1)
    assert clicks == 1
    assert element.clicks == 1
    assert selector in page.requested

    # Second page should use cached selector first
    element2 = StubElement()
    page2 = StubPage("https://example.com", {selector: StubLocator([element2])})

    clicks2 = await auto_dismiss_overlays(page2, max_clicks=1)
    assert clicks2 == 1
    # Cached selector should be the first (and only) request
    assert page2.requested[0] == selector
    assert element2.clicks == 1
