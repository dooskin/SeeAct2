import os
import sys
import types
import asyncio
from pathlib import Path

import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_shopping_flow_smoke(monkeypatch, tmp_path):
    # Make package importable
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    import importlib
    cli_mod = importlib.import_module("seeact.seeact")

    # Stub Playwright async context
    class DummyElement:
        async def click(self, *a, **k):
            return None
        async def clear(self, *a, **k):
            return None
        async def fill(self, *a, **k):
            return None
        async def press_sequentially(self, *a, **k):
            return None
        async def scroll_into_view_if_needed(self, *a, **k):
            return None
        async def hover(self, *a, **k):
            return None
        async def evaluate(self, *a, **k):
            return None

    class DummyKeyboard:
        async def press(self, *a, **k):
            return None

    class DummyPage:
        url = "about:blank"
        viewport_size = {"width": 800, "height": 600}
        keyboard = DummyKeyboard()
        async def goto(self, *a, **k):
            return None
        async def bring_to_front(self):
            return None
        async def title(self):
            return "Dummy"
        async def evaluate(self, expr):
            # scrollHeight
            return 1200
        async def screenshot(self, *a, **k):
            # ensure file path exists
            path = k.get("path") or (a[0] if a else None)
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b"\xFF\xD8\xFF\xD9")
            return None
        async def wait_for_load_state(self, *a, **k):
            return None

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

    # Avoid real network calls
    os.environ.setdefault("OPENAI_API_KEY", "sk-test")

    # Stub overlay dismiss to no-op
    async def _no_overlay(page, max_clicks=3):
        return 0
    monkeypatch.setattr(cli_mod, "auto_dismiss_overlays", _no_overlay, raising=True)

    # Prepare staged elements across calls
    staged = []
    # Stage 0: Google with a search input
    staged.append([
        {
            "center_point": (0.5, 0.2),
            "description": "Search q",
            "tag_with_role": "input",
            "box": [0, 0, 100, 20],
            "selector": DummyElement(),
            "tag": "input",
            "outer_html": "<input name=\"q\" placeholder=\"Search\">",
        }
    ])
    # Stage 1: results page (press enter action, element list can be anything)
    staged.append([
        {
            "center_point": (0.4, 0.3),
            "description": "Results container",
            "tag_with_role": "div",
            "box": [0, 0, 600, 400],
            "selector": DummyElement(),
            "tag": "div",
            "outer_html": "<div id=\"results\"></div>",
        }
    ])
    # Stage 2: official site link
    staged.append([
        {
            "center_point": (0.3, 0.3),
            "description": "On Running Official Site",
            "tag_with_role": "a",
            "box": [0, 0, 100, 20],
            "selector": DummyElement(),
            "tag": "a",
            "outer_html": "<a href=\"https://www.on.com/en-us/\">On Running Official Site</a>",
        }
    ])
    # Stage 3: terminate
    staged.append([
        {
            "center_point": (0.6, 0.8),
            "description": "Footer",
            "tag_with_role": "div",
            "box": [0, 0, 100, 20],
            "selector": DummyElement(),
            "tag": "div",
            "outer_html": "<div>Footer</div>",
        }
    ])

    call_count = {"n": 0}

    async def _elements(page, viewport=None):
        idx = min(call_count["n"], len(staged) - 1)
        call_count["n"] += 1
        return staged[idx]

    monkeypatch.setattr(cli_mod, "get_interactive_elements_with_playwright", _elements, raising=True)

    # Stub model with deterministic dialogue
    class StubEngine:
        def __init__(self, **kwargs):
            self.calls = 0
        def generate(self, prompt=None, image_path=None, turn_number=0, ouput_0=None, **kwargs):
            # Sequence: TYPE query, PRESS ENTER, CLICK link, TERMINATE
            script = [
                ("plan", "Click the search input and type query"),
                ("ground", "ELEMENT: A\nACTION: TYPE\nVALUE: On men's black running shoes size 11"),
                ("plan", "Press Enter to submit the search"),
                ("ground", "ACTION: PRESS ENTER\nVALUE: None"),
                ("plan", "Click link 'On Running Official Site'"),
                ("ground", "ELEMENT: A\nACTION: CLICK\nVALUE: None"),
                ("plan", "Terminate test"),
                ("ground", "ACTION: TERMINATE\nVALUE: None"),
            ]
            i = self.calls
            self.calls += 1
            kind, text = script[i % len(script)]
            return text

    monkeypatch.setattr(cli_mod, "OpenaiEngine", StubEngine, raising=True)

    # Minimal config
    cfg = {
        "basic": {
            "is_demo": True,
            "save_file_dir": str(tmp_path),
            "default_task": "On men's black running shoes size 11",
            "default_website": "https://www.google.com/",
        },
        "experiment": {
            "task_file_path": str(tmp_path / "tasks.json"),
            "overwrite": True,
            "top_k": 50,
            "fixed_choice_batch_size": 10,
            "dynamic_choice_batch_size": 0,
            "max_continuous_no_op": 2,
            "max_op": 6,
            "highlight": False,
            "monitor": False,
            "dev_mode": False,
            "include_dom_in_choices": False,
        },
        "openai": {"rate_limit": -1, "model": "gpt-4o", "temperature": 0},
        "oss_model": {},
        "playwright": {
            "save_video": False,
            "tracing": False,
            "locale": "en-US",
            "geolocation": {"longitude": 0, "latitude": 0},
            "viewport": {"width": 800, "height": 600},
            "trace": {"screenshots": False, "snapshots": False, "sources": False},
        },
        "runtime": {"provider": "local"},
    }

    base_dir = str(Path(cli_mod.__file__).resolve().parent)
    await cli_mod.main(cfg, base_dir)

