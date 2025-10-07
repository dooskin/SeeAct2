import json
import os
import sys
import types
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def _load_agent_module(monkeypatch):
    import importlib

    if "toml" not in sys.modules:
        sys.modules["toml"] = types.SimpleNamespace(
            load=lambda *a, **k: {},
            dump=lambda *a, **k: None,
            TomlDecodeError=Exception,
        )
    if "backoff" not in sys.modules:
        def _identity_deco(*args, **kwargs):
            def _wrap(fn):
                return fn
            return _wrap
        sys.modules["backoff"] = types.SimpleNamespace(on_exception=_identity_deco, expo=lambda *a, **k: None)
    if "dotenv" not in sys.modules:
        sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: None)
    if "openai" not in sys.modules:
        DummyErr = type("DummyErr", (Exception,), {})
        sys.modules["openai"] = types.SimpleNamespace(
            APIConnectionError=DummyErr,
            APIError=DummyErr,
            RateLimitError=DummyErr,
        )
    if "litellm" not in sys.modules:
        sys.modules["litellm"] = types.SimpleNamespace(
            completion=lambda **kwargs: types.SimpleNamespace(choices=[{"message": {"content": "ok"}}]),
            set_verbose=False,
        )
    if "requests" not in sys.modules:
        sys.modules["requests"] = types.SimpleNamespace(post=lambda **kwargs: types.SimpleNamespace(status_code=200, json=lambda: {"message": {"content": "ok"}}))

    agent_module = importlib.import_module("seeact.agent")

    class DummyEngine:
        def generate(self, *args, **kwargs):
            return "OK"

    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setattr(agent_module, "engine_factory", lambda **_: DummyEngine(), raising=True)

    return agent_module


def _base_agent_config(tmp_path: Path, input_info):
    return {
        "basic": {
            "save_file_dir": str(tmp_path),
            "default_task": "Test task",
            "default_website": "https://example.com/",
        },
        "agent": {
            "input_info": input_info,
        },
        "openai": {
            "model": "gpt-4o-mini",
            "rate_limit": -1,
            "temperature": 0,
        },
    }


@pytest.mark.parametrize("selector_key, expected", [
    ("search", "search input"),
    ("pdp", "Add to cart"),
])
def test_manifest_prompt_hint(monkeypatch, tmp_path, selector_key, expected):
    agent_module = _load_agent_module(monkeypatch)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    data = {
        "selectors": {
            "search": {"input": "input[name='q']"},
            "pdp": {"add_to_cart": "button.add-to-cart"},
        },
        "scraped_at": "2024-01-01",
    }
    (manifest_dir / "example.com.json").write_text(json.dumps(data), encoding="utf-8")

    config = _base_agent_config(tmp_path, input_info=["screenshot"])
    config["manifest"] = {"dir": str(manifest_dir)}
    agent = agent_module.SeeActAgent(config=config)
    agent._load_manifest_for_url("https://www.example.com/products/item")
    hint = agent._manifest_prompt_hint()
    assert hint is not None
    assert "input[name='q']" in hint or "button.add-to-cart" in hint
    prompts = agent.generate_prompt()
    assert any("Manifest hints" in p for p in prompts)


@pytest.mark.asyncio
async def test_macro_prefers_manifest(monkeypatch, tmp_path):
    agent_module = _load_agent_module(monkeypatch)
    config = _base_agent_config(tmp_path, input_info=[])
    agent = agent_module.SeeActAgent(config=config)

    agent._manifest_selectors = {"collections": {"product_link": "a.product"}}

    class StubLocator:
        def __init__(self, count=1):
            self._count = count

        async def count(self):
            return self._count

        async def is_visible(self, timeout=None):
            return self._count > 0

        async def inner_text(self, timeout=None):
            return "Product tile"

        async def bounding_box(self):
            return {"y": 100, "height": 50}

        def nth(self, idx):
            return self

        @property
        def first(self):
            return self

        def locator(self, selector):
            return StubLocator(0)

    class StubPage:
        url = "https://example.com/collections/shirts"

        def __init__(self):
            self.locators = {"a.product": StubLocator()}

        def locator(self, sel):
            return self.locators.get(sel, StubLocator(0))

        async def evaluate(self, *args, **kwargs):
            return {"w": 800, "h": 600}

    page = StubPage()
    agent.session_control['active_page'] = page

    action = await agent._macro_next_action()
    assert action["action"] == "CLICK"
    assert agent._manifest_step_used is True
