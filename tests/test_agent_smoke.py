import os
import sys
from pathlib import Path

import pytest
import types


def _ensure_pkg_path():
    # Add local package root so `import seeact` works without install
    repo_root = Path(__file__).resolve().parents[1]
    pkg_root = repo_root / "src"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))


@pytest.mark.smoke
def test_seeact_agent_initialization_and_prompts(monkeypatch, tmp_path):
    # Skip if Playwright is not available (import-level dep in agent module)
    pytest.importorskip("playwright.async_api")

    _ensure_pkg_path()

    # Import after ensuring path and skip
    import importlib
    # Provide a lightweight toml stub if toml isn't installed
    if "toml" not in sys.modules:
        sys.modules["toml"] = types.SimpleNamespace(
            load=lambda *a, **k: {},
            dump=lambda *a, **k: None,
            TomlDecodeError=Exception,
        )
    # Provide minimal stubs for optional deps to import module without installing all extras
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

    # Prefer real OpenAI engine if OPENAI_API_KEY is set; otherwise stub
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setattr(agent_module, "engine_factory", lambda **_: DummyEngine(), raising=True)

    # Create agent with local output dir and headless browser settings
    agent = agent_module.SeeActAgent(
        save_file_dir=str(tmp_path),
        headless=True,
        grounding_strategy="text_choice_som",
        model="gpt-4o",  # value irrelevant due to stubbed engine
    )

    # Basic properties
    assert agent.complete_flag is False
    assert Path(agent.main_path).exists()
    assert isinstance(agent.engine, DummyEngine)

    # Prompt generation
    prompts = agent.generate_prompt()
    assert isinstance(prompts, list)
    # text_choice_som yields [system, query, referring]
    assert len(prompts) == 3

    # Prompt update should be reflected
    assert agent.update_prompt_part("system_prompt", "SYSTEM_MARKER")
    prompts2 = agent.generate_prompt()
    assert "SYSTEM_MARKER" in prompts2[0]
