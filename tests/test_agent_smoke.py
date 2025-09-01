import os
import sys
from pathlib import Path

import pytest


def _ensure_pkg_path():
    # Add local package root so `import seeact` works without install
    repo_root = Path(__file__).resolve().parents[1]
    pkg_root = repo_root / "seeact_package"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))


@pytest.mark.smoke
def test_seeact_agent_initialization_and_prompts(monkeypatch, tmp_path):
    # Skip if Playwright is not available (import-level dep in agent module)
    pytest.importorskip("playwright.async_api")

    _ensure_pkg_path()

    # Import after ensuring path and skip
    import importlib
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
