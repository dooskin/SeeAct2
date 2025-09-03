import asyncio
import os
import sys
import types
from pathlib import Path

import pytest


def _ensure_pkg_path():
    repo_root = Path(__file__).resolve().parents[1]
    pkg_root = repo_root / "seeact_package"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))


def _sample_html():
    return """
    <!doctype html>
    <html>
      <head><meta charset="utf-8"><title>Hybrid Test</title></head>
      <body>
        <header>
          <a href="#home" id="home-link">Home</a>
          <a href="#about" id="about-link">About</a>
          <button id="signup">Sign up</button>
        </header>
        <main>
          <form>
            <label for="q">Search</label>
            <input id="q" type="text" placeholder="Search..." />
            <select id="category">
              <option>All</option>
              <option>Articles</option>
              <option>Videos</option>
            </select>
          </form>
        </main>
      </body>
    </html>
    """


@pytest.mark.integration
def test_hybrid_integration_screenshot_and_dom(tmp_path):
    # Require OpenAI key and Playwright for this test
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Set OPENAI_API_KEY to run hybrid integration test")

    pytest.importorskip("playwright.async_api")

    _ensure_pkg_path()

    async def run():
        from playwright.async_api import async_playwright
        # Ensure required deps exist for integration path
        pytest.importorskip("backoff")
        pytest.importorskip("litellm")
        # Provide toml stub if missing to allow importing seeact without full deps
        if "toml" not in sys.modules:
            sys.modules["toml"] = types.SimpleNamespace(
                load=lambda *a, **k: {},
                dump=lambda *a, **k: None,
                TomlDecodeError=Exception,
            )
        from seeact.agent import SeeActAgent
        from seeact.demo_utils.browser_helper import get_interactive_elements_with_playwright
        from seeact.demo_utils.format_prompt import format_choices

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(viewport={"width": 1280, "height": 720})
            page = await context.new_page()

            await page.set_content(_sample_html())
            viewport = await page.viewport_size

            elements = await get_interactive_elements_with_playwright(page, viewport)
            assert elements, "No interactive elements discovered from DOM"

            # Build prompt using DOM choices and pass screenshot to the model
            choices = format_choices(elements)
            agent = SeeActAgent(
                save_file_dir=str(tmp_path),
                grounding_strategy="text_choice_som",
                model="gpt-4o-mini",
            )

            prompt = agent.generate_prompt(task="Open the About section", previous=[], choices=choices)
            screenshot_path = Path(tmp_path) / "hybrid.png"
            await page.screenshot(path=str(screenshot_path))

            # Two-stage generation with both screenshot and DOM-based choices
            output0 = agent.engine.generate(
                prompt=prompt,
                image_path=str(screenshot_path),
                turn_number=0,
                max_new_tokens=64,
                temperature=0,
            )
            assert isinstance(output0, str) and len(output0) > 0

            output1 = agent.engine.generate(
                prompt=prompt,
                image_path=str(screenshot_path),
                turn_number=1,
                ouput_0=output0,
                max_new_tokens=64,
                temperature=0,
            )
            assert isinstance(output1, str) and len(output1) > 0

            await context.close()
            await browser.close()

    asyncio.run(run())
