import base64
import os
import sys
from pathlib import Path

import pytest


def _ensure_pkg_path():
    repo_root = Path(__file__).resolve().parents[1]
    pkg_root = repo_root / "seeact_package"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))


@pytest.mark.integration
def test_openai_engine_generate_with_image(tmp_path):
    # Require OPENAI_API_KEY for this integration test
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("Set OPENAI_API_KEY to run OpenAI integration test")

    # Avoid importing playwright; use inference engine directly
    pytest.importorskip("litellm")

    # Ensure local package import without install
    _ensure_pkg_path()

    # Lazy import to avoid optional deps
    from seeact.demo_utils.inference_engine import engine_factory

    # Write a tiny 1x1 JPEG to disk
    tiny_jpg_b64 = (
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAQEBAQEA8PDw8PEA8QEA8PDw8PFREWFhURFRUYHSggGBolHRUVITEhJSkrLi4uFx8zODMsNygtLisBCgoKDg0OGhAQGy0lICUtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAKgBLAMBIgACEQEDEQH/xAAbAAEAAgMBAQAAAAAAAAAAAAAABQYBBAcDAv/EADgQAAIBAgMFBQQIBgMAAAAAAAABAgMRBBIhMQVBUQYiYXGBE6GxwfAHIjJSYnLwFEOCkbLC8VOS/8QAGQEBAQEBAQEAAAAAAAAAAAAAAAECAwQF/8QAHBEBAQEBAQEBAQAAAAAAAAAAAAERAjEhQRIi/9oADAMBAAIRAxEAPwD9yIiAiIgIiICIiAiIgIiICIiAiU8o6Xo3k9pW3jzA7b0GqfueE6xG3bPkGfMZ75H13C1mXk4f8AjLBG+cbQG7oZ1w6z9oVHHK0bY+V8q2z7uaNF1oGq6qTjVblsQb5vO0z0r8Tq1M0cVn9b4Wytz00bM6U0b6+Zp8v8A9Z7vW4v1jS7k06t0J1Z9ePjY6m3dVg2nRCp5D0p8v9b7vW4v1jS8LrQ4q6Z2uHqCGRvox9c3M8dL6fmX//2Q=="
    )
    img_path = Path(tmp_path) / "tiny.jpg"
    img_path.write_bytes(base64.b64decode(tiny_jpg_b64))

    # Build engine and generate a tiny response
    engine = engine_factory(api_key=api_key, model="gpt-4o-mini")
    prompt = [
        "You are a test system.",
        "Respond with a short acknowledgment.",
        "N/A",
    ]
    out = engine.generate(prompt=prompt, image_path=str(img_path), turn_number=0, max_new_tokens=8, temperature=0)
    assert isinstance(out, str)
    assert len(out) > 0
