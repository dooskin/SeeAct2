import importlib.util
import sys
from pathlib import Path


def _ensure_pkg():
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def test_prompt_module_generation(tmp_path: Path):
    _ensure_pkg()
    from personas.prompts import render_shop_browse_prompt, write_prompt_module
    persona = {
        "persona_id": "pid123",
        "device": "mobile",
        "operatingSystem": "iOS",
        "source": "organic",
        "userAgeBracket": "25-34",
        "newVsReturning": "new",
        "gender": "female",
        "geo": "US:CA",
        "intent": "warm",
        "metrics": {
            "cr": 0.032,
            "bounce_rate": 0.55,
            "dwell_means": 35.0,
            "backtrack_rate": 0.12,
            "form_error_rate": 0.06,
            "sessions": 100,
        },
    }
    text = render_shop_browse_prompt(persona, "example.com")
    assert "[SYSTEM]" in text and "[USER]" in text
    # Required sections
    for sec in ["SHOP_PERSONA", "GOAL", "ACTIONS", "POLICY", "STOP", "OUTPUT"]:
        assert f"{sec}" in text

    mod_path = write_prompt_module(text, persona["persona_id"], out_dir=str(tmp_path))
    assert Path(mod_path).exists()
    # Import and call get_prompt
    spec = importlib.util.spec_from_file_location("shop_prompt_pid123", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    out = mod.get_prompt()
    assert isinstance(out, str) and "persona_id: pid123" in out

