import importlib.util
import json
import sys
from pathlib import Path


def _fake_args(**kwargs):
    defaults = {
        "data_dir": None,
        "site_domain": "example.com",
        "ids": None,
        "ids_file": None,
        "size": None,
        "include_vocab": False,
        "manifest_dir": None,
        "use_manifest_taxonomy": True,
        "use_llm": False,
        "llm_model": None,
        "llm_temperature": 0.2,
        "llm_base_url": None,
        "llm_max_tokens": 900,
        "out_dir": None,
    }
    defaults.update(kwargs)
    return type("Args", (), defaults)


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


def test_generate_prompts_merges_manifest_vocab(monkeypatch, tmp_path: Path):
    _ensure_pkg()
    pool_entry = {
        "persona_id": "pid1",
        "intent": "hot",
    }
    pool_path = tmp_path / "master_pool.jsonl"
    pool_path.write_text(json.dumps(pool_entry) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "personas.cli._load_pool",
        lambda _data_dir: [pool_entry],
    )

    monkeypatch.setattr(
        "personas.cli.prompt_vocab_from_taxonomy",
        lambda domain, intent, cache_dir=None: {
            "collections": ["Collection grid"],
            "filters": ["Filter controls"],
            "ctas": ["Add to Cart button"],
        },
    )

    args = _fake_args(
        data_dir=str(tmp_path),
        ids=["pid1"],
        out_dir=str(tmp_path / "prompts"),
    )

    from personas.cli import cmd_generate_prompts

    result = cmd_generate_prompts(args)
    assert result == 0
    prompt_path = Path(args.out_dir) / "shop_prompt_pid1.py"
    text = prompt_path.read_text(encoding="utf-8")
    assert "Collection grid" in text
    assert "Add to Cart button" in text
