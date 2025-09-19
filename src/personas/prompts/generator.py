from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _fmt_pct(x: float) -> str:
    try:
        return f"{100.0*float(x):.1f}%"
    except Exception:
        return "0.0%"


def render_shop_browse_prompt(persona: Dict[str, Any], site_domain: str, vocab: Optional[Dict[str, Any]] = None) -> str:
    tpl = _read_text(TEMPLATES_DIR / "shop_browse_prompt.txt")
    m = persona.get("metrics", {})
    tokens = {
        "persona_id": persona.get("persona_id", "unknown"),
        "newVsReturning": persona.get("newVsReturning", "unknown"),
        "device": persona.get("device", "unknown"),
        "operatingSystem": persona.get("operatingSystem", "unknown"),
        "gender": persona.get("gender", "unknown"),
        "userAgeBracket": persona.get("userAgeBracket", "unknown"),
        "geo": persona.get("geo", "unknown"),
        "source": persona.get("source", "unknown"),
        "intent": persona.get("intent", "cold"),
        "cr_pct": _fmt_pct(m.get("cr", 0.0)),
        "bounce_pct": _fmt_pct(m.get("bounce_rate", 0.0)),
        "dwell_sec": f"{float(m.get('dwell_means', 0.0) or 0.0):.0f}",
        "backtrack_pct": _fmt_pct(m.get("backtrack_rate", 0.0)),
        "form_err_pct": _fmt_pct(m.get("form_error_rate", 0.0)),
        "site_domain": site_domain,
        "collections_csv": "",
        "filters_csv": "",
        "ctas_csv": "",
    }
    if vocab:
        tokens.update({
            "collections_csv": ", ".join(vocab.get("collections", [])[:20]),
            "filters_csv": ", ".join(vocab.get("filters", [])[:20]),
            "ctas_csv": ", ".join(vocab.get("ctas", [])[:20]),
        })
    # Avoid Python format() clobbering literal JSON braces — do tokenwise replacement.
    out = tpl
    for k, v in tokens.items():
        out = out.replace("{" + k + "}", str(v))
    # Drop SITE_VOCAB block lines if no vocab provided
    if not vocab:
        lines = []
        skip = False
        for line in out.splitlines():
            if line.strip().startswith("# SITE_VOCAB"):
                skip = True
                continue
            if skip and line.strip().startswith("# STOP"):
                skip = False
                continue
            if not skip:
                lines.append(line)
        out = "\n".join(lines) + "\n"
    return out


def write_prompt_module(text: str, persona_id: str, out_dir: Optional[str] = None) -> str:
    base_dir = Path(os.getenv("PERSONAS_DATA_DIR") or out_dir or "data/personas/prompts")
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"shop_prompt_{persona_id}.py"
    body = (
        "# Generated UXAgent-style shop prompt\n"
        "def get_prompt():\n"
        "    return r'''" + text.replace("'''", "\\'\\'\\'") + "'''\n"
    )
    path.write_text(body, encoding="utf-8")
    return str(path)
