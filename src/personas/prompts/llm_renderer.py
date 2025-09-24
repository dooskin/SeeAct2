from __future__ import annotations

from typing import Any, Dict, Optional

import os

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - import surface handled in tests via monkeypatch
    OpenAI = None  # type: ignore


SYSTEM_PROMPT = (
    "You generate UXAgent-style shopper prompts. Output exactly the sections and "
    "constraints. Keep one action per step. Do not invent UI. Preserve the provided "
    "site domain verbatim."
)


REQUIRED_TOKENS = (
    "[SYSTEM]",
    "[USER]",
    "# SHOP_PERSONA",
    "# GOAL",
    "# ACTIONS",
    "# POLICY",
    "# STOP",
    "# OUTPUT",
)


def _fmt_pct(x: float) -> str:
    try:
        return f"{100.0*float(x):.1f}%"
    except Exception:
        return "0.0%"


def _build_user_content(persona: Dict[str, Any], site_domain: str, vocab: Optional[Dict[str, Any]]) -> str:
    m = persona.get("metrics", {}) or {}
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
        "collections_csv": ", ".join((vocab or {}).get("collections", [])[:20]),
        "filters_csv": ", ".join((vocab or {}).get("filters", [])[:20]),
        "ctas_csv": ", ".join((vocab or {}).get("ctas", [])[:20]),
    }
    # Provide a strict skeleton to minimize drift
    out = (
        "[SYSTEM]\n"
        "You are a simulated online shopper. Act through a browser by issuing short, explicit actions.\n"
        "Follow the sections and rules exactly. No meta commentary. Stay terse. Do not invent UI.\n"
        "Sections: SHOP_PERSONA / GOAL / ACTIONS / POLICY / STOP / OUTPUT.\n\n"
        "[USER]\n"
        "# SHOP_PERSONA\n"
        f"persona_id: {tokens['persona_id']}\n"
        f"profile: {tokens['newVsReturning']} {tokens['device']} user on {tokens['operatingSystem']}, {tokens['gender']}, {tokens['userAgeBracket']}, from {tokens['geo']}, source={tokens['source']}, intent={tokens['intent']}\n"
        f"metrics: cr≈{tokens['cr_pct']}  bounce≈{tokens['bounce_pct']}  dwell≈{tokens['dwell_sec']}s  backtrack≈{tokens['backtrack_pct']}  form_err≈{tokens['form_err_pct']}\n\n"
        "# GOAL\n"
        f"Navigate {tokens['site_domain']} and add one plausible product to cart. Do not pay. Return price/variant/subtotal.\n\n"
        "# ACTIONS  (only these verbs)\n"
        "OPEN(url) | CLICK(text|selector) | TYPE(selector, text) | ENTER | SCROLL(dir|amount) | FILTER(name=value) | ADD_TO_CART | VIEW_CART\n"
        "Each step <= 1 action. Prefer exact on-screen labels.\n\n"
        "# POLICY\n"
        "- hot intent: jump to search or “Buy Now” paths; minimize dithering.\n"
        "- warm intent: compare 2–3 items; skim specs; choose one.\n"
        "- cold intent: skim a collection; may bounce quickly after one PDP.\n"
        "- returning: reuse obvious nav routes; accept cookies quickly.\n"
        "- mobile: keep steps minimal; avoid opening many tabs.\n"
        "- prefer site-native CTAs and filter labels verbatim.\n\n"
    )
    if vocab:
        out += (
            "# SITE_VOCAB\n"
            f"collections: {tokens['collections_csv']}\n"
            f"filters: {tokens['filters_csv']}\n"
            f"ctas: {tokens['ctas_csv']}\n\n"
        )
    out += (
        "# STOP\n"
        "Stop when item is in cart OR you’re stuck after 2 failed attempts. Never proceed to payment.\n\n"
        "# OUTPUT  (strict JSON)\n"
        '{"persona_id":"' + tokens["persona_id"] + '","pdp_url":"...","_title":"...","price":"...","variant":"...","subtotal":"...","steps":N}'
        "\n"
    )
    return out


def _normalize_response(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    for quote in ("'''", '"""'):
        if stripped.startswith(quote) and stripped.endswith(quote):
            stripped = stripped[len(quote) : -len(quote)].strip()
    if stripped.startswith("\"") and stripped.endswith("\""):
        stripped = stripped[1:-1].strip()
    return stripped


def _is_valid_prompt(text: str, site_domain: str) -> bool:
    if not text:
        return False
    for token in REQUIRED_TOKENS:
        if token not in text:
            return False
    if site_domain and site_domain.lower() not in text.lower():
        return False
    return True


def _fallback_prompt(
    persona: Dict[str, Any],
    site_domain: str,
    vocab: Optional[Dict[str, Any]],
    user_content: str,
) -> str:
    try:
        from personas.prompts.generator import render_shop_browse_prompt  # type: ignore

        return render_shop_browse_prompt(persona, site_domain, vocab=vocab)
    except Exception:  # pragma: no cover - fallback safeguard
        return user_content


def generate_prompt_with_llm(
    persona: Dict[str, Any],
    site_domain: str,
    *,
    vocab: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    base_url: Optional[str] = None,
    max_tokens: int = 900,
) -> str:
    """
    Use an LLM to render the full UXAgent-style prompt text.
    Falls back to deterministic skeleton if the LLM client is unavailable.
    """
    user_content = _build_user_content(persona, site_domain, vocab)

    if OpenAI is None:  # pragma: no cover
        # Minimal, deterministic fallback
        return _fallback_prompt(persona, site_domain, vocab, user_content)

    # Initialize client (auth comes from OPENAI_API_KEY environment)
    try:
        client = OpenAI(base_url=base_url) if base_url else OpenAI()
        mdl = model or os.getenv("PERSONAS_LLM_MODEL") or "gpt-4o-mini"
        resp = client.chat.completions.create(
            model=mdl,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Render the prompt text using the exact sections above. "
                        "Use only the allowed ACTION verbs. Keep the JSON OUTPUT schema unchanged.\n\n" + user_content
                    ),
                },
            ],
        )
        raw_text = resp.choices[0].message.content or ""
        text = _normalize_response(raw_text)
        if not _is_valid_prompt(text, site_domain):
            return _fallback_prompt(persona, site_domain, vocab, user_content)
        return text
    except Exception:
        # Fallback if the LLM call fails
        return _fallback_prompt(persona, site_domain, vocab, user_content)
