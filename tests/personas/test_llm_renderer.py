from __future__ import annotations

from pathlib import Path
import sys
from pathlib import Path as _P


def _ensure_pkg():
    repo_root = _P(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_pkg()

from personas.prompts.llm_renderer import generate_prompt_with_llm  # type: ignore
from personas.prompts.generator import write_prompt_module  # type: ignore


class _FakeResp:
    def __init__(self, text: str) -> None:
        class _Msg:
            def __init__(self, c: str) -> None:
                self.content = c

        class _Choice:
            def __init__(self, t: str) -> None:
                self.message = _Msg(t)

        self.choices = [_Choice(text)]


def _make_fake_openai(monkeypatch, text: str) -> None:
    class _FakeOpenAI:
        def __init__(self, base_url: str | None = None) -> None:  # noqa: ARG002
            self._text = text
            self.chat = self
            self.completions = self

        def create(self, **_: object) -> _FakeResp:  # pragma: no cover - shape shim
            return _FakeResp(self._text)

    import personas.prompts.llm_renderer as llm

    monkeypatch.setattr(llm, "OpenAI", _FakeOpenAI, raising=True)


def _demo_persona() -> dict[str, object]:
    return {
        "persona_id": "p1",
        "newVsReturning": "new",
        "device": "mobile",
        "operatingSystem": "iOS",
        "gender": "female",
        "userAgeBracket": "25-34",
        "geo": "US:CA",
        "source": "organic",
        "intent": "warm",
        "metrics": {"cr": 0.05, "bounce_rate": 0.4, "dwell_means": 30, "backtrack_rate": 0.1, "form_error_rate": 0.05},
    }


def test_generate_prompt_with_llm_invalid_falls_back(monkeypatch, tmp_path: Path) -> None:
    invalid_text = '{"persona_id":"p1"}'
    _make_fake_openai(monkeypatch, invalid_text)

    persona = _demo_persona()
    text = generate_prompt_with_llm(persona, "example.com", model="dummy", temperature=0.0)
    # Fallback should be deterministic template with required sections and domain
    for token in ("[SYSTEM]", "[USER]", "# SHOP_PERSONA", "# STOP", "# OUTPUT"):
        assert token in text
    assert "example.com" in text

    out = write_prompt_module(text, persona_id="p1", out_dir=str(tmp_path))
    p = Path(out)
    assert p.exists()
    assert "def get_prompt()" in p.read_text(encoding="utf-8")


def test_generate_prompt_with_llm_valid_passthrough(monkeypatch) -> None:
    valid_text = """[SYSTEM]\nYou are a simulated online shopper.\n\n[USER]\n# SHOP_PERSONA\npersona_id: p1\nprofile: new mobile user on iOS, female, 25-34, from US:CA, source=organic, intent=warm\n\n# GOAL\nNavigate example.com and add one plausible product to cart. Do not pay. Return price/variant/subtotal.\n\n# ACTIONS  (only these verbs)\nOPEN(url) | CLICK(text|selector) | TYPE(selector, text) | ENTER | SCROLL(dir|amount) | FILTER(name=value) | ADD_TO_CART | VIEW_CART\n\n# POLICY\n- warm intent: compare 2–3 items; skim specs; choose one.\n- mobile: keep steps minimal; avoid opening many tabs.\n\n# STOP\nStop when item is in cart OR you’re stuck after 2 failed attempts. Never proceed to payment.\n\n# OUTPUT  (strict JSON)\n{\"persona_id\":\"p1\",\"pdp_url\":\"...\",\"_title\":\"...\",\"price\":\"...\",\"variant\":\"...\",\"subtotal\":\"...\",\"steps\":N}\n""".strip()

    _make_fake_openai(monkeypatch, valid_text)

    persona = _demo_persona()
    text = generate_prompt_with_llm(persona, "example.com", model="dummy", temperature=0.0)
    assert text == valid_text
