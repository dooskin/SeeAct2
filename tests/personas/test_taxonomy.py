import sys
from pathlib import Path


def _ensure_pkg():
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_pkg()

from personas.taxonomy import prompt_vocab_from_taxonomy


def test_site_taxonomy_prompt_vocab_hot_intent(monkeypatch):
    class DummyManifest:
        domain = "example.com"
        data = {
            "selectors": {
                "collections": {"product_link": "a[href*='/product']", "grid": "main .grid"},
                "pdp": {"add_to_cart": "button.add", "variant_widget": "[role='radio']"},
                "cart": {"checkout": "button.checkout"},
                "search": {"input": "input[type='search']"},
            }
        }

    monkeypatch.setattr("personas.taxonomy.load_manifest", lambda domain, cache_dir=None: DummyManifest())

    vocab = prompt_vocab_from_taxonomy("example.com", "hot")
    assert vocab
    assert "Add to Cart button" in vocab["ctas"]
    assert "Variant selector" in vocab["ctas"]
    assert "Product tiles" in vocab["collections"]


def test_site_taxonomy_prompt_vocab_cold_intent(monkeypatch):
    class DummyManifest:
        domain = "demo.com"
        data = {
            "selectors": {
                "collections": {"product_link": "a[href*='/product']"},
                "search": {"input": "input[name='q']"},
            }
        }

    monkeypatch.setattr("personas.taxonomy.load_manifest", lambda domain, cache_dir=None: DummyManifest())

    vocab = prompt_vocab_from_taxonomy("demo.com", "cold")
    assert vocab
    assert vocab["collections"][0] == "Search bar"
