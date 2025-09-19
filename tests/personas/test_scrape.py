import sys
from pathlib import Path


def _ensure_pkg():
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def test_extract_vocab_from_sample_html(monkeypatch):
    _ensure_pkg()
    from personas.scrape.shopify_scraper import scrape_shopify_vocab

    html_home = """
    <html><body>
      <a href="/collections/men">Men</a>
      <a href="/collections/women">Women</a>
      <a href="/products/shoe-1">Shoe 1</a>
    </body></html>
    """
    html_coll = """
    <html><body>
      <label>Size</label>
      <label>Color</label>
      <button>Add to Cart</button>
      <a href="/products/shoe-2">Shoe 2</a>
    </body></html>
    """
    pages = {
        "https://shop.example/": html_home,
        "https://shop.example/collections/men": html_coll,
    }

    def fake_get(url, headers=None, timeout=10):
        class R:
            def __init__(self, text):
                self.status_code = 200
                self.text = text
        # robots
        if url.endswith("/robots.txt"):
            return R("User-agent: *\nAllow: /\n")
        return R(pages.get(url, ""))

    monkeypatch.setenv("SCRAPER_IGNORE_ROBOTS", "1")
    monkeypatch.setattr("personas.scrape.shopify_scraper.requests.get", fake_get)
    vocab = scrape_shopify_vocab("https://shop.example/", max_pages=3)
    assert "collections" in vocab and vocab["collections"]
    assert any("Men" in c for c in vocab["collections"])  # extracted
    assert any("Add to Cart" in c for c in vocab["ctas"])  # CTA dedup
    assert any("Size" in f for f in vocab["filters"])  # filters present

