from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:  # Lazy import: personas package may be used without seeact installed
    from seeact.manifest import load_manifest, Manifest  # type: ignore
except Exception:  # pragma: no cover - fallback when seeact isn't installed
    Manifest = None  # type: ignore


@dataclass
class SiteTaxonomy:
    domain: str
    collections: List[str]
    filters: List[str]
    ctas: List[str]
    variants: List[str]
    search: List[str]

    def to_prompt_vocab(self, intent: str) -> Dict[str, List[str]]:
        intent = (intent or "").lower()
        collections = list(dict.fromkeys(self.collections))
        filters = list(dict.fromkeys(self.filters))
        ctas = list(dict.fromkeys(self.ctas))

        if intent == "hot":
            ctas = list(dict.fromkeys(ctas + self.variants))
        elif intent == "warm":
            filters = list(dict.fromkeys(filters + self.variants))
        elif intent == "cold":
            collections = list(dict.fromkeys(self.search + collections))
        else:
            # Unknown intent → highlight a balanced set
            filters = list(dict.fromkeys(filters + self.variants))
            collections = list(dict.fromkeys(self.search + collections))

        return {
            "collections": collections,
            "filters": filters,
            "ctas": ctas,
        }


def _safe_manifest(domain: str, cache_dir: Optional[Path]) -> Optional[Manifest]:
    if 'load_manifest' not in globals():  # pragma: no cover - safety when seeact absent
        return None
    try:
        return load_manifest(domain, cache_dir=cache_dir)
    except Exception:
        return None


def derive_site_taxonomy(domain: str, *, cache_dir: Optional[Path] = None) -> Optional[SiteTaxonomy]:
    manifest = _safe_manifest(domain, cache_dir)
    if not manifest:
        return None

    selectors = manifest.data.get("selectors", {}) or {}
    collections: List[str] = []
    filters: List[str] = []
    ctas: List[str] = []
    variants: List[str] = []
    search: List[str] = []

    coll = selectors.get("collections") or {}
    if coll.get("product_link"):
        collections.append("Product tiles")
    if coll.get("grid"):
        collections.append("Collection grid")
    if coll.get("list"):
        collections.append("Product list view")

    if selectors.get("filters"):
        filters.append("Filter controls")
    if coll.get("sort"):
        filters.append("Sort selector")

    pdp = selectors.get("pdp") or {}
    if pdp.get("add_to_cart"):
        ctas.append("Add to Cart button")
    if pdp.get("buy_now"):
        ctas.append("Buy Now button")

    cart = selectors.get("cart") or {}
    if cart.get("checkout"):
        ctas.append("Checkout CTA")

    if pdp.get("variant_widget"):
        variants.append("Variant selector")

    search_sel = selectors.get("search") or {}
    if search_sel.get("input"):
        search.append("Search bar")

    overlays = selectors.get("overlays") or {}
    if overlays.get("close_button"):
        ctas.append("Overlay close button")

    return SiteTaxonomy(
        domain=manifest.domain,
        collections=collections,
        filters=filters,
        ctas=ctas,
        variants=variants,
        search=search,
    )


def prompt_vocab_from_taxonomy(domain: str, intent: str, cache_dir: Optional[Path] = None) -> Optional[Dict[str, List[str]]]:
    taxonomy = derive_site_taxonomy(domain, cache_dir=cache_dir)
    if not taxonomy:
        return None
    return taxonomy.to_prompt_vocab(intent)

