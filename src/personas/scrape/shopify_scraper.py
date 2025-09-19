from __future__ import annotations

import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


CTA_PATTERNS = [
    r"add to cart",
    r"buy now",
    r"checkout",
    r"view cart",
    r"subscribe",
]


def _respect_robots() -> bool:
    return os.getenv("SCRAPER_IGNORE_ROBOTS") not in ("1", "true", "TRUE")


def _can_fetch(base: str) -> bool:
    if not _respect_robots():
        return True
    try:
        rp = requests.get(urljoin(base, "/robots.txt"), timeout=5)
        if rp.status_code != 200:
            return True
        text = rp.text.lower()
        # Very naive: disallow all? else allow
        return "disallow: /" not in text
    except Exception:
        return True


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def scrape_shopify_vocab(site: str, max_pages: int = 50, user_agent: str = "SeeAct2Bot/1.0") -> Dict[str, List[str]]:
    if not site.startswith("http"):
        site = "https://" + site
    if not _can_fetch(site):
        return {"collections": [], "products": [], "filters": [], "ctas": [], "misc_terms": [], "site_domain": urlparse(site).netloc}

    seen: Set[str] = set()
    q = deque([site])
    collections: Set[str] = set()
    products: Set[str] = set()
    filters: Set[str] = set()
    ctas: Set[str] = set()
    misc: Set[str] = set()

    headers = {"User-Agent": user_agent}
    n = 0
    while q and n < max_pages:
        url = q.popleft()
        if url in seen:
            continue
        seen.add(url)
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                continue
            html = r.text
        except Exception:
            continue
        soup = BeautifulSoup(html, "html.parser")

        # Collect links for BFS prioritizing collections/products
        for a in soup.find_all("a", href=True):
            href = a.get("href") or ""
            abs_url = urljoin(url, href)
            if urlparse(abs_url).netloc != urlparse(site).netloc:
                continue
            if any(seg in href for seg in ["/collections/", "/products/"]):
                q.append(abs_url)
            elif abs_url.startswith(site):
                q.append(abs_url)

        # Extract collections
        for a in soup.select("a[href*='/collections/']"):
            txt = _clean_text(a.get_text(" "))
            if txt:
                collections.add(txt)

        # Extract products (titles on PDP)
        for sel in ["h1", ".product-title", "meta[property='og:title']"]:
            for node in soup.select(sel):
                if node.name == "meta":
                    txt = node.get("content") or ""
                else:
                    txt = _clean_text(node.get_text(" "))
                if txt:
                    products.add(txt)

        # Extract filters: facet labels and common terms
        for lab in soup.find_all("label"):
            txt = _clean_text(lab.get_text(" "))
            if txt and any(k in txt.lower() for k in ["size", "color", "brand", "price"]):
                filters.add(txt)
        # Elements that look like facets (class contains 'facets') or have data-* attributes
        for node in soup.find_all(True):
            try:
                classes = node.get("class") or []
                has_facets_class = any("facets" in str(c).lower() for c in classes)
                has_data_attr = any(str(k).lower().startswith("data-") and "facet" in str(k).lower() for k in (node.attrs or {}).keys())
            except Exception:
                has_facets_class = False
                has_data_attr = False
            if not (has_facets_class or has_data_attr):
                continue
            txt = _clean_text(node.get_text(" "))
            if not txt:
                continue
            for part in re.split(r",|/|\|", txt):
                part = _clean_text(part)
                if part and len(part) <= 30 and any(k in part.lower() for k in ["size", "color", "brand", "price"]):
                    filters.add(part)

        # CTAs
        for btn in soup.find_all(["button", "a"]):
            txt = _clean_text(btn.get_text(" "))
            if not txt:
                continue
            low = txt.lower()
            if any(re.search(p, low) for p in CTA_PATTERNS):
                # Keep site casing, dedup case-insensitively
                if all(txt.lower() != c.lower() for c in ctas):
                    ctas.add(txt)

        # Misc terms
        for node in soup.find_all(text=True):
            s = _clean_text(str(node))
            low = s.lower()
            if any(k in low for k in ["free shipping", "returns", "points", "save "]):
                misc.add(s)

        n += 1
        time.sleep(0.2)

    domain = urlparse(site).netloc
    return {
        "collections": sorted(collections)[:200],
        "products": sorted(products)[:500],
        "filters": sorted(filters)[:200],
        "ctas": sorted(ctas, key=lambda x: x.lower())[:200],
        "misc_terms": sorted(misc)[:200],
        "site_domain": domain,
    }
