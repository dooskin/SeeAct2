from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from seeact.utils.manifest_loader import ManifestRecord


CAPABILITY_MAP = {
    "search": lambda selectors: bool((selectors.get("search") or {}).get("input")),
    "filters": lambda selectors: bool(selectors.get("filters") or (selectors.get("collections") or {}).get("sort")),
    "variant": lambda selectors: bool((selectors.get("pdp") or {}).get("variant_widget")),
    "add_to_cart": lambda selectors: bool((selectors.get("pdp") or {}).get("add_to_cart")),
    "checkout": lambda selectors: bool((selectors.get("cart") or {}).get("checkout")),
    "overlay_close": lambda selectors: bool((selectors.get("overlays") or {}).get("close_button")),
}


@dataclass
class Recommendation:
    id: str
    title: str
    capabilities: List[str]
    payload: Dict[str, object]


def _to_manifest_selectors(manifest: ManifestRecord | Dict[str, object] | None) -> Dict[str, object]:
    if manifest is None:
        return {}
    if isinstance(manifest, ManifestRecord):
        return manifest.selectors or {}
    data = getattr(manifest, "data", None)
    if isinstance(data, dict):
        return data.get("selectors", {}) or {}
    return manifest.get("selectors", {}) if isinstance(manifest, dict) else {}


def gate_recommendations(
    recommendations: Iterable[Dict[str, object]],
    manifest: Manifest | Dict[str, object] | None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    selectors = _to_manifest_selectors(manifest)
    allowed: List[Dict[str, object]] = []
    blocked: List[Dict[str, object]] = []

    for rec in recommendations:
        capabilities = [c.lower() for c in rec.get("capabilities", [])]
        missing = [cap for cap in capabilities if not CAPABILITY_MAP.get(cap, lambda _: False)(selectors)]
        rec_copy = dict(rec)
        if missing:
            rec_copy["status"] = "blocked_manifest_capability"
            rec_copy["missing_capabilities"] = missing
            blocked.append(rec_copy)
        else:
            rec_copy["status"] = "allowed"
            allowed.append(rec_copy)

    return allowed, blocked
