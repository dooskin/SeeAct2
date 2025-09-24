from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEVICE_MAP = {
    "tablet": "mobile",
    "mobile": "mobile",
    "desktop": "desktop",
}


def normalize_dims(cohort: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(cohort)
    # device_category -> device
    raw_dev = str(d.get("device_category") or d.get("device") or "").lower()
    device = DEVICE_MAP.get(raw_dev, "unknown")
    d["device"] = device
    # OS
    d["operating_system"] = map_os_family(d.get("operating_system"))
    # source
    d["source"] = map_source_taxonomy(d.get("session_source_medium") or d.get("source"))
    # geo bucket already provided (country[:state])
    geo = d.get("geo_bucket") or d.get("geo") or "unknown"
    d["geo"] = geo
    # Age bracket, new_vs_returning, gender passthrough with unknown default
    for k in ("user_age_bracket", "new_vs_returning", "gender"):
        v = d.get(k) or d.get(k.replace("user_", "")) or "unknown"
        d[k] = v
    return d


def map_os_family(os_name: Any) -> str:
    s = str(os_name or "").lower()
    if not s:
        return "unknown"
    if "ios" in s or "iphone" in s or "ipad" in s:
        return "iOS"
    if "android" in s:
        return "Android"
    if "windows" in s:
        return "Windows"
    if "mac" in s:
        return "macOS"
    if "linux" in s:
        return "Linux"
    return "unknown"


def map_source_taxonomy(src: Any) -> str:
    s = str(src or "").lower()
    if not s:
        return "unknown"
    if "shopping" in s:
        return "shopping"
    if any(k in s for k in ["cpc", "paid", "ppc", "affiliate", "ad"]):
        return "ads"
    if "social" in s:
        return "social"
    if "(direct)" in s or "direct" in s or "(none)" in s:
        return "direct"
    if "organic" in s or "search" in s:
        return "organic"
    return "referral"


def compute_unknown_share(rows: Iterable[Dict[str, Any]]) -> float:
    total_sessions = 0
    unknown_sessions = 0
    for r in rows:
        s = int(r.get("sessions", 0) or 0)
        total_sessions += s
        dims = (
            str(r.get("device")),
            str(r.get("source")),
            str(r.get("operating_system")),
            str(r.get("user_age_bracket")),
            str(r.get("new_vs_returning")),
            str(r.get("gender")),
            str(r.get("geo")),
        )
        if any(x == "unknown" for x in dims):
            unknown_sessions += s
    return (unknown_sessions / total_sessions) if total_sessions else 0.0


def _intent_from_metrics(cr: float, bounce_rate: float) -> str:
    # Deterministic thresholds from the lock sheet
    if cr >= 0.05 or (cr >= 0.03 and bounce_rate < 0.50):
        return "hot"
    if 0.015 <= cr < 0.05 and bounce_rate <= 0.70:
        return "warm"
    return "cold"


def _sha1_key(key_fields: Tuple[str, ...], window_end: str) -> str:
    h = hashlib.sha1()
    h.update("|".join(key_fields + (window_end,)).encode("utf-8"))
    return h.hexdigest()


def _merge_k_anon(items: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    # Items are assumed bucketed by (device, source, intent)
    # Iteratively merge until each item >= k sessions
    items = [dict(x) for x in items]
    changed = True
    while changed:
        changed = False
        small = [x for x in items if int(x.get("sessions", 0)) < k]
        if not small:
            break
        # Merge the smallest into the largest within bucket
        small.sort(key=lambda x: int(x.get("sessions", 0)))
        victim = small[0]
        # Find nearest-by-sessions (largest to absorb)
        candidates = [x for x in items if x is not victim]
        if not candidates:
            break
        target = max(candidates, key=lambda x: int(x.get("sessions", 0)))
        vs = int(victim.get("sessions", 0))
        ts = int(target.get("sessions", 0)) or 1
        total = vs + ts
        def wavg(a, b, wa, wb):
            return (a * wa + b * wb) / (wa + wb)
        for m in ("cr", "bounce_rate", "avg_dwell_sec", "backtrack_rate", "form_error_rate"):
            va = float(victim.get(m, 0) or 0)
            tb = float(target.get(m, 0) or 0)
            target[m] = wavg(tb, va, ts, vs)
        target["sessions"] = total
        # Remove victim
        items.remove(victim)
        changed = True
    return items


def _weighted_sample(ids: List[str], weights: List[float], k: int) -> List[str]:
    import random
    if not ids:
        return []
    total = sum(weights) or 1.0
    probs = [w / total for w in weights]
    return [random.choices(ids, weights=probs, k=1)[0] for _ in range(k)]


def build_master_pool(
    cohorts: Iterable[Dict[str, Any]],
    window_end: Optional[datetime] = None,
    k_anon: int = 50,
    unknown_drop_threshold: float = 0.7,
    pool_size: int = 1000,
) -> List[Dict[str, Any]]:
    """Normalize cohorts → personas with persona_id and metrics; then produce exactly 1000 entries.

    Returns a list of 1000 persona dicts (replicas allowed).
    """
    rows: List[Dict[str, Any]] = []
    # Normalize, compute metrics and intent
    for c in cohorts:
        n = normalize_dims(c)
        sessions = int(c.get("sessions", 0) or 0)
        conversions = int(c.get("conversions", 0) or 0)
        bounce_sessions = int(c.get("bounce_sessions", 0) or 0)
        avg_dwell_sec = float(c.get("avg_dwell_sec", 0) or 0)
        backtracks = int(c.get("backtracks", 0) or 0)
        form_errors = int(c.get("form_errors", 0) or 0)
        cr = conversions / sessions if sessions else 0.0
        bounce_rate = bounce_sessions / sessions if sessions else 0.0
        backtrack_rate = backtracks / sessions if sessions else 0.0
        form_error_rate = form_errors / sessions if sessions else 0.0
        intent = _intent_from_metrics(cr, bounce_rate)
        n.update({
            "sessions": sessions,
            "cr": cr,
            "bounce_rate": bounce_rate,
            "avg_dwell_sec": avg_dwell_sec,
            "backtrack_rate": backtrack_rate,
            "form_error_rate": form_error_rate,
            "intent": intent,
        })
        rows.append(n)

    # Unknown handling (drop fully-unknown ONLY if unknown_share > threshold)
    u_share = compute_unknown_share(rows)
    if u_share > unknown_drop_threshold:
        def fully_unknown(r: Dict[str, Any]) -> bool:
            dims = (
                r.get("device"), r.get("source"), r.get("operating_system"),
                r.get("user_age_bracket"), r.get("new_vs_returning"), r.get("gender"), r.get("geo")
            )
        
            return all(str(x) == "unknown" for x in dims)
        rows = [r for r in rows if not fully_unknown(r)]

    # Bucket by (device, source, intent) and enforce k-anon
    from collections import defaultdict
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[(r.get("device"), r.get("source"), r.get("intent"))].append(r)
    merged: List[Dict[str, Any]] = []
    for key, grp in buckets.items():
        # Merge within bucket iteratively
        merged_grp = _merge_k_anon(list(grp), k_anon)
        # Residual < k: fold into nearest-by-sessions
        small = [x for x in merged_grp if int(x.get("sessions", 0)) < k_anon]
        large = [x for x in merged_grp if int(x.get("sessions", 0)) >= k_anon]
        if small and large:
            largest = max(large, key=lambda x: int(x.get("sessions", 0)))
            for s in small:
                vs = int(s.get("sessions", 0))
                ts = int(largest.get("sessions", 0)) or 1
                total = vs + ts
                def wavg(a, b, wa, wb):
                    return (a * wa + b * wb) / (wa + wb)
                for m in ("cr", "bounce_rate", "avg_dwell_sec", "backtrack_rate", "form_error_rate"):
                    va = float(s.get(m, 0) or 0)
                    tb = float(largest.get(m, 0) or 0)
                    largest[m] = wavg(tb, va, ts, vs)
                largest["sessions"] = total
            merged_grp = [largest]
        merged.extend(merged_grp)

    # Build personas with persona_id and weight
    total_sessions = sum(int(x.get("sessions", 0) or 0) for x in merged) or 1
    window_end_iso = (window_end or datetime.now(timezone.utc)).isoformat()
    personas: List[Dict[str, Any]] = []
    for r in merged:
        key_fields = (
            str(r.get("device")), str(r.get("source")), str(r.get("operating_system")),
            str(r.get("user_age_bracket")), str(r.get("new_vs_returning")), str(r.get("gender")), str(r.get("geo")),
        )
        persona_id = _sha1_key(key_fields, window_end_iso)
        personas.append({
            "persona_id": persona_id,
            "device": r.get("device"),
            "source": r.get("source"),
            "operatingSystem": r.get("operating_system"),
            "userAgeBracket": r.get("user_age_bracket"),
            "newVsReturning": r.get("new_vs_returning"),
            "gender": r.get("gender"),
            "geo": r.get("geo"),
            "intent": r.get("intent"),
            "metrics": {
                "sessions": int(r.get("sessions", 0) or 0),
                "cr": float(r.get("cr", 0) or 0),
                "bounce_rate": float(r.get("bounce_rate", 0) or 0),
                "dwell_means": float(r.get("avg_dwell_sec", 0) or 0),
                "backtrack_rate": float(r.get("backtrack_rate", 0) or 0),
                "form_error_rate": float(r.get("form_error_rate", 0) or 0),
            },
            "weight": (int(r.get("sessions", 0) or 0) / total_sessions),
            "window_end": window_end_iso,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

    # Produce exactly 1000 entries
    personas_sorted = sorted(personas, key=lambda x: x["metrics"]["sessions"], reverse=True)
    N = len(personas_sorted)
    pool: List[Dict[str, Any]] = []
    if N >= pool_size:
        pool = personas_sorted[:pool_size]
        # If fewer than 1000 after cut (not possible), sample remainder; else exactly 1000
    else:
        pool = personas_sorted.copy()
        # sample remainder with replacement weighted by sessions
        ids = [p["persona_id"] for p in personas_sorted]
        weights = [p["metrics"]["sessions"] for p in personas_sorted]
        replicas = _weighted_sample(ids, weights, pool_size - N)
        index = {p["persona_id"]: p for p in personas_sorted}
        for ridx, pid in enumerate(replicas):
            base = index[pid]
            # Add a replica marker internally (not required to expose in prompt)
            replica = dict(base)
            replica["_replica_index"] = ridx + 1
            pool.append(replica)
    return pool


def save_pool_artifacts(pool: List[Dict[str, Any]], data_dir: Optional[str] = None) -> Dict[str, str]:
    out_dir = Path(os.getenv("PERSONAS_DATA_DIR") or data_dir or "data/personas")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "master_pool.jsonl"
    yaml_path = out_dir / "master_pool.yaml"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in pool:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # Minimal YAML round-trip
    def _yaml_str(rec: Dict[str, Any]) -> str:
        # Compact YAML-like for readability; not strict
        return json.dumps(rec, ensure_ascii=False)
    with open(yaml_path, "w", encoding="utf-8") as f:
        for rec in pool:
            f.write(_yaml_str(rec) + "\n")
    return {"jsonl": str(jsonl_path), "yaml": str(yaml_path)}
