from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None  # type: ignore


@dataclass
class SegmentSnapshotRow:
    name: str
    rule: Any
    breakdowns: Any
    window_end: str


def _norm_device(rule: Any) -> str:
    try:
        for cond in rule or []:
            dim = cond.get("dim")
            vals = cond.get("values") or []
            if dim == "deviceCategory":
                v = ",".join([str(x).lower() for x in vals])
                if any(x in v for x in ["mobile", "tablet"]):
                    return "mobile"
                return "desktop"
    except Exception:
        pass
    return "desktop"


def _norm_source(label: str) -> str:
    l = (label or "").strip().lower()
    if not l:
        return "organic"
    if "direct" in l:
        return "direct"
    if "referral" in l:
        return "referral"
    if "paid" in l:
        return "ads"
    if "social" in l:
        # map organic social to social
        return "social"
    if "shopping" in l:
        return "shopping"
    if "search" in l:
        return "organic"
    return "organic"


def parse_cohorts_from_snapshots(rows: Iterable[SegmentSnapshotRow]) -> List[Dict[str, Any]]:
    """Translate SegmentSnapshot rows into persona cohorts expected by personas_cli.

    Missing rates (cr, bounce_rate, etc.) are set to 0.0 for now; these should be
    filled when upstream Neon tables provide them. Sessions derive from the
    marketingChannel breakdown counts when available.
    """
    cohorts: List[Dict[str, Any]] = []
    for r in rows:
        device = _norm_device(r.rule)
        breakdowns = r.breakdowns or {}
        channels = (breakdowns.get("marketingChannel") or {}).get("values") or []
        # Fallback: if no channels, synthesize a single organic bucket with share=1
        if not channels:
            channels = [{"label": "Organic", "count": 0, "share": 1.0}]
        for ch in channels:
            label = ch.get("label") or ""
            sessions = float(ch.get("count") or 0)
            src = _norm_source(label)
            # Optional geo bucket from name (e.g., contains region/state), best-effort
            geo_bucket = None
            try:
                name = (r.name or "").lower()
                # naive extraction: last token after • might be region/state
                if "•" in name:
                    parts = [p.strip() for p in r.name.split("•")]
                    # pick the token that isn't device or channel if present
                    for p in parts:
                        pl = p.lower()
                        if pl in ("mobile", "desktop", "tablet"):
                            continue
                        if "paid" in pl or "organic" in pl or "referral" in pl or "search" in pl or "social" in pl:
                            continue
                        geo_bucket = p
                        break
            except Exception:
                geo_bucket = None

            cohorts.append(
                {
                    "device": device,
                    "source": src,
                    "session_depth": 1,
                    "country": geo_bucket or "",
                    "sessions": sessions,
                    # Placeholders until metrics are joined from Neon
                    "cr": 0.0,
                    "bounce_rate": 0.0,
                    "dwell_means": 0.0,
                    "backtrack_rate": 0.0,
                    "form_error_rate": 0.0,
                }
            )
    return cohorts


def fetch_latest_snapshots(dsn: str, window_days: Optional[int] = None) -> List[SegmentSnapshotRow]:
    if psycopg is None:
        raise RuntimeError("psycopg is not installed; install package extras or see README.")
    rows: List[SegmentSnapshotRow] = []
    # Connect read-only
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Find the latest window_end
            cur.execute("SELECT max(window_end) FROM \"SegmentSnapshot\";")
            res = cur.fetchone()
            if not res or not res[0]:
                return rows
            latest = res[0]
            # Load latest snapshot rows
            cur.execute(
                """
                SELECT name, rule, breakdowns, window_end
                FROM "SegmentSnapshot"
                WHERE window_end = %s
                """,
                (latest,),
            )
            for name, rule, breakdowns, window_end in cur.fetchall():
                try:
                    r = SegmentSnapshotRow(
                        name=name,
                        rule=rule if isinstance(rule, dict) else json.loads(rule or "{}"),
                        breakdowns=breakdowns if isinstance(breakdowns, dict) else json.loads(breakdowns or "{}"),
                        window_end=str(window_end),
                    )
                    rows.append(r)
                except Exception:
                    continue
    return rows

