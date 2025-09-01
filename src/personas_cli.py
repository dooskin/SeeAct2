#!/usr/bin/env python3
"""
Persona builder CLI

Reads GA4/Shopify cohort stats JSON and emits YAML personas.

Input expectations:
- Either a list of cohort objects, or a dict mapping keys -> cohort objects.
- Each cohort object should include:
  device, source, session_depth, country, sessions, cr, bounce_rate,
  dwell_means, backtrack_rate, form_error_rate

Usage:
  python src/personas_cli.py --input cohorts.json > personas.yaml
  cat cohorts.json | python src/personas_cli.py > personas.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Iterable, List, Tuple


ALLOWED_DEVICES = {"mobile", "desktop"}
ALLOWED_SOURCES = {"ads", "organic", "direct", "referral"}


def _quantiles(values: List[float], qs: Tuple[float, float]) -> Tuple[float, float]:
    """Return approximate quantile cutoffs for q1 and q2 in [0,1].
    Simple linear approach without external deps.
    """
    if not values:
        return 0.0, 0.0
    xs = sorted(values)
    def q_at(q: float) -> float:
        if not xs:
            return 0.0
        pos = (len(xs) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
    return q_at(qs[0]), q_at(qs[1])


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _norm_device(s: Any) -> str:
    s = str(s or "").strip().lower()
    if s in ALLOWED_DEVICES:
        return s
    return "desktop"


def _norm_source(s: Any) -> str:
    s = str(s or "").strip().lower()
    if s in ALLOWED_SOURCES:
        return s
    # heuristics
    if any(k in s for k in ["ad", "cpc", "paid", "ppc", "sem"]):
        return "ads"
    if "direct" in s:
        return "direct"
    if any(k in s for k in ["ref", "partner"]):
        return "referral"
    return "organic"


@dataclass
class Cohort:
    device: str
    source: str
    session_depth: Any
    country: Any
    sessions: float
    cr: float
    bounce_rate: float
    dwell_means: float
    backtrack_rate: float
    form_error_rate: float
    intent: str | None = None  # to be filled later


def _cohorts_from_obj(obj: Any) -> List[Cohort]:
    items: Iterable[Any]
    # Support nested GA4-style wrapper { window_days, global_median_dwell_sec, cohorts: [...] }
    if isinstance(obj, dict) and "cohorts" in obj and isinstance(obj["cohorts"], list):
        items = obj["cohorts"]
    elif isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        items = obj.values()
    else:
        raise ValueError("Input JSON must be a list or dict of cohorts")

    out: List[Cohort] = []
    for it in items:
        try:
            key = it.get("key", {}) if isinstance(it, dict) else {}
            out.append(
                Cohort(
                    device=_norm_device(it.get("device") or key.get("device")),
                    source=_norm_source(it.get("source") or key.get("source")),
                    session_depth=it.get("session_depth") or key.get("session_depth"),
                    country=it.get("country") or key.get("country"),
                    sessions=float(it.get("sessions", 0) or 0),
                    cr=float(it.get("cr", 0) or 0),
                    bounce_rate=float(it.get("bounce_rate", 0) or 0),
                    dwell_means=float(
                        it.get("dwell_means")
                        or it.get("dwell_mean")
                        or it.get("dwell_mean_sec")
                        or 0
                    ),
                    backtrack_rate=float(it.get("backtrack_rate", 0) or 0),
                    form_error_rate=float(it.get("form_error_rate", 0) or 0),
                )
            )
        except Exception:
            continue
    return out


def _label_intent(cohorts: List[Cohort]) -> None:
    crs = [c.cr for c in cohorts]
    brs = [c.bounce_rate for c in cohorts]
    q1_cr, q2_cr = _quantiles(crs, (1/3, 2/3))
    q1_br, q2_br = _quantiles(brs, (1/3, 2/3))

    def bin3(v: float, q1: float, q2: float) -> int:
        if v <= q1:
            return 0
        if v <= q2:
            return 1
        return 2

    for c in cohorts:
        cr_bin = bin3(c.cr, q1_cr, q2_cr)
        br_bin = bin3(c.bounce_rate, q1_br, q2_br)
        score = cr_bin - br_bin
        if score >= 1:
            c.intent = "hot"
        elif score <= -1:
            c.intent = "cold"
        else:
            c.intent = "warm"


def _merge_tiny(cohorts: List[Cohort], min_sessions: int, total_sessions: float) -> List[Tuple[str, Dict[str, Any]]]:
    """Return list of (persona_id, aggregated_metrics_dict).
    Aggregates cohorts with sessions < min_sessions by (device, source, intent).
    """
    big: List[Cohort] = [c for c in cohorts if c.sessions >= min_sessions]
    tiny: List[Cohort] = [c for c in cohorts if c.sessions < min_sessions]

    personas: List[Tuple[str, Dict[str, Any]]] = []

    # Keep big cohorts as one persona each
    for idx, c in enumerate(big):
        persona_id = f"{c.device}_{c.source}_{c.intent}_big_{idx+1}"
        personas.append((persona_id, {
            "device": c.device,
            "source": c.source,
            "intent": c.intent,
            "sessions": c.sessions,
            "cr": c.cr,
            "bounce_rate": c.bounce_rate,
            "dwell_means": c.dwell_means,
            "backtrack_rate": c.backtrack_rate,
            "form_error_rate": c.form_error_rate,
        }))

    # Group tiny cohorts by similarity
    groups: Dict[Tuple[str, str, str], List[Cohort]] = defaultdict(list)
    for c in tiny:
        groups[(c.device, c.source, c.intent or "warm")].append(c)

    for (device, source, intent), grp in groups.items():
        sess = sum(c.sessions for c in grp) or 1.0
        wavg = lambda arr: sum(x.sessions * getattr(x, arr) for x in grp) / sess
        personas.append((
            f"{device}_{source}_{intent}_tiny",
            {
                "device": device,
                "source": source,
                "intent": intent,
                "sessions": sess,
                "cr": wavg("cr"),
                "bounce_rate": wavg("bounce_rate"),
                "dwell_means": wavg("dwell_means"),
                "backtrack_rate": wavg("backtrack_rate"),
                "form_error_rate": wavg("form_error_rate"),
            }
        ))

    return personas


def _emit_yaml(personas: List[Tuple[str, Dict[str, Any]]], global_median_dwell: float, total_sessions: float) -> str:
    def ffmt(x: float) -> str:
        return f"{x:.6f}"

    lines: List[str] = []
    lines.append("personas:")
    for pid, m in personas:
        buy_prop = _clip(m["cr"], 0.005, 0.20)
        dwell_scale = _clip((m["dwell_means"] / global_median_dwell) if global_median_dwell > 0 else 1.0, 0.8, 1.5)
        backtrack_p = _clip(m["backtrack_rate"], 0.05, 0.35)
        form_error_p = _clip(m["form_error_rate"], 0.02, 0.20)
        weight = (m["sessions"] / total_sessions) if total_sessions > 0 else 0.0

        lines.append(f"  {pid}:")
        lines.append(f"    device: {m['device']}")
        lines.append(f"    source: {m['source']}")
        lines.append(f"    intent: {m['intent']}")
        lines.append(f"    buy_propensity: {ffmt(buy_prop)}")
        lines.append(f"    dwell_scale: {ffmt(dwell_scale)}")
        lines.append(f"    backtrack_p: {ffmt(backtrack_p)}")
        lines.append(f"    form_error_p: {ffmt(form_error_p)}")
        lines.append(f"    weight: {ffmt(weight)}")
    return "\n".join(lines) + "\n"


def build_personas_yaml(obj: Any, min_sessions: int = 500) -> str:
    cohorts = _cohorts_from_obj(obj)
    if not cohorts:
        return "personas: {}\n"
    _label_intent(cohorts)
    total_sessions = sum(c.sessions for c in cohorts) or 1.0
    # Prefer provided global median if present
    global_median_dwell = None
    if isinstance(obj, dict) and obj.get("global_median_dwell_sec"):
        try:
            global_median_dwell = float(obj.get("global_median_dwell_sec"))
        except Exception:
            global_median_dwell = None
    if not global_median_dwell:
        global_median_dwell = median([c.dwell_means for c in cohorts if c.dwell_means is not None] or [1.0])
    personas = _merge_tiny(cohorts, min_sessions=min_sessions, total_sessions=total_sessions)
    return _emit_yaml(personas, global_median_dwell=global_median_dwell, total_sessions=total_sessions)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build personas YAML from cohort stats JSON")
    p.add_argument("--input", "-i", help="Path to COHORT_STATS JSON (or '-' for stdin)", default="-")
    p.add_argument("--min-sessions", type=int, default=500, help="Merge cohorts with sessions below this threshold")
    args = p.parse_args(argv)

    data_str = sys.stdin.read() if args.input == "-" else open(args.input, "r").read()
    obj = json.loads(data_str)
    yaml_out = build_personas_yaml(obj, min_sessions=args.min_sessions)
    sys.stdout.write(yaml_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
