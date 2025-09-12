from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Iterable, List, Tuple


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _quantiles(values: List[float], qs: Tuple[float, float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    xs = sorted(values)
    def q_at(q: float) -> float:
        pos = (len(xs) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
    return q_at(qs[0]), q_at(qs[1])


ALLOWED_DEVICES = {"mobile", "desktop"}
ALLOWED_SOURCES = {"ads", "organic", "direct", "referral"}


def _norm_device(s: Any) -> str:
    s = str(s or "").strip().lower()
    return s if s in ALLOWED_DEVICES else "desktop"


def _norm_source(s: Any) -> str:
    s = str(s or "").strip().lower()
    if s in ALLOWED_SOURCES:
        return s
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
    intent: str | None = None


def _cohorts_from_obj(obj: Any) -> Tuple[List[Cohort], float]:
    items: Iterable[Any]
    global_median = None
    if isinstance(obj, dict) and "cohorts" in obj and isinstance(obj["cohorts"], list):
        items = obj["cohorts"]
        try:
            global_median = float(obj.get("global_median_dwell_sec") or 0) or None
        except Exception:
            global_median = None
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
                    dwell_means=float(it.get("dwell_means") or it.get("dwell_mean") or it.get("dwell_mean_sec") or 0),
                    backtrack_rate=float(it.get("backtrack_rate", 0) or 0),
                    form_error_rate=float(it.get("form_error_rate", 0) or 0),
                )
            )
        except Exception:
            continue

    # Compute global median dwell if not provided
    if not global_median:
        ds = [c.dwell_means for c in out if c.dwell_means > 0]
        global_median = float(median(ds)) if ds else 1.0
    return out, global_median


def _label_intent(cohorts: List[Cohort]) -> None:
    crs = [c.cr for c in cohorts]
    brs = [c.bounce_rate for c in cohorts]
    q1_cr, q2_cr = _quantiles(crs, (1/3, 2/3))
    q1_br, q2_br = _quantiles(brs, (1/3, 2/3))
    def bin3(v: float, q1: float, q2: float) -> int:
        if v <= q1: return 0
        if v <= q2: return 1
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


def _merge_tiny(cohorts: List[Cohort], min_sessions: int) -> List[Tuple[str, Dict[str, Any]]]:
    big: List[Cohort] = [c for c in cohorts if c.sessions >= min_sessions]
    tiny: List[Cohort] = [c for c in cohorts if c.sessions < min_sessions]
    personas: List[Tuple[str, Dict[str, Any]]] = []
    # Keep big cohorts
    for idx, c in enumerate(big):
        personas.append((
            f"{c.device}_{c.source}_{c.intent}_big_{idx+1}",
            {
                "device": c.device,
                "source": c.source,
                "intent": c.intent,
                "sessions": c.sessions,
                "cr": c.cr,
                "bounce_rate": c.bounce_rate,
                "dwell_means": c.dwell_means,
                "backtrack_rate": c.backtrack_rate,
                "form_error_rate": c.form_error_rate,
            },
        ))
    # Group tiny by (device, source, intent)
    buckets: Dict[Tuple[str, str, str], List[Cohort]] = {}
    for c in tiny:
        key = (c.device, c.source, c.intent or "warm")
        buckets.setdefault(key, []).append(c)
    for (device, source, intent), grp in buckets.items():
        sess = sum(c.sessions for c in grp) or 1.0
        def wavg(attr: str) -> float:
            return sum(c.sessions * getattr(c, attr) for c in grp) / sess
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
            },
        ))
    return personas


def build_personas_yaml(data: Any, min_sessions: int = 500) -> str:
    """Build personas YAML from cohorts or GA4-style wrapper.

    Returns a YAML string starting with 'personas:'.
    """
    cohorts, global_median = _cohorts_from_obj(data)
    if not cohorts:
        return "personas: {}\n"
    _label_intent(cohorts)
    personas = _merge_tiny(cohorts, min_sessions=min_sessions)
    total_sessions = sum(m.get("sessions", 0) for _, m in personas) or 1.0

    lines: List[str] = ["personas:"]
    for pid, m in personas:
        buy_propensity = _clip(m.get("cr", 0.0), 0.005, 0.20)
        dwell_scale = _clip((m.get("dwell_means", 0.0) or 0.0) / (global_median or 1.0), 0.8, 1.5)
        backtrack_p = _clip(m.get("backtrack_rate", 0.0), 0.05, 0.35)
        form_error_p = _clip(m.get("form_error_rate", 0.0), 0.02, 0.20)
        weight = (m.get("sessions", 0.0) or 0.0) / total_sessions
        lines.append(f"  {pid}:")
        lines.append(f"    device: {m.get('device')}")
        lines.append(f"    source: {m.get('source')}")
        lines.append(f"    intent: {m.get('intent')}")
        lines.append(f"    buy_propensity: {buy_propensity:.6f}")
        lines.append(f"    dwell_scale: {dwell_scale:.6f}")
        lines.append(f"    backtrack_p: {backtrack_p:.6f}")
        lines.append(f"    form_error_p: {form_error_p:.6f}")
        lines.append(f"    weight: {weight:.6f}")
    return "\n".join(lines) + "\n"

