from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


MIN_BUY = 0.005
MAX_BUY = 0.35
MIN_DWELL = 0.8
MAX_DWELL = 1.5
MIN_BACKTRACK = 0.05
MAX_BACKTRACK = 0.35


@dataclass
class PersonaStats:
    persona_id: str
    attempts: int
    successes: int
    total_duration_ms: float

    @property
    def observed_cr(self) -> Optional[float]:
        if self.attempts <= 0:
            return None
        return self.successes / self.attempts

    @property
    def observed_dwell(self) -> Optional[float]:
        if self.attempts <= 0:
            return None
        return (self.total_duration_ms / self.attempts) / 1000.0


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def load_personas(path: Path) -> Dict[str, Dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("personas", {}) if isinstance(data, dict) else {}


def save_personas(path: Path, personas: Dict[str, Dict[str, Any]]) -> None:
    payload = {"personas": personas}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_ga_targets(path: Path) -> Dict[str, Dict[str, Any]]:
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data.get("personas", data)


def load_synthetic_stats(path: Path) -> Dict[str, PersonaStats]:
    attempts: Dict[str, int] = {}
    successes: Dict[str, int] = {}
    duration: Dict[str, float] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = record.get("persona_id")
            if not pid:
                continue
            if record.get("event") == "task_complete":
                attempts[pid] = attempts.get(pid, 0) + 1
                if record.get("success"):
                    successes[pid] = successes.get(pid, 0) + 1
                duration[pid] = duration.get(pid, 0.0) + float(record.get("duration_ms", 0) or 0.0)
            elif record.get("event") == "task_error":
                attempts[pid] = attempts.get(pid, 0) + 1
    stats: Dict[str, PersonaStats] = {}
    for pid, att in attempts.items():
        stats[pid] = PersonaStats(
            persona_id=pid,
            attempts=att,
            successes=successes.get(pid, 0),
            total_duration_ms=duration.get(pid, 0.0),
        )
    return stats


def _adjust_buy_propensity(current: float, target_cr: float, observed_cr: Optional[float]) -> float:
    if observed_cr is None or observed_cr <= 0:
        return _clip(current * 1.2, MIN_BUY, MAX_BUY)
    ratio = target_cr / observed_cr if observed_cr else 1.0
    factor = ratio ** 0.5
    return _clip(current * factor, MIN_BUY, MAX_BUY)


def _adjust_dwell_scale(current: float, target_dwell: Optional[float], observed_dwell: Optional[float]) -> float:
    if not target_dwell or target_dwell <= 0 or not observed_dwell or observed_dwell <= 0:
        return current
    ratio = target_dwell / observed_dwell
    factor = ratio ** 0.5
    return _clip(current * factor, MIN_DWELL, MAX_DWELL)


def _adjust_backtrack(current: float, target_cr: float, observed_cr: Optional[float], epsilon: float) -> float:
    if observed_cr is None:
        return current
    error = target_cr - observed_cr
    if error > epsilon:
        return _clip(current * 0.9, MIN_BACKTRACK, MAX_BACKTRACK)
    if error < -epsilon:
        return _clip(current * 1.1, MIN_BACKTRACK, MAX_BACKTRACK)
    return current


def calibrate_personas(
    personas: Dict[str, Dict[str, Any]],
    ga_targets: Dict[str, Dict[str, Any]],
    synthetic_stats: Dict[str, PersonaStats],
    *,
    epsilon: float = 0.01,
) -> Dict[str, Dict[str, Any]]:
    calibrated = {}
    for pid, data in personas.items():
        row = dict(data)
        targets = ga_targets.get(pid) or {}
        stats = synthetic_stats.get(pid)

        target_cr = float(targets.get("target_cr", targets.get("cr", 0)) or 0)
        target_dwell = targets.get("target_dwell") or targets.get("dwell_sec")
        target_backtrack = targets.get("target_backtrack")

        if target_cr > 0:
            row["buy_propensity"] = _adjust_buy_propensity(
                float(row.get("buy_propensity", MIN_BUY) or MIN_BUY),
                target_cr,
                stats.observed_cr if stats else None,
            )
            row["backtrack_p"] = _adjust_backtrack(
                float(row.get("backtrack_p", MIN_BACKTRACK) or MIN_BACKTRACK),
                target_cr,
                stats.observed_cr if stats else None,
                epsilon,
            )

        if target_dwell:
            row["dwell_scale"] = _adjust_dwell_scale(
                float(row.get("dwell_scale", 1.0) or 1.0),
                float(target_dwell),
                stats.observed_dwell if stats else None,
            )

        if target_backtrack is not None:
            row["backtrack_p"] = _clip(float(target_backtrack), MIN_BACKTRACK, MAX_BACKTRACK)

        calibration_meta = row.setdefault("calibration", {})
        calibration_meta.update(
            {
                "target_cr": target_cr if target_cr else None,
                "observed_cr": stats.observed_cr if stats else None,
                "attempts": stats.attempts if stats else 0,
                "successes": stats.successes if stats else 0,
                "target_dwell": float(target_dwell) if target_dwell else None,
                "observed_dwell": stats.observed_dwell if stats else None,
                "calibrated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        calibrated[pid] = row
    return calibrated


def calibrate(
    personas_path: Path,
    ga_targets_path: Path,
    metrics_path: Path,
    output_path: Path,
    *,
    epsilon: float = 0.01,
) -> Dict[str, Dict[str, Any]]:
    personas = load_personas(personas_path)
    ga_targets = load_ga_targets(ga_targets_path) if ga_targets_path else {}
    synthetic_stats = load_synthetic_stats(metrics_path)
    calibrated = calibrate_personas(personas, ga_targets, synthetic_stats, epsilon=epsilon)
    save_personas(output_path, calibrated)
    return calibrated

