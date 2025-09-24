import sys
from pathlib import Path


def _ensure_pkg():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_pkg()

from seeact.personas.calibrator import calibrate, load_personas


def test_calibrate_adjusts_persona(tmp_path: Path):
    personas_yaml = tmp_path / "personas.yaml"
    personas_yaml.write_text(
        """
personas:
  pid1:
    device: mobile
    source: organic
    intent: warm
    buy_propensity: 0.020000
    dwell_scale: 1.000000
    backtrack_p: 0.200000
""".strip()
        + "\n",
        encoding="utf-8",
    )

    ga_targets = tmp_path / "ga.json"
    ga_targets.write_text(
        """
{
  "personas": {
    "pid1": {
      "target_cr": 0.08,
      "target_dwell": 30
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        """
{"event": "task_complete", "persona_id": "pid1", "success": false, "duration_ms": 20000}
{"event": "task_complete", "persona_id": "pid1", "success": false, "duration_ms": 18000}
{"event": "task_error", "persona_id": "pid1"}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "calibrated.yaml"
    calibrate(personas_yaml, ga_targets, metrics, out_path, epsilon=0.01)

    calibrated = load_personas(out_path)
    row = calibrated["pid1"]
    assert row["buy_propensity"] > 0.02  # nudged upward toward higher CR
    assert row["dwell_scale"] != 1.0
    assert "calibrated_at" in row["calibration"]
