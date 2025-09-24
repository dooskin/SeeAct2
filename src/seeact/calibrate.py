from __future__ import annotations

import argparse
from pathlib import Path

from seeact.personas.calibrator import calibrate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate personas against GA conversion targets")
    parser.add_argument("--personas", required=True, help="Input personas YAML")
    parser.add_argument("--ga-targets", required=True, help="GA targets JSON/YAML")
    parser.add_argument("--metrics", required=True, help="Runner metrics JSONL file")
    parser.add_argument("--out", required=True, help="Output calibrated personas YAML")
    parser.add_argument("--epsilon", type=float, default=0.01, help="CR tolerance for backtrack adjustments")
    args = parser.parse_args(argv)

    calibrate(
        personas_path=Path(args.personas).resolve(),
        ga_targets_path=Path(args.ga_targets).resolve(),
        metrics_path=Path(args.metrics).resolve(),
        output_path=Path(args.out).resolve(),
        epsilon=args.epsilon,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
