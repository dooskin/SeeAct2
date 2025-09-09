#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path


def _get_dsn(cli_dsn: str | None) -> str:
    dsn = cli_dsn or os.getenv("NEON_DATABASE_URL") or os.getenv("NEON_DB_URL")
    if dsn:
        return dsn
    print("Neon DSN not found in env. Paste your DSN (input hidden):")
    # getpass hides input; user can paste full DSN safely
    dsn = getpass.getpass(prompt="Neon DSN: ")
    if not dsn:
        raise SystemExit("Missing DSN. Set NEON_DATABASE_URL or pass --dsn.")
    return dsn


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build personas YAML from Neon SegmentSnapshot")
    p.add_argument("--dsn", help="Neon DSN (or set NEON_DATABASE_URL)")
    p.add_argument("--window-days", type=int, default=None, help="Window size in days (optional; latest window is used)")
    p.add_argument("--min-sessions", type=int, default=500, help="Merge cohorts with sessions below this threshold")
    p.add_argument("--out", "-o", default=str(Path("data/personas/personas.yaml")), help="Output YAML path")
    args = p.parse_args(argv)

    # Import personas builder from existing CLI
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from personas_cli import build_personas_yaml  # type: ignore
    from personas.neon_adapter import fetch_latest_snapshots, parse_cohorts_from_snapshots  # type: ignore

    dsn = _get_dsn(args.dsn)
    rows = fetch_latest_snapshots(dsn, window_days=args.window_days)
    if not rows:
        print("No SegmentSnapshot rows found.")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("personas: {}\n", encoding="utf-8")
        return 0

    cohorts = parse_cohorts_from_snapshots(rows)
    yaml_out = build_personas_yaml(cohorts, min_sessions=args.min_sessions)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(yaml_out, encoding="utf-8")
    print(f"Wrote personas to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
