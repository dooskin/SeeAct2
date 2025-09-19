#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import sys
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fastapi.testclient import TestClient
from api.main import create_app
from personas.builder import build_master_pool, save_pool_artifacts


def main():
    tmp = Path("./runs/personas_e2e").resolve()
    data_dir = tmp / "data"
    os.makedirs(data_dir, exist_ok=True)
    os.environ["PERSONAS_DATA_DIR"] = str(data_dir)
    # Build a tiny pool (no DB) and persist snapshots
    cohorts = [
        {
            "device_category": "mobile",
            "operating_system": "iOS",
            "session_source_medium": "google / organic",
            "user_age_bracket": "25-34",
            "new_vs_returning": "new",
            "gender": "female",
            "geo_bucket": "US:CA",
            "sessions": 500,
            "conversions": 30,
            "bounce_sessions": 200,
            "avg_dwell_sec": 35.0,
            "backtracks": 50,
            "form_errors": 25,
        },
        {
            "device_category": "desktop",
            "operating_system": "Windows 10",
            "session_source_medium": "(direct) / (none)",
            "user_age_bracket": "35-44",
            "new_vs_returning": "returning",
            "gender": "male",
            "geo_bucket": "US:NY",
            "sessions": 600,
            "conversions": 20,
            "bounce_sessions": 250,
            "avg_dwell_sec": 45.0,
            "backtracks": 40,
            "form_errors": 30,
        },
    ]
    pool = build_master_pool(cohorts, k_anon=50, unknown_drop_threshold=0.7)
    save_pool_artifacts(pool, data_dir=str(data_dir))

    app = create_app()
    client = TestClient(app)

    # List
    r = client.get("/v1/personas/?limit=5")
    r.raise_for_status()
    items = r.json().get("items", [])
    assert items, "Expected personas in snapshot"
    pid = items[0]["persona_id"]

    # Generate prompts (no vocab)
    r = client.post(
        "/v1/personas/generate-prompts",
        json={"persona_ids": [pid], "site_domain": "example.com", "include_vocab": False, "out_dir": str(data_dir / "prompts")},
    )
    r.raise_for_status()
    out = r.json()
    assert out["count"] >= 1

    # Get prompt
    r = client.get(f"/v1/personas/{pid}/prompt")
    r.raise_for_status()
    prompt = r.json().get("prompt", "")
    assert "[SYSTEM]" in prompt and "# SHOP_PERSONA" in prompt

    # Scrape vocab (best-effort; ignore robots by default for E2E)
    os.environ.setdefault("SCRAPER_IGNORE_ROBOTS", "1")
    r = client.post("/v1/personas/scrape-vocab", json={"site": "https://example.com/", "persist": False})
    r.raise_for_status()
    vocab = r.json().get("vocab", {})

    print("E2E OK:\n- personas:", len(items), "\n- prompt size:", len(prompt), "\n- vocab keys:", list(vocab.keys()))


if __name__ == "__main__":
    main()
