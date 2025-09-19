import sys
from pathlib import Path


def _ensure_pkg():
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def test_mapping_unknown_k_anon_and_pool_size():
    _ensure_pkg()
    from personas.builder import build_master_pool, normalize_dims, map_os_family, map_source_taxonomy, compute_unknown_share
    # Construct a few cohorts with tiny sessions to force merging and unknowns
    cohorts = [
        {
            "device_category": "tablet",
            "operating_system": "Mac OS X",
            "session_source_medium": "google / organic",
            "user_age_bracket": "25-34",
            "new_vs_returning": "new",
            "gender": "female",
            "geo_bucket": "US:CA",
            "sessions": 30,
            "conversions": 2,
            "bounce_sessions": 10,
            "avg_dwell_sec": 40.0,
            "backtracks": 2,
            "form_errors": 1,
        },
        {
            "device_category": "desktop",
            "operating_system": "Windows 11",
            "session_source_medium": "(direct) / (none)",
            "user_age_bracket": "35-44",
            "new_vs_returning": "returning",
            "gender": "male",
            "geo_bucket": "US:NY",
            "sessions": 40,
            "conversions": 1,
            "bounce_sessions": 20,
            "avg_dwell_sec": 50.0,
            "backtracks": 1,
            "form_errors": 3,
        },
        {
            "device_category": "mobile",
            "operating_system": "Android",
            "session_source_medium": "cpc / google",
            "user_age_bracket": "unknown",
            "new_vs_returning": "unknown",
            "gender": "unknown",
            "geo_bucket": "unknown",
            "sessions": 60,
            "conversions": 5,
            "bounce_sessions": 12,
            "avg_dwell_sec": 30.0,
            "backtracks": 6,
            "form_errors": 2,
        },
    ]
    pool = build_master_pool(cohorts, k_anon=50, unknown_drop_threshold=0.7)
    assert len(pool) == 1000
    # Check normalized mapping basics
    n = normalize_dims(cohorts[0])
    assert n["device"] == "mobile"  # tablet → mobile
    assert map_os_family("Mac OS X") == "macOS"
    assert map_source_taxonomy("cpc / google") == "ads"
    # Unknown share computes
    rows = [normalize_dims(c) | {"sessions": c["sessions"]} for c in cohorts]
    u = compute_unknown_share(rows)
    assert 0 <= u <= 1

