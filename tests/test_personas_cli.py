import json
from pathlib import Path

from src.personas_cli import build_personas_yaml


def test_personas_yaml_basic(tmp_path: Path):
    data = [
        {"device": "mobile", "source": "ads", "session_depth": 1, "country": "US", "sessions": 1200,
         "cr": 0.12, "bounce_rate": 0.35, "dwell_means": 45.0, "backtrack_rate": 0.12, "form_error_rate": 0.05},
        {"device": "desktop", "source": "organic", "session_depth": 1, "country": "US", "sessions": 800,
         "cr": 0.03, "bounce_rate": 0.55, "dwell_means": 60.0, "backtrack_rate": 0.2, "form_error_rate": 0.08},
        {"device": "mobile", "source": "direct", "session_depth": 2, "country": "CA", "sessions": 300,
         "cr": 0.01, "bounce_rate": 0.65, "dwell_means": 30.0, "backtrack_rate": 0.25, "form_error_rate": 0.12},
        {"device": "desktop", "source": "referral", "session_depth": 2, "country": "UK", "sessions": 200,
         "cr": 0.08, "bounce_rate": 0.40, "dwell_means": 50.0, "backtrack_rate": 0.10, "form_error_rate": 0.04},
    ]

    yaml_out = build_personas_yaml(data, min_sessions=500)
    assert yaml_out.startswith("personas:")
    # Ensure required fields appear
    for key in [
        "device:", "source:", "intent:", "buy_propensity:", "dwell_scale:",
        "backtrack_p:", "form_error_p:", "weight:",
    ]:
        assert key in yaml_out

    # We should get 2 big personas (>=500 sessions) and grouped tiny ones => total >= 3 personas
    persona_count = sum(1 for line in yaml_out.splitlines() if line.strip().endswith(":") and not line.strip().startswith("personas"))
    assert persona_count >= 3


def test_personas_yaml_nested_shape_uses_global_median():
    data = {
        "window_days": 30,
        "global_median_dwell_sec": 12.0,
        "cohorts": [
            {
                "key": {"device": "mobile", "source": "ads", "country": "US", "session_depth": "shallow"},
                "sessions": 1200,
                "cr": 0.02,
                "bounce_rate": 0.62,
                "dwell_mean_sec": 13.5,
                "backtrack_rate": 0.23,
                "form_error_rate": 0.11,
            },
            {
                "key": {"device": "desktop", "source": "organic", "country": "US", "session_depth": "deep"},
                "sessions": 3000,
                "cr": 0.09,
                "bounce_rate": 0.33,
                "dwell_mean_sec": 12.9,
                "backtrack_rate": 0.14,
                "form_error_rate": 0.06,
            },
        ],
    }
    yaml_out = build_personas_yaml(data, min_sessions=500)
    assert yaml_out.startswith("personas:")
    # Check that dwell_scale used 12.0 as denominator (e.g., 13.5/12=1.125)
    assert "dwell_scale: 1.125000" in yaml_out
