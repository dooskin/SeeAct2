import os
import sys
from pathlib import Path


def _ensure_pkg():
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def test_dsn_precedence_env_overrides_flag_and_config(monkeypatch):
    _ensure_pkg()
    from personas.adapter import resolve_dsn
    monkeypatch.setenv("NEON_DATABASE_URL", "postgres://env")
    assert resolve_dsn("postgres://flag", "postgres://cfg") == "postgres://env"
    monkeypatch.delenv("NEON_DATABASE_URL")
    assert resolve_dsn("postgres://flag", "postgres://cfg") == "postgres://flag"
    assert resolve_dsn(None, "postgres://cfg") == "postgres://cfg"
    assert resolve_dsn(None, None) is None


def test_sql_contains_table_and_events():
    _ensure_pkg()
    from personas.adapter import NeonGAAdapter, GAConfig
    cfg = GAConfig()
    cfg.events_table = "ga_events"
    cfg.conversion_events = ("purchase", "checkout_progress")
    adapter = NeonGAAdapter(dsn=None, cfg=cfg)
    assert "FROM ga_events" in adapter.sql
    # ensure conversion events present
    assert "purchase" in adapter.sql and "checkout_progress" in adapter.sql

