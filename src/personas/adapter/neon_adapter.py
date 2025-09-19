from __future__ import annotations

import os
import json
import contextlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


# DSN precedence: env NEON_DATABASE_URL > explicit --dsn > config
def resolve_dsn(explicit_dsn: Optional[str] = None, config_url: Optional[str] = None) -> Optional[str]:
    env_dsn = os.getenv("NEON_DATABASE_URL") or ""
    if env_dsn:
        return env_dsn
    if explicit_dsn:
        return explicit_dsn
    if config_url:
        return config_url
    return None


@dataclass
class GAConfig:
    events_table: str = os.getenv("GA_EVENTS_TABLE", "ga_events")
    conversion_events: Tuple[str, ...] = ("purchase", "checkout_progress")
    include_events_extra: Tuple[str, ...] = tuple()
    window_days: int = 30
    window_end: Optional[datetime] = None


DDL_STATEMENTS = [
    # Names must match the lock sheet
    """
    CREATE TABLE IF NOT EXISTS "SegmentSnapshot" (
      name TEXT NOT NULL,
      rule_json JSONB NOT NULL,
      breakdowns_json JSONB NOT NULL,
      window_end TIMESTAMPTZ NOT NULL,
      PRIMARY KEY(name, window_end)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS "CohortMetrics" (
      cohort_key TEXT NOT NULL,
      sessions INT NOT NULL,
      conversions INT NOT NULL,
      bounce_sessions INT NOT NULL,
      avg_dwell_sec DOUBLE PRECISION NOT NULL,
      backtracks INT NOT NULL,
      form_errors INT NOT NULL,
      window_end TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS "Personas" (
      persona_id TEXT PRIMARY KEY,
      payload JSONB NOT NULL,
      window_end TIMESTAMPTZ NOT NULL,
      generated_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS "PersonaPrompts" (
      persona_id TEXT NOT NULL,
      site_domain TEXT,
      prompt TEXT NOT NULL,
      temperature DOUBLE PRECISION NOT NULL,
      regenerated BOOLEAN NOT NULL,
      generated_at TIMESTAMPTZ NOT NULL,
      PRIMARY KEY(persona_id, site_domain, generated_at)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS "SiteVocab" (
      site_domain TEXT NOT NULL,
      vocab JSONB NOT NULL,
      scraped_at TIMESTAMPTZ NOT NULL,
      PRIMARY KEY(site_domain, scraped_at)
    )
    """,
]


def ensure_tables(conn) -> None:
    cur = conn.cursor()
    for ddl in DDL_STATEMENTS:
        cur.execute(ddl)
    conn.commit()


def _window_bounds(cfg: GAConfig) -> Tuple[datetime, datetime]:
    end = cfg.window_end or datetime.now(timezone.utc)
    start = end - timedelta(days=int(cfg.window_days))
    return start, end


def _sql_build(cfg: GAConfig) -> str:
    # Server-side aggregation across the 7 dimensions
    start, end = _window_bounds(cfg)
    conv = ",".join([f"'{e}'" for e in cfg.conversion_events + cfg.include_events_extra])
    events = cfg.events_table
    sql = f"""
WITH base AS (
  SELECT
    session_id,
    event_name,
    event_timestamp,
    page_location,
    page_referrer,
    COALESCE(engagement_time_msec,0) AS engagement_time_msec,
    device_category,
    operating_system,
    session_source_medium,
    user_age_bracket,
    new_vs_returning,
    gender,
    CONCAT(COALESCE(geo_country,''), CASE WHEN COALESCE(geo_region,'')<>'' THEN ':'||geo_region ELSE '' END) AS geo_bucket,
    custom_event
  FROM {events}
  WHERE event_timestamp >= %(start)s AND event_timestamp < %(end)s
),
by_session AS (
  SELECT
    session_id,
    MAX(CASE WHEN event_name IN ({conv}) THEN 1 ELSE 0 END) AS is_convert,
    MAX(CASE WHEN engagement_time_msec = 0 THEN 1 ELSE 0 END) AS is_bounce,
    SUM(engagement_time_msec)/1000.0 AS dwell_sec,
    SUM(CASE WHEN COALESCE((custom_event->>'backtrack')::int,0)=1 THEN 1 ELSE 0 END) AS backtracks,
    SUM(CASE WHEN COALESCE((custom_event->>'form_error')::int,0)=1 THEN 1 ELSE 0 END) AS form_errors,
    MAX(device_category) AS device_category,
    MAX(operating_system) AS operating_system,
    MAX(session_source_medium) AS session_source_medium,
    MAX(user_age_bracket) AS user_age_bracket,
    MAX(new_vs_returning) AS new_vs_returning,
    MAX(gender) AS gender,
    MAX(geo_bucket) AS geo_bucket,
    ARRAY_AGG(page_location ORDER BY event_timestamp) AS pages,
    ARRAY_AGG(page_referrer ORDER BY event_timestamp) AS refs
  FROM base
  GROUP BY session_id
),
backtrack_heuristic AS (
  SELECT
    s.*,
    CASE
      WHEN backtracks > 0 THEN backtracks
      ELSE (
        SELECT COUNT(*) FROM generate_subscripts(pages,1) i
        WHERE i>1 AND refs[i] = pages[i-1]
      )
    END AS backtracks_final
  FROM by_session s
),
cohorts AS (
  SELECT
    device_category,
    operating_system,
    session_source_medium,
    user_age_bracket,
    new_vs_returning,
    gender,
    geo_bucket,
    COUNT(*) AS sessions,
    SUM(is_convert) AS conversions,
    SUM(is_bounce) AS bounce_sessions,
    AVG(dwell_sec) AS avg_dwell_sec,
    SUM(backtracks_final) AS backtracks,
    SUM(form_errors) AS form_errors
  FROM backtrack_heuristic
  GROUP BY 1,2,3,4,5,6,7
)
SELECT * FROM cohorts;
"""
    return sql


class NeonGAAdapter:
    def __init__(self, dsn: Optional[str], cfg: Optional[GAConfig] = None):
        self.dsn = dsn
        self.cfg = cfg or GAConfig()
        self._sql = _sql_build(self.cfg)

    @property
    def sql(self) -> str:
        return self._sql

    def connect(self):
        import psycopg
        if not self.dsn:
            raise RuntimeError("No DSN provided; NEON_DATABASE_URL or --dsn is required for DB ops")
        return psycopg.connect(self.dsn)

    def ensure_tables(self) -> None:
        with self.connect() as conn:
            ensure_tables(conn)

    def fetch_cohorts(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                start, end = _window_bounds(self.cfg)
                cur.execute(self._sql, {"start": start, "end": end})
                cols = [d[0] for d in cur.description]
                out = [dict(zip(cols, row)) for row in cur.fetchall()]
                return out

    def upsert_personas(self, records: Iterable[Dict[str, Any]], window_end: datetime) -> int:
        with self.connect() as conn:
            cur = conn.cursor()
            now = datetime.now(timezone.utc)
            n = 0
            for rec in records:
                pid = rec.get("persona_id")
                payload = json.dumps(rec)
                cur.execute(
                    """
                    INSERT INTO "Personas"(persona_id, payload, window_end, generated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT(persona_id) DO UPDATE SET payload=EXCLUDED.payload, window_end=EXCLUDED.window_end, generated_at=EXCLUDED.generated_at
                    """,
                    (pid, payload, window_end, now),
                )
                n += 1
            conn.commit()
            return n

    def upsert_site_vocab(self, site_domain: str, vocab: Dict[str, Any]) -> None:
        with self.connect() as conn:
            cur = conn.cursor()
            now = datetime.now(timezone.utc)
            cur.execute(
                """
                INSERT INTO "SiteVocab"(site_domain, vocab, scraped_at)
                VALUES (%s, %s, %s)
                """,
                (site_domain, json.dumps(vocab), now),
            )
            conn.commit()

    def insert_persona_prompt(self, persona_id: str, site_domain: Optional[str], prompt: str, temperature: float, regenerated: bool) -> None:
        with self.connect() as conn:
            cur = conn.cursor()
            now = datetime.now(timezone.utc)
            cur.execute(
                """
                INSERT INTO "PersonaPrompts"(persona_id, site_domain, prompt, temperature, regenerated, generated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (persona_id, site_domain, prompt, float(temperature), bool(regenerated), now),
            )
            conn.commit()

