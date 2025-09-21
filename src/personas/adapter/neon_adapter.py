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
    # Shopify events to summarize for behavior-match charts
    shopify_events: Tuple[str, ...] = (
        "page_view",
        "view_item_list",
        "view_item",
        "add_to_cart",
        "begin_checkout",
        "purchase",
    )


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

    def fetch_event_rates(self) -> Dict[str, float]:
        """Aggregate event rates for the configured Shopify events across the window.

        Returns a mapping of event_name -> rate (events per session), computed as
        distinct sessions with event / total sessions for the window.
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                start, end = _window_bounds(self.cfg)
                events = ",".join([f"'{e}'" for e in self.cfg.shopify_events])
                sql = f"""
WITH base AS (
  SELECT session_id, event_name
  FROM {self.cfg.events_table}
  WHERE event_timestamp >= %(start)s AND event_timestamp < %(end)s
    AND event_name IN ({events})
),
session_events AS (
  SELECT event_name, COUNT(DISTINCT session_id) AS sessions_with_event
  FROM base GROUP BY event_name
),
total AS (
  SELECT COUNT(DISTINCT session_id) AS total_sessions
  FROM {self.cfg.events_table}
  WHERE event_timestamp >= %(start)s AND event_timestamp < %(end)s
)
SELECT e.event_name, (e.sessions_with_event::double precision / NULLIF(t.total_sessions,0)) AS rate
FROM session_events e CROSS JOIN total t;
"""
                cur.execute(sql, {"start": start, "end": end})
                rates = {row[0]: float(row[1] or 0.0) for row in cur.fetchall()}
                # Ensure all keys present
                out = {k: float(rates.get(k, 0.0)) for k in self.cfg.shopify_events}
                return out

    def fetch_distributions(self) -> Dict[str, Dict[str, float]]:
        """Return real traffic distributions (percentages) by key dims using sessions as weight.

        Keys: device_category, session_source_medium, geo_bucket, operating_system, user_age_bracket, new_vs_returning, gender
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                start, end = _window_bounds(self.cfg)
                sql = f"""
WITH base AS (
  SELECT
    session_id,
    MAX(device_category) AS device_category,
    MAX(session_source_medium) AS session_source_medium,
    MAX(operating_system) AS operating_system,
    MAX(user_age_bracket) AS user_age_bracket,
    MAX(new_vs_returning) AS new_vs_returning,
    MAX(gender) AS gender,
    MAX(CONCAT(COALESCE(geo_country,''), CASE WHEN COALESCE(geo_region,'')<>'' THEN ':'||geo_region ELSE '' END)) AS geo_bucket
  FROM {self.cfg.events_table}
  WHERE event_timestamp >= %(start)s AND event_timestamp < %(end)s
  GROUP BY session_id
),
tot AS (SELECT COUNT(*) AS n FROM base)
SELECT 'device_category' AS dim, device_category AS val, COUNT(*)::double precision / NULLIF((SELECT n FROM tot),0) AS pct FROM base GROUP BY device_category
UNION ALL
SELECT 'session_source_medium', session_source_medium, COUNT(*)::double precision / NULLIF((SELECT n FROM tot),0) FROM base GROUP BY session_source_medium
UNION ALL
SELECT 'geo_bucket', geo_bucket, COUNT(*)::double precision / NULLIF((SELECT n FROM tot),0) FROM base GROUP BY geo_bucket
UNION ALL
SELECT 'operating_system', operating_system, COUNT(*)::double precision / NULLIF((SELECT n FROM tot),0) FROM base GROUP BY operating_system
UNION ALL
SELECT 'user_age_bracket', user_age_bracket, COUNT(*)::double precision / NULLIF((SELECT n FROM tot),0) FROM base GROUP BY user_age_bracket
UNION ALL
SELECT 'new_vs_returning', new_vs_returning, COUNT(*)::double precision / NULLIF((SELECT n FROM tot),0) FROM base GROUP BY new_vs_returning
UNION ALL
SELECT 'gender', gender, COUNT(*)::double precision / NULLIF((SELECT n FROM tot),0) FROM base GROUP BY gender;
"""
                cur.execute(sql, {"start": start, "end": end})
                out: Dict[str, Dict[str, float]] = {}
                for dim, val, pct in cur.fetchall():
                    d = out.setdefault(dim, {})
                    d[str(val or 'unknown')] = float(pct or 0.0)
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
