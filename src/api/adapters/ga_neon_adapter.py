"""
Enhanced GA-Neon adapter for calibration process.
Handles traffic snapshots, funnel metrics, and real-time data processing.
"""

from __future__ import annotations

import os
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import psycopg
from psycopg.rows import dict_row


@dataclass
class TrafficSnapshot:
    """Represents a traffic snapshot for a site"""
    site_id: str
    snapshot_id: str
    window_start: datetime
    window_end: datetime
    total_sessions: int
    total_events: int
    fingerprint: str
    created_at: datetime


@dataclass
class FunnelMetrics:
    """Represents funnel metrics for a site"""
    site_id: str
    snapshot_id: str
    event_name: str
    sessions_with_event: int
    total_sessions: int
    rate: float
    created_at: datetime


class EnhancedGANeonAdapter:
    """Enhanced GA-Neon adapter for calibration process"""
    
    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.getenv("NEON_DATABASE_URL")
        if not self.dsn:
            raise ValueError("Database DSN not provided")
        
        # Ensure tables exist
        self._ensure_tables()
    
    def _get_connection(self):
        """Get database connection"""
        return psycopg.connect(self.dsn, row_factory=dict_row)
    
    def _ensure_tables(self):
        """Ensure required tables exist"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Traffic snapshots table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS traffic_snapshots (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        site_id TEXT NOT NULL,
                        snapshot_id TEXT NOT NULL UNIQUE,
                        window_start TIMESTAMPTZ NOT NULL,
                        window_end TIMESTAMPTZ NOT NULL,
                        total_sessions INTEGER NOT NULL,
                        total_events INTEGER NOT NULL,
                        fingerprint TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                
                # Create indexes separately
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_traffic_snapshots_site_id 
                    ON traffic_snapshots (site_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_traffic_snapshots_created_at 
                    ON traffic_snapshots (created_at)
                """)
                
                # Funnel metrics table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS funnel_metrics (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        site_id TEXT NOT NULL,
                        snapshot_id TEXT NOT NULL,
                        event_name TEXT NOT NULL,
                        sessions_with_event INTEGER NOT NULL,
                        total_sessions INTEGER NOT NULL,
                        rate DOUBLE PRECISION NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                
                # Create indexes separately
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_funnel_metrics_site_id 
                    ON funnel_metrics (site_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_funnel_metrics_snapshot_id 
                    ON funnel_metrics (snapshot_id)
                """)
                
                # Calibration results table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS calibration_results (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        calibration_id TEXT NOT NULL UNIQUE,
                        site_id TEXT NOT NULL,
                        snapshot_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        real_distributions JSONB,
                        synthetic_distributions JSONB,
                        real_funnel_rates JSONB,
                        synthetic_funnel_rates JSONB,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                
                # Create indexes separately
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_calibration_results_calibration_id 
                    ON calibration_results (calibration_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_calibration_results_site_id 
                    ON calibration_results (site_id)
                """)
                
                conn.commit()
    
    def _generate_fingerprint(self, site_id: str, window_start: datetime, window_end: datetime) -> str:
        """Generate a unique fingerprint for the snapshot"""
        data = f"{site_id}:{window_start.isoformat()}:{window_end.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def create_traffic_snapshot(self, site_id: str, window_days: int = 30) -> TrafficSnapshot:
        """Create a traffic snapshot for the given site"""
        window_end = datetime.now(timezone.utc)
        window_start = window_end - timedelta(days=window_days)
        snapshot_id = f"{site_id}_{window_start.strftime('%Y%m%d')}_{window_end.strftime('%Y%m%d')}"
        fingerprint = self._generate_fingerprint(site_id, window_start, window_end)
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Check if snapshot already exists
                cur.execute("""
                    SELECT * FROM traffic_snapshots 
                    WHERE site_id = %s AND fingerprint = %s
                    ORDER BY created_at DESC LIMIT 1
                """, (site_id, fingerprint))
                
                existing = cur.fetchone()
                if existing:
                    return TrafficSnapshot(
                        site_id=existing['site_id'],
                        snapshot_id=existing['snapshot_id'],
                        window_start=existing['window_start'],
                        window_end=existing['window_end'],
                        total_sessions=existing['total_sessions'],
                        total_events=existing['total_events'],
                        fingerprint=existing['fingerprint'],
                        created_at=existing['created_at']
                    )
                
                # Check if snapshot_id already exists and generate a new one if needed
                cur.execute("SELECT COUNT(*) FROM traffic_snapshots WHERE snapshot_id = %s", (snapshot_id,))
                count_result = cur.fetchone()
                if count_result and count_result['count'] > 0:
                    # Generate a unique snapshot_id
                    import time
                    snapshot_id = f"{site_id}_{window_start.strftime('%Y%m%d')}_{window_end.strftime('%Y%m%d')}_{int(time.time())}"
                
                # Get session and event counts from GA events table
                cur.execute("""
                    SELECT 
                        COUNT(DISTINCT session_id) as total_sessions,
                        COUNT(*) as total_events
                    FROM ga_events 
                    WHERE event_timestamp >= %s AND event_timestamp < %s
                """, (window_start, window_end))
                
                counts = cur.fetchone()
                total_sessions = counts['total_sessions'] if counts else 0
                total_events = counts['total_events'] if counts else 0
                
                # Insert new snapshot
                cur.execute("""
                    INSERT INTO traffic_snapshots 
                    (site_id, snapshot_id, window_start, window_end, total_sessions, total_events, fingerprint)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (site_id, snapshot_id, window_start, window_end, total_sessions, total_events, fingerprint))
                
                conn.commit()
                
                return TrafficSnapshot(
                    site_id=site_id,
                    snapshot_id=snapshot_id,
                    window_start=window_start,
                    window_end=window_end,
                    total_sessions=total_sessions,
                    total_events=total_events,
                    fingerprint=fingerprint,
                    created_at=datetime.now(timezone.utc)
                )
    
    def get_traffic_distributions(self, snapshot_id: str) -> Dict[str, Dict[str, float]]:
        """Get traffic distributions for a snapshot"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Get snapshot window
                cur.execute("""
                    SELECT window_start, window_end FROM traffic_snapshots 
                    WHERE snapshot_id = %s
                """, (snapshot_id,))
                
                snapshot = cur.fetchone()
                if not snapshot:
                    raise ValueError(f"Snapshot {snapshot_id} not found")
                
                window_start = snapshot['window_start']
                window_end = snapshot['window_end']
                
                # Get distributions by dimension
                distributions = {}
                
                # Device category distribution
                cur.execute("""
                    WITH base AS (
                        SELECT session_id, MAX(device_category) as device_category
                        FROM ga_events 
                        WHERE event_timestamp >= %s AND event_timestamp < %s
                        GROUP BY session_id
                    ),
                    totals AS (SELECT COUNT(*) as total FROM base)
                    SELECT device_category, COUNT(*)::float / (SELECT total FROM totals) as pct
                    FROM base
                    GROUP BY device_category
                """, (window_start, window_end))
                
                device_dist = {}
                for row in cur.fetchall():
                    device_dist[str(row['device_category'] or 'unknown')] = float(row['pct'])
                distributions['device_category'] = device_dist
                
                # Source/medium distribution
                cur.execute("""
                    WITH base AS (
                        SELECT session_id, MAX(session_source_medium) as session_source_medium
                        FROM ga_events 
                        WHERE event_timestamp >= %s AND event_timestamp < %s
                        GROUP BY session_id
                    ),
                    totals AS (SELECT COUNT(*) as total FROM base)
                    SELECT session_source_medium, COUNT(*)::float / (SELECT total FROM totals) as pct
                    FROM base
                    GROUP BY session_source_medium
                """, (window_start, window_end))
                
                source_dist = {}
                for row in cur.fetchall():
                    source_dist[str(row['session_source_medium'] or 'unknown')] = float(row['pct'])
                distributions['session_source_medium'] = source_dist
                
                # Geo distribution
                cur.execute("""
                    WITH base AS (
                        SELECT session_id, 
                               MAX(CONCAT(COALESCE(geo_country,''), 
                                     CASE WHEN COALESCE(geo_region,'')<>'' 
                                          THEN ':'||geo_region ELSE '' END)) as geo_bucket
                        FROM ga_events 
                        WHERE event_timestamp >= %s AND event_timestamp < %s
                        GROUP BY session_id
                    ),
                    totals AS (SELECT COUNT(*) as total FROM base)
                    SELECT geo_bucket, COUNT(*)::float / (SELECT total FROM totals) as pct
                    FROM base
                    GROUP BY geo_bucket
                """, (window_start, window_end))
                
                geo_dist = {}
                for row in cur.fetchall():
                    geo_dist[str(row['geo_bucket'] or 'unknown')] = float(row['pct'])
                distributions['geo_bucket'] = geo_dist
                
                # Operating system distribution
                cur.execute("""
                    WITH base AS (
                        SELECT session_id, MAX(operating_system) as operating_system
                        FROM ga_events 
                        WHERE event_timestamp >= %s AND event_timestamp < %s
                        GROUP BY session_id
                    ),
                    totals AS (SELECT COUNT(*) as total FROM base)
                    SELECT operating_system, COUNT(*)::float / (SELECT total FROM totals) as pct
                    FROM base
                    GROUP BY operating_system
                """, (window_start, window_end))
                
                os_dist = {}
                for row in cur.fetchall():
                    os_dist[str(row['operating_system'] or 'unknown')] = float(row['pct'])
                distributions['operating_system'] = os_dist
                
                # Gender distribution
                cur.execute("""
                    WITH base AS (
                        SELECT session_id, MAX(gender) as gender
                        FROM ga_events 
                        WHERE event_timestamp >= %s AND event_timestamp < %s
                        GROUP BY session_id
                    ),
                    totals AS (SELECT COUNT(*) as total FROM base)
                    SELECT gender, COUNT(*)::float / (SELECT total FROM totals) as pct
                    FROM base
                    GROUP BY gender
                """, (window_start, window_end))
                
                gender_dist = {}
                for row in cur.fetchall():
                    gender_dist[str(row['gender'] or 'unknown')] = float(row['pct'])
                distributions['gender'] = gender_dist
                
                return distributions
    
    def get_funnel_rates(self, snapshot_id: str) -> Dict[str, float]:
        """Get funnel event rates for a snapshot"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Get snapshot window
                cur.execute("""
                    SELECT window_start, window_end FROM traffic_snapshots 
                    WHERE snapshot_id = %s
                """, (snapshot_id,))
                
                snapshot = cur.fetchone()
                if not snapshot:
                    raise ValueError(f"Snapshot {snapshot_id} not found")
                
                window_start = snapshot['window_start']
                window_end = snapshot['window_end']
                
                # Define Shopify events
                shopify_events = [
                    "page_view",
                    "view_item_list", 
                    "view_item",
                    "add_to_cart",
                    "begin_checkout",
                    "purchase"
                ]
                
                # Get event rates
                cur.execute("""
                    WITH base AS (
                        SELECT session_id, event_name
                        FROM ga_events 
                        WHERE event_timestamp >= %s AND event_timestamp < %s
                        AND event_name = ANY(%s)
                    ),
                    session_events AS (
                        SELECT event_name, COUNT(DISTINCT session_id) as sessions_with_event
                        FROM base GROUP BY event_name
                    ),
                    total AS (
                        SELECT COUNT(DISTINCT session_id) as total_sessions
                        FROM ga_events
                        WHERE event_timestamp >= %s AND event_timestamp < %s
                    )
                    SELECT e.event_name, 
                           (e.sessions_with_event::float / NULLIF(t.total_sessions,0)) as rate
                    FROM session_events e CROSS JOIN total t
                """, (window_start, window_end, shopify_events, window_start, window_end))
                
                rates = {}
                for row in cur.fetchall():
                    rates[row['event_name']] = float(row['rate'] or 0.0)
                
                # Ensure all events are present
                for event in shopify_events:
                    if event not in rates:
                        rates[event] = 0.0
                
                return rates
    
    def save_calibration_results(self, calibration_id: str, site_id: str, snapshot_id: str, 
                                real_distributions: Dict[str, Any], 
                                synthetic_distributions: Dict[str, Any],
                                real_funnel_rates: Dict[str, float],
                                synthetic_funnel_rates: Dict[str, float]) -> None:
        """Save calibration results to database"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO calibration_results 
                    (calibration_id, site_id, snapshot_id, status, real_distributions, 
                     synthetic_distributions, real_funnel_rates, synthetic_funnel_rates)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (calibration_id) DO UPDATE SET
                        site_id = EXCLUDED.site_id,
                        snapshot_id = EXCLUDED.snapshot_id,
                        status = EXCLUDED.status,
                        real_distributions = EXCLUDED.real_distributions,
                        synthetic_distributions = EXCLUDED.synthetic_distributions,
                        real_funnel_rates = EXCLUDED.real_funnel_rates,
                        synthetic_funnel_rates = EXCLUDED.synthetic_funnel_rates,
                        updated_at = NOW()
                """, (calibration_id, site_id, snapshot_id, "complete",
                      json.dumps(real_distributions), json.dumps(synthetic_distributions),
                      json.dumps(real_funnel_rates), json.dumps(synthetic_funnel_rates)))
                
                conn.commit()
    
    def get_calibration_results(self, calibration_id: str) -> Optional[Dict[str, Any]]:
        """Get calibration results from database"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM calibration_results WHERE calibration_id = %s
                """, (calibration_id,))
                
                result = cur.fetchone()
                if not result:
                    return None
                
                return {
                    "calibration_id": result['calibration_id'],
                    "site_id": result['site_id'],
                    "snapshot_id": result['snapshot_id'],
                    "status": result['status'],
                    "real_distributions": json.loads(result['real_distributions'] or '{}'),
                    "synthetic_distributions": json.loads(result['synthetic_distributions'] or '{}'),
                    "real_funnel_rates": json.loads(result['real_funnel_rates'] or '{}'),
                    "synthetic_funnel_rates": json.loads(result['synthetic_funnel_rates'] or '{}'),
                    "created_at": result['created_at'],
                    "updated_at": result['updated_at']
                }
