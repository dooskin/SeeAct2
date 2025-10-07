"""
Database adapter for calibration operations
"""
import asyncio
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import psycopg
from ..models.calibration_models import CalibrationStatus, CalibrationFeatures, CalibrationBehaviorMatch


class CalibrationDBAdapter:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    async def create_calibration(self, calibration_id: str, user_id: str, site_id: str, seed: Optional[int] = None) -> str:
        """Create a new calibration record"""
        conn = await psycopg.AsyncConnection.connect(self.connection_string)
        try:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO calibrations (calibration_id, user_id, site_id, status, started_at, seed)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (calibration_id) DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        site_id = EXCLUDED.site_id,
                        status = EXCLUDED.status,
                        started_at = EXCLUDED.started_at,
                        seed = EXCLUDED.seed
                """, (
                    calibration_id,
                    user_id,
                    site_id,
                    "queued",
                    datetime.now(timezone.utc),
                    seed
                ))
                await conn.commit()
                return calibration_id
        finally:
            await conn.close()

    async def update_calibration_status(self, calibration_id: str, status: str, completed_at: Optional[datetime] = None) -> bool:
        """Update calibration status"""
        conn = await psycopg.AsyncConnection.connect(self.connection_string)
        try:
            async with conn.cursor() as cur:
                if completed_at:
                    await cur.execute("""
                        UPDATE calibrations 
                        SET status = %s, completed_at = %s
                        WHERE calibration_id = %s
                    """, (status, completed_at, calibration_id))
                else:
                    await cur.execute("""
                        UPDATE calibrations 
                        SET status = %s
                        WHERE calibration_id = %s
                    """, (status, calibration_id))
                
                await conn.commit()
                return cur.rowcount > 0
        finally:
            await conn.close()

    async def get_calibration_status(self, calibration_id: str) -> Optional[CalibrationStatus]:
        """Get calibration status by ID"""
        conn = await psycopg.AsyncConnection.connect(self.connection_string)
        try:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT calibration_id, user_id, site_id, status, started_at, completed_at, seed
                    FROM calibrations 
                    WHERE calibration_id = %s
                """, (calibration_id,))
                
                row = await cur.fetchone()
                if not row:
                    return None
                
                return CalibrationStatus(
                    calibration_id=row[0],
                    user_id=row[1],
                    site_id=row[2],
                    status=row[3],
                    steps=[],  # Empty steps for now
                    metrics={},  # Empty metrics for now
                    started_at=row[4],
                    finished_at=row[5],
                    seed=row[6]
                )
        finally:
            await conn.close()

    async def list_calibrations(self, user_id: str, site_id: str) -> List[CalibrationStatus]:
        """List calibrations for a user and site"""
        conn = await psycopg.AsyncConnection.connect(self.connection_string)
        try:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT calibration_id, user_id, site_id, status, started_at, completed_at, seed
                    FROM calibrations 
                    WHERE user_id = %s AND site_id = %s
                    ORDER BY started_at DESC
                """, (user_id, site_id))
                
                rows = await cur.fetchall()
                calibrations = []
                for row in rows:
                    calibrations.append(CalibrationStatus(
                        calibration_id=row[0],
                        user_id=row[1],
                        site_id=row[2],
                        status=row[3],
                        steps=[],  # Empty steps for now
                        metrics={},  # Empty metrics for now
                        started_at=row[4],
                        finished_at=row[5],
                        seed=row[6]
                    ))
                
                return calibrations
        finally:
            await conn.close()

    async def save_calibration_features(self, calibration_id: str, features: Dict[str, Any]) -> bool:
        """Save calibration features"""
        conn = await psycopg.AsyncConnection.connect(self.connection_string)
        try:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO calibration_features (calibration_id, features)
                    VALUES (%s, %s)
                    ON CONFLICT (calibration_id) DO UPDATE SET
                        features = EXCLUDED.features,
                        updated_at = CURRENT_TIMESTAMP
                """, (calibration_id, json.dumps(features)))
                
                await conn.commit()
                return cur.rowcount > 0
        finally:
            await conn.close()

    async def get_calibration_features(self, calibration_id: str) -> Optional[Dict[str, Any]]:
        """Get calibration features"""
        conn = await psycopg.AsyncConnection.connect(self.connection_string)
        try:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT features FROM calibration_features 
                    WHERE calibration_id = %s
                """, (calibration_id,))
                
                row = await cur.fetchone()
                if not row:
                    return None
                
                return json.loads(row[0]) if row[0] else None
        finally:
            await conn.close()

    async def save_calibration_behavior_match(self, calibration_id: str, behavior_match: Dict[str, Any]) -> bool:
        """Save calibration behavior match"""
        conn = await psycopg.AsyncConnection.connect(self.connection_string)
        try:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO calibration_behavior_match (calibration_id, behavior_match)
                    VALUES (%s, %s)
                    ON CONFLICT (calibration_id) DO UPDATE SET
                        behavior_match = EXCLUDED.behavior_match,
                        updated_at = CURRENT_TIMESTAMP
                """, (calibration_id, json.dumps(behavior_match)))
                
                await conn.commit()
                return cur.rowcount > 0
        finally:
            await conn.close()

    async def get_calibration_behavior_match(self, calibration_id: str) -> Optional[Dict[str, Any]]:
        """Get calibration behavior match"""
        conn = await psycopg.AsyncConnection.connect(self.connection_string)
        try:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT behavior_match FROM calibration_behavior_match 
                    WHERE calibration_id = %s
                """, (calibration_id,))
                
                row = await cur.fetchone()
                if not row:
                    return None
                
                return json.loads(row[0]) if row[0] else None
        finally:
            await conn.close()
