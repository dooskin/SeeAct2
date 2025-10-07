"""
Cached database adapter for experiments API with connection pooling and caching.
"""

from __future__ import annotations

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
import asyncio
from functools import lru_cache

from ..models.experiment_models import (
    ExperimentRequest, ExperimentResponse, ExperimentStatusResponse,
    ExperimentStatus, VariantType, VariantAggregates, ExperimentListItem, ProviderType
)


class ExperimentDBAdapter:
    """Database adapter for experiments with connection pooling and caching"""
    
    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.getenv("NEON_DATABASE_URL")
        if not self.dsn:
            raise ValueError("Database DSN not provided")
        
        # Initialize connection pool for better performance
        self.pool = ConnectionPool(
            self.dsn,
            min_size=5,
            max_size=20,
            timeout=60,
            max_lifetime=1800,
            max_idle=300,
            kwargs={"row_factory": dict_row}
        )
        
        # Cache for frequently accessed data
        self._cache = {}
        self._cache_ttl = 30  # 30 seconds cache TTL
        
        # Ensure tables exist
        self._ensure_tables()
    
    def _get_connection(self):
        """Get database connection from pool"""
        return self.pool.getconn()
    
    def _return_connection(self, conn):
        """Return connection to pool"""
        if conn is None:
            return
        try:
            # Only rollback if connection is still open and in transaction
            if not conn.closed and conn.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS:
                conn.rollback()
        except Exception as e:
            # Log the error but don't fail
            print(f"[DB] Warning: Error during connection cleanup: {e}")
        finally:
            try:
                self.pool.putconn(conn)
            except Exception as e:
                print(f"[DB] Error returning connection to pool: {e}")
    
    def _ensure_tables(self):
        """Ensure required tables exist"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Read and execute the schema file
                schema_file = os.path.join(os.path.dirname(__file__), "..", "database", "schema.sql")
                with open(schema_file, 'r') as f:
                    schema_sql = f.read()
                
                # Execute the schema with error handling for existing objects
                try:
                    cur.execute(schema_sql)
                    conn.commit()
                except Exception as e:
                    # Ignore errors for existing objects (tables, triggers, etc.)
                    if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                        conn.rollback()
                        print(f"Schema objects already exist, continuing...")
                    else:
                        raise e
        finally:
            self._return_connection(conn)
    
    def _get_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key"""
        return f"{prefix}:{':'.join(str(arg) for arg in args)}"
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is valid"""
        if key not in self._cache:
            return False
        entry_time, _ = self._cache[key]
        return (datetime.now() - entry_time).seconds < self._cache_ttl
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache"""
        if self._is_cache_valid(key):
            _, data = self._cache[key]
            return data
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Set data in cache"""
        self._cache[key] = (datetime.now(), data)
    
    async def create_experiment(self, request: ExperimentRequest, idempotency_key: Optional[str] = None) -> ExperimentResponse:
        """Create a new experiment in the database"""
        experiment_id = str(uuid.uuid4())
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Check for idempotency
                if idempotency_key:
                    cur.execute("""
                        SELECT id FROM experiments 
                        WHERE idempotency_key = %s AND site_id = %s AND user_id = %s
                        AND status IN ('queued', 'running')
                    """, (idempotency_key, request.site_id, request.user_id))
                    existing = cur.fetchone()
                    if existing:
                        return ExperimentResponse(experiment_id=existing['id'], status=ExperimentStatus.QUEUED)
                
                # Insert new experiment
                cur.execute("""
                    INSERT INTO experiments (
                        id, user_id, site_id, name, status, variant_a_url, variant_b_url,
                        n_agents, seed, provider, model, max_cost_usd, calibration_id,
                        concurrency, min_per_arm, min_completion_ratio, idempotency_key
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    experiment_id, request.user_id, request.site_id, request.name,
                    ExperimentStatus.QUEUED, request.variant_a_url, request.variant_b_url,
                    request.n_agents, request.seed or 12345, request.provider.value,
                    request.model, request.max_cost_usd, request.calibration_id,
                    request.concurrency, request.min_per_arm, request.min_completion_ratio,
                    idempotency_key
                ))
                
                # Insert variant metrics
                cur.execute("""
                    INSERT INTO variant_metrics (experiment_id, variant, n, purchases, cr)
                    VALUES (%s, 'A', 0, 0, 0.0), (%s, 'B', 0, 0, 0.0)
                """, (experiment_id, experiment_id))
                
                conn.commit()
                
                # Clear cache for this user
                self._clear_user_cache(request.user_id)
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self._return_connection(conn)
        
        return ExperimentResponse(experiment_id=experiment_id, status=ExperimentStatus.QUEUED)
    
    def _clear_user_cache(self, user_id: str):
        """Clear cache for a specific user"""
        keys_to_remove = [key for key in self._cache.keys() if f"user:{user_id}" in key]
        for key in keys_to_remove:
            del self._cache[key]
    
    async def get_experiment_status(self, experiment_id: str, user_id: str) -> Optional[ExperimentStatusResponse]:
        """Get experiment status for a specific user"""
        cache_key = self._get_cache_key("experiment", experiment_id, user_id)
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT e.*, 
                           COALESCE(vm_a.n, 0) as variant_a_finished,
                           COALESCE(vm_a.purchases, 0) as variant_a_purchases,
                           COALESCE(vm_a.cr, 0.0) as variant_a_cr,
                           COALESCE(vm_b.n, 0) as variant_b_finished,
                           COALESCE(vm_b.purchases, 0) as variant_b_purchases,
                           COALESCE(vm_b.cr, 0.0) as variant_b_cr
                    FROM experiments e
                    LEFT JOIN variant_metrics vm_a ON e.id = vm_a.experiment_id AND vm_a.variant = 'A'
                    LEFT JOIN variant_metrics vm_b ON e.id = vm_b.experiment_id AND vm_b.variant = 'B'
                    WHERE e.id = %s AND e.user_id = %s
                """, (experiment_id, user_id))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                # Get aggregates
                aggregates = await self._get_experiment_aggregates(experiment_id)
                
                result = ExperimentStatusResponse(
                    experiment_id=str(row['id']),
                    user_id=row['user_id'],
                    site_id=row['site_id'],
                    name=row['name'],
                    status=ExperimentStatus(row['status']),
                    variant_a_url=row['variant_a_url'],
                    variant_b_url=row['variant_b_url'],
                    n_agents=row['n_agents'],
                    calibration_id=row['calibration_id'],
                    concurrency=row['concurrency'],
                    provider=ProviderType(row['provider']),
                    model=row['model'],
                    max_cost_usd=float(row['max_cost_usd']),
                    seed=row['seed'],
                    min_per_arm=row['min_per_arm'],
                    min_completion_ratio=float(row['min_completion_ratio']),
                    aggregates=aggregates,
                    winner=VariantType(row['result']) if row['result'] else None,
                    lift_abs=float(row['lift_abs']) if row['lift_abs'] else None,
                    lift_rel=float(row['lift_rel']) if row['lift_rel'] else None,
                    p_value=float(row['p_value']) if row['p_value'] else None,
                    started_at=row['started_at'],
                    finished_at=row['finished_at'],
                    error=row['error']
                )
                
                # Cache the result
                self._set_cache(cache_key, result)
                return result
        finally:
            self._return_connection(conn)
    
    async def list_experiments(self, user_id: str, status_filter: Optional[str] = None) -> List[ExperimentListItem]:
        """List experiments for a specific user with caching"""
        cache_key = self._get_cache_key("experiments", user_id, status_filter or "all")
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Single optimized query with JOINs to get all data at once
                query = """
                    SELECT e.id, e.user_id, e.name, e.site_id, 
                           e.status, e.started_at, e.finished_at, e.created_at,
                           e.n_agents, e.provider, e.model, e.concurrency, e.max_cost_usd,
                           e.lift_rel, e.result,
                           COALESCE(vm_a.n, 0) as variant_a_finished,
                           COALESCE(vm_a.purchases, 0) as variant_a_purchases,
                           COALESCE(vm_a.cr, 0.0) as variant_a_cr,
                           COALESCE(vm_b.n, 0) as variant_b_finished,
                           COALESCE(vm_b.purchases, 0) as variant_b_purchases,
                           COALESCE(vm_b.cr, 0.0) as variant_b_cr
                    FROM experiments e
                    LEFT JOIN variant_metrics vm_a ON e.id = vm_a.experiment_id AND vm_a.variant = 'A'
                    LEFT JOIN variant_metrics vm_b ON e.id = vm_b.experiment_id AND vm_b.variant = 'B'
                    WHERE e.user_id = %s
                """
                params = [user_id]
                
                if status_filter:
                    query += " AND e.status = %s"
                    params.append(status_filter)
                
                query += " ORDER BY e.created_at DESC"
                
                cur.execute(query, params)
                rows = cur.fetchall()
                
                experiments = []
                for row in rows:
                    # Build aggregates from the joined data (no additional DB calls!)
                    aggregates = {
                        "A": {
                            "finished": row['variant_a_finished'],
                            "purchases": row['variant_a_purchases'],
                            "cr": float(row['variant_a_cr'])
                        },
                        "B": {
                            "finished": row['variant_b_finished'],
                            "purchases": row['variant_b_purchases'],
                            "cr": float(row['variant_b_cr'])
                        }
                    }
                    
                    experiments.append(ExperimentListItem(
                        id=str(row['id']),
                        experiment_id=str(row['id']),  # For frontend compatibility
                        user_id=row['user_id'],
                        name=row['name'],
                        site_id=row['site_id'],
                        date=row['created_at'].strftime('%Y-%m-%d') if row['created_at'] else '',
                        status=ExperimentStatus(row['status']),
                        lift_rel=float(row['lift_rel']) if row['lift_rel'] else None,
                        result=row['result'],
                        aggregates=aggregates
                    ))
                
                # Cache the result
                self._set_cache(cache_key, experiments)
                return experiments
        finally:
            self._return_connection(conn)
    
    async def _get_experiment_aggregates(self, experiment_id: str) -> Dict[str, Any]:
        """Get aggregates data for an experiment"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Get variant metrics
                cur.execute("""
                    SELECT variant, n, purchases, cr
                    FROM variant_metrics 
                    WHERE experiment_id = %s
                """, (experiment_id,))
                
                rows = cur.fetchall()
                aggregates = {}
                
                for row in rows:
                    variant = row['variant']
                    aggregates[variant] = {
                        "finished": row['n'] or 0,
                        "purchases": row['purchases'] or 0,
                        "cr": float(row['cr'] or 0.0)
                    }
                
                # Ensure both variants are present
                if "A" not in aggregates:
                    aggregates["A"] = {"finished": 0, "purchases": 0, "cr": 0.0}
                if "B" not in aggregates:
                    aggregates["B"] = {"finished": 0, "purchases": 0, "cr": 0.0}
                
                return aggregates
        finally:
            self._return_connection(conn)
    
    async def update_experiment_status(self, experiment_id: str, user_id: str, status: ExperimentStatus, 
                                     error: Optional[str] = None, finished_at: Optional[datetime] = None):
        """Update experiment status"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE experiments 
                    SET status = %s, error = %s, finished_at = %s
                    WHERE id = %s AND user_id = %s
                """, (status.value, error, finished_at, experiment_id, user_id))
                conn.commit()
                
                # Clear cache for this user
                self._clear_user_cache(user_id)
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self._return_connection(conn)
    
    async def update_variant_metrics(self, experiment_id: str, variant: str, 
                                   finished: int, purchases: int, cr: float):
        """Update variant metrics"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE variant_metrics 
                    SET n = %s, purchases = %s, cr = %s
                    WHERE experiment_id = %s AND variant = %s
                """, (finished, purchases, cr, experiment_id, variant))
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self._return_connection(conn)
    
    async def update_experiment_results(self, experiment_id: str, user_id: str, 
                                      winner: Optional[str], lift_abs: Optional[float],
                                      lift_rel: Optional[float], p_value: Optional[float]):
        """Update experiment final results"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE experiments 
                    SET result = %s, lift_abs = %s, lift_rel = %s, p_value = %s
                    WHERE id = %s AND user_id = %s
                """, (winner, lift_abs, lift_rel, p_value, experiment_id, user_id))
                conn.commit()
                
                # Clear cache for this user
                self._clear_user_cache(user_id)
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self._return_connection(conn)
    
    async def save_agent_sessions(self, experiment_id: str, sessions: List[Any]):
        """Save agent sessions to database"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                for session in sessions:
                    cur.execute("""
                        INSERT INTO agent_sessions (
                            experiment_id, session_id, variant, status, 
                            started_at, finished_at, error, events_jsonb
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        experiment_id,
                        session.get('session_id'),
                        session.get('variant'),
                        session.get('status'),
                        session.get('started_at'),
                        session.get('finished_at'),
                        session.get('error'),
                        json.dumps(session.get('metrics', {}))
                    ))
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self._return_connection(conn)
    
    async def delete_experiment(self, experiment_id: str, user_id: str) -> bool:
        """Delete an experiment and all related data"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                print(f"[DB] Attempting to delete experiment {experiment_id} for user {user_id}")
                
                # Check if experiment exists and belongs to user
                cur.execute("""
                    SELECT id FROM experiments 
                    WHERE id = %s AND user_id = %s
                """, (experiment_id, user_id))
                
                if not cur.fetchone():
                    print(f"[DB] Experiment {experiment_id} not found or user mismatch")
                    return False
                
                # Delete related data (cascading delete should handle this, but being explicit)
                cur.execute("DELETE FROM agent_sessions WHERE experiment_id = %s", (experiment_id,))
                cur.execute("DELETE FROM experiment_events WHERE experiment_id = %s", (experiment_id,))
                cur.execute("DELETE FROM variant_metrics WHERE experiment_id = %s", (experiment_id,))
                cur.execute("DELETE FROM experiments WHERE id = %s", (experiment_id,))
                
                conn.commit()
                print(f"[DB] Successfully deleted experiment {experiment_id}")
                
                # Clear cache for this user
                self._clear_user_cache(user_id)
                return True
        except Exception as e:
            print(f"[DB] Error deleting experiment {experiment_id}: {e}")
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)
    
    def close(self):
        """Close the connection pool"""
        if hasattr(self, 'pool'):
            self.pool.close()
