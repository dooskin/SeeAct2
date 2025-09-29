"""
Calibration service implementing the 7-step calibration process.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import random

from ..adapters.ga_neon_adapter import EnhancedGANeonAdapter, TrafficSnapshot
from ..models.calibration_models import CalibrationStep, CalibrationStatus


class CalibrationService:
    """Service for managing calibration processes"""
    
    def __init__(self, ga_adapter: EnhancedGANeonAdapter):
        self.ga_adapter = ga_adapter
        self.active_calibrations: Dict[str, CalibrationStatus] = {}
    
    async def start_calibration(self, site_id: str, seed: Optional[int] = None) -> str:
        """Start a new calibration process"""
        calibration_id = str(uuid.uuid4())
        
        # Initialize calibration status
        status = CalibrationStatus(
            calibration_id=calibration_id,
            site_id=site_id,
            status="queued",
            steps=[
                CalibrationStep(name="fetch_ga_snapshot_neon", status="pending"),
                CalibrationStep(name="generate_prompts", status="pending"),
                CalibrationStep(name="write_real_distributions", status="pending"),
                CalibrationStep(name="write_synthetic_distributions", status="pending"),
                CalibrationStep(name="write_real_funnel_rates", status="pending"),
                CalibrationStep(name="write_synthetic_funnel_rates", status="pending"),
                CalibrationStep(name="complete", status="pending")
            ],
            metrics={},
            started_at=datetime.now(timezone.utc),
            finished_at=None,
            error=None
        )
        
        self.active_calibrations[calibration_id] = status
        
        # Start background process
        asyncio.create_task(self._run_calibration_process(calibration_id, site_id, seed))
        
        return calibration_id
    
    async def get_calibration_status(self, calibration_id: str) -> Optional[CalibrationStatus]:
        """Get calibration status"""
        return self.active_calibrations.get(calibration_id)
    
    async def _run_calibration_process(self, calibration_id: str, site_id: str, seed: Optional[int] = None):
        """Run the 7-step calibration process"""
        try:
            status = self.active_calibrations[calibration_id]
            status.status = "running"
            
            # Step 1: Fetch GA snapshot
            await self._step_1_fetch_ga_snapshot(calibration_id, site_id)
            
            # Step 2: Generate prompts
            await self._step_2_generate_prompts(calibration_id, site_id, seed)
            
            # Step 3: Write real traffic distributions
            await self._step_3_write_real_distributions(calibration_id, site_id)
            
            # Step 4: Write synthetic distributions
            await self._step_4_write_synthetic_distributions(calibration_id, site_id)
            
            # Step 5: Write real funnel rates
            await self._step_5_write_real_funnel_rates(calibration_id, site_id)
            
            # Step 6: Write synthetic funnel rates
            await self._step_6_write_synthetic_funnel_rates(calibration_id, site_id)
            
            # Step 7: Complete
            await self._step_7_complete(calibration_id, site_id)
            
        except Exception as e:
            status = self.active_calibrations.get(calibration_id)
            if status:
                status.status = "error"
                status.error = str(e)
                status.finished_at = datetime.now(timezone.utc)
    
    async def _step_1_fetch_ga_snapshot(self, calibration_id: str, site_id: str):
        """Step 1: Fetch & snapshot Neon traffic for site_id"""
        status = self.active_calibrations[calibration_id]
        status.steps[0].status = "running"
        
        # Simulate async work
        await asyncio.sleep(1)
        
        # Create traffic snapshot
        snapshot = self.ga_adapter.create_traffic_snapshot(site_id)
        
        # Store snapshot ID for later use
        status.metrics["snapshot_id"] = snapshot.snapshot_id
        status.metrics["total_sessions"] = snapshot.total_sessions
        status.metrics["total_events"] = snapshot.total_events
        
        status.steps[0].status = "complete"
    
    async def _step_2_generate_prompts(self, calibration_id: str, site_id: str, seed: Optional[int] = None):
        """Step 2: Generate 1,000 prompts proportional to distributions"""
        status = self.active_calibrations[calibration_id]
        status.steps[1].status = "running"
        
        # Simulate async work
        await asyncio.sleep(2)
        
        # Generate mock prompts (in real implementation, this would use LLM)
        if seed:
            random.seed(seed)
        
        num_prompts = 1000
        prompts = []
        
        # Generate prompts based on traffic distributions
        snapshot_id = status.metrics.get("snapshot_id")
        if snapshot_id:
            distributions = self.ga_adapter.get_traffic_distributions(snapshot_id)
            
            # Generate prompts proportional to device category distribution
            device_dist = distributions.get("device_category", {})
            for device, pct in device_dist.items():
                count = int(num_prompts * pct)
                for i in range(count):
                    prompts.append({
                        "id": str(uuid.uuid4()),
                        "device_category": device,
                        "prompt_text": f"Browse the {site_id} website as a {device} user"
                    })
        
        status.metrics["num_prompts"] = len(prompts)
        status.metrics["prompts"] = prompts
        
        status.steps[1].status = "complete"
    
    async def _step_3_write_real_distributions(self, calibration_id: str, site_id: str):
        """Step 3: Write real traffic distributions (inner pie)"""
        status = self.active_calibrations[calibration_id]
        status.steps[2].status = "running"
        
        # Simulate async work
        await asyncio.sleep(1)
        
        snapshot_id = status.metrics.get("snapshot_id")
        if snapshot_id:
            distributions = self.ga_adapter.get_traffic_distributions(snapshot_id)
            status.metrics["real_distributions"] = distributions
        
        status.steps[2].status = "complete"
    
    async def _step_4_write_synthetic_distributions(self, calibration_id: str, site_id: str):
        """Step 4: Write synthetic distributions (outer pie)"""
        status = self.active_calibrations[calibration_id]
        status.steps[3].status = "running"
        
        # Simulate async work
        await asyncio.sleep(1)
        
        # For now, synthetic distributions match real distributions
        # In real implementation, this would be generated from prompts
        real_distributions = status.metrics.get("real_distributions", {})
        status.metrics["synthetic_distributions"] = real_distributions.copy()
        
        status.steps[3].status = "complete"
    
    async def _step_5_write_real_funnel_rates(self, calibration_id: str, site_id: str):
        """Step 5: Write real funnel rates (six Shopify events)"""
        status = self.active_calibrations[calibration_id]
        status.steps[4].status = "running"
        
        # Simulate async work
        await asyncio.sleep(1)
        
        snapshot_id = status.metrics.get("snapshot_id")
        if snapshot_id:
            funnel_rates = self.ga_adapter.get_funnel_rates(snapshot_id)
            status.metrics["real_funnel_rates"] = funnel_rates
        
        status.steps[4].status = "complete"
    
    async def _step_6_write_synthetic_funnel_rates(self, calibration_id: str, site_id: str):
        """Step 6: Write synthetic target funnel rates"""
        status = self.active_calibrations[calibration_id]
        status.steps[5].status = "running"
        
        # Simulate async work
        await asyncio.sleep(1)
        
        # For now, synthetic funnel rates match real rates
        # In real implementation, this would be derived from prompt metadata
        real_funnel_rates = status.metrics.get("real_funnel_rates", {})
        status.metrics["synthetic_funnel_rates"] = real_funnel_rates.copy()
        
        status.steps[5].status = "complete"
    
    async def _step_7_complete(self, calibration_id: str, site_id: str):
        """Step 7: Emit SSE waypoints; mark complete"""
        status = self.active_calibrations[calibration_id]
        status.steps[6].status = "running"
        
        # Simulate async work
        await asyncio.sleep(1)
        
        # Save results to database
        snapshot_id = status.metrics.get("snapshot_id")
        real_distributions = status.metrics.get("real_distributions", {})
        synthetic_distributions = status.metrics.get("synthetic_distributions", {})
        real_funnel_rates = status.metrics.get("real_funnel_rates", {})
        synthetic_funnel_rates = status.metrics.get("synthetic_funnel_rates", {})
        
        if snapshot_id:
            self.ga_adapter.save_calibration_results(
                calibration_id, site_id, snapshot_id,
                real_distributions, synthetic_distributions,
                real_funnel_rates, synthetic_funnel_rates
            )
        
        status.steps[6].status = "complete"
        status.status = "complete"
        status.finished_at = datetime.now(timezone.utc)
    
    def get_calibration_features(self, calibration_id: str) -> Optional[Dict[str, Any]]:
        """Get calibration features (distributions)"""
        status = self.active_calibrations.get(calibration_id)
        if not status:
            return None
        
        real_distributions = status.metrics.get("real_distributions", {})
        synthetic_distributions = status.metrics.get("synthetic_distributions", {})
        
        # Convert to API format
        distributions = []
        
        for dimension, real_data in real_distributions.items():
            # Real distribution
            real_buckets = [{"bucket": k, "pct": v} for k, v in real_data.items()]
            distributions.append({
                "dimension": dimension,
                "kind": "real",
                "buckets": real_buckets
            })
            
            # Synthetic distribution
            synthetic_data = synthetic_distributions.get(dimension, {})
            synthetic_buckets = [{"bucket": k, "pct": v} for k, v in synthetic_data.items()]
            distributions.append({
                "dimension": dimension,
                "kind": "synthetic",
                "buckets": synthetic_buckets
            })
        
        return {"distributions": distributions}
    
    def get_calibration_behavior_match(self, calibration_id: str) -> Optional[Dict[str, Any]]:
        """Get calibration behavior match (funnel rates)"""
        status = self.active_calibrations.get(calibration_id)
        if not status:
            return None
        
        real_funnel_rates = status.metrics.get("real_funnel_rates", {})
        synthetic_funnel_rates = status.metrics.get("synthetic_funnel_rates", {})
        
        # Convert to API format
        real_events = [{"event": k, "rate": v} for k, v in real_funnel_rates.items()]
        synthetic_events = [{"event": k, "rate": v} for k, v in synthetic_funnel_rates.items()]
        
        return {
            "real": real_events,
            "synthetic": synthetic_events
        }
