from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Header, Query
from fastapi.responses import StreamingResponse
from typing import List

from ..adapters.ga_neon_adapter import EnhancedGANeonAdapter
from ..adapters.calibration_db_adapter import CalibrationDBAdapter
from ..services.calibration_service import CalibrationService
from ..database.connection import get_database_connection
from ..models.calibration_models import (
    CalibrationRequest, CalibrationResponse, CalibrationStatus,
    CalibrationFeatures, CalibrationBehaviorMatch, CalibrationEvent
)

router = APIRouter()

# Initialize services lazily
ga_adapter = None
calibration_service = None

def get_calibration_service():
    global calibration_service
    if calibration_service is None:
        global ga_adapter
        if ga_adapter is None:
            ga_adapter = EnhancedGANeonAdapter()
        
        # Get database connection string from unified service
        db_connection = get_database_connection()
        db_adapter = CalibrationDBAdapter(db_connection.get_connection_string())
        
        calibration_service = CalibrationService(ga_adapter, db_adapter)
    return calibration_service

# Health check endpoint (must be before parameterized routes)
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "calibrations"}

@router.get("", response_model=List[CalibrationStatus])
@router.get("/", response_model=List[CalibrationStatus])
async def list_calibrations(user_id: str = Header(..., alias="X-User-ID"), site_id: str = Query(...)):
    """List calibrations for a user and site"""
    try:
        service = get_calibration_service()
        calibrations = await service.list_calibrations(user_id, site_id)
        return calibrations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=CalibrationResponse)
@router.post("/", response_model=CalibrationResponse)
async def start_calibration(
    request: CalibrationRequest, 
    user_id: str = Header(..., alias="X-User-ID")
):
    """Start a calibration job"""
    try:
        service = get_calibration_service()
        calibration_id = await service.start_calibration(
            user_id, request.site_id, request.seed
        )
        return CalibrationResponse(
            calibration_id=calibration_id,
            status="queued"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{calibration_id}", response_model=CalibrationStatus)
async def get_calibration_status(calibration_id: str):
    """Get calibration status"""
    service = get_calibration_service()
    status = await service.get_calibration_status(calibration_id)
    if not status:
        raise HTTPException(status_code=404, detail="Calibration not found")
    
    return status

@router.get("/{calibration_id}/features", response_model=CalibrationFeatures)
async def get_calibration_features(calibration_id: str):
    """Get calibration features"""
    service = get_calibration_service()
    features = service.get_calibration_features(calibration_id)
    if not features:
        raise HTTPException(status_code=404, detail="Features not ready")
    
    return CalibrationFeatures(**features)

@router.get("/{calibration_id}/behavior-match", response_model=CalibrationBehaviorMatch)
async def get_calibration_behavior_match(calibration_id: str):
    """Get calibration behavior match"""
    service = get_calibration_service()
    behavior_match = service.get_calibration_behavior_match(calibration_id)
    if not behavior_match:
        raise HTTPException(status_code=404, detail="Behavior match not ready")
    
    return CalibrationBehaviorMatch(**behavior_match)

@router.get("/{calibration_id}/events")
async def get_calibration_events(calibration_id: str):
    """Get calibration events via SSE"""
    service = get_calibration_service()
    status = await service.get_calibration_status(calibration_id)
    if not status:
        raise HTTPException(status_code=404, detail="Calibration not found")
    
    async def event_generator():
        # Send queued event
        yield f"event: queued\n"
        yield f"data: {json.dumps({'calibration_id': calibration_id})}\n\n"
        
        # Monitor calibration progress
        last_step = -1
        while status.status in ["queued", "running"]:
            current_step = -1
            for i, step in enumerate(status.steps):
                if step.status == "complete":
                    current_step = i
                elif step.status == "running":
                    current_step = i
                    break
            
            # Send events for completed steps
            if current_step > last_step:
                if current_step >= 0 and status.steps[current_step].status == "complete":
                    step_name = status.steps[current_step].name
                    if step_name == "fetch_ga_snapshot_neon":
                        yield f"event: ga_snapshot_neon_complete\n"
                        yield f"data: {json.dumps({'calibration_id': calibration_id})}\n\n"
                    elif step_name == "generate_prompts":
                        yield f"event: prompts_generated\n"
                        yield f"data: {json.dumps({'count': status.metrics.get('num_prompts', 0)})}\n\n"
                    elif step_name == "write_real_distributions":
                        yield f"event: features_real_ready\n"
                        yield f"data: {json.dumps({'calibration_id': calibration_id})}\n\n"
                    elif step_name == "write_synthetic_distributions":
                        yield f"event: features_synth_ready\n"
                        yield f"data: {json.dumps({'calibration_id': calibration_id})}\n\n"
                    elif step_name == "write_real_funnel_rates":
                        yield f"event: behavior_real_ready\n"
                        yield f"data: {json.dumps({'calibration_id': calibration_id})}\n\n"
                    elif step_name == "write_synthetic_funnel_rates":
                        yield f"event: behavior_synth_ready\n"
                        yield f"data: {json.dumps({'calibration_id': calibration_id})}\n\n"
                    elif step_name == "complete":
                        yield f"event: complete\n"
                        yield f"data: {json.dumps({'calibration_id': calibration_id})}\n\n"
                        break
                
                last_step = current_step
            
            # Send keep-alive
            yield f":ka\n\n"
            await asyncio.sleep(1)
        
        # Send final events
        if status.status == "complete":
            yield f"event: complete\n"
            yield f"data: {json.dumps({'calibration_id': calibration_id})}\n\n"
        elif status.status == "error":
            yield f"event: error\n"
            yield f"data: {json.dumps({'message': status.error or 'Unknown error'})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )
