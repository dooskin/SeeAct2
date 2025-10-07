"""
Experiments API routes.
"""

from __future__ import annotations

import json
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Request, Header
from fastapi.responses import StreamingResponse

from ..models.experiment_models import (
    ExperimentRequest, ExperimentResponse, ExperimentStatusResponse,
    ExperimentListResponse, ExperimentListItem, ExperimentArtifacts
)
from ..services.experiment_service import ExperimentService
from ..middleware.auth import get_idempotency_key, validate_idempotency_key

router = APIRouter()

# Initialize service
experiment_service = ExperimentService()

# Health check endpoint (must be before parameterized routes)
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "experiments"}


@router.post("", response_model=ExperimentResponse)
@router.post("/", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentRequest,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Create a new experiment"""
    # TODO: Add authentication validation
    # if not authorization or not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Authorization header required")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header required")
    
    # Validate idempotency key
    validate_idempotency_key(idempotency_key)
    
    try:
        response = await experiment_service.create_experiment(request, idempotency_key)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}", response_model=ExperimentStatusResponse)
async def get_experiment(
    experiment_id: str,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Get experiment status and details with live data for running experiments"""
    # TODO: Add authentication validation
    # if not authorization or not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Authorization header required")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header required")
    
    status = await experiment_service.get_experiment_status(experiment_id, user_id)
    if not status:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # If experiment is running, try to get any partial data
    if status.status == "running":
        print(f"[API] Experiment {experiment_id} is running, checking for partial data...")
        
        # Check for partial metrics data
        try:
            partial_metrics = await experiment_service.get_experiment_metrics(experiment_id, user_id)
            if partial_metrics:
                # Add partial data to the response
                status_dict = status.dict()
                status_dict["partial_metrics"] = partial_metrics
                print(f"[API] Found partial metrics for running experiment {experiment_id}")
                return status_dict
        except Exception as e:
            print(f"[API] Error getting partial metrics: {e}")
        
        # Check for partial summary data
        try:
            partial_summary = await experiment_service.get_experiment_summary(experiment_id, user_id)
            if partial_summary:
                # Add partial data to the response
                status_dict = status.dict()
                status_dict["partial_summary"] = partial_summary
                print(f"[API] Found partial summary for running experiment {experiment_id}")
                return status_dict
        except Exception as e:
            print(f"[API] Error getting partial summary: {e}")
    
    return status


@router.get("", response_model=ExperimentListResponse)
async def list_experiments(
    status: Optional[str] = Query(None, description="Filter by status: in_progress, ended"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """List experiments with optional status filter"""
    # TODO: Add authentication validation
    # if not authorization or not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Authorization header required")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header required")
    
    # Map status filter
    status_filter = None
    if status == "in_progress":
        status_filter = "running"
    elif status == "ended":
        status_filter = "complete"
    
    experiments = await experiment_service.list_experiments(user_id, status_filter)
    
    return ExperimentListResponse(
        experiments=experiments,
        total=len(experiments)
    )


@router.get("/{experiment_id}/events")
async def get_experiment_events(
    experiment_id: str,
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    """Get experiment events via SSE"""
    # TODO: Add authentication validation
    # if not authorization or not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Check if experiment exists
    status = await experiment_service.get_experiment_status(experiment_id)
    if not status:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    async def event_generator():
        async for event in experiment_service.get_experiment_events(experiment_id):
            yield event
    
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


@router.get("/{experiment_id}/artifacts", response_model=ExperimentArtifacts)
async def get_experiment_artifacts(
    experiment_id: str,
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    """Get experiment artifacts URLs"""
    # TODO: Add authentication validation
    # if not authorization or not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Check if experiment exists
    status = await experiment_service.get_experiment_status(experiment_id)
    if not status:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    artifacts = await experiment_service.get_experiment_artifacts(experiment_id)
    
    return ExperimentArtifacts(
        experiment_id=experiment_id,
        artifacts=artifacts
    )


@router.get("/{experiment_id}/summary.csv")
async def get_experiment_summary_csv(
    experiment_id: str,
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    """Get experiment summary as CSV"""
    # TODO: Add authentication validation
    # if not authorization or not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Check if experiment exists
    status = await experiment_service.get_experiment_status(experiment_id)
    if not status:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Generate CSV content
    csv_content = f"""experiment_id,site_id,name,status,variant_a_url,variant_b_url,n_agents,concurrency,provider,model,max_cost_usd,seed,min_per_arm,min_completion_ratio,started_at,finished_at,winner,lift_abs,lift_rel,p_value
{status.experiment_id},{status.site_id},{status.name},{status.status},{status.variant_a_url},{status.variant_b_url},{status.n_agents},{status.concurrency},{status.provider},{status.model},{status.max_cost_usd},{status.seed},{status.min_per_arm},{status.min_completion_ratio},{status.started_at},{status.finished_at},{status.winner},{status.lift_abs},{status.lift_rel},{status.p_value}
"""
    
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=experiment_{experiment_id}_summary.csv"
        }
    )


@router.get("/{experiment_id}/A.json")
async def get_variant_a_data(
    experiment_id: str,
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    """Get variant A data as JSON"""
    # TODO: Add authentication validation
    # if not authorization or not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Check if experiment exists
    status = await experiment_service.get_experiment_status(experiment_id)
    if not status:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    variant_data = {
        "experiment_id": experiment_id,
        "variant": "A",
        "url": status.variant_a_url,
        "aggregates": status.aggregates["A"].dict(),
        "sessions": []  # TODO: Add actual session data
    }
    
    return variant_data


@router.get("/{experiment_id}/B.json")
async def get_variant_b_data(
    experiment_id: str,
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    """Get variant B data as JSON"""
    # TODO: Add authentication validation
    # if not authorization or not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Check if experiment exists
    status = await experiment_service.get_experiment_status(experiment_id)
    if not status:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    variant_data = {
        "experiment_id": experiment_id,
        "variant": "B",
        "url": status.variant_b_url,
        "aggregates": status.aggregates["B"].dict(),
        "sessions": []  # TODO: Add actual session data
    }
    
    return variant_data


@router.get("/{experiment_id}/metrics.zip")
async def get_experiment_metrics_zip(
    experiment_id: str,
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    """Get experiment metrics as ZIP file"""
    # TODO: Add authentication validation
    # if not authorization or not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Check if experiment exists
    status = await experiment_service.get_experiment_status(experiment_id)
    if not status:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # TODO: Generate actual ZIP file with metrics
    # For now, return a placeholder
    raise HTTPException(status_code=501, detail="Metrics ZIP generation not yet implemented")


@router.get("/{experiment_id}/metrics")
async def get_experiment_metrics(
    experiment_id: str,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Get experiment metrics for details page"""
    print(f"[API] Get experiment metrics request: {experiment_id}, user_id: {user_id}")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header required")
    
    # Get experiment metrics
    metrics = await experiment_service.get_experiment_metrics(experiment_id, user_id)
    
    if not metrics:
        raise HTTPException(status_code=404, detail="Experiment metrics not found")
    
    return metrics

@router.get("/{experiment_id}/summary")
async def get_experiment_summary(
    experiment_id: str,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Get experiment summary for details page"""
    print(f"[API] Get experiment summary request: {experiment_id}, user_id: {user_id}")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header required")
    
    # Get experiment summary
    summary = await experiment_service.get_experiment_summary(experiment_id, user_id)
    
    if not summary:
        raise HTTPException(status_code=404, detail="Experiment summary not found")
    
    return summary

@router.delete("/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Delete an experiment"""
    print(f"[API] Delete experiment request: {experiment_id}, user_id: {user_id}")
    
    # TODO: Add authentication validation
    # if not authorization or not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Authorization header required")
    
    if not user_id:
        print(f"[API] No user_id provided")
        raise HTTPException(status_code=400, detail="X-User-ID header required")
    
    # Delete experiment
    success = await experiment_service.delete_experiment(experiment_id, user_id)
    print(f"[API] Delete result: {success}")
    
    if not success:
        raise HTTPException(status_code=404, detail="Experiment not found or access denied")
    
    return {"message": "Experiment deleted successfully"}
