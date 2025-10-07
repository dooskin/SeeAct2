"""
Pydantic models for calibration API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class CalibrationStep(BaseModel):
    """Represents a single step in the calibration process"""
    name: str
    status: str  # "pending", "running", "complete", "error"


class CalibrationStatus(BaseModel):
    """Represents the status of a calibration process"""
    calibration_id: str
    user_id: str
    site_id: str
    status: str  # "queued", "running", "complete", "error"
    steps: List[CalibrationStep]
    metrics: Dict[str, Any]
    started_at: datetime
    finished_at: Optional[datetime] = None
    error: Optional[str] = None


class CalibrationRequest(BaseModel):
    """Request to start a calibration"""
    site_id: str
    seed: Optional[int] = None


class CalibrationResponse(BaseModel):
    """Response from starting a calibration"""
    calibration_id: str
    status: str


class CalibrationFeatures(BaseModel):
    """Calibration features (distributions)"""
    distributions: List[Dict[str, Any]]


class CalibrationBehaviorMatch(BaseModel):
    """Calibration behavior match (funnel rates)"""
    real: List[Dict[str, Any]]
    synthetic: List[Dict[str, Any]]


class CalibrationEvent(BaseModel):
    """SSE event for calibration progress"""
    event: str
    data: Dict[str, Any]
    timestamp: datetime
