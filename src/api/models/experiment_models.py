"""
Pydantic models for experiments API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


class ExperimentStatus(str, Enum):
    """Experiment status enumeration"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


class VariantType(str, Enum):
    """Variant type enumeration"""
    A = "A"
    B = "B"


class ProviderType(str, Enum):
    """Provider type enumeration"""
    BROWSERBASE = "browserbase"
    LOCAL = "local"
    CDP = "cdp"


class ExperimentRequest(BaseModel):
    """Request to create an experiment"""
    user_id: str
    site_id: str
    name: str
    variant_a_url: str
    variant_b_url: str
    n_agents: int = Field(ge=1, le=10000, description="Number of agents to run")
    calibration_id: str
    concurrency: int = Field(ge=1, le=100, description="Concurrent agents")
    provider: ProviderType = ProviderType.BROWSERBASE
    model: str = "gpt-4o"
    max_cost_usd: float = Field(ge=0.01, le=10000.0, description="Maximum cost in USD")
    seed: Optional[int] = None
    min_per_arm: int = Field(ge=1, le=1000, default=200, description="Minimum samples per variant")
    min_completion_ratio: float = Field(ge=0.1, le=1.0, default=0.8, description="Minimum completion ratio")


class ExperimentResponse(BaseModel):
    """Response from creating an experiment"""
    experiment_id: str
    status: ExperimentStatus


class VariantAggregates(BaseModel):
    """Aggregated metrics for a variant"""
    finished: int
    purchases: int
    cr: float  # conversion rate
    add_to_cart_rate: Optional[float] = None
    begin_checkout_rate: Optional[float] = None
    purchase_rate: Optional[float] = None


class ExperimentStatusResponse(BaseModel):
    """Response for experiment status"""
    experiment_id: str
    user_id: str
    site_id: str
    name: str
    status: ExperimentStatus
    variant_a_url: str
    variant_b_url: str
    n_agents: int
    calibration_id: str
    concurrency: int
    provider: ProviderType
    model: str
    max_cost_usd: float
    seed: Optional[int]
    min_per_arm: int
    min_completion_ratio: float
    aggregates: Dict[VariantType, VariantAggregates]
    winner: Optional[VariantType] = None
    lift_abs: Optional[float] = None
    lift_rel: Optional[float] = None
    p_value: Optional[float] = None
    started_at: datetime
    finished_at: Optional[datetime] = None
    error: Optional[str] = None


class ExperimentListItem(BaseModel):
    """Item in experiment list response"""
    id: str
    experiment_id: str
    user_id: str
    name: str
    site_id: str
    date: str
    status: ExperimentStatus
    lift_rel: Optional[float] = None
    result: Optional[str] = None
    aggregates: Optional[Dict[str, Any]] = None


class ExperimentListResponse(BaseModel):
    """Response for experiment list"""
    experiments: List[ExperimentListItem]
    total: int


class AgentSession(BaseModel):
    """Agent session data"""
    session_id: str
    experiment_id: str
    prompt_id: str
    variant: VariantType
    status: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    purchased: bool = False
    metrics_path: Optional[str] = None
    events_jsonb: Optional[Dict[str, Any]] = None


class FunnelEvent(BaseModel):
    """Funnel event data"""
    session_id: str
    variant: VariantType
    event: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class ProgressEvent(BaseModel):
    """Progress event data"""
    variant: VariantType
    finished: int
    purchases: int
    cr: float
    total_agents: int


class ExperimentEvent(BaseModel):
    """SSE event for experiment progress"""
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime


class ExperimentArtifacts(BaseModel):
    """Experiment artifacts response"""
    experiment_id: str
    artifacts: Dict[str, str]  # artifact_type -> URL


class StatisticalResult(BaseModel):
    """Statistical analysis result"""
    winner: Optional[VariantType]
    lift_abs: float
    lift_rel: float
    p_value: float
    confidence_interval: Optional[Dict[str, float]] = None
    power: Optional[float] = None
    sample_size: int


class CostTracking(BaseModel):
    """Cost tracking data"""
    total_cost_usd: float
    estimated_remaining_cost: float
    cost_per_agent: float
    budget_utilization: float  # percentage


class ExperimentMetrics(BaseModel):
    """Comprehensive experiment metrics"""
    experiment_id: str
    total_agents: int
    completed_agents: int
    completion_rate: float
    cost_tracking: CostTracking
    statistical_result: Optional[StatisticalResult] = None
    variant_metrics: Dict[VariantType, VariantAggregates]
    funnel_events: List[FunnelEvent]
    agent_sessions: List[AgentSession]
