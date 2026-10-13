"""Agent layer contracts: what agents hand to the backend (and humans)."""

from enum import Enum

from pydantic import BaseModel, Field

from .common import Citation


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class AgentStep(BaseModel):
    step_type: str  # plan | tool_call | observation | verify
    content: str


class AgentResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    steps: list[AgentStep] = Field(default_factory=list)
    needs_approval: bool = False


class AgentQuery(BaseModel):
    question: str
    session_id: str | None = None
    top_k: int = 10


class ApprovalRequest(BaseModel):
    id: str
    action: str
    payload: dict
    reason: str
    status: ApprovalStatus = ApprovalStatus.PENDING


class ApprovalDecision(BaseModel):
    approved: bool
    decided_by: str
    note: str | None = None
