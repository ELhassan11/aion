"""Customer Intelligence contracts: the unified customer view."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class CustomerProfile(BaseModel):
    """Customer 360: the single unified view."""

    id: str
    name: str
    phone: str | None = None
    email: str | None = None
    segment: str | None = None
    lifetime_value: float = 0.0
    churn_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    last_interaction_at: datetime | None = None
    # Computed aggregates (filled by the Customer 360 builder, P1-T05).
    ticket_count: int = Field(default=0, ge=0)
    open_ticket_count: int = Field(default=0, ge=0)
    channels: list[str] = Field(default_factory=list)


class TicketMessage(BaseModel):
    role: str  # customer | agent | ai
    text: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Ticket(BaseModel):
    id: str
    customer_id: str
    channel: str  # whatsapp | messenger | ...
    subject: str
    status: TicketStatus = TicketStatus.OPEN
    priority: TicketPriority = TicketPriority.MEDIUM
    messages: list[TicketMessage] = Field(default_factory=list)
    ai_summary: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class NextBestAction(BaseModel):
    customer_id: str
    action: str
    reason: str
    priority_score: float = Field(ge=0.0, le=1.0)
