"""AION backend data contracts (v1). See docs/api-contracts.md."""

from .agent import (
    AgentQuery,
    AgentResponse,
    AgentStep,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
)
from .common import Citation, ErrorResponse, PaginationParams
from .customer import (
    CustomerProfile,
    NextBestAction,
    Ticket,
    TicketMessage,
    TicketPriority,
    TicketStatus,
)
from .document import DocumentChunk, IngestRequest, IngestResponse, SourceType
from .eval import EvalCase, EvalResult
from .retrieval import RetrievedChunk, RetrievalQuery, RetrievalResponse

__all__ = [
    "AgentQuery",
    "AgentResponse",
    "AgentStep",
    "ApprovalDecision",
    "ApprovalRequest",
    "ApprovalStatus",
    "Citation",
    "CustomerProfile",
    "DocumentChunk",
    "ErrorResponse",
    "EvalCase",
    "EvalResult",
    "IngestRequest",
    "IngestResponse",
    "NextBestAction",
    "PaginationParams",
    "RetrievalQuery",
    "RetrievalResponse",
    "RetrievedChunk",
    "SourceType",
    "Ticket",
    "TicketMessage",
    "TicketPriority",
    "TicketStatus",
]
