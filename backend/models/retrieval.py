"""RAG layer contracts: what retrieval hands to the agents."""

from pydantic import BaseModel, Field

from .document import DocumentChunk


class RetrievalQuery(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=50)
    filters: dict = Field(default_factory=dict)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    score: float
    rank: int


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
