"""Shared base schemas used across all layers."""

from pydantic import BaseModel, Field


class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict | None = None


class Citation(BaseModel):
    """Links a generated statement back to its source chunk. Mandatory."""

    chunk_id: str
    document_id: str
    source_uri: str
    page: int | None = None
    excerpt: str
