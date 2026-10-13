"""Ingestion layer contracts: what ingestion hands to storage / RAG."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"
    DATABASE = "database"
    API = "api"


class DocumentChunk(BaseModel):
    """Immutable unit of text. Corrections create a new chunk, never edit."""

    id: str
    document_id: str
    source_type: SourceType
    source_uri: str
    text: str
    chunk_index: int
    parent_chunk_id: str | None = None
    page: int | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IngestRequest(BaseModel):
    source_type: SourceType
    source_uri: str
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int
    status: str
