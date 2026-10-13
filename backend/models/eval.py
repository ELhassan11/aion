"""Evaluation contracts: cases and per-methodology results."""

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str
    question: str
    expected_answer: str
    context_document_ids: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    case_id: str
    methodology: str  # e.g. "naive-rag" | "hybrid+rerank"
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    notes: str | None = None
