"""
Pydantic response/request models for the NyayaQuest API.

These provide auto-generated OpenAPI docs, client-side type safety,
and validated contracts for all public endpoints.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class Citation(BaseModel):
    section: str
    act: str
    chunk_id: str = ""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class RetrievalMeta(BaseModel):
    total_candidates: int
    reranked_count: int
    fusion_method: Literal["rrf", "weighted"] = "rrf"
    query_rewrites_used: int = 4


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieval_metadata: Optional[RetrievalMeta] = None
    request_id: str = ""


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    conversation_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
