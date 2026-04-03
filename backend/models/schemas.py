"""
VISIO.ID — Pydantic Schemas
Centralized request/response models untuk semua API endpoints.

Semua request models menggunakan field_validator untuk input sanitization.
Semua response mengikuti format: {status, data, error}.
"""

from typing import Any, Optional
from pydantic import BaseModel, HttpUrl, field_validator


# ─── Standard Response Wrapper ───────────────────────────────────────

class PipelineResponse(BaseModel):
    """Standard API response format untuk semua endpoints."""
    status: str           # "ok" | "error"
    data: Optional[Any]   # payload
    error: Optional[str]  # pesan error, null kalau sukses


# ─── Pipeline Request Models ─────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    """
    Request untuk trigger full pipeline (Stage 1-3).
    Digunakan oleh endpoint POST /pipeline/run.
    """
    brand_url: HttpUrl
    brand_name: str
    industry: str = "skincare"
    max_pages: int = 50
    stages: list[int] = [1, 2, 3]

    @field_validator("brand_name")
    @classmethod
    def brand_name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("brand_name tidak boleh kosong")
        return v.strip()

    @field_validator("max_pages")
    @classmethod
    def max_pages_valid(cls, v: int) -> int:
        if v < 1 or v > 50:
            raise ValueError("max_pages harus antara 1 dan 50")
        return v

    @field_validator("stages")
    @classmethod
    def stages_valid(cls, v: list[int]) -> list[int]:
        for s in v:
            if s not in [1, 2, 3]:
                raise ValueError("stages hanya boleh berisi 1, 2, atau 3")
        return sorted(set(v))


class PipelineStageRequest(BaseModel):
    """
    Request untuk trigger satu stage saja (Process atau Embed).
    Digunakan oleh endpoint POST /pipeline/process dan /pipeline/embed.
    """
    brand_name: str
    industry: str = "skincare"

    @field_validator("brand_name")
    @classmethod
    def brand_name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("brand_name tidak boleh kosong")
        return v.strip()


class QueryRequest(BaseModel):
    """
    Request untuk GEO audit query.
    Digunakan oleh endpoint POST /pipeline/query.
    """
    brand_name: str
    query: str = "Analisis visibilitas brand ini di AI search engines"
    industry: str = "skincare"

    @field_validator("brand_name")
    @classmethod
    def brand_name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("brand_name tidak boleh kosong")
        return v.strip()

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query tidak boleh kosong")
        if len(v) > 500:
            raise ValueError("query maksimal 500 karakter")
        return v.strip()


# ─── GEO Audit Response Models ────────────────────────────────────────

class GeoScore(BaseModel):
    """
    Hasil kalkulasi GEO Score untuk satu brand.
    Score 0–100, grade A–F.
    """
    score: int                    # 0–100
    grade: str                    # A | B | C | D | F
    presence_score: int           # seberapa sering brand muncul di AI results (0-40)
    accuracy_score: int           # keakuratan informasi yang diambil AI (0-40)
    sentiment_score: int          # sentimen konteks brand di AI results (0-20)
    issues: list[str]             # daftar masalah yang ditemukan
    strengths: list[str]          # daftar kekuatan brand
    recommendations: list[str]    # rekomendasi perbaikan (paywall: top 3 gratis)


class GeoAuditResponse(BaseModel):
    """
    Response lengkap dari GEO audit pipeline.
    Wrapper untuk hasil stage4_rag + geo_scorer.
    """
    brand_name: str
    query: str
    geo_score: GeoScore
    chunks_retrieved: int         # jumlah chunks yang ditemukan
    llm_response: str             # raw LLM audit text
    sources: list[str]            # URL sumber yang direferensikan
