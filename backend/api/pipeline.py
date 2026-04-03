"""
VISIO.ID — Pipeline API Endpoints
Endpoint untuk trigger pipeline ingestion, processing, embedding, dan GEO audit.

Semua endpoint memiliki rate limiting dan Pydantic input validation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.config import get_logger
from backend.services.cache import CacheService

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

# Singleton cache service
_cache = CacheService()


# ─── Request Models ──────────────────────────────────────────────────

class PipelineRunRequest(BaseModel):
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
            if s not in [1, 2, 3, 4]:
                raise ValueError("stages hanya boleh berisi 1, 2, 3, atau 4")
        return sorted(set(v))


class PipelineStageRequest(BaseModel):
    brand_name: str
    industry: str = "skincare"

    @field_validator("brand_name")
    @classmethod
    def brand_name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("brand_name tidak boleh kosong")
        return v.strip()


class QueryRequest(BaseModel):
    brand_name: str
    query: str = "Analisis visibilitas brand ini di AI search engines"
    industry: str = "skincare"
    use_cache: bool = True

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


# ─── Background Tasks ────────────────────────────────────────────────

def _run_full_pipeline(
    brand_url: str,
    brand_name: str,
    industry: str,
    max_pages: int,
    stages: list[int],
):
    """Background task untuk full pipeline run (Stage 1–4)."""
    try:
        if 1 in stages:
            from backend.pipeline.stage1_ingestion import crawl_brand_site
            crawl_brand_site(
                base_url=brand_url,
                brand_name=brand_name,
                max_pages=max_pages,
                industry=industry,
            )

        if 2 in stages:
            from backend.pipeline.stage2_processing import process_brand_data
            process_brand_data(brand_name)

        if 3 in stages:
            from backend.pipeline.stage3_embedding import embed_brand_data
            embed_brand_data(brand_name)

        if 4 in stages:
            from backend.pipeline.stage4_rag import run_geo_audit
            _query = "Analisis visibilitas brand ini di AI search engines"
            result = run_geo_audit(
                query=_query,
                brand_name=brand_name,
                industry=industry,
            )
            # Cache hasil audit setelah pipeline selesai
            cache_key = CacheService.make_key(brand_name, _query)
            _cache.set(cache_key, result)
            logger.info(f"Audit cached untuk brand: {brand_name}")

        logger.info(f"Pipeline selesai untuk brand: {brand_name}")
    except Exception as e:
        logger.error(f"Pipeline error untuk {brand_name}: {e}", exc_info=True)


# ─── Endpoints ───────────────────────────────────────────────────────

@router.post("/run")
@limiter.limit("5/minute")
async def run_pipeline(
    request: Request,
    body: PipelineRunRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger pipeline untuk satu brand (Stage 1–4 bisa dipilih).
    Pipeline berjalan di background — response langsung dikembalikan.
    """
    logger.info(
        f"Pipeline triggered: brand={body.brand_name}, "
        f"stages={body.stages}, max_pages={body.max_pages}"
    )

    background_tasks.add_task(
        _run_full_pipeline,
        str(body.brand_url),
        body.brand_name,
        body.industry,
        body.max_pages,
        body.stages,
    )

    return JSONResponse(
        status_code=202,
        content={
            "status": "ok",
            "data": {
                "message": f"Pipeline dimulai untuk brand '{body.brand_name}'",
                "brand_name": body.brand_name,
                "stages": body.stages,
            },
            "error": None,
        },
    )


@router.post("/process")
@limiter.limit("5/minute")
async def process_brand(
    request: Request,
    body: PipelineStageRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger Stage 2 (processing) saja untuk brand yang sudah di-scrape.
    """
    logger.info(f"Stage 2 triggered: brand={body.brand_name}")

    background_tasks.add_task(
        _run_full_pipeline,
        "",
        body.brand_name,
        body.industry,
        0,
        [2],
    )

    return JSONResponse(
        status_code=202,
        content={
            "status": "ok",
            "data": {"message": f"Processing dimulai untuk '{body.brand_name}'"},
            "error": None,
        },
    )


@router.post("/query")
@limiter.limit("5/minute")
async def query_geo_audit(
    request: Request,
    body: QueryRequest,
):
    """
    Jalankan GEO audit menggunakan RAG pipeline (Stage 4).
    Cek cache Redis dulu — jika hit, return cached result.
    Sinkron — hasil langsung dikembalikan.
    """
    logger.info(f"GEO audit requested: brand={body.brand_name}, query={body.query[:50]}")

    # ── Cache check ──────────────────────────────────────────────────
    cache_key = CacheService.make_key(body.brand_name, body.query)
    if body.use_cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            logger.info(f"Cache HIT untuk brand: {body.brand_name}")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "ok",
                    "data": {**cached, "cache_hit": True},
                    "error": None,
                },
            )

    # ── Run audit ────────────────────────────────────────────────────
    try:
        from backend.pipeline.stage4_rag import run_geo_audit

        result = run_geo_audit(
            query=body.query,
            brand_name=body.brand_name,
            industry=body.industry,
        )

        # Simpan ke cache
        if body.use_cache:
            _cache.set(cache_key, result)
            logger.info(f"Cache SET untuk brand: {body.brand_name}")

        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "data": {**result, "cache_hit": False},
                "error": None,
            },
        )

    except Exception as e:
        logger.error(f"GEO audit error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "data": None,
                "error": str(e),
            },
        )


@router.get("/status/{brand_name}")
@limiter.limit("10/minute")
async def get_pipeline_status(request: Request, brand_name: str):
    """
    Cek status pipeline untuk suatu brand — apakah file sudah ada.
    """
    from backend.config import RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR

    brand_key = brand_name.lower().replace(" ", "_")

    raw_exists = (RAW_DIR / brand_key / "pages.json").exists()
    processed_exists = (PROCESSED_DIR / brand_key / "chunks.json").exists()
    embedded_exists = (EMBEDDINGS_DIR / brand_key / "vectors.npy").exists()

    # Cache info
    _default_query = "Analisis visibilitas brand ini di AI search engines"
    cache_key = CacheService.make_key(brand_name, _default_query)
    has_cached_audit = _cache.get(cache_key) is not None

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "data": {
                "brand_name": brand_name,
                "stages_completed": {
                    "stage1_ingestion": raw_exists,
                    "stage2_processing": processed_exists,
                    "stage3_embedding": embedded_exists,
                },
                "cache": {
                    "has_cached_audit": has_cached_audit,
                    "redis_available": _cache.is_available,
                },
            },
            "error": None,
        },
    )
