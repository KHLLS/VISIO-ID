"""
VISIO.ID — FastAPI Application
Entry point untuk backend API.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.config import get_logger
from backend.api.health import router as health_router
from backend.api.pipeline import router as pipeline_router

logger = get_logger(__name__)

# ─── Rate Limiter ────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ─── App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="VISIO.ID API",
    description="AI Search Visibility & GEO Monitoring Platform for local Indonesian skincare brands",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── CORS ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────────────────
app.include_router(health_router, tags=["Health"])
app.include_router(pipeline_router, prefix="/pipeline", tags=["Pipeline"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "data": None, "error": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
