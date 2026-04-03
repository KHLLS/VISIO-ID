"""
VISIO.ID — Health Check Endpoint
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health_check():
    """Cek status API."""
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "data": {"service": "visio-id-api"}, "error": None},
    )
