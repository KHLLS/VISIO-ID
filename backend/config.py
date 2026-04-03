"""
VISIO.ID — Centralized Configuration
Semua environment variables dan konstanta pipeline ada di sini.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
LOGS_DIR = BASE_DIR / "logs"

# Buat directories kalau belum ada
for d in [RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Supabase ─────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_TABLE = "visio_documents"

# ─── LLM / Embedding ─────────────────────────────────────────────────
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768
EMBEDDING_PREFIX_QUERY = "query: "
EMBEDDING_PREFIX_DOC = "passage: "

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ─── RAG Pipeline Parameters ─────────────────────────────────────────
CHUNK_SIZE_CHARS = 2048          # ~512 tokens
CHUNK_OVERLAP_CHARS = 256        # ~64 tokens
REQUEST_DELAY = 2.0              # detik antar request
MAX_PAGES_PER_DOMAIN = 50
BATCH_SIZE_EMBEDDING = 32
BATCH_SIZE_UPLOAD = 50
VECTOR_SEARCH_TOP_K = 20
BM25_TOP_K = 10
RERANK_FINAL_TOP_K = 5
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1500

# ─── Scraper Settings ────────────────────────────────────────────────
USER_AGENT = (
    "Mozilla/5.0 (compatible; VisioBot/1.0; "
    "+https://visio.id/bot; skincare-research)"
)
SKIP_URL_PATTERNS = [
    "/cart", "/checkout", "/login", "/register", "/account",
    "/wishlist", "/compare", "/search", "/wp-admin",
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
    ".zip", ".css", ".js",
]

# ─── Rate Limiting ───────────────────────────────────────────────────
RATE_LIMIT_DEFAULT = "5/minute"

# ─── Logging ─────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8"),
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name)
