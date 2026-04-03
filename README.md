# VISIO.ID

AI Search Visibility & GEO Monitoring Platform untuk brand skincare lokal Indonesia.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11 + FastAPI |
| Frontend | Next.js 14 + Tailwind (belum dibangun) |
| Database | Supabase + pgvector (768 dim) |
| Embedding | multilingual-e5-base (lokal) |
| LLM dev | Gemini Flash API |
| LLM prod | GPT-4o mini |
| Cache | Redis |
| Deploy | Vercel + Railway |

## Setup

### 1. Clone & install dependencies

```bash
pip install -r requirements.txt
```

### 2. Konfigurasi environment

```bash
cp .env.example .env
# Edit .env dan isi kredensial:
# - SUPABASE_URL
# - SUPABASE_SERVICE_KEY
# - GEMINI_API_KEY
```

### 3. Jalankan pipeline

```bash
# Full pipeline (scrape + process + embed)
python run_pipeline.py --brand-url https://www.somethinc.com --brand-name "Somethinc" --industry skincare

# Dry-run (tanpa upload ke Supabase, limit 3 pages)
python run_pipeline.py --brand-url https://www.somethinc.com --brand-name "Somethinc" --dry-run --max-pages 3

# Run stage tertentu
python run_pipeline.py --stage 1 --brand-url URL --brand-name "Nama"
python run_pipeline.py --stage 2 --brand-name "Nama"
python run_pipeline.py --stage 3 --brand-name "Nama"

# GEO audit (stage 4 — requires stages 1-3 completed + Gemini API key)
python run_pipeline.py --stage 4 --brand-name "Somethinc" --query "Analisis visibilitas brand ini"
```

### 4. Jalankan API server

```bash
uvicorn backend.main:app --reload --port 8000
```

API docs tersedia di: http://localhost:8000/docs

## Project Structure

```
visio-id/
├── backend/
│   ├── config.py             # Semua env vars & konstanta
│   ├── main.py               # FastAPI app entry point
│   ├── api/
│   │   ├── health.py         # GET /health
│   │   └── pipeline.py       # POST /pipeline/run, /pipeline/process, GET /pipeline/status/{brand}
│   ├── pipeline/
│   │   ├── stage1_ingestion.py   # Web scraper (BFS, robots.txt)
│   │   ├── stage2_processing.py  # Text cleaning & chunking
│   │   ├── stage3_embedding.py   # multilingual-e5-base + Supabase upsert
│   │   └── stage4_rag.py         # Hybrid retrieval + rerank + GEO audit
│   └── tests/
│       ├── test_stage1_ingestion.py
│       └── test_stage2_processing.py
├── data/
│   ├── raw/          # Output stage 1
│   ├── processed/    # Output stage 2
│   └── embeddings/   # Output stage 3
├── logs/
├── scripts/migrations/
├── run_pipeline.py   # CLI entry point
├── .env.example
└── requirements.txt
```

## Pipeline Parameters (dari VISIO_RULES)

| Parameter | Value |
|-----------|-------|
| Chunk size | 2048 karakter (~512 token) |
| Chunk overlap | 256 karakter (~64 token) |
| Request delay | min 2 detik |
| Max pages/domain | 50 |
| Embedding model | intfloat/multilingual-e5-base |
| Embedding prefix doc | `"passage: "` |
| Embedding prefix query | `"query: "` |
| Batch size embedding | 32 |
| Batch size upload | 50 |
| Vector search top-k | 20 |
| BM25 top-k | 10 |
| Rerank final top-k | 5 |
| LLM temperature | 0.3 |
| LLM max tokens | 1500 |

## Running Tests

```bash
pytest backend/tests/ -v
```

## API Endpoints

| Method | Path | Rate Limit | Deskripsi |
|--------|------|-----------|-----------|
| GET | `/health` | - | Health check |
| POST | `/pipeline/run` | 5/min | Trigger full pipeline |
| POST | `/pipeline/process` | 5/min | Trigger stage 2 saja |
| POST | `/pipeline/query` | 5/min | GEO audit via RAG (Stage 4) |
| GET | `/pipeline/status/{brand_name}` | 10/min | Cek status pipeline |
