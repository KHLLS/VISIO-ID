# VISIO.ID — Agent Rules
# Antigravity IDE · Claude Opus 4.6 + Sonnet 4.6
# Last updated: Maret 2026

---

## 🤖 MODEL ROUTING — BACA INI DULU

```
RULE UTAMA: Opus 4.6 untuk berpikir. Sonnet 4.6 untuk mengeksekusi.
Kalau ragu, pakai Sonnet dulu. Switch ke Opus hanya kalau Sonnet stuck.
```

### Gunakan CLAUDE OPUS 4.6 untuk:
- Merancang arsitektur RAG pipeline dari nol
- Menulis logic hybrid retrieval + reranker
- Prompt engineering untuk GEO audit system prompt
- Debugging logic error yang kompleks dan tidak jelas penyebabnya
- Keputusan arsitektur yang berdampak ke banyak file sekaligus
- Analisis tradeoff (embedding model, strategi chunking, dll)
- Review kode pipeline sebelum deploy ke production

### Gunakan CLAUDE SONNET 4.6 untuk:
- Semua UI components (Next.js + Tailwind)
- Boilerplate FastAPI endpoints
- Scraper baru berdasarkan pola yang sudah ada
- Nulis test cases
- Refactor atau rename variabel
- Update requirements.txt, .env.example, README
- One-off scripts di folder scripts/
- Bug fix yang penyebabnya sudah jelas
- Semua task yang bisa dijelaskan dalam 1-2 kalimat

### Estimasi cost:
```
Sonnet 4.6  : ~10x lebih murah dari Opus
Target      : 80% task Sonnet, 20% Opus
```

---

## 🧠 PROJECT IDENTITY

```
Product   : VISIO.ID
Type      : AI Search Visibility & GEO Monitoring Platform
Target    : Brand skincare lokal Indonesia (UMKM)
Mission   : Bantu brand lokal terlihat di ChatGPT, Perplexity, Gemini
Stage     : Pre-build — discovery phase (10 user interviews in progress)
Team      : 3 orang (AIDA + CS + DM), Semester 2, CCIT Venture Studio FTUI
```

---

## 🗂️ PROJECT STRUCTURE

```
visio-id/
├── backend/
│   ├── config.py             # SEMUA env vars di sini
│   ├── main.py
│   ├── api/
│   ├── pipeline/
│   │   ├── stage1_ingestion.py
│   │   ├── stage2_processing.py
│   │   ├── stage3_embedding.py
│   │   └── stage4_rag.py
│   ├── services/
│   ├── models/
│   └── tests/
├── frontend/
│   ├── app/
│   ├── components/
│   └── lib/
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
├── logs/
├── scripts/
│   └── migrations/
├── .env
├── .env.example
└── requirements.txt
```

---

## ⚙️ TECH STACK

| Layer | Technology | Notes |
|-------|-----------|-------|
| Backend | Python 3.11 + FastAPI | |
| Frontend | Next.js 14 + Tailwind | App Router |
| Database | Supabase + pgvector | Vector dim: 768 |
| Embedding | multilingual-e5-base | Lokal, gratis, wajib |
| LLM dev | Gemini Flash API | |
| LLM prod | GPT-4o mini | |
| RAG | LangChain + LangGraph | |
| Cache | Redis | TTL: 30 hari |
| Auth | NextAuth.js | Email verify wajib |
| Rate Limit | SlowAPI | Wajib semua endpoint |
| Deploy | Vercel + Railway | |

---

## 🗄️ DATABASE SCHEMA

```sql
create table visio_documents (
    id          bigserial primary key,
    chunk_id    text unique not null,
    doc_id      text,
    source      text,
    category    text,
    content     text,
    language    text,
    metadata    jsonb,
    embedding   vector(768),
    created_at  timestamptz default now()
);

create index on visio_documents
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);
```

Perubahan schema → wajib buat file di scripts/migrations/YYYY-MM-DD_desc.sql

---

## 🔧 RAG PIPELINE RULES

| Parameter | Value |
|-----------|-------|
| Chunk size | 512 token ≈ 2048 karakter |
| Chunk overlap | 64 token |
| Request delay | min 2 detik |
| Max pages/domain | 50 |
| Embedding prefix query | "query: " + teks |
| Embedding prefix doc | "passage: " + teks |
| Batch size embedding | 32 |
| Batch size upload | 50 |
| Vector search top-k | 20 |
| BM25 top-k | 10 |
| Rerank final top-k | 5 |
| LLM temperature | 0.3 |
| Max tokens | 1500 |

Upload Supabase: SELALU upsert, bukan insert.
Satu URL gagal: log warning + skip, jangan crash pipeline.

---

## 🔒 SECURITY WAJIB

- Rate limiting di setiap endpoint: @limiter.limit("5/minute")
- Tidak ada anonymous audit — email verify wajib
- Redis cache — cek sebelum panggil LLM API
- Spending cap di semua LLM provider
- Anomaly monitoring kalau cost melebihi threshold

---

## 💻 CODE STANDARDS

Python:
- Docstring + logging wajib di setiap file
- Error handling: try/catch + log, jangan crash silent
- Semua secret dari config.py, tidak ada hardcode
- Fungsi harus idempotent (aman di-run ulang)

TypeScript:
- App Router, Server Components by default
- API calls lewat /api/ routes — jangan fetch Supabase dari frontend

---

## 🤖 AGENT BEHAVIOR

Sebelum edit file existing:
1. Baca file dulu
2. Cek apakah fungsi dipanggil dari tempat lain
3. Jangan rename fungsi publik tanpa cek semua references

Sebelum tambah dependency:
1. Cek requirements.txt dulu
2. Tambah dengan versi spesifik

Saat generate scraper:
1. Tambah time.sleep(REQUEST_DELAY) antar request
2. Handle exception per URL — satu gagal tidak crash semua
3. Test 3-5 URL dulu sebelum full batch

Saat generate endpoint:
1. Tambah @limiter.limit()
2. Input validation dengan Pydantic
3. Response: {"status": "ok", "data": ..., "error": null}

DILARANG:
- Hardcode API keys atau credentials
- Delete file tanpa konfirmasi
- Bypass auth untuk testing
- Push ke branch main langsung

---

## 📊 BUSINESS CONTEXT

```
Segment MVP    : Brand skincare lokal Indonesia
Pricing        : Free / Rp199k / Rp499k per bulan
Kompetitor     : Searchable ($50/mo), Profound ($99-399/mo) — English-only
Moat           : Data lokal + distribusi UMKM + Bahasa Indonesia native
Free tier rule : Tampil GEO Score + top 3 masalah dulu — rekomendasi = paywall
Red line       : Jangan klaim "optimize AI training" — framing: "optimize retrieval signals"
```

---

## 🎯 CURRENT SPRINT

```
Status          : Discovery phase — user interviews berlangsung
Priority        : Skincare vertikal dulu
Boleh build     : Stage 1–3 pipeline sebagai prep
Jangan build    : Frontend, auth, payment — tunggu interviews selesai
```

---

## ⚡ QUICK REFERENCE

```bash
# Full pipeline
python run_pipeline.py --brand-url URL --brand-name "Nama" --industry skincare

# Audit only
python run_pipeline.py --audit --brand-name "Nama" --industry skincare

# Dev servers
uvicorn main:app --reload --port 8000
cd frontend && npm run dev
```

---

*VISIO.ID · Rules v1.1 · 2026 · CCIT Venture Studio FTUI*
