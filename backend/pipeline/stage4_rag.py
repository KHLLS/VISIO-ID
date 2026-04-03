"""
VISIO.ID — Stage 4: RAG (Retrieval-Augmented Generation)
Hybrid retrieval + reranking + LLM-based GEO audit.

Fitur:
- Vector search via Supabase pgvector (top-k: 20)
- BM25 keyword search lokal (top-k: 10)
- Reciprocal Rank Fusion (RRF) untuk merge results
- Cross-encoder reranking (top-k: 5)
- GEO audit via Gemini Flash API
- Redis-ready cache check (placeholder)
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import (
    PROCESSED_DIR,
    EMBEDDINGS_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_PREFIX_QUERY,
    EMBEDDING_PREFIX_DOC,
    VECTOR_SEARCH_TOP_K,
    BM25_TOP_K,
    RERANK_FINAL_TOP_K,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    SUPABASE_TABLE,
    GEMINI_API_KEY,
    get_logger,
)

logger = get_logger(__name__)


# ─── Lazy-loaded Models ──────────────────────────────────────────────

_embedding_model = None
_cross_encoder = None


def _get_embedding_model():
    """Load embedding model (lazy, singleton)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded ✓")
    return _embedding_model


def _get_cross_encoder():
    """Load cross-encoder reranker (lazy, singleton)."""
    global _cross_encoder
    if _cross_encoder is None:
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info(f"Loading cross-encoder: {model_name}")
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(model_name)
        logger.info("Cross-encoder loaded ✓")
    return _cross_encoder


# ─── Vector Search (Supabase pgvector) ───────────────────────────────


def vector_search(
    query: str,
    top_k: int = VECTOR_SEARCH_TOP_K,
    brand_name: Optional[str] = None,
) -> list[dict]:
    """
    Lakukan vector similarity search di Supabase pgvector.

    Args:
        query: Teks query dari user
        top_k: Jumlah hasil teratas (default: 20)
        brand_name: Filter berdasarkan brand (optional)

    Returns:
        List of dicts: [{chunk_id, content, source, category, metadata, score}]
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        logger.warning("Supabase credentials not set — skipping vector search")
        return []

    try:
        from supabase import create_client
        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

        # Generate query embedding dengan prefix "query: "
        model = _get_embedding_model()
        query_embedding = model.encode(
            f"{EMBEDDING_PREFIX_QUERY}{query}",
            normalize_embeddings=True,
        ).tolist()

        # RPC call ke Supabase (requires a pgvector similarity function)
        # Fallback: manual query jika RPC tidak tersedia
        rpc_result = client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.3,
                "match_count": top_k,
            },
        ).execute()

        results = []
        for row in rpc_result.data or []:
            result = {
                "chunk_id": row.get("chunk_id", ""),
                "content": row.get("content", ""),
                "source": row.get("source", ""),
                "category": row.get("category", ""),
                "metadata": row.get("metadata", {}),
                "score": row.get("similarity", 0.0),
                "retrieval_method": "vector",
            }
            # Filter by brand if specified
            if brand_name:
                meta = result["metadata"]
                if isinstance(meta, dict) and meta.get("brand_name", "").lower() != brand_name.lower():
                    continue
            results.append(result)

        logger.info(f"Vector search: {len(results)} results for query '{query[:50]}...'")
        return results

    except Exception as e:
        logger.warning(f"Vector search error: {e}")
        return []


# ─── BM25 Search (Local) ────────────────────────────────────────────


def bm25_search(
    query: str,
    brand_name: str,
    top_k: int = BM25_TOP_K,
) -> list[dict]:
    """
    Lakukan BM25 keyword search dari local chunks file.

    Args:
        query: Teks query dari user
        brand_name: Nama brand (untuk locate chunks file)
        top_k: Jumlah hasil teratas (default: 10)

    Returns:
        List of dicts: [{chunk_id, content, source, category, metadata, score}]
    """
    brand_key = brand_name.lower().replace(" ", "_")
    chunks_file = PROCESSED_DIR / brand_key / "chunks.json"

    if not chunks_file.exists():
        logger.warning(f"Chunks file not found: {chunks_file}")
        return []

    try:
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            return []

        # Tokenize untuk BM25
        tokenized_corpus = [
            chunk.get("content", "").lower().split() for chunk in chunks
        ]
        bm25 = BM25Okapi(tokenized_corpus)

        # Query
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Rank dan ambil top-k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = chunks[idx]
            results.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "content": chunk.get("content", ""),
                "source": chunk.get("source", ""),
                "category": chunk.get("category", ""),
                "metadata": chunk.get("metadata", {}),
                "score": float(scores[idx]),
                "retrieval_method": "bm25",
            })

        logger.info(f"BM25 search: {len(results)} results for query '{query[:50]}...'")
        return results

    except Exception as e:
        logger.warning(f"BM25 search error: {e}")
        return []


# ─── Reciprocal Rank Fusion ─────────────────────────────────────────


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Merge multiple ranked result lists menggunakan Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) across all lists.

    Args:
        result_lists: List of ranked result lists
        k: RRF constant (default: 60, standard value)

    Returns:
        Merged list sorted by RRF score (descending)
    """
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            # Simpan data chunk (ambil dari list pertama yang punya)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    merged = []
    for chunk_id in sorted_ids:
        entry = dict(chunk_data[chunk_id])
        entry["rrf_score"] = rrf_scores[chunk_id]
        entry["retrieval_method"] = "hybrid"
        merged.append(entry)

    logger.info(f"RRF merge: {len(merged)} unique results from {len(result_lists)} lists")
    return merged


# ─── Reranker ────────────────────────────────────────────────────────


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = RERANK_FINAL_TOP_K,
) -> list[dict]:
    """
    Rerank candidates menggunakan cross-encoder.

    Args:
        query: Query text
        candidates: List of candidate chunks (dari hybrid retrieval)
        top_k: Jumlah hasil akhir (default: 5)

    Returns:
        Top-k reranked results
    """
    if not candidates:
        return []

    if len(candidates) <= top_k:
        return candidates

    try:
        cross_encoder = _get_cross_encoder()

        # Prepare pairs for cross-encoder
        pairs = [(query, c["content"]) for c in candidates]

        # Score
        scores = cross_encoder.predict(pairs)

        # Attach scores dan sort
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            f"Reranked: {len(candidates)} → {top_k} "
            f"(top score: {reranked[0]['rerank_score']:.4f})"
        )
        return reranked[:top_k]

    except Exception as e:
        logger.warning(f"Reranker error, returning top-{top_k} by RRF score: {e}")
        return candidates[:top_k]


# ─── GEO Audit via LLM ──────────────────────────────────────────────


GEO_AUDIT_SYSTEM_PROMPT = """Kamu adalah VISIO.ID, AI assistant yang menganalisis visibilitas brand skincare lokal Indonesia di AI search engines (ChatGPT, Perplexity, Gemini).

TUGASMU:
Berdasarkan konten website brand yang diberikan, buat audit GEO (Generative Engine Optimization) yang mencakup:

1. **GEO Score** (0-100): Seberapa mudah konten ini ditemukan dan dikutip oleh AI search engines
2. **Top 3 Masalah**: Masalah utama yang mengurangi visibilitas AI
3. **Rekomendasi**: Langkah konkret untuk meningkatkan retrieval signals

ATURAN:
- Fokus pada "optimizing retrieval signals" — bukan "optimizing AI training"
- Nilai berdasarkan: kejelasan konten, struktur data, coverage keyword, kualitas deskripsi produk
- Berikan skor realistis — brand lokal rata-rata skornya 20-40
- Gunakan Bahasa Indonesia
- Format output sebagai JSON dengan keys: geo_score, issues, recommendations, summary

KONTEKS INDUSTRI:
- Target: Brand skincare lokal Indonesia (UMKM)
- AI engines yang monitored: ChatGPT, Perplexity, Gemini
- Kompetitor global: brand internasional yang sudah optimize konten mereka"""


def generate_geo_audit(
    query: str,
    context_chunks: list[dict],
    brand_name: str,
    industry: str = "skincare",
) -> dict:
    """
    Generate GEO audit menggunakan LLM (Gemini Flash API).

    Args:
        query: Query audit (misal: "analisis visibilitas brand X")
        context_chunks: Top-k reranked chunks sebagai context
        brand_name: Nama brand
        industry: Industri (default: skincare)

    Returns:
        dict: {geo_score, issues, recommendations, summary, raw_response}
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY tidak diset di .env!")
        return {
            "geo_score": 0,
            "issues": ["API key tidak tersedia"],
            "recommendations": ["Set GEMINI_API_KEY di file .env"],
            "summary": "Audit gagal — API key tidak tersedia.",
            "raw_response": "",
        }

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Siapkan context dari chunks
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            title = chunk.get("title", chunk.get("metadata", {}).get("title", ""))
            url = chunk.get("metadata", {}).get("url", "")
            context_text += f"\n--- Konten {i} ---\n"
            if title:
                context_text += f"Judul: {title}\n"
            if url:
                context_text += f"URL: {url}\n"
            context_text += f"Kategori: {chunk.get('category', 'unknown')}\n"
            context_text += f"Konten:\n{chunk['content'][:1500]}\n"

        user_prompt = f"""Brand: {brand_name}
Industri: {industry}
Query: {query}

Berikut konten dari website brand yang berhasil di-retrieve:
{context_text}

Berikan audit GEO dalam format JSON:
{{
    "geo_score": <number 0-100>,
    "issues": ["masalah 1", "masalah 2", "masalah 3"],
    "recommendations": ["rekomendasi 1", "rekomendasi 2", "rekomendasi 3"],
    "summary": "ringkasan audit dalam 2-3 kalimat"
}}"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(role="user", parts=[types.Part(text=GEO_AUDIT_SYSTEM_PROMPT)]),
                types.Content(role="model", parts=[types.Part(text="Saya siap membantu audit GEO. Silakan berikan konten brand yang akan dianalisis.")]),
                types.Content(role="user", parts=[types.Part(text=user_prompt)]),
            ],
            config=types.GenerateContentConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
            ),
        )

        raw_text = response.text.strip()
        logger.info(f"LLM response received ({len(raw_text)} chars)")

        # Parse JSON dari response
        audit_result = _parse_audit_response(raw_text)
        audit_result["raw_response"] = raw_text
        return audit_result

    except Exception as e:
        logger.error(f"GEO audit error: {e}", exc_info=True)
        return {
            "geo_score": 0,
            "issues": [f"Error: {str(e)}"],
            "recommendations": ["Coba jalankan ulang audit"],
            "summary": f"Audit gagal karena error: {str(e)}",
            "raw_response": "",
        }


def _parse_audit_response(raw_text: str) -> dict:
    """Parse LLM response menjadi structured audit result."""
    # Coba parse JSON langsung
    try:
        # Handle markdown code block wrapping
        text = raw_text
        if "```json" in text:
            text = text.split("```json", 1)[1]
            text = text.split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1]
            text = text.split("```", 1)[0]

        result = json.loads(text.strip())

        return {
            "geo_score": int(result.get("geo_score", 0)),
            "issues": result.get("issues", []),
            "recommendations": result.get("recommendations", []),
            "summary": result.get("summary", ""),
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse audit JSON: {e}")
        return {
            "geo_score": 0,
            "issues": ["Response LLM tidak dapat di-parse"],
            "recommendations": ["Coba jalankan ulang audit"],
            "summary": raw_text[:500],
        }


# ─── Orchestrator ────────────────────────────────────────────────────


def run_geo_audit(
    query: str,
    brand_name: str,
    industry: str = "skincare",
) -> dict:
    """
    Orchestrate full RAG pipeline: hybrid retrieval → rerank → GEO audit.

    Args:
        query: Query audit dari user
        brand_name: Nama brand yang diaudit
        industry: Industri (default: skincare)

    Returns:
        dict: Full audit result termasuk retrieval metadata
    """
    logger.info(f"═══ Stage 4: GEO Audit for {brand_name} ═══")
    logger.info(f"Query: {query}")
    start_time = time.time()

    # Step 1: Hybrid Retrieval
    logger.info("Step 1: Hybrid retrieval...")
    vector_results = vector_search(query, brand_name=brand_name)
    bm25_results = bm25_search(query, brand_name)

    # Step 2: RRF Merge
    logger.info("Step 2: RRF merge...")
    all_results = []
    if vector_results:
        all_results.append(vector_results)
    if bm25_results:
        all_results.append(bm25_results)

    if not all_results:
        logger.warning("No retrieval results — cannot run audit")
        return {
            "brand_name": brand_name,
            "query": query,
            "audit": {
                "geo_score": 0,
                "issues": ["Tidak ada data brand ditemukan"],
                "recommendations": ["Jalankan pipeline stage 1-3 dulu untuk brand ini"],
                "summary": "Audit gagal — tidak ada data.",
            },
            "retrieval": {"total_candidates": 0, "final_chunks": 0},
            "elapsed_seconds": time.time() - start_time,
        }

    merged = reciprocal_rank_fusion(all_results)

    # Step 3: Rerank
    logger.info("Step 3: Reranking...")
    top_chunks = rerank(query, merged)

    # Step 4: GEO Audit via LLM
    logger.info("Step 4: GEO audit via LLM...")
    audit = generate_geo_audit(query, top_chunks, brand_name, industry)

    elapsed = time.time() - start_time
    logger.info(f"═══ Stage 4 selesai dalam {elapsed:.1f}s ═══")

    return {
        "brand_name": brand_name,
        "query": query,
        "industry": industry,
        "audit": {
            "geo_score": audit.get("geo_score", 0),
            "issues": audit.get("issues", []),
            "recommendations": audit.get("recommendations", []),
            "summary": audit.get("summary", ""),
        },
        "retrieval": {
            "vector_results": len(vector_results),
            "bm25_results": len(bm25_results),
            "merged_candidates": len(merged),
            "final_chunks": len(top_chunks),
        },
        "elapsed_seconds": round(elapsed, 2),
    }


# ─── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VISIO.ID Stage 4: RAG / GEO Audit")
    parser.add_argument("--brand-name", required=True, help="Nama brand")
    parser.add_argument(
        "--query",
        default="Analisis visibilitas brand ini di AI search engines",
        help="Query untuk audit GEO",
    )
    parser.add_argument("--industry", default="skincare", help="Industri")
    args = parser.parse_args()

    result = run_geo_audit(
        query=args.query,
        brand_name=args.brand_name,
        industry=args.industry,
    )

    print("\n" + "=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
