"""
VISIO.ID — Stage 3: Embedding & Upload
Generate embeddings menggunakan multilingual-e5-base dan upload ke Supabase pgvector.

Fitur:
- Batch embedding (32 per batch)
- Prefix "passage: " untuk dokumen
- Upsert ke Supabase (bukan insert)
- Batch upload (50 per batch)
- Local embedding cache di data/embeddings/
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import (
    PROCESSED_DIR,
    EMBEDDINGS_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_PREFIX_DOC,
    BATCH_SIZE_EMBEDDING,
    BATCH_SIZE_UPLOAD,
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    SUPABASE_TABLE,
    get_logger,
)

logger = get_logger(__name__)

# Lazy-load model (besar, hanya load kalau diperlukan)
_model = None


def _get_model():
    """Load embedding model (lazy, singleton)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        logger.info("(Download pertama kali ~1GB, harap tunggu...)")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Model loaded ✓")
    return _model


# ─── Embedding ───────────────────────────────────────────────────────


def generate_embeddings(
    chunks: list[dict],
    batch_size: int = BATCH_SIZE_EMBEDDING,
) -> list[dict]:
    """
    Generate embeddings untuk list of chunks.

    Args:
        chunks: List of chunk dicts (harus punya key 'content')
        batch_size: Ukuran batch (default: 32)

    Returns:
        List of chunk dicts dengan tambahan key 'embedding' (list of floats)
    """
    if not chunks:
        return []

    model = _get_model()

    # Siapkan texts dengan prefix
    texts = [
        f"{EMBEDDING_PREFIX_DOC}{chunk['content']}" for chunk in chunks
    ]

    logger.info(f"Generating embeddings: {len(texts)} chunks, batch_size={batch_size}")

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.extend(embeddings)

    # Attach embeddings ke chunks
    embedded_chunks = []
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk_with_emb = dict(chunk)
        chunk_with_emb["embedding"] = embedding.tolist()
        embedded_chunks.append(chunk_with_emb)

    logger.info(f"Embeddings generated ✓ (dim={EMBEDDING_DIM})")
    return embedded_chunks


# ─── Supabase Upload ─────────────────────────────────────────────────


def _get_supabase_client():
    """Buat Supabase client. Raise error kalau credentials kosong."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError(
            "SUPABASE_URL dan SUPABASE_SERVICE_KEY harus diset di .env!\n"
            "Lihat .env.example untuk template."
        )
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def upload_to_supabase(
    embedded_chunks: list[dict],
    batch_size: int = BATCH_SIZE_UPLOAD,
) -> int:
    """
    Upsert embedded chunks ke Supabase table visio_documents.

    Args:
        embedded_chunks: List of chunk dicts dengan 'embedding' key
        batch_size: Ukuran batch upload (default: 50)

    Returns:
        Jumlah rows yang berhasil di-upsert
    """
    client = _get_supabase_client()

    logger.info(
        f"Uploading to Supabase: {len(embedded_chunks)} chunks, "
        f"batch_size={batch_size}"
    )

    uploaded = 0
    errors = 0

    for i in tqdm(range(0, len(embedded_chunks), batch_size), desc="Upload"):
        batch = embedded_chunks[i : i + batch_size]

        rows = []
        for chunk in batch:
            rows.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "source": chunk["source"],
                "category": chunk.get("category", "general"),
                "content": chunk["content"],
                "language": chunk.get("language", "id"),
                "metadata": {
                    "brand_name": chunk.get("brand_name", ""),
                    "industry": chunk.get("industry", "skincare"),
                    "title": chunk.get("title", ""),
                    "url": chunk.get("metadata", {}).get("url", ""),
                    "chunk_index": chunk.get("metadata", {}).get("chunk_index", 0),
                    "total_chunks": chunk.get("metadata", {}).get("total_chunks", 1),
                },
                "embedding": chunk["embedding"],
            })

        try:
            result = (
                client.table(SUPABASE_TABLE)
                .upsert(rows, on_conflict="chunk_id")
                .execute()
            )
            uploaded += len(batch)
        except Exception as e:
            logger.warning(f"✗ Batch upload error (rows {i}-{i+len(batch)}): {e}")
            errors += 1
            continue

    logger.info(f"Upload selesai: {uploaded} rows upserted, {errors} batch errors")
    return uploaded


# ─── Orchestrator ────────────────────────────────────────────────────


def embed_brand_data(
    brand_name: str,
    skip_upload: bool = False,
) -> list[dict]:
    """
    Orchestrate embedding pipeline: load chunks → embed → upload.

    Fungsi ini idempotent — menjalankan ulang akan overwrite data lama.

    Args:
        brand_name: Nama brand (harus cocok dengan folder di data/processed/)
        skip_upload: Jika True, skip upload ke Supabase (untuk dry-run/testing)

    Returns:
        List of embedded chunk dicts
    """
    brand_key = brand_name.lower().replace(" ", "_")
    chunks_file = PROCESSED_DIR / brand_key / "chunks.json"

    if not chunks_file.exists():
        logger.error(f"File tidak ditemukan: {chunks_file}")
        logger.error("Jalankan Stage 2 (processing) dulu!")
        return []

    logger.info(f"═══ Stage 3: Embedding {brand_name} ═══")

    # Load chunks
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks dari {chunks_file}")

    # Generate embeddings
    embedded_chunks = generate_embeddings(chunks)

    # Save embeddings locally (cache)
    cache_dir = EMBEDDINGS_DIR / brand_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "embedded_chunks.json"

    # Save tanpa embedding vectors (terlalu besar untuk JSON readable)
    # Simpan embedding vectors sebagai numpy file
    embeddings_array = np.array([c["embedding"] for c in embedded_chunks])
    np.save(cache_dir / "vectors.npy", embeddings_array)

    # Save metadata (tanpa embedding)
    meta_chunks = []
    for c in embedded_chunks:
        meta = dict(c)
        meta.pop("embedding", None)
        meta_chunks.append(meta)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(meta_chunks, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Cache saved: {cache_file} + vectors.npy "
        f"({embeddings_array.shape})"
    )

    # Upload ke Supabase
    if skip_upload:
        logger.info("Upload di-skip (dry-run mode)")
    else:
        upload_to_supabase(embedded_chunks)

    logger.info(f"═══ Stage 3 selesai: {len(embedded_chunks)} chunks embedded ═══")
    return embedded_chunks


# ─── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VISIO.ID Stage 3: Embedding")
    parser.add_argument("--brand-name", required=True, help="Nama brand")
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Skip upload ke Supabase (dry-run)"
    )
    args = parser.parse_args()

    embed_brand_data(args.brand_name, skip_upload=args.skip_upload)
