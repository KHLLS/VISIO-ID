"""
VISIO.ID — Stage 2: Data Processing
Membersihkan dan memotong teks hasil scraping menjadi chunks yang siap di-embed.

Fitur:
- Text cleaning (remove HTML artifacts, normalize whitespace)
- Sliding window chunking (2048 chars, 256 chars overlap)
- Deterministic chunk_id untuk idempotency
- Output: JSON files di data/processed/{brand_name}/
"""

import json
import re
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import (
    RAW_DIR,
    PROCESSED_DIR,
    CHUNK_SIZE_CHARS,
    CHUNK_OVERLAP_CHARS,
    get_logger,
)

logger = get_logger(__name__)


# ─── Data Models ─────────────────────────────────────────────────────


@dataclass
class ProcessedChunk:
    """Satu chunk teks yang sudah diproses, siap di-embed."""
    chunk_id: str          # unique, deterministic
    doc_id: str            # ID dokumen asal
    source: str            # domain asal
    brand_name: str
    industry: str
    category: str          # "product_page", "about", "blog", dll
    title: str
    content: str           # teks chunk
    language: str
    metadata: dict         # info tambahan (url, chunk_index, dll)


# ─── Text Cleaning ──────────────────────────────────────────────────


def clean_text(raw_text: str) -> str:
    """
    Bersihkan teks dari HTML artifacts dan normalize formatting.

    Args:
        raw_text: Teks mentah dari scraper

    Returns:
        Teks yang sudah dibersihkan
    """
    if not raw_text:
        return ""

    text = raw_text

    # Remove remaining HTML entities
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Remove excessive special characters
    text = re.sub(r"[│┃┆┊╎║▎▏─━┈┄╌═]+", "", text)

    # Normalize whitespace: multiple spaces → single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize newlines: 3+ newlines → 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that are just whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    text = "\n".join(lines)

    # Remove very short lines (likely menu items, buttons)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        # Keep lines with actual content (>10 chars atau bagian dari paragraf)
        if len(line) > 10 or (cleaned_lines and len(cleaned_lines[-1]) > 50):
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    return text.strip()


def _detect_page_category(url: str, title: str, text: str) -> str:
    """Deteksi kategori halaman berdasarkan URL dan konten."""
    url_lower = url.lower()
    title_lower = title.lower()
    combined = f"{url_lower} {title_lower}"

    if any(k in combined for k in ["/product", "/produk", "/shop", "/toko"]):
        return "product_page"
    if any(k in combined for k in ["/about", "/tentang", "/cerita"]):
        return "about"
    if any(k in combined for k in ["/blog", "/article", "/artikel", "/berita"]):
        return "blog"
    if any(k in combined for k in ["/faq", "/bantuan", "/help"]):
        return "faq"
    if any(k in combined for k in ["/ingredient", "/kandungan", "/bahan"]):
        return "ingredient"
    if any(k in combined for k in ["/review", "/testimoni", "/ulasan"]):
        return "review"
    if any(k in combined for k in ["/contact", "/kontak", "/hubungi"]):
        return "contact"

    return "general"


# ─── Chunking ────────────────────────────────────────────────────────


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """
    Potong teks menjadi chunks dengan sliding window.

    Memotong di batas paragraf/kalimat ketika memungkinkan.

    Args:
        text: Teks yang akan dipotong
        chunk_size: Ukuran maksimum chunk (default: 2048 chars)
        overlap: Ukuran overlap antar chunk (default: 256 chars)

    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk
            chunks.append(text[start:].strip())
            break

        # Coba potong di batas paragraf
        split_pos = text.rfind("\n\n", start + chunk_size // 2, end)

        # Kalau tidak ada, coba di batas kalimat
        if split_pos == -1:
            for sep in [". ", ".\n", "! ", "? "]:
                split_pos = text.rfind(sep, start + chunk_size // 2, end)
                if split_pos != -1:
                    split_pos += len(sep)
                    break

        # Kalau masih tidak ada, potong di batas kata
        if split_pos == -1:
            split_pos = text.rfind(" ", start + chunk_size // 2, end)

        # Fallback: hard cut
        if split_pos == -1 or split_pos <= start:
            split_pos = end

        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Geser start dengan overlap
        start = split_pos - overlap
        if start < 0:
            start = 0

    # Filter chunks yang terlalu pendek (< 50 chars)
    chunks = [c for c in chunks if len(c) >= 50]

    return chunks


def _generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate deterministic chunk_id untuk idempotency."""
    raw = f"{doc_id}:chunk:{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ─── Main Processing ────────────────────────────────────────────────


def process_brand_data(brand_name: str) -> list[ProcessedChunk]:
    """
    Proses semua raw data dari stage 1 menjadi chunks.

    Fungsi ini idempotent — menjalankan ulang akan overwrite data lama.

    Args:
        brand_name: Nama brand (harus cocok dengan folder di data/raw/)

    Returns:
        List of ProcessedChunk
    """
    brand_key = brand_name.lower().replace(" ", "_")
    raw_file = RAW_DIR / brand_key / "pages.json"

    if not raw_file.exists():
        logger.error(f"File tidak ditemukan: {raw_file}")
        logger.error("Jalankan Stage 1 (ingestion) dulu!")
        return []

    logger.info(f"═══ Stage 2: Processing {brand_name} ═══")

    with open(raw_file, "r", encoding="utf-8") as f:
        pages = json.load(f)

    logger.info(f"Loaded {len(pages)} pages dari {raw_file}")

    all_chunks: list[ProcessedChunk] = []

    for page in pages:
        try:
            # Clean text
            cleaned = clean_text(page.get("text", ""))
            if not cleaned or len(cleaned) < 50:
                logger.debug(f"Skip (terlalu pendek): {page.get('url', 'unknown')}")
                continue

            # Detect category
            category = _detect_page_category(
                page.get("url", ""),
                page.get("title", ""),
                cleaned,
            )

            # Detect language (simple heuristic)
            language = "id"  # Default Indonesian
            id_words = ["dan", "atau", "yang", "untuk", "dengan", "dari"]
            en_words = ["and", "the", "for", "with", "from", "this"]
            text_lower = cleaned.lower()
            id_count = sum(1 for w in id_words if f" {w} " in text_lower)
            en_count = sum(1 for w in en_words if f" {w} " in text_lower)
            if en_count > id_count * 2:
                language = "en"

            # Prepend title ke konten untuk context
            full_content = cleaned
            title = page.get("title", "").strip()
            if title:
                full_content = f"{title}\n\n{cleaned}"

            # Chunk
            text_chunks = chunk_text(full_content)

            for idx, chunk_text_content in enumerate(text_chunks):
                chunk = ProcessedChunk(
                    chunk_id=_generate_chunk_id(page.get("doc_id", ""), idx),
                    doc_id=page.get("doc_id", ""),
                    source=page.get("source", ""),
                    brand_name=brand_name,
                    industry=page.get("industry", "skincare"),
                    category=category,
                    title=title,
                    content=chunk_text_content,
                    language=language,
                    metadata={
                        "url": page.get("url", ""),
                        "chunk_index": idx,
                        "total_chunks": len(text_chunks),
                        "meta_description": page.get("meta_description", ""),
                        "scraped_at": page.get("scraped_at", ""),
                    },
                )
                all_chunks.append(chunk)

            logger.debug(
                f"  {page.get('url', '')}: {len(text_chunks)} chunks, "
                f"category={category}, lang={language}"
            )

        except Exception as e:
            logger.warning(
                f"✗ Error processing {page.get('url', 'unknown')}: {e}"
            )
            continue

    # Save processed chunks
    output_dir = PROCESSED_DIR / brand_key
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f, ensure_ascii=False, indent=2)

    logger.info(
        f"═══ Stage 2 selesai: {len(all_chunks)} chunks → {output_file} ═══"
    )
    return all_chunks


# ─── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VISIO.ID Stage 2: Processing")
    parser.add_argument("--brand-name", required=True, help="Nama brand")
    args = parser.parse_args()

    process_brand_data(args.brand_name)
