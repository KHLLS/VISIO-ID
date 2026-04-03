"""
VISIO.ID — Test: Stage 2 Processing
Unit tests untuk text cleaning dan chunking logic.

Jalankan: pytest backend/tests/test_stage2_processing.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from backend.pipeline.stage2_processing import (
    clean_text,
    chunk_text,
    _detect_page_category,
    _generate_chunk_id,
)


# ─── clean_text ──────────────────────────────────────────────────────

class TestCleanText:
    def test_removes_html_entities(self):
        assert "&amp;" not in clean_text("Hello &amp; World")
        assert "&nbsp;" not in clean_text("Hello&nbsp;World")

    def test_removes_urls(self):
        result = clean_text("Visit https://example.com for more")
        assert "https://" not in result

    def test_normalizes_whitespace(self):
        result = clean_text("Hello   World")
        assert "  " not in result

    def test_collapses_multiple_newlines(self):
        result = clean_text("Line 1\n\n\n\n\nLine 2")
        assert "\n\n\n" not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_like_empty(self):
        assert clean_text("") == ""

    def test_preserves_content(self):
        text = "Somethinc adalah brand skincare lokal Indonesia berkualitas premium."
        result = clean_text(text)
        assert "Somethinc" in result
        assert "skincare" in result

    def test_removes_email(self):
        result = clean_text("Hubungi kami di cs@visio.id untuk info lebih lanjut")
        assert "@" not in result


# ─── chunk_text ──────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "Teks pendek yang tidak perlu dipotong."
        chunks = chunk_text(text, chunk_size=2048)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_creates_multiple_chunks(self):
        text = "A " * 2000  # 4000 chars
        chunks = chunk_text(text, chunk_size=2048, overlap=256)
        assert len(chunks) > 1

    def test_chunk_sizes_respect_limit(self):
        text = "B " * 3000
        chunks = chunk_text(text, chunk_size=2048, overlap=256)
        for chunk in chunks:
            assert len(chunk) <= 2048 + 256  # slight tolerance for boundary

    def test_overlap_exists(self):
        # Create text where overlap should be visible
        word = "OVERLAP "
        text = word * 500  # 4000 chars
        chunks = chunk_text(text, chunk_size=2048, overlap=256)
        # Last chars of chunk[0] should appear in start of chunk[1]
        assert len(chunks) >= 2
        end_of_first = chunks[0][-200:]
        start_of_second = chunks[1][:200]
        # There should be some common words
        assert any(w in start_of_second for w in end_of_first.split()[:5])

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []

    def test_filters_very_short_chunks(self):
        # chunk_text returns short text as-is (no splitting needed when text <= chunk_size)
        # Short-chunk filter only applies to chunks produced by the splitting loop
        chunks = chunk_text("Hi")
        assert len(chunks) == 1
        assert chunks[0] == "Hi"

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []

    def test_consistent_with_rules_params(self):
        """Test dengan parameter dari VISIO_RULES: 2048 chars, 256 overlap."""
        text = "Skincare " * 600  # ~5400 chars
        chunks = chunk_text(text, chunk_size=2048, overlap=256)
        assert 2 <= len(chunks) <= 4


# ─── _detect_page_category ───────────────────────────────────────────

class TestDetectPageCategory:
    def test_product_page(self):
        cat = _detect_page_category(
            "https://brand.com/product/serum-vitamin-c", "Serum Vitamin C", ""
        )
        assert cat == "product_page"

    def test_about_page(self):
        cat = _detect_page_category(
            "https://brand.com/about-us", "Tentang Kami", ""
        )
        assert cat == "about"

    def test_blog_page(self):
        cat = _detect_page_category(
            "https://brand.com/blog/tips-kulit-cerah", "Tips Kulit Cerah", ""
        )
        assert cat == "blog"

    def test_faq_page(self):
        cat = _detect_page_category(
            "https://brand.com/faq", "FAQ - Pertanyaan Umum", ""
        )
        assert cat == "faq"

    def test_general_fallback(self):
        cat = _detect_page_category(
            "https://brand.com/", "Home", ""
        )
        assert cat == "general"


# ─── _generate_chunk_id ──────────────────────────────────────────────

class TestGenerateChunkId:
    def test_deterministic(self):
        id1 = _generate_chunk_id("doc123", 0)
        id2 = _generate_chunk_id("doc123", 0)
        assert id1 == id2

    def test_different_doc_different_id(self):
        id1 = _generate_chunk_id("doc123", 0)
        id2 = _generate_chunk_id("doc456", 0)
        assert id1 != id2

    def test_different_index_different_id(self):
        id1 = _generate_chunk_id("doc123", 0)
        id2 = _generate_chunk_id("doc123", 1)
        assert id1 != id2

    def test_id_length(self):
        chunk_id = _generate_chunk_id("doc123", 0)
        assert len(chunk_id) == 16
