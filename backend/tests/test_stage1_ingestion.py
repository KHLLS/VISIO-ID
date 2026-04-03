"""
VISIO.ID — Test: Stage 1 Ingestion (helper functions)
Unit tests untuk URL filtering dan helper functions scraper.

Jalankan: pytest backend/tests/test_stage1_ingestion.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from backend.pipeline.stage1_ingestion import (
    _generate_doc_id,
    _should_skip_url,
    _is_same_domain,
)


# ─── _generate_doc_id ────────────────────────────────────────────────

class TestGenerateDocId:
    def test_deterministic(self):
        id1 = _generate_doc_id("https://example.com/page")
        id2 = _generate_doc_id("https://example.com/page")
        assert id1 == id2

    def test_different_urls_different_ids(self):
        id1 = _generate_doc_id("https://example.com/page1")
        id2 = _generate_doc_id("https://example.com/page2")
        assert id1 != id2

    def test_id_length(self):
        doc_id = _generate_doc_id("https://example.com")
        assert len(doc_id) == 12


# ─── _should_skip_url ────────────────────────────────────────────────

class TestShouldSkipUrl:
    def test_skip_cart(self):
        assert _should_skip_url("https://brand.com/cart") is True

    def test_skip_checkout(self):
        assert _should_skip_url("https://brand.com/checkout") is True

    def test_skip_login(self):
        assert _should_skip_url("https://brand.com/login") is True

    def test_skip_register(self):
        assert _should_skip_url("https://brand.com/register") is True

    def test_skip_image(self):
        assert _should_skip_url("https://brand.com/assets/hero.jpg") is True

    def test_skip_pdf(self):
        assert _should_skip_url("https://brand.com/catalog.pdf") is True

    def test_skip_css(self):
        assert _should_skip_url("https://brand.com/styles.css") is True

    def test_skip_js(self):
        assert _should_skip_url("https://brand.com/app.js") is True

    def test_allow_product_page(self):
        assert _should_skip_url("https://brand.com/product/serum-c") is False

    def test_allow_about_page(self):
        assert _should_skip_url("https://brand.com/about") is False

    def test_allow_blog_page(self):
        assert _should_skip_url("https://brand.com/blog/skincare-tips") is False

    def test_allow_homepage(self):
        assert _should_skip_url("https://brand.com/") is False


# ─── _is_same_domain ─────────────────────────────────────────────────

class TestIsSameDomain:
    def test_same_domain(self):
        assert _is_same_domain("https://brand.com/page", "brand.com") is True

    def test_different_domain(self):
        assert _is_same_domain("https://other.com/page", "brand.com") is False

    def test_relative_url(self):
        # Relative URLs have empty netloc
        assert _is_same_domain("/about", "brand.com") is True

    def test_subdomain_is_different(self):
        assert _is_same_domain("https://shop.brand.com/page", "brand.com") is False

    def test_www_prefix(self):
        assert _is_same_domain("https://www.brand.com/page", "www.brand.com") is True
