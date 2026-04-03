"""
VISIO.ID — Test: Stage 4 RAG
Unit tests untuk hybrid retrieval, RRF merge, reranker, dan prompt parsing.

Jalankan: pytest backend/tests/test_stage4_rag.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from backend.pipeline.stage4_rag import (
    reciprocal_rank_fusion,
    _parse_audit_response,
)


# ─── reciprocal_rank_fusion ─────────────────────────────────────────

class TestReciprocalRankFusion:
    def _make_result(self, chunk_id: str, score: float = 1.0) -> dict:
        return {
            "chunk_id": chunk_id,
            "content": f"Content for {chunk_id}",
            "source": "test.com",
            "category": "general",
            "metadata": {},
            "score": score,
            "retrieval_method": "test",
        }

    def test_single_list(self):
        """Satu list harus di-return dengan RRF scores."""
        results = [
            self._make_result("a"),
            self._make_result("b"),
            self._make_result("c"),
        ]
        merged = reciprocal_rank_fusion([results])
        assert len(merged) == 3
        # First item should have highest RRF score
        assert merged[0]["chunk_id"] == "a"
        assert all("rrf_score" in r for r in merged)

    def test_two_lists_overlap(self):
        """Chunk yang muncul di kedua list harus punya RRF score lebih tinggi."""
        list1 = [self._make_result("a"), self._make_result("b")]
        list2 = [self._make_result("b"), self._make_result("c")]
        merged = reciprocal_rank_fusion([list1, list2])
        assert len(merged) == 3
        # "b" appears in both lists, should have highest score
        assert merged[0]["chunk_id"] == "b"

    def test_empty_lists(self):
        """Empty lists harus return empty."""
        merged = reciprocal_rank_fusion([])
        assert len(merged) == 0

    def test_all_unique(self):
        """Semua unique chunks harus ada di merged result."""
        list1 = [self._make_result("a")]
        list2 = [self._make_result("b")]
        list3 = [self._make_result("c")]
        merged = reciprocal_rank_fusion([list1, list2, list3])
        ids = {r["chunk_id"] for r in merged}
        assert ids == {"a", "b", "c"}

    def test_rrf_scores_decreasing(self):
        """RRF scores harus descending (sorted)."""
        results = [self._make_result(f"item_{i}") for i in range(10)]
        merged = reciprocal_rank_fusion([results])
        scores = [r["rrf_score"] for r in merged]
        assert scores == sorted(scores, reverse=True)

    def test_retrieval_method_set_to_hybrid(self):
        """Semua merged results harus punya retrieval_method = hybrid."""
        list1 = [self._make_result("a")]
        merged = reciprocal_rank_fusion([list1])
        assert merged[0]["retrieval_method"] == "hybrid"


# ─── _parse_audit_response ──────────────────────────────────────────

class TestParseAuditResponse:
    def test_valid_json(self):
        raw = '{"geo_score": 35, "issues": ["a", "b"], "recommendations": ["x"], "summary": "Test"}'
        result = _parse_audit_response(raw)
        assert result["geo_score"] == 35
        assert len(result["issues"]) == 2
        assert result["summary"] == "Test"

    def test_json_in_code_block(self):
        raw = '```json\n{"geo_score": 42, "issues": ["a"], "recommendations": ["b"], "summary": "OK"}\n```'
        result = _parse_audit_response(raw)
        assert result["geo_score"] == 42

    def test_json_in_generic_code_block(self):
        raw = '```\n{"geo_score": 50, "issues": [], "recommendations": [], "summary": "Good"}\n```'
        result = _parse_audit_response(raw)
        assert result["geo_score"] == 50

    def test_invalid_json_returns_fallback(self):
        raw = "This is not JSON at all, just plain text response."
        result = _parse_audit_response(raw)
        assert result["geo_score"] == 0
        assert "tidak dapat di-parse" in result["issues"][0]

    def test_partial_json(self):
        raw = '{"geo_score": 60}'
        result = _parse_audit_response(raw)
        assert result["geo_score"] == 60
        assert result["issues"] == []

    def test_geo_score_as_string(self):
        raw = '{"geo_score": "75", "issues": [], "recommendations": [], "summary": "Nice"}'
        result = _parse_audit_response(raw)
        assert result["geo_score"] == 75


# ─── BM25 Search (basic logic) ──────────────────────────────────────

class TestBM25SearchLogic:
    """Test BM25 search logic tanpa file dependency."""

    def test_bm25_okapi_basic(self):
        """Verify BM25Okapi works dengan data sederhana."""
        from rank_bm25 import BM25Okapi
        import numpy as np

        corpus = [
            "serum vitamin c untuk kulit cerah".split(),
            "moisturizer untuk kulit kering".split(),
            "sunscreen perlindungan UV".split(),
        ]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores("vitamin c serum".split())

        # First doc should score highest (most relevant)
        assert np.argmax(scores) == 0

    def test_bm25_empty_query(self):
        """Empty query harus tetap aman."""
        from rank_bm25 import BM25Okapi

        corpus = [["hello", "world"]]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores([])
        assert len(scores) == 1
