"""
VISIO.ID — GEO Scorer Service
Menghitung GEO Score (0–100) dari hasil audit RAG Stage 4.

GEO Score mengukur seberapa baik sebuah brand terlihat di AI search engines
(ChatGPT, Perplexity, Gemini). Score dibagi 3 dimensi:
  - Presence  (0–40): Seberapa sering brand muncul di AI responses
  - Accuracy  (0–40): Keakuratan info yang diambil AI tentang brand
  - Sentiment (0–20): Sentimen konteks brand di AI results

Grade:
  A = 80–100  (brand sangat visible di AI search)
  B = 65–79
  C = 50–64
  D = 35–49
  F = 0–34    (brand hampir tidak terdeteksi AI)

Penggunaan:
    from backend.services.geo_scorer import calculate_geo_score

    score_data = calculate_geo_score(audit_result)
    print(score_data["score"])  # 72
    print(score_data["grade"])  # B
"""

from typing import Any
from backend.config import get_logger

logger = get_logger(__name__)

# ─── Grade thresholds ─────────────────────────────────────────────────
GRADE_THRESHOLDS = [
    (80, "A"),
    (65, "B"),
    (50, "C"),
    (35, "D"),
    (0,  "F"),
]

# ─── Keywords untuk heuristik scoring ────────────────────────────────
_PRESENCE_POSITIVE_SIGNALS = [
    "disebutkan", "dikenal", "brand", "muncul", "tersedia",
    "ditemukan", "referenced", "mentioned", "visible",
]
_PRESENCE_NEGATIVE_SIGNALS = [
    "tidak ditemukan", "tidak ada data", "tidak dikenal",
    "tidak tersedia", "no data", "not found",
]
_ACCURACY_POSITIVE_SIGNALS = [
    "akurat", "sesuai", "benar", "konsisten", "tepat",
    "accurate", "correct", "consistent",
]
_ACCURACY_NEGATIVE_SIGNALS = [
    "tidak akurat", "salah", "tidak konsisten", "misleading",
    "inaccurate", "incorrect", "wrong",
]
_SENTIMENT_POSITIVE_SIGNALS = [
    "positif", "baik", "unggul", "terpercaya", "direkomendasikan",
    "positive", "good", "trusted", "recommended", "terbaik",
]
_SENTIMENT_NEGATIVE_SIGNALS = [
    "negatif", "buruk", "keluhan", "masalah",
    "negative", "bad", "complaint", "issue",
]


class GeoScorerService:
    """
    Service untuk menghitung GEO Score dari hasil audit RAG.
    """

    def calculate(self, audit_result: dict[str, Any]) -> dict[str, Any]:
        """
        Hitung GEO Score dari raw audit result.

        Args:
            audit_result: Dict output dari stage4_rag.run_geo_audit().
                          Harus punya key: 'llm_response', 'chunks_retrieved',
                          'sources', 'brand_name'

        Returns:
            Dict dengan keys: score, grade, presence_score, accuracy_score,
            sentiment_score, issues, strengths, recommendations
        """
        llm_text = audit_result.get("llm_response", "").lower()
        chunks_retrieved = audit_result.get("chunks_retrieved", 0)
        brand_name = audit_result.get("brand_name", "brand ini")

        # ── Presence Score (0–40) ────────────────────────────────────
        presence = self._score_presence(llm_text, chunks_retrieved)

        # ── Accuracy Score (0–40) ────────────────────────────────────
        accuracy = self._score_accuracy(llm_text)

        # ── Sentiment Score (0–20) ───────────────────────────────────
        sentiment = self._score_sentiment(llm_text)

        total = presence + accuracy + sentiment
        grade = self._calculate_grade(total)

        issues, strengths, recommendations = self._extract_insights(
            llm_text, presence, accuracy, sentiment, brand_name
        )

        result = {
            "score": total,
            "grade": grade,
            "presence_score": presence,
            "accuracy_score": accuracy,
            "sentiment_score": sentiment,
            "issues": issues,
            "strengths": strengths,
            "recommendations": recommendations,
        }

        logger.info(
            f"GEO Score kalkulasi selesai: brand={brand_name}, "
            f"score={total} ({grade}), "
            f"presence={presence}, accuracy={accuracy}, sentiment={sentiment}"
        )

        return result

    def _score_presence(self, text: str, chunks_retrieved: int) -> int:
        """
        Score presence (0–40).
        Berdasarkan: chunks yang ditemukan + sinyal kata di LLM response.
        """
        # Chunk-based score (0–20)
        if chunks_retrieved >= 10:
            chunk_score = 20
        elif chunks_retrieved >= 5:
            chunk_score = 14
        elif chunks_retrieved >= 2:
            chunk_score = 8
        elif chunks_retrieved == 1:
            chunk_score = 4
        else:
            chunk_score = 0

        # Keyword-based score (0–20)
        positive_hits = sum(1 for s in _PRESENCE_POSITIVE_SIGNALS if s in text)
        negative_hits = sum(1 for s in _PRESENCE_NEGATIVE_SIGNALS if s in text)
        keyword_score = max(0, min(20, (positive_hits * 4) - (negative_hits * 6)))

        return chunk_score + keyword_score

    def _score_accuracy(self, text: str) -> int:
        """
        Score akurasi informasi (0–40).
        Berdasarkan sinyal kata akurasi di LLM response.
        """
        positive_hits = sum(1 for s in _ACCURACY_POSITIVE_SIGNALS if s in text)
        negative_hits = sum(1 for s in _ACCURACY_NEGATIVE_SIGNALS if s in text)
        raw = (positive_hits * 8) - (negative_hits * 10)
        return max(0, min(40, raw + 20))  # baseline 20

    def _score_sentiment(self, text: str) -> int:
        """
        Score sentimen konteks brand (0–20).
        Berdasarkan sinyal kata sentimen di LLM response.
        """
        positive_hits = sum(1 for s in _SENTIMENT_POSITIVE_SIGNALS if s in text)
        negative_hits = sum(1 for s in _SENTIMENT_NEGATIVE_SIGNALS if s in text)
        raw = (positive_hits * 4) - (negative_hits * 5)
        return max(0, min(20, raw + 10))  # baseline 10

    def _calculate_grade(self, score: int) -> str:
        """Konversi score numerik ke grade huruf."""
        for threshold, grade in GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return "F"

    def _extract_insights(
        self,
        text: str,
        presence: int,
        accuracy: int,
        sentiment: int,
        brand_name: str,
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Ekstrak issues, strengths, dan recommendations dari skor dan teks audit.

        Returns:
            Tuple: (issues, strengths, recommendations)
        """
        issues = []
        strengths = []
        recommendations = []

        # ── Issues ───────────────────────────────────────────────────
        if presence <= 10:
            issues.append(
                f"Brand '{brand_name}' hampir tidak terdeteksi di AI search engines"
            )
        if accuracy <= 15:
            issues.append("Informasi brand yang diambil AI tidak akurat atau tidak konsisten")
        if sentiment <= 5:
            issues.append("Sentimen brand dalam konteks AI cenderung negatif atau netral")
        if presence <= 5 and accuracy <= 15:
            issues.append("Konten website belum dioptimasi untuk AI retrieval (GEO)")

        # ── Strengths ────────────────────────────────────────────────
        if presence >= 25:
            strengths.append("Brand memiliki kehadiran yang kuat di AI search results")
        if accuracy >= 30:
            strengths.append("Informasi brand yang diambil AI akurat dan konsisten")
        if sentiment >= 14:
            strengths.append("Brand mendapat konteks positif di AI responses")
        if not strengths:
            strengths.append("Brand memiliki potensi untuk ditingkatkan visibilitasnya di AI")

        # ── Recommendations (top 3 gratis, sisanya paywall) ──────────
        if presence <= 15:
            recommendations.append(
                "Tambahkan konten terstruktur (FAQ, definisi produk) yang mudah diekstrak AI"
            )
        if accuracy <= 25:
            recommendations.append(
                "Konsistenkan nama produk dan klaim di seluruh halaman website"
            )
        if sentiment <= 10:
            recommendations.append(
                "Tambahkan halaman testimonial dan review yang terstruktur"
            )

        # Rekomendasi generik kalau kurang dari 3
        generic_recs = [
            "Buat halaman 'Tentang Kami' yang detail dengan informasi brand yang terstruktur",
            "Tambahkan schema markup (JSON-LD) untuk produk dan brand",
            "Optimalkan meta description setiap halaman dengan keyword yang relevan",
            "Perbarui konten secara rutin agar AI indexer menemukan informasi terbaru",
        ]
        for rec in generic_recs:
            if len(recommendations) >= 3:
                break
            if rec not in recommendations:
                recommendations.append(rec)

        return issues, strengths, recommendations[:3]  # max 3 rekomendasi gratis


# ─── Convenience function ─────────────────────────────────────────────

def calculate_geo_score(audit_result: dict[str, Any]) -> dict[str, Any]:
    """
    Shortcut function untuk menghitung GEO Score.
    Gunakan ini kalau tidak perlu instance GeoScorerService.

    Args:
        audit_result: Output dari stage4_rag.run_geo_audit()

    Returns:
        Dict GEO Score dengan keys: score, grade, presence_score,
        accuracy_score, sentiment_score, issues, strengths, recommendations
    """
    scorer = GeoScorerService()
    return scorer.calculate(audit_result)
