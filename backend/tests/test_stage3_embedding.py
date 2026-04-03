"""
VISIO.ID — Test: Stage 3 Embedding
Unit tests untuk generate_embeddings, upload_to_supabase, dan embed_brand_data.

Semua tests menggunakan unittest.mock — tidak perlu GPU, model download, atau internet.

Jalankan: pytest backend/tests/test_stage3_embedding.py -v
"""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from backend.pipeline.stage3_embedding import (
    generate_embeddings,
    upload_to_supabase,
    embed_brand_data,
)


# ─── Fixtures ────────────────────────────────────────────────────────

def _make_chunks(n: int = 3) -> list[dict]:
    """Helper: buat dummy chunks untuk testing."""
    return [
        {
            "chunk_id": f"chunk_{i:04d}",
            "doc_id": f"doc_001",
            "source": "https://brand.com/page",
            "category": "product_page",
            "content": f"Ini adalah konten chunk nomor {i} untuk brand skincare.",
            "language": "id",
            "brand_name": "TestBrand",
            "industry": "skincare",
            "title": "Produk Unggulan",
            "metadata": {"url": "https://brand.com/page", "chunk_index": i, "total_chunks": n},
        }
        for i in range(n)
    ]


# ─── generate_embeddings ─────────────────────────────────────────────

class TestGenerateEmbeddings:
    def test_empty_list_returns_empty(self):
        """Input kosong → output kosong tanpa memanggil model."""
        result = generate_embeddings([])
        assert result == []

    @patch("backend.pipeline.stage3_embedding._get_model")
    def test_attaches_embedding_key(self, mock_get_model):
        """Setiap output chunk harus punya key 'embedding'."""
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((2, 768), dtype="float32")
        mock_get_model.return_value = mock_model

        chunks = _make_chunks(2)
        result = generate_embeddings(chunks)

        assert all("embedding" in c for c in result)

    @patch("backend.pipeline.stage3_embedding._get_model")
    def test_output_count_matches_input(self, mock_get_model):
        """Jumlah output == jumlah input chunks."""
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((5, 768), dtype="float32")
        mock_get_model.return_value = mock_model

        chunks = _make_chunks(5)
        result = generate_embeddings(chunks)

        assert len(result) == 5

    @patch("backend.pipeline.stage3_embedding._get_model")
    def test_embedding_is_list_of_floats(self, mock_get_model):
        """Embedding harus berupa list of floats (bukan numpy array)."""
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 768), dtype="float32")
        mock_get_model.return_value = mock_model

        chunks = _make_chunks(1)
        result = generate_embeddings(chunks)

        embedding = result[0]["embedding"]
        assert isinstance(embedding, list)
        assert all(isinstance(v, float) for v in embedding)

    @patch("backend.pipeline.stage3_embedding._get_model")
    def test_embedding_dimension_768(self, mock_get_model):
        """Embedding harus berdimensi 768 (multilingual-e5-base)."""
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 768), dtype="float32")
        mock_get_model.return_value = mock_model

        chunks = _make_chunks(1)
        result = generate_embeddings(chunks)

        assert len(result[0]["embedding"]) == 768

    @patch("backend.pipeline.stage3_embedding._get_model")
    def test_original_chunk_fields_preserved(self, mock_get_model):
        """Field asli chunk (chunk_id, content, dll) tidak hilang setelah embed."""
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 768), dtype="float32")
        mock_get_model.return_value = mock_model

        chunks = _make_chunks(1)
        result = generate_embeddings(chunks)

        assert result[0]["chunk_id"] == chunks[0]["chunk_id"]
        assert result[0]["content"] == chunks[0]["content"]
        assert result[0]["source"] == chunks[0]["source"]

    @patch("backend.pipeline.stage3_embedding._get_model")
    def test_doc_prefix_added_to_text(self, mock_get_model):
        """Model harus dipanggil dengan prefix 'passage: ' sesuai rules."""
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 768), dtype="float32")
        mock_get_model.return_value = mock_model

        chunks = _make_chunks(1)
        generate_embeddings(chunks)

        # Ambil args yang dikirim ke encode
        call_args = mock_model.encode.call_args
        texts_sent = call_args[0][0]
        assert texts_sent[0].startswith("passage: ")


# ─── upload_to_supabase ──────────────────────────────────────────────

class TestUploadToSupabase:
    def test_raises_without_credentials(self):
        """
        upload_to_supabase harus raise ValueError kalau SUPABASE_URL
        atau SUPABASE_SERVICE_KEY kosong.
        """
        with patch("backend.pipeline.stage3_embedding.SUPABASE_URL", ""):
            with patch("backend.pipeline.stage3_embedding.SUPABASE_SERVICE_KEY", ""):
                with pytest.raises(ValueError, match="SUPABASE_URL"):
                    upload_to_supabase([{"chunk_id": "x", "embedding": [0.1] * 768}])

    @patch("backend.pipeline.stage3_embedding._get_supabase_client")
    def test_returns_uploaded_count(self, mock_get_client):
        """Return value harus sama dengan jumlah chunks yang berhasil di-upload."""
        mock_table = MagicMock()
        mock_table.upsert.return_value.execute.return_value = MagicMock()
        mock_client = MagicMock()
        mock_client.table.return_value = mock_table
        mock_get_client.return_value = mock_client

        import numpy as np
        chunks = _make_chunks(3)
        for c in chunks:
            c["embedding"] = [0.1] * 768

        result = upload_to_supabase(chunks)
        assert result == 3

    @patch("backend.pipeline.stage3_embedding._get_supabase_client")
    def test_uses_upsert_not_insert(self, mock_get_client):
        """Harus pakai upsert (bukan insert) sesuai VISIO_RULES."""
        mock_table = MagicMock()
        mock_table.upsert.return_value.execute.return_value = MagicMock()
        mock_client = MagicMock()
        mock_client.table.return_value = mock_table
        mock_get_client.return_value = mock_client

        chunks = _make_chunks(1)
        for c in chunks:
            c["embedding"] = [0.1] * 768

        upload_to_supabase(chunks)
        mock_table.upsert.assert_called_once()

    @patch("backend.pipeline.stage3_embedding._get_supabase_client")
    def test_batch_error_does_not_crash(self, mock_get_client):
        """Satu batch error → log warning + skip, tidak crash seluruh upload."""
        mock_table = MagicMock()
        mock_table.upsert.return_value.execute.side_effect = Exception("Network error")
        mock_client = MagicMock()
        mock_client.table.return_value = mock_table
        mock_get_client.return_value = mock_client

        chunks = _make_chunks(2)
        for c in chunks:
            c["embedding"] = [0.1] * 768

        # Tidak boleh raise exception
        result = upload_to_supabase(chunks)
        assert result == 0  # 0 berhasil karena semua batch error


# ─── embed_brand_data ────────────────────────────────────────────────

class TestEmbedBrandData:
    def test_returns_empty_if_chunks_not_found(self):
        """Kalau chunks.json tidak ada, return [] dan tidak crash."""
        result = embed_brand_data("brand_yang_tidak_ada_xyz", skip_upload=True)
        assert result == []

    @patch("backend.pipeline.stage3_embedding.generate_embeddings")
    @patch("backend.pipeline.stage3_embedding.upload_to_supabase")
    def test_skip_upload_when_flag_set(
        self, mock_upload, mock_embed, tmp_path
    ):
        """skip_upload=True → upload_to_supabase tidak dipanggil."""
        import numpy as np
        # Buat struktur file sementara
        brand_key = "testbrand"
        processed_dir = tmp_path / "processed" / brand_key
        processed_dir.mkdir(parents=True)
        chunks_file = processed_dir / "chunks.json"
        chunks = _make_chunks(2)
        chunks_file.write_text(json.dumps(chunks), encoding="utf-8")

        # Mock embedding output
        embedded = [dict(c) for c in chunks]
        for c in embedded:
            c["embedding"] = [0.1] * 768
        mock_embed.return_value = embedded

        with patch("backend.pipeline.stage3_embedding.PROCESSED_DIR", tmp_path / "processed"):
            with patch("backend.pipeline.stage3_embedding.EMBEDDINGS_DIR", tmp_path / "embeddings"):
                result = embed_brand_data("testbrand", skip_upload=True)

        mock_upload.assert_not_called()
        assert len(result) == 2
