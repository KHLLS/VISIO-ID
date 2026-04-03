"""
VISIO.ID — Test Supabase Connection
Script one-off untuk verifikasi koneksi ke Supabase + tabel visio_documents.
Jalankan: python scripts/test_supabase_connection.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_TABLE

print("=" * 55)
print("VISIO.ID — Supabase Connection Test")
print("=" * 55)

# ── 1. Cek env vars ───────────────────────────────────────
print("\n[1] Cek environment variables...")
if not SUPABASE_URL:
    print("  ✗ SUPABASE_URL kosong — cek .env!")
    sys.exit(1)
if not SUPABASE_SERVICE_KEY:
    print("  ✗ SUPABASE_SERVICE_KEY kosong — cek .env!")
    sys.exit(1)
print(f"  ✓ SUPABASE_URL    : {SUPABASE_URL}")
print(f"  ✓ SERVICE_KEY     : {SUPABASE_SERVICE_KEY[:20]}...")

# ── 2. Koneksi ke Supabase ────────────────────────────────
print("\n[2] Koneksi ke Supabase...")
try:
    from supabase import create_client
    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("  ✓ Supabase client berhasil dibuat")
except ImportError:
    print("  ✗ Package 'supabase' belum terinstall.")
    print("    Jalankan: pip install supabase")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Koneksi gagal: {e}")
    sys.exit(1)

# ── 3. Cek tabel visio_documents ─────────────────────────
print(f"\n[3] Cek tabel '{SUPABASE_TABLE}'...")
try:
    result = client.table(SUPABASE_TABLE).select("id").limit(1).execute()
    count = len(result.data)
    print(f"  ✓ Tabel ditemukan (rows di sample: {count})")
except Exception as e:
    print(f"  ✗ Tabel tidak ditemukan atau error: {e}")
    print("    Pastikan migration SQL sudah dijalankan di Supabase!")
    sys.exit(1)

# ── 4. Test upsert dummy row ──────────────────────────────
print("\n[4] Test upsert dummy row...")
dummy = {
    "chunk_id": "test_connection_chunk_001",
    "doc_id": "test_doc_001",
    "source": "https://visio.id/test",
    "category": "test",
    "content": "Ini adalah test koneksi VISIO.ID",
    "language": "id",
    "metadata": {"brand_name": "TestBrand", "industry": "test"},
    "embedding": [0.0] * 768,
}
try:
    client.table(SUPABASE_TABLE).upsert(dummy, on_conflict="chunk_id").execute()
    print("  ✓ Upsert berhasil")
except Exception as e:
    print(f"  ✗ Upsert gagal: {e}")
    sys.exit(1)

# ── 5. Hapus dummy row ────────────────────────────────────
print("\n[5] Cleanup dummy row...")
try:
    client.table(SUPABASE_TABLE).delete().eq("chunk_id", "test_connection_chunk_001").execute()
    print("  ✓ Cleanup berhasil")
except Exception as e:
    print(f"  ⚠ Cleanup gagal (tidak kritis): {e}")

print("\n" + "=" * 55)
print("✅ SEMUA TEST PASSED — Supabase siap digunakan!")
print("=" * 55)
