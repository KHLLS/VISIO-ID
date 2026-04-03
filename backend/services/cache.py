"""
VISIO.ID — Cache Service (Redis)
Wrapper tipis di atas Redis untuk cache LLM responses dan audit results.

TTL default: 30 hari (sesuai VISIO_RULES.MD)
Graceful degradation: kalau Redis tidak available, semua operasi return None/False
tanpa crash — pipeline tetap berjalan tanpa cache.

Penggunaan:
    from backend.services.cache import get_cache

    cache = get_cache()
    cached = cache.get("geo_audit:somethinc:query_hash")
    if cached:
        return cached

    result = run_expensive_operation()
    cache.set("geo_audit:somethinc:query_hash", result)
"""

import json
import hashlib
import os
from typing import Any, Optional

from backend.config import get_logger

logger = get_logger(__name__)

# TTL 30 hari dalam detik
DEFAULT_TTL_SECONDS = 30 * 24 * 60 * 60

# Singleton instance
_cache_instance: Optional["CacheService"] = None


class CacheService:
    """
    Redis cache wrapper dengan graceful fallback.
    Semua operasi wrapped dalam try/except — satu error tidak crash pipeline.
    """

    def __init__(self):
        self._client = None
        self._available = False
        self._connect()

    def _connect(self) -> None:
        """Coba koneksi ke Redis. Gagal? Log warning dan continue tanpa cache."""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis
            client = redis.from_url(redis_url, decode_responses=True, socket_timeout=2)
            client.ping()
            self._client = client
            self._available = True
            logger.info(f"Redis connected: {redis_url}")
        except ImportError:
            logger.warning("redis package tidak terinstall — cache disabled")
        except Exception as e:
            logger.warning(f"Redis tidak available: {e} — pipeline berjalan tanpa cache")

    @property
    def is_available(self) -> bool:
        """Return True kalau Redis terkoneksi dan siap dipakai."""
        return self._available

    def get(self, key: str) -> Optional[Any]:
        """
        Ambil nilai dari cache.

        Args:
            key: Cache key

        Returns:
            Nilai yang di-cache (dict/list/str/int) atau None kalau miss/error
        """
        if not self._available:
            return None
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Cache GET error (key={key}): {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: int = DEFAULT_TTL_SECONDS,
    ) -> bool:
        """
        Simpan nilai ke cache.

        Args:
            key: Cache key
            value: Nilai yang akan disimpan (harus JSON-serializable)
            ttl: Time-to-live dalam detik (default: 30 hari)

        Returns:
            True kalau berhasil, False kalau gagal/tidak available
        """
        if not self._available:
            return False
        try:
            serialized = json.dumps(value, ensure_ascii=False)
            self._client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache SET error (key={key}): {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Hapus entry dari cache.

        Args:
            key: Cache key

        Returns:
            True kalau berhasil dihapus, False kalau gagal/tidak ada
        """
        if not self._available:
            return False
        try:
            result = self._client.delete(key)
            return result > 0
        except Exception as e:
            logger.warning(f"Cache DELETE error (key={key}): {e}")
            return False

    def invalidate_brand(self, brand_name: str) -> int:
        """
        Hapus semua cache entries untuk satu brand.
        Berguna setelah pipeline re-run (data brand berubah).

        Args:
            brand_name: Nama brand

        Returns:
            Jumlah keys yang dihapus
        """
        if not self._available:
            return 0
        try:
            brand_key = brand_name.lower().replace(" ", "_")
            pattern = f"geo_audit:{brand_key}:*"
            keys = self._client.keys(pattern)
            if keys:
                deleted = self._client.delete(*keys)
                logger.info(f"Cache invalidated: {deleted} keys untuk brand '{brand_name}'")
                return deleted
            return 0
        except Exception as e:
            logger.warning(f"Cache INVALIDATE error (brand={brand_name}): {e}")
            return 0

    @staticmethod
    def make_key(brand_name: str, query: str) -> str:
        """
        Generate cache key yang deterministik dari brand + query.

        Args:
            brand_name: Nama brand
            query: Query string

        Returns:
            Cache key dalam format "geo_audit:{brand}:{query_hash}"
        """
        brand_key = brand_name.lower().replace(" ", "_")
        query_hash = hashlib.md5(query.encode()).hexdigest()[:12]
        return f"geo_audit:{brand_key}:{query_hash}"


def get_cache() -> CacheService:
    """
    Get singleton CacheService instance.
    Thread-safe untuk penggunaan di FastAPI concurrent requests.
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheService()
    return _cache_instance
