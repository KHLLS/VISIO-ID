"""
VISIO.ID — Stage 1: Data Ingestion
Web scraper untuk mengambil konten dari website brand skincare lokal.

Fitur:
- Crawl halaman dari satu domain (max 50 pages)
- Respectful crawling dengan delay antar request
- robots.txt compliance
- Per-URL error handling (satu gagal tidak crash semua)
- Output: JSON files di data/raw/{brand_name}/
"""

import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import (
    RAW_DIR,
    REQUEST_DELAY,
    MAX_PAGES_PER_DOMAIN,
    USER_AGENT,
    SKIP_URL_PATTERNS,
    get_logger,
)

logger = get_logger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────


def _generate_doc_id(url: str) -> str:
    """Generate deterministic doc_id dari URL untuk idempotency."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _should_skip_url(url: str) -> bool:
    """Cek apakah URL harus di-skip (cart, login, assets, dll)."""
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in SKIP_URL_PATTERNS)


def _is_same_domain(url: str, base_domain: str) -> bool:
    """Cek apakah URL masih di domain yang sama."""
    parsed = urlparse(url)
    return parsed.netloc == base_domain or parsed.netloc == ""


def _get_robots_disallowed(base_url: str, session: requests.Session) -> list[str]:
    """Parse robots.txt dan return list path yang di-disallow."""
    disallowed = []
    robots_url = urljoin(base_url, "/robots.txt")
    try:
        resp = session.get(robots_url, timeout=10)
        if resp.status_code == 200:
            current_agent_applies = False
            for line in resp.text.splitlines():
                line = line.strip()
                if line.lower().startswith("user-agent:"):
                    agent = line.split(":", 1)[1].strip().lower()
                    current_agent_applies = agent == "*" or "visiobot" in agent
                elif line.lower().startswith("disallow:") and current_agent_applies:
                    path = line.split(":", 1)[1].strip()
                    if path:
                        disallowed.append(path)
            logger.info(f"robots.txt: {len(disallowed)} disallowed paths")
    except Exception as e:
        logger.warning(f"Gagal baca robots.txt: {e}")
    return disallowed


def _is_disallowed(url: str, disallowed_paths: list[str]) -> bool:
    """Cek apakah URL di-disallow oleh robots.txt."""
    parsed = urlparse(url)
    path = parsed.path
    return any(path.startswith(d) for d in disallowed_paths)


# ─── Core Functions ──────────────────────────────────────────────────


def extract_page_content(
    url: str, session: requests.Session
) -> Optional[dict]:
    """
    Ambil dan extract konten dari satu halaman web.

    Args:
        url: URL halaman yang akan di-scrape
        session: requests.Session untuk reuse connection

    Returns:
        dict dengan keys: url, title, text, meta_description, headings, links
        None jika gagal
    """
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()

        # Skip non-HTML responses
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            logger.debug(f"Skip non-HTML: {url} ({content_type})")
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "iframe", "noscript"]):
            tag.decompose()

        # Extract title
        title = ""
        if soup.title:
            title = soup.title.get_text(strip=True)

        # Extract meta description
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"].strip()

        # Extract main text
        body = soup.find("body")
        text = body.get_text(separator="\n", strip=True) if body else ""

        # Extract headings for structure
        headings = []
        for level in range(1, 4):
            for h in soup.find_all(f"h{level}"):
                h_text = h.get_text(strip=True)
                if h_text:
                    headings.append({"level": level, "text": h_text})

        # Extract internal links for crawling
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(url, href)
            links.append(full_url)

        logger.info(f"✓ Scraped: {url} ({len(text)} chars)")

        return {
            "url": url,
            "doc_id": _generate_doc_id(url),
            "title": title,
            "meta_description": meta_desc,
            "text": text,
            "headings": headings,
            "links": links,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }

    except requests.exceptions.RequestException as e:
        logger.warning(f"✗ Gagal scrape {url}: {e}")
        return None
    except Exception as e:
        logger.warning(f"✗ Error parsing {url}: {e}")
        return None


def crawl_brand_site(
    base_url: str,
    brand_name: str,
    max_pages: int = MAX_PAGES_PER_DOMAIN,
    industry: str = "skincare",
) -> list[dict]:
    """
    Crawl website brand skincare dan simpan hasilnya sebagai JSON.

    Fungsi ini idempotent — menjalankan ulang akan overwrite data lama.

    Args:
        base_url: URL utama brand (contoh: https://www.somethinc.com)
        brand_name: Nama brand untuk folder output
        max_pages: Maksimum halaman yang di-crawl (default: 50)
        industry: Industri brand (default: skincare)

    Returns:
        List of dicts berisi konten halaman yang berhasil di-scrape
    """
    logger.info(f"═══ Stage 1: Crawling {brand_name} ({base_url}) ═══")
    logger.info(f"Max pages: {max_pages}, Delay: {REQUEST_DELAY}s")

    # Setup session
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
    })

    # Parse base domain
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc

    # Check robots.txt
    disallowed = _get_robots_disallowed(base_url, session)

    # BFS crawl
    visited = set()
    queue = [base_url]
    results = []

    while queue and len(results) < max_pages:
        url = queue.pop(0)

        # Normalize URL (strip fragment, trailing slash)
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        normalized = normalized.rstrip("/")

        # Skip checks
        if normalized in visited:
            continue
        if _should_skip_url(normalized):
            logger.debug(f"Skip (pattern): {normalized}")
            continue
        if _is_disallowed(normalized, disallowed):
            logger.debug(f"Skip (robots.txt): {normalized}")
            continue
        if not _is_same_domain(normalized, base_domain):
            continue

        visited.add(normalized)

        # Crawl with delay
        if len(results) > 0:
            time.sleep(REQUEST_DELAY)

        page_data = extract_page_content(normalized, session)
        if page_data is None:
            continue

        # Add metadata
        page_data["brand_name"] = brand_name
        page_data["industry"] = industry
        page_data["source"] = base_domain
        results.append(page_data)

        # Add discovered links to queue
        for link in page_data.get("links", []):
            parsed_link = urlparse(link)
            norm_link = f"{parsed_link.scheme}://{parsed_link.netloc}{parsed_link.path}".rstrip("/")
            if norm_link not in visited and _is_same_domain(link, base_domain):
                queue.append(link)

        # Remove links from stored data (tidak perlu di-persist)
        page_data.pop("links", None)

        logger.info(f"Progress: {len(results)}/{max_pages} pages")

    # Save results
    output_dir = RAW_DIR / brand_name.lower().replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "pages.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"═══ Stage 1 selesai: {len(results)} pages → {output_file} ═══")
    return results


# ─── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VISIO.ID Stage 1: Ingestion")
    parser.add_argument("--brand-url", required=True, help="URL website brand")
    parser.add_argument("--brand-name", required=True, help="Nama brand")
    parser.add_argument("--industry", default="skincare", help="Industri")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES_PER_DOMAIN)
    args = parser.parse_args()

    crawl_brand_site(
        base_url=args.brand_url,
        brand_name=args.brand_name,
        max_pages=args.max_pages,
        industry=args.industry,
    )
