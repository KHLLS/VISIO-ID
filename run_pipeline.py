"""
VISIO.ID — Pipeline Runner
CLI entry point untuk menjalankan pipeline ingestion → processing → embedding.

Usage:
    # Full pipeline
    python run_pipeline.py --brand-url URL --brand-name "Nama" --industry skincare

    # Dry-run (tanpa upload ke Supabase)
    python run_pipeline.py --brand-url URL --brand-name "Nama" --dry-run

    # Run stage tertentu
    python run_pipeline.py --stage 1 --brand-url URL --brand-name "Nama"
    python run_pipeline.py --stage 2 --brand-name "Nama"
    python run_pipeline.py --stage 3 --brand-name "Nama" --dry-run

    # Limit pages (untuk testing)
    python run_pipeline.py --brand-url URL --brand-name "Nama" --max-pages 3 --dry-run
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.config import get_logger

logger = get_logger("run_pipeline")


def run_stage1(args) -> bool:
    """Jalankan Stage 1: Ingestion."""
    if not args.brand_url:
        logger.error("--brand-url wajib untuk Stage 1!")
        return False

    from backend.pipeline.stage1_ingestion import crawl_brand_site

    results = crawl_brand_site(
        base_url=args.brand_url,
        brand_name=args.brand_name,
        max_pages=args.max_pages,
        industry=args.industry,
    )
    return len(results) > 0


def run_stage2(args) -> bool:
    """Jalankan Stage 2: Processing."""
    from backend.pipeline.stage2_processing import process_brand_data

    chunks = process_brand_data(args.brand_name)
    return len(chunks) > 0


def run_stage3(args) -> bool:
    """Jalankan Stage 3: Embedding & Upload."""
    from backend.pipeline.stage3_embedding import embed_brand_data

    embedded = embed_brand_data(
        brand_name=args.brand_name,
        skip_upload=args.dry_run,
    )
    return len(embedded) > 0


def run_stage4(args) -> bool:
    """Jalankan Stage 4: GEO Audit (RAG)."""
    from backend.pipeline.stage4_rag import run_geo_audit

    query = getattr(args, "query", None) or "Analisis visibilitas brand ini di AI search engines"
    result = run_geo_audit(
        query=query,
        brand_name=args.brand_name,
        industry=args.industry,
    )

    # Print result
    import json
    print("\n" + "=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result.get("audit", {}).get("geo_score", 0) >= 0


def main():
    parser = argparse.ArgumentParser(
        description="VISIO.ID — RAG Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --brand-url https://somethinc.com --brand-name "Somethinc" --industry skincare
  python run_pipeline.py --brand-url https://somethinc.com --brand-name "Somethinc" --dry-run --max-pages 3
  python run_pipeline.py --stage 2 --brand-name "Somethinc"
        """,
    )
    parser.add_argument(
        "--brand-url", type=str, default=None,
        help="URL website brand (wajib untuk stage 1)"
    )
    parser.add_argument(
        "--brand-name", type=str, required=True,
        help="Nama brand"
    )
    parser.add_argument(
        "--industry", type=str, default="skincare",
        help="Industri brand (default: skincare)"
    )
    parser.add_argument(
        "--stage", type=int, default=None, choices=[1, 2, 3, 4],
        help="Jalankan stage tertentu saja (default: 1-3, stage 4 = on-demand audit)"
    )
    parser.add_argument(
        "--max-pages", type=int, default=50,
        help="Max halaman per domain (default: 50)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip upload ke Supabase"
    )
    parser.add_argument(
        "--audit", action="store_true",
        help="Mode audit (skip ingestion, langsung processing + embedding)"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Query untuk GEO audit (stage 4)"
    )

    args = parser.parse_args()

    # Determine stages to run
    if args.stage:
        stages = [args.stage]
    elif args.audit:
        stages = [2, 3]
    else:
        stages = [1, 2, 3]

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║         VISIO.ID — RAG Pipeline             ║")
    logger.info("╚══════════════════════════════════════════════╝")
    logger.info(f"Brand    : {args.brand_name}")
    logger.info(f"Industry : {args.industry}")
    logger.info(f"Stages   : {stages}")
    logger.info(f"Dry-run  : {args.dry_run}")
    if args.brand_url:
        logger.info(f"URL      : {args.brand_url}")
    logger.info("")

    start_time = time.time()
    stage_runners = {1: run_stage1, 2: run_stage2, 3: run_stage3, 4: run_stage4}

    for stage_num in stages:
        runner = stage_runners[stage_num]
        success = runner(args)
        if not success:
            logger.error(f"Stage {stage_num} gagal. Pipeline berhenti.")
            sys.exit(1)
        logger.info("")

    elapsed = time.time() - start_time
    logger.info(f"✓ Pipeline selesai dalam {elapsed:.1f} detik")


if __name__ == "__main__":
    main()
