#!/usr/bin/env python3
"""Build the top signals >10% table + published JSON artifact.

Standalone, non-production daily persistence: filters already-evaluated
signals (data/performance/signal_results.csv) down to forward_return_pct >
10%, ranks them, and mirrors the result to a local SQLite table + committed
JSON. Does NOT touch signal_engine.py, scanner_config.yaml, ml_ranker.py, or
any promotion path, and is not the knowledge_base table (Learning Agent
Phase 1) — see stock_scanner/pipeline/top_signals.py and
stock_scanner/db/top_signals.py.

Intended to run once per day, after the evening evaluation step that
(re)writes data/performance/signal_results.csv — see
.github/workflows/performance.yml, step "Build top signals (>10%) table +
JSON mirror".

Usage:
    python scripts/build_top_signals.py
    python scripts/build_top_signals.py --results data/performance/signal_results.csv
"""
import argparse
import os
import sys
import uuid
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from stock_scanner.db.init_db import create_schema, get_connection  # noqa: E402
from stock_scanner.db.top_signals import (  # noqa: E402
    export_top_signals,
    import_top_signals,
    upsert_top_signals,
)
from stock_scanner.pipeline.top_signals import RETURN_THRESHOLD, build_top_signals  # noqa: E402

_DEFAULT_RESULTS = repo_root / "data" / "performance" / "signal_results.csv"
_DEFAULT_RANKED_DIR = repo_root / "data" / "ranked"


def main(results_path: Path, ranked_dir: Path) -> None:
    if not results_path.exists():
        logger.warning(f"top_signals: {results_path} not found — nothing to do.")
        return

    results = pd.read_csv(results_path)
    logger.info(f"top_signals: {len(results)} total signal_results rows loaded")

    top = build_top_signals(results, ranked_dir, threshold=RETURN_THRESHOLD)
    logger.info(
        f"top_signals: {len(top)} evaluated signal(s) with forward_return_pct > "
        f"{RETURN_THRESHOLD:.0%}"
    )
    if not top.empty:
        n_enriched = int((top["quality_source"] == "ranked_csv").sum())
        logger.info(f"top_signals: {n_enriched}/{len(top)} enriched with quality scores from ranked CSVs")

    source_run_id = os.environ.get("GITHUB_RUN_ID", str(uuid.uuid4()))

    conn = get_connection()
    create_schema(conn)   # idempotent — CREATE TABLE IF NOT EXISTS, adds top_signals if missing
    n_imported = import_top_signals(conn)
    logger.info(f"top_signals: imported {n_imported} historical row(s) from published mirror")

    upsert_top_signals(conn, top, source_run_id=source_run_id, threshold_pct=RETURN_THRESHOLD * 100)
    export_path = export_top_signals(conn)
    logger.info(f"top_signals: exported published artifact → {export_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=_DEFAULT_RESULTS,
                        help="Path to data/performance/signal_results.csv")
    parser.add_argument("--ranked-dir", type=Path, default=_DEFAULT_RANKED_DIR,
                        help="Path to data/ranked/ (best-effort quality-score enrichment)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.results, args.ranked_dir)
