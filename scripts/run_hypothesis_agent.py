#!/usr/bin/env python3
"""Learning Agent — Phase 2 (LLM articulation over de-duplicated clusters).

Read-only research tool: writes only to the knowledge_base table, never to
signals/model_registry/promotion_decisions/scanner_config.yaml. Manual
invocation only — no workflow/schedule exists for this yet. See
docs/LEARNING_AGENT_ARCHITECTURE.md.

Usage:
    python scripts/run_hypothesis_agent.py --mock            # exercise full path, no real LLM
    python scripts/run_hypothesis_agent.py                   # real 9router call (raises until wired)
    python scripts/run_hypothesis_agent.py --dedup-report data/reports/pattern_dedup_2026-07-13.json
"""
import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from loguru import logger  # noqa: E402

from stock_scanner.db.init_db import create_schema, get_connection  # noqa: E402
from stock_scanner.db.knowledge_base import export_knowledge_base, write_hypotheses  # noqa: E402
from stock_scanner.learning.hypothesis_agent import (  # noqa: E402
    MockLLMClient,
    NineRouterClient,
    generate_hypotheses,
)
from stock_scanner.learning.pattern_dedup import ClusteredPattern  # noqa: E402
from stock_scanner.learning.pattern_miner import PatternCandidate  # noqa: E402

_REPORTS_DIR = repo_root / "data" / "reports"


def _load_clusters(dedup_report_path: Path) -> list[ClusteredPattern]:
    data = json.loads(dedup_report_path.read_text())
    clusters = []
    for c in data.get("clusters", []):
        representative = PatternCandidate.from_dict(c["representative"])
        members = [PatternCandidate.from_dict(m) for m in c["members"]]
        clusters.append(ClusteredPattern(
            cluster_id=c["cluster_id"], representative=representative,
            member_count=c["member_count"], members=members,
        ))
    return clusters


def main(dedup_report_path: Path, use_mock: bool, api_key: str | None) -> None:
    if not dedup_report_path.exists():
        logger.error(f"Dedup report not found: {dedup_report_path}. Run scripts/run_pattern_dedup.py first.")
        return

    clusters = _load_clusters(dedup_report_path)
    logger.info(f"Loaded {len(clusters)} clusters from {dedup_report_path}")
    if not clusters:
        logger.warning("No clusters to process.")
        return

    client = MockLLMClient() if use_mock else NineRouterClient(api_key=api_key)
    if not use_mock:
        logger.warning(
            "Using NineRouterClient — this will raise NotImplementedError until 9router API "
            "details are configured (see docs/LEARNING_AGENT_ARCHITECTURE.md). Use --mock to "
            "exercise the rest of the pipeline in the meantime."
        )

    hypotheses = generate_hypotheses(clusters, client)
    if not hypotheses:
        logger.warning("No hypotheses generated (all responses failed validation, or all clusters skipped).")
        return

    pattern_json_by_cluster_id = {c.cluster_id: c.to_dict() for c in clusters}
    source_run_id = f"run_hypothesis_agent_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    conn = get_connection()
    create_schema(conn)   # idempotent — CREATE TABLE IF NOT EXISTS, adds knowledge_base if missing
    n_inserted = write_hypotheses(conn, hypotheses, source_run_id, pattern_json_by_cluster_id)
    logger.info(f"knowledge_base: {n_inserted} new row(s) inserted (source_run_id={source_run_id})")
    export_path = export_knowledge_base(conn)
    conn.close()
    logger.info(f"knowledge_base mirror exported -> {export_path}")


if __name__ == "__main__":
    today_str = date.today().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Learning Agent Phase 2 — LLM hypothesis articulation")
    parser.add_argument("--dedup-report", type=Path, default=_REPORTS_DIR / f"pattern_dedup_{today_str}.json")
    parser.add_argument("--mock", action="store_true", help="Use MockLLMClient — no real network call.")
    parser.add_argument("--api-key", default=None, help="9router API key (unused until the client is wired).")
    args = parser.parse_args()
    main(args.dedup_report, args.mock, args.api_key)
