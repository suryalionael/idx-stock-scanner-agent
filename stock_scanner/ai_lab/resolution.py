"""AI Lab — recommendation resolver ("Performance Tracker" in the AI Lab
architecture docs — distinct from stock_scanner.pipeline.performance, the
unrelated production performance tracker).

Advances every AI Lab recommendation's lifecycle against forward OHLCV
data: PENDING -> ACTIVE (entry_price set once generated_date's close is
available) and ACTIVE -> CLOSED | EXPIRED (TP/SL simulation, same fallback
exit policy as stock_scanner.pipeline.evaluator). Also refreshes running
highest_price/lowest_price/max_runup_pct/max_drawdown_pct/holding_days on
every ACTIVE row each run, even when it doesn't resolve this time — see
stock_scanner/ai_lab/resolver.py.

Completely standalone: reads data/raw/*.parquet (read-only, produced by the
existing scan pipeline) and writes only to ai_recommendations /
ai_learning_events — never touches signals/outcomes/model_registry or any
production scanner file.

Extracted from scripts/resolve_ai_lab.py so stock_scanner.ai_lab.pipeline
can call it in-process as part of the automated AI pipeline. This stage has
no async work (no LLM calls) — scripts/resolve_ai_lab.py remains the CLI
entry point, now a thin wrapper around run() below.
"""
from pathlib import Path

from loguru import logger

from stock_scanner.ai_lab.pipeline_status import PipelineStageStatus
from stock_scanner.ai_lab.resolver import activate_pending, resolve_active
from stock_scanner.db.ai_lab import (
    export_ai_recommendations,
    export_learning_events,
    import_ai_recommendations,
    import_learning_events,
    load_recommendations,
    log_learning_event,
    update_recommendation_status,
)
from stock_scanner.db.init_db import create_schema, get_connection

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_RAW_DIR = _REPO_ROOT / "data" / "raw"


def run(raw_dir: Path | None = None, horizon_days: int = 10, risk_pct: float = 3.0) -> dict:
    """Activate PENDING recommendations and track/resolve ACTIVE ones.

    Always reports OK — resolving zero pending/active recommendations is a
    normal, successful outcome for this bookkeeping pass, not a skip.
    """
    raw_dir = raw_dir or _DEFAULT_RAW_DIR
    conn = get_connection()
    create_schema(conn)
    import_ai_recommendations(conn)
    import_learning_events(conn)

    pending = load_recommendations(conn, status="PENDING")
    activations = activate_pending(pending, raw_dir) if not pending.empty else []
    for row in activations:
        rec_id = row.pop("id")
        status = row.pop("status")
        update_recommendation_status(conn, rec_id, status, **row)
    logger.info(f"ai_lab: activated {len(activations)}/{len(pending)} PENDING recommendation(s)")

    active = load_recommendations(conn, status="ACTIVE")
    tracked = resolve_active(active, raw_dir, horizon_days=horizon_days, risk_pct=risk_pct) if not active.empty else []
    closed = won = lost = expired = still_active = 0
    for row in tracked:
        rec_id = row.pop("id")
        status = row.pop("status", None)
        if status is None:
            status = "ACTIVE"
            still_active += 1
        else:
            closed += 1
            if status == "EXPIRED":
                expired += 1
            if row.get("trade_outcome") == "WIN":
                won += 1
            elif row.get("trade_outcome") == "LOSS":
                lost += 1
        update_recommendation_status(conn, rec_id, status, **row)
    logger.info(
        f"ai_lab: tracked {len(tracked)}/{len(active)} ACTIVE recommendation(s) — "
        f"{closed} resolved ({won} WIN, {lost} LOSS, {expired} EXPIRED), {still_active} still tracking"
    )

    if activations or tracked:
        log_learning_event(
            conn, event_type="outcome_resolved",
            description=f"Resolved {len(activations)} activation(s) and {len(tracked)} tracking update(s): "
                        f"{closed} closed ({won} WIN, {lost} LOSS, {expired} EXPIRED), {still_active} still tracking.",
            metadata={"activated": len(activations), "closed": closed, "won": won,
                      "lost": lost, "expired": expired, "still_active": still_active},
        )

    export_ai_recommendations(conn)
    export_learning_events(conn)
    logger.info("ai_lab: published data/published/ai_recommendations.json + ai_learning_events.json")

    return {
        "status": PipelineStageStatus.OK,
        "activated": len(activations),
        "closed": closed,
        "won": won,
        "lost": lost,
        "expired": expired,
        "still_active": still_active,
    }
