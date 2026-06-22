"""Round-trip model_registry + promotion_decisions to a committed JSON file.

Why this exists: data/db/signals.db is gitignored (rebuildable from
committed sources — see scripts/init_db_and_backfill.py), so a GitHub
Actions runner starts every job with a FRESH, empty database. signals/
feature_snapshots/outcomes can all be rebuilt from committed CSV/parquet,
but model_registry and promotion_decisions are NOT derived from anything
else — they're the only durable record of "what models were trained, with
what metrics, and what was promoted/rejected and why." Losing that on every
ephemeral runner would defeat the entire point of having a registry.

Fix: export these two (small, metadata-only — never bulk feature data) to
data/published/model_registry.json after every training/promotion run, and
import that file's rows back into a fresh DB before training. The training
workflow commits the updated file back to main, same pattern already used
by scan.yml/performance.yml for data/published/.
"""
import json
import sqlite3
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "published" / "model_registry.json"

_REGISTRY_COLS = [
    "model_version_id", "model_type", "feature_list_json",
    "train_start_date", "train_end_date", "val_start_date", "val_end_date",
    "test_start_date", "test_end_date", "label_threshold_pct", "label_horizon",
    "train_metrics_json", "val_metrics_json", "test_metrics_json",
    "sensitivity_json", "artifact_path", "status", "promoted_at", "retired_at", "trained_at",
]
_DECISION_COLS = [
    "decision_id", "challenger_model_id", "production_model_id", "decision",
    "challenger_metrics_json", "production_metrics_json", "reason", "decided_at",
]
_MONITORING_COLS = [
    "monitor_date", "production_model_id", "n_signals", "n_evaluated",
    "realized_win_rate", "realized_precision_at_threshold", "avg_predicted_prob",
    "feature_drift_flag", "alert_triggered",
]


def export_registry(conn: sqlite3.Connection, path: Path | None = None) -> Path:
    path = path or _DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = conn.cursor()
    models = [dict(zip(_REGISTRY_COLS, row))
              for row in cur.execute(f"SELECT {','.join(_REGISTRY_COLS)} FROM model_registry").fetchall()]
    decisions = [dict(zip(_DECISION_COLS, row))
                 for row in cur.execute(f"SELECT {','.join(_DECISION_COLS)} FROM promotion_decisions").fetchall()]
    monitoring = [dict(zip(_MONITORING_COLS, row))
                  for row in cur.execute(f"SELECT {','.join(_MONITORING_COLS)} FROM live_monitoring").fetchall()]
    path.write_text(json.dumps({"model_registry": models, "promotion_decisions": decisions,
                                 "live_monitoring": monitoring}, indent=2))
    return path


def import_registry(conn: sqlite3.Connection, path: Path | None = None) -> dict:
    """Idempotent — safe to call on a DB that already has some/all of these
    rows (INSERT OR IGNORE on the primary key)."""
    path = path or _DEFAULT_PATH
    if not path.exists():
        return {"model_registry": 0, "promotion_decisions": 0, "live_monitoring": 0}
    data = json.loads(path.read_text())
    cur = conn.cursor()

    n_models = 0
    for row in data.get("model_registry", []):
        cur.execute(
            f"""INSERT OR IGNORE INTO model_registry ({','.join(_REGISTRY_COLS)})
                VALUES ({','.join('?' * len(_REGISTRY_COLS))})""",
            tuple(row.get(c) for c in _REGISTRY_COLS),
        )
        n_models += cur.rowcount

    n_decisions = 0
    for row in data.get("promotion_decisions", []):
        cur.execute(
            f"""INSERT OR IGNORE INTO promotion_decisions ({','.join(_DECISION_COLS)})
                VALUES ({','.join('?' * len(_DECISION_COLS))})""",
            tuple(row.get(c) for c in _DECISION_COLS),
        )
        n_decisions += cur.rowcount

    # live_monitoring uses REPLACE (not IGNORE): monitor_date is the PK and a
    # re-run on the same day should refresh the row, not skip it.
    n_monitoring = 0
    for row in data.get("live_monitoring", []):
        cur.execute(
            f"""INSERT OR REPLACE INTO live_monitoring ({','.join(_MONITORING_COLS)})
                VALUES ({','.join('?' * len(_MONITORING_COLS))})""",
            tuple(row.get(c) for c in _MONITORING_COLS),
        )
        n_monitoring += 1

    conn.commit()
    return {"model_registry": n_models, "promotion_decisions": n_decisions, "live_monitoring": n_monitoring}
