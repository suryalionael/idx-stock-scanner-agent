"""Read-only lookup of the currently promoted model — from the COMMITTED
JSON mirror of model_registry (data/published/model_registry.json), never
from SQLite.

Why: the morning scan (scan.yml) is intentionally dependency-light (see
docs/AUTOMATION.md — scanner extras only, no DB rebuild step). SQLite
itself is gitignored and rebuilt fresh from committed sources by
performance.yml/train_challenger.yml (see scripts/init_db_and_backfill.py).
The JSON mirror that stock_scanner/db/registry_io.py already maintains for
CI durability (model_registry/promotion_decisions survive ephemeral
runners) doubles as this lookup's read path — no new infrastructure, no new
dependency in the scan job.
"""
import json
from pathlib import Path

_DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent.parent / "data" / "published" / "model_registry.json"


def get_promoted_model(model_type: str, registry_path: Path | None = None) -> dict | None:
    """Return the model_registry row with status='promoted' for model_type.

    Never raises: a missing file, malformed JSON, or the absence of a
    promoted row are all just None — a lookup failure here must never break
    the morning scan (see run_daily_scan.py's promoted-challenger step).
    """
    path = registry_path or _DEFAULT_REGISTRY_PATH
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None

    for row in data.get("model_registry", []):
        if row.get("model_type") == model_type and row.get("status") == "promoted":
            return row
    return None
