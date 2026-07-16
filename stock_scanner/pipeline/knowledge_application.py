"""Knowledge Application Engine v1 — applies curated AI Lab knowledge
(data/published/knowledge_report.json) as a bounded ranking adjustment in
the morning scan.

Scope (v1, no Knowledge Compiler yet):
    - Reads the existing published JSON directly — no SQLite, no live
      knowledge_entries query, ever. Mirrors stock_scanner.db.model_lookup
      .get_promoted_model()'s "never raises" contract.
    - Only applies knowledge_entries rows with lifecycle_status in
      {confirmed, strong} — never emerging/weakening/contradicted/archived.
      lifecycle_status answers "how statistically mature is this belief" —
      it is NOT a deployment decision (see next bullet).
    - Only applies entries with promotion_status == 'promoted' exactly —
      a second, orthogonal gate answering "has a human approved this for
      production." candidate/rejected/archived/missing/null/anything else
      all mean not promoted (fail closed). Checked via
      stock_scanner.ai_lab.schemas.is_entry_promoted() — the single
      reusable implementation of this check; this module does not
      duplicate it. knowledge_base_engine.py only ever writes 'candidate';
      nothing in this codebase sets 'promoted' automatically — that
      requires a future human-run promotion tool
      (scripts/promote_knowledge.py, not built yet). Statistical maturity
      alone is never sufficient to influence production ranking.
    - Only applies conditions that are either a boolean technical-indicator
      column (exact True/False match) or `sector` (via
      stock_scanner.reference.issuers.get_sector). Any entry containing an
      unsupported condition (numeric Low/Mid/High tercile — no persisted
      bucket boundary exists yet; or an AI-Lab-internal dimension —
      ai_model/recommendation/historical_verdict, which has no production
      analogue and would require replaying AI Lab live) is dropped in
      full, not partially applied — applying a subset of a validated
      condition-set would extrapolate beyond what was actually tested.
    - Never calls an LLM, never touches signal/final_status (classification
      and hard gates), never changes which tier a ticker belongs to — only
      adds an additive, capped adjustment on top of whichever score
      run_daily_scan._save_ranked would otherwise rank by first.

Kept modular so a future Knowledge Compiler can replace `_load_entries`'s
source (a precompiled artifact instead of the raw JSON) without changing
`apply_knowledge_ranking`'s signature or the columns it produces.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from stock_scanner.ai_lab.schemas import is_entry_promoted
from stock_scanner.reference.issuers import get_sector

_DEFAULT_KNOWLEDGE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "published" / "knowledge_report.json"
)

_AI_ONLY_DIMENSIONS = {"ai_model", "recommendation", "historical_verdict"}
_NUMERIC_TERCILE_VALUES = {"Low", "Mid", "High"}
_BOOLEAN_VALUES = {"True", "False"}

_APPLICABLE_LIFECYCLE_STATUSES = {"confirmed", "strong"}
_LIFECYCLE_MULTIPLIER = {"strong": 1.0, "confirmed": 0.5}

_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "bonus_scale": 1.0,
    "max_total_bonus": 0.5,
}

# Columns _save_ranked already resolves in this priority order — Knowledge
# Application reuses the same resolution so its adjustment always applies
# on top of whatever rules/ML already decided, never in place of it.
_BASE_SCORE_CASCADE = ["quality_adjusted_score", "promoted_rule_score", "ml_prob", "total_score"]


# ---------------------------------------------------------------------------
# Loading (no SQLite, never raises)
# ---------------------------------------------------------------------------

def _load_entries(path: Path | None = None) -> list[dict]:
    """Read data/published/knowledge_report.json. Missing file, malformed
    JSON, or an unexpected shape all just return [] — a knowledge lookup
    failure must never break the morning scan (same contract as
    stock_scanner.db.model_lookup.get_promoted_model)."""
    path = path or _DEFAULT_KNOWLEDGE_PATH
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []
    entries = data.get("entries")
    return entries if isinstance(entries, list) else []


# ---------------------------------------------------------------------------
# Condition classification + entry filtering (done once, not per-candidate)
# ---------------------------------------------------------------------------

def _condition_supported(dimension: str, value: str) -> bool:
    if dimension in _AI_ONLY_DIMENSIONS:
        return False
    if dimension == "sector":
        return True
    if value in _NUMERIC_TERCILE_VALUES:
        return False
    if value in _BOOLEAN_VALUES:
        return True
    return False  # unknown shape — conservative default, never assumed safe


def _entry_applicable(entry: dict) -> bool:
    if str(entry.get("lifecycle_status", "")).lower() not in _APPLICABLE_LIFECYCLE_STATUSES:
        return False
    if not is_entry_promoted(entry):  # centralized check — see stock_scanner.ai_lab.schemas
        return False
    conditions = entry.get("conditions") or []
    if not conditions:
        return False
    return all(_condition_supported(dim, value) for dim, value in conditions)


def filter_applicable_entries(entries: list[dict]) -> list[dict]:
    """Keep only entries this engine is allowed to apply in production:
    confirmed/strong lifecycle AND promotion_status == 'promoted' (fail
    closed on anything else) AND every condition boolean- or sector-based.
    Computed once per apply_knowledge_ranking() call, not once per
    candidate row."""
    applicable = [e for e in entries if _entry_applicable(e)]
    if entries and not applicable:
        logger.info("Knowledge Application: 0/{} entries applicable (lifecycle/dimension gates)", len(entries))
    return applicable


# ---------------------------------------------------------------------------
# Matching + scoring
# ---------------------------------------------------------------------------

def _condition_matches(dimension: str, value: str, row: pd.Series, sector: str) -> bool:
    if dimension == "sector":
        return sector == value
    if dimension not in row.index or pd.isna(row[dimension]):
        return False  # missing/NaN column is never a match, never a default
    return bool(row[dimension]) == (value == "True")


def _matched_entries(entries: list[dict], row: pd.Series, sector: str) -> list[dict]:
    matched = []
    for entry in entries:
        conditions = entry.get("conditions") or []
        if all(_condition_matches(dim, val, row, sector) for dim, val in conditions):
            matched.append(entry)
    return matched


def _entry_bonus(entry: dict, bonus_scale: float) -> float:
    win_rate = entry.get("shrunk_win_rate")
    if win_rate is None:
        return 0.0
    multiplier = _LIFECYCLE_MULTIPLIER.get(str(entry.get("lifecycle_status", "")).lower(), 0.0)
    return (float(win_rate) - 0.5) * bonus_scale * multiplier


def _resolve_base_score_column(df: pd.DataFrame) -> str | None:
    for col in _BASE_SCORE_CASCADE:
        if col in df.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_knowledge_ranking(
    df: pd.DataFrame,
    knowledge_path: Path | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Additive, deterministic ranking adjustment from curated AI Lab
    knowledge. Never modifies `signal`/`final_status` or any existing
    column — only adds new columns, so a caller that ignores them sees no
    behavior change at all.

    Returns `df` completely unchanged (not even a copy) if knowledge
    application is disabled, no entries are applicable, or `df` has no
    resolvable base score column — this is what makes output byte-identical
    when no applicable knowledge exists.
    """
    cfg = {**_DEFAULTS, **(config or {})}
    if not cfg.get("enabled", True):
        return df
    if df is None or df.empty:
        return df

    base_col = _resolve_base_score_column(df)
    if base_col is None:
        return df

    entries = filter_applicable_entries(_load_entries(knowledge_path))
    if not entries:
        return df

    bonus_scale = float(cfg["bonus_scale"])
    max_total_bonus = float(cfg["max_total_bonus"])

    df = df.copy()
    bonuses: list[float] = []
    matched_ids: list[str] = []
    applied_rules: list[str] = []

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", ""))
        sector = get_sector(ticker) if ticker else ""
        matched = _matched_entries(entries, row, sector)

        rules = []
        total = 0.0
        for entry in matched:
            individual_bonus = _entry_bonus(entry, bonus_scale)
            total += individual_bonus
            rules.append({
                "knowledge_id": entry.get("knowledge_id"),
                "conditions": entry.get("conditions"),
                "lifecycle_status": entry.get("lifecycle_status"),
                "shrunk_win_rate": entry.get("shrunk_win_rate"),
                "individual_bonus": round(individual_bonus, 4),
            })
        total = max(-max_total_bonus, min(max_total_bonus, total))

        bonuses.append(round(total, 4))
        matched_ids.append(json.dumps([e.get("knowledge_id") for e in matched]))
        applied_rules.append(json.dumps(rules, default=str))

    df["knowledge_bonus"] = bonuses
    df["knowledge_adjusted_score"] = df[base_col] + df["knowledge_bonus"]
    df["knowledge_matched_ids"] = matched_ids
    df["knowledge_applied_rules"] = applied_rules

    n_applied = sum(1 for b in bonuses if b != 0.0)
    logger.info(
        "Knowledge Application: {} applicable entries, {} candidates adjusted (base={})",
        len(entries), n_applied, base_col,
    )
    return df
