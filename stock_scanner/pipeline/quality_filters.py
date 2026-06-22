"""Quality filter layer — hard exclusion + risk-flag penalty.

Tiga fungsi utama dipanggil setelah signal_engine, sebelum ML ranking:

    evaluate_hard_filters(row, config)
        → passes_hard_filters, exclusion_reason, risk_flags (dari data)

    apply_quality_penalty(row, config)
        → quality_adjusted_score (total_score dikurangi penalty)

    assign_final_status(row)
        → final_status string

    enrich_df_with_quality_filters(df, config, risk_overrides)
        → DataFrame dengan kolom baru ditambahkan

Taxonomy final_status
    eligible               — lolos semua filter, tidak ada risk flag
    watch_with_risk        — lolos hard filter tapi ada ≥1 risk flag
    excluded_fundamental   — DER > der_max OR PBV > pbv_max
    excluded_regulatory    — is_uma OR is_special_monitoring (manual override)
    excluded_float_structure — public_float_pct < float_hard_min_pct
    insufficient_data      — fundamental_status == "missing"

Status yang TIDAK diimplementasikan (karena tidak ada data real):
    excluded_distribution  — butuh RIDR dari data broker real (semua mock saat ini)

Kolom yang ditambahkan ke DataFrame
    passes_hard_filters    bool
    exclusion_reason       str | None
    risk_flags             str   (comma-separated, untuk display)
    quality_adjusted_score float (total_score - penalties, clipped 0-10)
    final_status           str
    is_uma                 bool  (dari uma_overrides.csv)
    is_special_monitoring  bool  (dari uma_overrides.csv)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Default config values (override via scanner_config.yaml quality_filters:)
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "der_max":                      4.0,
    "pbv_max":                      3.5,
    "ebitda_negative_exclude":      False,
    "vol_spike_exhaustion_ratio":   5.0,
    "vol_spike_cap_threshold_idr":  1_000_000_000_000,  # Rp 1T
    "float_hard_min_pct":           5.0,
    "float_watch_min_pct":          10.0,
    "float_warn_max_pct":           70.0,
    "penalty_ebitda_negative":      0.5,
    "penalty_der_elevated":         0.3,
    "penalty_float_low":            0.3,
    "penalty_vol_exhaustion":       0.2,
    "penalty_uma":                  1.0,
    "penalty_data_missing":         0.5,
}

# Final status values — ordered from most to least severe
STATUS_INSUFFICIENT   = "insufficient_data"
STATUS_EXCL_FUND      = "excluded_fundamental"
STATUS_EXCL_FLOAT     = "excluded_float_structure"
STATUS_EXCL_REG       = "excluded_regulatory"
STATUS_WATCH          = "watch_with_risk"
STATUS_ELIGIBLE       = "eligible"

# Statuses that should NOT appear in Telegram alerts (can be filtered)
EXCLUDED_STATUSES = {STATUS_EXCL_FUND, STATUS_EXCL_FLOAT, STATUS_EXCL_REG}


# ---------------------------------------------------------------------------
# UMA / special monitoring override loader
# ---------------------------------------------------------------------------

def load_risk_overrides(risk_dir: Path | str | None = None) -> dict[str, dict]:
    """Load UMA / special monitoring overrides from data/risk/uma_overrides.csv.

    Returns:
        {ticker: {"is_uma": bool, "is_special_monitoring": bool}}
        Empty dict if file not found or unparseable.
    """
    if risk_dir is None:
        risk_dir = Path("data/risk")
    path = Path(risk_dir) / "uma_overrides.csv"

    if not path.exists():
        return {}

    try:
        # Skip lines starting with '#'
        lines = [ln for ln in path.read_text().splitlines() if not ln.strip().startswith("#")]
        if not lines:
            return {}
        from io import StringIO
        df = pd.read_csv(StringIO("\n".join(lines)))
        result: dict[str, dict] = {}
        for _, row in df.iterrows():
            ticker = str(row.get("ticker", "")).strip()
            if not ticker or ticker.startswith("#"):
                continue
            uma = str(row.get("is_uma", "false")).strip().lower() in ("true", "1", "yes")
            special = str(row.get("is_special_monitoring", "false")).strip().lower() in ("true", "1", "yes")
            result[ticker] = {"is_uma": uma, "is_special_monitoring": special}
        logger.debug("UMA overrides loaded: %d tickers", len(result))
        return result
    except Exception as exc:
        logger.warning("Could not load uma_overrides.csv: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Core filter function
# ---------------------------------------------------------------------------

def evaluate_hard_filters(
    row: dict | pd.Series,
    config: dict | None = None,
) -> dict:
    """Apply hard exclusion filters to one ticker row.

    Args:
        row    : dict or pd.Series with fundamental + signal columns
        config : scanner_config dict (reads quality_filters sub-section)

    Returns:
        {
            "passes_hard_filters": bool,
            "exclusion_reason":    str | None,   # None if passes
            "_risk_flags_list":    list[str],    # internal; joined later
        }
    """
    cfg = _get_cfg(config)
    risk_flags: list[str] = []

    # ── 0. Missing fundamental data ───────────────────────────────────────
    fund_status = str(_get(row, "fundamental_status", "missing")).lower()
    if fund_status == "missing":
        # Don't hard-exclude — mark as insufficient_data but keep in output
        return {
            "passes_hard_filters":  False,
            "exclusion_reason":     STATUS_INSUFFICIENT,
            "_risk_flags_list":     ["fundamental_data_missing"],
        }

    exclusions: list[str] = []

    # ── 1. DER (stored as ratio ×, post-conversion from yfinance %) ──────
    der = _float(row, "der")
    der_max = cfg["der_max"]
    if der is not None:
        if der < 0:
            risk_flags.append(f"DER negatif ({der:.2f}×) — ekuitas negatif")
        elif der > der_max:
            exclusions.append(f"DER {der:.2f}× > {der_max}×")
        elif der > 2.0:
            risk_flags.append(f"DER tinggi ({der:.2f}×)")

    # ── 2. PBV ────────────────────────────────────────────────────────────
    pbv = _float(row, "pbv")
    pbv_max = cfg["pbv_max"]
    if pbv is not None:
        if pbv < 0:
            risk_flags.append(f"PBV negatif ({pbv:.2f}) — ekuitas negatif")
        elif pbv > pbv_max:
            exclusions.append(f"PBV {pbv:.2f}× > {pbv_max}×")

    # ── 3. EBITDA (optional hard exclude, default off) ────────────────────
    ebitda = _float(row, "ebitda")
    if ebitda is not None:
        if ebitda < 0:
            if cfg.get("ebitda_negative_exclude", False):
                exclusions.append(f"EBITDA negatif ({ebitda/1e9:.1f}B IDR)")
            else:
                risk_flags.append(f"EBITDA negatif ({ebitda/1e9:.1f}B IDR)")

    # ── 4. Public float structure (hard exclude only for very low float) ──
    float_pct = _float(row, "public_float_pct")
    float_hard_min = cfg["float_hard_min_pct"]
    float_watch_min = cfg["float_watch_min_pct"]
    float_warn_max = cfg["float_warn_max_pct"]
    if float_pct is not None:
        if float_pct < float_hard_min:
            exclusions.append(f"Float terlalu rendah {float_pct:.1f}% < {float_hard_min}% (bandar territory)")
        elif float_pct < float_watch_min:
            risk_flags.append(f"Float rendah {float_pct:.1f}% ({float_watch_min}% batas aman)")
        elif float_pct > float_warn_max:
            risk_flags.append(f"Float sangat tinggi {float_pct:.1f}% (dominasi publik)")

    # ── 5. UMA / Regulatory (dari is_uma / is_special_monitoring yang sudah di-merge) ──
    if _bool(row, "is_uma"):
        risk_flags.append("⚠️ UMA — Unusual Market Activity (IDX)")
    if _bool(row, "is_special_monitoring"):
        risk_flags.append("⚠️ Special Monitoring IDX")

    # ── 6. Volume exhaustion (risk flag only, not hard exclude) ──────────
    vol_ratio = _float(row, "vol_ratio_20d") or 0.0
    market_cap = _float(row, "market_cap") or 0.0
    vol_thresh = cfg["vol_spike_exhaustion_ratio"]
    cap_thresh = cfg["vol_spike_cap_threshold_idr"]
    if vol_ratio > vol_thresh and 0 < market_cap < cap_thresh:
        risk_flags.append(
            f"Vol spike ekstrem {vol_ratio:.1f}× (mktcap < Rp {cap_thresh/1e12:.0f}T, waspada distribusi)"
        )

    if exclusions:
        # Determine primary exclusion reason
        excl_reason = _primary_exclusion(exclusions, float_pct, float_hard_min)
        return {
            "passes_hard_filters":  False,
            "exclusion_reason":     excl_reason,
            "_risk_flags_list":     risk_flags + [f"[excluded] {e}" for e in exclusions],
        }

    return {
        "passes_hard_filters":  True,
        "exclusion_reason":     None,
        "_risk_flags_list":     risk_flags,
    }


def _primary_exclusion(exclusions: list[str], float_pct: float | None, float_hard_min: float) -> str:
    """Determine the single primary exclusion status."""
    has_float_excl = float_pct is not None and float_pct < float_hard_min
    has_der_pbv    = any("DER" in e or "PBV" in e or "EBITDA" in e for e in exclusions)

    if has_float_excl and not has_der_pbv:
        return STATUS_EXCL_FLOAT
    return STATUS_EXCL_FUND


# ---------------------------------------------------------------------------
# Quality score adjustment
# ---------------------------------------------------------------------------

def apply_quality_penalty(
    row: dict | pd.Series,
    config: dict | None = None,
    risk_flags_list: list[str] | None = None,
) -> dict:
    """Compute quality_adjusted_score = total_score - penalties.

    Penalties never exclude — they only reduce the score used for ranking.
    The more risk flags, the lower quality_adjusted_score.

    Args:
        row             : ticker row with signal + fundamental columns
        config          : scanner_config dict
        risk_flags_list : pre-computed risk flags from evaluate_hard_filters
                          (pass None to recompute from row)

    Returns:
        {
            "quality_adjusted_score": float,
            "quality_penalty_total":  float,
        }
    """
    cfg = _get_cfg(config)
    base_score = float(_get(row, "total_score", 0.0) or 0.0)
    penalty = 0.0

    # NOTE: risk_flags_list is accepted for API compatibility but every
    # penalty check below independently reads from `row` directly — this
    # function does not actually consult a pre-computed flags list. Found
    # during the 2026-06-22 lint cleanup; not changed here since it's a
    # behavior question (should pre-computed flags short-circuit these
    # checks?) for a deliberate follow-up, not a silent lint-pass fix.

    # EBITDA negative
    ebitda = _float(row, "ebitda")
    if ebitda is not None and ebitda < 0:
        penalty += cfg["penalty_ebitda_negative"]

    # DER elevated (2.0 – der_max)
    der = _float(row, "der")
    der_max = cfg["der_max"]
    if der is not None and 2.0 < der <= der_max:
        penalty += cfg["penalty_der_elevated"]

    # Public float low (watch range)
    float_pct = _float(row, "public_float_pct")
    float_watch_min = cfg["float_watch_min_pct"]
    float_hard_min = cfg["float_hard_min_pct"]
    if float_pct is not None and float_hard_min <= float_pct < float_watch_min:
        penalty += cfg["penalty_float_low"]

    # Volume exhaustion
    vol_ratio = _float(row, "vol_ratio_20d") or 0.0
    market_cap = _float(row, "market_cap") or 0.0
    if vol_ratio > cfg["vol_spike_exhaustion_ratio"] and 0 < market_cap < cfg["vol_spike_cap_threshold_idr"]:
        penalty += cfg["penalty_vol_exhaustion"]

    # UMA flag
    if _bool(row, "is_uma"):
        penalty += cfg["penalty_uma"]

    # Missing fundamental data
    fund_status = str(_get(row, "fundamental_status", "missing")).lower()
    if fund_status == "missing":
        penalty += cfg["penalty_data_missing"]

    quality_adjusted_score = round(max(0.0, min(10.0, base_score - penalty)), 2)
    return {
        "quality_adjusted_score": quality_adjusted_score,
        "quality_penalty_total":  round(penalty, 2),
    }


# ---------------------------------------------------------------------------
# Final status assignment
# ---------------------------------------------------------------------------

def assign_final_status(row: dict | pd.Series) -> str:
    """Assign final_status from filter results already on the row.

    Reads: passes_hard_filters, exclusion_reason, is_uma,
           is_special_monitoring, risk_flags (comma string).

    Returns one of the taxonomy strings defined at module top.
    """
    passes   = bool(_get(row, "passes_hard_filters", True))
    excl     = str(_get(row, "exclusion_reason", "") or "")
    is_uma   = _bool(row, "is_uma")
    is_spec  = _bool(row, "is_special_monitoring")
    flags_str = str(_get(row, "risk_flags", "") or "")

    # Regulatory always wins (manual override)
    if is_uma or is_spec:
        return STATUS_EXCL_REG

    if not passes:
        if excl in (STATUS_INSUFFICIENT, STATUS_EXCL_FUND,
                    STATUS_EXCL_FLOAT, STATUS_EXCL_REG):
            return excl
        return STATUS_EXCL_FUND

    # Passes hard filter but has risk flags
    if flags_str.strip():
        return STATUS_WATCH

    return STATUS_ELIGIBLE


# ---------------------------------------------------------------------------
# Batch enrichment (main entry point for run_daily_scan.py)
# ---------------------------------------------------------------------------

def enrich_df_with_quality_filters(
    df: pd.DataFrame,
    config: dict | None = None,
    risk_overrides: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Apply all quality filters to every row in the signals DataFrame.

    Adds columns:
        is_uma, is_special_monitoring  (from risk_overrides)
        passes_hard_filters            (bool)
        exclusion_reason               (str | None → stored as "" for CSV compat)
        risk_flags                     (comma-separated string)
        quality_adjusted_score         (float)
        quality_penalty_total          (float)
        final_status                   (str)

    Args:
        df              : signals DataFrame (after compute_signal + ML ranking)
        config          : scanner_config dict
        risk_overrides  : output of load_risk_overrides(); None → no overrides
    """
    if df.empty:
        return df

    df = df.copy()
    overrides = risk_overrides or {}

    # ── Inject UMA / special monitoring from manual overrides ─────────────
    df["is_uma"] = df["ticker"].map(
        lambda t: overrides.get(t, {}).get("is_uma", False)
    )
    df["is_special_monitoring"] = df["ticker"].map(
        lambda t: overrides.get(t, {}).get("is_special_monitoring", False)
    )

    # ── Evaluate filters row by row ───────────────────────────────────────
    passes_list:   list[bool]         = []
    excl_list:     list[str]          = []
    flags_list:    list[str]          = []
    qa_score_list: list[float]        = []
    qa_pen_list:   list[float]        = []
    status_list:   list[str]          = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        # Hard filters
        hf = evaluate_hard_filters(row_dict, config)
        passes   = hf["passes_hard_filters"]
        excl     = hf["exclusion_reason"] or ""
        rf_list  = hf["_risk_flags_list"]

        # Quality penalty (apply even if excluded — for score reference)
        qp = apply_quality_penalty(row_dict, config, rf_list)

        # Merge filter results back onto row_dict for assign_final_status
        row_dict["passes_hard_filters"] = passes
        row_dict["exclusion_reason"]    = excl
        row_dict["risk_flags"]          = ", ".join(rf_list)

        status = assign_final_status(row_dict)

        passes_list.append(passes)
        excl_list.append(excl)
        flags_list.append(", ".join(rf_list))
        qa_score_list.append(qp["quality_adjusted_score"])
        qa_pen_list.append(qp["quality_penalty_total"])
        status_list.append(status)

    df["passes_hard_filters"]    = passes_list
    df["exclusion_reason"]       = excl_list
    df["risk_flags"]             = flags_list
    df["quality_adjusted_score"] = qa_score_list
    df["quality_penalty_total"]  = qa_pen_list
    df["final_status"]           = status_list

    # ── Log summary ───────────────────────────────────────────────────────
    counts = df["final_status"].value_counts().to_dict()
    logger.info(
        "Quality filter summary: eligible=%d watch=%d excl_fund=%d excl_float=%d "
        "excl_reg=%d insufficient=%d",
        counts.get(STATUS_ELIGIBLE, 0),
        counts.get(STATUS_WATCH, 0),
        counts.get(STATUS_EXCL_FUND, 0),
        counts.get(STATUS_EXCL_FLOAT, 0),
        counts.get(STATUS_EXCL_REG, 0),
        counts.get(STATUS_INSUFFICIENT, 0),
    )

    # Log a few notable exclusions for auditability
    _excl_cols = ["ticker", "final_status", "exclusion_reason"] + [
        c for c in ["pbv", "der"] if c in df.columns
    ]
    excluded = df[df["final_status"].isin(EXCLUDED_STATUSES)][_excl_cols].head(10)
    if not excluded.empty:
        logger.info("Notable exclusions:\n%s", excluded.to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_cfg(config: dict | None) -> dict:
    """Extract quality_filters sub-section, merged with defaults."""
    base = dict(_DEFAULTS)
    if config:
        qf = config.get("quality_filters", {})
        base.update(qf)
    return base


def _get(row: dict | pd.Series, key: str, default: Any = None) -> Any:
    try:
        val = row[key] if hasattr(row, "__getitem__") else getattr(row, key, default)
        return default if (val is None or (isinstance(val, float) and math.isnan(val))) else val
    except (KeyError, AttributeError):
        return default


def _float(row: dict | pd.Series, key: str) -> float | None:
    val = _get(row, key)
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _bool(row: dict | pd.Series, key: str) -> bool:
    val = _get(row, key, False)
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")
