"""Statistics-only pattern mining over historical signals/outcomes.

Phase 1 of the Learning Agent (see docs/LEARNING_AGENT_ARCHITECTURE.md):
answers one question — "does the historical database contain enough
statistical signal to justify an LLM hypothesis stage?" — using nothing but
Fisher's exact test, Wilson confidence intervals, and Benjamini-Hochberg FDR
correction. No LLM, no hypothesis text, no database writes. Read-only,
research-only: nothing here is imported by stock_scanner/pipeline/, and
nothing here can influence signal_engine.py, model_registry, or
scanner_config.yaml.

Reuses scripts/train_challenger.py::load_training_examples() unchanged for
the base population (signals ⋈ feature_snapshots ⋈ outcomes ⋈
market_context, evaluated rows only) — that function has three other
consumers already and stays exactly as they need it; this module adds its
own sector_reference join on top rather than modifying a shared dependency.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import fisher_exact

# ---------------------------------------------------------------------------
# Dimensions searched — deliberately curated, not FEATURE_COLS wholesale.
# Derived/normalized columns are kept; their redundant raw inputs are
# skipped (e.g. price_vs_ma200 kept, ma20/ma50/ma200 skipped) — including
# both would inflate the test count (worse FDR burden) without adding
# independent information.
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "rsi14", "macd_histogram", "roc5", "roc20", "vol_ratio_20d", "atr_pct",
    "bb_width", "hist_vol_20d", "pct_from_52w_high", "price_vs_ma200",
    "slope_ma20", "stoch_rsi_k", "adx", "price_vs_vwap",
]
BOOLEAN_FEATURES = [
    "ma_full_alignment", "ma_partial_alignment", "golden_cross", "atr_breakout",
    "vol_spike", "obv_trend", "supertrend_bullish", "squeeze_on", "squeeze_release",
]
CATEGORICAL_FEATURES = ["sector", "regime", "strategy", "signal_label"]

_MIN_SLICE_N = 5          # skip trivially tiny slices before spending a stats call on them
_TIME_SPLIT_MIN_HALF_N_SUCCESS = 3  # below this in either half, direction check is inconclusive
_TICKER_CONCENTRATION_FLAG_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Stats primitives — hand-implemented (no new dependency beyond the scipy
# already used elsewhere in this repo's ML stack).
# ---------------------------------------------------------------------------

def wilson_ci(n_success: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval — appropriate for the small-n, low-base-rate
    counts this database actually has (a normal-approximation interval can
    go negative or exceed 1 at these sample sizes)."""
    if n == 0:
        return (0.0, 0.0)
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    phat = n_success / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * ((phat * (1 - phat) + z**2 / (4 * n)) / n) ** 0.5) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Standard BH step-up adjusted p-values (q-values), same order as
    input. Applied separately per interaction-order tier by mine_patterns —
    never pooled across tiers, so the triple-order tier's much larger test
    count can't drown out real single-feature signal."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    q = [0.0] * n
    running_min = 1.0
    for rank in range(n, 0, -1):
        idx = order[rank - 1]
        raw_q = p_values[idx] * n / rank
        running_min = min(running_min, raw_q)
        q[idx] = min(running_min, 1.0)
    return q


def shrunk_win_rate(n_success: int, n: int, baseline_rate: float, prior_strength: float = 20.0) -> float:
    """Beta-Binomial posterior mean under a weak prior centered on the
    global baseline — prevents a 2/2 slice from displaying as a misleading
    '100% win rate' in the report."""
    a0 = baseline_rate * prior_strength
    b0 = (1 - baseline_rate) * prior_strength
    return (a0 + n_success) / (a0 + b0 + n)


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Boolean feature columns round-trip through JSON/CSV as literal
    "True"/"False" strings — same gotcha stock_scanner.pipeline.
    challenger_score.compute_rule_score() already handles."""
    return series.map({True: True, False: False, "True": True, "False": False})


# ---------------------------------------------------------------------------
# Candidate pattern
# ---------------------------------------------------------------------------

@dataclass
class PatternCandidate:
    dimensions: tuple[str, ...]
    slice_definition: dict
    interaction_order: int
    n: int
    n_success: int
    win_rate: float
    win_rate_shrunk: float
    baseline_win_rate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    p_value_adjusted: float | None = None
    ticker_concentration: float | None = None
    ticker_concentration_flag: bool = False
    time_split_stable: bool | None = None   # None = inconclusive (too few positives per half)
    passed_gate: bool = False
    signal_ids: frozenset = field(default_factory=frozenset)  # positive-example ids behind n_success — used by pattern_dedup.py's Jaccard/overlap clustering, never read by hypothesis_agent.py's prompt builder

    def to_dict(self) -> dict:
        return {
            "dimensions": list(self.dimensions),
            "slice_definition": self.slice_definition,
            "interaction_order": self.interaction_order,
            "n": self.n,
            "n_success": self.n_success,
            "win_rate": round(self.win_rate, 4),
            "win_rate_shrunk": round(self.win_rate_shrunk, 4),
            "baseline_win_rate": round(self.baseline_win_rate, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "p_value": self.p_value,
            "p_value_adjusted": self.p_value_adjusted,
            "ticker_concentration": self.ticker_concentration,
            "ticker_concentration_flag": self.ticker_concentration_flag,
            "time_split_stable": self.time_split_stable,
            "passed_gate": self.passed_gate,
            "signal_ids": sorted(self.signal_ids),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PatternCandidate":
        return cls(
            dimensions=tuple(d["dimensions"]), slice_definition=d["slice_definition"],
            interaction_order=d["interaction_order"], n=d["n"], n_success=d["n_success"],
            win_rate=d["win_rate"], win_rate_shrunk=d["win_rate_shrunk"],
            baseline_win_rate=d["baseline_win_rate"], ci_lower=d["ci_lower"], ci_upper=d["ci_upper"],
            p_value=d["p_value"], p_value_adjusted=d.get("p_value_adjusted"),
            ticker_concentration=d.get("ticker_concentration"),
            ticker_concentration_flag=d.get("ticker_concentration_flag", False),
            time_split_stable=d.get("time_split_stable"), passed_gate=d.get("passed_gate", False),
            signal_ids=frozenset(d.get("signal_ids", [])),
        )


@dataclass
class MiningResult:
    total_n: int
    total_success: int
    baseline_win_rate: float
    candidates_by_order: dict[int, list[PatternCandidate]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dimension preparation
# ---------------------------------------------------------------------------

def _prepare_dimensions(df: pd.DataFrame, sector_df: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """Return (bucketed_df, dimension_names). bucketed_df has one column per
    searchable dimension: numeric features replaced by quintile-bucket
    labels, boolean features coerced to real bool, categoricals as-is."""
    out = df.copy()
    dims: list[str] = []

    for col in NUMERIC_FEATURES:
        if col not in out.columns:
            continue
        try:
            bucketed = pd.qcut(out[col], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
        except (ValueError, IndexError) as e:
            logger.debug(f"pattern_miner: skipping numeric dimension {col} (qcut failed: {e})")
            continue
        if bucketed.nunique(dropna=True) < 2:
            logger.debug(f"pattern_miner: skipping numeric dimension {col} (< 2 distinct buckets)")
            continue
        out[col] = bucketed.astype(object)
        dims.append(col)

    for col in BOOLEAN_FEATURES:
        if col not in out.columns:
            continue
        out[col] = _coerce_bool(out[col])
        dims.append(col)

    if "ihsg_pct_change_eval" in out.columns:
        out["regime"] = np.where(
            out["ihsg_pct_change_eval"].isna(), np.nan,
            np.where(out["ihsg_pct_change_eval"] > 0, "up", "down"),
        )
        dims.append("regime")

    for col in ("strategy", "signal_label"):
        if col in out.columns:
            dims.append(col)

    if sector_df is not None and not sector_df.empty and "ticker" in out.columns:
        out = out.merge(sector_df[["ticker", "sector"]], on="ticker", how="left")
        dims.append("sector")

    return out, dims


# ---------------------------------------------------------------------------
# Slice enumeration + scoring
# ---------------------------------------------------------------------------

def _enumerate_slices(bucketed: pd.DataFrame, combo: tuple[str, ...]):
    grouped = bucketed.groupby(list(combo), dropna=True, observed=True)
    for key, grp in grouped:
        if len(grp) < _MIN_SLICE_N:
            continue
        key_tuple = key if isinstance(key, tuple) else (key,)
        yield dict(zip(combo, key_tuple)), grp


def _time_split_stability(grp: pd.DataFrame, baseline_rate: float) -> bool | None:
    if "signal_date" not in grp.columns or grp["signal_date"].isna().all():
        return None
    median_date = grp["signal_date"].median()
    first_half = grp[grp["signal_date"] <= median_date]
    second_half = grp[grp["signal_date"] > median_date]
    if len(first_half) == 0 or len(second_half) == 0:
        return None
    s1, s2 = first_half["label_success"].sum(), second_half["label_success"].sum()
    if s1 < _TIME_SPLIT_MIN_HALF_N_SUCCESS or s2 < _TIME_SPLIT_MIN_HALF_N_SUCCESS:
        return None
    wr1 = s1 / len(first_half)
    wr2 = s2 / len(second_half)
    return (wr1 > baseline_rate) and (wr2 > baseline_rate)


def _score_slice(
    slice_def: dict, grp: pd.DataFrame, combo: tuple[str, ...], order: int,
    baseline_rate: float, total_n: int, total_success: int,
) -> PatternCandidate:
    n = len(grp)
    n_success = int(grp["label_success"].sum())
    win_rate = n_success / n
    ci_lower, ci_upper = wilson_ci(n_success, n)
    win_rate_shrunk = shrunk_win_rate(n_success, n, baseline_rate)

    rest_n = total_n - n
    rest_success = total_success - n_success
    table = [[n_success, n - n_success], [rest_success, max(rest_n - rest_success, 0)]]
    _, p_value = fisher_exact(table, alternative="two-sided")

    pos = grp[grp["label_success"] == 1] if n_success > 0 else grp.iloc[0:0]

    ticker_concentration = None
    ticker_flag = False
    if len(pos) and "ticker" in grp.columns:
        top_share = pos["ticker"].value_counts(normalize=True).iloc[0]
        ticker_concentration = float(top_share)
        ticker_flag = top_share > _TICKER_CONCENTRATION_FLAG_THRESHOLD

    signal_ids = frozenset(pos["signal_id"]) if "signal_id" in grp.columns else frozenset()

    time_stable = _time_split_stability(grp, baseline_rate)

    return PatternCandidate(
        dimensions=combo, slice_definition=slice_def, interaction_order=order,
        n=n, n_success=n_success, win_rate=win_rate, win_rate_shrunk=win_rate_shrunk,
        baseline_win_rate=baseline_rate, ci_lower=ci_lower, ci_upper=ci_upper,
        p_value=float(p_value), ticker_concentration=ticker_concentration,
        ticker_concentration_flag=ticker_flag, time_split_stable=time_stable,
        signal_ids=signal_ids,
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def mine_patterns(
    df: pd.DataFrame,
    sector_df: pd.DataFrame | None = None,
    min_n_success: int = 8,
    alpha: float = 0.05,
    max_order: int = 3,
) -> MiningResult:
    """Search single/pairwise/triple feature-dimension slices for a win-rate
    lift over baseline, gated by: n_success >= min_n_success, Wilson-CI
    lower bound > baseline, and Benjamini-Hochberg-adjusted p < alpha
    (correction applied separately within each interaction-order tier).

    Order-3 (triple) results are always computed but are exploratory — at
    this database's current volume, a 5x5x5 quintile cross averages a
    handful of positives per cell, so callers should treat order-3 findings
    as a lead to investigate, not a validated pattern (see
    docs/LEARNING_AGENT_ARCHITECTURE.md).
    """
    if "label_success" not in df.columns or df.empty:
        return MiningResult(total_n=0, total_success=0, baseline_win_rate=0.0, candidates_by_order={1: [], 2: [], 3: []})

    total_n = len(df)
    total_success = int(df["label_success"].sum())
    baseline_rate = total_success / total_n if total_n else 0.0

    bucketed, dims = _prepare_dimensions(df, sector_df)
    logger.info(f"pattern_miner: {len(dims)} dimensions, {total_n} rows, {total_success} positives (baseline={baseline_rate:.4f})")

    candidates_by_order: dict[int, list[PatternCandidate]] = {}
    for order in range(1, max_order + 1):
        order_candidates: list[PatternCandidate] = []
        combos = list(itertools.combinations(dims, order))
        for combo_i, combo in enumerate(combos):
            if order >= 3 and combo_i and combo_i % 500 == 0:
                logger.info(f"pattern_miner: order={order} combo {combo_i}/{len(combos)}...")
            for slice_def, grp in _enumerate_slices(bucketed, combo):
                order_candidates.append(
                    _score_slice(slice_def, grp, combo, order, baseline_rate, total_n, total_success)
                )
        q_values = benjamini_hochberg([c.p_value for c in order_candidates])
        for c, q in zip(order_candidates, q_values):
            c.p_value_adjusted = q
            c.passed_gate = (
                c.n_success >= min_n_success
                and q < alpha
                and c.ci_lower > c.baseline_win_rate
            )
        candidates_by_order[order] = order_candidates
        logger.info(
            f"pattern_miner: order={order} tested={len(order_candidates)} "
            f"passed_gate={sum(c.passed_gate for c in order_candidates)}"
        )

    return MiningResult(
        total_n=total_n, total_success=total_success, baseline_win_rate=baseline_rate,
        candidates_by_order=candidates_by_order,
    )
