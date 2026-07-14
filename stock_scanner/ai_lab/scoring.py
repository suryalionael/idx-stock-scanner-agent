"""Deterministic, code-computed scoring for AI Lab decisions.

Every number here is derived from evidence already computed/validated
elsewhere in the pipeline — the production scanner's own component scores
(stock_scanner.pipeline.signal_engine) where a persona has a direct
equivalent, its quality/risk flags (stock_scanner.pipeline.quality_filters)
for risk, and knowledge_base pattern statistics for statistical/historical
evidence — never invented from scratch and never LLM-generated. This is
what makes the AI Score, confidence, recommendation level, and risk level
fully auditable: the same inputs always produce the same outputs, and
every component traces back to a real, inspectable number.

Weights (technical 35%, statistical 35%, pattern_similarity 15%, risk 15%)
are a deliberate, documented choice — technical and statistical evidence
weighted equally as the two primary pillars, pattern similarity and risk
as secondary modifiers — not tuned/backtested. Revisit once AI Lab has
enough resolved recommendations to evaluate against.
"""
from __future__ import annotations

import pandas as pd

from stock_scanner.ai_lab.models import AIModelSpec
from stock_scanner.ai_lab.schemas import (
    ConfidenceBreakdown,
    DecisionTrace,
    Evidence,
    HistoricalComparison,
    HistoricalComparisonVerdict,
    RecommendationLevel,
    RiskLevel,
)

WEIGHTS = {"technical": 0.35, "statistical": 0.35, "pattern_similarity": 0.15, "risk": 0.15}


def _f(row: pd.Series, key: str, default: float = 0.0) -> float:
    val = row.get(key)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return float(val)


# ---------------------------------------------------------------------------
# Component scores (0-100 each)
# ---------------------------------------------------------------------------

def compute_technical_score(model_spec: AIModelSpec, feature_row: pd.Series, indicators: dict) -> float:
    """Reuses the production scanner's own 0-10 component scores
    (trend_score, momentum_score, breakout_score, volume_score — already
    computed and tuned in stock_scanner.pipeline.signal_engine) wherever a
    persona has a direct equivalent, rather than re-deriving parallel
    logic. reversal_ai has no production equivalent (the scanner doesn't
    score reversal setups), so it gets a dedicated formula from its own
    focus-feature indicators."""
    key = model_spec.key
    if key == "momentum_ai":
        return round((_f(feature_row, "trend_score") + _f(feature_row, "momentum_score")) / 2 * 10, 2)
    if key == "breakout_ai":
        return round(_f(feature_row, "breakout_score") * 10, 2)
    if key == "volume_ai":
        return round(_f(feature_row, "volume_score") * 10, 2)
    if key == "reversal_ai":
        return _reversal_technical_score(indicators)
    return 50.0  # unknown persona: neutral, not zero — absence of a formula isn't evidence of weakness


def _reversal_technical_score(indicators: dict) -> float:
    """No production equivalent exists for reversal setups, so this is a
    dedicated (documented, not reused) formula: oversold RSI/StochRSI and a
    turning-up MACD histogram are the classic reversal-entry tells."""
    parts = []
    rsi = indicators.get("rsi14")
    if rsi is not None:
        parts.append(max(0.0, min(100.0, 100.0 - float(rsi))))
    stoch_k = indicators.get("stoch_rsi_k")
    if stoch_k is not None:
        parts.append(max(0.0, min(100.0, 100.0 - float(stoch_k))))
    macd_hist = indicators.get("macd_histogram")
    if macd_hist is not None:
        parts.append(max(0.0, min(100.0, 50.0 + float(macd_hist) * 10)))
    return round(sum(parts) / len(parts), 2) if parts else 50.0


def compute_statistical_score(evidence: Evidence) -> float:
    """Mean win_rate_shrunk across exactly-matched validated patterns,
    scaled to 0-100. 0 (not a neutral 50) when no pattern matches at all —
    "no evidence" is not the same as "known to work," and defaulting to a
    midpoint would silently manufacture unearned confidence."""
    rates = [
        e["win_rate_shrunk"] for e in evidence.statistical_evidence
        if e.get("win_rate_shrunk") is not None
    ]
    if not rates:
        return 0.0
    return round(sum(rates) / len(rates) * 100, 2)


def compute_pattern_similarity_score(evidence: Evidence) -> float:
    return round(evidence.best_pattern_similarity_pct, 2)


def compute_risk_score(feature_row: pd.Series) -> float:
    """Grounded in the production pipeline's own quality/risk flags
    (stock_scanner.pipeline.quality_filters) — is_uma, is_special_monitoring,
    quality_penalty_total — plus ATR volatility as a secondary contributor,
    rather than an ad hoc new risk model."""
    risk = 0.0
    if bool(feature_row.get("is_uma")):
        risk += 40.0
    if bool(feature_row.get("is_special_monitoring")):
        risk += 30.0
    risk += min(30.0, _f(feature_row, "quality_penalty_total") * 3)
    atr_pct = feature_row.get("atr_pct")
    if atr_pct is not None and not (isinstance(atr_pct, float) and pd.isna(atr_pct)):
        risk += min(20.0, float(atr_pct) * 2)
    return round(min(100.0, risk), 2)


# ---------------------------------------------------------------------------
# Composite trace / confidence
# ---------------------------------------------------------------------------

def compute_decision_trace(model_spec: AIModelSpec, feature_row: pd.Series, evidence: Evidence) -> DecisionTrace:
    technical = compute_technical_score(model_spec, feature_row, evidence.technical_indicators)
    statistical = compute_statistical_score(evidence)
    pattern_similarity = compute_pattern_similarity_score(evidence)
    risk = compute_risk_score(feature_row)
    final = (
        technical * WEIGHTS["technical"]
        + statistical * WEIGHTS["statistical"]
        + pattern_similarity * WEIGHTS["pattern_similarity"]
        + (100.0 - risk) * WEIGHTS["risk"]
    )
    return DecisionTrace(
        technical_score=technical, statistical_score=statistical,
        pattern_similarity_score=pattern_similarity, risk_score=risk,
        final_score=round(final, 2),
    )


def compute_confidence_breakdown(trace: DecisionTrace) -> ConfidenceBreakdown:
    """Same component scores as DecisionTrace, rescaled to 0-1, with risk
    expressed as an explicit penalty (risk_adjustment <= 0) rather than an
    inverted bonus — makes the "risk always subtracts" rule visible in the
    stored number itself, not just in the final_score formula."""
    technical = trace.technical_score / 100.0
    statistical = trace.statistical_score / 100.0
    pattern_similarity = trace.pattern_similarity_score / 100.0
    risk_adjustment = -round(trace.risk_score / 100.0 * 0.3, 4)  # penalty capped at -0.3

    raw = (
        technical * WEIGHTS["technical"]
        + statistical * WEIGHTS["statistical"]
        + pattern_similarity * WEIGHTS["pattern_similarity"]
        + risk_adjustment
    )
    final_confidence = max(0.0, min(1.0, raw))
    return ConfidenceBreakdown(
        technical=round(technical, 4), statistical=round(statistical, 4),
        pattern_similarity=round(pattern_similarity, 4), risk_adjustment=risk_adjustment,
        final_confidence=round(final_confidence, 4),
    )


# ---------------------------------------------------------------------------
# Classifications (rule-based, not LLM-judged)
# ---------------------------------------------------------------------------

def classify_risk_level(trace: DecisionTrace) -> RiskLevel:
    if trace.risk_score >= 66:
        return RiskLevel.HIGH
    if trace.risk_score >= 33:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def classify_recommendation_level(trace: DecisionTrace, confidence: ConfidenceBreakdown) -> RecommendationLevel:
    """Rule-based threshold classifier over DecisionTrace/ConfidenceBreakdown
    — never an LLM judgment call, so the same evidence always yields the
    same tier (see RecommendationLevel's docstring)."""
    if trace.risk_score >= 80:
        return RecommendationLevel.AVOID
    if confidence.final_confidence >= 0.70 and trace.final_score >= 70 and trace.statistical_score >= 50:
        return RecommendationLevel.STRONG_BUY
    if confidence.final_confidence >= 0.50 and trace.final_score >= 50:
        return RecommendationLevel.BUY
    if trace.final_score < 30:
        return RecommendationLevel.AVOID
    return RecommendationLevel.WATCH


def compute_expected_return(evidence: Evidence) -> float | None:
    """Always None in the current pipeline: knowledge_base pattern stats
    carry win/loss RATE (win_rate, win_rate_shrunk) but not return
    MAGNITUDE, so any number produced here would be an estimate dressed up
    as data — exactly what this upgrade forbids ("never estimate historical
    performance... return null rather than guessing"). Kept as a field
    (not removed) so a future evidence source carrying real
    return-magnitude statistics can populate it without another schema
    change; unused parameter kept for that same forward-compatibility."""
    del evidence
    return None


def compute_historical_comparison(evidence: Evidence, trace: DecisionTrace) -> HistoricalComparison:
    """Stats + verdict are entirely code-computed; only `explanation` is
    filled in later by an LLM call constrained to these exact numbers (see
    prompts.build_decision_prompt) — never left to invent its own."""
    if not evidence.statistical_evidence:
        return HistoricalComparison(verdict=HistoricalComparisonVerdict.NO_DATA)

    best = max(evidence.statistical_evidence, key=lambda e: e.get("win_rate_shrunk") or 0.0)
    pattern_desc = evidence.similar_patterns[0] if evidence.similar_patterns else None

    if trace.technical_score >= 65 and trace.pattern_similarity_score >= 80:
        verdict = HistoricalComparisonVerdict.STRONGER
    elif trace.technical_score < 40:
        verdict = HistoricalComparisonVerdict.WEAKER
    else:
        verdict = HistoricalComparisonVerdict.SIMILAR

    return HistoricalComparison(
        pattern_description=pattern_desc,
        sample_size=best.get("n"),
        win_rate=best.get("win_rate"),
        ci_lower=best.get("ci_lower"),
        ci_upper=best.get("ci_upper"),
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Grounded strengths/weaknesses/risks candidates
# ---------------------------------------------------------------------------

def generate_evidence_highlights(feature_row: pd.Series, evidence: Evidence, trace: DecisionTrace) -> dict:
    """Rule-based candidate observations, grounded ONLY in supplied
    evidence — passed to the LLM as the pool it must select/rephrase from
    when writing strengths/weaknesses/risks (never invent from scratch),
    and used as a deterministic fallback if the LLM returns an empty list
    for any of the three (see decision_agent.assemble_recommendation)."""
    strengths: list[str] = []
    weaknesses: list[str] = []
    risks: list[str] = []
    ind = evidence.technical_indicators

    if ind.get("ma_full_alignment"):
        strengths.append("Full moving-average alignment confirms an established uptrend.")
    elif "ma_full_alignment" in ind:
        weaknesses.append("Moving averages are not fully aligned — trend structure is incomplete.")

    adx = ind.get("adx")
    if adx is not None:
        if adx >= 40:
            strengths.append(f"Strong ADX ({adx:.1f}) confirms high trend strength.")
        elif adx < 20:
            weaknesses.append(f"Weak ADX ({adx:.1f}) suggests the trend lacks conviction.")

    if ind.get("golden_cross") is False:
        weaknesses.append("No golden cross confirmation.")
    elif ind.get("golden_cross") is True:
        strengths.append("Golden cross confirmed.")

    roc5, roc20 = ind.get("roc5"), ind.get("roc20")
    if roc5 is not None and roc20 is not None:
        if roc5 > 0 and roc20 > 0:
            strengths.append("Positive momentum across both short- and medium-term lookback windows.")
        elif roc5 < 0 and roc20 < 0:
            weaknesses.append("Negative momentum across both short- and medium-term lookback windows.")

    if ind.get("atr_breakout"):
        strengths.append("ATR breakout confirms a genuine volatility expansion.")
    if ind.get("squeeze_release"):
        strengths.append("Bollinger squeeze release signals the start of a breakout move.")
    elif ind.get("squeeze_on"):
        weaknesses.append("Still in a volatility squeeze — breakout not yet confirmed.")

    if ind.get("vol_spike"):
        strengths.append("Volume spike confirms conviction behind the move.")
    else:
        vr = ind.get("vol_ratio_20d")
        if vr is not None and vr < 1.0:
            weaknesses.append(f"Below-average volume ({vr:.2f}x) — move lacks participation.")

    rsi = ind.get("rsi14")
    if rsi is not None:
        if rsi < 30:
            strengths.append(f"RSI ({rsi:.1f}) is in oversold territory — reversal setup.")
        elif rsi > 70:
            risks.append(f"RSI ({rsi:.1f}) is overbought — pullback risk.")

    if evidence.statistical_evidence:
        best = max(evidence.statistical_evidence, key=lambda e: e.get("win_rate_shrunk") or 0.0)
        wr = best.get("win_rate")
        if wr is not None:
            if wr < 0.3:
                weaknesses.append(f"Historical win rate only {wr:.0%} for this validated pattern.")
                risks.append("Pattern has weak historical follow-through.")
            elif wr >= 0.5:
                strengths.append(f"Historical win rate of {wr:.0%} supports this setup.")
    else:
        weaknesses.append("No statistically validated historical pattern matches this setup.")
        risks.append("No historical track record to gauge follow-through.")

    if bool(feature_row.get("is_uma")):
        risks.append("Ticker is flagged under Unusual Market Activity (UMA) monitoring.")
    if bool(feature_row.get("is_special_monitoring")):
        risks.append("Ticker is under special exchange monitoring.")
    if trace.risk_score >= 66:
        risks.append("High composite risk score — elevated volatility/quality-flag exposure.")

    return {"strengths": strengths[:5], "weaknesses": weaknesses[:5], "risks": risks[:5]}
