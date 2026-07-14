"""Structured I/O contracts for AI Lab — every LLM call in this package
produces one of these Pydantic models, never a free-form string. See
stock_scanner/ai_lab/client.py for how responses are validated against
these before anything downstream can touch them.

Architecture (explainability upgrade): scoring, confidence, recommendation
level, risk level, and expected return are ALL computed deterministically
in stock_scanner/ai_lab/scoring.py — never by the LLM. The LLM's role here
is narrowed to pure narrative: explaining, in prose, numbers it did not
produce and cannot change. This is what makes DecisionTrace/
ConfidenceBreakdown auditable — the same Evidence always yields the same
scores, independent of any LLM call.

Guardrail (mirrors stock_scanner/learning/hypothesis_agent.py): the LLM is
only ever asked to produce QUALITATIVE fields — narrative selected/
rephrased from code-generated candidate observations (see
scoring.generate_evidence_highlights), never invented from scratch.
`Evidence` is built entirely by code from committed, already-validated
data (knowledge_base pattern stats, feature snapshots, the production
scanner's own component scores) and handed TO the model as context; the
model cannot invent numbers that end up in `Evidence`, `DecisionTrace`, or
`ConfidenceBreakdown` fields.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RecommendationLevel(str, Enum):
    """Replaces the old BUY/WATCH/SELL/AVOID set. Always assigned by
    stock_scanner.ai_lab.scoring.classify_recommendation_level() — a
    rule-based threshold classifier over DecisionTrace/ConfidenceBreakdown,
    never an LLM judgment call, so the same evidence always yields the
    same tier."""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WATCH = "WATCH"
    AVOID = "AVOID"


class RecommendationStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"


class HistoricalComparisonVerdict(str, Enum):
    STRONGER = "stronger"
    WEAKER = "weaker"
    SIMILAR = "similar"
    NO_DATA = "no_data"


class Evidence(BaseModel):
    """Code-computed, never LLM-generated. Passed INTO prompts as context;
    also stored verbatim in ai_recommendations.reasoning so the dashboard's
    "statistical evidence" panel renders the same numbers the model saw —
    never numbers the model produced itself."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    technical_indicators: dict[str, float | bool | None] = Field(default_factory=dict)
    statistical_evidence: list[dict] = Field(
        default_factory=list,
        description="Representative stats from validated knowledge_base pattern clusters "
        "(win_rate, win_rate_shrunk, ci_lower, ci_upper, p_value_adjusted, n, n_success, ...) — "
        "restricted to EXACT slice_definition matches. See "
        "stock_scanner.learning.pattern_dedup.ClusteredPattern.representative.",
    )
    similar_patterns: list[str] = Field(
        default_factory=list, description="Human-readable slice descriptions of exactly-matching patterns."
    )
    best_pattern_similarity_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Best partial-match percentage across ALL knowledge_base patterns (not just "
        "exact matches) — fraction of a pattern's slice_definition keys that equal this ticker's "
        "current feature values, for the single closest pattern. Distinct from "
        "similar_patterns/statistical_evidence, which are restricted to 100%-exact matches only; "
        "this is a continuous 'how close to any known pattern' measure, feeding "
        "DecisionTrace.pattern_similarity_score.",
    )


class DecisionTrace(BaseModel):
    """Transparent component breakdown behind the single AI Score — see
    stock_scanner.ai_lab.scoring.compute_decision_trace(). All fields 0-100,
    code-computed, reproducible from the same Evidence + feature_row."""

    technical_score: float = Field(ge=0.0, le=100.0)
    statistical_score: float = Field(ge=0.0, le=100.0)
    pattern_similarity_score: float = Field(ge=0.0, le=100.0)
    risk_score: float = Field(ge=0.0, le=100.0)
    final_score: float = Field(ge=0.0, le=100.0)


class ConfidenceBreakdown(BaseModel):
    """Transparent component breakdown behind the single confidence value —
    see stock_scanner.ai_lab.scoring.compute_confidence_breakdown().
    `risk_adjustment` is always <= 0 (a penalty, never a bonus)."""

    technical: float = Field(ge=0.0, le=1.0)
    statistical: float = Field(ge=0.0, le=1.0)
    pattern_similarity: float = Field(ge=0.0, le=1.0)
    risk_adjustment: float = Field(le=0.0)
    final_confidence: float = Field(ge=0.0, le=1.0)


class HistoricalComparison(BaseModel):
    """Code-computed stats (pattern_description/sample_size/win_rate/
    ci_lower/ci_upper/verdict) + one LLM-written explanation sentence that
    must not contradict them — see
    stock_scanner.ai_lab.scoring.compute_historical_comparison() for the
    stats+verdict and prompts.py for the explanation-only LLM call."""

    pattern_description: str | None = None
    sample_size: int | None = None
    win_rate: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    verdict: HistoricalComparisonVerdict
    explanation: str = Field(default="", max_length=400)


class HypothesisOutput(BaseModel):
    """LLM output for stock_scanner.ai_lab.agents.hypothesis_agent — purely
    qualitative narrative over a given Evidence + candidate-highlights
    object (see scoring.generate_evidence_highlights). strengths/
    weaknesses/risks must be selected/rephrased from the supplied
    candidates only, never invented — enforced by prompt instruction and,
    if the model returns nothing, backfilled from the same candidates in
    decision_agent.assemble_recommendation rather than left empty."""

    why: str = Field(min_length=1, max_length=800)
    strengths: list[str] = Field(default_factory=list, max_length=5)
    weaknesses: list[str] = Field(default_factory=list, max_length=5)
    risks: list[str] = Field(default_factory=list, max_length=5)


class DecisionOutput(BaseModel):
    """LLM output for stock_scanner.ai_lab.agents.decision_agent — pure
    narrative over already-computed DecisionTrace/ConfidenceBreakdown/
    HistoricalComparison. No score, confidence, recommendation, expected
    return, or risk level here anymore — all of those are code-computed in
    scoring.py before this call ever happens, and this call cannot change
    them, only explain them."""

    reasoning_summary: str = Field(min_length=1, max_length=600)
    historical_comparison_explanation: str = Field(min_length=1, max_length=400)
    confidence_explanation: str = Field(min_length=1, max_length=400)


class AIRecommendation(BaseModel):
    """Full row shape for the ai_recommendations table / published JSON —
    the merge of code-computed Evidence + DecisionTrace + ConfidenceBreakdown
    + HistoricalComparison with the LLM's narrative fields into one
    storable, dashboard-renderable record.

    Backward compatible with the pre-upgrade shape: `score`/`confidence`/
    `recommendation`/`expected_return`/`risk_level` all keep their original
    field names and DB columns — `score` always equals
    `decision_trace.final_score` and `confidence` always equals
    `confidence_breakdown.final_confidence`, kept in sync by
    decision_agent.assemble_recommendation. Only `recommendation`'s enum
    values changed (RecommendationAction -> RecommendationLevel, per the
    "Recommendation Levels" upgrade) and `expected_return` may now be null.
    """

    id: str
    ticker: str
    ai_model: str
    score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    recommendation: RecommendationLevel
    reasoning: dict = Field(default_factory=dict)
    decision_trace: DecisionTrace
    confidence_breakdown: ConfidenceBreakdown
    historical_comparison: HistoricalComparison
    expected_return: float | None = None
    risk_level: RiskLevel | None = None
    generated_date: str
    status: RecommendationStatus = RecommendationStatus.PENDING
    entry_price: float | None = None
    exit_price: float | None = None
    return_percentage: float | None = None
    model: str
