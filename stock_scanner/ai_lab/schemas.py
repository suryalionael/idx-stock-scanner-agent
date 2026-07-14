"""Structured I/O contracts for AI Lab — every LLM call in this package
produces one of these Pydantic models, never a free-form string. See
stock_scanner/ai_lab/client.py for how responses are validated against
these before anything downstream can touch them.

Guardrail (mirrors stock_scanner/learning/hypothesis_agent.py): the LLM is
only ever asked to produce QUALITATIVE fields here — narrative, strengths,
weaknesses, risk labels, a bounded confidence/score. It is never the source
of a statistic. `Evidence` below is built entirely by code from committed,
already-validated data (knowledge_base pattern stats, feature snapshots)
and handed TO the model as context; the model cannot invent numbers that
end up in `Evidence` fields.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RecommendationAction(str, Enum):
    BUY = "BUY"
    WATCH = "WATCH"
    SELL = "SELL"
    AVOID = "AVOID"


class RecommendationStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"


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
        "(win_rate, ci_lower, p_value_adjusted, n, n_success, ...) — see "
        "stock_scanner.learning.pattern_dedup.ClusteredPattern.representative.",
    )
    similar_patterns: list[str] = Field(
        default_factory=list, description="Human-readable slice descriptions of matching patterns."
    )


class HypothesisOutput(BaseModel):
    """LLM output for stock_scanner.ai_lab.agents.hypothesis_agent — purely
    qualitative narrative over a given Evidence object. No numeric field
    here is trusted as a fact; `confidence` is the model's own qualitative
    self-assessment, explicitly labeled as such downstream."""

    why: str = Field(min_length=1, max_length=800)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_explanation: str = Field(min_length=1, max_length=400)
    strengths: list[str] = Field(default_factory=list, max_length=10)
    weaknesses: list[str] = Field(default_factory=list, max_length=10)
    risks: list[str] = Field(default_factory=list, max_length=10)


class DecisionOutput(BaseModel):
    """LLM output for stock_scanner.ai_lab.agents.decision_agent — the
    final ranked call. `score`/`confidence`/`expected_return`/`risk_level`
    are the model's judgment, always presented as experimental/AI-generated
    (never silently treated as validated statistics)."""

    score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    recommendation: RecommendationAction
    expected_return: float = Field(description="Expected forward return, as a fraction, e.g. 0.08 for 8%.")
    risk_level: RiskLevel
    reasoning_summary: str = Field(min_length=1, max_length=500)


class AIRecommendation(BaseModel):
    """Full row shape for the ai_recommendations table / published JSON —
    the merge of code-computed Evidence + HypothesisOutput + DecisionOutput
    into one storable, dashboard-renderable record."""

    id: str
    ticker: str
    ai_model: str
    score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    recommendation: RecommendationAction
    reasoning: dict = Field(default_factory=dict)
    expected_return: float | None = None
    risk_level: RiskLevel | None = None
    generated_date: str
    status: RecommendationStatus = RecommendationStatus.PENDING
    entry_price: float | None = None
    exit_price: float | None = None
    return_percentage: float | None = None
    model: str
