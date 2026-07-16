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


class TradeOutcome(str, Enum):
    """Win/loss classification of a resolved recommendation's
    return_percentage — a deliberately separate axis from
    RecommendationStatus (see stock_scanner.ai_lab.resolver), set only once
    a recommendation reaches CLOSED/EXPIRED. Kept distinct from status so an
    EXPIRED mark-to-market row that ended up profitable is still WIN, not
    inferred as a loss just because it never hit its target."""

    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"


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
    highest_price: float | None = None
    lowest_price: float | None = None
    max_runup_pct: float | None = None
    max_drawdown_pct: float | None = None
    holding_days: int | None = None
    trade_outcome: TradeOutcome | None = None
    model: str


class ObservationCategory(str, Enum):
    """Which scored tier a ReflectionObservation came from — see
    stock_scanner.ai_lab.reflection_engine. MODEL_PERFORMANCE/
    SECTOR_PERFORMANCE/RECOMMENDATION_LEVEL_PERFORMANCE/
    HISTORICAL_VERDICT_ACCURACY share one categorical-dimension scorer and
    one Benjamini-Hochberg pool; TECHNICAL_PATTERN and
    CONFIDENCE_CALIBRATION are each their own pool — pooling separately
    mirrors stock_scanner.learning.pattern_miner's per-order-tier BH
    separation, so a noisy tier can't drown a real signal in another."""

    MODEL_PERFORMANCE = "model_performance"
    SECTOR_PERFORMANCE = "sector_performance"
    RECOMMENDATION_LEVEL_PERFORMANCE = "recommendation_level_performance"
    HISTORICAL_VERDICT_ACCURACY = "historical_verdict_accuracy"
    TECHNICAL_PATTERN = "technical_pattern"
    CONFIDENCE_CALIBRATION = "confidence_calibration"


class ReflectionObservation(BaseModel):
    """One statistically gated finding over RESOLVED ai_recommendations —
    see stock_scanner.ai_lab.reflection_engine.generate_observations().
    Every field except `llm_note` is code-computed (Wilson CI / Fisher's
    exact / Benjamini-Hochberg / one-sample binomial test, reusing
    stock_scanner.learning.pattern_miner's primitives) — never LLM-invented.
    `confidence` here is 1 - p_value_adjusted (clamped [0,1]): a
    statistical "how likely this is real, not noise" measure, distinct
    from an individual recommendation's own AIRecommendation.confidence.
    Not frozen: `llm_note` is attached after construction via
    `.model_copy(update={"llm_note": ...})` once
    stock_scanner.ai_lab.agents.reflection_agent's narrative call
    (best-effort, may fail) returns."""

    observation_id: str
    category: ObservationCategory
    title: str
    description: str
    supporting_statistics: dict = Field(default_factory=dict)
    affected_trade_count: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    generated_at: str
    llm_note: str | None = None


class ObservationNote(BaseModel):
    """One LLM-written plain-English gloss for a single, already-existing
    ReflectionObservation — the LLM may only reference observation_ids it
    was given, never invent new ones (enforced by prompt instruction, see
    prompts.REFLECTION_SYSTEM_PROMPT)."""

    observation_id: str
    note: str = Field(min_length=1, max_length=300)


class ReflectionNarrativeOutput(BaseModel):
    """LLM output for stock_scanner.ai_lab.agents.reflection_agent — pure
    narrative over an already-gated, already-scored list of
    ReflectionObservation objects. No numeric field anywhere: the LLM may
    only summarize, explain, and prioritize/order the given
    observation_ids, never compute or restate a statistic itself (mirrors
    HypothesisOutput/DecisionOutput's all-string/list-of-string
    discipline)."""

    overall_summary: str = Field(min_length=1, max_length=800)
    prioritized_observation_ids: list[str] = Field(default_factory=list, max_length=20)
    observation_notes: list[ObservationNote] = Field(default_factory=list, max_length=20)


class HypothesisStatus(str, Enum):
    """Validation outcome, not a review lifecycle — a Hypothesis is either
    a statistically supported finding or a rejected candidate, decided
    once by stock_scanner.ai_lab.statistical_validation.validate_hypotheses()
    and never changed afterward (append-only, see Hypothesis's docstring)."""

    VALIDATED = "validated"
    REJECTED = "rejected"


class EvidenceStrength(str, Enum):
    """Qualitative tier over an already-VALIDATED hypothesis's
    bh_adjusted_p — STRONG if < 0.01, else MODERATE (both already cleared
    the alpha=0.05 gate to be VALIDATED at all). None for REJECTED
    hypotheses — there's no "strength" to a finding that didn't hold up."""

    STRONG = "STRONG"
    MODERATE = "MODERATE"


class HypothesisCandidate(BaseModel):
    """Pure structural output of stock_scanner.ai_lab.hypothesis_engine —
    a condition combination (order <= 3) plus raw counts only. No p-values,
    no CI, no id: those belong to statistical_validation.py, which is the
    only place Wilson CI / Fisher's exact / Benjamini-Hochberg get called
    (see hypothesis_engine.py's module docstring for why this split is
    deliberate, not accidental)."""

    conditions: list[list[str]]  # sorted [[dimension, value], ...] pairs — deterministic ordering
    order: int = Field(ge=2, le=3)
    n: int = Field(ge=0)
    n_success: int = Field(ge=0)
    n_failure: int = Field(ge=0)
    win_rate: float = Field(ge=0.0, le=1.0)
    avg_return_percentage: float | None = None
    avg_holding_days: float | None = None
    source_reflection_ids: list[str] = Field(default_factory=list)


class Hypothesis(BaseModel):
    """One row of validated_hypotheses — a HypothesisCandidate scored by
    statistical_validation.py. Append-only, same reasoning as
    ReflectionObservation: hypothesis_id hashes in created_at, so the same
    condition-set re-validated on a larger resolved-trade population later
    is a new, worth-keeping data point rather than an overwrite (see
    docs/AI_LAB_ARCHITECTURE.md "Hypothesis Generator + Statistical
    Validation" for why this differs from knowledge_base.py's
    content-only hash). Not frozen: `llm_note` is attached after
    construction, same pattern as ReflectionObservation."""

    hypothesis_id: str
    created_at: str
    description: str
    conditions: list[list[str]]
    sample_size: int = Field(ge=0)
    successes: int = Field(ge=0)
    failures: int = Field(ge=0)
    win_rate: float = Field(ge=0.0, le=1.0)
    shrunk_win_rate: float = Field(ge=0.0, le=1.0)
    wilson_lower: float = Field(ge=0.0, le=1.0)
    wilson_upper: float = Field(ge=0.0, le=1.0)
    fisher_p: float = Field(ge=0.0, le=1.0)
    bh_adjusted_p: float = Field(ge=0.0, le=1.0)
    evidence_strength: EvidenceStrength | None = None
    status: HypothesisStatus
    rejection_reason: str | None = None
    failed_gate: str | None = None
    source_reflection_ids: list[str] = Field(default_factory=list)
    metadata_json: dict = Field(default_factory=dict)
    llm_note: str | None = None


class HypothesisNote(BaseModel):
    """One LLM-written plain-English gloss for a single, already-existing
    Hypothesis — same contract as ObservationNote: the LLM may only
    reference hypothesis_ids it was given, never invent new ones."""

    hypothesis_id: str
    note: str = Field(min_length=1, max_length=300)


class HypothesisCluster(BaseModel):
    """One LLM-proposed grouping of hypothesis_ids that describe the same
    underlying finding from different angles — narrative-only clustering
    (a short label), not a statistical claim; contrast with
    stock_scanner.learning.pattern_dedup's Jaccard/overlap clustering,
    which is a numeric similarity measure over positive-example sets and
    is deliberately NOT reused here (see docs/AI_LAB_ARCHITECTURE.md's
    scope note on why "duplicate prevention" stays structural, not
    fuzzy-semantic, for this component)."""

    label: str = Field(min_length=1, max_length=120)
    hypothesis_ids: list[str] = Field(default_factory=list, max_length=20)


class HypothesisNarrativeOutput(BaseModel):
    """LLM output for
    stock_scanner.ai_lab.agents.hypothesis_review_agent — pure narrative
    over an already-validated/rejected list of Hypothesis objects. No
    numeric field anywhere: summarize, explain, prioritize, and cluster
    only (mirrors ReflectionNarrativeOutput's discipline, extended with
    the one new responsibility this spec adds — clustering)."""

    overall_summary: str = Field(min_length=1, max_length=800)
    prioritized_hypothesis_ids: list[str] = Field(default_factory=list, max_length=20)
    hypothesis_notes: list[HypothesisNote] = Field(default_factory=list, max_length=20)
    clusters: list[HypothesisCluster] = Field(default_factory=list, max_length=10)


class KnowledgeLifecycleStatus(str, Enum):
    """Deterministic curation status over accumulated evidence for one
    normalized condition-set — see
    stock_scanner.ai_lab.knowledge_base_engine.generate_knowledge_entries()
    for the exact integer-threshold ladder. EMERGING is only reachable at
    confirmation_count == 1 (a real, distinct state, not a dead one)."""

    EMERGING = "emerging"
    CONFIRMED = "confirmed"
    STRONG = "strong"
    WEAKENING = "weakening"
    CONTRADICTED = "contradicted"
    ARCHIVED = "archived"


class KnowledgePromotionStatus(str, Enum):
    """Deployment axis — orthogonal to KnowledgeLifecycleStatus. Answers
    "has a human approved this for production?", never "how statistically
    mature is this?" (that's lifecycle_status, unchanged). Set to CANDIDATE
    by knowledge_base_engine.py at creation time ONLY — never advanced
    automatically by any statistical process; only a future human-run
    promotion tool (scripts/promote_knowledge.py, not built yet) may set
    PROMOTED/REJECTED/ARCHIVED. stock_scanner.pipeline.knowledge_application
    treats anything other than an exact PROMOTED match as not promoted —
    including this field being absent entirely on rows written before this
    field existed."""

    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ARCHIVED = "archived"


def is_entry_promoted(entry: dict) -> bool:
    """The single, reusable, fail-closed promotion check over a raw
    knowledge-entry dict — as loaded from knowledge_report.json or a
    pandas row converted to a dict. Exact string match against
    KnowledgePromotionStatus.PROMOTED.value only.

    A dict (not a KnowledgeEntry instance) is the deliberate input shape
    here: stock_scanner.pipeline.knowledge_application reads raw JSON
    without ever constructing/validating a KnowledgeEntry — Pydantic
    validation would raise on a partial or foreign-shaped row, which is
    exactly the "never raises" contract that module depends on. This
    function reproduces that same leniency for one field only: missing,
    None, or any value that isn't exactly the PROMOTED string (including
    other valid statuses, or an unrecognized/corrupt one) all return
    False. Never assume promotion.

    For code that already holds a validated KnowledgeEntry instance, use
    KnowledgeEntry.is_promoted() instead — same rule, no string parsing
    needed since the field is already a real enum member there."""
    return entry.get("promotion_status") == KnowledgePromotionStatus.PROMOTED.value


class KnowledgeEntry(BaseModel):
    """One curated belief — an accumulation of validated_hypotheses rows
    across ALL historical runs that share the exact same normalized
    condition-set (see knowledge_base_engine.py's module docstring for why
    exact equality is the full "deterministic similarity" mechanism for
    this system). Append-only, same reasoning as ReflectionObservation/
    Hypothesis: knowledge_id hashes in created_at (THIS curation run's
    timestamp, distinct from first_seen/last_confirmed which track the
    underlying evidence trail, not the engine's own run cadence).

    cumulative_sample_size/successes/failures, confidence_interval,
    shrunk_win_rate, and evidence_strength all come from the latest
    CONFIRMING hypothesis row only — never summed across runs, since each
    run re-scores against the same growing population and summing would
    double-count the same underlying trades ("never strengthen knowledge
    from duplicate evidence"). average_win_rate is a genuine average of
    win_rate across every independent confirming run (a ratio, not a raw
    count, so this doesn't double-count). confirmation_count/
    contradiction_count count distinct validation runs, never raw trades.

    Not frozen: `llm_note` is attached after construction, same pattern as
    ReflectionObservation/Hypothesis."""

    knowledge_id: str
    created_at: str
    title: str
    description: str
    conditions: list[list[str]]
    originating_hypotheses: list[str] = Field(default_factory=list)
    evidence_count: int = Field(ge=0)
    cumulative_sample_size: int = Field(ge=0)
    cumulative_successes: int = Field(ge=0)
    cumulative_failures: int = Field(ge=0)
    average_win_rate: float = Field(ge=0.0, le=1.0)
    shrunk_win_rate: float = Field(ge=0.0, le=1.0)
    confidence_interval: list[float]  # [wilson_lower, wilson_upper]
    first_seen: str
    last_confirmed: str
    confirmation_count: int = Field(ge=0)
    contradiction_count: int = Field(ge=0)
    evidence_strength: EvidenceStrength | None = None
    lifecycle_status: KnowledgeLifecycleStatus
    previous_lifecycle_status: KnowledgeLifecycleStatus | None = None
    llm_note: str | None = None

    # Deployment gate — see KnowledgePromotionStatus. Defaulted so every
    # existing construction call (knowledge_base_engine.py, tests) stays
    # valid unchanged; new entries always get CANDIDATE explicitly, never
    # rely on this default firing silently.
    promotion_status: KnowledgePromotionStatus = KnowledgePromotionStatus.CANDIDATE
    # Informational only — stock_scanner.pipeline.knowledge_application
    # never reads these, only promotion_status itself.
    promoted_at: str | None = None
    promoted_by: str | None = None
    promotion_reason: str | None = None

    def is_promoted(self) -> bool:
        """Typed equivalent of is_entry_promoted() for code that already
        holds a validated KnowledgeEntry instance — promotion_status is
        already a real KnowledgePromotionStatus member here (Pydantic
        guarantees it), so this is a plain enum comparison, no string
        parsing or fail-closed handling needed the way the raw-dict
        version requires."""
        return self.promotion_status == KnowledgePromotionStatus.PROMOTED


class KnowledgeNote(BaseModel):
    """One LLM-written plain-English gloss for a single, already-existing
    KnowledgeEntry — same contract as HypothesisNote/ObservationNote: the
    LLM may only reference knowledge_ids it was given, never invent new
    ones."""

    knowledge_id: str
    note: str = Field(min_length=1, max_length=300)


class KnowledgeChangeHighlight(BaseModel):
    """One LLM-written note calling out a knowledge_id whose
    lifecycle_status differs from its previous_lifecycle_status — the
    CHANGE itself is code-computed (KnowledgeEntry.previous_lifecycle_status
    != .lifecycle_status), the LLM only narrates what it means."""

    knowledge_id: str
    note: str = Field(min_length=1, max_length=300)


class KnowledgeGroup(BaseModel):
    """One LLM-proposed thematic grouping of knowledge_ids (e.g. "sector-
    driven findings") — narrative labeling only, not a merge: the
    deterministic engine already decided what counts as the same belief
    via exact normalized condition-set equality before this call ever
    happens."""

    label: str = Field(min_length=1, max_length=120)
    knowledge_ids: list[str] = Field(default_factory=list, max_length=20)


class KnowledgeNarrativeOutput(BaseModel):
    """LLM output for
    stock_scanner.ai_lab.agents.knowledge_review_agent — pure narrative
    over already-curated KnowledgeEntry objects. No numeric field
    anywhere, and no field that could create knowledge, change a
    lifecycle_status, compute confidence, or merge entries — the
    deterministic engine already decided all of that. Maps 1:1 onto the
    four allowed LLM responsibilities: summarize -> overall_summary,
    explain -> knowledge_notes, organize -> organized_groups, highlight
    important changes -> highlighted_changes."""

    overall_summary: str = Field(min_length=1, max_length=800)
    knowledge_notes: list[KnowledgeNote] = Field(default_factory=list, max_length=20)
    organized_groups: list[KnowledgeGroup] = Field(default_factory=list, max_length=10)
    highlighted_changes: list[KnowledgeChangeHighlight] = Field(default_factory=list, max_length=20)
