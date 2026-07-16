"""Prompt builders for AI Lab agents.

Rule enforced by construction, not just instruction text: every number and
every candidate observation the model is allowed to reference is embedded
in the prompt FROM code-computed data (Evidence, DecisionTrace,
ConfidenceBreakdown, HistoricalComparison, and the rule-based candidate
highlights from scoring.generate_evidence_highlights). The model is asked
to narrate/select/explain over given facts — never to state new ones. See
schemas.py's module docstring for how this is enforced again on the
response side (HypothesisOutput/DecisionOutput carry no numeric scoring
fields at all anymore).
"""
from __future__ import annotations

from stock_scanner.ai_lab.models import AIModelSpec
from stock_scanner.ai_lab.schemas import (
    ConfidenceBreakdown,
    DecisionTrace,
    Evidence,
    HistoricalComparison,
    Hypothesis,
    KnowledgeEntry,
    ReflectionObservation,
)

HYPOTHESIS_SYSTEM_PROMPT = """You are a quantitative research assistant for an experimental, \
non-production stock recommendation system (IDX / Indonesian stock market). You are given \
ONLY validated, code-computed evidence and a pool of pre-generated candidate observations below \
— you must not invent, estimate, or restate any fact that was not given to you. Your job is to \
write a short rationale and select/lightly rephrase (never invent new ones) the most relevant \
candidate strengths, weaknesses, and risks. Respond with ONLY a JSON object matching the \
requested schema, no other text."""

DECISION_SYSTEM_PROMPT = """You are a narration assistant for an experimental, non-production \
stock recommendation system (IDX / Indonesian stock market). All scoring, confidence, \
recommendation level, risk level, and historical-comparison verdict have ALREADY been computed \
by code and are given to you below — you cannot change them and must not contradict them. Your \
only job is to explain, in plain language, why those numbers came out the way they did. Do not \
invent statistics or restate numbers incorrectly. Respond with ONLY a JSON object matching the \
requested schema, no other text."""

REFLECTION_SYSTEM_PROMPT = """You are a research-summary assistant for an experimental, \
non-production stock recommendation system (IDX / Indonesian stock market). You are given a \
fixed list of already-gated, already-scored statistical observations below — every number and \
every observation_id has ALREADY been computed by code (Wilson confidence intervals, Fisher's \
exact test, Benjamini-Hochberg correction, or a one-sample binomial test) from real historical \
trade outcomes. You must not invent, estimate, or restate any number differently, and you must \
NEVER reference an observation_id that is not in the list given to you. You have exactly three \
jobs: (1) write a short overall_summary synthesizing the pattern of findings, (2) order the \
GIVEN observation_ids by importance in prioritized_observation_ids (this list may only contain \
ids from the list below — do not add, drop silently is fine, but never invent), (3) optionally \
write one short plain-English note per observation_id explaining what it means in \
observation_notes, without restating or altering any number. You have no ability to modify \
thresholds, change production logic, generate code, or update any knowledge base — you are \
narrating already-final research findings, nothing else. Respond with ONLY a JSON object \
matching the requested schema, no other text."""

HYPOTHESIS_REVIEW_SYSTEM_PROMPT = """You are a research-summary assistant for an experimental, \
non-production stock recommendation system (IDX / Indonesian stock market). You are given a \
fixed list of already-validated-or-rejected hypotheses below — every number, condition, and \
hypothesis_id has ALREADY been computed by code (Wilson confidence intervals, Fisher's exact \
test, shrunk win rate, Benjamini-Hochberg correction) from real historical trade outcomes. You \
must not invent, estimate, or restate any number differently, and you must NEVER reference a \
hypothesis_id that is not in the list given to you. You have exactly four jobs: (1) write a \
short overall_summary synthesizing the pattern across these hypotheses, (2) order the GIVEN \
hypothesis_ids by importance/actionability in prioritized_hypothesis_ids (only ids from the \
list below — never invent one), (3) optionally write one short plain-English note per \
hypothesis_id explaining what it means in hypothesis_notes, without restating or altering any \
number, (4) optionally group hypothesis_ids that describe the same underlying finding from \
different angles into clusters, each with a short label — this is a narrative grouping only, \
not a new statistical claim. You determine NOTHING about whether a hypothesis is true — that \
was already decided by code before you saw this list. You have no ability to modify \
thresholds, change production logic, generate code, or update any knowledge base — you are \
narrating already-final research findings, nothing else. Respond with ONLY a JSON object \
matching the requested schema, no other text."""

KNOWLEDGE_REVIEW_SYSTEM_PROMPT = """You are a research-summary assistant for an experimental, \
non-production stock recommendation system (IDX / Indonesian stock market). You are given a \
fixed list of already-curated knowledge entries below — every number, condition, lifecycle \
status, and knowledge_id has ALREADY been computed by a deterministic engine (grouping by exact \
normalized condition-set, Wilson confidence intervals, shrunk win rates, and a fixed integer \
lifecycle ladder) from real historical validation runs. You must not invent, estimate, or \
restate any number, condition, or lifecycle status differently, and you must NEVER reference a \
knowledge_id that is not in the list given to you. You have exactly four jobs: (1) write a short \
overall_summary synthesizing the state of accumulated knowledge, (2) write one short \
plain-English note per knowledge_id explaining what it means in knowledge_notes, without \
restating or altering any number or status, (3) optionally group knowledge_ids that share a \
theme (e.g. the same sector, or a similar kind of setup) into organized_groups, each with a \
short label — this is a narrative grouping only, not a merge, since the engine already decided \
what counts as the same belief before you ever saw this list, (4) optionally call out, in \
highlighted_changes, any knowledge_id whose lifecycle status differs from its previous one — the \
CHANGE itself is already computed and given to you; you only explain what it means. You \
determine NOTHING: not whether a pattern is true, not its lifecycle status, not its confidence, \
and you may never merge, split, or create a knowledge entry — the deterministic engine already \
decided all of that before this call happened. You have no ability to modify thresholds, change \
production logic, generate code, or promote anything into the production knowledge base — you \
are narrating already-final research findings, nothing else. Respond with ONLY a JSON object \
matching the requested schema, no other text."""


def build_hypothesis_prompt(evidence: Evidence, model_spec: AIModelSpec, highlights: dict) -> str:
    indicators = "\n".join(f"  - {k}: {v}" for k, v in evidence.technical_indicators.items())
    stats = "\n".join(
        f"  - n={s.get('n')}, n_success={s.get('n_success')}, win_rate={s.get('win_rate')}, "
        f"win_rate_shrunk={s.get('win_rate_shrunk')}, ci_lower={s.get('ci_lower')}, "
        f"ci_upper={s.get('ci_upper')}, p_value_adjusted={s.get('p_value_adjusted')}"
        for s in evidence.statistical_evidence
    ) or "  (none available for this ticker yet)"
    patterns = "\n".join(f"  - {p}" for p in evidence.similar_patterns) or "  (none)"

    candidate_strengths = "\n".join(f"  - {s}" for s in highlights.get("strengths", [])) or "  (none generated)"
    candidate_weaknesses = "\n".join(f"  - {w}" for w in highlights.get("weaknesses", [])) or "  (none generated)"
    candidate_risks = "\n".join(f"  - {r}" for r in highlights.get("risks", [])) or "  (none generated)"

    return f"""Persona: {model_spec.display_name} — {model_spec.description}
{model_spec.persona_instructions}

Ticker: {evidence.ticker}

Technical indicators (code-computed, current):
{indicators}

Statistically validated pattern evidence (from Learning Agent Phase 1's gated pattern clusters,
exact matches only):
{stats}

Similar historical patterns matched (exact):
{patterns}

Closest known pattern similarity (partial match across ALL patterns, not just exact ones):
{evidence.best_pattern_similarity_pct:.1f}%

Candidate strengths (code-generated from the evidence above — choose the most relevant, you may
rephrase for fluency but must not invent new ones or add facts not listed):
{candidate_strengths}

Candidate weaknesses (same rule):
{candidate_weaknesses}

Candidate risks (same rule):
{candidate_risks}

Respond with ONLY a JSON object with exactly these keys:
  "why": one or two sentences on why this ticker is interesting given the evidence above
  "strengths": up to 5 strings, chosen/rephrased ONLY from the candidate strengths above
  "weaknesses": up to 5 strings, chosen/rephrased ONLY from the candidate weaknesses above
  "risks": up to 5 strings, chosen/rephrased ONLY from the candidate risks above

If a candidate list above is "(none generated)", return an empty list for that key — do not
invent an entry. Do not include any other fields or any numbers not present in the evidence
above."""


def build_decision_prompt(
    evidence: Evidence,
    hypothesis_why: str,
    model_spec: AIModelSpec,
    trace: DecisionTrace,
    confidence: ConfidenceBreakdown,
    comparison: HistoricalComparison,
) -> str:
    comparison_stats = (
        f"sample_size={comparison.sample_size}, win_rate={comparison.win_rate}, "
        f"ci_lower={comparison.ci_lower}, ci_upper={comparison.ci_upper}, verdict={comparison.verdict.value}"
        if comparison.sample_size is not None
        else "no validated historical pattern matches this ticker (verdict=no_data)"
    )

    return f"""Persona: {model_spec.display_name} — {model_spec.description}

Ticker: {evidence.ticker}
Prior hypothesis rationale: {hypothesis_why}

Code-computed decision trace (already final, do not alter):
  technical_score={trace.technical_score}, statistical_score={trace.statistical_score},
  pattern_similarity_score={trace.pattern_similarity_score}, risk_score={trace.risk_score},
  final_score={trace.final_score}

Code-computed confidence breakdown (already final, do not alter):
  technical={confidence.technical}, statistical={confidence.statistical},
  pattern_similarity={confidence.pattern_similarity}, risk_adjustment={confidence.risk_adjustment},
  final_confidence={confidence.final_confidence}

Code-computed historical comparison (already final, do not alter):
  {comparison_stats}

Based ONLY on the numbers above, respond with ONLY a JSON object with exactly these keys:
  "reasoning_summary": 1-3 sentences explaining why the technical indicators are bullish or
    bearish, why the historical evidence supports or contradicts them, and why the final
    score/confidence make this recommendation conservative or aggressive
  "historical_comparison_explanation": 1-2 sentences explaining the historical_comparison
    verdict above ({comparison.verdict.value}) using only the given sample_size/win_rate/CI —
    if verdict is no_data, say so plainly rather than inventing a comparison
  "confidence_explanation": 1 sentence explaining the final_confidence value above in terms of
    its technical/statistical/pattern_similarity/risk_adjustment components

Do not include any other fields, and do not state any score/confidence/rate different from the
numbers given above."""


def build_reflection_prompt(observations: list[ReflectionObservation]) -> str:
    blocks = []
    for o in observations:
        stats = ", ".join(f"{k}={v}" for k, v in o.supporting_statistics.items())
        blocks.append(
            f"""observation_id: {o.observation_id}
  category: {o.category.value}
  title: {o.title}
  description: {o.description}
  affected_trade_count: {o.affected_trade_count}
  confidence: {o.confidence:.4f}
  supporting_statistics (code-computed, already final): {stats}"""
        )
    observations_block = "\n\n".join(blocks)

    return f"""Below are {len(observations)} statistically gated research observations over \
resolved AI Lab recommendations. Every field was computed by code — treat all of it as given, \
final fact.

{observations_block}

Respond with ONLY a JSON object with exactly these keys:
  "overall_summary": 2-4 sentences synthesizing the pattern across these observations —
    e.g. which models/sectors/patterns stand out, and whether confidence looks well-calibrated
  "prioritized_observation_ids": the observation_id values above, ordered from most to least
    important/actionable for a human researcher to look at next — must only contain
    observation_ids from the list above, never a new or invented id
  "observation_notes": a list of {{"observation_id": ..., "note": ...}} objects, one short
    plain-English sentence per observation explaining what it means in practice — again, only
    for observation_ids given above, and the note must not restate any number incorrectly or add
    a number not already given

Do not include any other fields, and do not invent any observation_id, statistic, ticker, or
threshold not present above."""


def build_hypothesis_review_prompt(hypotheses: list[Hypothesis]) -> str:
    blocks = []
    for h in hypotheses:
        cond_str = ", ".join(f"{dim}={value}" for dim, value in h.conditions)
        blocks.append(
            f"""hypothesis_id: {h.hypothesis_id}
  status: {h.status.value}
  conditions: {cond_str}
  description: {h.description}
  sample_size: {h.sample_size}, successes: {h.successes}, failures: {h.failures}
  win_rate: {h.win_rate:.4f}, shrunk_win_rate: {h.shrunk_win_rate:.4f}
  wilson_ci: [{h.wilson_lower:.4f}, {h.wilson_upper:.4f}]
  fisher_p: {h.fisher_p:.4f}, bh_adjusted_p: {h.bh_adjusted_p:.4f}
  evidence_strength: {h.evidence_strength.value if h.evidence_strength else 'N/A'}
  rejection_reason: {h.rejection_reason or 'N/A'}"""
        )
    hypotheses_block = "\n\n".join(blocks)

    return f"""Below are {len(hypotheses)} statistically validated-or-rejected hypotheses over \
resolved AI Lab recommendations. Every field was computed by code — treat all of it as given, \
final fact.

{hypotheses_block}

Respond with ONLY a JSON object with exactly these keys:
  "overall_summary": 2-4 sentences synthesizing the pattern across these hypotheses — e.g.
    which condition combinations look strongest, and any recurring themes across validated ones
  "prioritized_hypothesis_ids": the hypothesis_id values above, ordered from most to least
    important/actionable for a human researcher to look at next — must only contain
    hypothesis_ids from the list above, never a new or invented id
  "hypothesis_notes": a list of {{"hypothesis_id": ..., "note": ...}} objects, one short
    plain-English sentence per hypothesis explaining what it means in practice — only for
    hypothesis_ids given above, and the note must not restate any number incorrectly or add a
    number not already given
  "clusters": a list of {{"label": ..., "hypothesis_ids": [...]}} objects grouping
    hypothesis_ids that describe the same underlying finding from different angles (e.g. two
    hypotheses that largely overlap in which recommendations they cover) — only group
    hypothesis_ids from the list above, and this is a narrative grouping only, not a new
    statistical claim; return an empty list if nothing meaningfully clusters

Do not include any other fields, and do not invent any hypothesis_id, statistic, condition, or
threshold not present above."""


def build_knowledge_review_prompt(entries: list[KnowledgeEntry]) -> str:
    blocks = []
    for e in entries:
        cond_str = ", ".join(f"{dim}={value}" for dim, value in e.conditions)
        prev_status = e.previous_lifecycle_status.value if e.previous_lifecycle_status else "N/A (first time seen)"
        blocks.append(
            f"""knowledge_id: {e.knowledge_id}
  conditions: {cond_str}
  description: {e.description}
  lifecycle_status: {e.lifecycle_status.value} (previous: {prev_status})
  evidence_count: {e.evidence_count}, confirmation_count: {e.confirmation_count}, contradiction_count: {e.contradiction_count}
  average_win_rate: {e.average_win_rate:.4f}, shrunk_win_rate: {e.shrunk_win_rate:.4f}
  confidence_interval: [{e.confidence_interval[0]:.4f}, {e.confidence_interval[1]:.4f}]
  cumulative: {e.cumulative_successes}/{e.cumulative_sample_size} (latest snapshot)
  first_seen: {e.first_seen}, last_confirmed: {e.last_confirmed}
  evidence_strength: {e.evidence_strength.value if e.evidence_strength else 'N/A'}"""
        )
    entries_block = "\n\n".join(blocks)

    return f"""Below are {len(entries)} curated knowledge entries accumulated from resolved AI Lab \
recommendations. Every field was computed by a deterministic engine — treat all of it as given, \
final fact, including which entries changed lifecycle_status (shown as "previous:" above).

{entries_block}

Respond with ONLY a JSON object with exactly these keys:
  "overall_summary": 2-4 sentences synthesizing the current state of accumulated knowledge — e.g.
    how many entries are strengthening vs. weakening, and any notable persistent patterns
  "knowledge_notes": a list of {{"knowledge_id": ..., "note": ...}} objects, one short
    plain-English sentence per entry explaining what it means in practice — only for
    knowledge_ids given above, and the note must not restate any number or status incorrectly
  "organized_groups": a list of {{"label": ..., "knowledge_ids": [...]}} objects grouping
    knowledge_ids that share a theme (e.g. same sector, similar setup) — only knowledge_ids from
    the list above; return an empty list if nothing meaningfully groups
  "highlighted_changes": a list of {{"knowledge_id": ..., "note": ...}} objects, one per
    knowledge_id whose lifecycle_status differs from its previous_lifecycle_status shown above,
    explaining what that transition means — only for knowledge_ids given above where a change is
    shown; return an empty list if no entry changed status this run

Do not include any other fields, and do not invent any knowledge_id, statistic, condition, or
lifecycle status not present above. You may never decide that a knowledge entry is true, change
its lifecycle_status, compute a confidence value, or merge/split entries — all of that has
already been decided by the deterministic engine before this prompt was built."""
