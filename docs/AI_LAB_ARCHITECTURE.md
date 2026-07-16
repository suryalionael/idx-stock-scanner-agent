# AI Lab — Architecture

**Status:** storage + agents + client + dashboard implemented, tested, and verified
end-to-end against the live 9router endpoint (base URL/auth/request format confirmed
— see "9router configuration status"). Explainability upgrade (decision trace,
confidence breakdown, historical comparison, rule-based recommendation levels — see
"Explainability upgrade" below) is live. Performance Tracker automation (see
"Recommendation lifecycle") is also live: `scripts/resolve_ai_lab.py` resolves
PENDING/ACTIVE recommendations against forward OHLCV data instead of the old
manual-only runbook, and every resolved (or still-tracking) recommendation now carries
running MFE/MAE-style trade analytics — the designed input surface for the future
Calibration Engine / Self Reflection stages (see "Trade analytics" below). Reflection
Engine (see "Reflection Engine" below) is also live: `scripts/run_reflection_engine.py`
reviews RESOLVED recommendations and produces statistically gated
`reflection_observations` — the first component of the closed learning loop. Hypothesis
Generator + Statistical Validation (see "Hypothesis Generator + Statistical Validation"
below) is also live: `scripts/run_hypothesis_engine.py` refines
`reflection_observations`' single-dimension findings into multi-condition (order 2-3)
`validated_hypotheses`. Knowledge Base Engine (see "Knowledge Base Engine" below) is
also live: `scripts/run_knowledge_base_engine.py` curates ALL historical
`validated_hypotheses` into long-lived `knowledge_entries` with a deterministic
lifecycle (Emerging/Confirmed/Strong/Weakening/Contradicted/Archived) — a candidate
pool for a possible future promotion into the production `knowledge_base` table
(always human-reviewed, never automatic), and the closing stage of the current pipeline
before the future Calibration Engine.
**Not yet done:** no scheduled GitHub Actions workflow — `scripts/run_ai_lab.py`,
`scripts/resolve_ai_lab.py`, `scripts/run_reflection_engine.py`,
`scripts/run_hypothesis_engine.py`, and `scripts/run_knowledge_base_engine.py` are all
manual-only. A deliberate follow-up, not an oversight — see "What's phased for later."

---

## Explainability upgrade

The AI Score, confidence, recommendation level, risk level, and expected return are
**never LLM-generated** — all computed deterministically in
`stock_scanner/ai_lab/scoring.py` from Evidence before either LLM call happens. The
LLM's role is narrowed to pure narrative: explaining, in prose, numbers it did not
produce and cannot change. Same evidence in -> same numbers out, always.

- **`DecisionTrace`** (`technical_score`, `statistical_score`,
  `pattern_similarity_score`, `risk_score`, `final_score`, all 0-100): `technical_score`
  reuses the production scanner's own `trend_score`/`momentum_score`/`breakout_score`/
  `volume_score` (already computed and tuned in `signal_engine.py`) wherever a persona
  has a direct equivalent — only `reversal_ai` gets a dedicated formula, since the
  scanner has no reversal-setup concept. `statistical_score` is the mean
  `win_rate_shrunk` of exactly-matched `knowledge_base` patterns (0, not a neutral 50,
  when nothing matches). `risk_score` reuses `quality_filters.py`'s own `is_uma`/
  `is_special_monitoring`/`quality_penalty_total` flags plus ATR volatility.
  `final_score` is a documented weighted blend (technical 35%, statistical 35%,
  pattern_similarity 15%, risk 15% as a penalty) — see `scoring.WEIGHTS`.
- **`ConfidenceBreakdown`** — the same four components rescaled to 0-1, with
  `risk_adjustment` always <= 0 (explicit penalty, never a bonus).
- **`RecommendationLevel`** (`STRONG_BUY`/`BUY`/`WATCH`/`AVOID`, replacing the old
  `BUY`/`WATCH`/`SELL`/`AVOID`) and **`RiskLevel`** are both rule-based threshold
  classifiers over the trace/confidence (`scoring.classify_recommendation_level`/
  `classify_risk_level`) — never an LLM judgment call, so reproducible by construction.
- **`expected_return`** is always `null`: `knowledge_base` pattern stats carry win/loss
  *rate*, not return *magnitude*, so any number here would be an estimate dressed up as
  data. Kept as a field for a future evidence source that does carry magnitude data.
- **`HistoricalComparison`** — sample_size/win_rate/CI/verdict (stronger/weaker/
  similar/no_data) are code-computed; only the one-sentence `explanation` is an LLM
  call constrained to those exact numbers.
- **`generate_evidence_highlights()`** — a rule-based pool of grounded candidate
  strengths/weaknesses/risks (e.g. "Strong ADX (50.4) confirms high trend strength").
  The Hypothesis Agent may only select/rephrase from this pool, never invent; if it
  returns an empty list, `decision_agent.assemble_recommendation` falls back to the
  candidates directly, so the dashboard is never left with empty arrays when grounded
  observations exist.

Verified live: `docs/AI_LAB_ARCHITECTURE.md`'s claims above were checked against a real
9router response (BAPA.JK/momentum_ai) — the reasoning text correctly cited the exact
supplied numbers (ADX 50.4, win rate 13.04%, n=115) and explained *why* bullish
technicals were being overridden by weak historical evidence into a conservative WATCH,
matching the "Better Reasoning" requirement almost verbatim.

## Why this exists, and what it is not

AI Lab is an **experimental, standalone recommendation engine** — not a replacement
for, extension of, or input into the Production Scanner. Nothing in
`stock_scanner/pipeline/`, `stock_scanner/alerts/`, `signal_engine.py`, or
`scanner_config.yaml` is read or modified by anything under `stock_scanner/ai_lab/`,
and nothing in `stock_scanner/pipeline/` reads AI Lab's tables. The long-term goal
(per product spec) is for AI Lab to *prove* superior long-term performance through
its own forward-tested track record before any future integration into production is
even considered — this document does not implement that integration; it only prepares
the architecture (see "Future Ready").

## Pipeline

```
Historical Market Data
  -> Feature Engineering        stock_scanner.pipeline.feature_builder      (existing, reused)
  -> Pattern Miner              stock_scanner.learning.pattern_miner        (existing, reused)
  -> Statistical Validation     stock_scanner.learning.pattern_dedup        (existing, reused)
  -> Knowledge Base              stock_scanner.db.knowledge_base            (existing, reused — read-only)
  -> Hypothesis Agent             stock_scanner.ai_lab.agents.hypothesis_agent   (new)
  -> Decision Agent               stock_scanner.ai_lab.agents.decision_agent     (new)
  -> AI Recommendation Engine     stock_scanner.db.ai_lab                        (new)
  -> AI Lab Dashboard             dashboard.ai_lab_view                          (new)
```

Everything above "Hypothesis Agent" already existed as Learning Agent Phase 1
(`docs/LEARNING_AGENT_ARCHITECTURE.md`) and is reused as-is, not duplicated. AI Lab's
own `hypothesis_agent.py`/`decision_agent.py` are **distinct** from
`stock_scanner/learning/hypothesis_agent.py`: the Learning Agent one produces a
human-reviewed research narrative (`knowledge_base` rows, no lifecycle, no score); AI
Lab's produces a ranked, lifecycle-tracked, dashboard-facing recommendation
(`ai_recommendations` rows: score, confidence, BUY/WATCH/SELL/AVOID, expected return,
risk level, PENDING → ACTIVE → CLOSED/EXPIRED).

## Module map

```
stock_scanner/ai_lab/
  __init__.py       — package overview (this pipeline diagram, in code)
  schemas.py        — Pydantic contracts: Evidence, HypothesisOutput, DecisionOutput,
                       AIRecommendation, RiskLevel/RecommendationAction/RecommendationStatus enums
  client.py         — NineRouterClient (async, retry, timeout, structured JSON)
                       + MockNineRouterClient (deterministic, no network)
  prompts.py        — prompt builders; every number in a prompt comes FROM Evidence
  models.py         — AI_MODEL_REGISTRY: plug-and-play model personas
  performance.py    — win rate / avg return / avg loss / profit factor / expected
                       value / sharpe ratio, computed per ai_model or overall
  resolver.py       — activate_pending() + resolve_active(): PENDING→ACTIVE→CLOSED/EXPIRED
                       lifecycle + running MFE/MAE/holding_days trade analytics, pure
                       functions over already-loaded DataFrames + raw parquet
  reflection_engine.py — generate_observations(): 3-tier statistically gated findings
                       over RESOLVED recommendations, reuses pattern_miner.py's Wilson
                       CI/BH/shrinkage primitives, pure function, no DB/LLM I/O. Also
                       exposes score_group()/passes_slice_gate()/prepare_dataframe()
                       (public) for statistical_validation.py/hypothesis_engine.py to reuse.
  hypothesis_engine.py — generate_candidate_hypotheses(): apriori-seeded, order-2/3
                       condition-combination discovery from reflection_observations,
                       raw counts only, no Wilson/Fisher/BH, pure function, no DB/LLM I/O
  statistical_validation.py — validate_hypotheses(): Wilson CI/Fisher's exact/shrunk win
                       rate/BH-corrected scoring of hypothesis_engine.py's candidates,
                       reusing reflection_engine.py's score_group()/passes_slice_gate()
  knowledge_base_engine.py — generate_knowledge_entries(): curates ALL historical
                       validated_hypotheses into long-lived KnowledgeEntry rows with a
                       deterministic lifecycle ladder — no new statistics, pure
                       aggregation, no DB/LLM I/O
  agents/
    hypothesis_agent.py  — build_evidence() + generate_hypothesis() (per-recommendation
                            narrator for the Decision pipeline — NOT the same thing as
                            hypothesis_engine.py/hypothesis_review_agent.py below)
    decision_agent.py    — generate_decision() + assemble_recommendation() + recommendation_id()
    reflection_agent.py  — generate_reflection_narrative(): LLM summarize/prioritize
                            over already-gated ReflectionObservation objects only
    hypothesis_review_agent.py — generate_hypothesis_narrative(): LLM summarize/
                            prioritize/cluster over already-validated/rejected
                            Hypothesis objects only
    knowledge_review_agent.py — generate_knowledge_narrative(): LLM summarize/explain/
                            organize/highlight over already-curated KnowledgeEntry
                            objects only

stock_scanner/db/ai_lab.py             — SQLite IO: upsert/update/load/export/import for ai_recommendations + ai_learning_events
stock_scanner/db/reflection.py         — SQLite IO: upsert/load/export/import for reflection_observations
stock_scanner/db/hypotheses.py         — SQLite IO: upsert/load/export/import for validated_hypotheses
stock_scanner/db/knowledge_entries.py  — SQLite IO: upsert/load/export/import for knowledge_entries (NOT knowledge_base.py — see "Knowledge Base Engine" above for the naming collision)
stock_scanner/db/schema.sql            — appended, not modified, existing tables
scripts/run_ai_lab.py                   — manual orchestrator (--mock or live)
scripts/resolve_ai_lab.py               — manual resolver orchestrator (activate + resolve + track, see resolver.py)
scripts/run_reflection_engine.py        — manual reflection orchestrator (see reflection_engine.py/reflection_agent.py)
scripts/run_hypothesis_engine.py        — manual hypothesis orchestrator (see hypothesis_engine.py/statistical_validation.py/hypothesis_review_agent.py)
scripts/run_knowledge_base_engine.py    — manual curation orchestrator (see knowledge_base_engine.py/knowledge_review_agent.py)
dashboard/ai_lab_view.py     — read-only dashboard tab (8 sections; sections 6/7/8 delegate to reflection_view.py/hypothesis_view.py/knowledge_entries_view.py)
dashboard/reflection_view.py — read-only Reflection section (7 subsections, see spec)
dashboard/hypothesis_view.py — read-only Hypothesis Generator section (7 subsections, see spec)
dashboard/knowledge_entries_view.py — read-only Knowledge Base section (7 subsections, see spec; NOT knowledge_base_view.py, which already serves the production table)
dashboard/data_loader.py     — load_ai_recommendations_payload / load_ai_learning_events_payload / load_reflection_report_payload / load_hypotheses_report_payload / load_knowledge_report_payload
data/published/ai_recommendations.json, ai_learning_events.json, reflection_report.json, hypotheses_report.json, knowledge_report.json — committed mirrors
```

## Isolation guarantees (verified, not just documented)

- `grep -rn "ai_lab" stock_scanner/pipeline/ stock_scanner/alerts/` → no matches.
- `ai_recommendations`/`ai_learning_events` have no foreign keys to/from
  `signals`/`outcomes`/`model_registry`.
- The dashboard tab only ever calls `dashboard.data_loader.load_ai_*_payload()` (reads
  committed JSON) and `stock_scanner.ai_lab.performance.compute_performance()` (pure
  pandas, no I/O) — it never imports `client.py` or `agents/`, so there is no live
  9router call from the dashboard, matching this repo's existing no-live-fetch-from-
  dashboard rule (see `daily_movers_view.py`/`knowledge_base_view.py`).

## Evidence guardrail — "no hallucinated statistics"

Every number an LLM call is allowed to reason about is embedded in the prompt by code,
from `Evidence` (technical indicators from the current ranked-CSV snapshot;
`statistical_evidence`/`similar_patterns` from `knowledge_base` rows whose
`slice_definition` matches the ticker's current feature values via exact equality —
see `hypothesis_agent.build_evidence()`). The LLM is only asked to narrate/judge over
given numbers. `AIRecommendation.reasoning` stores `Evidence` verbatim alongside the
LLM's qualitative fields, so the dashboard's detail panel always renders the same
numbers the model saw — never numbers the model invented.

## 9router configuration status

9router was already a recorded product decision before this work
(`stock_scanner/learning/hypothesis_agent.py`'s `NineRouterClient` stub,
`docs/LEARNING_AGENT_ARCHITECTURE.md` "Provider: 9router") — but its base URL, auth
header format, and request/response shape were never documented anywhere in this repo.
`stock_scanner/ai_lab/client.py`:

- reads `NINEROUTER_API_KEY`, `NINEROUTER_MODEL`, **and** `NINEROUTER_BASE_URL` from
  environment variables only — none hardcoded, including the base URL (a wrong
  hardcoded host would leak a real API key to an unintended domain);
- raises `NineRouterConfigError` immediately if any are unset — same fail-fast
  contract as the existing Learning Agent stub, never a silent guess;
- **assumes** an OpenAI-compatible `/chat/completions` endpoint (the lowest common
  denominator across LLM routing services) — if 9router's real contract differs,
  update `NineRouterClient._call`/`_parse` only; nothing else in `ai_lab/` depends on
  the wire format.

Until `NINEROUTER_BASE_URL` is confirmed and set, use `scripts/run_ai_lab.py --mock`
(backed by `MockNineRouterClient`) to exercise the full pipeline deterministically —
this is exactly how it was verified end-to-end for this pass (see runbook below).

## AI Models — plug-and-play

`stock_scanner/ai_lab/models.py`'s `AI_MODEL_REGISTRY` currently has four entries
(Momentum, Breakout, Reversal, Volume AI). Each is a named "lens" (a `focus_features`
list + `persona_instructions` string) applied to the same evidence/agent pipeline —
not a separately-trained statistical model. Adding a new one is a single new
`AIModelSpec` entry; performance tracking (`performance.py`) and the dashboard both
group by the `ai_model` column value with no hardcoded model list anywhere, so a new
entry appears in per-model performance the first time it produces a recommendation —
zero other code changes required.

## Recommendation lifecycle

`PENDING -> ACTIVE -> CLOSED | EXPIRED`, via `stock_scanner.db.ai_lab.update_recommendation_status()`
(validates the status string, returns the row count actually updated — 0 for a
nonexistent id, matching `knowledge_base.update_status`'s guardrail). A recommendation
row's identity (`id = sha1(ticker|ai_model|generated_date)`) is created once;
`upsert_recommendations()` re-running the same day updates content fields
(score/reasoning/etc.) but explicitly never touches `status`/`entry_price`/
`exit_price`/`return_percentage` or any tracking field (below) on conflict — those are
lifecycle fields owned only by `update_recommendation_status()`. This is what makes
"never overwrite history, append only" true in practice: content can be regenerated,
lifecycle progress cannot be reset.

Transitions are driven automatically by `stock_scanner.ai_lab.resolver` /
`scripts/resolve_ai_lab.py`, not manually:

- **`PENDING -> ACTIVE`** (`resolver.activate_pending`): once `generated_date`'s close
  exists in `data/raw/{ticker}.parquet`, `entry_price` is set to that close. Rows for
  tickers whose raw data doesn't cover `generated_date` yet stay `PENDING` — safe to
  re-run tomorrow.
- **`ACTIVE -> CLOSED | EXPIRED`** (`resolver.resolve_active`): simulates a trade over
  the forward window using the *same fallback exit policy*
  `stock_scanner.pipeline.evaluator` uses for signals with no explicit levels (AI Lab
  recommendations never carry `tp`/`cutloss` — Evidence produces a score/confidence
  tier, not price levels): `tp = entry * (1 + risk_pct/100 * 1.8)`,
  `cl = entry * (1 - risk_pct/100)`, checked TP-then-SL, first hit within
  `horizon_days` wins (`horizon_days=10`, `risk_pct=3.0` by default — same constants as
  `evaluator._DEFAULT_HORIZON`/`_DEFAULT_RISK_PCT`, so `return_percentage` is directly
  comparable to the production scanner's own realized returns). No hit within the
  horizon -> `EXPIRED`, mark-to-market at the horizon's last close.

### Trade analytics (Calibration Engine / Self Reflection input surface)

Every ACTIVE row also gets 6 tracking columns refreshed on **every** resolver run,
whether or not it resolves that run — the intended primary input for the future
Calibration Engine and Self Reflection stages, so neither has to re-derive trade
history from raw parquet later:

- `highest_price` / `lowest_price` — running max(high) / min(low) since entry.
- `max_runup_pct` — MFE (Maximum Favorable Excursion), `(highest_price - entry_price) / entry_price * 100`, clamped >= 0.
- `max_drawdown_pct` — MAE (Maximum Adverse Excursion), `(entry_price - lowest_price) / entry_price * 100`, clamped >= 0.
- `holding_days` — trading days elapsed since entry; keeps climbing while `ACTIVE`, freezes once resolved.
- `trade_outcome` (`WIN` | `LOSS` | `BREAKEVEN`) — **deliberately a separate axis from
  `status`**, set only once a row resolves, based on the sign of `return_percentage`.
  An `EXPIRED` mark-to-market row that happened to be profitable is still `WIN`, never
  inferred as a loss just because it never hit its target — so "did this
  recommendation end up right" can be queried without re-deriving it from
  `status` + return sign every time.

## Reflection Engine

First component of the closed learning loop: `Performance Tracker -> Reflection
Engine -> Hypothesis Generator -> Statistical Validation -> (future) Knowledge Base`.
Reviews **only RESOLVED** `ai_recommendations` (`status IN
('CLOSED','EXPIRED')`) and produces `reflection_observations` — statistically gated
findings, never an LLM-invented statistic.

**Methodology — reused, not reinvented.** `stock_scanner/ai_lab/reflection_engine.py`
imports `wilson_ci`/`benjamini_hochberg`/`shrunk_win_rate` directly from
`stock_scanner.learning.pattern_miner` (the same primitives that already gate the
production Knowledge Base) rather than re-deriving them. `pattern_miner.py` lives under
`stock_scanner/learning/`, not `stock_scanner/pipeline/`/`stock_scanner/alerts/`, so
this is the same one-way reuse the Hypothesis Agent already does for Pattern
Miner/Statistical Validation — see "Pipeline" above.

Three independently Benjamini-Hochberg-corrected tiers (mirrors `pattern_miner`'s
per-order-tier separation, so one noisy tier can't drown a real signal in another):

1. **Categorical dimensions** — `ai_model`, `sector` (via
   `stock_scanner.reference.issuers.get_sector`, a read-only reference lookup),
   `recommendation` level, and `historical_comparison.verdict` (closing the loop on
   whether the AI's own stronger/weaker/similar call was actually right). Wilson CI +
   Fisher's exact vs. rest of population, same shape as `pattern_miner._score_slice`.
2. **Technical indicator combinations** (order <= 2) — boolean keys auto-detected from
   each row's `reasoning.technical_indicators` (not a hardcoded feature list — AI Lab
   personas expose different indicators than the production scanner).
3. **Confidence calibration** — quartile buckets of stated `confidence`, tested against
   realized win rate via a one-sample binomial test (`scipy.stats.binomtest`), flagging
   `overconfident`/`underconfident` when the gate passes.

Gate: BH-adjusted `q < alpha` AND (Wilson CI lower bound above baseline, with enough
*successes* to trust it — a success pattern; OR CI upper bound below baseline, with
enough *non-successes* to trust it — a failure pattern). Using `n_success` as the
sample-size floor for success patterns but `(n - n_success)` for failure patterns is
deliberate: a slice with almost no wins can still have plenty of losses to reliably
show it underperforms, and gating failure patterns on `n_success` would make them
nearly undiscoverable by construction. Defaults `min_n_success=3` (lower than
`pattern_miner`'s 8 — AI Lab's resolved-trade population starts near zero and grows
slowly), `alpha=0.05` unchanged (sample-size floor and statistical rigor are different
knobs). Both are CLI flags (`--min-n-success`/`--alpha`), not YAML — same
no-scanner-config-coupling convention as the rest of AI Lab. With few resolved trades,
the report legitimately comes back empty — correct, not a bug.

Every observation's `confidence` field = `1 - p_value_adjusted` (clamped `[0,1]`) — one
consistent, documented definition across all three tiers, distinct from an individual
recommendation's own per-trade `confidence`.

**LLM role — summarize/explain/prioritize only, code-only fallback.** One 9router call
per run (`stock_scanner/ai_lab/agents/reflection_agent.py`), receiving every already-
gated observation's id/title/description/stats as grounded context.
`ReflectionNarrativeOutput` (`overall_summary`, `prioritized_observation_ids`,
`observation_notes`) has zero numeric or identifier-inventing fields — same discipline
as `HypothesisOutput`/`DecisionOutput`. **Deliberate deviation from
`run_ai_lab.py`'s precedent** (which drops an entire recommendation if its LLM call
fails): `generate_observations()` always runs and always produces the report's
`observations` array before any LLM call happens; the narrative call is a separate,
best-effort step — a failed or unconfigured (`NineRouterConfigError`) call just leaves
`"narrative": null` in the published report rather than discarding the (real,
code-computed) observations.

**Storage** — `reflection_observations` (append-only, mirrors `ai_learning_events`'
shape, not `ai_recommendations`' upsert-with-lifecycle shape: observations are
immutable analysis snapshots, so `observation_id = sha1(category|dimension|value|generated_at)`
is naturally unique per run, making this table double as the "Recent Reflections"
timeline). The run-level narrative isn't its own table — it's logged as one
`ai_learning_events` row (`event_type='reflection_generated'`) and passed straight into
`data/published/reflection_report.json`.

**Isolation** — same guarantees as the rest of AI Lab: `reflection_observations` has no
FKs to `signals`/`outcomes`/`model_registry`/`knowledge_base`, nothing in
`stock_scanner/pipeline/`/`stock_scanner/alerts/` reads it, and `dashboard/reflection_view.py`
only reads the committed JSON mirror (never imports `reflection_engine.py`/`client.py`/
`agents/`).

## Hypothesis Generator + Statistical Validation

Second/third components of the closed learning loop. Refines `reflection_observations`'
single-dimension findings into multi-condition (order 2-3), statistically validated
`validated_hypotheses` — e.g. Reflection says "sector Technology underperforms,"
Hypothesis Generator + Statistical Validation together answer "...specifically when
RSI is high and ATR is high, and that combination is statistically significant." A
candidate pool for a possible future Knowledge Base promotion — always via human
review, never automatic; nothing in this pass writes to `knowledge_base`.

**Two-component split, spec-literal.** `stock_scanner/ai_lab/hypothesis_engine.py` is
pure candidate *generation* — deterministic, no Wilson/Fisher/BH calls at all, just
condition-combination discovery + raw counts. `stock_scanner/ai_lab/statistical_validation.py`
is the *only* place that scores candidates, reusing (not re-deriving) two functions
promoted to public in `reflection_engine.py`: `score_group()` (Wilson CI + Fisher's
exact + shrunk win rate — already generic) and `passes_slice_gate()` (the same
symmetric success/failure gate Reflection Engine uses). `benjamini_hochberg` still comes
from `pattern_miner.py` directly. This is what "reuse existing statistical utilities...
do not duplicate implementations" means in practice — the same three primitives now
power `pattern_miner.py` -> `reflection_engine.py` -> `statistical_validation.py`, never
reimplemented a third time.

**Apriori-seeded candidate generation ("prune intelligently"), not brute-force.**
Cross-producting every `ai_model` x `sector` x `recommendation` x `verdict` x indicator
value up to order 3 would be a combinatorial explosion. Instead, every gated
`ReflectionObservation` (excluding `CONFIDENCE_CALIBRATION` — that category describes
confidence buckets, not a recommendation-attribute condition) is a seed:
categorical-tier observations seed an order-1 condition, `TECHNICAL_PATTERN`
observations (already order-2 pairs) seed directly at order 2. Each seed expands by
exactly one more atomic condition (any dimension not already present); a freshly
generated order-2 candidate is added back to the frontier for one more hop to order 3
only if it clears a cheap, *non-statistical* raw-lift heuristic
(`|win_rate - baseline_rate| >= 0.10`) — a pruning heuristic, not a significance claim;
real significance testing happens once, later, on every order-2 **and** order-3
candidate alike. Hard-capped at `max_order=3`, at most 2 expansion rounds. Duplicate
condition-sets reached via different seed paths are deduped at generation time (a
canonical sorted-condition-tuple key) — this, plus `INSERT OR IGNORE` at the DB layer,
is the full scope of "duplicate prevention" here; `pattern_dedup.py`'s Jaccard/overlap
semantic clustering is deliberately **not** reused for this — the spec's "reuse
existing statistical utilities" list names Fisher/Wilson/Shrunk/BH, not Jaccard/overlap,
and pulling that in would be new scope, not reuse.

**Numeric indicators, not just booleans.** `hypothesis_engine.py` auto-detects numeric
keys in `reasoning.technical_indicators` (not just booleans, unlike Reflection Engine's
technical tier) and buckets them into Low/Mid/High terciles via `pd.qcut` — this is
what makes "RSI is High AND ATR is High" a discoverable atomic condition, a
percentile-bucketed generalization rather than a hardcoded per-indicator threshold.

**Validation gate** — identical to Reflection Engine's, reused via `passes_slice_gate()`:
BH-adjusted `q < alpha` (pooled **separately per interaction order** — order-2 pool,
order-3 pool, same per-tier rationale as `pattern_miner`/Reflection Engine) AND a
directional Wilson CI separation from baseline with enough supporting trades on the
relevant side. Every candidate becomes a `Hypothesis` row — validated
(`evidence_strength` `STRONG` if `bh_adjusted_p < 0.01` else `MODERATE`) or rejected
(`rejection_reason` + `failed_gate` in `{"not_significant", "no_directional_lift"}`,
built from the same real numbers, never invented text — nothing is silently dropped).
`hypothesis_id = sha1(sorted_conditions|created_at)` — append-only per run, same
reasoning as `reflection_observations` ("Never overwrite rows. History is valuable" —
the same condition-set re-validated on a larger resolved-trade population later is a
new, worth-keeping data point). This differs from `knowledge_base.py`'s own
content-only hash specifically because that table dedupes *identical human-reviewed
text*, a different problem.

**LLM layer — summarize/explain/prioritize/cluster only, code-only fallback.**
`stock_scanner/ai_lab/agents/hypothesis_review_agent.py` — named this, not
`hypothesis_agent.py`, because that path is already taken by the unrelated
per-recommendation narrator used by the Decision pipeline
(`stock_scanner/ai_lab/agents/hypothesis_agent.py`'s `build_evidence()`/
`generate_hypothesis()`). Same contract as `reflection_agent.py`: one call per run,
`HypothesisNarrativeOutput` has zero numeric or identifier-inventing fields
(`overall_summary`, `prioritized_hypothesis_ids`, `hypothesis_notes`, plus the one new
responsibility this stage adds — `clusters`: grouping `hypothesis_id`s that describe
the same underlying finding from different angles, a narrative grouping only, never a
new statistical claim), and the statistical output is always published regardless of
whether the LLM call succeeds.

**Storage** — `validated_hypotheses` (append-only, mirrors `reflection_observations`'
shape). Stores BOTH validated and rejected hypotheses (`status` column) — one table
serves the dashboard's Candidate (unfiltered) / Validated / Rejected filters, no
separate rejected-hypotheses table.

**Isolation** — same guarantees as the rest of AI Lab: `validated_hypotheses` has no
FKs to `signals`/`outcomes`/`model_registry`/`knowledge_base`, nothing in
`stock_scanner/pipeline/`/`stock_scanner/alerts/` reads it, and
`dashboard/hypothesis_view.py` only reads the committed JSON mirror (never imports
`hypothesis_engine.py`/`statistical_validation.py`/`client.py`/`agents/`).

## Knowledge Base Engine

Sixth component, and the current end of the closed learning loop: `Performance Tracker
→ Reflection Engine → Hypothesis Generator → Statistical Validation → Knowledge Base
Engine → (future) Calibration Engine`. **Not another statistical engine** — no new
Fisher/Wilson/BH math here. It curates `validated_hypotheses` across ALL historical
runs into long-lived `knowledge_entries`, each with a deterministic lifecycle
(`emerging → confirmed → strong`, or `→ weakening → contradicted → archived`).

**Naming collisions found and resolved during planning** — asked of, and decided by,
the user: the spec's table name `knowledge_base` and the natural module name
`knowledge_base.py` both collide with **pre-existing, unrelated** files —
`stock_scanner/db/knowledge_base.py` (Learning Agent Phase 1's human-reviewed
hypothesis store, a completely different table with its own
`candidate`/`reviewed`/.../`promoted` lifecycle) and `dashboard/knowledge_base_view.py`
(that table's dashboard view). The new table/module/view are **`knowledge_entries`**
(`stock_scanner/db/knowledge_entries.py`, `dashboard/knowledge_entries_view.py`)
instead — chosen by the user, applied consistently everywhere this phase touches.
Nothing in this pass reads or writes the production `knowledge_base` table.

**"Deterministic similarity based on normalized conditions," not LLM clustering.**
`stock_scanner/ai_lab/hypothesis_engine.py` already normalizes every hypothesis's
`conditions` into a canonical sorted list at generation time — unlike
`stock_scanner.learning.pattern_dedup`'s problem (raw signal-id-set overlap producing
genuinely different-looking candidate rows for the same underlying event), there is no
"RSI>70 vs RSI High vs RSI Top Quartile" divergence to reconcile in this system: every
run that rediscovers the same idea produces the *exact same* normalized condition-set.
So similarity here is **exact equality on the normalized condition-set** — the
simplest, strictest form, no threshold tuning, no reuse of `pattern_dedup.py`'s
Jaccard/overlap machinery (same scope boundary already drawn in the Hypothesis
Generator phase — that tool solves a different problem this system doesn't have).
Because `validated_hypotheses` is append-only, the same condition-set naturally recurs
as a new row every time a later run re-validates it — grouping by that exact equality
across runs **is** the accumulation mechanism.

**"Never strengthen knowledge from duplicate evidence."** Each Hypothesis Generator run
re-scores a condition-set against the *entire current* resolved-recommendations
population, not an incremental slice — so summing `sample_size`/`successes` across
every historical row for a condition-set would count the same underlying trades many
times over. Instead: `cumulative_sample_size`/`successes`/`failures`,
`confidence_interval`, `shrunk_win_rate`, and `evidence_strength` all come from the
**latest confirming row only** (a reused snapshot, never re-derived);
`average_win_rate` **is** a genuine average of `win_rate` across every independent
confirming run (a ratio, not a raw count, so averaging doesn't double-count);
`confirmation_count`/`contradiction_count` count distinct validation runs, never raw
trades.

**Algorithm** (`stock_scanner/ai_lab/knowledge_base_engine.py`,
`generate_knowledge_entries()`) — replays the *entire* historical `validated_hypotheses`
population from scratch on every call, not an incremental update against a prior
snapshot (this is what makes it trivially deterministic/reproducible: identical input
always produces an identical entry sequence). Groups all rows by normalized
condition-set; a group with no `status='validated'` row ever is skipped (nothing to
know yet). Walks each group in `created_at` order: the first validated row with a
determinable direction (`win_rate` vs. that run's `metadata_json.baseline_rate` — one
small, additive field added to `statistical_validation.py`'s existing per-row
`metadata_json` this pass) establishes `established_direction`, which **never flips
again** — matching the spec's own "Energy + Breakout = Positive [then] = Negative"
example: the entry doesn't become a "Negative" entry, it becomes an increasingly
contradicted "Positive" one. Lifecycle ladder, integer thresholds only:

```
contradiction_count >= confirmation_count + archive_margin(3)  -> ARCHIVED
contradiction_count >= confirmation_count and > 0               -> CONTRADICTED
contradiction_count > 0                                          -> WEAKENING
confirmation_count >= strong_threshold(5)                        -> STRONG
confirmation_count >= 2                                          -> CONFIRMED
else (confirmation_count == 1)                                   -> EMERGING
```

`strong_threshold=5` is the spec's own worked example ("Validation #5 → Strong
Knowledge"), kept literal. `EMERGING` is only reachable at `confirmation_count==1` — a
distinct, real first-class status (the spec lists six), not a dead state; the worked
example's diagram jumping straight to "Confirmed" after "Validation #1" is read as an
illustrative simplification of the strengthening trajectory, not a literal state
enumeration. Both thresholds are CLI flags (`--strong-threshold`/`--archive-margin`),
not YAML. `previous_lifecycle_status` replays the same ladder one evidence-point
earlier — a real, code-computed "did this change" signal (not inferred) that serves
both the dashboard's Recent Changes section and the LLM prompt's "highlight important
changes" context from one computation.

**LLM layer — summarize/explain/organize/highlight, code-only fallback.**
`stock_scanner/ai_lab/agents/knowledge_review_agent.py` (not `knowledge_agent.py` or
anything colliding — same naming care as above), same contract as
`reflection_agent.py`/`hypothesis_review_agent.py`: one call per run, curated entries
always published regardless of LLM outcome. `KnowledgeNarrativeOutput` maps 1:1 onto
the four allowed responsibilities: summarize → `overall_summary`; explain →
`knowledge_notes`; organize → `organized_groups` (thematic labeling, not a merge — the
deterministic engine already decided what's "the same belief"); highlight important
changes → `highlighted_changes` (computed by the caller from
`previous_lifecycle_status != lifecycle_status`, never left for the LLM to detect
itself). Zero numeric fields, same discipline as every other AI Lab LLM schema — the
LLM never computes confidence, creates knowledge, changes a lifecycle_status, or merges
entries.

**Storage** — `knowledge_entries` (append-only, mirrors `validated_hypotheses`' shape;
`knowledge_id` hashes in *this curation run's own* `created_at`, distinct from
`first_seen`/`last_confirmed` which track the underlying evidence trail — giving the
"Knowledge Timeline" dashboard section a real history of how each belief's status
evolved across runs, never overwritten).

**Isolation** — same guarantees as the rest of AI Lab: `knowledge_entries` has no FKs
to `signals`/`outcomes`/`model_registry`/the production `knowledge_base` table, nothing
in `stock_scanner/pipeline/`/`stock_scanner/alerts/` reads it, and
`dashboard/knowledge_entries_view.py` only reads the committed JSON mirror (never
imports `knowledge_base_engine.py`/`client.py`/`agents/`).

## What's phased for later (explicitly out of scope for this pass)

- **Live 9router wiring**: needs the real base URL/contract confirmed (see above).
- **Scheduled execution**: none of `scripts/run_ai_lab.py` (generation),
  `scripts/resolve_ai_lab.py` (resolution), `scripts/run_reflection_engine.py`
  (reflection), `scripts/run_hypothesis_engine.py` (hypothesis generation +
  validation), or `scripts/run_knowledge_base_engine.py` (curation) has a GitHub
  Actions workflow yet — deliberately manual-only until a live generation run has been
  reviewed at least once; scheduling a downstream stage before the stage before it is
  itself scheduled would be automating half a loop.
- **Auto Promotion Engine** (see below): architecture prepared, not implemented.
- **Promotion into the production `knowledge_base` table**: `knowledge_entries`
  (`lifecycle_status='strong'` rows) is a candidate pool only — nothing writes to the
  production `knowledge_base` table yet, and when that path is built it stays
  human-review-gated, same as every other promotion path in this repo.
- **Calibration Engine / Human Review / Production Proposal**: the remaining stages in
  the target architecture, not started.
- **Fuzzy/semantic duplicate detection for hypotheses or knowledge entries**:
  `hypothesis_engine.py` only dedupes exact condition-sets at generation time, and
  `knowledge_base_engine.py` only merges by exact normalized condition-set equality;
  near-duplicate findings that overlap heavily in which recommendations they cover but
  aren't literally the same condition-set (the problem `pattern_dedup.py`'s
  Jaccard/overlap clustering solves for production patterns) aren't merged — the LLM's
  `clusters`/`organized_groups` narrative grouping is the only mitigation today, and
  it's advisory, not a statistical dedup.
- **Smarter learning-timeline detection**: today `run_ai_lab.py` logs one honest,
  count-based `hypothesis_generated` event per run, `resolve_ai_lab.py` logs one
  honest, count-based `outcome_resolved` event per run, `run_reflection_engine.py` logs
  one `reflection_generated` event per run, `run_hypothesis_engine.py` logs one
  `hypothesis_validated` event per run, and `run_knowledge_base_engine.py` logs one
  `knowledge_base_updated` event per run (deliberately not reusing the existing
  `hypothesis_generated` event type — that one already means "AI Lab generated
  recommendations," a different thing). Detecting genuinely new production
  `knowledge_base` patterns since the last run (for real `pattern_learned` events) or
  confidence drift (`confidence_updated`) is a natural follow-up, not yet built — the
  schema (`ai_learning_events.event_type`) already supports it.

## Future Ready — Auto Promotion Engine (not implemented)

A future module could compare AI Lab's per-`ai_model` performance
(`compute_performance(df, ai_model=...)`) against the Production Scanner's own
realized win rate (`stock_scanner.pipeline.evaluator.summarize_evaluations`) and
recommend promoting a model — the same two-sided comparison
`scripts/promote_challenger.py` already does for the ML ranker, extended to AI Lab's
models. The schema is already shaped for this: `ai_recommendations.ai_model` +
`.status` + `.return_percentage` (now consistently populated by the resolver, see
"Recommendation lifecycle") give per-model, per-recommendation resolved returns with
no further migration needed; `ai_learning_events` gives an audit trail. Nothing in this
pass writes to `model_registry` or `promotion_decisions`, and no comparison logic
exists yet — this section is architecture notes only, per the product spec's explicit
"do NOT implement auto-promotion now."

## Future Ready — Calibration Engine (not implemented)

`reflection_observations` (`CONFIDENCE_CALIBRATION` category rows specifically —
`avg_confidence`/`win_rate`/`ci_lower`/`ci_upper`/`calibration_issue`) and
`validated_hypotheses` are the designed input surface for a future Calibration Engine,
the same way `ai_recommendations` is Auto Promotion Engine's. It could re-weight
`scoring.WEIGHTS` based on which `overconfident`/`underconfident` findings recur across
runs. Nothing in this pass writes to `scoring.py` — this section, like Auto Promotion
Engine's, is architecture notes only.

## Future Ready — Promotion into the production `knowledge_base` table (not implemented)

`knowledge_entries` (`lifecycle_status='strong'` rows especially —
`conditions`/`average_win_rate`/`shrunk_win_rate`/`confidence_interval`/
`confirmation_count`/`contradiction_count`/`evidence_strength`) is now the actual,
implemented candidate pool this repo has for a future promotion path into the
production `knowledge_base` table (this section previously anticipated
`validated_hypotheses` filling this role directly — Knowledge Base Engine, built this
pass, is the curation layer that makes a *durable, multi-run-confirmed* belief the
right unit of promotion instead of a single run's hypothesis). The shape maps closely
enough onto what `stock_scanner/learning/hypothesis_agent.py` already writes there
(`hypothesis_id`/`confidence`/`supporting_trades`/`affected_sector`/`pattern_json`/
`status`) that a future promotion script could map one onto the other field-for-field.
Any such promotion would go through the exact same human-review gate
`knowledge_base.update_status()` already enforces for Learning Agent Phase 1's own
hypotheses — never automatic, and nothing in this pass writes to the production
`knowledge_base` table at all.

## Runbook

```bash
# Exercise the full pipeline with no live API calls (deterministic):
python scripts/run_ai_lab.py --mock

# Restrict to specific models / candidate count:
python scripts/run_ai_lab.py --mock --top-n 10 --models momentum_ai,breakout_ai

# Live (once NINEROUTER_BASE_URL/API_KEY/MODEL are confirmed and exported):
python scripts/run_ai_lab.py

# Resolve PENDING -> ACTIVE -> CLOSED/EXPIRED against forward OHLCV data —
# run this after new raw price data lands (see "Recommendation lifecycle").
# Lifecycle + trade analytics are now handled automatically; the old manual
# update_recommendation_status() call below is only needed for a one-off
# manual override, not routine resolution:
python scripts/resolve_ai_lab.py
python scripts/resolve_ai_lab.py --horizon-days 15 --risk-pct 2.5

# Manual override (rare — routine resolution is scripts/resolve_ai_lab.py above):
python -c "
from stock_scanner.db.init_db import get_connection
from stock_scanner.db.ai_lab import update_recommendation_status, export_ai_recommendations
conn = get_connection()
update_recommendation_status(conn, '<id>', 'ACTIVE', entry_price=1234.0)
export_ai_recommendations(conn)
"

# Review RESOLVED recommendations and publish reflection_report.json —
# run this after scripts/resolve_ai_lab.py has closed/expired some
# positions (an empty observations list is expected/correct until there's
# enough resolved history to clear the statistical gate):
python scripts/run_reflection_engine.py --mock
python scripts/run_reflection_engine.py --min-n-success 5 --alpha 0.1

# Refine reflection_observations into multi-condition validated_hypotheses
# and publish hypotheses_report.json — run this after
# scripts/run_reflection_engine.py has produced at least one gated
# observation (no observations = no seeds = an empty hypotheses report,
# expected/correct):
python scripts/run_hypothesis_engine.py --mock
python scripts/run_hypothesis_engine.py --min-n-success 5 --alpha 0.1 --max-order 2

# Curate ALL historical validated_hypotheses into long-lived knowledge_entries
# and publish knowledge_report.json — run this after
# scripts/run_hypothesis_engine.py has produced at least one validated
# hypothesis (no validated rows = nothing to curate = an empty knowledge
# report, expected/correct). Safe to re-run anytime — it always replays
# the full history from scratch, never an incremental update:
python scripts/run_knowledge_base_engine.py --mock
python scripts/run_knowledge_base_engine.py --strong-threshold 3 --archive-margin 2
```

Tests: `pytest tests/test_ai_lab_*.py tests/test_db_reflection.py tests/test_db_hypotheses.py tests/test_db_knowledge_entries.py`
(schemas, client retry/config/parsing, agents evidence-matching, DB
upsert-idempotency/lifecycle-preservation, performance metrics, resolver
activation/TP/SL/expiry/trade-analytics, reflection engine gating/determinism,
reflection agent narrative, reflection DB round-trip, hypothesis engine candidate
generation/seeding/duplicate-prevention/determinism, statistical validation
Fisher/Wilson/BH correctness and validated-vs-rejected classification, hypothesis
review agent narrative, hypotheses DB round-trip, knowledge base engine deterministic
merging/contradiction handling/lifecycle transitions, knowledge review agent
narrative, knowledge_entries DB round-trip).
