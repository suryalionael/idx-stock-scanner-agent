# AI Lab — Architecture

**Status:** storage + agents + client + dashboard implemented, tested, and verified
end-to-end against the live 9router endpoint (base URL/auth/request format confirmed
— see "9router configuration status"). Explainability upgrade (decision trace,
confidence breakdown, historical comparison, rule-based recommendation levels — see
"Explainability upgrade" below) is live.
**Not yet done:** no scheduled GitHub Actions workflow — `scripts/run_ai_lab.py` is
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
  agents/
    hypothesis_agent.py  — build_evidence() + generate_hypothesis()
    decision_agent.py    — generate_decision() + assemble_recommendation() + recommendation_id()

stock_scanner/db/ai_lab.py   — SQLite IO: upsert/update/load/export/import for both tables
stock_scanner/db/schema.sql  — ai_recommendations + ai_learning_events (appended, not modified existing tables)
scripts/run_ai_lab.py        — manual orchestrator (--mock or live)
dashboard/ai_lab_view.py     — read-only dashboard tab (5 sections, see spec)
dashboard/data_loader.py     — load_ai_recommendations_payload / load_ai_learning_events_payload
data/published/ai_recommendations.json, ai_learning_events.json — committed mirrors
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
`exit_price`/`return_percentage` on conflict — those are lifecycle fields owned only
by `update_recommendation_status()`. This is what makes "never overwrite history, append
only" true in practice: content can be regenerated, lifecycle progress cannot be reset.

## What's phased for later (explicitly out of scope for this pass)

- **Live 9router wiring**: needs the real base URL/contract confirmed (see above).
- **Scheduled execution**: `scripts/run_ai_lab.py` has no GitHub Actions workflow yet
  — deliberately manual-only until a live run has been reviewed at least once.
- **Auto Promotion Engine** (see below): architecture prepared, not implemented.
- **Smarter learning-timeline detection**: today `run_ai_lab.py` logs one honest,
  count-based `hypothesis_generated` event per run. Detecting genuinely new
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
`.status` + `.return_percentage` give per-model, per-recommendation resolved returns
with no further migration needed; `ai_learning_events` gives an audit trail. Nothing
in this pass writes to `model_registry` or `promotion_decisions`, and no comparison
logic exists yet — this section is architecture notes only, per the product spec's
explicit "do NOT implement auto-promotion now."

## Runbook

```bash
# Exercise the full pipeline with no live API calls (deterministic):
python scripts/run_ai_lab.py --mock

# Restrict to specific models / candidate count:
python scripts/run_ai_lab.py --mock --top-n 10 --models momentum_ai,breakout_ai

# Live (once NINEROUTER_BASE_URL/API_KEY/MODEL are confirmed and exported):
python scripts/run_ai_lab.py

# Advance a recommendation's lifecycle manually (no CLI yet — same as
# review_knowledge_base.py's precedent, a small CLI could be added if this
# becomes a frequent manual task):
python -c "
from stock_scanner.db.init_db import get_connection
from stock_scanner.db.ai_lab import update_recommendation_status, export_ai_recommendations
conn = get_connection()
update_recommendation_status(conn, '<id>', 'ACTIVE', entry_price=1234.0)
export_ai_recommendations(conn)
"
```

Tests: `pytest tests/test_ai_lab_*.py` (schemas, client retry/config/parsing, agents
evidence-matching, DB upsert-idempotency/lifecycle-preservation, performance metrics).
