-- Self-improving signal system — SQLite schema.
-- See docs/SELF_IMPROVING_ARCHITECTURE.md for the full design rationale.
--
-- Apply with: sqlite3 data/db/signals.db < stock_scanner/db/schema.sql
-- (or via stock_scanner.db.init_db.create_schema(), which runs this file)

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Core signal lifecycle
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS signals (
    signal_id        TEXT PRIMARY KEY,
    ticker           TEXT NOT NULL,
    signal_date      DATE NOT NULL,
    strategy         TEXT NOT NULL,
    signal_label     TEXT NOT NULL,
    total_score      REAL,
    ml_prob          REAL,
    model_version_id TEXT,
    scan_run_id      TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, signal_date, strategy)
);
CREATE INDEX IF NOT EXISTS idx_signals_date   ON signals(signal_date);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    signal_id            TEXT PRIMARY KEY REFERENCES signals(signal_id),
    feature_set_version  TEXT NOT NULL,
    features_json        TEXT NOT NULL,
    raw_close            REAL,
    raw_open             REAL,
    raw_volume           REAL,
    snapshot_source_path TEXT
);

CREATE TABLE IF NOT EXISTS outcomes (
    signal_id      TEXT PRIMARY KEY REFERENCES signals(signal_id),
    eval_date      DATE,
    status         TEXT NOT NULL DEFAULT 'pending',
    prev_close     REAL,
    eval_open      REAL,
    eval_high      REAL,
    eval_close     REAL,
    pct_high       REAL,
    pct_close      REAL,
    wl             TEXT,
    label_success  INTEGER,
    labeled_at     TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_outcomes_status    ON outcomes(status);
CREATE INDEX IF NOT EXISTS idx_outcomes_eval_date ON outcomes(eval_date);

-- ---------------------------------------------------------------------------
-- Context (market regime, sector, broker)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS market_context (
    context_date    DATE PRIMARY KEY,
    ihsg_close      REAL,
    ihsg_pct_change REAL,
    ihsg_trend_5d   REAL,
    ihsg_trend_20d  REAL,
    regime_label    TEXT
);

CREATE TABLE IF NOT EXISTS sector_reference (
    ticker       TEXT PRIMARY KEY,
    company_name TEXT,
    sector       TEXT
);

-- Coverage is currently too sparse (62 tickers, sparse dates — see leakage
-- audit) to use for training. Table exists so no schema change is needed
-- once data/broker/ coverage improves. Do not join this into
-- training_examples yet.
CREATE TABLE IF NOT EXISTS broker_context (
    ticker        TEXT NOT NULL,
    context_date  DATE NOT NULL,
    net_lot_top10 REAL,
    PRIMARY KEY (ticker, context_date)
);

-- ---------------------------------------------------------------------------
-- Daily movers >10% (non-production, standalone feature — see
-- stock_scanner/pipeline/daily_movers.py, scripts/build_daily_movers.py).
-- Not read by signal_engine.py, ml_ranker.py, or any promotion path. Rows
-- are stored only for (trade_date, ticker) pairs that actually hit one of
-- the two >10% definitions.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS daily_movers (
    trade_date          DATE NOT NULL,
    ticker              TEXT NOT NULL,
    prev_close          REAL,
    open                REAL,
    high                REAL,
    low                 REAL,
    close               REAL,
    volume              REAL,
    pct_change_close    REAL,
    pct_change_high     REAL,
    hit_10pct_close     INTEGER NOT NULL DEFAULT 0,
    hit_10pct_intraday  INTEGER NOT NULL DEFAULT 0,
    source              TEXT,
    inserted_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trade_date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_daily_movers_date ON daily_movers(trade_date);

-- ---------------------------------------------------------------------------
-- Top signals >10% (non-production, standalone daily persistence — see
-- stock_scanner/pipeline/top_signals.py, scripts/build_top_signals.py).
-- Deliberately NOT the knowledge_base table (Learning Agent Phase 1) — this
-- is a plain filtered/ranked mirror of already-evaluated signal outcomes,
-- with no hypothesis-generation or LLM step. Not read by signal_engine.py,
-- ml_ranker.py, or any promotion path. signal_id is deterministic
-- (stock_scanner.db.init_db.signal_id) but there is NO foreign key to
-- signals(signal_id) — this table is built directly and independently from
-- the always-committed data/performance/signal_results.csv, so it never
-- depends on the signals/outcomes tables being populated first.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS top_signals (
    signal_id               TEXT PRIMARY KEY,
    ticker                  TEXT NOT NULL,
    strategy                TEXT NOT NULL,
    signal_date             DATE NOT NULL,
    eval_date               DATE NOT NULL,
    signal_label            TEXT,
    prev_close              REAL,
    eval_close              REAL,
    eval_high               REAL,
    pct_close               REAL,
    pct_high                REAL,
    forward_return_pct      REAL NOT NULL,
    quality_adjusted_score  REAL,
    total_score             REAL,
    enhanced_total_score    REAL,
    ml_prob                 REAL,
    quality_source          TEXT NOT NULL DEFAULT 'unavailable',
    rank_in_day             INTEGER NOT NULL,
    filter_threshold_pct    REAL NOT NULL DEFAULT 10.0,
    source_run_id           TEXT,
    computed_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_top_signals_eval_date ON top_signals(eval_date);
CREATE INDEX IF NOT EXISTS idx_top_signals_return    ON top_signals(forward_return_pct DESC);

-- ---------------------------------------------------------------------------
-- Model lifecycle
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS model_registry (
    model_version_id    TEXT PRIMARY KEY,
    model_type          TEXT NOT NULL,
    feature_list_json   TEXT NOT NULL,
    train_start_date    DATE,
    train_end_date      DATE,
    val_start_date       DATE,
    val_end_date         DATE,
    test_start_date      DATE,
    test_end_date        DATE,
    label_threshold_pct  REAL,
    label_horizon        TEXT,
    train_metrics_json   TEXT,
    val_metrics_json     TEXT,
    test_metrics_json    TEXT,
    sensitivity_json      TEXT,
    artifact_path          TEXT,
    status                 TEXT NOT NULL DEFAULT 'candidate',
    promoted_at             TIMESTAMP,
    retired_at              TIMESTAMP,
    trained_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS promotion_decisions (
    decision_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    challenger_model_id     TEXT REFERENCES model_registry(model_version_id),
    production_model_id     TEXT REFERENCES model_registry(model_version_id),
    decision                TEXT NOT NULL,
    challenger_metrics_json TEXT,
    production_metrics_json TEXT,
    reason                  TEXT,
    decided_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS live_monitoring (
    monitor_date                    DATE PRIMARY KEY,
    production_model_id             TEXT REFERENCES model_registry(model_version_id),
    n_signals                       INTEGER,
    n_evaluated                     INTEGER,
    realized_win_rate               REAL,
    realized_precision_at_threshold REAL,
    avg_predicted_prob              REAL,
    feature_drift_flag              INTEGER,
    alert_triggered                 INTEGER
);

-- ---------------------------------------------------------------------------
-- Learning Agent (research-only — see docs/LEARNING_AGENT_ARCHITECTURE.md)
-- ---------------------------------------------------------------------------

-- LLM-articulated hypotheses over statistically-gated, de-duplicated
-- pattern clusters (stock_scanner.learning.pattern_miner / pattern_dedup).
-- Read-only research output: nothing in stock_scanner/pipeline/ reads this
-- table, and status never leaves 'candidate' automatically — promotion to
-- production still requires a human running scripts/train_challenger.py +
-- scripts/promote_challenger.py, unchanged.
CREATE TABLE IF NOT EXISTS knowledge_base (
    hypothesis_id           TEXT PRIMARY KEY,   -- sha1(hypothesis || generated_at)
    generated_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hypothesis               TEXT NOT NULL,
    confidence               REAL,               -- LLM's qualitative framing, NOT a p-value
    supporting_trades        INTEGER,
    affected_sector          TEXT,
    affected_dimension       TEXT,
    pattern_json             TEXT NOT NULL,       -- source ClusteredPattern, full audit trail
    expected_effect          TEXT,
    status                   TEXT NOT NULL DEFAULT 'candidate',
        -- 'candidate' | 'reviewed' | 'testing' | 'tested_passed' | 'tested_failed' | 'promoted' | 'archived'
    reviewed_by              TEXT,
    linked_model_version_id  TEXT REFERENCES model_registry(model_version_id),
    source_run_id            TEXT
);
CREATE INDEX IF NOT EXISTS idx_knowledge_base_status ON knowledge_base(status);

-- ---------------------------------------------------------------------------
-- AI Lab (experimental — see stock_scanner/ai_lab/, docs/AI_LAB_ARCHITECTURE.md)
--
-- Completely separate recommendation engine, isolated from the Production
-- Scanner: nothing in stock_scanner/pipeline/ or stock_scanner/alerts/
-- reads these tables, and nothing here writes to signals/outcomes/
-- knowledge_base/model_registry. AI Lab consumes Learning Agent Phase 1's
-- already-validated statistics (knowledge_base rows) as evidence, but is a
-- distinct output: a ranked, lifecycle-tracked recommendation, not a
-- research narrative. No auto-promotion path exists yet — see the "Future
-- Ready" section of docs/AI_LAB_ARCHITECTURE.md for the architecture a
-- later Auto Promotion Engine would plug into (this schema's per-ai_model
-- grouping + status/return_percentage columns are exactly what it would
-- read; nothing further is implemented now).
--
-- Lifecycle + tracking fields (status, entry/exit_price, return_percentage,
-- highest/lowest_price, max_runup_pct/max_drawdown_pct, holding_days,
-- trade_outcome) are written exclusively by stock_scanner.ai_lab.resolver /
-- scripts/resolve_ai_lab.py — never by the generation pipeline
-- (scripts/run_ai_lab.py), which only ever INSERTs a row or updates its
-- content fields on conflict (see upsert_recommendations). The tracking
-- fields are the designed input surface for the future Calibration Engine
-- and Self Reflection stages (see docs/AI_LAB_ARCHITECTURE.md).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ai_recommendations (
    id                   TEXT PRIMARY KEY,   -- sha1(ticker|ai_model|generated_date)[:16] — deterministic, one row per ticker/model/day
    ticker               TEXT NOT NULL,
    ai_model             TEXT NOT NULL,       -- 'momentum_ai' | 'breakout_ai' | 'reversal_ai' | 'volume_ai' | ... (plug-and-play, see ai_lab/models.py)
    score                REAL NOT NULL,        -- 0..100, always == decision_trace_json.final_score (see stock_scanner/ai_lab/scoring.py)
    confidence           REAL NOT NULL,        -- 0..1, always == confidence_breakdown_json.final_confidence
    recommendation       TEXT NOT NULL,        -- 'STRONG_BUY' | 'BUY' | 'WATCH' | 'AVOID' — rule-based, see scoring.classify_recommendation_level
    reasoning            TEXT NOT NULL,        -- JSON: {why, technical_indicators, statistical_evidence,
                                                --        similar_patterns, best_pattern_similarity_pct,
                                                --        strengths, weaknesses, risks, reasoning_summary,
                                                --        historical_comparison_explanation, confidence_explanation}
    decision_trace        TEXT NOT NULL,        -- JSON: {technical_score, statistical_score, pattern_similarity_score, risk_score, final_score} — explainability upgrade
    confidence_breakdown  TEXT NOT NULL,        -- JSON: {technical, statistical, pattern_similarity, risk_adjustment, final_confidence}
    historical_comparison TEXT NOT NULL,        -- JSON: {pattern_description, sample_size, win_rate, ci_lower, ci_upper, verdict, explanation}
    expected_return       REAL,                 -- always NULL currently — see scoring.compute_expected_return's docstring for why
    risk_level            TEXT,                -- 'LOW' | 'MEDIUM' | 'HIGH' — rule-based, see scoring.classify_risk_level
    generated_date        DATE NOT NULL,
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status                  TEXT NOT NULL DEFAULT 'PENDING',  -- 'PENDING' | 'ACTIVE' | 'CLOSED' | 'EXPIRED'
    entry_price              REAL,
    exit_price                REAL,
    return_percentage          REAL,
    highest_price             REAL,               -- running max(high) since entry — updated every resolver pass, frozen once resolved
    lowest_price               REAL,               -- running min(low) since entry
    max_runup_pct               REAL,               -- MFE: (highest_price - entry_price) / entry_price * 100, clamped >= 0
    max_drawdown_pct             REAL,               -- MAE: (entry_price - lowest_price) / entry_price * 100, clamped >= 0
    holding_days                  INTEGER,            -- trading days elapsed since entry; keeps climbing while ACTIVE, freezes on resolution
    trade_outcome                  TEXT,               -- 'WIN' | 'LOSS' | 'BREAKEVEN' — set only once resolved; independent of status (an EXPIRED mark-to-market row can still be a WIN)
    model                       TEXT NOT NULL      -- 9router model string used for this recommendation (e.g. 'deepseek-v4-flash-free'); NEVER hardcoded, always the configured NINEROUTER_MODEL at generation time
);
CREATE INDEX IF NOT EXISTS idx_ai_recommendations_date       ON ai_recommendations(generated_date);
CREATE INDEX IF NOT EXISTS idx_ai_recommendations_ai_model   ON ai_recommendations(ai_model);
CREATE INDEX IF NOT EXISTS idx_ai_recommendations_status     ON ai_recommendations(status);

-- AI Learning Timeline (dashboard section 5) — append-only log of
-- observable AI Lab events (a hypothesis was generated, a model's
-- confidence shifted, accuracy improved after a performance recompute).
-- Purely descriptive/informational; nothing reads this table to make a
-- decision — it exists only to render the timeline.
CREATE TABLE IF NOT EXISTS ai_learning_events (
    event_id       TEXT PRIMARY KEY,   -- sha1(ai_model|event_type|description|created_at)[:16]
    ai_model       TEXT,
    event_type     TEXT NOT NULL,       -- 'pattern_learned' | 'confidence_updated' | 'hypothesis_generated' | 'accuracy_improved' | 'outcome_resolved' | 'reflection_generated' | 'hypothesis_validated' | 'knowledge_base_updated'
    description    TEXT NOT NULL,
    metadata_json  TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ai_learning_events_created_at ON ai_learning_events(created_at);

-- Reflection Engine (stock_scanner/ai_lab/reflection_engine.py, see
-- docs/AI_LAB_ARCHITECTURE.md "Reflection Engine") — statistically gated
-- observations over RESOLVED ai_recommendations rows only (status IN
-- ('CLOSED','EXPIRED')). Isolated the same way as the rest of AI Lab:
-- nothing here reads/writes signals/outcomes/knowledge_base/
-- model_registry, and nothing in stock_scanner/pipeline/ or
-- stock_scanner/alerts/ reads this table. Append-only, not upsert-with-
-- lifecycle like ai_recommendations: each reflection run inserts new rows
-- (observation_id is unique per run since it hashes generated_at), so this
-- table doubles as a "Recent Reflections" timeline, not just a latest
-- snapshot. Designed as the primary input surface for the future
-- Calibration Engine / Hypothesis Generator / Statistical Validation
-- stages — nothing reads this table to make an automated decision yet.
CREATE TABLE IF NOT EXISTS reflection_observations (
    observation_id        TEXT PRIMARY KEY,   -- sha1(category|dimension|value|generated_at)[:16]
    category               TEXT NOT NULL,       -- 'model_performance' | 'sector_performance' | 'recommendation_level_performance' | 'historical_verdict_accuracy' | 'technical_pattern' | 'confidence_calibration'
    title                    TEXT NOT NULL,
    description               TEXT NOT NULL,
    supporting_statistics      TEXT NOT NULL,      -- JSON: {n, n_success, win_rate, win_rate_shrunk, baseline_rate, ci_lower, ci_upper, p_value_adjusted, avg_return_percentage, avg_holding_days, ...}
    affected_trade_count         INTEGER NOT NULL,
    confidence                     REAL NOT NULL,      -- 1 - p_value_adjusted, clamped [0,1] — statistical confidence this is real, not noise; distinct from ai_recommendations.confidence
    llm_note                        TEXT,               -- optional one-sentence LLM explanation (stock_scanner.ai_lab.agents.reflection_agent) — never a source of new numbers
    generated_at                      TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reflection_observations_category ON reflection_observations(category);
CREATE INDEX IF NOT EXISTS idx_reflection_observations_generated_at ON reflection_observations(generated_at);

-- Hypothesis Generator + Statistical Validation
-- (stock_scanner/ai_lab/hypothesis_engine.py + statistical_validation.py,
-- see docs/AI_LAB_ARCHITECTURE.md "Hypothesis Generator + Statistical
-- Validation") — refines reflection_observations' single-dimension
-- findings into multi-condition (order 2-3), statistically validated
-- hypotheses, seeded from reflection_observations (apriori expansion, not
-- brute-force). Isolated the same way as the rest of AI Lab: no FKs
-- to/from signals/outcomes/knowledge_base/model_registry, nothing in
-- stock_scanner/pipeline/ or stock_scanner/alerts/ reads this table.
-- Append-only, same reasoning as reflection_observations: hypothesis_id
-- hashes in created_at, so the same condition-set re-validated on a
-- larger resolved-trade population later is a new row, not an overwrite.
-- Stores BOTH validated and rejected hypotheses (status column) — one
-- table serves all three dashboard filters (candidate/validated/
-- rejected), no separate rejected-hypotheses table needed. Designed
-- candidate pool for a future (not implemented here) Knowledge Base
-- promotion path: only status='validated' rows would ever be eligible,
-- and even then only via human review, never automatically.
--
-- NOTE: ai_learning_events.event_type='hypothesis_generated' (above)
-- predates this table and means something different — AI Lab
-- recommendation generation (scripts/run_ai_lab.py), not statistical
-- hypothesis validation. This pipeline logs 'hypothesis_validated'
-- instead to avoid colliding with that existing meaning.
CREATE TABLE IF NOT EXISTS validated_hypotheses (
    hypothesis_id            TEXT PRIMARY KEY,   -- sha1(sorted_conditions|created_at)[:16]
    created_at                TIMESTAMP NOT NULL,
    description                 TEXT NOT NULL,
    conditions                   TEXT NOT NULL,      -- JSON: [[dimension, value], ...], order 2-3
    sample_size                    INTEGER NOT NULL,
    successes                        INTEGER NOT NULL,
    failures                           INTEGER NOT NULL,
    win_rate                            REAL NOT NULL,
    shrunk_win_rate                       REAL NOT NULL,
    wilson_lower                            REAL NOT NULL,
    wilson_upper                              REAL NOT NULL,
    fisher_p                                    REAL NOT NULL,
    bh_adjusted_p                                 REAL NOT NULL,
    evidence_strength                               TEXT,               -- 'STRONG' | 'MODERATE' — validated rows only, NULL for rejected
    status                                            TEXT NOT NULL,      -- 'validated' | 'rejected'
    rejection_reason                                    TEXT,               -- rejected rows only
    failed_gate                                           TEXT,               -- 'not_significant' | 'no_directional_lift' — rejected rows only
    source_reflection_ids                                   TEXT,               -- JSON list — reflection_observations.observation_id this hypothesis was seeded/expanded from
    metadata_json                                             TEXT,               -- JSON: {avg_return_percentage, avg_holding_days, interaction_order}
    llm_note                                                    TEXT                -- optional one-sentence LLM explanation (hypothesis_review_agent) — never a source of new numbers
);
CREATE INDEX IF NOT EXISTS idx_validated_hypotheses_status ON validated_hypotheses(status);
CREATE INDEX IF NOT EXISTS idx_validated_hypotheses_created_at ON validated_hypotheses(created_at);

-- Knowledge Base Engine (stock_scanner/ai_lab/knowledge_base_engine.py,
-- see docs/AI_LAB_ARCHITECTURE.md "Knowledge Base Engine") — curates
-- validated_hypotheses across ALL historical runs into long-lived
-- KnowledgeEntry rows with a deterministic lifecycle (emerging/confirmed/
-- strong/weakening/contradicted/archived). NOT another statistical
-- engine — no new Fisher/Wilson/BH math here, only aggregation over what
-- statistical_validation.py already decided.
--
-- NOTE: this table is named `knowledge_entries`, deliberately NOT
-- `knowledge_base` — that name already belongs to a completely different,
-- pre-existing table (above, this file) written by
-- stock_scanner/db/knowledge_base.py / stock_scanner/learning/ (Learning
-- Agent Phase 1's human-reviewed hypothesis store, with its own
-- 'candidate'|'reviewed'|...|'promoted' status lifecycle). The two are
-- unrelated schemas serving unrelated purposes; nothing in this pass
-- reads or writes the production knowledge_base table, and nothing in
-- knowledge_base.py reads or writes knowledge_entries. Promotion from
-- knowledge_entries (lifecycle_status='strong') into the production
-- knowledge_base table is a possible future step, always human-reviewed,
-- never automatic, and not implemented by this pass.
--
-- Isolated the same way as the rest of AI Lab: no FKs to/from
-- signals/outcomes/model_registry/knowledge_base, nothing in
-- stock_scanner/pipeline/ or stock_scanner/alerts/ reads this table.
-- Append-only, same reasoning as reflection_observations/
-- validated_hypotheses: knowledge_id hashes in created_at (THIS curation
-- run's own timestamp, distinct from first_seen/last_confirmed which
-- track the underlying evidence trail), so this table doubles as the
-- "Knowledge Timeline" — a real history of how each belief's lifecycle
-- status evolved across runs, never overwritten.
CREATE TABLE IF NOT EXISTS knowledge_entries (
    knowledge_id              TEXT PRIMARY KEY,   -- sha1(sorted_conditions|created_at)[:16]
    created_at                 TIMESTAMP NOT NULL,  -- this curation run's timestamp, not the evidence trail
    title                        TEXT NOT NULL,
    description                   TEXT NOT NULL,
    conditions                     TEXT NOT NULL,      -- JSON: [[dimension, value], ...] — the normalized belief this entry tracks
    originating_hypotheses           TEXT,               -- JSON list — every validated_hypotheses.hypothesis_id contributing (validated AND rejected, for full traceability)
    evidence_count                     INTEGER NOT NULL,   -- confirmation_count + contradiction_count
    cumulative_sample_size               INTEGER NOT NULL,   -- latest CONFIRMING row's sample_size — never summed across runs (would double-count trades)
    cumulative_successes                   INTEGER NOT NULL,   -- latest CONFIRMING row's successes
    cumulative_failures                      INTEGER NOT NULL,   -- latest CONFIRMING row's failures
    average_win_rate                           REAL NOT NULL,      -- mean of win_rate across every independent confirming run (a ratio average, not a count sum)
    shrunk_win_rate                              REAL NOT NULL,      -- latest CONFIRMING row's shrunk_win_rate, reused not recomputed
    confidence_interval                            TEXT NOT NULL,      -- JSON [wilson_lower, wilson_upper] — latest CONFIRMING row's
    first_seen                                       TIMESTAMP NOT NULL,  -- created_at of the first-ever validated row for this condition-set
    last_confirmed                                     TIMESTAMP NOT NULL,  -- created_at of the most recent CONFIRMING (not contradicting) row
    confirmation_count                                   INTEGER NOT NULL,   -- distinct validation runs that agreed with the established direction
    contradiction_count                                    INTEGER NOT NULL,   -- distinct validation runs that disagreed — established_direction never flips, see knowledge_base_engine.py
    evidence_strength                                        TEXT,               -- 'STRONG' | 'MODERATE' — latest CONFIRMING row's, a snapshot; distinct from lifecycle_status which is the accumulated trajectory
    lifecycle_status                                           TEXT NOT NULL,      -- 'emerging' | 'confirmed' | 'strong' | 'weakening' | 'contradicted' | 'archived'
    previous_lifecycle_status                                    TEXT,               -- lifecycle_status as of one evidence-point earlier — NULL if this is the entry's first-ever appearance
    llm_note                                                       TEXT,               -- optional one-sentence LLM explanation (knowledge_review_agent) — never a source of new numbers
    -- Deployment gate — orthogonal to lifecycle_status (statistical maturity,
    -- automatic). promotion_status answers "has a human approved this for
    -- production?" and is set to 'candidate' by knowledge_base_engine.py at
    -- creation time ONLY; only a future human-run promote_knowledge.py (not
    -- built yet) may set 'promoted'/'rejected'/'archived'. No automatic
    -- promotion exists anywhere. stock_scanner.pipeline.knowledge_application
    -- requires an exact 'promoted' match — missing/NULL/unknown all mean not
    -- promoted (fail closed), which is also why this column still defaults
    -- to 'candidate' rather than allowing NULL: a row imported from a
    -- pre-this-change JSON mirror gets 'candidate' explicitly, never NULL.
    -- CHECK keeps this list in lockstep with
    -- stock_scanner.ai_lab.schemas.KnowledgePromotionStatus's four values
    -- — update both together if this vocabulary ever changes. The Python
    -- layer (stock_scanner.db.knowledge_entries.import_knowledge_entries)
    -- already normalizes any unrecognized value to 'candidate' before it
    -- reaches this table, so this CHECK is a defense-in-depth backstop
    -- against a direct/manual INSERT, not a live failure path for the
    -- sanctioned import/upsert functions.
    promotion_status TEXT NOT NULL DEFAULT 'candidate'
        CHECK (promotion_status IN ('candidate', 'promoted', 'rejected', 'archived')),
    promoted_at                                                      TIMESTAMP,          -- informational only — knowledge_application.py never reads this
    promoted_by                                                        TEXT,               -- informational only
    promotion_reason                                                     TEXT                -- informational only
);
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_lifecycle_status ON knowledge_entries(lifecycle_status);
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_created_at ON knowledge_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_promotion_status ON knowledge_entries(promotion_status);
