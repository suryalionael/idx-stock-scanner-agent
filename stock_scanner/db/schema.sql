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
    event_type     TEXT NOT NULL,       -- 'pattern_learned' | 'confidence_updated' | 'hypothesis_generated' | 'accuracy_improved'
    description    TEXT NOT NULL,
    metadata_json  TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ai_learning_events_created_at ON ai_learning_events(created_at);
