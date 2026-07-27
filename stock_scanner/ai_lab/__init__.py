"""AI Lab — experimental, standalone AI recommendation engine.

Completely isolated from the Production Scanner: nothing in
stock_scanner/pipeline/, stock_scanner/alerts/, signal_engine.py, or
scanner_config.yaml is read or modified by anything in this package, and
nothing in stock_scanner/pipeline/ reads AI Lab's output. See
docs/AI_LAB_ARCHITECTURE.md for the full design and the "Future Ready"
notes on the (not yet implemented) Auto Promotion Engine.

Pipeline (this package implements everything from Hypothesis Agent
onward; Feature Engineering / Pattern Miner / Statistical Validation /
Knowledge Base already exist as stock_scanner.pipeline.feature_builder /
stock_scanner.learning.pattern_miner / stock_scanner.learning.pattern_dedup
/ stock_scanner.db.knowledge_base — reused as-is, not duplicated):

    Historical Market Data
      -> Feature Engineering        (stock_scanner.pipeline.feature_builder)
      -> Pattern Miner              (stock_scanner.learning.pattern_miner)
      -> Statistical Validation     (stock_scanner.learning.pattern_dedup)
      -> Knowledge Base             (stock_scanner.db.knowledge_base)
      -> Hypothesis Agent           (stock_scanner.ai_lab.agents.hypothesis_agent)
      -> Decision Agent             (stock_scanner.ai_lab.agents.decision_agent)
      -> AI Recommendation Engine   (stock_scanner.db.ai_lab)
      -> AI Lab Dashboard           (dashboard.ai_lab_view)

Modules:
    schemas.py    — Pydantic contracts for every LLM input/output (never free-form parsing)
    client.py     — async 9router client: retry, timeout, structured JSON, Pydantic-validated
    prompts.py    — prompt builders (evidence in, narrative out — never the reverse)
    models.py     — plug-and-play AI model registry (Momentum/Breakout/Reversal/Volume AI)
    performance.py — win rate / avg return / profit factor / Sharpe etc., grouped per ai_model
    agents/hypothesis_agent.py — evidence -> HypothesisOutput
    agents/decision_agent.py   — evidence + hypothesis -> DecisionOutput -> AIRecommendation

Status: runs automatically as the final step of the Daily Scan orchestrator
(pipeline.py::run_ai_pipeline(), gated by ai_lab.enabled in
scanner_config.yaml) — see the "Architecture Decision Record" in
docs/ADR_AI_AUTOMATION_AND_STOCK_DICTIONARY.md and the "Scheduled
execution" note in docs/AI_LAB_ARCHITECTURE.md. scripts/run_ai_lab.py and
its siblings remain available for manual/standalone runs.
"""
