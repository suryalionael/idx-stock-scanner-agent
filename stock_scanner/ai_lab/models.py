"""Plug-and-play AI model registry.

Each entry is a named "lens" the Hypothesis/Decision agents apply when
looking at the same underlying evidence — not a separately-trained
statistical model. This keeps the architecture genuinely extensible: a new
strategy persona is a new AI_MODEL_REGISTRY entry, nothing else changes.
stock_scanner.db.ai_lab groups performance by the `ai_model` column value
alone (no hardcoded model list anywhere in storage/performance code), so a
new entry here is immediately visible in the dashboard's per-model
performance breakdown with zero other code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AIModelSpec:
    key: str                 # stored verbatim in ai_recommendations.ai_model
    display_name: str
    description: str
    focus_features: list[str] = field(default_factory=list)
    persona_instructions: str = ""


AI_MODEL_REGISTRY: dict[str, AIModelSpec] = {
    "momentum_ai": AIModelSpec(
        key="momentum_ai",
        display_name="Momentum AI",
        description="Looks for sustained directional strength — moving-average alignment, "
                     "ROC, ADX trend strength.",
        focus_features=["ma_full_alignment", "ma_partial_alignment", "slope_ma20", "roc5",
                         "roc20", "adx", "adx_pos", "adx_neg", "golden_cross"],
        persona_instructions="You favor names showing sustained upward price momentum over "
                              "multiple timeframes, confirmed by trend-strength indicators.",
    ),
    "breakout_ai": AIModelSpec(
        key="breakout_ai",
        display_name="Breakout AI",
        description="Looks for range compression followed by expansion — ATR breakout, "
                     "Bollinger squeeze release, 52-week high proximity.",
        focus_features=["atr_breakout", "atr_pct", "bb_width", "squeeze_on", "squeeze_release",
                         "pct_from_52w_high", "high_52w"],
        persona_instructions="You favor names breaking out of a consolidation range with "
                              "confirming volatility expansion.",
    ),
    "reversal_ai": AIModelSpec(
        key="reversal_ai",
        display_name="Reversal AI",
        description="Looks for oversold/overbought exhaustion and early reversal signals — "
                     "RSI, Stochastic RSI, MACD histogram inflection.",
        focus_features=["rsi14", "stoch_rsi_k", "stoch_rsi_d", "macd", "macd_signal",
                         "macd_histogram", "price_vs_ma200"],
        persona_instructions="You favor names showing exhaustion of the prior trend and early "
                              "signs of reversal, not names still trending strongly.",
    ),
    "volume_ai": AIModelSpec(
        key="volume_ai",
        display_name="Volume AI",
        description="Looks for volume-confirmed conviction — volume spikes, OBV trend, "
                     "foreign net flow.",
        focus_features=["vol_ratio_20d", "vol_spike", "obv_trend", "foreign_net",
                         "foreign_net_5d", "foreign_net_20d", "foreign_flow_score"],
        persona_instructions="You favor names where price action is confirmed by unusually "
                              "high volume and/or sustained foreign net buying.",
    ),
}


def get_model_spec(key: str) -> AIModelSpec:
    if key not in AI_MODEL_REGISTRY:
        raise KeyError(
            f"Unknown AI Lab model key {key!r}. Registered models: {sorted(AI_MODEL_REGISTRY)}. "
            f"To add a new model, add an AIModelSpec entry to AI_MODEL_REGISTRY — no other "
            f"code changes are required for it to appear in performance tracking or the dashboard."
        )
    return AI_MODEL_REGISTRY[key]
