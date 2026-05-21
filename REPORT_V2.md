# IDX Stock Scanner — Audit & Fix Report V2

**Date**: 2026-05-21  
**Branch**: `audit-fixes-v2`  
**Author**: Claude (audit-driven refactor)

---

## Summary

Full audit of the IDX signal scanner pipeline covering:
1. Root-cause analysis of poor signal quality (248 → 64 ranked stocks after fixes)
2. Signal engine hardening (liquidity gate, penalty weights, thresholds)
3. Historical backfill evaluation across 6 dates (2221 signals evaluated)
4. ML ranker training from 3 years of OHLCV data (AUC 0.594)
5. `ml_prob` column now active in daily scan pipeline

---

## Root Cause Analysis

### Problem: Why the old system ranked 248 stocks/day (May 20)

| Issue | Detail |
|---|---|
| WATCH threshold too low | 3.5/10 — almost any stock with mild trend passed |
| Penalty weight = 0.10 | Liquidity/volatility penalty had ≈ no effect on `total_score` |
| No hard liquidity gate | BRAM (Rp 117M turnover, #1 BREAKOUT) passed all filters |
| `squeeze_release` always False | Computed via `shift(1)` on single-row DataFrames per ticker → always 0 |

### BRAM concrete example
- Close = 15,350, Volume = 7,600 → turnover **Rp 116.7M**
- `vol_ratio_20d = 8.88x` → volume spike → high `breakout_score`
- `penalty_score = 7` but weight 0.10 → only -0.70 deduction
- Net result: BREAKOUT rank #1 despite being uninvestable

---

## Files Changed

### Modified

| File | Change |
|---|---|
| `stock_scanner/pipeline/signal_engine.py` | Hard liquidity gate (turnover < Rp 500M → AVOID); penalty weight 0.10→0.20; `squeeze_release` reads pre-computed column; new defaults: WATCH ≥5.0, PRE_MARKUP ≥6.0 |
| `stock_scanner/pipeline/feature_builder.py` | Added `squeeze_release` column in `_add_tv_indicators()` using full ticker history |
| `stock_scanner/pipeline/run_daily_scan.py` | `_save_ranked()` takes `config`; per-tier caps applied (BREAKOUT=15, PRE_MARKUP=30, WATCH=50); passes `min_rr` to level calculator |
| `stock_scanner/pipeline/ml_ranker.py` | `score_candidates()` fills missing feature columns with 0 (backward-compat for old parquets) |
| `stock_scanner/alerts/level_calculator.py` | R:R validated after tick rounding; if actual R:R < `min_rr` → `trade_setup_status = "low_rr"` |
| `stock_scanner/configs/scanner_config.yaml` | Added `liquidity` gate, `signal_caps`, `min_rr`; updated thresholds |
| `stock_scanner/configs/model_config.yaml` | Feature list updated (removed signal-engine-only columns, added feature_builder columns) |

### New Files

| File | Purpose |
|---|---|
| `stock_scanner/pipeline/evaluator.py` | Compute realized outcomes (TP/STOP/OPEN, realized_R, MFE, MAE) for historical ranked files |
| `stock_scanner/pipeline/cooldown.py` | Suppress re-appearing tickers within N trading days at same or higher tier |
| `stock_scanner/pipeline/train_ranker_from_history.py` | Build training dataset from raw OHLCV, train XGBoost, save `models/ranker.pkl` |
| `scripts/backfill_evaluate.py` | CLI runner for evaluating all historical ranked files; prints summary table + quintile report |
| `data/evaluation/eval_*.csv` (6 files) | Realized outcomes for May 7–13 ranked signals |
| `models/ranker.pkl` | Trained XGBoost classifier (AUC 0.594) |
| `models/ranker_metrics.yaml` | Training metrics and configuration |

---

## Backfill Evaluation Results (Old System)

Evaluated May 7–13 ranked files against forward OHLCV (horizon = 10 trading days).  
Files May 14+ skipped: insufficient forward data (raw data ends 2026-05-20).

### Per-date summary

| Date | n_signals | win_rate | avg_R | profit_factor | pct_TP | pct_STOP |
|---|---|---|---|---|---|---|
| 2026-05-07 | 1 | 0.000 | -0.333 | 0.00 | 0.0% | 100.0% |
| 2026-05-08 | 474 | 0.244 | -0.269 | 0.62 | 21.9% | 69.8% |
| 2026-05-10 | 449 | 0.247 | -0.276 | 0.63 | 22.3% | 71.9% |
| 2026-05-11 | 449 | 0.244 | -0.276 | 0.63 | 21.8% | 71.9% |
| 2026-05-12 | 432 | 0.243 | -0.270 | 0.62 | 21.8% | 71.5% |
| 2026-05-13 | 416 | 0.240 | -0.281 | 0.60 | 21.4% | 71.9% |

### Aggregate (n=2221 signals)

```
Win rate      : 24.3%
Avg realized R: -0.273
Profit factor : 0.62
Avg MFE       :  2.47%
Avg MAE       :  3.14%
TP hits       : 474  (21.3%)
Stop hits     : 1582 (71.2%)
Still open    : 165  ( 7.4%)
```

### Per signal type

```
BREAKOUT:   n=10   WR=40.0% avgR=-0.125  PF=0.79
PRE_MARKUP: n=556  WR=27.3% avgR=-0.228  PF=0.67
WATCH:      n=1655 WR=23.1% avgR=-0.290  PF=0.61
```

### Score quintile monotonicity: ❌ NOT monotonic

Higher `total_score` does **not** consistently predict better realized_R.  
This confirms the old signal engine lacked discriminative power.

### Interpretation

The old system's evaluation used "same-day close > open" (wrong metric).  
When evaluated with correct forward swing-trading metrics (entry/TP/SL simulation):
- 71% of stops hit → overfit to momentum features that decay within 1-2 days
- WATCH (low-conviction) is the drag: 1655/2221 signals, worst R (-0.290)
- The new system removes WATCH unless total_score ≥ 5.0 (previously 3.5)

---

## New System Comparison (May 20 Scan)

| Metric | Old | New |
|---|---|---|
| Total ranked | 248 | 64 |
| BREAKOUT | 13 | 0 |
| PRE_MARKUP | 43 | 24 |
| WATCH | 192 | 40 |
| BRAM result | BREAKOUT #1 | ✅ AVOID (Rp 116M turnover) |
| `ml_prob` active | ❌ No | ✅ Yes |

---

## ML Ranker Results

**Training**: 886 tickers × 3 years of OHLCV → 255,545 rows  
**Features**: 16 technical indicators from `feature_builder`  
**Target**: 5-day forward return > 3% (binary)  
**Split**: chronological (last 20% = test)

```
Train period : 2023-05-08 → 2025-11-26  (204,436 rows)
Test period  : 2025-11-26 → 2026-05-13  ( 51,109 rows)

AUC-ROC      : 0.5940   ← above random (0.50)
Precision(1) : 0.5046
Recall(1)    : 0.0283
F1(1)        : 0.0536
Positive rate: 25.4%
```

**Top features by importance**:

| Feature | Importance |
|---|---|
| `atr_pct` | 0.192 |
| `bb_width` | 0.158 |
| `hist_vol_20d` | 0.111 |
| `roc20` | 0.066 |
| `price_vs_ma200` | 0.063 |

**Notes**:
- AUC 0.594 is modest but meaningful for financial data (typical range: 0.52–0.65)
- Recall is very low at 0.028 at 0.5 threshold — this is by design for a **ranker**, not a classifier
- `ml_prob` is used to rank candidates within each signal tier, not as a binary gate
- Model saved to `models/ranker.pkl`; retrain monthly or on major market regime change

---

## Config Changes Summary

```yaml
# scanner_config.yaml — key changes
liquidity:
  min_turnover_idr: 500_000_000   # NEW: Rp 500M hard gate

signal_thresholds:
  watch:      { total: 5.0 }      # was 3.5 → +43% bar
  pre_markup: { total: 6.0 }      # was 5.5

signal_caps:                       # NEW: per-tier safety valves
  breakout: 15
  pre_markup: 30
  watch: 50

min_rr: 1.5                       # NEW: R:R gate after tick rounding
```

---

## Next Steps

1. **Accumulate forward data** — the new (stricter) ranked files from May 14+ will be evaluatable once 5+ trading days of forward OHLCV exist. Run `backfill_evaluate.py --overwrite` weekly.

2. **Cooldown filter integration** — wire `apply_cooldown()` into `run_daily_scan.py`:
   ```python
   from stock_scanner.pipeline.cooldown import apply_cooldown
   signals_df = apply_cooldown(signals_df, ranked_dir, cooldown_days=5, scan_date=scan_date)
   ```

3. **ML retraining cadence** — set up monthly cron:
   ```bash
   python -m stock_scanner.pipeline.train_ranker_from_history --horizon 5 --target-pct 3.0
   ```

4. **Walk-forward backtest** — once enough evaluation data accumulates, validate whether new WATCH threshold (5.0) + ml_prob ranking actually improves realized_R.

5. **Feature engineering** — consider adding:
   - RS (Relative Strength vs IHSG) 
   - Foreign net-buy signal as feature
   - Sector momentum

6. **Threshold calibration** — after 2–4 weeks of new system data, revisit WATCH 5.0 vs PRE_MARKUP 6.0 thresholds using evaluation data.
