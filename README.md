# idx-stock-scanner-agent

AI-powered daily scanner untuk saham IDX (Bursa Efek Indonesia).
**Decision-support tool — bukan jaminan profit.**

---

## Arsitektur

Pipeline tiga lapis: **Rules → ML → LLM**

```
DataFetcher → Validator → FeatureEngine → SignalEngine → MLRanker → ExplainAgent
     ↓             ↓            ↓               ↓            ↓            ↓
  OHLCV raw    clean data    indicators      rule signal    ml_prob     narasi
  (yfinance)   + gaps        (24 kolom)     + score        (XGBoost)   (Claude)
```

| Layer | Peran |
|---|---|
| **Rules** (SignalEngine) | Filter awal + guardrail. Reject saham tidak layak sebelum ML menyentuhnya. |
| **ML** (MLRanker) | Ranking probabilistik: seberapa besar kemungkinan return > X% dalam N hari. |
| **LLM** (ExplainAgent) | Narasi singkat per ticker: kenapa signal ini muncul, apa yang perlu diperhatikan. |

---

## Komponen

### `stock_scanner/pipeline/`

| File | Fungsi |
|---|---|
| `fetch_base.py` | `BaseFetcher` abstract class — kontrak data provider |
| `fetch_yfinance.py` | `YFinanceFetcher` + `incremental_update()` — download & simpan ke Parquet |
| `validator.py` | `validate()` + `validate_batch()` — data quality check |
| `feature_builder.py` | `build_features()` — 20+ indikator teknikal (MA, RSI, MACD, ATR, dll.) |
| `signal_engine.py` | `compute_signal()` — rule-based scoring + label (`BREAKOUT/PRE_MARKUP/WATCH/AVOID`) |
| `ml_ranker.py` | `train_ranker()` + `score_candidates()` — XGBoost probability ranking |
| `explain_agent.py` | `explain_signal()` — prompt builder → Claude API |
| `run_daily_scan.py` | Entry point: fetch → validate → features → signal → rank → explain |

---

## Alur Kerja Harian

```
1. Load ticker universe (configs/idx_universe.csv)
2. Incremental update data OHLCV (yfinance, Parquet storage)
3. Validate — bersihkan NaN, gap, harga negatif
4. Build features — 20+ kolom teknikal
5. SignalEngine — scoring + label per ticker
6. MLRanker (opsional) — tambah kolom ml_prob
7. ExplainAgent (opsional) — narasi per kandidat
8. Simpan output: data/signals/ + data/ranked/
```

Jalankan:
```bash
python -m stock_scanner.pipeline.run_daily_scan --config stock_scanner/configs/scanner_config.yaml
```

---

## Setup

```bash
# Clone dan buat virtual env
git clone <repo-url>
cd idx-stock-scanner-agent
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Siapkan universe ticker
cp stock_scanner/configs/idx_universe.csv.example stock_scanner/configs/idx_universe.csv
# Edit: tambahkan ticker IDX kamu (format: ticker,is_active)

# Jalankan scan pertama
python -m stock_scanner.pipeline.run_daily_scan
```

---

## Output

```
data/
  raw/          # OHLCV per ticker (Parquet)
  features/     # feature matrix per scan date (Parquet)
  signals/      # signals + scores per scan date (Parquet + CSV)
  ranked/       # top candidates per scan date (CSV)
models/
  ranker.pkl    # trained XGBoost model (setelah train_ranker dijalankan)
```

---

## Disclaimer

Tool ini dibuat untuk edukasi dan riset pribadi.
Output bukan rekomendasi investasi. Selalu lakukan due diligence sendiri.
