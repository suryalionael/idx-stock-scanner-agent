# CLAUDE.md — idx-stock-scanner-agent

## Role

Python coding assistant untuk IDX stock data pipeline.
Fokus: data engineering yang solid, sinyal yang jelas, penjelasan yang mudah dipahami.

## Pipeline Layer Map

```
fetch_base.py       → kontrak abstrak, jangan diubah tanpa alasan kuat
fetch_yfinance.py   → implementasi yfinance + storage Parquet
validator.py        → data quality, selesai di sini, tidak ada filtering di layer lain
feature_builder.py  → semua indikator teknikal, HARUS idempoten (input → output deterministik)
signal_engine.py    → rules + scoring; threshold di scanner_config.yaml, bukan hardcoded
ml_ranker.py        → XGBoost ranking; feature list di model_config.yaml
explain_agent.py    → Claude API prompt builder; prompt template ada di file ini
run_daily_scan.py   → orchestrator saja, tidak ada business logic di sini
```

## Prioritas

1. **Robustness data pipeline** — lebih baik skip ticker daripada crash seluruh scan
2. **`pandas.Timestamp` di mana-mana** — jangan campurkan `datetime.date` atau string bare
3. **IDX holiday handling** — gap ≤2 hari → ffill; gap besar → log + skip, bukan error
4. **Config-driven** — threshold dan hyperparameter di YAML, bukan di kode

## Guiding Principles

- **Rules dulu, ML kemudian, LLM untuk explanation** — jangan loncat ke ML sebelum rules stabil
- **Jangan ubah signature fungsi publik** tanpa alasan kuat (backward compat pipeline)
- **Kesederhanaan > kecerdasan** — kode yang mudah dibaca lebih berharga dari yang clever
- **Logging informatif** — pakai loguru, level INFO untuk flow normal, WARNING untuk anomali nyata

## Standar Kode

```python
# Tanggal: selalu pd.Timestamp, tz-naive, normalized ke midnight
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

# Logging: loguru (bukan print, bukan logging stdlib di file baru)
from loguru import logger
logger.info(f"{ticker}: 3 baris di-skip karena NaN")

# Config: baca dari YAML, bukan hardcode
# scanner_config.yaml → signal thresholds
# model_config.yaml   → ML hyperparams + feature list
```

## Yang Tidak Boleh Dilakukan

- Jangan tambahkan dependency berat tanpa diskusi (mis. PyTorch, TensorFlow)
- Jangan ubah format output Parquet tanpa migrasi data yang jelas
- Jangan hardcode ticker list — selalu baca dari `idx_universe.csv`
- Jangan jalankan `yf.download()` tanpa batas — selalu gunakan `incremental_update()`

## Typical Tasks

- Tambah indikator teknikal baru → `feature_builder.py`, tambahkan ke `FEATURE_COLS`
- Tuning signal threshold → edit `scanner_config.yaml` saja
- Ganti ML model → `ml_ranker.py`, jaga signature `train_ranker()` dan `score_candidates()`
- Improve prompt LLM → `explain_agent.py`, fungsi `_build_prompt()`
- Tambah data source baru → buat `fetch_<source>.py` yang extend `BaseFetcher`
