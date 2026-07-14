"""Tests for the daily movers >10% feature — non-production, standalone.
See stock_scanner/pipeline/daily_movers.py (computation) and
stock_scanner/db/daily_movers.py (persistence)."""
import json
import sqlite3

import pandas as pd
import pytest

from stock_scanner.db.daily_movers import (
    export_daily_movers,
    import_daily_movers,
    load_daily_movers,
    upsert_daily_movers,
)
from stock_scanner.db.init_db import create_schema
from stock_scanner.pipeline.daily_movers import compute_daily_movers


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    return conn


def _ohlc_row(ticker, date, open_, high, low, close, volume=1_000_000):
    return {"ticker": ticker, "date": date, "open": open_, "high": high,
            "low": low, "close": close, "volume": volume}


# ---------------------------------------------------------------------------
# compute_daily_movers — pct calculation + classification
# ---------------------------------------------------------------------------

def test_pct_change_close_and_high_calculated_correctly():
    df = pd.DataFrame([
        _ohlc_row("AAAA.JK", "2026-07-10", 100, 105, 99, 100),
        _ohlc_row("AAAA.JK", "2026-07-13", 105, 115, 104, 112),   # prev_close=100
    ])
    out = compute_daily_movers(df)
    row = out.iloc[0]
    assert row["prev_close"] == 100
    assert row["pct_change_close"] == pytest.approx(0.12)
    assert row["pct_change_high"] == pytest.approx(0.15)


def test_hit_flags_true_when_move_meets_threshold_exactly():
    df = pd.DataFrame([
        _ohlc_row("BBBB.JK", "2026-07-10", 100, 100, 100, 100),
        _ohlc_row("BBBB.JK", "2026-07-13", 108, 110, 107, 110),   # close +10% exactly, high +10%
    ])
    out = compute_daily_movers(df)
    row = out.iloc[0]
    assert row["hit_10pct_close"] == True  # noqa: E712
    assert row["hit_10pct_intraday"] == True  # noqa: E712


def test_intraday_hit_without_close_hit():
    # Spiked >10% intraday but closed back below the threshold.
    df = pd.DataFrame([
        _ohlc_row("CCCC.JK", "2026-07-10", 100, 100, 100, 100),
        _ohlc_row("CCCC.JK", "2026-07-13", 105, 112, 104, 105),   # high +12%, close +5%
    ])
    out = compute_daily_movers(df)
    row = out.iloc[0]
    assert row["hit_10pct_intraday"] == True  # noqa: E712
    assert row["hit_10pct_close"] == False  # noqa: E712


def test_non_movers_excluded_from_output():
    df = pd.DataFrame([
        _ohlc_row("DDDD.JK", "2026-07-10", 100, 101, 99, 100),
        _ohlc_row("DDDD.JK", "2026-07-13", 100, 102, 99, 101),   # +1%, not a mover
    ])
    out = compute_daily_movers(df)
    assert out.empty


def test_first_row_per_ticker_skipped_gracefully_no_prev_close():
    # A ticker's very first observation has no prior row — must be skipped,
    # not raise or produce a NaN-based false hit.
    df = pd.DataFrame([
        _ohlc_row("EEEE.JK", "2026-07-13", 100, 130, 95, 125),
    ])
    out = compute_daily_movers(df)
    assert out.empty


def test_mixed_universe_only_returns_actual_movers():
    df = pd.DataFrame([
        _ohlc_row("FFFF.JK", "2026-07-10", 100, 101, 99, 100),
        _ohlc_row("FFFF.JK", "2026-07-13", 100, 102, 99, 101),   # +1%, not a mover
        _ohlc_row("GGGG.JK", "2026-07-10", 200, 202, 198, 200),
        _ohlc_row("GGGG.JK", "2026-07-13", 210, 230, 208, 225),  # +12.5% close, mover
    ])
    out = compute_daily_movers(df)
    assert list(out["ticker"]) == ["GGGG.JK"]


# ---------------------------------------------------------------------------
# DB layer — upsert idempotency, load, export/import JSON shape
# ---------------------------------------------------------------------------

def _sample_movers() -> pd.DataFrame:
    return pd.DataFrame([{
        "trade_date": "2026-07-13", "ticker": "GGGG.JK", "prev_close": 200.0,
        "open": 210.0, "high": 230.0, "low": 208.0, "close": 225.0, "volume": 5_000_000,
        "pct_change_close": 0.125, "pct_change_high": 0.15,
        "hit_10pct_close": True, "hit_10pct_intraday": True,
    }])


def test_upsert_inserts_row():
    conn = _conn()
    n = upsert_daily_movers(conn, _sample_movers())
    assert n == 1
    df = load_daily_movers(conn)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "GGGG.JK"
    assert df.iloc[0]["hit_10pct_close"] == 1


def test_upsert_is_idempotent_overwrites_not_duplicates():
    conn = _conn()
    upsert_daily_movers(conn, _sample_movers())
    # Re-run with an updated close for the same (trade_date, ticker) key.
    updated = _sample_movers()
    updated.loc[0, "close"] = 240.0
    updated.loc[0, "pct_change_close"] = 0.20
    upsert_daily_movers(conn, updated)

    df = load_daily_movers(conn)
    assert len(df) == 1   # no duplicate row
    assert df.iloc[0]["close"] == 240.0
    assert df.iloc[0]["pct_change_close"] == pytest.approx(0.20)


def test_upsert_empty_dataframe_is_noop():
    conn = _conn()
    n = upsert_daily_movers(conn, pd.DataFrame())
    assert n == 0
    assert load_daily_movers(conn).empty


def test_export_json_shape(tmp_path):
    conn = _conn()
    upsert_daily_movers(conn, _sample_movers())
    export_path = export_daily_movers(conn, path=tmp_path / "daily_movers.json")

    payload = json.loads(export_path.read_text())
    assert payload["as_of_date"] == "2026-07-13"
    assert payload["source"] == "yfinance"
    assert "generated_at" in payload
    assert payload["summary"]["total_rows"] == 1
    assert payload["summary"]["hit_10pct_close_count"] == 1
    assert payload["summary"]["hit_10pct_intraday_count"] == 1
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["ticker"] == "GGGG.JK"
    assert payload["rows"][0]["hit_10pct_close"] is True


def test_export_import_round_trip(tmp_path):
    conn = _conn()
    upsert_daily_movers(conn, _sample_movers())
    export_path = export_daily_movers(conn, path=tmp_path / "daily_movers.json")

    fresh_conn = _conn()
    n_imported = import_daily_movers(fresh_conn, path=export_path)
    assert n_imported == 1
    assert len(load_daily_movers(fresh_conn)) == 1


def test_import_is_idempotent(tmp_path):
    conn = _conn()
    upsert_daily_movers(conn, _sample_movers())
    export_path = export_daily_movers(conn, path=tmp_path / "daily_movers.json")

    fresh_conn = _conn()
    import_daily_movers(fresh_conn, path=export_path)
    n_second = import_daily_movers(fresh_conn, path=export_path)
    assert n_second == 0
    assert len(load_daily_movers(fresh_conn)) == 1


def test_import_missing_file_returns_zero(tmp_path):
    conn = _conn()
    n = import_daily_movers(conn, path=tmp_path / "does_not_exist.json")
    assert n == 0
