"""Tests for the top signals >10% daily persistence feature —
non-production, standalone, NOT the knowledge_base (Learning Agent Phase 1)
table. See stock_scanner/pipeline/top_signals.py (filter/enrich/rank) and
stock_scanner/db/top_signals.py (persistence)."""
import json
import sqlite3

import pandas as pd
import pytest

from stock_scanner.db.init_db import create_schema
from stock_scanner.db.top_signals import (
    export_top_signals,
    import_top_signals,
    load_top_signals,
    upsert_top_signals,
)
from stock_scanner.pipeline.top_signals import (
    build_top_signals,
    enrich_with_quality_scores,
    filter_top_signals,
    rank_top_signals,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    return conn


def _result_row(ticker, signal_date, eval_date, strategy, prev, close, high, pct_close, pct_high,
                 status="evaluated", signal="PRE_MARKUP", wl=None):
    return {
        "signal_date": signal_date, "eval_date": eval_date, "strategy": strategy, "ticker": ticker,
        "signal": signal, "prev": prev, "close": close, "high": high,
        "pct_high": pct_high, "pct_close": pct_close,
        "wl": wl or ("W" if pct_close is not None and pct_close > 0 else "L"),
        "status": status,
    }


# ---------------------------------------------------------------------------
# filter_top_signals — exact >10% rule
# ---------------------------------------------------------------------------

def test_filter_excludes_at_exactly_10pct():
    # Strict >10% — "above 10%", not >=10%. A signal at exactly 10.0 must
    # NOT qualify (unlike the daily_movers.py >=10% feature, which is a
    # separate, differently-specified rule).
    df = pd.DataFrame([
        _result_row("AAAA.JK", "2026-07-01", "2026-07-06", "swing", 100, 110.0, 110.0, 10.0, 10.0),
    ])
    out = filter_top_signals(df)
    assert out.empty


def test_filter_includes_just_above_10pct():
    df = pd.DataFrame([
        _result_row("BBBB.JK", "2026-07-01", "2026-07-06", "swing", 100, 110.01, 111.0, 10.01, 11.0),
    ])
    out = filter_top_signals(df)
    assert len(out) == 1
    assert out.iloc[0]["forward_return_pct"] == pytest.approx(0.1001)


def test_filter_excludes_pending_signals():
    df = pd.DataFrame([
        _result_row("CCCC.JK", "2026-07-01", None, "swing", 100, None, None, None, None,
                     status="pending", wl=None),
    ])
    out = filter_top_signals(df)
    assert out.empty


def test_filter_uses_pct_close_not_pct_high():
    # A name that spiked >10% intraday but closed back below threshold must
    # NOT appear here — this table is about realized close-to-close return,
    # unlike daily_movers.py's intraday metric.
    df = pd.DataFrame([
        _result_row("DDDD.JK", "2026-07-01", "2026-07-06", "swing", 100, 105.0, 118.0, 5.0, 18.0),
    ])
    out = filter_top_signals(df)
    assert out.empty


def test_filter_deterministic_signal_id():
    df = pd.DataFrame([
        _result_row("EEEE.JK", "2026-07-01", "2026-07-06", "swing", 100, 115.0, 116.0, 15.0, 16.0),
    ])
    out1 = filter_top_signals(df)
    out2 = filter_top_signals(df)
    assert out1.iloc[0]["signal_id"] == out2.iloc[0]["signal_id"]


# ---------------------------------------------------------------------------
# rank_top_signals — ordering + graceful degradation without quality scores
# ---------------------------------------------------------------------------

def test_rank_orders_by_return_desc_within_day():
    df = pd.DataFrame([
        _result_row("FFFF.JK", "2026-07-01", "2026-07-06", "swing", 100, 112.0, 112.0, 12.0, 12.0),
        _result_row("GGGG.JK", "2026-07-01", "2026-07-06", "swing", 100, 130.0, 130.0, 30.0, 30.0),
        _result_row("HHHH.JK", "2026-07-01", "2026-07-06", "swing", 100, 120.0, 120.0, 20.0, 20.0),
    ])
    filtered = filter_top_signals(df)
    ranked = rank_top_signals(filtered)
    assert list(ranked["ticker"]) == ["GGGG.JK", "HHHH.JK", "FFFF.JK"]
    assert list(ranked["rank_in_day"]) == [1, 2, 3]


def test_rank_falls_back_to_return_when_quality_score_missing():
    # No promoted model / no ranked CSV available that day -> quality scores
    # stay NULL. Ranking must not crash or misorder; it degrades to
    # return-only ordering, per the "no additional promoted model" contract.
    df = pd.DataFrame([
        _result_row("IIII.JK", "2026-07-01", "2026-07-06", "swing", 100, 111.0, 111.0, 11.0, 11.0),
        _result_row("JJJJ.JK", "2026-07-01", "2026-07-06", "swing", 100, 125.0, 125.0, 25.0, 25.0),
    ])
    filtered = filter_top_signals(df)
    assert filtered["quality_adjusted_score"].isna().all()
    ranked = rank_top_signals(filtered)
    assert list(ranked["ticker"]) == ["JJJJ.JK", "IIII.JK"]


def test_rank_uses_quality_score_as_tiebreak():
    df = pd.DataFrame([
        _result_row("KKKK.JK", "2026-07-01", "2026-07-06", "swing", 100, 120.0, 120.0, 20.0, 20.0),
        _result_row("LLLL.JK", "2026-07-01", "2026-07-06", "swing", 100, 120.0, 120.0, 20.0, 20.0),
    ])
    filtered = filter_top_signals(df)
    filtered.loc[filtered["ticker"] == "KKKK.JK", "quality_adjusted_score"] = 50.0
    filtered.loc[filtered["ticker"] == "LLLL.JK", "quality_adjusted_score"] = 90.0
    ranked = rank_top_signals(filtered)
    assert list(ranked["ticker"]) == ["LLLL.JK", "KKKK.JK"]


def test_rank_separate_cohorts_per_eval_date():
    df = pd.DataFrame([
        _result_row("MMMM.JK", "2026-07-01", "2026-07-06", "swing", 100, 111.0, 111.0, 11.0, 11.0),
        _result_row("NNNN.JK", "2026-06-01", "2026-06-06", "swing", 100, 150.0, 150.0, 50.0, 50.0),
    ])
    filtered = filter_top_signals(df)
    ranked = rank_top_signals(filtered)
    # Both are rank 1 in their own eval_date cohort, despite very different returns.
    assert set(ranked["rank_in_day"]) == {1}


# ---------------------------------------------------------------------------
# enrich_with_quality_scores — best-effort join, missing file handled
# ---------------------------------------------------------------------------

def test_enrich_fills_quality_columns_when_ranked_csv_present(tmp_path):
    df = pd.DataFrame([
        _result_row("OOOO.JK", "2026-07-01", "2026-07-06", "swing", 100, 115.0, 115.0, 15.0, 15.0),
    ])
    filtered = filter_top_signals(df)

    ranked_dir = tmp_path
    ranked = pd.DataFrame([{
        "ticker": "OOOO.JK", "quality_adjusted_score": 72.5, "total_score": 60.0,
        "enhanced_total_score": 65.0, "ml_prob": 0.81,
    }])
    ranked.to_csv(ranked_dir / "ranked_2026-07-01.csv", index=False)

    enriched = enrich_with_quality_scores(filtered, ranked_dir)
    row = enriched.iloc[0]
    assert row["quality_adjusted_score"] == 72.5
    assert row["ml_prob"] == 0.81
    assert row["quality_source"] == "ranked_csv"


def test_enrich_leaves_unavailable_when_ranked_csv_missing(tmp_path):
    df = pd.DataFrame([
        _result_row("PPPP.JK", "2026-07-01", "2026-07-06", "swing", 100, 115.0, 115.0, 15.0, 15.0),
    ])
    filtered = filter_top_signals(df)
    enriched = enrich_with_quality_scores(filtered, tmp_path)  # empty dir, no ranked file
    row = enriched.iloc[0]
    assert pd.isna(row["quality_adjusted_score"])
    assert row["quality_source"] == "unavailable"


def test_build_top_signals_end_to_end(tmp_path):
    df = pd.DataFrame([
        _result_row("QQQQ.JK", "2026-07-01", "2026-07-06", "swing", 100, 111.0, 111.0, 11.0, 11.0),
        _result_row("RRRR.JK", "2026-07-01", "2026-07-06", "swing", 100, 105.0, 105.0, 5.0, 5.0),
    ])
    out = build_top_signals(df, tmp_path)
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "QQQQ.JK"
    assert out.iloc[0]["rank_in_day"] == 1


# ---------------------------------------------------------------------------
# DB layer — upsert idempotency, load, export/import JSON shape
# ---------------------------------------------------------------------------

def _sample_top_signal() -> pd.DataFrame:
    return pd.DataFrame([{
        "signal_id": "abc123", "ticker": "SSSS.JK", "strategy": "swing",
        "signal_date": "2026-07-01", "eval_date": "2026-07-06", "signal_label": "PRE_MARKUP",
        "prev_close": 100.0, "eval_close": 120.0, "eval_high": 122.0,
        "pct_close": 20.0, "pct_high": 22.0, "forward_return_pct": 0.20,
        "quality_adjusted_score": 70.0, "total_score": 60.0, "enhanced_total_score": 65.0,
        "ml_prob": 0.75, "quality_source": "ranked_csv", "rank_in_day": 1,
    }])


def test_upsert_inserts_row():
    conn = _conn()
    n = upsert_top_signals(conn, _sample_top_signal(), source_run_id="run1")
    assert n == 1
    df = load_top_signals(conn)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "SSSS.JK"


def test_upsert_is_idempotent_overwrites_not_duplicates():
    conn = _conn()
    upsert_top_signals(conn, _sample_top_signal(), source_run_id="run1")
    updated = _sample_top_signal()
    updated.loc[0, "quality_adjusted_score"] = 88.0
    updated.loc[0, "quality_source"] = "ranked_csv"
    upsert_top_signals(conn, updated, source_run_id="run2")

    df = load_top_signals(conn)
    assert len(df) == 1
    assert df.iloc[0]["quality_adjusted_score"] == 88.0


def test_upsert_empty_dataframe_is_noop():
    conn = _conn()
    n = upsert_top_signals(conn, pd.DataFrame(), source_run_id="run1")
    assert n == 0
    assert load_top_signals(conn).empty


def test_export_json_shape(tmp_path):
    conn = _conn()
    upsert_top_signals(conn, _sample_top_signal(), source_run_id="run1")
    export_path = export_top_signals(conn, path=tmp_path / "top_signals.json")

    payload = json.loads(export_path.read_text())
    assert payload["as_of_date"] == "2026-07-06"
    assert "forward_return_pct > 0.10" in payload["filter_rule"]
    assert "generated_at" in payload
    assert payload["summary"]["total_rows"] == 1
    assert payload["summary"]["quality_enriched_count"] == 1
    assert payload["rows"][0]["ticker"] == "SSSS.JK"


def test_export_import_round_trip(tmp_path):
    conn = _conn()
    upsert_top_signals(conn, _sample_top_signal(), source_run_id="run1")
    export_path = export_top_signals(conn, path=tmp_path / "top_signals.json")

    fresh_conn = _conn()
    n_imported = import_top_signals(fresh_conn, path=export_path)
    assert n_imported == 1
    assert len(load_top_signals(fresh_conn)) == 1


def test_import_is_idempotent(tmp_path):
    conn = _conn()
    upsert_top_signals(conn, _sample_top_signal(), source_run_id="run1")
    export_path = export_top_signals(conn, path=tmp_path / "top_signals.json")

    fresh_conn = _conn()
    import_top_signals(fresh_conn, path=export_path)
    n_second = import_top_signals(fresh_conn, path=export_path)
    assert n_second == 0
    assert len(load_top_signals(fresh_conn)) == 1


def test_import_missing_file_returns_zero(tmp_path):
    conn = _conn()
    n = import_top_signals(conn, path=tmp_path / "does_not_exist.json")
    assert n == 0
