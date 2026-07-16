"""Tests for stock_scanner/ai_lab/resolver.py — Performance Tracker
automation (activate_pending / resolve_active)."""
import pandas as pd
import pytest

from stock_scanner.ai_lab.resolver import activate_pending, resolve_active


def _raw_parquet(tmp_path, ticker: str, rows: list[dict]):
    df = pd.DataFrame(rows)
    path = tmp_path / f"{ticker}.parquet"
    df.to_parquet(path)
    return tmp_path


def _pending_row(**overrides) -> pd.DataFrame:
    base = dict(id="rec1", ticker="BBCA.JK", generated_date="2026-07-01")
    base.update(overrides)
    return pd.DataFrame([base])


def _active_row(**overrides) -> pd.DataFrame:
    base = dict(id="rec1", ticker="BBCA.JK", generated_date="2026-07-01", entry_price=1000.0)
    base.update(overrides)
    return pd.DataFrame([base])


# ---------------------------------------------------------------------------
# activate_pending
# ---------------------------------------------------------------------------

def test_activate_sets_entry_price_when_generated_date_close_available(tmp_path):
    raw_dir = _raw_parquet(tmp_path, "BBCA.JK", [
        {"date": "2026-07-01", "open": 990, "high": 1010, "low": 985, "close": 1000},
    ])
    out = activate_pending(_pending_row(), raw_dir)
    assert len(out) == 1
    assert out[0]["id"] == "rec1"
    assert out[0]["status"] == "ACTIVE"
    assert out[0]["entry_price"] == 1000.0
    assert out[0]["highest_price"] == 1000.0
    assert out[0]["lowest_price"] == 1000.0
    assert out[0]["holding_days"] == 0


def test_activate_skips_when_generated_date_not_yet_in_raw_data(tmp_path):
    raw_dir = _raw_parquet(tmp_path, "BBCA.JK", [
        {"date": "2026-06-30", "open": 990, "high": 1010, "low": 985, "close": 995},
    ])
    out = activate_pending(_pending_row(), raw_dir)
    assert out == []


def test_activate_skips_when_raw_file_missing(tmp_path):
    out = activate_pending(_pending_row(), tmp_path)
    assert out == []


def test_activate_empty_input_is_noop(tmp_path):
    out = activate_pending(pd.DataFrame(), tmp_path)
    assert out == []


# ---------------------------------------------------------------------------
# resolve_active — TP / SL / still-open / expiry
# ---------------------------------------------------------------------------

def test_resolve_hits_tp_and_marks_win(tmp_path):
    # entry=1000, tp = 1000*(1+0.03*1.8) = 1054, hit on day 2
    raw_dir = _raw_parquet(tmp_path, "BBCA.JK", [
        {"date": "2026-07-02", "open": 1000, "high": 1020, "low": 995, "close": 1010},
        {"date": "2026-07-03", "open": 1010, "high": 1060, "low": 1005, "close": 1055},
    ])
    out = resolve_active(_active_row(), raw_dir)
    assert len(out) == 1
    row = out[0]
    assert row["status"] == "CLOSED"
    assert row["exit_price"] == pytest.approx(1054.0)
    assert row["trade_outcome"] == "WIN"
    assert row["holding_days"] == 2
    assert row["return_percentage"] > 0


def test_resolve_hits_sl_and_marks_loss(tmp_path):
    # entry=1000, cl = 1000*(1-0.03) = 970, hit on day 1
    raw_dir = _raw_parquet(tmp_path, "BBCA.JK", [
        {"date": "2026-07-02", "open": 1000, "high": 1005, "low": 960, "close": 965},
    ])
    out = resolve_active(_active_row(), raw_dir)
    assert len(out) == 1
    row = out[0]
    assert row["status"] == "CLOSED"
    assert row["exit_price"] == pytest.approx(970.0)
    assert row["trade_outcome"] == "LOSS"
    assert row["return_percentage"] < 0


def test_resolve_leaves_active_and_tracks_running_mfe_mae_when_short_of_horizon(tmp_path):
    # No TP/SL hit, only 2 rows available (< default horizon_days=10) -> stays open.
    raw_dir = _raw_parquet(tmp_path, "BBCA.JK", [
        {"date": "2026-07-02", "open": 1000, "high": 1020, "low": 990, "close": 1010},
        {"date": "2026-07-03", "open": 1010, "high": 1015, "low": 980, "close": 1000},
    ])
    out = resolve_active(_active_row(), raw_dir)
    assert len(out) == 1
    row = out[0]
    assert "status" not in row
    assert "exit_price" not in row
    assert "trade_outcome" not in row
    assert row["highest_price"] == 1020.0
    assert row["lowest_price"] == 980.0
    assert row["max_runup_pct"] == pytest.approx(2.0)
    assert row["max_drawdown_pct"] == pytest.approx(2.0)
    assert row["holding_days"] == 2


def test_resolve_expires_at_horizon_with_no_hit_and_can_still_be_a_win(tmp_path):
    # 10 flat, non-triggering rows, mark-to-market close on the 10th day
    # ends up above entry -> EXPIRED but WIN, proving trade_outcome is
    # independent of status.
    rows = [
        {"date": f"2026-07-{2 + i:02d}", "open": 1000, "high": 1010, "low": 995, "close": 1000 + i}
        for i in range(10)
    ]
    raw_dir = _raw_parquet(tmp_path, "BBCA.JK", rows)
    out = resolve_active(_active_row(), raw_dir, horizon_days=10, risk_pct=3.0)
    assert len(out) == 1
    row = out[0]
    assert row["status"] == "EXPIRED"
    assert row["exit_price"] == pytest.approx(1009.0)
    assert row["trade_outcome"] == "WIN"
    assert row["holding_days"] == 10


def test_resolve_skips_when_no_new_forward_data(tmp_path):
    raw_dir = _raw_parquet(tmp_path, "BBCA.JK", [
        {"date": "2026-07-01", "open": 1000, "high": 1010, "low": 995, "close": 1000},
    ])
    out = resolve_active(_active_row(), raw_dir)
    assert out == []


def test_resolve_skips_when_raw_file_missing(tmp_path):
    out = resolve_active(_active_row(), tmp_path)
    assert out == []


def test_resolve_empty_input_is_noop(tmp_path):
    out = resolve_active(pd.DataFrame(), tmp_path)
    assert out == []
