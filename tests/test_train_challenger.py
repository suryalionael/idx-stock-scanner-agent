"""Tests for scripts/train_challenger.py's dataset/split feasibility guards
— see the crash this fixes: chronological_split() indexing into
`dates[int(n * train_frac) - 1]` raises IndexError on an empty `dates`
list, which is exactly what happens on a fresh/empty database (0 training
examples). A missing or insufficient training dataset is an expected
production state during early deployment, not an error — this must log and
exit cleanly (exit code 0), never raise.
"""
import sqlite3

import pandas as pd
import pytest

import scripts.train_challenger as train_challenger
from scripts.train_challenger import (
    _MIN_ROWS,
    _MIN_UNIQUE_DATES,
    _dataset_feasible,
    _split_feasible,
    chronological_split,
)
from stock_scanner.db.init_db import create_schema


def _df_with_dates(n_dates: int, rows_per_date: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="B")
    rows = []
    for i, d in enumerate(dates):
        for j in range(rows_per_date):
            rows.append({
                "signal_id": f"s{i}_{j}", "ticker": "AAAA.JK", "signal_date": d,
                "label_success": 1 if (i + j) % 3 == 0 else 0,
                "vol_ratio_20d": 1.0 + (j * 0.1), "squeeze_on": False,
                "atr_breakout": False, "vol_spike": False,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _dataset_feasible — checked immediately after load_training_examples()
# ---------------------------------------------------------------------------

def test_empty_dataframe_is_infeasible():
    df = pd.DataFrame(columns=["signal_date", "label_success"])
    reason = _dataset_feasible(df)
    assert reason is not None
    assert "No training examples" in reason


def test_insufficient_rows_is_infeasible():
    # Plenty of unique dates, but too few total rows.
    df = _df_with_dates(n_dates=_MIN_UNIQUE_DATES + 5, rows_per_date=1)
    assert len(df) < _MIN_ROWS
    reason = _dataset_feasible(df)
    assert reason is not None
    assert "training rows available" in reason
    assert str(_MIN_ROWS) in reason


def test_insufficient_unique_dates_is_infeasible():
    # Plenty of rows, but too few distinct trading days.
    df = _df_with_dates(n_dates=7, rows_per_date=20)
    assert len(df) >= _MIN_ROWS
    reason = _dataset_feasible(df)
    assert reason is not None
    assert "unique trading days" in reason
    assert "7" in reason
    assert str(_MIN_UNIQUE_DATES) in reason


def test_sufficient_dataset_is_feasible():
    df = _df_with_dates(n_dates=_MIN_UNIQUE_DATES + 10, rows_per_date=5)
    assert _dataset_feasible(df) is None


def test_dataset_feasible_never_raises_on_missing_columns():
    # Defensive: a malformed/empty df missing signal_date entirely must
    # still be handled by the emptiness check before ever touching
    # df["signal_date"] — never an uncaught KeyError.
    df = pd.DataFrame()
    assert _dataset_feasible(df) is not None


# ---------------------------------------------------------------------------
# chronological_split — confirms the exact reported crash, and that the
# guard is what actually prevents it (the split function itself is
# deliberately left unguarded, per its own docstring).
# ---------------------------------------------------------------------------

def test_chronological_split_raises_indexerror_on_empty_df():
    # This is the bug report's exact traceback, reproduced directly —
    # confirms _dataset_feasible() is checked BEFORE this call in main(),
    # not that this function itself changed.
    df = pd.DataFrame(columns=["signal_date"])
    with pytest.raises(IndexError):
        chronological_split(df, 0.6, 0.2)


def test_dataset_feasible_would_have_caught_the_crashing_case():
    df = pd.DataFrame(columns=["signal_date", "label_success"])
    assert _dataset_feasible(df) is not None  # main() would skip, never reach the line above


# ---------------------------------------------------------------------------
# _split_feasible — checked immediately after chronological_split()
# ---------------------------------------------------------------------------

def test_split_feasible_detects_empty_train():
    empty = pd.DataFrame({"signal_date": []})
    non_empty = pd.DataFrame({"signal_date": [pd.Timestamp("2026-01-01")]})
    reason = _split_feasible(empty, non_empty, non_empty)
    assert reason is not None
    assert "TRAIN" in reason


def test_split_feasible_detects_empty_val():
    empty = pd.DataFrame({"signal_date": []})
    non_empty = pd.DataFrame({"signal_date": [pd.Timestamp("2026-01-01")]})
    reason = _split_feasible(non_empty, empty, non_empty)
    assert reason is not None
    assert "VAL" in reason


def test_split_feasible_detects_empty_test():
    empty = pd.DataFrame({"signal_date": []})
    non_empty = pd.DataFrame({"signal_date": [pd.Timestamp("2026-01-01")]})
    reason = _split_feasible(non_empty, non_empty, empty)
    assert reason is not None
    assert "TEST" in reason


def test_split_feasible_passes_when_all_partitions_non_empty():
    non_empty = pd.DataFrame({"signal_date": [pd.Timestamp("2026-01-01")]})
    assert _split_feasible(non_empty, non_empty, non_empty) is None


def test_degenerate_split_from_skewed_dates_is_caught():
    # A realistic degenerate case: _dataset_feasible() passes (enough total
    # unique dates), but a non-default train_frac still starves one
    # partition. _split_feasible() must catch this even though
    # _dataset_feasible() didn't (and couldn't, generically).
    df = _df_with_dates(n_dates=_MIN_UNIQUE_DATES, rows_per_date=5)
    assert _dataset_feasible(df) is None
    train, val, test = chronological_split(df, train_frac=0.99, val_frac=0.005)
    reason = _split_feasible(train, val, test)
    assert reason is not None


# ---------------------------------------------------------------------------
# Sufficient dataset — chronological_split behaves exactly as before,
# unguarded logic untouched.
# ---------------------------------------------------------------------------

def test_sufficient_dataset_splits_normally_with_no_empty_partitions():
    df = _df_with_dates(n_dates=_MIN_UNIQUE_DATES + 10, rows_per_date=5)
    assert _dataset_feasible(df) is None

    train, val, test = chronological_split(df, train_frac=0.6, val_frac=0.2)
    assert _split_feasible(train, val, test) is None
    assert len(train) + len(val) + len(test) == len(df)
    assert train["signal_date"].max() <= val["signal_date"].min()
    assert val["signal_date"].max() <= test["signal_date"].min()


# ---------------------------------------------------------------------------
# main() end-to-end — the exact reported crash, reproduced against a real
# (in-memory) freshly-created database, confirming it now exits cleanly
# instead of raising IndexError. Never touches the real data/db/signals.db.
# ---------------------------------------------------------------------------

def test_main_exits_cleanly_on_fresh_empty_database(monkeypatch):
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    monkeypatch.setattr(train_challenger, "get_connection", lambda: conn)

    # Must not raise — this is the literal reported bug: an empty DB used
    # to reach chronological_split() and IndexError here. main() closes
    # the connection on the skip path (same as the pre-existing
    # sensitivity-battery-failed branch), so there's nothing further to
    # query afterward — not raising is the whole assertion.
    train_challenger.main()
