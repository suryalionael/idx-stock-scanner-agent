"""Tests for the review CLI's safety polish — see
docs/LEARNING_AGENT_RUNBOOK.md. These cover two real gaps found while
manually exercising the tool: a missing knowledge_base table crashing with
a raw traceback instead of a clear message, and a typo'd --status filter
silently reporting "zero rows" indistinguishably from a correctly-spelled
status with no candidates yet.
"""
import sqlite3

import pytest

from scripts.review_knowledge_base import _safe_load, _validate_status_filter
from stock_scanner.db.init_db import create_schema
from stock_scanner.db.knowledge_base import write_hypotheses
from stock_scanner.learning.hypothesis_agent import Hypothesis


def _hypothesis() -> Hypothesis:
    return Hypothesis(
        hypothesis="Test", confidence=0.7, supporting_trades=10,
        affected_sector=None, expected_effect="Higher win rate",
        status="candidate", source_cluster_id="c1",
    )


# ---------------------------------------------------------------------------
# _safe_load — missing table must not raise past this function
# ---------------------------------------------------------------------------

def test_safe_load_returns_none_when_table_missing(capsys):
    conn = sqlite3.connect(":memory:")   # no create_schema() — table genuinely absent
    result = _safe_load(conn)
    assert result is None
    captured = capsys.readouterr()
    assert "run_learning_agent.py" in captured.out or "run_hypothesis_agent.py" in captured.out


def test_safe_load_returns_dataframe_when_table_exists():
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    write_hypotheses(conn, [_hypothesis()], source_run_id="run1")
    result = _safe_load(conn)
    assert result is not None
    assert len(result) == 1


def test_safe_load_empty_table_returns_empty_dataframe_not_none():
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    result = _safe_load(conn)
    assert result is not None
    assert result.empty


# ---------------------------------------------------------------------------
# _validate_status_filter — typo protection
# ---------------------------------------------------------------------------

def test_validate_status_filter_accepts_none():
    assert _validate_status_filter(None) is True


def test_validate_status_filter_accepts_known_status():
    assert _validate_status_filter("candidate") is True
    assert _validate_status_filter("archived") is True


def test_validate_status_filter_rejects_typo_and_warns(capsys):
    assert _validate_status_filter("condidate") is False   # typo of "candidate"
    captured = capsys.readouterr()
    assert "not a recognized status" in captured.out
    assert "candidate" in captured.out   # valid options listed for the user to compare against
