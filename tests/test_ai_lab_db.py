"""Tests for stock_scanner/db/ai_lab.py — ai_recommendations +
ai_learning_events persistence."""
import json
import sqlite3

import pytest

from stock_scanner.ai_lab.schemas import (
    AIRecommendation,
    ConfidenceBreakdown,
    DecisionTrace,
    HistoricalComparison,
    HistoricalComparisonVerdict,
    RecommendationLevel,
    RecommendationStatus,
    RiskLevel,
)
from stock_scanner.db.ai_lab import (
    export_ai_recommendations,
    export_learning_events,
    import_ai_recommendations,
    import_learning_events,
    load_learning_events,
    load_recommendations,
    log_learning_event,
    update_recommendation_status,
    upsert_recommendations,
)
from stock_scanner.db.init_db import create_schema


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    return conn


def _trace(**overrides) -> DecisionTrace:
    base = dict(technical_score=80.0, statistical_score=20.0, pattern_similarity_score=90.0,
                risk_score=60.0, final_score=55.0)
    base.update(overrides)
    return DecisionTrace(**base)


def _confidence(**overrides) -> ConfidenceBreakdown:
    base = dict(technical=0.8, statistical=0.2, pattern_similarity=0.9,
                risk_adjustment=-0.18, final_confidence=0.45)
    base.update(overrides)
    return ConfidenceBreakdown(**base)


def _comparison(**overrides) -> HistoricalComparison:
    base = dict(pattern_description="MA Alignment + ATR Breakout", sample_size=115, win_rate=0.13,
                ci_lower=0.081, ci_upper=0.187, verdict=HistoricalComparisonVerdict.SIMILAR,
                explanation="test")
    base.update(overrides)
    return HistoricalComparison(**base)


def _rec(**overrides) -> AIRecommendation:
    base = dict(
        id="rec1", ticker="BBCA.JK", ai_model="momentum_ai", score=55.0, confidence=0.45,
        recommendation=RecommendationLevel.WATCH, reasoning={"why": "test"},
        decision_trace=_trace(), confidence_breakdown=_confidence(), historical_comparison=_comparison(),
        expected_return=None, risk_level=RiskLevel.MEDIUM, generated_date="2026-07-14",
        status=RecommendationStatus.PENDING, model="oc/hy3-free",
    )
    base.update(overrides)
    return AIRecommendation(**base)


# ---------------------------------------------------------------------------
# ai_recommendations — upsert idempotency + lifecycle preservation
# ---------------------------------------------------------------------------

def test_upsert_inserts_row():
    conn = _conn()
    n = upsert_recommendations(conn, [_rec()])
    assert n == 1
    df = load_recommendations(conn)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "BBCA.JK"
    assert json.loads(df.iloc[0]["reasoning"]) == {"why": "test"}
    assert json.loads(df.iloc[0]["decision_trace"])["final_score"] == 55.0


def test_upsert_overwrites_content_not_duplicates():
    conn = _conn()
    upsert_recommendations(conn, [_rec()])
    upsert_recommendations(conn, [_rec(score=95.0)])
    df = load_recommendations(conn)
    assert len(df) == 1
    assert df.iloc[0]["score"] == 95.0


def test_upsert_never_resets_lifecycle_fields_set_by_status_update():
    conn = _conn()
    upsert_recommendations(conn, [_rec()])
    update_recommendation_status(conn, "rec1", "ACTIVE", entry_price=1000.0)

    # A regeneration re-run for the same day (e.g. re-running the script)
    # must not silently reset status/entry_price back to PENDING/None.
    upsert_recommendations(conn, [_rec(score=91.0)])

    df = load_recommendations(conn)
    row = df.iloc[0]
    assert row["status"] == "ACTIVE"
    assert row["entry_price"] == 1000.0
    assert row["score"] == 91.0


def test_upsert_empty_list_is_noop():
    conn = _conn()
    n = upsert_recommendations(conn, [])
    assert n == 0


def test_load_recommendations_filters_by_ai_model_and_status():
    conn = _conn()
    upsert_recommendations(conn, [_rec(id="a", ai_model="momentum_ai"), _rec(id="b", ai_model="breakout_ai")])
    update_recommendation_status(conn, "a", "ACTIVE")

    momentum_only = load_recommendations(conn, ai_model="momentum_ai")
    assert len(momentum_only) == 1
    active_only = load_recommendations(conn, status="ACTIVE")
    assert len(active_only) == 1
    assert active_only.iloc[0]["id"] == "a"


# ---------------------------------------------------------------------------
# update_recommendation_status
# ---------------------------------------------------------------------------

def test_update_status_returns_zero_for_nonexistent_id():
    conn = _conn()
    n = update_recommendation_status(conn, "does-not-exist", "ACTIVE")
    assert n == 0


def test_update_status_returns_one_on_success():
    conn = _conn()
    upsert_recommendations(conn, [_rec()])
    assert update_recommendation_status(conn, "rec1", "CLOSED", return_percentage=0.12) == 1
    row = load_recommendations(conn).iloc[0]
    assert row["status"] == "CLOSED"
    assert row["return_percentage"] == 0.12


def test_update_status_rejects_unknown_status():
    conn = _conn()
    upsert_recommendations(conn, [_rec()])
    with pytest.raises(ValueError):
        update_recommendation_status(conn, "rec1", "NOT_A_REAL_STATUS")


# ---------------------------------------------------------------------------
# export / import round trip
# ---------------------------------------------------------------------------

def test_export_json_shape(tmp_path):
    conn = _conn()
    upsert_recommendations(conn, [_rec()])
    path = export_ai_recommendations(conn, path=tmp_path / "ai_recommendations.json")
    payload = json.loads(path.read_text())
    assert payload["as_of_date"] == "2026-07-14"
    assert payload["summary"]["total_rows"] == 1
    assert payload["summary"]["by_status"] == {"PENDING": 1}
    row = payload["rows"][0]
    assert row["reasoning"] == {"why": "test"}
    assert row["decision_trace"]["final_score"] == 55.0
    assert row["confidence_breakdown"]["final_confidence"] == 0.45
    assert row["historical_comparison"]["verdict"] == "similar"


def test_export_import_round_trip(tmp_path):
    conn = _conn()
    upsert_recommendations(conn, [_rec()])
    path = export_ai_recommendations(conn, path=tmp_path / "ai_recommendations.json")

    fresh_conn = _conn()
    n = import_ai_recommendations(fresh_conn, path=path)
    assert n == 1
    df = load_recommendations(fresh_conn)
    assert len(df) == 1
    assert json.loads(df.iloc[0]["decision_trace"])["final_score"] == 55.0


def test_import_does_not_overwrite_existing_rows(tmp_path):
    # import is INSERT OR IGNORE only — it must never clobber lifecycle
    # state already present in a live DB.
    conn = _conn()
    upsert_recommendations(conn, [_rec()])
    path = export_ai_recommendations(conn, path=tmp_path / "ai_recommendations.json")
    update_recommendation_status(conn, "rec1", "ACTIVE")

    n = import_ai_recommendations(conn, path=path)
    assert n == 0
    assert load_recommendations(conn).iloc[0]["status"] == "ACTIVE"


def test_import_missing_file_returns_zero(tmp_path):
    conn = _conn()
    n = import_ai_recommendations(conn, path=tmp_path / "does_not_exist.json")
    assert n == 0


# ---------------------------------------------------------------------------
# ai_learning_events — append-only
# ---------------------------------------------------------------------------

def test_log_learning_event_appends():
    conn = _conn()
    log_learning_event(conn, "hypothesis_generated", "Generated 3 recommendations.", ai_model="momentum_ai")
    df = load_learning_events(conn)
    assert len(df) == 1
    assert df.iloc[0]["event_type"] == "hypothesis_generated"


def test_learning_events_export_import_round_trip(tmp_path):
    conn = _conn()
    log_learning_event(conn, "pattern_learned", "test event", metadata={"n": 5})
    path = export_learning_events(conn, path=tmp_path / "ai_learning_events.json")

    fresh_conn = _conn()
    n = import_learning_events(fresh_conn, path=path)
    assert n == 1
    df = load_learning_events(fresh_conn)
    assert len(df) == 1


def test_load_learning_events_respects_limit():
    conn = _conn()
    for i in range(5):
        log_learning_event(conn, "hypothesis_generated", f"event {i}")
    df = load_learning_events(conn, limit=2)
    assert len(df) == 2
