"""Tests for stock_scanner/db/reflection.py — reflection_observations
persistence. Mirrors tests/test_ai_lab_db.py's conventions."""
import json
import sqlite3

from stock_scanner.ai_lab.schemas import ObservationCategory, ReflectionObservation
from stock_scanner.db.init_db import create_schema
from stock_scanner.db.reflection import (
    export_reflection_report,
    import_reflection_observations,
    load_observations,
    upsert_observations,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    return conn


def _obs(**overrides) -> ReflectionObservation:
    base = dict(
        observation_id="obs1", category=ObservationCategory.MODEL_PERFORMANCE,
        title="AI model 'momentum_ai': consistently succeeds",
        description="20 recommendations ... 80.0% win rate vs 50.0% baseline ...",
        supporting_statistics={"n": 20, "n_success": 16, "win_rate": 0.8, "baseline_rate": 0.5},
        affected_trade_count=20, confidence=0.95, generated_at="2026-07-15T00:00:00+00:00",
    )
    base.update(overrides)
    return ReflectionObservation(**base)


# ---------------------------------------------------------------------------
# upsert / load — append-only
# ---------------------------------------------------------------------------

def test_upsert_inserts_row():
    conn = _conn()
    n = upsert_observations(conn, [_obs()])
    assert n == 1
    df = load_observations(conn)
    assert len(df) == 1
    assert df.iloc[0]["category"] == "model_performance"
    assert json.loads(df.iloc[0]["supporting_statistics"])["win_rate"] == 0.8


def test_upsert_empty_list_is_noop():
    conn = _conn()
    assert upsert_observations(conn, []) == 0


def test_upsert_is_append_only_new_run_adds_new_rows():
    """Two runs with different observation_ids (different generated_at)
    for the "same" underlying slice both persist — this table is a
    timeline, not a latest-snapshot upsert like ai_recommendations."""
    conn = _conn()
    upsert_observations(conn, [_obs(observation_id="run1", generated_at="2026-07-14T00:00:00+00:00")])
    upsert_observations(conn, [_obs(observation_id="run2", generated_at="2026-07-15T00:00:00+00:00")])
    df = load_observations(conn)
    assert len(df) == 2


def test_upsert_same_id_twice_does_not_duplicate():
    conn = _conn()
    upsert_observations(conn, [_obs()])
    upsert_observations(conn, [_obs(title="different title, same id")])
    df = load_observations(conn)
    assert len(df) == 1
    assert df.iloc[0]["title"] == "AI model 'momentum_ai': consistently succeeds"  # first write wins, INSERT OR IGNORE


def test_load_observations_filters_by_category():
    conn = _conn()
    upsert_observations(conn, [
        _obs(observation_id="a", category=ObservationCategory.MODEL_PERFORMANCE),
        _obs(observation_id="b", category=ObservationCategory.SECTOR_PERFORMANCE),
    ])
    model_only = load_observations(conn, category="model_performance")
    assert len(model_only) == 1
    assert model_only.iloc[0]["observation_id"] == "a"


def test_load_observations_respects_limit():
    conn = _conn()
    upsert_observations(conn, [_obs(observation_id=f"o{i}", generated_at=f"2026-07-{i+1:02d}T00:00:00+00:00") for i in range(5)])
    limited = load_observations(conn, limit=2)
    assert len(limited) == 2


# ---------------------------------------------------------------------------
# export / import round trip
# ---------------------------------------------------------------------------

def test_export_json_shape(tmp_path):
    conn = _conn()
    upsert_observations(conn, [_obs()])
    path = export_reflection_report(
        conn, path=tmp_path / "reflection_report.json",
        narrative={"overall_summary": "test", "prioritized_observation_ids": ["obs1"]},
        resolved_trade_count=40,
    )
    payload = json.loads(path.read_text())
    assert payload["summary"]["total_observations"] == 1
    assert payload["summary"]["by_category"] == {"model_performance": 1}
    assert payload["summary"]["resolved_trade_count"] == 40
    assert payload["narrative"]["overall_summary"] == "test"
    row = payload["observations"][0]
    assert row["supporting_statistics"]["win_rate"] == 0.8


def test_export_json_narrative_null_when_not_given(tmp_path):
    conn = _conn()
    upsert_observations(conn, [_obs()])
    path = export_reflection_report(conn, path=tmp_path / "reflection_report.json")
    payload = json.loads(path.read_text())
    assert payload["narrative"] is None


def test_export_with_no_observations_is_not_an_error(tmp_path):
    conn = _conn()
    path = export_reflection_report(conn, path=tmp_path / "reflection_report.json", resolved_trade_count=0)
    payload = json.loads(path.read_text())
    assert payload["summary"]["total_observations"] == 0
    assert payload["observations"] == []


def test_import_export_round_trip(tmp_path):
    conn1 = _conn()
    upsert_observations(conn1, [_obs()])
    path = export_reflection_report(conn1, path=tmp_path / "reflection_report.json")

    conn2 = _conn()
    n = import_reflection_observations(conn2, path=path)
    assert n == 1
    df = load_observations(conn2)
    assert len(df) == 1
    assert df.iloc[0]["observation_id"] == "obs1"


def test_import_is_idempotent_never_duplicates(tmp_path):
    conn1 = _conn()
    upsert_observations(conn1, [_obs()])
    path = export_reflection_report(conn1, path=tmp_path / "reflection_report.json")

    conn2 = _conn()
    import_reflection_observations(conn2, path=path)
    import_reflection_observations(conn2, path=path)
    assert len(load_observations(conn2)) == 1


def test_import_missing_file_returns_zero(tmp_path):
    conn = _conn()
    n = import_reflection_observations(conn, path=tmp_path / "does_not_exist.json")
    assert n == 0
