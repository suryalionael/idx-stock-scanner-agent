"""Tests for stock_scanner/db/hypotheses.py — validated_hypotheses
persistence. Mirrors tests/test_db_reflection.py's conventions."""
import json
import sqlite3

from stock_scanner.ai_lab.schemas import EvidenceStrength, Hypothesis, HypothesisStatus
from stock_scanner.db.hypotheses import (
    export_hypotheses_report,
    import_hypotheses,
    load_hypotheses,
    upsert_hypotheses,
)
from stock_scanner.db.init_db import create_schema


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    return conn


def _hyp(**overrides) -> Hypothesis:
    base = dict(
        hypothesis_id="h1", created_at="2026-07-15T00:00:00+00:00",
        description="Recommendations where sector=Technology AND rsi14=High realized a 20.0% win rate...",
        conditions=[["sector", "Technology"], ["rsi14", "High"]],
        sample_size=20, successes=4, failures=16, win_rate=0.2, shrunk_win_rate=0.3,
        wilson_lower=0.08, wilson_upper=0.42, fisher_p=0.001, bh_adjusted_p=0.002,
        evidence_strength=EvidenceStrength.STRONG, status=HypothesisStatus.VALIDATED,
        source_reflection_ids=["r1"], metadata_json={"interaction_order": 2},
    )
    base.update(overrides)
    return Hypothesis(**base)


# ---------------------------------------------------------------------------
# upsert / load — append-only
# ---------------------------------------------------------------------------

def test_upsert_inserts_row():
    conn = _conn()
    n = upsert_hypotheses(conn, [_hyp()])
    assert n == 1
    df = load_hypotheses(conn)
    assert len(df) == 1
    assert df.iloc[0]["status"] == "validated"
    assert json.loads(df.iloc[0]["conditions"]) == [["sector", "Technology"], ["rsi14", "High"]]


def test_upsert_empty_list_is_noop():
    conn = _conn()
    assert upsert_hypotheses(conn, []) == 0


def test_upsert_is_append_only_new_run_adds_new_rows():
    conn = _conn()
    upsert_hypotheses(conn, [_hyp(hypothesis_id="run1", created_at="2026-07-14T00:00:00+00:00")])
    upsert_hypotheses(conn, [_hyp(hypothesis_id="run2", created_at="2026-07-15T00:00:00+00:00")])
    df = load_hypotheses(conn)
    assert len(df) == 2


def test_upsert_same_id_twice_does_not_duplicate():
    conn = _conn()
    upsert_hypotheses(conn, [_hyp()])
    upsert_hypotheses(conn, [_hyp(description="different description, same id")])
    df = load_hypotheses(conn)
    assert len(df) == 1
    assert "different description" not in df.iloc[0]["description"]  # first write wins, INSERT OR IGNORE


def test_load_hypotheses_filters_by_status():
    conn = _conn()
    upsert_hypotheses(conn, [
        _hyp(hypothesis_id="a", status=HypothesisStatus.VALIDATED),
        _hyp(hypothesis_id="b", status=HypothesisStatus.REJECTED, evidence_strength=None,
             rejection_reason="not significant", failed_gate="not_significant"),
    ])
    validated_only = load_hypotheses(conn, status="validated")
    assert len(validated_only) == 1
    assert validated_only.iloc[0]["hypothesis_id"] == "a"
    rejected_only = load_hypotheses(conn, status="rejected")
    assert len(rejected_only) == 1
    assert rejected_only.iloc[0]["rejection_reason"] == "not significant"


def test_load_hypotheses_respects_limit():
    conn = _conn()
    upsert_hypotheses(conn, [
        _hyp(hypothesis_id=f"h{i}", created_at=f"2026-07-{i+1:02d}T00:00:00+00:00") for i in range(5)
    ])
    limited = load_hypotheses(conn, limit=2)
    assert len(limited) == 2


# ---------------------------------------------------------------------------
# export / import round trip
# ---------------------------------------------------------------------------

def test_export_json_shape(tmp_path):
    conn = _conn()
    upsert_hypotheses(conn, [_hyp()])
    path = export_hypotheses_report(
        conn, path=tmp_path / "hypotheses_report.json",
        narrative={"overall_summary": "test", "prioritized_hypothesis_ids": ["h1"], "clusters": []},
        resolved_trade_count=40,
    )
    payload = json.loads(path.read_text())
    assert payload["summary"]["total_hypotheses"] == 1
    assert payload["summary"]["by_status"] == {"validated": 1}
    assert payload["summary"]["resolved_trade_count"] == 40
    assert payload["narrative"]["overall_summary"] == "test"
    row = payload["hypotheses"][0]
    assert row["conditions"] == [["sector", "Technology"], ["rsi14", "High"]]
    assert row["metadata_json"]["interaction_order"] == 2


def test_export_json_narrative_null_when_not_given(tmp_path):
    conn = _conn()
    upsert_hypotheses(conn, [_hyp()])
    path = export_hypotheses_report(conn, path=tmp_path / "hypotheses_report.json")
    payload = json.loads(path.read_text())
    assert payload["narrative"] is None


def test_export_with_no_hypotheses_is_not_an_error(tmp_path):
    conn = _conn()
    path = export_hypotheses_report(conn, path=tmp_path / "hypotheses_report.json", resolved_trade_count=0)
    payload = json.loads(path.read_text())
    assert payload["summary"]["total_hypotheses"] == 0
    assert payload["hypotheses"] == []


def test_import_export_round_trip(tmp_path):
    conn1 = _conn()
    upsert_hypotheses(conn1, [_hyp()])
    path = export_hypotheses_report(conn1, path=tmp_path / "hypotheses_report.json")

    conn2 = _conn()
    n = import_hypotheses(conn2, path=path)
    assert n == 1
    df = load_hypotheses(conn2)
    assert len(df) == 1
    assert df.iloc[0]["hypothesis_id"] == "h1"


def test_import_is_idempotent_never_duplicates(tmp_path):
    conn1 = _conn()
    upsert_hypotheses(conn1, [_hyp()])
    path = export_hypotheses_report(conn1, path=tmp_path / "hypotheses_report.json")

    conn2 = _conn()
    import_hypotheses(conn2, path=path)
    import_hypotheses(conn2, path=path)
    assert len(load_hypotheses(conn2)) == 1


def test_import_missing_file_returns_zero(tmp_path):
    conn = _conn()
    n = import_hypotheses(conn, path=tmp_path / "does_not_exist.json")
    assert n == 0
