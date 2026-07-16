"""Tests for stock_scanner/db/knowledge_entries.py — knowledge_entries
persistence. Mirrors tests/test_db_hypotheses.py's conventions."""
import json
import sqlite3

import pytest

from stock_scanner.ai_lab.schemas import (
    EvidenceStrength,
    KnowledgeEntry,
    KnowledgeLifecycleStatus,
    KnowledgePromotionStatus,
)
from stock_scanner.db.init_db import create_schema
from stock_scanner.db.knowledge_entries import (
    export_knowledge_report,
    import_knowledge_entries,
    load_knowledge_entries,
    upsert_knowledge_entries,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    return conn


def _entry(**overrides) -> KnowledgeEntry:
    base = dict(
        knowledge_id="k1", created_at="2026-07-15T00:00:00+00:00",
        title="sector=Technology AND rsi14=High: strong",
        description="First observed 2026-07-10T00:00:00+00:00, independently confirmed 5 time(s)...",
        conditions=[["sector", "Technology"], ["rsi14", "High"]], originating_hypotheses=["h1", "h2"],
        evidence_count=5, cumulative_sample_size=20, cumulative_successes=16, cumulative_failures=4,
        average_win_rate=0.78, shrunk_win_rate=0.7, confidence_interval=[0.6, 0.9],
        first_seen="2026-07-10T00:00:00+00:00", last_confirmed="2026-07-14T00:00:00+00:00",
        confirmation_count=5, contradiction_count=0, evidence_strength=EvidenceStrength.STRONG,
        lifecycle_status=KnowledgeLifecycleStatus.STRONG, previous_lifecycle_status=KnowledgeLifecycleStatus.CONFIRMED,
    )
    base.update(overrides)
    return KnowledgeEntry(**base)


# ---------------------------------------------------------------------------
# upsert / load — append-only
# ---------------------------------------------------------------------------

def test_upsert_inserts_row():
    conn = _conn()
    n = upsert_knowledge_entries(conn, [_entry()])
    assert n == 1
    df = load_knowledge_entries(conn)
    assert len(df) == 1
    assert df.iloc[0]["lifecycle_status"] == "strong"
    assert json.loads(df.iloc[0]["conditions"]) == [["sector", "Technology"], ["rsi14", "High"]]
    assert json.loads(df.iloc[0]["confidence_interval"]) == [0.6, 0.9]


def test_upsert_empty_list_is_noop():
    conn = _conn()
    assert upsert_knowledge_entries(conn, []) == 0


def test_upsert_is_append_only_new_run_adds_new_rows():
    conn = _conn()
    upsert_knowledge_entries(conn, [_entry(knowledge_id="run1", created_at="2026-07-14T00:00:00+00:00")])
    upsert_knowledge_entries(conn, [_entry(knowledge_id="run2", created_at="2026-07-15T00:00:00+00:00")])
    df = load_knowledge_entries(conn)
    assert len(df) == 2


def test_upsert_same_id_twice_does_not_duplicate():
    conn = _conn()
    upsert_knowledge_entries(conn, [_entry()])
    upsert_knowledge_entries(conn, [_entry(title="different title, same id")])
    df = load_knowledge_entries(conn)
    assert len(df) == 1
    assert "different title" not in df.iloc[0]["title"]  # first write wins, INSERT OR IGNORE


def test_load_knowledge_entries_filters_by_lifecycle_status():
    conn = _conn()
    upsert_knowledge_entries(conn, [
        _entry(knowledge_id="a", lifecycle_status=KnowledgeLifecycleStatus.STRONG),
        _entry(knowledge_id="b", lifecycle_status=KnowledgeLifecycleStatus.EMERGING,
               previous_lifecycle_status=None, confirmation_count=1),
    ])
    strong_only = load_knowledge_entries(conn, lifecycle_status="strong")
    assert len(strong_only) == 1
    assert strong_only.iloc[0]["knowledge_id"] == "a"


def test_load_knowledge_entries_respects_limit():
    conn = _conn()
    upsert_knowledge_entries(conn, [
        _entry(knowledge_id=f"k{i}", created_at=f"2026-07-{i+1:02d}T00:00:00+00:00") for i in range(5)
    ])
    limited = load_knowledge_entries(conn, limit=2)
    assert len(limited) == 2


# ---------------------------------------------------------------------------
# export / import round trip
# ---------------------------------------------------------------------------

def test_export_json_shape(tmp_path):
    conn = _conn()
    upsert_knowledge_entries(conn, [_entry()])
    path = export_knowledge_report(
        conn, path=tmp_path / "knowledge_report.json",
        narrative={"overall_summary": "test", "organized_groups": [], "highlighted_changes": []},
        resolved_trade_count=40,
    )
    payload = json.loads(path.read_text())
    assert payload["summary"]["total_entries"] == 1
    assert payload["summary"]["by_lifecycle_status"] == {"strong": 1}
    assert payload["summary"]["resolved_trade_count"] == 40
    assert payload["narrative"]["overall_summary"] == "test"
    row = payload["entries"][0]
    assert row["conditions"] == [["sector", "Technology"], ["rsi14", "High"]]
    assert row["confidence_interval"] == [0.6, 0.9]
    assert row["originating_hypotheses"] == ["h1", "h2"]


def test_export_json_narrative_null_when_not_given(tmp_path):
    conn = _conn()
    upsert_knowledge_entries(conn, [_entry()])
    path = export_knowledge_report(conn, path=tmp_path / "knowledge_report.json")
    payload = json.loads(path.read_text())
    assert payload["narrative"] is None


def test_export_with_no_entries_is_not_an_error(tmp_path):
    conn = _conn()
    path = export_knowledge_report(conn, path=tmp_path / "knowledge_report.json", resolved_trade_count=0)
    payload = json.loads(path.read_text())
    assert payload["summary"]["total_entries"] == 0
    assert payload["entries"] == []


def test_import_export_round_trip(tmp_path):
    conn1 = _conn()
    upsert_knowledge_entries(conn1, [_entry()])
    path = export_knowledge_report(conn1, path=tmp_path / "knowledge_report.json")

    conn2 = _conn()
    n = import_knowledge_entries(conn2, path=path)
    assert n == 1
    df = load_knowledge_entries(conn2)
    assert len(df) == 1
    assert df.iloc[0]["knowledge_id"] == "k1"


def test_import_is_idempotent_never_duplicates(tmp_path):
    conn1 = _conn()
    upsert_knowledge_entries(conn1, [_entry()])
    path = export_knowledge_report(conn1, path=tmp_path / "knowledge_report.json")

    conn2 = _conn()
    import_knowledge_entries(conn2, path=path)
    import_knowledge_entries(conn2, path=path)
    assert len(load_knowledge_entries(conn2)) == 1


def test_import_missing_file_returns_zero(tmp_path):
    conn = _conn()
    n = import_knowledge_entries(conn, path=tmp_path / "does_not_exist.json")
    assert n == 0


# ---------------------------------------------------------------------------
# promotion_status — deployment gate, orthogonal to lifecycle_status
# ---------------------------------------------------------------------------

def test_new_entry_defaults_to_candidate_promotion_status():
    conn = _conn()
    upsert_knowledge_entries(conn, [_entry()])  # no promotion_status passed
    df = load_knowledge_entries(conn)
    assert df.iloc[0]["promotion_status"] == "candidate"


def test_export_includes_promotion_status(tmp_path):
    conn = _conn()
    upsert_knowledge_entries(conn, [_entry()])
    path = export_knowledge_report(conn, path=tmp_path / "knowledge_report.json")
    payload = json.loads(path.read_text())
    assert payload["entries"][0]["promotion_status"] == "candidate"


def test_import_defaults_missing_promotion_status_to_candidate(tmp_path):
    # Simulates a real knowledge_report.json written before promotion_status
    # existed — the row has no such key at all. Import must not crash on
    # the NOT NULL constraint, and the row must land as 'candidate' (never
    # NULL, never silently 'promoted').
    path = tmp_path / "knowledge_report.json"
    old_shaped_row = {
        "knowledge_id": "legacy1", "created_at": "2026-07-01T00:00:00+00:00",
        "title": "legacy entry", "description": "pre-promotion-gate row",
        "conditions": [["vol_spike", "True"]], "originating_hypotheses": ["h1"],
        "evidence_count": 3, "cumulative_sample_size": 20, "cumulative_successes": 15,
        "cumulative_failures": 5, "average_win_rate": 0.75, "shrunk_win_rate": 0.7,
        "confidence_interval": [0.6, 0.85], "first_seen": "2026-06-01T00:00:00+00:00",
        "last_confirmed": "2026-07-01T00:00:00+00:00", "confirmation_count": 3,
        "contradiction_count": 0, "evidence_strength": "STRONG", "lifecycle_status": "strong",
        "previous_lifecycle_status": "confirmed", "llm_note": None,
        # promotion_status intentionally absent
    }
    path.write_text(json.dumps({"entries": [old_shaped_row]}))

    conn = _conn()
    n = import_knowledge_entries(conn, path=path)
    assert n == 1
    df = load_knowledge_entries(conn)
    assert df.iloc[0]["promotion_status"] == "candidate"


def test_import_coerces_unrecognized_promotion_status_to_candidate(tmp_path):
    # A row with promotion_status PRESENT but not one of the four valid
    # values (corrupt data, a hand-edited file, a future schema drift) —
    # distinct from the missing-key case above. Must still not crash, and
    # must still land as 'candidate', not pass the invalid string through
    # to the database (which would otherwise hit the new CHECK constraint
    # and raise).
    path = tmp_path / "knowledge_report.json"
    row = dict(
        knowledge_id="corrupt1", created_at="2026-07-01T00:00:00+00:00",
        title="t", description="d", conditions=[["vol_spike", "True"]],
        originating_hypotheses=[], evidence_count=1, cumulative_sample_size=10,
        cumulative_successes=8, cumulative_failures=2, average_win_rate=0.8,
        shrunk_win_rate=0.75, confidence_interval=[0.5, 0.9],
        first_seen="2026-07-01T00:00:00+00:00", last_confirmed="2026-07-01T00:00:00+00:00",
        confirmation_count=1, contradiction_count=0, evidence_strength="STRONG",
        lifecycle_status="strong", previous_lifecycle_status=None, llm_note=None,
        promotion_status="approved",  # not a real KnowledgePromotionStatus value
    )
    path.write_text(json.dumps({"entries": [row]}))

    conn = _conn()
    n = import_knowledge_entries(conn, path=path)
    assert n == 1
    df = load_knowledge_entries(conn)
    assert df.iloc[0]["promotion_status"] == "candidate"


def test_database_rejects_invalid_promotion_status_via_check_constraint():
    # Bypasses the Python normalization layer entirely — a raw INSERT
    # attempting an out-of-vocabulary value must be rejected by the
    # database itself, not just by application code upstream of it.
    conn = _conn()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO knowledge_entries
               (knowledge_id, created_at, title, description, conditions,
                originating_hypotheses, evidence_count, cumulative_sample_size,
                cumulative_successes, cumulative_failures, average_win_rate, shrunk_win_rate,
                confidence_interval, first_seen, last_confirmed, confirmation_count,
                contradiction_count, lifecycle_status, promotion_status)
               VALUES ('x', '2026-07-01', 't', 'd', '[]', '[]', 0, 0, 0, 0, 0.5, 0.5,
                       '[0,1]', '2026-07-01', '2026-07-01', 0, 0, 'strong', 'not_a_real_status')"""
        )


def test_database_accepts_every_valid_promotion_status():
    # The CHECK constraint's allowlist must stay in lockstep with
    # KnowledgePromotionStatus — this fails loudly if the two ever drift.
    conn = _conn()
    for i, status in enumerate(KnowledgePromotionStatus):
        conn.execute(
            """INSERT INTO knowledge_entries
               (knowledge_id, created_at, title, description, conditions,
                originating_hypotheses, evidence_count, cumulative_sample_size,
                cumulative_successes, cumulative_failures, average_win_rate, shrunk_win_rate,
                confidence_interval, first_seen, last_confirmed, confirmation_count,
                contradiction_count, lifecycle_status, promotion_status)
               VALUES (?, '2026-07-01', 't', 'd', '[]', '[]', 0, 0, 0, 0, 0.5, 0.5,
                       '[0,1]', '2026-07-01', '2026-07-01', 0, 0, 'strong', ?)""",
            (f"x{i}", status.value),
        )
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM knowledge_entries").fetchone()[0] == len(KnowledgePromotionStatus)
