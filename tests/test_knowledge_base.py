"""Tests for the knowledge_base read/write path — the only table
stock_scanner/learning/ is allowed to touch. See
docs/LEARNING_AGENT_ARCHITECTURE.md.
"""
import json
import sqlite3

import pytest

from stock_scanner.db.init_db import create_schema
from stock_scanner.db.knowledge_base import (
    export_knowledge_base,
    import_knowledge_base,
    load_knowledge_base,
    update_status,
    write_hypotheses,
)
from stock_scanner.learning.hypothesis_agent import Hypothesis


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    return conn


def _hypothesis(source_cluster_id: str = "cluster1") -> Hypothesis:
    return Hypothesis(
        hypothesis="Test hypothesis", confidence=0.7, supporting_trades=18,
        affected_sector=None, expected_effect="Higher win rate",
        status="candidate", source_cluster_id=source_cluster_id,
    )


def test_write_hypotheses_inserts_rows():
    conn = _conn()
    n = write_hypotheses(conn, [_hypothesis()], source_run_id="run1")
    assert n == 1
    df = load_knowledge_base(conn)
    assert len(df) == 1
    assert df.iloc[0]["status"] == "candidate"
    assert df.iloc[0]["supporting_trades"] == 18


def test_load_knowledge_base_filters_by_status():
    conn = _conn()
    write_hypotheses(conn, [_hypothesis("c1")], source_run_id="run1")
    hid = load_knowledge_base(conn).iloc[0]["hypothesis_id"]
    update_status(conn, hid, "tested_passed")
    write_hypotheses(conn, [_hypothesis("c2")], source_run_id="run2")

    candidates = load_knowledge_base(conn, status="candidate")
    passed = load_knowledge_base(conn, status="tested_passed")
    assert len(candidates) == 1
    assert len(passed) == 1


def test_update_status_sets_reviewed_by():
    conn = _conn()
    write_hypotheses(conn, [_hypothesis()], source_run_id="run1")
    hid = load_knowledge_base(conn).iloc[0]["hypothesis_id"]
    update_status(conn, hid, "reviewed", reviewed_by="analyst_1")
    row = load_knowledge_base(conn).iloc[0]
    assert row["status"] == "reviewed"
    assert row["reviewed_by"] == "analyst_1"


def test_export_import_round_trip(tmp_path):
    conn = _conn()
    write_hypotheses(conn, [_hypothesis()], source_run_id="run1")
    export_path = export_knowledge_base(conn, path=tmp_path / "knowledge_base.json")

    payload = json.loads(export_path.read_text())
    assert len(payload["knowledge_base"]) == 1


def test_export_publishes_generated_at_and_summary(tmp_path):
    """The dashboard reads status from these fields directly (no client-side
    value_counts()/max()) — see dashboard/knowledge_base_view.py."""
    conn = _conn()
    write_hypotheses(conn, [_hypothesis()], source_run_id="run1")
    export_path = export_knowledge_base(conn, path=tmp_path / "knowledge_base.json")

    payload = json.loads(export_path.read_text())
    assert "generated_at" in payload and payload["generated_at"]
    assert payload["summary"]["total_entries"] == 1
    assert payload["summary"]["by_status"] == {"candidate": 1}

    fresh_conn = _conn()
    n_imported = import_knowledge_base(fresh_conn, path=export_path)
    assert n_imported == 1
    assert len(load_knowledge_base(fresh_conn)) == 1


def test_import_is_idempotent(tmp_path):
    conn = _conn()
    write_hypotheses(conn, [_hypothesis()], source_run_id="run1")
    export_path = export_knowledge_base(conn, path=tmp_path / "knowledge_base.json")

    fresh_conn = _conn()
    import_knowledge_base(fresh_conn, path=export_path)
    n_second_import = import_knowledge_base(fresh_conn, path=export_path)
    assert n_second_import == 0   # already present, INSERT OR IGNORE
    assert len(load_knowledge_base(fresh_conn)) == 1


def test_write_hypotheses_is_idempotent_across_repeated_runs():
    # Regression test for a real bug found manually validating this module:
    # hypothesis_id used to hash in a fresh generated_at timestamp, so
    # re-running against unchanged input doubled the table every time.
    conn = _conn()
    write_hypotheses(conn, [_hypothesis()], source_run_id="run1")
    n_second_run = write_hypotheses(conn, [_hypothesis()], source_run_id="run2")
    assert n_second_run == 0
    assert len(load_knowledge_base(conn)) == 1


def test_update_status_returns_zero_for_nonexistent_hypothesis_id():
    # An UPDATE with no matching WHERE row is not a SQL error — callers must
    # check the returned count rather than assume success. Regression guard
    # for a real gap found while manually exercising the review CLI: it
    # printed a false "success" message for a hypothesis_id that matched no
    # row at all.
    conn = _conn()
    n_updated = update_status(conn, "does-not-exist", "reviewed")
    assert n_updated == 0


def test_update_status_returns_one_on_success():
    conn = _conn()
    write_hypotheses(conn, [_hypothesis()], source_run_id="run1")
    hid = load_knowledge_base(conn).iloc[0]["hypothesis_id"]
    assert update_status(conn, hid, "reviewed") == 1


def test_update_status_rejects_unknown_status():
    conn = _conn()
    write_hypotheses(conn, [_hypothesis()], source_run_id="run1")
    hid = load_knowledge_base(conn).iloc[0]["hypothesis_id"]
    with pytest.raises(ValueError):
        update_status(conn, hid, "not_a_real_status")


def _insert_minimal_model_registry_row(conn: sqlite3.Connection, model_version_id: str) -> None:
    conn.execute(
        "INSERT INTO model_registry (model_version_id, model_type, feature_list_json, status) "
        "VALUES (?, 'rule_score', '[]', 'candidate')",
        (model_version_id,),
    )
    conn.commit()


def test_update_status_accepts_real_linked_model_version_id():
    conn = _conn()
    _insert_minimal_model_registry_row(conn, "rule_score_test123")
    write_hypotheses(conn, [_hypothesis()], source_run_id="run1")
    hid = load_knowledge_base(conn).iloc[0]["hypothesis_id"]
    update_status(conn, hid, "tested_passed", linked_model_version_id="rule_score_test123")
    row = load_knowledge_base(conn).iloc[0]
    assert row["linked_model_version_id"] == "rule_score_test123"


def test_update_status_rejects_nonexistent_linked_model_version_id():
    # PRAGMA foreign_keys=ON (set in get_connection, and replicated here for
    # the in-memory test connection) must reject a linked_model_version_id
    # that doesn't exist in model_registry — a real DB-level guardrail
    # against silently mislinking a hypothesis to a model that was never
    # actually trained.
    conn = _conn()
    conn.execute("PRAGMA foreign_keys = ON")
    write_hypotheses(conn, [_hypothesis()], source_run_id="run1")
    hid = load_knowledge_base(conn).iloc[0]["hypothesis_id"]
    with pytest.raises(sqlite3.IntegrityError):
        update_status(conn, hid, "tested_passed", linked_model_version_id="does_not_exist")
