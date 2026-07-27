"""Light smoke tests for the AI Lab stage modules extracted from
scripts/run_*.py (generation.py, resolution.py, reflection_runner.py,
hypothesis_runner.py, knowledge_runner.py) — confirming the extraction
preserved each script's existing "insufficient data" guard behavior and
that each now returns the new PipelineStageStatus-shaped result dict.

The heavy business logic (activate_pending/resolve_active, generate_
observations, generate_candidate_hypotheses/validate_hypotheses,
generate_knowledge_entries) is already covered by test_ai_lab_resolver.py,
test_ai_lab_reflection_engine.py, test_ai_lab_hypothesis_engine.py,
test_ai_lab_statistical_validation.py, test_ai_lab_knowledge_base_engine.py
— not re-tested here.

Each stage's run() opens a real sqlite connection (get_connection()) and
writes to hardcoded data/published/*.json mirror paths, matching the
original scripts exactly (no path-injection parameter existed before this
extraction, so none is added here). Tests redirect both to tmp_path via
monkeypatch so nothing touches the real project files.
"""
import sqlite3

import stock_scanner.ai_lab.hypothesis_runner as hypothesis_runner
import stock_scanner.ai_lab.knowledge_runner as knowledge_runner
import stock_scanner.ai_lab.reflection_runner as reflection_runner
import stock_scanner.ai_lab.resolution as resolution
from stock_scanner.ai_lab import generation


def test_generation_skips_when_no_ranked_csv(tmp_path):
    import asyncio

    result = asyncio.run(generation.run(ranked_dir=tmp_path))
    assert result["status"].value == "skipped"
    assert "no ranked_*.csv found" in result["reason"]


def _use_memory_db(monkeypatch, module) -> None:
    monkeypatch.setattr(module, "get_connection", lambda db_path=None: sqlite3.connect(":memory:"))


def test_resolution_run_returns_ok_on_empty_db(monkeypatch, tmp_path):
    import stock_scanner.db.ai_lab as db_ai_lab

    _use_memory_db(monkeypatch, resolution)
    monkeypatch.setattr(db_ai_lab, "_RECS_MIRROR_PATH", tmp_path / "ai_recommendations.json")
    monkeypatch.setattr(db_ai_lab, "_EVENTS_MIRROR_PATH", tmp_path / "ai_learning_events.json")

    result = resolution.run(raw_dir=tmp_path)

    assert result["status"].value == "ok"
    assert result["activated"] == 0
    assert result["closed"] == 0


def test_reflection_runner_skips_on_empty_db(monkeypatch, tmp_path):
    import stock_scanner.db.ai_lab as db_ai_lab
    import stock_scanner.db.reflection as db_reflection

    _use_memory_db(monkeypatch, reflection_runner)
    monkeypatch.setattr(db_ai_lab, "_RECS_MIRROR_PATH", tmp_path / "ai_recommendations.json")
    monkeypatch.setattr(db_reflection, "_MIRROR_PATH", tmp_path / "reflection_report.json")

    import asyncio
    result = asyncio.run(reflection_runner.run(use_mock=True))

    assert result["status"].value == "skipped"
    assert result["reason"] == "insufficient resolved recommendations"


def test_hypothesis_runner_skips_on_empty_db(monkeypatch, tmp_path):
    import stock_scanner.db.ai_lab as db_ai_lab
    import stock_scanner.db.hypotheses as db_hypotheses
    import stock_scanner.db.reflection as db_reflection

    _use_memory_db(monkeypatch, hypothesis_runner)
    monkeypatch.setattr(db_ai_lab, "_RECS_MIRROR_PATH", tmp_path / "ai_recommendations.json")
    monkeypatch.setattr(db_reflection, "_MIRROR_PATH", tmp_path / "reflection_report.json")
    monkeypatch.setattr(db_hypotheses, "_MIRROR_PATH", tmp_path / "hypotheses_report.json")

    import asyncio
    result = asyncio.run(hypothesis_runner.run(use_mock=True))

    assert result["status"].value == "skipped"
    assert "no candidate cleared validation" in result["reason"] or "no gated reflection" in result["reason"]


def test_knowledge_runner_skips_on_empty_db(monkeypatch, tmp_path):
    import stock_scanner.db.ai_lab as db_ai_lab
    import stock_scanner.db.hypotheses as db_hypotheses
    import stock_scanner.db.knowledge_entries as db_knowledge_entries

    _use_memory_db(monkeypatch, knowledge_runner)
    monkeypatch.setattr(db_ai_lab, "_RECS_MIRROR_PATH", tmp_path / "ai_recommendations.json")
    monkeypatch.setattr(db_hypotheses, "_MIRROR_PATH", tmp_path / "hypotheses_report.json")
    monkeypatch.setattr(db_knowledge_entries, "_MIRROR_PATH", tmp_path / "knowledge_report.json")

    import asyncio
    result = asyncio.run(knowledge_runner.run(use_mock=True))

    assert result["status"].value == "skipped"
    assert result["reason"] == "no validated hypotheses available to curate"
