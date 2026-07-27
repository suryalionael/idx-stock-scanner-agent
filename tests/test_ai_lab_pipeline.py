"""Tests for stock_scanner/ai_lab/pipeline.py — the AI Lab automation
orchestrator. Async calls are driven with asyncio.run() inside plain
`def test_...` functions, matching this repo's existing convention (see
tests/test_ai_lab_client.py) — no pytest-asyncio/anyio plugin needed.
"""
import asyncio
import json

import stock_scanner.ai_lab.pipeline as pipeline
from stock_scanner.ai_lab.pipeline_status import PipelineStageStatus


def _patch_status_path(monkeypatch, tmp_path):
    """Redirect the published status artifact into tmp_path so tests never
    write to the real data/published/ directory."""
    monkeypatch.setattr(pipeline, "_STATUS_PATH", tmp_path / "ai_pipeline_status.json")


def _ok_stage(*, calls: list[str], name: str, **extra):
    async def _stub(ctx):
        calls.append(name)
        return {"status": PipelineStageStatus.OK, **extra}
    return _stub


def _skipped_stage(*, calls: list[str], name: str, reason: str):
    async def _stub(ctx):
        calls.append(name)
        return {"status": PipelineStageStatus.SKIPPED, "reason": reason}
    return _stub


def _raising_stage(*, calls: list[str], name: str, exc: Exception):
    async def _stub(ctx):
        calls.append(name)
        raise exc
    return _stub


def test_all_stages_succeed(monkeypatch, tmp_path):
    _patch_status_path(monkeypatch, tmp_path)
    calls: list[str] = []
    stages = [(n, _ok_stage(calls=calls, name=n)) for n in
              ["generation", "resolution", "reflection", "hypothesis", "knowledge_base"]]
    monkeypatch.setattr(pipeline, "PIPELINE_STAGES", stages)

    results = asyncio.run(pipeline.run_ai_pipeline("2026-07-25", config={}))

    assert calls == ["generation", "resolution", "reflection", "hypothesis", "knowledge_base"]
    assert set(results) == {"generation", "resolution", "reflection", "hypothesis", "knowledge_base"}
    for stage_result in results.values():
        assert stage_result["status"] == "ok"
        assert "started_at" in stage_result and "finished_at" in stage_result
        assert isinstance(stage_result["duration_ms"], int)


def test_one_stage_failing_does_not_block_the_rest(monkeypatch, tmp_path):
    _patch_status_path(monkeypatch, tmp_path)
    calls: list[str] = []
    stages = [
        ("generation", _ok_stage(calls=calls, name="generation")),
        ("resolution", _raising_stage(calls=calls, name="resolution", exc=RuntimeError("boom"))),
        ("reflection", _ok_stage(calls=calls, name="reflection")),
        ("hypothesis", _ok_stage(calls=calls, name="hypothesis")),
        ("knowledge_base", _ok_stage(calls=calls, name="knowledge_base")),
    ]
    monkeypatch.setattr(pipeline, "PIPELINE_STAGES", stages)

    results = asyncio.run(pipeline.run_ai_pipeline("2026-07-25", config={}))

    # The isolation guarantee: every stage was called despite one raising.
    assert calls == ["generation", "resolution", "reflection", "hypothesis", "knowledge_base"]
    assert results["resolution"]["status"] == "failed"
    assert "boom" in results["resolution"]["error"]
    for name in ["generation", "reflection", "hypothesis", "knowledge_base"]:
        assert results[name]["status"] == "ok"


def test_skipped_stage_reports_reason_not_failure(monkeypatch, tmp_path):
    _patch_status_path(monkeypatch, tmp_path)
    calls: list[str] = []
    stages = [
        ("generation", _ok_stage(calls=calls, name="generation")),
        ("resolution", _ok_stage(calls=calls, name="resolution")),
        ("reflection", _skipped_stage(calls=calls, name="reflection",
                                       reason="insufficient resolved recommendations")),
        ("hypothesis", _ok_stage(calls=calls, name="hypothesis")),
        ("knowledge_base", _ok_stage(calls=calls, name="knowledge_base")),
    ]
    monkeypatch.setattr(pipeline, "PIPELINE_STAGES", stages)

    results = asyncio.run(pipeline.run_ai_pipeline("2026-07-25", config={}))

    assert results["reflection"]["status"] == "skipped"
    assert results["reflection"]["reason"] == "insufficient resolved recommendations"
    assert "error" not in results["reflection"]


def test_registry_is_extensible_without_orchestrator_changes(monkeypatch, tmp_path):
    """Appending a stub 6th stage to PIPELINE_STAGES exercises it with zero
    changes to run_ai_pipeline() itself — proves the registry design
    actually delivers the extensibility it's meant to."""
    _patch_status_path(monkeypatch, tmp_path)
    calls: list[str] = []
    stages = [(n, _ok_stage(calls=calls, name=n)) for n in
              ["generation", "resolution", "reflection", "hypothesis", "knowledge_base"]]
    stages.append(("calibration_engine", _ok_stage(calls=calls, name="calibration_engine")))
    monkeypatch.setattr(pipeline, "PIPELINE_STAGES", stages)

    results = asyncio.run(pipeline.run_ai_pipeline("2026-07-25", config={}))

    assert "calibration_engine" in results
    assert results["calibration_engine"]["status"] == "ok"
    assert calls[-1] == "calibration_engine"


def test_status_report_serializes_enum_to_plain_string(monkeypatch, tmp_path):
    _patch_status_path(monkeypatch, tmp_path)
    calls: list[str] = []
    stages = [(n, _ok_stage(calls=calls, name=n)) for n in
              ["generation", "resolution", "reflection", "hypothesis", "knowledge_base"]]
    monkeypatch.setattr(pipeline, "PIPELINE_STAGES", stages)

    asyncio.run(pipeline.run_ai_pipeline("2026-07-25", config={}))

    written = json.loads((tmp_path / "ai_pipeline_status.json").read_text())
    assert written["schema_version"] == 1
    assert written["scan_date"] == "2026-07-25"
    assert "last_run" in written
    for name in ["generation", "resolution", "reflection", "hypothesis", "knowledge_base"]:
        # Plain string, never a Python Enum repr like "PipelineStageStatus.OK".
        assert written[name]["status"] == "ok"
        assert "PipelineStageStatus" not in json.dumps(written[name])
