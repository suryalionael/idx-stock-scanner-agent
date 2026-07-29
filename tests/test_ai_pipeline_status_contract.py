"""End-to-end contract test between stock_scanner.ai_lab.pipeline (the
writer of data/published/ai_pipeline_status.json) and
dashboard.data_loader.load_ai_pipeline_status_payload() (the sole reader).

test_ai_lab_pipeline.py already verifies the writer in isolation, and the
dashboard views are smoke-tested against whatever payload the loader
returns — but nothing previously proved the two sides agree on the same
file, in the same shape, without either side being mocked out. This test
runs the real orchestrator (stages stubbed, since the real stages need
live DB/LLM state — see test_ai_lab_pipeline.py for why) and points the
real dashboard loader's local-file path at that same output, so a change
to either side's file shape breaks this test instead of only surfacing at
runtime in the deployed dashboard.
"""
import asyncio

import dashboard.data_loader as data_loader
import stock_scanner.ai_lab.pipeline as pipeline
from stock_scanner.ai_lab.pipeline_status import PipelineStageStatus


def test_pipeline_output_is_readable_by_the_dashboard_loader(monkeypatch, tmp_path):
    status_path = tmp_path / "ai_pipeline_status.json"
    monkeypatch.setattr(pipeline, "_STATUS_PATH", status_path)
    # Point the loader's *local* fallback at the same file, and force it
    # past the "remote first" branch (no network in a test) the same way
    # every other loader test in this repo does — see the URL-placeholder
    # skip in data_loader.py's own load_*_payload() functions.
    monkeypatch.setattr(data_loader, "_LOCAL_AI_PIPELINE_STATUS_PATH", status_path)

    async def _ok(ctx):
        return {"status": PipelineStageStatus.OK}

    async def _skipped(ctx):
        return {"status": PipelineStageStatus.SKIPPED, "reason": "insufficient resolved recommendations"}

    stages = [
        ("generation", _ok), ("resolution", _ok),
        ("reflection", _skipped), ("hypothesis", _skipped), ("knowledge_base", _ok),
    ]
    monkeypatch.setattr(pipeline, "PIPELINE_STAGES", stages)

    # The writer side: run the real orchestrator (stages stubbed).
    asyncio.run(pipeline.run_ai_pipeline("2026-07-25", config={}))
    assert status_path.exists(), "run_ai_pipeline() did not write ai_pipeline_status.json"

    # The reader side: the actual dashboard function, unmodified, reading
    # the file the orchestrator just wrote.
    payload = data_loader.load_ai_pipeline_status_payload(url="PLACEHOLDER_USER")

    assert payload["schema_version"] == 1
    assert payload["scan_date"] == "2026-07-25"
    assert "last_run" in payload
    assert payload["generation"]["status"] == "ok"
    assert payload["reflection"]["status"] == "skipped"
    assert payload["reflection"]["reason"] == "insufficient resolved recommendations"


def test_dashboard_loader_degrades_gracefully_when_pipeline_has_never_run(monkeypatch, tmp_path):
    """Before the first automated Daily Scan with this feature (or in a
    fresh local clone), the status file simply doesn't exist yet — the
    dashboard must show an empty-but-valid state, not crash."""
    monkeypatch.setattr(data_loader, "_LOCAL_AI_PIPELINE_STATUS_PATH", tmp_path / "does_not_exist.json")

    payload = data_loader.load_ai_pipeline_status_payload(url="PLACEHOLDER_USER")

    assert payload == {}
