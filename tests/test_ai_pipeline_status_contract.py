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


def _capture_streamlit_calls(monkeypatch):
    """Monkeypatch st.caption/markdown/expander to record what
    _render_pipeline_status_row() would actually display, without needing
    a live Streamlit server — same technique used to manually verify this
    against the real production ai_pipeline_status.json during development."""
    import streamlit as st

    captured: list[tuple] = []

    class _FakeExpander:
        def __init__(self, label):
            captured.append(("expander_open", label))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            captured.append(("expander_close",))

    monkeypatch.setattr(st, "caption", lambda msg, **k: captured.append(("caption", msg)))
    monkeypatch.setattr(st, "markdown", lambda msg, **k: captured.append(("markdown", msg)))
    monkeypatch.setattr(st, "expander", lambda label, **k: _FakeExpander(label))
    return captured


def test_status_row_surfaces_recorded_error_for_failed_stage(monkeypatch, tmp_path):
    """The UI must show the pipeline's own recorded `error` string for a
    failed stage — not just a red dot — so it's actionable."""
    import dashboard.ai_lab_view as ai_lab_view

    status_path = tmp_path / "ai_pipeline_status.json"
    monkeypatch.setattr(data_loader, "_LOCAL_AI_PIPELINE_STATUS_PATH", status_path)
    monkeypatch.setattr(data_loader, "_REMOTE_AI_PIPELINE_STATUS_URL", "PLACEHOLDER_USER")
    monkeypatch.setattr(pipeline, "_STATUS_PATH", status_path)

    async def _failing(ctx):
        raise RuntimeError("Missing required 9router configuration: NINEROUTER_API_KEY")

    async def _ok(ctx):
        return {"status": PipelineStageStatus.OK}

    monkeypatch.setattr(pipeline, "PIPELINE_STAGES", [
        ("generation", _failing), ("resolution", _ok),
        ("reflection", _ok), ("hypothesis", _ok), ("knowledge_base", _ok),
    ])
    asyncio.run(pipeline.run_ai_pipeline("2026-07-28", config={}))

    captured = _capture_streamlit_calls(monkeypatch)
    ai_lab_view._load_pipeline_status_payload.clear()  # bypass st.cache_data across test runs
    ai_lab_view._render_pipeline_status_row()

    detail_lines = [c[1] for c in captured if c[0] == "markdown"]
    assert any("generation" in line and "NINEROUTER_API_KEY" in line for line in detail_lines)


def test_status_row_surfaces_recorded_reason_for_skipped_stage(monkeypatch, tmp_path):
    """The UI must show the pipeline's own recorded `reason` string for a
    skipped stage — not just a yellow dot."""
    import dashboard.ai_lab_view as ai_lab_view

    status_path = tmp_path / "ai_pipeline_status.json"
    monkeypatch.setattr(data_loader, "_LOCAL_AI_PIPELINE_STATUS_PATH", status_path)
    monkeypatch.setattr(data_loader, "_REMOTE_AI_PIPELINE_STATUS_URL", "PLACEHOLDER_USER")
    monkeypatch.setattr(pipeline, "_STATUS_PATH", status_path)

    async def _ok(ctx):
        return {"status": PipelineStageStatus.OK}

    async def _skipped(ctx):
        return {"status": PipelineStageStatus.SKIPPED, "reason": "insufficient resolved recommendations"}

    monkeypatch.setattr(pipeline, "PIPELINE_STAGES", [
        ("generation", _ok), ("resolution", _ok),
        ("reflection", _skipped), ("hypothesis", _ok), ("knowledge_base", _ok),
    ])
    asyncio.run(pipeline.run_ai_pipeline("2026-07-28", config={}))

    captured = _capture_streamlit_calls(monkeypatch)
    ai_lab_view._load_pipeline_status_payload.clear()
    ai_lab_view._render_pipeline_status_row()

    detail_lines = [c[1] for c in captured if c[0] == "markdown"]
    assert any("reflection" in line and "insufficient resolved recommendations" in line for line in detail_lines)


def test_status_row_shows_no_detail_expander_when_everything_ok(monkeypatch, tmp_path):
    """No noise when there's nothing actionable — an all-ok run shouldn't
    render an empty/pointless expander."""
    import dashboard.ai_lab_view as ai_lab_view

    status_path = tmp_path / "ai_pipeline_status.json"
    monkeypatch.setattr(data_loader, "_LOCAL_AI_PIPELINE_STATUS_PATH", status_path)
    monkeypatch.setattr(data_loader, "_REMOTE_AI_PIPELINE_STATUS_URL", "PLACEHOLDER_USER")
    monkeypatch.setattr(pipeline, "_STATUS_PATH", status_path)

    async def _ok(ctx):
        return {"status": PipelineStageStatus.OK}

    monkeypatch.setattr(pipeline, "PIPELINE_STAGES", [
        ("generation", _ok), ("resolution", _ok),
        ("reflection", _ok), ("hypothesis", _ok), ("knowledge_base", _ok),
    ])
    asyncio.run(pipeline.run_ai_pipeline("2026-07-28", config={}))

    captured = _capture_streamlit_calls(monkeypatch)
    ai_lab_view._load_pipeline_status_payload.clear()
    ai_lab_view._render_pipeline_status_row()

    assert not any(c[0] == "expander_open" for c in captured)
