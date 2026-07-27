"""AI Lab automation — the single orchestration layer that chains
generation -> resolution -> reflection -> hypothesis (+ statistical
validation) -> knowledge base together in-process, after every Daily Scan.

Called as the final step of stock_scanner.pipeline.run_daily_scan.py::main(),
strictly after all production scoring/ranking/publishing has already
completed — this ordering is what guarantees AI Lab never influences
production output, not a runtime check. See
docs/ADR_AI_AUTOMATION_AND_STOCK_DICTIONARY.md.

Each stage is a plain (name, async function) pair in PIPELINE_STAGES —
adding a future stage (Calibration Engine, Decision Agent, Model Promotion)
is one adapter function + one appended tuple, no class hierarchy required.
Every stage is independently crash-isolated: an exception in any one stage
never prevents the remaining stages from running, mirroring the
try/except/log/continue pattern run_daily_scan.py already uses for its own
optional steps.

This module writes exactly one artifact of its own,
data/published/ai_pipeline_status.json — strictly execution metadata
(status/timing/reason per stage). Business data (recommendation counts,
hypothesis counts, etc.) continues to live exclusively in each stage's own
published payload (ai_recommendations.json, reflection_report.json, ...).
"""
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from stock_scanner.ai_lab import generation, hypothesis_runner, knowledge_runner, reflection_runner, resolution
from stock_scanner.ai_lab.pipeline_status import PipelineStageStatus

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STATUS_SCHEMA_VERSION = 1
_STATUS_PATH = _REPO_ROOT / "data" / "published" / "ai_pipeline_status.json"


@dataclass
class PipelinePaths:
    """Directories a pipeline stage may need — long-lived, shared context,
    not stage-specific parameters (those stay as each run()'s own defaults)."""

    raw_dir: Path
    ranked_dir: Path
    published_dir: Path


@dataclass
class PipelineContext:
    """Shared context threaded through every stage. Future stages read
    whatever they need off this object instead of the orchestrator gaining
    a new parameter per stage."""

    scan_date: str
    config: dict
    paths: PipelinePaths
    metadata: dict[str, Any] = field(default_factory=dict)


StageFunc = Callable[[PipelineContext], Awaitable[dict]]
# Each stage function returns at minimum {"status": PipelineStageStatus, ...},
# and when status is SKIPPED, a "reason" string — never a new ad hoc status.


async def _generation(ctx: PipelineContext) -> dict:
    return await generation.run(scan_date=ctx.scan_date, ranked_dir=ctx.paths.ranked_dir)


async def _resolution(ctx: PipelineContext) -> dict:
    return resolution.run(raw_dir=ctx.paths.raw_dir)   # sync stage; wrapped only for a uniform interface


async def _reflection(ctx: PipelineContext) -> dict:
    return await reflection_runner.run()


async def _hypothesis(ctx: PipelineContext) -> dict:
    return await hypothesis_runner.run()


async def _knowledge_base(ctx: PipelineContext) -> dict:
    return await knowledge_runner.run()


# The whole extensibility story: a future stage is one adapter function +
# one appended tuple here — no class, no inheritance, no orchestrator changes.
PIPELINE_STAGES: list[tuple[str, StageFunc]] = [
    ("generation", _generation),
    ("resolution", _resolution),
    ("reflection", _reflection),
    ("hypothesis", _hypothesis),
    ("knowledge_base", _knowledge_base),
]


def _build_paths(config: dict) -> PipelinePaths:
    return PipelinePaths(
        raw_dir=_REPO_ROOT / "data" / "raw",
        ranked_dir=_REPO_ROOT / "data" / "ranked",
        published_dir=_REPO_ROOT / "data" / "published",
    )


async def run_ai_pipeline(scan_date: str, config: dict | None = None) -> dict:
    """Run every AI Lab automation stage in order, isolating failures per
    stage. Returns the same per-stage results dict that gets published to
    ai_pipeline_status.json (plus schema_version/last_run/scan_date)."""
    config = config or {}
    ctx = PipelineContext(scan_date=scan_date, config=config, paths=_build_paths(config))

    results: dict[str, dict] = {}
    for stage_name, stage_fn in PIPELINE_STAGES:
        started = datetime.now(timezone.utc)
        try:
            stage_result = await stage_fn(ctx)
            status = stage_result.get("status", PipelineStageStatus.OK)
            extra: dict[str, Any] = {}
            if status is PipelineStageStatus.SKIPPED:
                extra["reason"] = stage_result.get("reason", "skipped")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"AI pipeline stage '{stage_name}' failed: {exc}")
            status = PipelineStageStatus.FAILED
            extra = {"error": str(exc)}
        finished = datetime.now(timezone.utc)

        results[stage_name] = {
            "status": status.value,   # enum serialized to its string value only at this JSON boundary
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_ms": int((finished - started).total_seconds() * 1000),
            **extra,
        }

    _publish_pipeline_status(scan_date, results)
    return results


def _publish_pipeline_status(scan_date: str, results: dict[str, dict]) -> None:
    """Write ai_pipeline_status.json atomically (write to a sibling .tmp
    file, then os.replace) — this runs unattended in CI, so a crash or
    kill mid-write must never leave the dashboard reading a truncated
    JSON file."""
    payload = {
        "schema_version": _STATUS_SCHEMA_VERSION,
        "last_run": datetime.now(timezone.utc).isoformat(),
        "scan_date": scan_date,
        **results,
    }
    _STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _STATUS_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, _STATUS_PATH)
    logger.info(f"ai_pipeline: published {_STATUS_PATH}")
