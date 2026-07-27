"""Shared status vocabulary for the AI Lab automation pipeline.

Kept in its own module (rather than inside pipeline.py) so every extracted
stage module (generation.py, resolution.py, reflection_runner.py,
hypothesis_runner.py, knowledge_runner.py) can import it without a circular
import back to pipeline.py, which itself imports those stage modules.
"""
from enum import Enum


class PipelineStageStatus(Enum):
    """The only status vocabulary a pipeline stage may report.

    Anything more specific belongs in a "reason" (for SKIPPED) or "error"
    (for FAILED) field on the stage's result dict — never a new status value.
    """

    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
