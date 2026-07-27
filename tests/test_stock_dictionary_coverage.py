"""Regression test for the Stock Dictionary's coverage of every user-visible
metric/indicator/score/term the dashboard actually exposes.

This runs the exact same structured-metadata-first discovery + diff as
scripts/check_dictionary_coverage.py (see that script's module docstring
for the discovery-tier priority order and the ADR for why the allowlist
exists). The point of this test is to fail the moment a new metric is
introduced to the dashboard/backend without a corresponding Dictionary
entry or a reviewed allowlist reason — the coverage report alone is only a
point-in-time snapshot; this is what keeps it true going forward.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from check_dictionary_coverage import run_coverage_check  # noqa: E402


def test_every_discovered_metric_is_covered_or_allowlisted():
    result = run_coverage_check()
    assert result["missing_count"] == 0, (
        f"{result['missing_count']} metric(s) discovered in the dashboard/backend have no "
        f"Stock Dictionary entry and are not on the reviewed allowlist: "
        f"{sorted(result['missing'])}. Add a stock_scanner/configs/dictionary/"
        f"stock_dictionary.yaml entry (or an aliases.yaml alias), or, if it's genuine UI "
        f"chrome/bookkeeping rather than a user-facing metric, add it to "
        f"scripts/check_dictionary_coverage.py's _ALLOWLIST with a reason."
    )


def test_coverage_check_actually_discovers_something():
    """Guards against the discovery logic silently breaking and reporting a
    false 100% because it found nothing at all."""
    result = run_coverage_check()
    assert result["discovered_count"] > 100
