"""Tests for the promoted-model lookup + shared challenger scoring formula —
closes the self-improving loop's read path (see
docs/SELF_IMPROVING_ARCHITECTURE.md). These guard two properties:
  1. get_promoted_model() never raises and only ever returns a row whose
     status is literally 'promoted' for the requested model_type.
  2. compute_rule_score() (relocated from scripts/train_challenger.py) keeps
     scoring identically to the original formula — a regression here would
     silently desync training/promotion from what production applies live.
"""
import json
from pathlib import Path

import pandas as pd

from stock_scanner.db.model_lookup import get_promoted_model
from stock_scanner.pipeline.challenger_score import compute_rule_score


def _write_registry(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "model_registry.json"
    path.write_text(json.dumps({"model_registry": rows}))
    return path


def test_missing_registry_file_returns_none(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    assert get_promoted_model("rule_score", registry_path=missing) is None


def test_malformed_json_returns_none(tmp_path):
    path = tmp_path / "model_registry.json"
    path.write_text("{not valid json")
    assert get_promoted_model("rule_score", registry_path=path) is None


def test_only_candidate_and_rejected_rows_returns_none(tmp_path):
    path = _write_registry(tmp_path, [
        {"model_version_id": "rule_score_1", "model_type": "rule_score", "status": "candidate"},
        {"model_version_id": "rule_score_2", "model_type": "rule_score", "status": "rejected"},
    ])
    assert get_promoted_model("rule_score", registry_path=path) is None


def test_promoted_row_returned_for_matching_type(tmp_path):
    path = _write_registry(tmp_path, [
        {"model_version_id": "rule_score_1", "model_type": "rule_score", "status": "candidate"},
        {"model_version_id": "rule_score_2", "model_type": "rule_score", "status": "promoted"},
        {"model_version_id": "xgb_1", "model_type": "xgboost_ranker", "status": "promoted"},
    ])
    row = get_promoted_model("rule_score", registry_path=path)
    assert row is not None
    assert row["model_version_id"] == "rule_score_2"


def test_promoted_row_not_returned_for_other_model_type(tmp_path):
    path = _write_registry(tmp_path, [
        {"model_version_id": "xgb_1", "model_type": "xgboost_ranker", "status": "promoted"},
    ])
    assert get_promoted_model("rule_score", registry_path=path) is None


def test_empty_registry_file_returns_none(tmp_path):
    path = _write_registry(tmp_path, [])
    assert get_promoted_model("rule_score", registry_path=path) is None


def test_compute_rule_score_formula():
    df = pd.DataFrame({
        "atr_breakout":  [True,  False, False, True],
        "vol_spike":     [False, True,  False, False],
        "vol_ratio_20d": [1.0,   3.0,   1.0,   1.0],
        "squeeze_on":    [False, False, True,  False],
    })
    score = compute_rule_score(df, vol_thresh=2.0)
    # row0: atr_breakout(1) + vol_spike(0) + (1.0>2.0 -> 0) - 2*squeeze(0) = 1
    # row1: 0 + 1 + (3.0>2.0 -> 1) - 0 = 2
    # row2: 0 + 0 + 0 - 2*1 = -2
    # row3: 1 + 0 + 0 - 0 = 1
    assert list(score) == [1, 2, -2, 1]


def test_compute_rule_score_handles_string_booleans_and_missing_values():
    # feature_snapshots round-trips through JSON/CSV, so bool columns can
    # arrive as literal "True"/"False" strings or be absent/NaN for a given
    # row (e.g. a ticker with too little history) — must not raise.
    df = pd.DataFrame({
        "atr_breakout":  ["True", "False", None],
        "vol_spike":     [False, True, False],
        "vol_ratio_20d": [1.0, None, 5.0],
        "squeeze_on":    [False, False, False],
    })
    score = compute_rule_score(df, vol_thresh=2.0)
    assert list(score) == [1, 1, 1]
