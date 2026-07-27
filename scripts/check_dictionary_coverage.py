#!/usr/bin/env python3
"""Stock Dictionary coverage checker.

Discovers candidate metric/column/label names actually exposed by the
dashboard and backend configs, then checks each against
stock_scanner/configs/dictionary/ (entries + aliases). Prints a
human-readable coverage report and exits non-zero if anything discovered
is neither covered nor explicitly allowlisted — see
tests/test_stock_dictionary_coverage.py, which runs this same check as a
regression test.

Discovery is structured-metadata-first, string/AST inspection only as a
fallback, to minimize false positives (see docs/ADR_AI_AUTOMATION_AND_STOCK_DICTIONARY.md):

  1. Dashboard table schemas       — dashboard/data_loader.py TABLE_COLS/HISTORY_COLS
  2. Streamlit column config       — label/help args to st.column_config.*Column(...)
  3. Published JSON schemas       — real row/column keys in data/published/*.json
  4. Backend config files         — metric keys in broker/smart-money config,
                                     validated even if not currently rendered anywhere
  5. Dashboard metadata           — tab titles (st.tabs([...]))
  6. AST/string inspection        — fallback only: st.metric(...) label literals

Usage:
    python scripts/check_dictionary_coverage.py            # print report, exit 1 if incomplete
    python scripts/check_dictionary_coverage.py --json      # machine-readable report
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).parent.parent
_DASHBOARD_DIR = _REPO_ROOT / "dashboard"
_PUBLISHED_DIR = _REPO_ROOT / "data" / "published"
_DICTIONARY_DIR = _REPO_ROOT / "stock_scanner" / "configs" / "dictionary"
_BROKER_CONFIG = _REPO_ROOT / "stock_scanner" / "configs" / "broker_config.yaml"
_SMART_MONEY_CONFIG = _REPO_ROOT / "stock_scanner" / "configs" / "smart_money_config.yaml"


def _norm(s: str) -> str:
    return str(s).strip().lower().replace("_", " ").replace("-", " ")


# Structural/administrative columns and generic UI chrome that are not
# glossary-worthy terms — reviewed by hand, grouped by why each is excluded.
# A term here is a deliberate judgment call, not an oversight: either it's
# raw record-keeping (an id/timestamp/blob, not something a user looks up
# in a glossary), pure navigation (a tab title), or a display-only
# abbreviation/sub-label of a concept already documented under its
# canonical name elsewhere in stock_dictionary.yaml.
_ALLOWLIST = {
    _norm(s) for s in [
        # --- Structural / OHLCV / generic table columns ---
        "ticker", "close", "date", "open", "high", "low", "volume", "rank", "status",
        "generated_time", "current_price", "action", "value", "no", "id", "model",
        "ai_model", "recommendation", "type", "label", "score", "grade", "share",
        "combined", "fundamentals", "entry", "tp", "cl", "prev", "pe",
        "kode", "broker", "lot", "freq", "avg", "tanggal", "sesi", "kandidat",
        "avg beli", "avg jual", "freq beli", "freq jual", "nilai beli", "nilai jual",
        "kategori", "ringkasan", "skor", "nama emiten", "bulan", "top brk",
        "saham (lbr)", "jumlah holder", "growth mom", "ownership",
        "beli (lot)", "jual (lot)", "net (lot)", "close (rp)", "eval date",
        "entry hi", "area entry (rp)", "cutloss (rp)", "target profit (rp)",
        "atr brk", "broker absorb", "intrinsic (rp)", "valuasi", "siklikalitas",
        "roe%", "div yield%", "rev growth%", "profit growth%", "roc5 (%)",
        "f.score", "enh.score", "vol×", "w/l", "%", "% close", "% high", "δ%",
        "alasan", "indikasi", "ditampilkan", "sinyal direview",
        "pending (menunggu sesi)", "distinct tickers", "rows shown",
        # --- Emoji tab titles / navigation (not metrics) ---
        "📈 scalping", "🔄 swing trading", "📊 long term", "🎯 smart money",
        "🔁 naik/turun beruntun", "📋 signal performance", "🔍 search emiten",
        "🕐 history", "🧠 learning agent", "🚀 daily movers >10%", "🧪 ai lab",
        "📖 stock dictionary", "📈 chart", "🏢 shareholders", "📋 broker",
        "📈 sinyal & chart", "📋 broker activity",
        # --- Emoji signal/label badges (already covered by the `signal` /
        # `smart_money_label` / `long_term_label` / `scalping_label` entries
        # under a different literal string) ---
        "🏷️ undervalued", "👀 scalping watch", "👀 watch", "💎 long term core",
        "📋 long term watchlist", "🔥 scalping high", "🔥 strong candidate",
        # --- AI Lab / Learning Agent count & sub-component labels — already
        # explained under their composite entry (decision_trace,
        # confidence_breakdown, hypothesis, reflection, knowledge_base_ai_lab)
        # or are simple "how many X" counts, not new concepts ---
        "active", "active positions", "closed positions", "expired", "pending",
        "resolved", "rejected", "validated", "number of recommendations",
        "categories", "evidence strength", "failures", "successes",
        "last run", "last update", "total hypotheses", "total knowledge entries",
        "total observations", "resolved trades analyzed", "verdict",
        "final confidence", "final score", "risk", "risk adjustment",
        "sample size", "sample size (n / n success)", "statistical", "technical",
        "confidence (llm assigned)", "win rate (raw / shrunk)",
        # --- Raw JSON record fields — ids/timestamps/blobs, not user-facing
        # metrics (model_registry.json, ai_learning_events.json,
        # knowledge_base.json, ai_recommendations.json, top_signals.json,
        # latest_scan.json, daily_movers.json bookkeeping columns) ---
        "affected dimension", "affected sector", "ai model", "alert triggered",
        "artifact path", "avg predicted prob", "challenger metrics json",
        "challenger model id", "computed at", "created at", "decided at",
        "decision", "decision id", "description", "entry price", "eval close",
        "eval high", "event id", "event type", "exit price", "expected effect",
        "feature drift flag", "feature list json", "filter threshold pct",
        "final status", "forward return pct", "generated at", "generated date",
        "highest price", "historical comparison", "holding days", "hypothesis id",
        "inserted at", "label horizon", "label threshold pct",
        "linked model version id", "lowest price", "metadata json", "ml prob",
        "model type", "model version id", "monitor date", "n evaluated",
        "n signals", "pattern json", "pct change close", "pct change high",
        "pct close", "pct high", "prev close", "production metrics json",
        "production model id", "promoted at", "public float pct",
        "quality source", "rank in day", "realized precision at threshold",
        "realized win rate", "reason", "reasoning", "retired at",
        "return percentage", "reviewed by", "risk flags", "scalping reason",
        "sensitivity json", "signal date", "signal id", "signal label",
        "source", "source run id", "strategy", "supporting trades",
        "test end date", "test metrics json", "test start date", "trade date",
        "trade outcome", "trained at", "train end date", "train metrics json",
        "train start date", "updated at", "val end date", "val metrics json",
        "val start date", "atr pct",
    ]
}


def _load_dictionary() -> tuple[set[str], dict[str, str]]:
    """Return (set of every searchable normalized key, {key: entry_id})."""
    entries_doc = yaml.safe_load((_DICTIONARY_DIR / "stock_dictionary.yaml").read_text())
    aliases_doc = yaml.safe_load((_DICTIONARY_DIR / "aliases.yaml").read_text())

    key_to_id: dict[str, str] = {}
    for entry in entries_doc.get("entries", []):
        for field in ("id", "short_name", "title"):
            val = entry.get(field)
            if val:
                key_to_id[_norm(val)] = entry["id"]
    for entry_id, alias_list in (aliases_doc.get("aliases") or {}).items():
        for alias in alias_list or []:
            key_to_id[_norm(alias)] = entry_id

    return set(key_to_id), key_to_id


def _discover_table_cols() -> dict[str, list[str]]:
    """Tier 1: dashboard/data_loader.py's TABLE_COLS/HISTORY_COLS constants."""
    src = (_DASHBOARD_DIR / "data_loader.py").read_text()
    tree = ast.parse(src)
    found: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ("TABLE_COLS", "HISTORY_COLS"):
                if isinstance(node.value, ast.List):
                    cols = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
                    found[f"data_loader.py::{target.id}"] = cols
    return found


def _iter_st_calls(tree: ast.AST, dotted_suffix: str):
    """Yield ast.Call nodes whose func is an attribute chain ending in dotted_suffix
    (e.g. "column_config.Column" matches st.column_config.TextColumn)."""
    parts = dotted_suffix.split(".")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        chain = []
        cur = func
        while isinstance(cur, ast.Attribute):
            chain.insert(0, cur.attr)
            cur = cur.value
        if len(chain) >= len(parts) and chain[-len(parts):] == parts:
            yield node
        elif len(chain) >= 1 and parts[-1] in chain:
            # loose match for e.g. st.column_config.TextColumn / NumberColumn / CheckboxColumn
            if dotted_suffix == "column_config" and len(chain) >= 2 and chain[0] == "column_config":
                yield node


def _first_str_arg(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    for kw in call.keywords:
        if kw.arg == "label" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _discover_column_config_labels() -> dict[str, list[str]]:
    """Tier 2: label= / positional-string args to st.column_config.*Column(...)."""
    found: dict[str, list[str]] = {}
    for path in sorted(_DASHBOARD_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        labels = []
        for node in _iter_st_calls(tree, "column_config"):
            label = _first_str_arg(node)
            if label:
                labels.append(label)
        if labels:
            found[f"{path.name}::column_config"] = labels
    return found


def _discover_published_json_keys() -> dict[str, list[str]]:
    """Tier 3: real row/column keys present in data/published/*.json."""
    found: dict[str, list[str]] = {}
    if not _PUBLISHED_DIR.exists():
        return found
    for path in sorted(_PUBLISHED_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        keys: set[str] = set()
        if isinstance(payload, dict):
            for v in payload.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    keys.update(v[0].keys())
        if keys:
            found[f"published/{path.name}"] = sorted(keys)
    return found


def _discover_backend_config_metrics() -> dict[str, list[str]]:
    """Tier 4: metric keys defined in broker/smart-money configs — validated
    even when not currently rendered anywhere in the dashboard."""
    found: dict[str, list[str]] = {}
    if _BROKER_CONFIG.exists():
        doc = yaml.safe_load(_BROKER_CONFIG.read_text()) or {}
        metrics = list((doc.get("metrics") or {}).keys())
        if metrics:
            found["broker_config.yaml::metrics"] = metrics
    if _SMART_MONEY_CONFIG.exists():
        doc = yaml.safe_load(_SMART_MONEY_CONFIG.read_text()) or {}
        # Top-level keys that look like detector names (dict-valued, not scalar settings).
        detector_keys = [k for k, v in doc.items() if isinstance(v, dict)]
        if detector_keys:
            found["smart_money_config.yaml::detectors"] = detector_keys
    return found


def _discover_tab_titles() -> dict[str, list[str]]:
    """Tier 5: dashboard tab titles (st.tabs([...]))."""
    found: dict[str, list[str]] = {}
    tree = ast.parse((_DASHBOARD_DIR / "app.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "tabs" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.List):
                    titles = [elt.value for elt in arg.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)]
                    if titles:
                        found.setdefault("app.py::tabs", []).extend(titles)
    return found


def _discover_metric_labels_fallback() -> dict[str, list[str]]:
    """Tier 6 (fallback only): st.metric(label, ...) string literals."""
    found: dict[str, list[str]] = {}
    for path in sorted(_DASHBOARD_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        labels = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "metric":
                    label = _first_str_arg(node)
                    if label:
                        labels.append(label)
        if labels:
            found[f"{path.name}::st.metric"] = labels
    return found


def run_coverage_check() -> dict:
    dict_keys, _ = _load_dictionary()

    tiers = [
        ("Dashboard table schemas", _discover_table_cols()),
        ("Streamlit column configuration metadata", _discover_column_config_labels()),
        ("Published JSON schemas", _discover_published_json_keys()),
        ("Scanner/Broker/Smart Money configuration files", _discover_backend_config_metrics()),
        ("Dashboard metadata (tab titles)", _discover_tab_titles()),
        ("AST/string inspection (fallback)", _discover_metric_labels_fallback()),
    ]

    discovered: dict[str, str] = {}  # normalized key -> source
    for tier_name, sources in tiers:
        for source, candidates in sources.items():
            for c in candidates:
                key = _norm(c)
                if key and key not in discovered:
                    discovered[key] = f"{tier_name} ({source})"

    covered = {k: v for k, v in discovered.items() if k in dict_keys}
    allowlisted = {k: v for k, v in discovered.items() if k not in dict_keys and k in _ALLOWLIST}
    missing = {k: v for k, v in discovered.items() if k not in dict_keys and k not in _ALLOWLIST}

    return {
        "discovered_count": len(discovered),
        "covered_count": len(covered),
        "allowlisted_count": len(allowlisted),
        "missing_count": len(missing),
        "missing": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON report.")
    args = parser.parse_args()

    result = run_coverage_check()
    coverage_pct = (
        100.0 * (result["covered_count"] + result["allowlisted_count"]) / result["discovered_count"]
        if result["discovered_count"] else 100.0
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Coverage Report")
        print("================")
        print(f"Discovered Metrics : {result['discovered_count']}")
        print(f"Covered            : {result['covered_count']}")
        print(f"Allowlisted        : {result['allowlisted_count']}")
        print(f"Missing            : {result['missing_count']}")
        print(f"Coverage           : {coverage_pct:.0f}%")
        if result["missing"]:
            print("\nMissing terms (need a Dictionary entry or an allowlist reason):")
            for key, source in sorted(result["missing"].items()):
                print(f"  - {key!r}  (from: {source})")

    return 0 if result["missing_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
