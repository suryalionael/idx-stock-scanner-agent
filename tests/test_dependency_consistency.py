"""Guards against requirements.txt / pyproject.toml drift.

On Streamlit Community Cloud, requirements.txt wins the dependency-file
priority contest (verified against docs.streamlit.io/deploy/streamlit-
community-cloud/deploy-your-app/app-dependencies: uv.lock > Pipfile >
environment.yml > requirements.txt > pyproject.toml) — it's what actually
installs streamlit/plotly/yfinance/openpyxl for the deployed dashboard.
Separately, Streamlit Cloud also runs `pip install .` (because of
pyproject.toml's [build-system] table) to register stock_scanner as an
importable package, which pulls pyproject.toml's own [project.dependencies].
These are two independent install steps reading two different files — if a
package listed in both ever gets a different version specifier in one file
and not the other, the two steps could silently disagree. This test makes
that impossible to do by accident.
"""
import re
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_REQUIREMENTS_PATH = _REPO_ROOT / "requirements.txt"
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"

_SPEC_RE = re.compile(r"^([A-Za-z0-9_.\-]+)\s*(.*)$")


def _split_name_and_spec(raw: str) -> tuple[str, str] | None:
    m = _SPEC_RE.match(raw.strip())
    if not m or not m.group(1):
        return None
    return m.group(1).lower(), m.group(2).strip()


def _parse_requirements(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()   # strip full-line and inline comments
        if not line:
            continue
        parsed = _split_name_and_spec(line)
        if parsed:
            result[parsed[0]] = parsed[1]
    return result


def _parse_pyproject_dependencies(path: Path) -> dict[str, str]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    result: dict[str, str] = {}
    for dep in data["project"]["dependencies"]:
        parsed = _split_name_and_spec(dep)
        if parsed:
            result[parsed[0]] = parsed[1]
    return result


def test_shared_dependencies_have_identical_version_specifiers():
    req = _parse_requirements(_REQUIREMENTS_PATH)
    pyproj = _parse_pyproject_dependencies(_PYPROJECT_PATH)
    shared = set(req) & set(pyproj)
    assert shared, "expected at least one package listed in both files (sanity check on the parser)"

    mismatches = {name: (req[name], pyproj[name]) for name in shared if req[name] != pyproj[name]}
    assert not mismatches, (
        f"requirements.txt and pyproject.toml disagree on version specifiers: {mismatches}. "
        "These are read by two independent install steps on Streamlit Community Cloud "
        "(requirements.txt for the deployed app; pyproject.toml's [project.dependencies] via "
        "`pip install .` to register stock_scanner) — any mismatch here is a real "
        "non-deterministic-deploy risk, not just a style nit."
    )


def test_pyproject_base_dependencies_are_satisfied_by_requirements_txt():
    # pyproject.toml's base [project.dependencies] exists so `pip install .`
    # can register stock_scanner as importable — every package it lists must
    # also be present in requirements.txt, since that's the file that
    # actually runs for the deployed dashboard.
    req = _parse_requirements(_REQUIREMENTS_PATH)
    pyproj = _parse_pyproject_dependencies(_PYPROJECT_PATH)
    missing = set(pyproj) - set(req)
    assert not missing, (
        f"pyproject.toml base dependencies missing from requirements.txt: {sorted(missing)} — "
        "the deployed app would not have these installed."
    )
