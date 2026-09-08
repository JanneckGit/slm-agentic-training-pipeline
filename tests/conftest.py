"""
tests/conftest.py
=================
Shared fixtures for the pytest suite (`bash ops/run_tests.sh`, or CI from requirements/tau2.lock.txt).

The db_bahn fixtures (db.json, tasks.json, answer_keys.json) are generated artifacts, not in git. They
resolve through `tau2_domain.environment.DATA_DIR` (overridable via DB_BAHN_DATA), so the same suite
runs against the local pool, a CI-regenerated pool or a scratch copy — no code changes.
"""
import json

import pytest

REGEN_HINT = ("generate them first: python3 sdg_pipeline/db_bahn/seed_worldstate.py --seed 42 && "
              "PYTHONPATH=. .venv-tau2/bin/python sdg_pipeline/db_bahn/gen_tasks.py --seed 42")


@pytest.fixture(scope="session")
def dom():
    """The registered db_bahn domain — needs the tau2 package, skipped in venvs without it."""
    pytest.importorskip("tau2")
    import sdg_pipeline.db_bahn.tau2_domain as dom
    return dom


@pytest.fixture(scope="session")
def data_dir(dom):
    from sdg_pipeline.db_bahn.tau2_domain.environment import DATA_DIR
    missing = [n for n in ("db.json", "tasks.json", "answer_keys.json") if not (DATA_DIR / n).exists()]
    if missing:
        pytest.skip(f"db_bahn fixtures missing in {DATA_DIR}: {missing} — {REGEN_HINT}")
    return DATA_DIR


@pytest.fixture(scope="session")
def tasks(data_dir):
    return {t["id"]: t for t in json.loads((data_dir / "tasks.json").read_text(encoding="utf-8"))}


@pytest.fixture(scope="session")
def keys(data_dir):
    return json.loads((data_dir / "answer_keys.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def pick(keys):
    """First task id of a template, optionally filtered on key fields (`injected=False`) or facts (`flavor="standort"`).
    A template without a matching task is a hard failure: the pool changed, and that is a finding."""
    def _pick(template, **pred):
        for tid, k in keys.items():
            if k["template"] == template and all(k.get(f, k["facts"].get(f)) == v for f, v in pred.items()):
                return tid
        raise AssertionError(f"no task for template {template} with {pred or 'no filter'} in the pool")
    return _pick
