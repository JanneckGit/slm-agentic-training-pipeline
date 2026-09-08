"""ToolACE think-backfill: verdict taxonomy, trainable-turn selection, prefix salvage — the module's own self-test.
Needs the tau2 venv: the module imports sdg_pipeline.db_bahn.rollout (-> tau2 domain) at import time."""
import pytest

pytest.importorskip("tau2")
from data_pipeline.backfill_toolace_think import _selftest  # noqa: E402


def test_backfill_toolace_selftest():
    _selftest()
