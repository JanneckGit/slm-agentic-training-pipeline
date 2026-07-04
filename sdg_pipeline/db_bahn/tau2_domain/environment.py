"""
sdg_pipeline/db_bahn/tau2_domain/environment.py
===============================================
tau2 `db_bahn` environment + task loaders. Data lives in OUR repo (not tau2's DATA_DIR):
  - db.json      : generated frozen world-state (gitignored)   -> $DB_BAHN_DATA/db.json
  - tasks.json   : generated tasks (Phase 2, gitignored)       -> $DB_BAHN_DATA/tasks.json
  - policy.md    : authored German agent policy (in repo)      -> next to this file
$DB_BAHN_DATA defaults to <repo>/data/raw/db_sandbox.
"""

import os
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.utils import load_file

from sdg_pipeline.db_bahn.tau2_domain.data_model import BahnDB
from sdg_pipeline.db_bahn.tau2_domain.tools import BahnTools

_REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("DB_BAHN_DATA", _REPO_ROOT / "data" / "raw" / "db_sandbox"))
DB_PATH = DATA_DIR / "db.json"
TASKS_PATH = DATA_DIR / "tasks.json"
SPLITS_PATH = DATA_DIR / "split_tasks.json"
POLICY_PATH = Path(__file__).parent / "policy.md"


def get_environment(db: Optional[BahnDB] = None, solo_mode: bool = False) -> Environment:
    if db is None:
        db = BahnDB.load(str(DB_PATH))
    tools = BahnTools(db)
    policy = POLICY_PATH.read_text(encoding="utf-8") if POLICY_PATH.exists() else ""
    env = Environment(domain_name="db_bahn", policy=policy, tools=tools, user_tools=None)
    if solo_mode:
        env.set_solo_mode(True)
    return env


def get_tasks(task_split_name: Optional[str] = None) -> list[Task]:
    if not TASKS_PATH.exists():
        return []
    tasks = [Task.model_validate(t) for t in load_file(str(TASKS_PATH))]
    if task_split_name is None:
        return tasks
    splits = get_tasks_split()
    if task_split_name not in splits:
        raise ValueError(f"Unbekannter Split '{task_split_name}'. Verfügbar: {list(splits)}")
    ids = set(splits[task_split_name])
    return [t for t in tasks if t.id in ids]


def get_tasks_split() -> dict[str, list[str]]:
    if not SPLITS_PATH.exists():
        return {}
    return load_file(str(SPLITS_PATH))
