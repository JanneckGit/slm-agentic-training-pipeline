"""
tau2 `db_bahn` domain — registers itself with the tau2 registry on import (runtime registration,
so the domain code stays in this repo and we never edit the pip-installed tau2 source).
Import this module before driving tau2 programmatically:  `import sdg_pipeline.db_bahn.tau2_domain`
"""

from tau2.registry import registry

from sdg_pipeline.db_bahn.tau2_domain.environment import (
    get_environment, get_tasks, get_tasks_split,
)

if "db_bahn" not in registry.get_domains():
    registry.register_domain(get_environment, "db_bahn")
if "db_bahn" not in registry.get_task_sets():
    registry.register_tasks(get_tasks, "db_bahn", get_tasks_split)
