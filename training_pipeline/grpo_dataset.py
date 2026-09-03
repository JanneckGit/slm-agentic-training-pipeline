"""verl dataset for the Stage-2 GRPO pool: weighted, ledger-driven task draws ("corridor sampler").

Why a dataset and not a sampler: verl 0.8.0 (frozen image) has no custom-sampler hook —
`create_rl_sampler` is Random/Sequential only — but it does have `data.custom_cls` and a custom
reward function. So the dataset does the drawing: its length is the epoch size (128 prompts) and
`__getitem__` ignores verl's index and hands out the next entry of a per-epoch plan built by
training_pipeline/grpo_sampling.py from (a) the corridor measurement baked into the pool parquet and
(b) the training ledger that evaluation/grpo_reward.py appends to (one JSONL per reward worker).
Validation parquets (pool_role != "train") make the class behave exactly like RLHFDataset.

Wiring (all set by ops/grpo_run.sh):
    data.custom_cls.path=/app/training_pipeline/grpo_dataset.py  data.custom_cls.name=CorridorDataset
    +data.grpo_run_dir=/app/data/final/grpo/runs/<run>     # ledger/ + sampler/ live here
    +data.grpo_sampling=dynamic|static  +data.grpo_epoch_size=128  +data.grpo_eps=0.005
    +data.grpo_mix=0.5  +data.grpo_max_wait=2  +data.grpo_seed=42
    (env GRPO_RUN_DIR / GRPO_SAMPLING are the fallback when the keys are absent)
Requires data.dataloader_num_workers=0 (the plan lives in the trainer process; the preflight checks it).

Resume: the sampler state (epoch counter, last-drawn epoch per task) is persisted after every plan;
a restart rebuilds a fresh plan for the remaining calls of the interrupted epoch — the ledger is the
source of truth, nothing is lost.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

from verl.utils.dataset.rl_dataset import RLHFDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_pipeline.grpo_sampling import (  # noqa: E402
    EPS_DEFAULT, MAX_WAIT_EPOCHS, MIX_DEFAULT, N_DRAWS_DEFAULT, PoolTask, build_plan, compute_weights,
)


def _cfg(config, key: str, env: str | None, default):
    v = config.get(key, None) if config is not None else None
    if v is None and env:
        v = os.environ.get(env)
    return default if v is None else v


class CorridorDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        super().__init__(data_files, tokenizer, config, processor, max_samples)
        first = self.dataframe[0].get("extra_info") or {} if len(self.dataframe) else {}
        self.active = first.get("pool_role") == "train"
        if not self.active:
            return
        run_dir = _cfg(config, "grpo_run_dir", "GRPO_RUN_DIR", None)
        if not run_dir:
            raise ValueError("CorridorDataset: set +data.grpo_run_dir=… (or env GRPO_RUN_DIR)")
        self.run_dir = Path(run_dir)
        self.ledger_dir = self.run_dir / "ledger"
        self.state_dir = self.run_dir / "sampler"
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.mode = str(_cfg(config, "grpo_sampling", "GRPO_SAMPLING", "dynamic"))
        self.epoch_size = int(_cfg(config, "grpo_epoch_size", "GRPO_EPOCH_SIZE", N_DRAWS_DEFAULT))
        self.eps = float(_cfg(config, "grpo_eps", None, EPS_DEFAULT))
        self.mix = float(_cfg(config, "grpo_mix", None, MIX_DEFAULT))
        self.max_wait = int(_cfg(config, "grpo_max_wait", None, MAX_WAIT_EPOCHS))
        self.seed = int(_cfg(config, "grpo_seed", None, 42))

        self.tasks: list[PoolTask] = []
        for i in range(len(self.dataframe)):
            e = self.dataframe[i]["extra_info"]
            self.tasks.append(PoolTask(self.dataframe[i]["reward_model"]["ground_truth"], e["template"],
                                       float(e["rate0"]), float(e["template_rate0"]), bool(e["ever_failed"])))
        self.by_id = {t.task_id: i for i, t in enumerate(self.tasks)}
        self.static_weights = compute_weights(self.tasks, self.eps, self.mix) if self.mode == "static" else None
        self.epoch = 0
        self.plan: list[int] | None = None
        self.calls = 0
        self._load_state()
        print(f"[corridor_dataset] active: {len(self.tasks)} tasks, mode={self.mode}, epoch_size={self.epoch_size}, "
              f"run_dir={self.run_dir}, resume_epoch={self.epoch}", flush=True)

    # --- verl contract ---------------------------------------------------------------------------
    def __len__(self):
        return self.epoch_size if self.active else super().__len__()

    def __getitem__(self, item):
        if not self.active:
            return super().__getitem__(item)
        if self.plan is None or self.calls >= len(self.plan):
            self._new_epoch()
        idx = self.plan[self.calls]
        self.calls += 1
        row = super().__getitem__(idx)
        row["extra_info"] = {**(row.get("extra_info") or {}), "ledger_dir": str(self.ledger_dir),
                             "pool_epoch": self.epoch - 1}
        return row

    # --- plan / ledger / state -----------------------------------------------------------------
    def _read_ledger(self) -> None:
        entries = []
        for f in glob.glob(str(self.ledger_dir / "*.jsonl")):
            try:
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            entries.append(json.loads(line))
            except Exception as e:  # a half-written line must not kill the trainer
                print(f"[corridor_dataset] ledger read warning {f}: {e}", file=sys.stderr)
        entries.sort(key=lambda r: r.get("ts", 0))
        for t in self.tasks:
            t.ledger = []
        for r in entries:
            i = self.by_id.get(r.get("task_id"))
            if i is not None:
                self.tasks[i].ledger.append(float(r.get("score", 0.0)))
                if r.get("score", 1.0) < 1:
                    self.tasks[i].ever_failed = True

    def _new_epoch(self) -> None:
        if self.mode != "static":
            self._read_ledger()
        self.plan, diag = build_plan(self.tasks, self.epoch, self.epoch_size, self.seed, self.eps, self.mix,
                                     self.max_wait, static_weights=self.static_weights)
        for i in set(self.plan):
            self.tasks[i].last_drawn_epoch = self.epoch
        self.calls = 0
        weights = self.static_weights or compute_weights(self.tasks, self.eps, self.mix)
        drawn = sorted(set(self.plan))
        record = {
            "epoch": self.epoch, "mode": self.mode, "diag": diag, "ts": time.time(),
            "plan_task_ids": [self.tasks[i].task_id for i in self.plan],
            "draws": {self.tasks[i].task_id: self.plan.count(i) for i in drawn},
            "weights": {t.task_id: round(w, 5) for t, w in zip(self.tasks, weights)},
            "n_ledger_entries": sum(len(t.ledger) for t in self.tasks),
            "share_sibling_draws": round(sum(1 for i in self.plan if not self.tasks[i].ever_failed) / max(1, len(self.plan)), 4),
        }
        (self.state_dir / f"epoch_{self.epoch:03d}.json").write_text(json.dumps(record, ensure_ascii=False))
        print(f"[corridor_dataset] epoch {self.epoch}: {diag} sibling_draws={record['share_sibling_draws']} "
              f"ledger_entries={record['n_ledger_entries']}", flush=True)
        self.epoch += 1
        self._save_state()

    def _state_path(self) -> Path:
        return self.state_dir / "state.json"

    def _save_state(self) -> None:
        self._state_path().write_text(json.dumps({
            "epoch": self.epoch, "mode": self.mode,
            "last_drawn_epoch": {t.task_id: t.last_drawn_epoch for t in self.tasks},
            "ever_failed": {t.task_id: t.ever_failed for t in self.tasks},
        }))

    def _load_state(self) -> None:
        p = self._state_path()
        if not p.exists():
            return
        try:
            s = json.loads(p.read_text())
            self.epoch = int(s.get("epoch", 0))
            for t in self.tasks:
                t.last_drawn_epoch = int(s.get("last_drawn_epoch", {}).get(t.task_id, -1))
                t.ever_failed = bool(s.get("ever_failed", {}).get(t.task_id, t.ever_failed))
        except Exception as e:
            print(f"[corridor_dataset] state load warning: {e}", file=sys.stderr)
