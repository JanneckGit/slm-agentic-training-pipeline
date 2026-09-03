"""Pure sampling math for the Stage-2 GRPO task pool — no verl/torch imports, testable anywhere.

Why this exists (2026-09-03): GRPO learns only from prompt groups whose rollouts do NOT all agree
(Dr.GRPO advantage = reward − group mean → a task the policy solves 8/8 or 0/8 contributes nothing).
The pool builder measured every rl_train task 8× (the "corridor" run); here we turn those rates into
draw weights and an epoch plan, and re-weight from the training ledger as the policy moves.

Three pieces, each a plain function:
  * weights  — p̂ = mix·own + (1−mix)·template (own = last LEDGER_LAST_N ledger scores, else the
               corridor rate; template = mean own-p of the template's tasks), w = max(p̂(1−p̂), ε).
               Smoothing with the template rate (160 rollouts) tames the ±1/8 noise of a single task.
  * aging    — a task that ever failed and was not drawn for MAX_WAIT_EPOCHS full epochs gets a
               guaranteed draw next epoch (never-failed "siblings" are probed only via their weight,
               otherwise 500+ of them would be forced into a 128-draw epoch).
  * allocate — low-variance (systematic) resampling: claim_i = n·w_i/Σw, every task receives
               floor(claim) or ceil(claim), the total is exactly n. Then shuffle, so a step never sees
               16 prompts of the same template.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

EPS_DEFAULT = 0.005          # weight floor: a saturated task is still probed every ~1/ε·Σw epochs
MIX_DEFAULT = 0.5            # weight of the task's OWN rate vs. its template's rate (1:1)
LEDGER_LAST_N = 16           # ledger scores per task used for its own rate (two 8-rollout groups)
MAX_WAIT_EPOCHS = 2          # aging: full epochs without a draw before a failed task is forced
N_DRAWS_DEFAULT = 128        # epoch length in prompts (= 8 steps × 16 prompts)


@dataclass
class PoolTask:
    task_id: str
    template: str
    rate0: float                 # corridor measurement: ok / k
    template_rate0: float        # template mean over all its tasks in the measurement
    ever_failed: bool            # rate0 < 1 or any ledger score < 1
    ledger: list[float] = field(default_factory=list)   # scores in arrival order
    last_drawn_epoch: int = -1   # -1 = never drawn in this run (the measurement counts as epoch -1)


def own_rate(t: PoolTask, last_n: int = LEDGER_LAST_N) -> tuple[float, int]:
    """(rate, n_evidence): ledger-based if the task has ledger scores, else the corridor rate (k=8)."""
    if t.ledger:
        recent = t.ledger[-last_n:]
        return sum(recent) / len(recent), len(recent)
    return t.rate0, 8


def template_rates(tasks: list[PoolTask], last_n: int = LEDGER_LAST_N) -> dict[str, float]:
    """Template rate = mean of the tasks' own rates (ledger where present, corridor rate otherwise)."""
    acc: dict[str, list[float]] = {}
    for t in tasks:
        acc.setdefault(t.template, []).append(own_rate(t, last_n)[0])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def smooth_p(own_p: float, tmpl_p: float, mix: float = MIX_DEFAULT) -> float:
    return mix * own_p + (1.0 - mix) * tmpl_p


def task_weight(p_hat: float, eps: float = EPS_DEFAULT) -> float:
    return max(p_hat * (1.0 - p_hat), eps)


def compute_weights(tasks: list[PoolTask], eps: float = EPS_DEFAULT, mix: float = MIX_DEFAULT,
                    last_n: int = LEDGER_LAST_N) -> list[float]:
    tr = template_rates(tasks, last_n)
    return [task_weight(smooth_p(own_rate(t, last_n)[0], tr[t.template], mix), eps) for t in tasks]


def aged(tasks: list[PoolTask], epoch: int, max_wait: int = MAX_WAIT_EPOCHS) -> list[int]:
    """Indices that must be drawn in `epoch`: ever failed AND not drawn in the last `max_wait` epochs.
    A never-drawn task counts from epoch -1 (the corridor measurement), so with max_wait=2 it is
    forced in epoch 2 at the latest."""
    return [i for i, t in enumerate(tasks) if t.ever_failed and (epoch - t.last_drawn_epoch) > max_wait]


def allocate(weights: list[float], n_draws: int, rng: random.Random) -> list[int]:
    """Systematic (low-variance) resampling: counts_i ∈ {floor(claim_i), ceil(claim_i)}, Σ = n_draws."""
    if n_draws <= 0 or not weights:
        return [0] * len(weights)
    total = float(sum(weights))
    if total <= 0:
        weights, total = [1.0] * len(weights), float(len(weights))
    counts = [0] * len(weights)
    u = rng.random()
    upper, ptr = 0.0, 0
    for i, w in enumerate(weights):
        upper += w * n_draws / total
        while ptr < n_draws and (u + ptr) < upper - 1e-12:
            counts[i] += 1
            ptr += 1
    while ptr < n_draws:           # float residue: give the last marks to the heaviest task
        counts[max(range(len(weights)), key=lambda k: weights[k])] += 1
        ptr += 1
    return counts


def build_plan(tasks: list[PoolTask], epoch: int, n_draws: int = N_DRAWS_DEFAULT, seed: int = 0,
               eps: float = EPS_DEFAULT, mix: float = MIX_DEFAULT, max_wait: int = MAX_WAIT_EPOCHS,
               static_weights: list[float] | None = None) -> tuple[list[int], dict]:
    """One epoch of task indices (length n_draws, shuffled) + diagnostics.
    `static_weights` freezes the weights (mode `static`); otherwise they come from the ledger."""
    rng = random.Random(f"{seed}:{epoch}")
    weights = list(static_weights) if static_weights is not None else compute_weights(tasks, eps, mix)
    forced = aged(tasks, epoch, max_wait) if static_weights is None else []
    if len(forced) > n_draws:
        rng.shuffle(forced)
        forced = forced[:n_draws]
    counts = allocate(weights, n_draws - len(forced), rng)
    plan: list[int] = list(forced)
    for i, c in enumerate(counts):
        plan.extend([i] * c)
    rng.shuffle(plan)
    total = sum(weights)
    failed_mass = sum(w for w, t in zip(weights, tasks) if t.ever_failed)
    diag = {
        "epoch": epoch, "n_draws": len(plan), "n_forced": len(forced),
        "weight_total": round(total, 4),
        "share_ever_failed": round(failed_mass / total, 4) if total else None,
        "distinct_tasks": len(set(plan)),
        "max_copies": max((plan.count(i) for i in set(plan)), default=0),
    }
    return plan, diag


def informative_share(groups: list[list[float]]) -> float:
    """Fraction of prompt groups whose rollouts disagree (the only ones that carry gradient)."""
    if not groups:
        return 0.0
    return sum(1 for g in groups if g and min(g) != max(g)) / len(groups)


# --- self-test: python3 training_pipeline/grpo_sampling.py --selftest ------------------------------
def _selftest() -> None:
    rng = random.Random(0)
    # weights from the worked examples in docs/stage2-RL.md (kaskade template rate 0.919)
    kask = 0.919
    assert abs(task_weight(smooth_p(0.5, kask)) - 0.206) < 0.002, task_weight(smooth_p(0.5, kask))
    assert abs(task_weight(smooth_p(0.875, kask)) - 0.092) < 0.002
    assert abs(task_weight(smooth_p(1.0, kask)) - 0.038) < 0.002
    assert task_weight(smooth_p(1.0, 0.994)) == EPS_DEFAULT          # 159/160-template sibling -> floor
    assert abs(task_weight(smooth_p(1.0, 0.981)) - 0.0094) < 0.001    # wartung_depot sibling ~ 2x floor
    assert abs(task_weight(smooth_p(1.0, 0.787)) - 0.095) < 0.002    # refusal_nicht_machbar sibling
    # allocate: exact total, floor/ceil bound, claim 0.8 never > 1
    w = [0.109] * 72 + [0.1875] * 32 + [0.234] * 5 + [0.25] * 7 + [0.234] * 3
    for trial in range(200):
        c = allocate(w, 128, random.Random(trial))
        assert sum(c) == 128, sum(c)
        tot = sum(w)
        for wi, ci in zip(w, c):
            claim = 128 * wi / tot
            assert math.floor(claim) <= ci <= math.ceil(claim), (wi, ci, claim)
    # build_plan on a synthetic pool: siblings never forced, aged failed task forced in epoch 2
    tasks = [PoolTask(f"c{i}", "t_a", 0.875, 0.92, True) for i in range(10)] + \
            [PoolTask(f"s{i}", "t_a", 1.0, 0.92, False) for i in range(10)] + \
            [PoolTask(f"m{i}", "t_b", 1.0, 0.99, False) for i in range(20)]
    plan0, d0 = build_plan(tasks, epoch=0, n_draws=128, seed=1)
    assert len(plan0) == 128 and d0["n_forced"] == 0 and d0["share_ever_failed"] > 0.5, d0
    plan2, d2 = build_plan(tasks, epoch=2, n_draws=128, seed=1)      # nobody drawn yet -> all 10 failed forced
    assert d2["n_forced"] == 10, d2
    assert all(tasks[i].ever_failed for i in aged(tasks, 2)), "siblings must never be aged in"
    tasks[0].last_drawn_epoch = 1
    assert 0 not in aged(tasks, 2) and 1 in aged(tasks, 2)
    # ledger moves weights the right way
    base = compute_weights(tasks)
    tasks[10].ledger = [1, 1, 0, 1, 1, 0, 1, 1]        # sibling starts failing -> weight up
    tasks[0].ledger = [1] * 16                          # corridor task saturates -> weight down
    upd = compute_weights(tasks)
    assert upd[10] > base[10] and upd[0] < base[0], (base[10], upd[10], base[0], upd[0])
    # shuffle + determinism
    assert plan0 != sorted(plan0) and build_plan(tasks, 0, 128, seed=1)[0] == build_plan(tasks, 0, 128, seed=1)[0]
    p_static, ds = build_plan(tasks, epoch=5, n_draws=128, seed=1, static_weights=base)
    assert ds["n_forced"] == 0 and len(p_static) == 128
    assert informative_share([[1, 1, 1], [1, 0, 1], [0, 0]]) == 1 / 3
    print("grpo_sampling selftest OK")


if __name__ == "__main__":
    import sys as _sys
    if "--selftest" in _sys.argv:
        _selftest()
    else:
        print(__doc__)
