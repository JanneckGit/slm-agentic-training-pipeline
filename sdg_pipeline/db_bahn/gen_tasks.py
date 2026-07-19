"""
sdg_pipeline/db_bahn/gen_tasks.py
=================================
Template-based German task generation for the tau2 `db_bahn` domain (wave 2, clean rebuild).
Every task ships its own machine-checkable ANSWER-KEY (KAG principle):

  - ACTION tasks -> `evaluation_criteria.actions` (reference write calls, replayed by tau2 to derive the
    target DB hash) + `env_assertions` (assert_* methods). reward_basis = [DB, ENV_ASSERTION].
  - INFO tasks   -> expected facts COMPUTED by running the real domain tools on a fresh DB copy
    (post fault-injection), stored in answer_keys.json for the tool-grounding checker, plus
    a few distinctive `communicate_info` substrings. reward_basis = [COMMUNICATE].

Wave-2 design (2026-07-08, supersedes the archived wave 1):
  - 26 templates: 10 polished easy/mid singles+chains, 16 hard ones — search tasks WITHOUT pre-given
    ids (zuege_suchen / mitarbeiter_suchen / wartung_liste), 3-4-tool chains, conditional writes,
    and RUNTIME faults (a WRITE tool rejects: wrong role/qualification, duplicate, terminal status).
  - Uniform answer-key schema for every task: kind, template, injected,
    fault in {none, state, runtime, state+runtime}, expected_tools, expected_calls,
    oracle_calls (the exact valid-path calls — drives rollout.py's oracle), facts.
  - Search->write determinism: tickets pin the choice via "erster Treffer (kleinste Mitarbeiter-ID)"
    (matches the tools' deterministic sort) or the generator verifies 1-3 hits; the DB-hash reward
    pins the exact entity either way. Rejected attempts are NEVER in reference actions (they do not
    mutate state, so the gold hash stays clean).

Replan by design: state faults via `initial_state.initialization_actions` (inject_verspaetung /
inject_lokfuehrer_ausfall) — tau2 applies them to live AND gold env; runtime faults via the WRITE-tool
validation gates in tools.py (the ticket steers the agent into a rejected call first).

Determinism: entity sampling + injections via sha256-seeded RNG (same helper as seed_worldstate);
task ids are content-derived (template__entity__nr), NOT uuids -> byte-identical re-runs.
Dedup/splits: unique (template, primary entity); splits bakeoff_dev / heldout_eval / rl_train /
sft_train are disjoint by construction and HARD-FAIL-checked before writing. rl_train is the
GRPO task reserve — it is never rolled out for SFT.

Run inside the tau2 venv (needs tau2 + the domain):
    PYTHONPATH=. <tau2-venv>/bin/python sdg_pipeline/db_bahn/gen_tasks.py --seed 42
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable, NamedTuple, Optional

from sdg_pipeline.db_bahn.seed_worldstate import rng
from sdg_pipeline.db_bahn.gen_tasks_lib import Gen
from sdg_pipeline.db_bahn.tau2_domain.environment import DATA_DIR, DB_PATH
from sdg_pipeline.db_bahn.gen_templates_easy import (
    t_action_crew, t_action_ersatz, t_action_wartung, t_action_wartung_status,
    t_info_ankunft, t_info_crew, t_info_mitarbeiter, t_info_standort, t_info_verspaetung,
    t_info_wartung)
from sdg_pipeline.db_bahn.gen_templates_hard import (
    t_action_crew_doppelt, t_action_ersatz_quali, t_action_gefahrgut,
    t_action_inspektion_bedingt, t_action_ueberfaellig, t_action_verstaerkung,
    t_action_wartung_batch, t_action_wartung_suche, t_action_wstatus_konflikt,
    t_info_ankunft_suche, t_info_mitarbeiter_suche, t_info_schichtcheck,
    t_info_verspaetung_suche, t_info_wartung_depot, t_info_zug_komplett,
    t_info_zugsuche_status)
from sdg_pipeline.db_bahn.gen_templates_wave3 import (
    t_action_batch_konflikt, t_action_batch_liste, t_action_doppelfault,
    t_action_iteration_ersatz, t_action_kaskade, t_action_lagebericht,
    t_action_umlauf_wartung, t_info_aggregation, t_info_anschluss,
    t_info_batch_phantom, t_info_batch_verspaetung, t_info_datenluecke,
    t_info_iteration_bedingt, t_info_ma_verfeinern, t_info_name_suche,
    t_info_teilerledigt, t_info_transient, t_info_umlauf_fahrzeug,
    t_info_wo_verfeinern, t_refusal_nicht_existent, t_refusal_nicht_machbar,
    t_refusal_policy)


# ---------------------------------------------------------------------------------------
# registry — one Spec per template; n is the per-template target (dedup + eligibility discount)
# ---------------------------------------------------------------------------------------
class Spec(NamedTuple):
    fn: Callable
    pool: str
    n: int
    injectable: Optional[bool]        # True = prob fault_rate, False = never, None = always
    fault_rate: Optional[float] = None  # per-template override; None -> --fault-rate


TEMPLATES = [
    # wave-2.5 n-table (validated by exact seeder+generator replication, 2026-07-11):
    # target T ~= 10.5k with multi-tool (>=3 calls) >= 50% and fault 38-42%. n is a SOFT cap —
    # pool-capped templates carry a generous n to harvest their full eligible ceiling
    # (actual = min(n, eligible)); n-capped templates carry the calibrated exact value.
    # easy/mid tier (1-2 calls; deliberately bounded so the easy tier can't drown the mix)
    Spec(t_info_verspaetung, "trips", 250, True),
    Spec(t_info_standort, "trips_pos", 300, False),               # pool-capped ~245
    Spec(t_info_ankunft, "trips_pos", 300, True),                 # pool-capped ~245 (3 calls!)
    Spec(t_info_wartung, "trips_orders", 450, False),
    Spec(t_info_crew, "trips_lokf", 300, False),
    Spec(t_info_mitarbeiter, "employees", 300, False),            # NEW (A1): lookup-by-ID gold path
    Spec(t_action_wartung, "trips", 400, False),
    Spec(t_action_crew, "trips_lokf", 300, False),
    Spec(t_action_wartung_status, "trips_orders", 450, False),
    Spec(t_action_ersatz, "trips_lokf", 250, None),
    # hard tier — the 3-4-call templates carry the >=50% multi-tool target,
    # so their n sits at/above their eligible ceilings (wave-2.5 recalibration)
    Spec(t_info_zugsuche_status, "trips_pos_route", 250, True),   # pool-capped ~181
    Spec(t_info_verspaetung_suche, "trips", 999, None),           # gate-capped ~890
    Spec(t_info_mitarbeiter_suche, "ma_filter_combos", 300, False),
    Spec(t_info_schichtcheck, "trips_lokf", 450, False),
    Spec(t_info_wartung_depot, "wartung_filter_combos", 150, False),
    Spec(t_info_zug_komplett, "trips_komplett", 300, True),       # pool-capped ~221
    Spec(t_info_ankunft_suche, "trips_route_page", 999, True),    # pool-capped ~852
    Spec(t_action_ersatz_quali, "trips_lokf", 1100, None),        # eligibility-capped ~1030
    Spec(t_action_crew_doppelt, "trips_lokf", 180, False),
    Spec(t_action_verstaerkung, "trips", 1100, False),            # pool-capped ~1070
    Spec(t_action_wartung_suche, "trips_route_page", 999, None),  # pool-capped ~852
    Spec(t_action_inspektion_bedingt, "trips", 800, True, 0.5),   # eligibility-capped ~705
    Spec(t_action_ueberfaellig, "trips_veh_ueberfaellig", 250, False),   # pool-capped ~136
    Spec(t_action_wstatus_konflikt, "trips_veh_konflikt", 120, False),   # pool-capped ~57
    Spec(t_action_wartung_batch, "trips_veh_2open", 250, False),         # pool-capped ~121
    Spec(t_action_gefahrgut, "trips_lokf_no_quali_produkt", 100, False), # pool-capped ~88 (GTFS-fix)
    # wave-3 hardening tier (2026-07-18): the mechanics the SFT data never demanded —
    # parallel/batch, iteration over hits, aggregation, refusal, cascades, Umlauf ambiguity,
    # >10-hit refinement, data gaps, transfer logic, transient errors, ambiguous names
    # S5-corridor rebalance (2026-07-18): the three IN-CORRIDOR classes (base 6-42%) carry the
    # real learning signal -> harvest their full pools; the easy-for-base classes stay at their
    # calibrated n as style/format carriers (think, parallel, refusal wording, retry policy).
    Spec(t_info_batch_verspaetung, "trips_batch3", 300, None),
    Spec(t_action_batch_liste, "wo_combos_batch", 999, False),           # pool-capped ~405 (CORRIDOR 42%)
    Spec(t_info_iteration_bedingt, "station_iter_groups", 999, None),    # pool-capped, window 3-10 (CORRIDOR 6%)
    Spec(t_info_aggregation, "trips_batch3", 250, None),
    Spec(t_refusal_nicht_existent, "trips", 150, False),
    Spec(t_refusal_nicht_machbar, "trips", 150, False),
    Spec(t_refusal_policy, "trips_veh_abgeschlossen", 200, False),       # pool-capped ~185
    Spec(t_action_kaskade, "trips_lokf", 200, None),
    Spec(t_info_umlauf_fahrzeug, "trips_same_vehicle", 250, False),      # pool-capped ~276
    Spec(t_info_ma_verfeinern, "ma_refine_combos", 300, False),
    Spec(t_info_wo_verfeinern, "wo_refine_combos", 400, False),          # (CORRIDOR 39%)
    Spec(t_info_datenluecke, "trips_pos", 200, None),
    Spec(t_info_anschluss, "anschluss_pairs", 300, True, 0.5),
    Spec(t_info_teilerledigt, "trips_lokf", 200, True),
    Spec(t_info_name_suche, "employees_common_name", 250, False),        # pool-capped ~304
    Spec(t_info_transient, "trips_pos", 150, None),
    # wave-3.5 tier (2026-07-18): conjunction tickets (multiplicative hardness) + depth
    # compositions — where the corridor measurement located the real base-4B weakness
    Spec(t_action_lagebericht, "trips_pos_lokf", 250, True),
    Spec(t_action_batch_konflikt, "wo_batch_konflikt", 200, False),
    Spec(t_info_batch_phantom, "trips_batch3", 200, None),
    Spec(t_action_iteration_ersatz, "station_iter_groups", 999, None),   # pool-capped ~60-75
    Spec(t_action_doppelfault, "trips_lokf", 250, None),
    Spec(t_action_umlauf_wartung, "veh_pairs_mixed", 250, False),
]


def main():
    ap = argparse.ArgumentParser(description="Generate db_bahn tasks with built-in answer keys (wave 2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fault-rate", type=float, default=0.35)
    # wave-2.5 defaults for the ~10.5k pool: bakeoff = one task per template (default tracks the
    # registry, so a new template can't silently drop out); heldout/rl scaled for eval power + GRPO
    ap.add_argument("--n-bakeoff", type=int, default=len(TEMPLATES))
    # wave-3.5 split decision (2026-07-19): heldout AND rl are UNIFORM — a fixed count PER
    # TEMPLATE instead of proportional-to-size. Rationale: per-class solve rates are the eval
    # instrument (a 2-slot class can only read 0/50/100%), and the rl reserve must not be
    # dominated by the big easy classes. Neutral rule, NOT hardness weighting. Consequence:
    # headline eval numbers break with the proportional-397 era (86.6% baseline is stale).
    ap.add_argument("--n-heldout-per-template", type=int, default=10)
    ap.add_argument("--n-rl-per-template", type=int, default=20,
                    help="GRPO task reserve — disjoint from SFT, never rolled out for SFT")
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    g = Gen(DB_PATH, args.seed)
    out_dir = Path(args.out_dir)

    tasks, keys, seen = [], {}, set()
    stats = Counter()
    for spec in TEMPLATES:
        tname = spec.fn.__name__
        pool = g.pool(spec.pool)
        order = rng(args.seed, "pool", tname).sample(range(len(pool)), len(pool))
        made = 0
        for j in order:
            if made >= spec.n:
                break
            item = pool[j]
            # trips -> zugnummer; employees -> emp_id (wave-2.5); combos -> .key
            entity = getattr(item, "zugnummer", None) or getattr(item, "emp_id", None) or item.key
            dedup_key = (tname, entity)
            if dedup_key in seen:  # unique (template, primary entity)
                stats["near_dup_skipped"] += 1
                continue
            if spec.injectable is None:
                inject = True
            elif spec.injectable:
                rate = spec.fault_rate if spec.fault_rate is not None else args.fault_rate
                inject = rng(args.seed, "faultroll", tname, entity).random() < rate
            else:
                inject = False
            res = spec.fn(g, item, made, inject)
            if res is None:
                stats[f"{tname}__ineligible"] += 1
                continue
            task, key = res
            seen.add(dedup_key)
            key["template"] = tname
            key["injected"] = bool(task.get("initial_state"))
            key.setdefault("fault", "state" if key["injected"] else "none")
            key["expected_calls"] = len(key.get("oracle_calls") or [])
            tasks.append(task)
            keys[task["id"]] = key
            made += 1
            stats[tname] += 1
            if key["injected"]:
                stats["injected_total"] += 1

    # splits: PER-TEMPLATE PROPORTIONAL — every template appears in EVERY disjoint split (small pools no
    # longer starved by round-robin), with floors. bakeoff_dev is a NON-disjoint stratified sample (its job,
    # teacher selection, is done → no need to burn unique tasks on it). HARD-FAIL disjointness + coverage.
    by_tpl = {}
    for t in tasks:
        by_tpl.setdefault(keys[t["id"]]["template"], []).append(t["id"])
    for tpl in by_tpl:
        by_tpl[tpl] = sorted(by_tpl[tpl])
        rng(args.seed, "split", tpl).shuffle(by_tpl[tpl])

    total = len(tasks)  # consumed by the disjointness hard-fail assert below
    heldout, rl, sft = [], [], []
    for tpl in sorted(by_tpl):
        ids = by_tpl[tpl]
        # UNIFORM per template, FIXED counts (Janneck 2026-07-19: no soft caps) — a template too
        # small for 10+20+sft trips the "templates missing from sft_train" HARD-FAIL below,
        # which is the intended loud signal (current smallest template: 57 -> 10/20/27)
        n_hel = args.n_heldout_per_template
        n_rl = args.n_rl_per_template
        heldout += ids[:n_hel]
        rl += ids[n_hel:n_hel + n_rl]
        sft += ids[n_hel + n_rl:]                     # the rest -> sft (guaranteed non-empty per template)
    # bakeoff_dev: 1 task per template from sft (non-disjoint smoke/dev subset)
    sft_by_tpl = {}
    for tid in sft:
        sft_by_tpl.setdefault(keys[tid]["template"], []).append(tid)
    bakeoff = [sft_by_tpl[tpl][0] for tpl in sorted(sft_by_tpl)][:args.n_bakeoff]
    splits = {"bakeoff_dev": bakeoff, "heldout_eval": heldout, "rl_train": rl, "sft_train": sft}

    disjoint = [i for k in ("heldout_eval", "rl_train", "sft_train") for i in splits[k]]
    assert len(disjoint) == len(set(disjoint)) == total, (
        f"HARD-FAIL: sft/rl/heldout overlap or loss ({len(disjoint)} vs {len(set(disjoint))} vs {total})")
    assert set(bakeoff) <= set(sft), "HARD-FAIL: bakeoff_dev must be a subset of sft_train"
    missing = set(by_tpl) - {keys[t]["template"] for t in sft}
    assert not missing, f"HARD-FAIL: templates missing from sft_train: {sorted(missing)}"

    tasks.sort(key=lambda t: t["id"])
    (out_dir / "tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    (out_dir / "split_tasks.json").write_text(
        json.dumps({k: sorted(v) for k, v in splits.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "answer_keys.json").write_text(
        json.dumps(keys, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")

    # acceptance stats (gate d): diversity/difficulty profile of the generated pool
    n = len(tasks)
    multi = sum(1 for k in keys.values() if k["expected_calls"] >= 3)
    single = sum(1 for k in keys.values() if k["expected_calls"] == 1)
    fault_counts = Counter(k["fault"] for k in keys.values())
    n_fault = n - fault_counts.get("none", 0)
    print(f"tasks: {n}  (near-dup skipped: {stats['near_dup_skipped']})")
    for spec in TEMPLATES:
        tname = spec.fn.__name__
        ks = [k for k in keys.values() if k["template"] == tname]
        f = Counter(k["fault"] for k in ks)
        ftxt = ", ".join(f"{v}× {t}" for t, v in sorted(f.items()) if t != "none") or "—"
        print(f"  {tname:28s}: {len(ks):4d}  (calls≥3: {sum(k['expected_calls'] >= 3 for k in ks):4d}, "
              f"fault: {ftxt})")
    print(f"multi-tool (expected_calls>=3): {multi}/{n} = {multi / max(n, 1):.0%}   "
          f"single-tool: {single}/{n} = {single / max(n, 1):.0%}")
    print(f"fault total: {n_fault}/{n} = {n_fault / max(n, 1):.0%}  "
          f"({', '.join(f'{v}× {t}' for t, v in sorted(fault_counts.items()) if t != 'none')})")
    print("splits:", {k: len(v) for k, v in splits.items()})
    print(f"wrote tasks.json / split_tasks.json / answer_keys.json -> {out_dir}")


if __name__ == "__main__":
    main()
