"""
sdg_pipeline/db_bahn/gen_tasks.py
=================================
Phase 2 of Plan (B): template-based German task generation for the tau2 `db_bahn` domain.
Every task ships its own machine-checkable ANSWER-KEY (KAG principle):

  - ACTION tasks -> `evaluation_criteria.actions` (reference write calls, replayed by tau2 to derive the
    target DB hash) + `env_assertions` (assert_* methods). reward_basis = [DB, ENV_ASSERTION].
  - INFO tasks   -> expected facts COMPUTED by running the real domain tools on a fresh DB copy
    (post fault-injection), stored in answer_keys.json for our Phase-4 tool-grounding checker, plus
    a few distinctive `communicate_info` substrings. reward_basis = [COMMUNICATE] (DB is vacuous
    for reads; the strict fact check lives in our own verifier).

Replan by design: a seeded fraction of tasks carries `initial_state.initialization_actions` calling the
NON-tool injection methods (inject_verspaetung / inject_lokfuehrer_ausfall). tau2 applies these to both
the live env (rollout) and the gold env (evaluation replay), so answer keys stay consistent.

Determinism: entity sampling + injections via sha256-seeded RNG (same helper as seed_worldstate);
task ids are content-derived (template__zug__nr), NOT uuids -> byte-identical re-runs.
Dedup/splits: unique (template, primary entity); splits bakeoff_dev / heldout_eval / sft_train are
disjoint by construction and HARD-FAIL-checked before writing (mirrors the GRPO leakage-guard lesson).

Run inside the tau2 venv (needs tau2 + the domain):
    PYTHONPATH=. <tau2-venv>/bin/python sdg_pipeline/db_bahn/gen_tasks.py --seed 42
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from sdg_pipeline.db_bahn.tau2_domain.data_model import BahnDB
from sdg_pipeline.db_bahn.tau2_domain.environment import DATA_DIR, DB_PATH
from sdg_pipeline.db_bahn.tau2_domain.tools import BahnTools

DELAY_CAUSES = ["Signalstörung", "Bauarbeiten", "technische Störung am Zug", "Notarzteinsatz"]
MAINT_TYPES = ["Inspektion", "Reparatur", "Radsatztausch", "Softwareupdate"]


def rng(*keys) -> random.Random:
    h = hashlib.sha256("|".join(str(k) for k in keys).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


# ---------------------------------------------------------------------------------------
# small helpers around the domain
# ---------------------------------------------------------------------------------------
class Gen:
    def __init__(self, db_path: Path, seed: int):
        self.master = BahnDB.load(str(db_path))
        self.seed = seed
        self.tools = BahnTools(self.master)  # read-only eligibility queries on the master copy

    def fresh(self) -> BahnTools:
        """Independent DB copy + toolkit (for fact computation / reference replay)."""
        return BahnTools(self.master.model_copy(deep=True))

    # eligibility pools (deterministic order: sort by id)
    def pool(self, kind: str) -> list:
        db = self.master
        if kind == "trips":
            return sorted(db.trips.values(), key=lambda t: t.trip_id)
        if kind == "trips_pos":
            pos = {p.trip_id for p in db.positions}
            return [t for t in self.pool("trips") if t.trip_id in pos]
        if kind == "trips_orders":
            veh = {o.vehicle_id for o in db.maintenance_orders.values()}
            return [t for t in self.pool("trips") if t.vehicle_id in veh]
        if kind == "trips_lokf":
            lokf = {a.trip_id for a in db.assignments if a.role == "Lokführer"}
            return [t for t in self.pool("trips") if t.trip_id in lokf]
        raise ValueError(kind)

    def spare_lokfuehrer(self, trip_id: str, r: random.Random):
        assigned = {a.emp_id for a in self.master.assignments if a.trip_id == trip_id}
        cands = sorted(e.emp_id for e in self.master.employees.values()
                       if e.role == "Lokführer" and e.emp_id not in assigned)
        return self.master.employees[r.choice(cands)]


def inj_call(func_name: str, **arguments) -> dict:
    return {"env_type": "assistant", "func_name": func_name, "arguments": arguments}


def env_assert(func_name: str, assert_value: bool = True, **arguments) -> dict:
    return {"env_type": "assistant", "func_name": func_name, "arguments": arguments,
            "assert_value": assert_value}


def ref_action(i: int, name: str, **arguments) -> dict:
    return {"action_id": f"a{i}", "requestor": "assistant", "name": name, "arguments": arguments}


def build_task(task_id, ticket, purpose, injections, ref_actions, env_assertions,
               communicate, reward_basis) -> dict:
    task = {
        "id": task_id,
        "description": {"purpose": purpose},
        "user_scenario": {"instructions": ticket},
        "ticket": ticket,
        "evaluation_criteria": {
            "actions": ref_actions or None,
            "env_assertions": env_assertions or None,
            "communicate_info": communicate or None,
            "nl_assertions": None,
            "reward_basis": reward_basis,
        },
    }
    if injections:
        task["initial_state"] = {"initialization_actions": injections}
    return task


# ---------------------------------------------------------------------------------------
# templates — each returns (task_dict, answer_key_dict) or None if entity not usable
# ---------------------------------------------------------------------------------------
def t_info_verspaetung(g: Gen, trip, idx, inject: bool):
    r = rng(g.seed, "inj", "info_verspaetung", trip.trip_id)
    injections = []
    if inject:
        injections = [inj_call("inject_verspaetung", zugnummer=trip.zugnummer,
                               minuten=r.choice([25, 35, 45, 60, 90]), grund=r.choice(DELAY_CAUSES))]
    tk = g.fresh()
    for c in injections:
        getattr(tk, c["func_name"])(**c["arguments"])
    v = tk.verspaetung(trip.zugnummer)
    comm = [v["grund"]] if v["verspaetung_minuten"] > 0 else ["pünktlich"]
    ticket = (f"Prüfe die aktuelle Verspätung von {trip.zugnummer}. "
              f"Wie viele Minuten Verspätung hat der Zug und aus welchem Grund?")
    task = build_task(f"info_verspaetung__{trip.zugnummer.replace(' ', '-')}__{idx:03d}", ticket,
                      "INFO: aktuelle Verspätung + Grund abfragen", injections, None, None,
                      comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["verspaetung"], "facts": v}
    return task, key


def t_info_standort(g: Gen, trip, idx, inject: bool):
    tk = g.fresh()
    s = tk.zugstandort(trip.zugnummer)
    if s.get("status") != "unterwegs" or not s.get("naechster_halt"):
        return None
    ticket = f"Wo befindet sich {trip.zugnummer} gerade? Nenne insbesondere den nächsten Halt."
    task = build_task(f"info_standort__{trip.zugnummer.replace(' ', '-')}__{idx:03d}", ticket,
                      "INFO: aktueller Standort + nächster Halt", [], None, None,
                      [s["naechster_halt"]], ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["zugstandort"], "facts": s}
    return task, key


def t_info_ankunft(g: Gen, trip, idx, inject: bool):
    r = rng(g.seed, "inj", "info_ankunft", trip.trip_id)
    injections = []
    if inject:
        injections = [inj_call("inject_verspaetung", zugnummer=trip.zugnummer,
                               minuten=r.choice([30, 45, 60, 75]), grund=r.choice(DELAY_CAUSES))]
    tk = g.fresh()
    for c in injections:
        getattr(tk, c["func_name"])(**c["arguments"])
    s = tk.zugstandort(trip.zugnummer)
    if s.get("status") != "unterwegs":
        return None
    v = tk.verspaetung(trip.zugnummer)
    fp = tk.fahrplan(trip.zugnummer)
    ziel = fp["nach"]
    comm = [fp["ankunft"], s["naechster_halt"]] + ([v["grund"]] if v["verspaetung_minuten"] > 0 else ["pünktlich"])
    ticket = (f"Wo ist {trip.zugnummer} gerade unterwegs, und kommt der Zug voraussichtlich pünktlich in "
              f"{ziel} an? Nenne in der Antwort ausdrücklich den nächsten Halt, die PLANMÄSSIGE "
              f"Ankunftszeit in {ziel} und die aktuelle Verspätung (Minuten und Grund, oder 'pünktlich'); "
              f"schätze dann kurz ein, ob die Ankunft pünktlich erfolgt.")
    task = build_task(f"info_ankunft__{trip.zugnummer.replace(' ', '-')}__{idx:03d}", ticket,
                      "INFO: Standort + Verspätung + Ankunft kombinieren (3 Tools)", injections,
                      None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["zugstandort", "verspaetung", "fahrplan"],
           "facts": {"standort": s, "verspaetung": v,
                     "ankunft_plan": fp["ankunft"], "ziel": ziel}}
    return task, key


def t_info_wartung(g: Gen, trip, idx, inject: bool):
    tk = g.fresh()
    w = tk.wartung_status(trip.zugnummer)
    offen = [o for o in w["wartungsauftraege"] if o["status"] != "abgeschlossen"]
    if not offen:
        return None
    comm = [offen[0]["order_id"], offen[0]["due_at"][:10]]
    ticket = (f"Welche offenen Wartungsaufträge gibt es für das Fahrzeug von {trip.zugnummer}? "
              f"Nenne Auftrags-ID, Typ, Status und Fälligkeit.")
    task = build_task(f"info_wartung__{trip.zugnummer.replace(' ', '-')}__{idx:03d}", ticket,
                      "INFO: offene Wartungsaufträge eines Fahrzeugs", [], None, None,
                      comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["wartung_status"],
           "facts": {"fahrzeug_id": w["fahrzeug_id"], "offene_auftraege": offen}}
    return task, key


def t_info_crew(g: Gen, trip, idx, inject: bool):
    tk = g.fresh()
    m = tk.mitarbeiter_info(trip.zugnummer)
    lokf = [c for c in m["besatzung"] if c["rolle"] == "Lokführer"]
    if not lokf:
        return None
    comm = [lokf[0]["mitarbeiter_id"], lokf[0]["name"]]
    ticket = (f"Wer ist auf {trip.zugnummer} als Lokführer eingeteilt? "
              f"Nenne Name und Mitarbeiter-ID.")
    task = build_task(f"info_crew__{trip.zugnummer.replace(' ', '-')}__{idx:03d}", ticket,
                      "INFO: Besatzung eines Zuges abfragen", [], None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["mitarbeiter_info"], "facts": {"lokfuehrer": lokf}}
    return task, key


def t_info_wartung_machbar(g: Gen, trip, idx, inject: bool):
    r = rng(g.seed, "inj", "info_machbar", trip.trip_id)
    injections = []
    if inject:
        injections = [inj_call("inject_verspaetung", zugnummer=trip.zugnummer,
                               minuten=r.choice([45, 60, 90]), grund=r.choice(DELAY_CAUSES))]
    tk = g.fresh()
    for c in injections:
        getattr(tk, c["func_name"])(**c["arguments"])
    w = tk.wartung_status(trip.zugnummer)
    offen = [o for o in w["wartungsauftraege"] if o["status"] in ("geplant", "überfällig")]
    if not offen:
        return None
    order = offen[0]
    v = tk.verspaetung(trip.zugnummer)
    fp = tk.fahrplan(trip.zugnummer)
    ticket = (f"Für das Fahrzeug von {trip.zugnummer} ist der Wartungsauftrag {order['order_id']} "
              f"({order['type']}) am {order['due_at']} fällig. Prüfe die planmäßige Ankunft des Zuges und "
              f"die aktuelle Verspätung: Ist die Wartung zeitlich machbar? Nenne in der Antwort ausdrücklich "
              f"die Auftrags-ID, die PLANMÄSSIGE Ankunftszeit und die aktuelle Verspätung (Minuten/Grund, "
              f"oder 'pünktlich') und begründe deine Einschätzung kurz.")
    comm = [order["order_id"], fp["ankunft"]] + ([v["grund"]] if v["verspaetung_minuten"] > 0 else [])
    task = build_task(f"info_machbar__{trip.zugnummer.replace(' ', '-')}__{idx:03d}", ticket,
                      "INFO: Wartungs-Machbarkeit aus Ankunft+Verspätung ableiten (Mehr-Tool)",
                      injections, None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["wartung_status", "verspaetung", "fahrplan"],
           "facts": {"order": order, "ankunft_plan": fp["ankunft"],
                     "verspaetung_min": v["verspaetung_minuten"]}}
    return task, key


def t_action_wartung(g: Gen, trip, idx, inject: bool):
    r = rng(g.seed, "act", "wartung", trip.trip_id)
    typ = r.choice(MAINT_TYPES)
    due = f"2026-07-{r.randint(4, 10):02d} {r.choice(['06:00', '08:00', '22:00'])}"
    ticket = (f"Plane für das Fahrzeug von {trip.zugnummer} eine Wartung vom Typ '{typ}' ein, "
              f"fällig am {due}. Ermittle zuerst die Fahrzeug-ID über den Wartungsstatus des Zuges.")
    refs = [ref_action(1, "wartung_einplanen", fahrzeug_id=trip.vehicle_id, typ=typ, faellig_am=due)]
    asserts = [env_assert("assert_maintenance_exists", fahrzeug_id=trip.vehicle_id, typ=typ)]
    task = build_task(f"action_wartung__{trip.zugnummer.replace(' ', '-')}__{idx:03d}", ticket,
                      "ACTION: Wartungsauftrag anlegen (Lookup + Write)", [], refs, asserts,
                      None, ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "expected_tools": ["wartung_status", "wartung_einplanen"],
           "facts": {"fahrzeug_id": trip.vehicle_id, "typ": typ, "faellig_am": due}}
    return task, key


def t_action_crew(g: Gen, trip, idx, inject: bool):
    r = rng(g.seed, "act", "crew", trip.trip_id)
    emp = g.spare_lokfuehrer(trip.trip_id, r)
    rolle = "Zugbegleiter" if r.random() < 0.5 else "Lokführer"
    ticket = (f"Teile {emp.name} (Mitarbeiter-ID {emp.emp_id}) dem Zug {trip.zugnummer} "
              f"als {rolle} zu und bestätige die Zuteilung.")
    refs = [ref_action(1, "crew_zuweisen", zugnummer=trip.zugnummer,
                       mitarbeiter_id=emp.emp_id, rolle=rolle)]
    asserts = [env_assert("assert_crew_assigned", zugnummer=trip.zugnummer, mitarbeiter_id=emp.emp_id)]
    task = build_task(f"action_crew__{trip.zugnummer.replace(' ', '-')}__{idx:03d}", ticket,
                      "ACTION: Mitarbeiter einem Zug zuteilen", [], refs, asserts,
                      [emp.emp_id], ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "expected_tools": ["crew_zuweisen"],
           "facts": {"emp_id": emp.emp_id, "rolle": rolle}}
    return task, key


def t_action_wartung_status(g: Gen, trip, idx, inject: bool):
    tk = g.fresh()
    w = tk.wartung_status(trip.zugnummer)
    offen = [o for o in w["wartungsauftraege"] if o["status"] != "abgeschlossen"]
    if not offen:
        return None
    order = offen[0]
    ticket = (f"Der Wartungsauftrag {order['order_id']} (Fahrzeug von {trip.zugnummer}) wurde erledigt. "
              f"Setze seinen Status auf 'abgeschlossen' und bestätige kurz.")
    refs = [ref_action(1, "wartung_status_setzen", auftrag_id=order["order_id"], status="abgeschlossen")]
    asserts = [env_assert("assert_maintenance_status", auftrag_id=order["order_id"], status="abgeschlossen")]
    task = build_task(f"action_wstatus__{trip.zugnummer.replace(' ', '-')}__{idx:03d}", ticket,
                      "ACTION: Wartungsstatus setzen", [], refs, asserts,
                      [order["order_id"]], ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "expected_tools": ["wartung_status_setzen"],
           "facts": {"auftrag_id": order["order_id"]}}
    return task, key


def t_action_ersatz(g: Gen, trip, idx, inject: bool):
    """Always fault-injected: the assigned Lokführer drops out -> agent must check + assign spare."""
    r = rng(g.seed, "act", "ersatz", trip.trip_id)
    tk0 = BahnTools(g.master)
    orig = [c for c in tk0.mitarbeiter_info(trip.zugnummer)["besatzung"] if c["rolle"] == "Lokführer"]
    if not orig:
        return None
    ersatz = g.spare_lokfuehrer(trip.trip_id, r)
    injections = [inj_call("inject_lokfuehrer_ausfall", zugnummer=trip.zugnummer)]
    ticket = (f"Der eingeteilte Lokführer von {trip.zugnummer} ist kurzfristig ausgefallen. "
              f"Prüfe die aktuelle Besatzung des Zuges und teile {ersatz.name} "
              f"(Mitarbeiter-ID {ersatz.emp_id}) als Ersatz-Lokführer zu.")
    refs = [ref_action(1, "crew_zuweisen", zugnummer=trip.zugnummer,
                       mitarbeiter_id=ersatz.emp_id, rolle="Lokführer")]
    asserts = [
        env_assert("assert_crew_assigned", zugnummer=trip.zugnummer, mitarbeiter_id=ersatz.emp_id),
        env_assert("assert_crew_assigned", assert_value=False,
                   zugnummer=trip.zugnummer, mitarbeiter_id=orig[0]["mitarbeiter_id"]),
    ]
    task = build_task(f"action_ersatz__{trip.zugnummer.replace(' ', '-')}__{idx:03d}", ticket,
                      "ACTION+REPLAN: Ausfall erkennen und Ersatz-Lokführer zuteilen (fault-injected)",
                      injections, refs, asserts, [ersatz.emp_id], ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "expected_tools": ["mitarbeiter_info", "crew_zuweisen"],
           "facts": {"ersatz_id": ersatz.emp_id, "ausgefallen": orig[0]["mitarbeiter_id"]}}
    return task, key


TEMPLATES = [
    # (template_fn, eligibility pool, injectable)
    (t_info_verspaetung, "trips", True),
    (t_info_standort, "trips_pos", False),
    (t_info_ankunft, "trips_pos", True),
    (t_info_wartung, "trips_orders", False),
    (t_info_crew, "trips_lokf", False),
    (t_info_wartung_machbar, "trips_orders", True),
    (t_action_wartung, "trips", False),
    (t_action_crew, "trips_lokf", False),
    (t_action_wartung_status, "trips_orders", False),
    (t_action_ersatz, "trips_lokf", None),  # None = always injected (built into template)
]


def main():
    ap = argparse.ArgumentParser(description="Generate db_bahn tasks with built-in answer keys")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-per-template", type=int, default=55)
    ap.add_argument("--fault-rate", type=float, default=0.35)
    ap.add_argument("--n-bakeoff", type=int, default=25)
    ap.add_argument("--n-heldout", type=int, default=40)
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    args = ap.parse_args()

    g = Gen(DB_PATH, args.seed)
    out_dir = Path(args.out_dir)

    tasks, keys, seen = [], {}, set()
    stats = Counter()
    for fn, pool_name, injectable in TEMPLATES:
        tname = fn.__name__
        pool = g.pool(pool_name)
        order = rng(args.seed, "pool", tname).sample(range(len(pool)), len(pool))
        made = 0
        for j in order:
            if made >= args.n_per_template:
                break
            trip = pool[j]
            dedup_key = (tname, trip.zugnummer)
            if dedup_key in seen:  # unique (template, primary entity)
                stats["near_dup_skipped"] += 1
                continue
            if injectable is None:
                inject = True
            elif injectable:
                inject = rng(args.seed, "faultroll", tname, trip.trip_id).random() < args.fault_rate
            else:
                inject = False
            res = fn(g, trip, made, inject)
            if res is None:
                stats[f"{tname}__ineligible"] += 1
                continue
            task, key = res
            seen.add(dedup_key)
            key["template"] = tname
            key["injected"] = bool(task.get("initial_state"))
            tasks.append(task)
            keys[task["id"]] = key
            made += 1
            stats[tname] += 1
            if key["injected"]:
                stats["injected_total"] += 1

    # splits: balanced round-robin over templates, disjoint by construction, then HARD-FAIL check
    by_tpl = {}
    for t in tasks:
        by_tpl.setdefault(keys[t["id"]]["template"], []).append(t["id"])
    for tpl in by_tpl:
        by_tpl[tpl] = sorted(by_tpl[tpl])
        rng(args.seed, "split", tpl).shuffle(by_tpl[tpl])

    def take(n):
        got = []
        while len(got) < n:
            progress = False
            for tpl in sorted(by_tpl):
                if by_tpl[tpl] and len(got) < n:
                    got.append(by_tpl[tpl].pop())
                    progress = True
            if not progress:
                break
        return got

    splits = {"bakeoff_dev": take(args.n_bakeoff), "heldout_eval": take(args.n_heldout)}
    splits["sft_train"] = [i for tpl in sorted(by_tpl) for i in by_tpl[tpl]]

    all_ids = [i for s in splits.values() for i in s]
    assert len(all_ids) == len(set(all_ids)) == len(tasks), (
        f"HARD-FAIL: split overlap or loss ({len(all_ids)} vs {len(set(all_ids))} vs {len(tasks)})")

    tasks.sort(key=lambda t: t["id"])
    (out_dir / "tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    (out_dir / "split_tasks.json").write_text(
        json.dumps({k: sorted(v) for k, v in splits.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "answer_keys.json").write_text(
        json.dumps(keys, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")

    print(f"tasks: {len(tasks)}  (injected: {stats['injected_total']}, "
          f"near-dup skipped: {stats['near_dup_skipped']})")
    for fn, _, _ in TEMPLATES:
        print(f"  {fn.__name__:28s}: {stats[fn.__name__]}")
    print("splits:", {k: len(v) for k, v in splits.items()})
    print(f"wrote tasks.json / split_tasks.json / answer_keys.json -> {out_dir}")


if __name__ == "__main__":
    main()
