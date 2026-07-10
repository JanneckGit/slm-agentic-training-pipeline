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
  - 25 templates: 9 polished easy/mid singles+chains, 16 hard ones — search tasks WITHOUT pre-given
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
import hashlib
import json
import random
import re
from collections import Counter, namedtuple
from pathlib import Path
from typing import Callable, NamedTuple, Optional

from sdg_pipeline.db_bahn.tau2_domain.data_model import BahnDB
from sdg_pipeline.db_bahn.tau2_domain.environment import DATA_DIR, DB_PATH
from sdg_pipeline.db_bahn.tau2_domain.tools import BahnTools, QUALI_PRODUKTE

DELAY_CAUSES = ["Signalstörung", "Bauarbeiten", "technische Störung am Zug", "Notarzteinsatz"]
MAINT_TYPES = ["Inspektion", "Reparatur", "Radsatztausch", "Softwareupdate"]
QUALS = ["ICE", "IC", "EC", "Nacht", "Gefahrgut"]
ROLES = ["Lokführer", "Zugbegleiter", "Techniker", "Disponent"]

MACombo = namedtuple("MACombo", "key rolle qualifikation heimatbasis verfuegbar_um emp_ids")
WOCombo = namedtuple("WOCombo", "key status depot faellig_vor order_ids")


def rng(*keys) -> random.Random:
    h = hashlib.sha256("|".join(str(k) for k in keys).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def sid(s: str) -> str:
    """Content-derived id fragment: umlaut-transliterated, [A-Za-z0-9_.-] only."""
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("Ä", "Ae"), ("Ö", "Oe"), ("Ü", "Ue"), ("ß", "ss")):
        s = s.replace(a, b)
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-")


def qual_for(trip) -> Optional[str]:
    """Product qualification required for a Lokführer assignment on this trip (None if ungated)."""
    return trip.product if trip.product in QUALI_PRODUKTE else None


# ---------------------------------------------------------------------------------------
# small helpers around the domain
# ---------------------------------------------------------------------------------------
class Gen:
    def __init__(self, db_path: Path, seed: int):
        self.master = BahnDB.load(str(db_path))
        self.seed = seed
        self.tools = BahnTools(self.master)  # read-only eligibility queries on the master copy
        self._pool_cache: dict[str, list] = {}

    def fresh(self) -> BahnTools:
        """Independent DB copy + toolkit (for fact computation / reference replay)."""
        return BahnTools(self.master.model_copy(deep=True))

    # eligibility pools (deterministic order)
    def pool(self, kind: str) -> list:
        if kind not in self._pool_cache:
            self._pool_cache[kind] = self._build_pool(kind)
        return self._pool_cache[kind]

    def _build_pool(self, kind: str) -> list:
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
        if kind == "trips_lokf_no_quali_produkt":
            # Gefahrgut template: products without a product-qualification gate (search can only
            # filter ONE qualification, so the assignment must not additionally require ICE/IC/EC)
            return [t for t in self.pool("trips_lokf") if t.product not in QUALI_PRODUKTE]
        if kind == "trips_route_page":
            # trips findable via zuege_suchen(von, nach, produkt) within one result page, and
            # uniquely pinned by the ticket's Abfahrt (unique dep_time within the route group)
            groups: dict[tuple, list] = {}
            for t in self.pool("trips"):
                groups.setdefault((t.origin_station, t.dest_station, t.product), []).append(t)
            out = []
            for ts in groups.values():
                if len(ts) > 10:
                    continue
                deps = Counter(t.dep_time for t in ts)
                out.extend(t for t in ts if deps[t.dep_time] == 1)
            return sorted(out, key=lambda t: t.trip_id)
        if kind == "trips_pos_route":
            pos = {p.trip_id for p in db.positions}
            return [t for t in self.pool("trips_route_page") if t.trip_id in pos]
        if kind == "trips_komplett":
            pos = {p.trip_id for p in db.positions}
            open_veh = {o.vehicle_id for o in db.maintenance_orders.values() if o.status != "abgeschlossen"}
            lokf = {a.trip_id for a in db.assignments if a.role == "Lokführer"}
            return [t for t in self.pool("trips")
                    if t.trip_id in pos and t.vehicle_id in open_veh and t.trip_id in lokf]
        if kind == "trips_veh_ueberfaellig":
            # exactly ONE überfällig order among >=2 orders -> "find the one" is a real selection
            ok = {v for v, os_ in self._veh_orders().items()
                  if sum(o.status == "überfällig" for o in os_) == 1 and len(os_) >= 2}
            return self._rep_trip_per_vehicle(ok)
        if kind == "trips_veh_konflikt":
            # >=1 abgeschlossen (terminal -> rejection) and exactly ONE geplant/überfällig fallback;
            # no in_Arbeit orders (they would make "der offene Auftrag" ambiguous / the write vacuous)
            ok = {v for v, os_ in self._veh_orders().items()
                  if any(o.status == "abgeschlossen" for o in os_)
                  and sum(o.status in ("geplant", "überfällig") for o in os_) == 1
                  and not any(o.status == "in_Arbeit" for o in os_)}
            return self._rep_trip_per_vehicle(ok)
        if kind == "trips_veh_2open":
            ok = {v for v, os_ in self._veh_orders().items()
                  if sum(o.status in ("geplant", "überfällig") for o in os_) == 2}
            return self._rep_trip_per_vehicle(ok)
        if kind == "ma_filter_combos":
            return self._ma_combos()
        if kind == "wartung_filter_combos":
            return self._wo_combos()
        raise ValueError(kind)

    def _veh_orders(self) -> dict[str, list]:
        m: dict[str, list] = {}
        for o in self.master.maintenance_orders.values():
            m.setdefault(o.vehicle_id, []).append(o)
        return m

    def _rep_trip_per_vehicle(self, vehicles: set) -> list:
        """One representative trip per qualifying vehicle (smallest zugnummer): the (template,
        zugnummer) dedup then automatically prevents near-duplicate answers for shared vehicles."""
        by_veh: dict[str, object] = {}
        for t in self.pool("trips"):
            if t.vehicle_id in vehicles:
                cur = by_veh.get(t.vehicle_id)
                if cur is None or t.zugnummer < cur.zugnummer:
                    by_veh[t.vehicle_id] = t
        return [by_veh[v] for v in sorted(by_veh)]

    def _shift_of(self) -> dict:
        shift_of = {}
        for s in self.master.shifts:
            shift_of.setdefault(s.emp_id, s)
        return shift_of

    def _ma_combos(self) -> list:
        """Filter combos (rolle, qualifikation, heimatbasis, verfuegbar_um) with 1-3 matches —
        the tool decides availability, communicate checks exact emp_ids (info_machbar lesson)."""
        shift_of = self._shift_of()
        emp_by: dict[tuple, list] = {}
        for eid in sorted(self.master.employees):
            e = self.master.employees[eid]
            emp_by.setdefault((e.role, e.home_base), []).append(e)
        times = ["04:30", "06:15", "08:45", "13:30", "18:45", "21:15", "23:30"]
        combos = []
        for (rolle, base), emps in sorted(emp_by.items()):
            for q in QUALS:
                having_q = [e for e in emps if q in e.qualifications]
                if not having_q:
                    continue
                for um in times:
                    hits = tuple(e.emp_id for e in having_q
                                 if (sh := shift_of.get(e.emp_id)) and sh.start <= um <= sh.end)
                    if 1 <= len(hits) <= 3:
                        combos.append(MACombo(f"{rolle}|{q}|{base}|{um}", rolle, q, base, um, hits))
        return combos

    def _wo_combos(self) -> list:
        """Filter combos (status, depot, faellig_vor) with 1-3 matching maintenance orders."""
        depots = sorted({o.depot for o in self.master.maintenance_orders.values()})
        cutoffs = ["2026-06-25", "2026-06-29", "2026-07-02", "2026-07-06", "2026-07-13"]
        combos = []
        for status in ("geplant", "in_Arbeit", "überfällig"):
            for depot in depots:
                for cutoff in cutoffs:
                    hits = tuple(sorted(o.order_id for o in self.master.maintenance_orders.values()
                                        if o.status == status and o.depot == depot and o.due_at < cutoff))
                    if 1 <= len(hits) <= 3:
                        combos.append(WOCombo(f"{status}|{depot}|{cutoff}", status, depot, cutoff, hits))
        return combos

    def spare_mitarbeiter(self, trip_id: str, r: random.Random, rolle: str,
                          qualifikation: Optional[str] = None):
        """Rule-conform reference pick: matching role (+ product qualification for Lokführer on
        ICE/IC/EC), not already assigned to the trip. Returns None if no candidate exists."""
        assigned = {a.emp_id for a in self.master.assignments if a.trip_id == trip_id}
        cands = sorted(e.emp_id for e in self.master.employees.values()
                       if e.role == rolle and e.emp_id not in assigned
                       and (qualifikation is None or qualifikation in e.qualifications))
        return self.master.employees[r.choice(cands)] if cands else None


def inj_call(func_name: str, **arguments) -> dict:
    return {"env_type": "assistant", "func_name": func_name, "arguments": arguments}


def env_assert(func_name: str, assert_value: bool = True, **arguments) -> dict:
    return {"env_type": "assistant", "func_name": func_name, "arguments": arguments,
            "assert_value": assert_value}


def ref_action(i: int, name: str, **arguments) -> dict:
    return {"action_id": f"a{i}", "requestor": "assistant", "name": name, "arguments": arguments}


def oc(name: str, **arguments) -> dict:
    """One oracle_calls entry: the exact valid-path call (drives rollout.py's dry-run oracle)."""
    return {"name": name, "arguments": arguments}


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


def _apply(tk: BahnTools, injections: list[dict]) -> BahnTools:
    for c in injections:
        getattr(tk, c["func_name"])(**c["arguments"])
    return tk


def _grund_comm(v: dict) -> list[str]:
    return [v["grund"]] if v["verspaetung_minuten"] > 0 else ["pünktlich"]


# ---------------------------------------------------------------------------------------
# easy/mid tier — polished wave-1 templates (regenerated under the 11-tool domain)
# ---------------------------------------------------------------------------------------
def t_info_verspaetung(g: Gen, trip, idx, inject: bool):
    r = rng(g.seed, "inj", "info_verspaetung", trip.trip_id)
    injections = []
    if inject:
        injections = [inj_call("inject_verspaetung", zugnummer=trip.zugnummer,
                               minuten=r.choice([25, 35, 45, 60, 90]), grund=r.choice(DELAY_CAUSES))]
    tk = _apply(g.fresh(), injections)
    v = tk.verspaetung(trip.zugnummer)
    ticket = (f"Prüfe die aktuelle Verspätung von {trip.zugnummer}. "
              f"Wie viele Minuten Verspätung hat der Zug und aus welchem Grund?")
    task = build_task(f"info_verspaetung__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "INFO: aktuelle Verspätung + Grund abfragen", injections, None, None,
                      _grund_comm(v), ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["verspaetung"],
           "oracle_calls": [oc("verspaetung", zugnummer=trip.zugnummer)], "facts": v}
    return task, key


def t_info_standort(g: Gen, trip, idx, inject: bool):
    tk = g.fresh()
    s = tk.zugstandort(trip.zugnummer)
    if s.get("status") != "unterwegs" or not s.get("naechster_halt"):
        return None
    ticket = f"Wo befindet sich {trip.zugnummer} gerade? Nenne insbesondere den nächsten Halt."
    task = build_task(f"info_standort__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "INFO: aktueller Standort + nächster Halt", [], None, None,
                      [s["naechster_halt"]], ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["zugstandort"],
           "oracle_calls": [oc("zugstandort", zugnummer=trip.zugnummer)], "facts": s}
    return task, key


def t_info_ankunft(g: Gen, trip, idx, inject: bool):
    r = rng(g.seed, "inj", "info_ankunft", trip.trip_id)
    injections = []
    if inject:
        injections = [inj_call("inject_verspaetung", zugnummer=trip.zugnummer,
                               minuten=r.choice([30, 45, 60, 75]), grund=r.choice(DELAY_CAUSES))]
    tk = _apply(g.fresh(), injections)
    s = tk.zugstandort(trip.zugnummer)
    if s.get("status") != "unterwegs":
        return None
    v = tk.verspaetung(trip.zugnummer)
    fp = tk.fahrplan(trip.zugnummer)
    ziel = fp["nach"]
    comm = [fp["ankunft"], s["naechster_halt"]] + _grund_comm(v)
    ticket = (f"Wo ist {trip.zugnummer} gerade unterwegs, und kommt der Zug voraussichtlich pünktlich in "
              f"{ziel} an? Nenne in der Antwort ausdrücklich den nächsten Halt, die PLANMÄSSIGE "
              f"Ankunftszeit in {ziel} und die aktuelle Verspätung (Minuten und Grund, oder 'pünktlich'); "
              f"schätze dann kurz ein, ob die Ankunft pünktlich erfolgt.")
    task = build_task(f"info_ankunft__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "INFO: Standort + Verspätung + Ankunft kombinieren (3 Tools)", injections,
                      None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["zugstandort", "verspaetung", "fahrplan"],
           "oracle_calls": [oc("zugstandort", zugnummer=trip.zugnummer),
                            oc("verspaetung", zugnummer=trip.zugnummer),
                            oc("fahrplan", zugnummer=trip.zugnummer)],
           "facts": {"standort": s, "verspaetung": v, "ankunft_plan": fp["ankunft"], "ziel": ziel}}
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
    task = build_task(f"info_wartung__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "INFO: offene Wartungsaufträge eines Fahrzeugs", [], None, None,
                      comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["wartung_status"],
           "oracle_calls": [oc("wartung_status", kennung=trip.zugnummer)],
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
    task = build_task(f"info_crew__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "INFO: Besatzung eines Zuges abfragen", [], None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["mitarbeiter_info"],
           "oracle_calls": [oc("mitarbeiter_info", zugnummer=trip.zugnummer)],
           "facts": {"lokfuehrer": lokf}}
    return task, key


def t_action_wartung(g: Gen, trip, idx, inject: bool):
    r = rng(g.seed, "act", "wartung", trip.trip_id)
    typ = r.choice(MAINT_TYPES)
    due = f"2026-07-{r.randint(4, 10):02d} {r.choice(['06:00', '08:00', '22:00'])}"
    ticket = (f"Plane für das Fahrzeug von {trip.zugnummer} eine Wartung vom Typ '{typ}' ein, "
              f"fällig am {due}. Ermittle zuerst die Fahrzeug-ID über den Wartungsstatus des Zuges.")
    refs = [ref_action(1, "wartung_einplanen", fahrzeug_id=trip.vehicle_id, typ=typ, faellig_am=due)]
    asserts = [env_assert("assert_maintenance_exists", fahrzeug_id=trip.vehicle_id, typ=typ)]
    task = build_task(f"action_wartung__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION: Wartungsauftrag anlegen (Lookup + Write)", [], refs, asserts,
                      None, ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "expected_tools": ["wartung_status", "wartung_einplanen"],
           "oracle_calls": [oc("wartung_status", kennung=trip.zugnummer),
                            oc("wartung_einplanen", fahrzeug_id=trip.vehicle_id, typ=typ, faellig_am=due)],
           "facts": {"fahrzeug_id": trip.vehicle_id, "typ": typ, "faellig_am": due}}
    return task, key


def t_action_crew(g: Gen, trip, idx, inject: bool):
    r = rng(g.seed, "act", "crew", trip.trip_id)
    rolle = "Zugbegleiter" if r.random() < 0.5 else "Lokführer"
    emp = g.spare_mitarbeiter(trip.trip_id, r, rolle,
                              qual_for(trip) if rolle == "Lokführer" else None)
    if emp is None:
        return None
    ticket = (f"Teile {emp.name} (Mitarbeiter-ID {emp.emp_id}) dem Zug {trip.zugnummer} "
              f"als {rolle} zu und bestätige die Zuteilung.")
    refs = [ref_action(1, "crew_zuweisen", zugnummer=trip.zugnummer,
                       mitarbeiter_id=emp.emp_id, rolle=rolle)]
    asserts = [env_assert("assert_crew_assigned", zugnummer=trip.zugnummer, mitarbeiter_id=emp.emp_id)]
    task = build_task(f"action_crew__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION: Mitarbeiter einem Zug zuteilen", [], refs, asserts,
                      [emp.emp_id], ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "expected_tools": ["crew_zuweisen"],
           "oracle_calls": [oc("crew_zuweisen", zugnummer=trip.zugnummer,
                               mitarbeiter_id=emp.emp_id, rolle=rolle)],
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
    task = build_task(f"action_wartung_status__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION: Wartungsstatus setzen", [], refs, asserts,
                      [order["order_id"]], ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "expected_tools": ["wartung_status_setzen"],
           "oracle_calls": [oc("wartung_status_setzen", auftrag_id=order["order_id"], status="abgeschlossen")],
           "facts": {"auftrag_id": order["order_id"]}}
    return task, key


def t_action_ersatz(g: Gen, trip, idx, inject: bool):
    """Always fault-injected: the assigned Lokführer drops out -> agent must check + assign spare."""
    r = rng(g.seed, "act", "ersatz", trip.trip_id)
    tk0 = BahnTools(g.master)
    orig = [c for c in tk0.mitarbeiter_info(trip.zugnummer)["besatzung"] if c["rolle"] == "Lokführer"]
    if not orig:
        return None
    ersatz = g.spare_mitarbeiter(trip.trip_id, r, "Lokführer", qual_for(trip))
    if ersatz is None:
        return None
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
    task = build_task(f"action_ersatz__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION+REPLAN: Ausfall erkennen und Ersatz-Lokführer zuteilen (fault-injected)",
                      injections, refs, asserts, [ersatz.emp_id], ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "expected_tools": ["mitarbeiter_info", "crew_zuweisen"],
           "oracle_calls": [oc("mitarbeiter_info", zugnummer=trip.zugnummer),
                            oc("crew_zuweisen", zugnummer=trip.zugnummer,
                               mitarbeiter_id=ersatz.emp_id, rolle="Lokführer")],
           "facts": {"ersatz_id": ersatz.emp_id, "ausgefallen": orig[0]["mitarbeiter_id"]}}
    return task, key


# ---------------------------------------------------------------------------------------
# hard tier — search without ids, long chains, conditional writes, runtime faults
# ---------------------------------------------------------------------------------------
def _route_search_ok(tk: BahnTools, trip, von: str, nach: str) -> bool:
    res = tk.zuege_suchen(von=von, nach=nach, produkt=trip.product)
    return any(row["zugnummer"] == trip.zugnummer for row in res["treffer"])


def t_info_zugsuche_status(g: Gen, trip, idx, inject: bool):
    r = rng(g.seed, "inj", "zugsuche_status", trip.trip_id)
    injections = []
    if inject:
        injections = [inj_call("inject_verspaetung", zugnummer=trip.zugnummer,
                               minuten=r.choice([25, 35, 45, 60]), grund=r.choice(DELAY_CAUSES))]
    tk = _apply(g.fresh(), injections)
    von = tk._station_name(trip.origin_station)
    nach = tk._station_name(trip.dest_station)
    s = tk.zugstandort(trip.zugnummer)
    if s.get("status") != "unterwegs" or not s.get("naechster_halt"):
        return None
    if not _route_search_ok(tk, trip, von, nach):
        return None
    v = tk.verspaetung(trip.zugnummer)
    comm = [trip.zugnummer, s["naechster_halt"]] + _grund_comm(v)
    ticket = (f"Ein {trip.product} von {von} nach {nach} mit Abfahrt {trip.dep_time} — finde den Zug "
              f"über die Zugsuche und berichte: Zugnummer, nächster Halt und aktuelle Verspätung "
              f"(Minuten und Grund, oder 'pünktlich').")
    task = build_task(f"info_zugsuche_status__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "INFO+SUCHE: Zug ohne Zugnummer finden, Standort + Verspätung berichten",
                      injections, None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["zuege_suchen", "zugstandort", "verspaetung"],
           "oracle_calls": [oc("zuege_suchen", von=von, nach=nach, produkt=trip.product),
                            oc("zugstandort", zugnummer=trip.zugnummer),
                            oc("verspaetung", zugnummer=trip.zugnummer)],
           "facts": {"zugnummer": trip.zugnummer, "standort": s, "verspaetung": v}}
    return task, key


def t_info_verspaetung_suche(g: Gen, trip, idx, inject: bool):
    """Always injected: which trains from station X are >=30 min late right now?"""
    r = rng(g.seed, "inj", "versp_suche", trip.trip_id)
    injections = [inj_call("inject_verspaetung", zugnummer=trip.zugnummer,
                           minuten=r.choice([35, 50, 70, 90]), grund=r.choice(DELAY_CAUSES))]
    tk = _apply(g.fresh(), injections)
    von = tk._station_name(trip.origin_station)
    res = tk.zuege_suchen(von=von, min_verspaetung_minuten=30)
    if not (1 <= res["treffer_gesamt"] <= 3):
        return None
    hits = res["treffer"]
    if not any(row["zugnummer"] == trip.zugnummer for row in hits):
        return None
    comm, oracle, gruende = [], [oc("zuege_suchen", von=von, min_verspaetung_minuten=30)], {}
    for row in hits:
        v = tk.verspaetung(row["zugnummer"])
        comm.append(row["zugnummer"])
        gruende[row["zugnummer"]] = v["grund"]
        oracle.append(oc("verspaetung", zugnummer=row["zugnummer"]))
    comm += sorted(set(gruende.values()))
    ticket = (f"Gibt es aktuell Züge ab {von} mit mindestens 30 Minuten Verspätung? "
              f"Nenne jede betroffene Zugnummer und jeweils den Verspätungsgrund.")
    task = build_task(f"info_verspaetung_suche__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "INFO+SUCHE: verspätete Züge ab Bahnhof finden (fault-injected)",
                      injections, None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["zuege_suchen", "verspaetung"],
           "oracle_calls": oracle,
           "facts": {"von": von, "betroffen": gruende}}
    return task, key


def t_info_mitarbeiter_suche(g: Gen, combo: MACombo, idx, inject: bool):
    tk = g.fresh()
    res = tk.mitarbeiter_suchen(rolle=combo.rolle, heimatbasis=combo.heimatbasis,
                                qualifikation=combo.qualifikation, verfuegbar_um=combo.verfuegbar_um)
    hit_ids = [row["mitarbeiter_id"] for row in res["treffer"]]
    if not (1 <= len(hit_ids) <= 3) or set(hit_ids) != set(combo.emp_ids):
        return None
    ticket = (f"Welche Mitarbeiter mit Rolle {combo.rolle} und Qualifikation {combo.qualifikation} "
              f"an der Heimatbasis {combo.heimatbasis} sind um {combo.verfuegbar_um} laut Schicht im "
              f"Dienst? Nenne die Mitarbeiter-IDs.")
    task = build_task(f"info_mitarbeiter_suche__{sid(combo.key)}__{idx:03d}", ticket,
                      "INFO+SUCHE: Mitarbeiter nach Rolle/Qualifikation/Basis/Schicht finden",
                      [], None, None, list(hit_ids), ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["mitarbeiter_suchen"],
           "oracle_calls": [oc("mitarbeiter_suchen", rolle=combo.rolle, heimatbasis=combo.heimatbasis,
                               qualifikation=combo.qualifikation, verfuegbar_um=combo.verfuegbar_um)],
           "facts": {"emp_ids": hit_ids}}
    return task, key


def t_info_schichtcheck(g: Gen, trip, idx, inject: bool):
    """Departure time must come from fahrplan (not the ticket) -> forces the 2-tool chain."""
    r = rng(g.seed, "combo", "schichtcheck", trip.trip_id)
    tk = g.fresh()
    bases = sorted({e.home_base for e in g.master.employees.values()})
    combos = [(b, q) for b in bases for q in QUALS]
    r.shuffle(combos)
    chosen = None
    for b, q in combos[:25]:
        res = tk.mitarbeiter_suchen(rolle="Lokführer", heimatbasis=b, qualifikation=q,
                                    verfuegbar_um=trip.dep_time)
        if 1 <= res["treffer_gesamt"] <= 3:
            chosen = (b, q, [row["mitarbeiter_id"] for row in res["treffer"]])
            break
    if chosen is None:
        return None
    b, q, emp_ids = chosen
    ticket = (f"Für {trip.zugnummer} werden kurzfristig Lokführer gesucht: Welche Lokführer mit "
              f"Qualifikation {q} an der Heimatbasis {b} sind zur Abfahrtszeit des Zuges laut Schicht "
              f"im Dienst? Ermittle zuerst die Abfahrtszeit über den Fahrplan und nenne dann die "
              f"Mitarbeiter-IDs.")
    task = build_task(f"info_schichtcheck__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "INFO+SUCHE: Schichtabdeckung zur Abfahrtszeit prüfen (Fahrplan -> Suche)",
                      [], None, None, list(emp_ids), ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["fahrplan", "mitarbeiter_suchen"],
           "oracle_calls": [oc("fahrplan", zugnummer=trip.zugnummer),
                            oc("mitarbeiter_suchen", rolle="Lokführer", heimatbasis=b,
                               qualifikation=q, verfuegbar_um=trip.dep_time)],
           "facts": {"abfahrt": trip.dep_time, "basis": b, "qualifikation": q, "emp_ids": emp_ids}}
    return task, key


def t_info_wartung_depot(g: Gen, combo: WOCombo, idx, inject: bool):
    tk = g.fresh()
    res = tk.wartung_liste(status=combo.status, depot=combo.depot, faellig_vor=combo.faellig_vor)
    rows = res["treffer"]
    if not (1 <= len(rows) <= 3) or {r_["order_id"] for r_ in rows} != set(combo.order_ids):
        return None
    comm = [r_["order_id"] for r_ in rows] + sorted({r_["due_at"][:10] for r_ in rows})
    ticket = (f"Welche Wartungsaufträge mit Status '{combo.status}' im Depot {combo.depot} sind vor dem "
              f"{combo.faellig_vor} fällig? Nenne je Auftrag die Auftrags-ID und das Fälligkeitsdatum.")
    task = build_task(f"info_wartung_depot__{sid(combo.key)}__{idx:03d}", ticket,
                      "INFO+SUCHE: Wartungsaufträge flottenweit filtern (Status/Depot/Fälligkeit)",
                      [], None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["wartung_liste"],
           "oracle_calls": [oc("wartung_liste", status=combo.status, depot=combo.depot,
                               faellig_vor=combo.faellig_vor)],
           "facts": {"orders": rows}}
    return task, key


def t_info_zug_komplett(g: Gen, trip, idx, inject: bool):
    r = rng(g.seed, "inj", "zug_komplett", trip.trip_id)
    injections = []
    if inject:
        injections = [inj_call("inject_verspaetung", zugnummer=trip.zugnummer,
                               minuten=r.choice([25, 40, 55]), grund=r.choice(DELAY_CAUSES))]
    tk = _apply(g.fresh(), injections)
    s = tk.zugstandort(trip.zugnummer)
    if s.get("status") != "unterwegs" or not s.get("naechster_halt"):
        return None
    w = tk.wartung_status(trip.zugnummer)
    offen = [o for o in w["wartungsauftraege"] if o["status"] != "abgeschlossen"]
    m = tk.mitarbeiter_info(trip.zugnummer)
    lokf = [c for c in m["besatzung"] if c["rolle"] == "Lokführer"]
    if not offen or not lokf:
        return None
    v = tk.verspaetung(trip.zugnummer)
    comm = [s["naechster_halt"], offen[0]["order_id"], lokf[0]["mitarbeiter_id"]] + _grund_comm(v)
    ticket = (f"Erstelle einen kurzen Statusbericht zu {trip.zugnummer}: aktuelle Verspätung (Minuten "
              f"und Grund, oder 'pünktlich'), nächster Halt, offene Wartungsaufträge des Fahrzeugs "
              f"(Auftrags-IDs) und der eingeteilte Lokführer (Mitarbeiter-ID).")
    task = build_task(f"info_zug_komplett__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "INFO: voller Statusbericht (4 Tools)", injections, None, None,
                      comm, ["COMMUNICATE"])
    key = {"kind": "info",
           "expected_tools": ["verspaetung", "zugstandort", "wartung_status", "mitarbeiter_info"],
           "oracle_calls": [oc("verspaetung", zugnummer=trip.zugnummer),
                            oc("zugstandort", zugnummer=trip.zugnummer),
                            oc("wartung_status", kennung=trip.zugnummer),
                            oc("mitarbeiter_info", zugnummer=trip.zugnummer)],
           "facts": {"verspaetung": v, "naechster_halt": s["naechster_halt"],
                     "offene_auftraege": [o["order_id"] for o in offen],
                     "lokfuehrer": lokf[0]["mitarbeiter_id"]}}
    return task, key


def t_info_ankunft_suche(g: Gen, trip, idx, inject: bool):
    r = rng(g.seed, "inj", "ankunft_suche", trip.trip_id)
    injections = []
    if inject:
        injections = [inj_call("inject_verspaetung", zugnummer=trip.zugnummer,
                               minuten=r.choice([30, 45, 60]), grund=r.choice(DELAY_CAUSES))]
    tk = _apply(g.fresh(), injections)
    von = tk._station_name(trip.origin_station)
    nach = tk._station_name(trip.dest_station)
    if not _route_search_ok(tk, trip, von, nach):
        return None
    fp = tk.fahrplan(trip.zugnummer)
    v = tk.verspaetung(trip.zugnummer)
    comm = [trip.zugnummer, fp["ankunft"]] + _grund_comm(v)
    ticket = (f"Ein {trip.product} von {von} nach {nach} mit Abfahrt {trip.dep_time}: Finde den Zug über "
              f"die Zugsuche und prüfe, ob er voraussichtlich pünktlich ankommt. Nenne ausdrücklich die "
              f"Zugnummer, die PLANMÄSSIGE Ankunftszeit in {nach} und die aktuelle Verspätung "
              f"(Minuten und Grund, oder 'pünktlich').")
    task = build_task(f"info_ankunft_suche__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "INFO+SUCHE: Ankunftsprognose ohne vorgegebene Zugnummer", injections,
                      None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["zuege_suchen", "fahrplan", "verspaetung"],
           "oracle_calls": [oc("zuege_suchen", von=von, nach=nach, produkt=trip.product),
                            oc("fahrplan", zugnummer=trip.zugnummer),
                            oc("verspaetung", zugnummer=trip.zugnummer)],
           "facts": {"zugnummer": trip.zugnummer, "ankunft_plan": fp["ankunft"], "verspaetung": v}}
    return task, key


def t_action_ersatz_quali(g: Gen, trip, idx, inject: bool):
    """Always state+runtime: driver drops out, dispo proposes an UNQUALIFIED replacement (wrong role
    or missing product qualification) -> rejection -> search -> assign first hit."""
    r = rng(g.seed, "act", "ersatz_quali", trip.trip_id)
    tk0 = BahnTools(g.master)
    orig = [c for c in tk0.mitarbeiter_info(trip.zugnummer)["besatzung"] if c["rolle"] == "Lokführer"]
    if not orig:
        return None
    orig_id = orig[0]["mitarbeiter_id"]
    qual = qual_for(trip)
    emps = g.master.employees
    assigned = {a.emp_id for a in g.master.assignments if a.trip_id == trip.trip_id}
    flavor = "rolle" if (qual is None or r.random() < 0.5) else "qual"
    if flavor == "rolle":
        cands_bad = sorted(eid for eid, e in emps.items()
                           if e.role == "Zugbegleiter" and eid not in assigned)
    else:
        cands_bad = sorted(eid for eid, e in emps.items()
                           if e.role == "Lokführer" and qual not in e.qualifications
                           and eid not in assigned)
    if not cands_bad:
        return None
    proposed = emps[r.choice(cands_bad)]
    base = emps[orig_id].home_base if orig_id in emps else None
    if base is None:
        return None
    # first hit under the ticket's exact search filters (mirrors mitarbeiter_suchen's emp_id sort)
    cand_valid = sorted(eid for eid, e in emps.items()
                        if e.role == "Lokführer" and e.home_base == base
                        and (qual is None or qual in e.qualifications))
    if not cand_valid:
        return None
    first = cand_valid[0]
    assigned_after = {a.emp_id for a in g.master.assignments
                      if a.trip_id == trip.trip_id and a.role != "Lokführer"}
    if first in (orig_id, proposed.emp_id) or first in assigned_after:
        return None
    ersatz = emps[first]
    injections = [inj_call("inject_lokfuehrer_ausfall", zugnummer=trip.zugnummer)]
    qual_txt = f" mit Qualifikation {qual}" if qual else ""
    such_args = {"rolle": "Lokführer", "heimatbasis": base}
    if qual:
        such_args["qualifikation"] = qual
    ticket = (f"Der eingeteilte Lokführer von {trip.zugnummer} ist kurzfristig ausgefallen. Die "
              f"Disposition schlägt {proposed.name} ({proposed.emp_id}) als Ersatz vor — prüfe die "
              f"Besatzung und teile ihn als Lokführer zu. Falls die Zuweisung abgelehnt wird: suche "
              f"per Mitarbeitersuche einen Lokführer{qual_txt} an der Heimatbasis {base} und teile den "
              f"ersten Treffer (kleinste Mitarbeiter-ID) als Lokführer zu.")
    refs = [ref_action(1, "crew_zuweisen", zugnummer=trip.zugnummer,
                       mitarbeiter_id=ersatz.emp_id, rolle="Lokführer")]
    asserts = [
        env_assert("assert_crew_assigned", zugnummer=trip.zugnummer, mitarbeiter_id=ersatz.emp_id),
        env_assert("assert_crew_assigned", assert_value=False,
                   zugnummer=trip.zugnummer, mitarbeiter_id=proposed.emp_id),
        env_assert("assert_crew_assigned", assert_value=False,
                   zugnummer=trip.zugnummer, mitarbeiter_id=orig_id),
    ]
    task = build_task(f"action_ersatz_quali__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION+REPLAN: unqualifizierter Ersatzvorschlag wird abgelehnt -> suchen -> zuteilen",
                      injections, refs, asserts, [ersatz.emp_id], ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "fault": "state+runtime",
           "expected_tools": ["mitarbeiter_info", "mitarbeiter_suchen", "crew_zuweisen"],
           "oracle_calls": [oc("mitarbeiter_info", zugnummer=trip.zugnummer),
                            oc("mitarbeiter_suchen", **such_args),
                            oc("crew_zuweisen", zugnummer=trip.zugnummer,
                               mitarbeiter_id=ersatz.emp_id, rolle="Lokführer")],
           "facts": {"vorschlag_ungueltig": proposed.emp_id, "flavor": flavor,
                     "ersatz_id": ersatz.emp_id, "ausgefallen": orig_id, "basis": base}}
    return task, key


def t_action_crew_doppelt(g: Gen, trip, idx, inject: bool):
    """Runtime fault baked into the ticket: the proposed employee is already on the crew."""
    r = rng(g.seed, "act", "crew_doppelt", trip.trip_id)
    tk0 = BahnTools(g.master)
    crew = sorted(tk0.mitarbeiter_info(trip.zugnummer)["besatzung"],
                  key=lambda c: c["mitarbeiter_id"])
    if not crew:
        return None
    dup = crew[0]
    emp_b = g.spare_mitarbeiter(trip.trip_id, r, "Zugbegleiter")
    if emp_b is None:
        return None
    ticket = (f"Teile {dup['name']} ({dup['mitarbeiter_id']}) dem Zug {trip.zugnummer} als "
              f"Zugbegleiter zu. Falls die Zuweisung abgelehnt wird (bereits eingeteilt), teile "
              f"stattdessen {emp_b.name} ({emp_b.emp_id}) als Zugbegleiter zu und bestätige die Zuteilung.")
    refs = [ref_action(1, "crew_zuweisen", zugnummer=trip.zugnummer,
                       mitarbeiter_id=emp_b.emp_id, rolle="Zugbegleiter")]
    asserts = [env_assert("assert_crew_assigned", zugnummer=trip.zugnummer, mitarbeiter_id=emp_b.emp_id)]
    task = build_task(f"action_crew_doppelt__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION+REPLAN: Doppel-Zuteilung wird abgelehnt -> Alternative zuteilen",
                      [], refs, asserts, [emp_b.emp_id], ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "fault": "runtime",
           "expected_tools": ["crew_zuweisen"],
           "oracle_calls": [oc("crew_zuweisen", zugnummer=trip.zugnummer,
                               mitarbeiter_id=emp_b.emp_id, rolle="Zugbegleiter")],
           "facts": {"bereits_eingeteilt": dup["mitarbeiter_id"], "alternative": emp_b.emp_id}}
    return task, key


def t_action_verstaerkung(g: Gen, trip, idx, inject: bool):
    """Search -> assign first hit -> confirm with the crew list (3-tool chain, no fault)."""
    r = rng(g.seed, "act", "verstaerkung", trip.trip_id)
    emps = g.master.employees
    assigned = {a.emp_id for a in g.master.assignments if a.trip_id == trip.trip_id}
    bases = sorted({e.home_base for e in emps.values()})
    r.shuffle(bases)
    chosen = None
    for base in bases[:15]:
        cands = sorted(eid for eid, e in emps.items()
                       if e.role == "Zugbegleiter" and e.home_base == base)
        if cands and cands[0] not in assigned:
            chosen = (base, cands[0])
            break
    if chosen is None:
        return None
    base, first = chosen
    ticket = (f"{trip.zugnummer} braucht kurzfristig einen zusätzlichen Zugbegleiter. Suche per "
              f"Mitarbeitersuche Zugbegleiter an der Heimatbasis {base}, teile den ersten Treffer "
              f"(kleinste Mitarbeiter-ID) dem Zug als Zugbegleiter zu und bestätige anschließend über "
              f"die Besatzungsliste, dass die Zuteilung erfolgt ist.")
    refs = [ref_action(1, "crew_zuweisen", zugnummer=trip.zugnummer,
                       mitarbeiter_id=first, rolle="Zugbegleiter")]
    asserts = [env_assert("assert_crew_assigned", zugnummer=trip.zugnummer, mitarbeiter_id=first)]
    task = build_task(f"action_verstaerkung__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION+SUCHE: Verstärkung suchen, ersten Treffer zuteilen, bestätigen",
                      [], refs, asserts, [first], ["DB", "ENV_ASSERTION"])
    key = {"kind": "action",
           "expected_tools": ["mitarbeiter_suchen", "crew_zuweisen", "mitarbeiter_info"],
           "oracle_calls": [oc("mitarbeiter_suchen", rolle="Zugbegleiter", heimatbasis=base),
                            oc("crew_zuweisen", zugnummer=trip.zugnummer,
                               mitarbeiter_id=first, rolle="Zugbegleiter"),
                            oc("mitarbeiter_info", zugnummer=trip.zugnummer)],
           "facts": {"basis": base, "emp_id": first}}
    return task, key


def t_action_wartung_suche(g: Gen, trip, idx, inject: bool):
    """Always injected: a route-described train reports a technical fault -> find it, look up the
    vehicle, schedule a repair."""
    r = rng(g.seed, "act", "wartung_suche", trip.trip_id)
    grund = "technische Störung am Zug"
    injections = [inj_call("inject_verspaetung", zugnummer=trip.zugnummer,
                           minuten=r.choice([20, 30, 40]), grund=grund)]
    tk = _apply(g.fresh(), injections)
    von = tk._station_name(trip.origin_station)
    nach = tk._station_name(trip.dest_station)
    if not _route_search_ok(tk, trip, von, nach):
        return None
    due = f"2026-07-{r.randint(4, 10):02d} {r.choice(['06:00', '08:00', '22:00'])}"
    ticket = (f"Ein {trip.product} von {von} nach {nach} (Abfahrt {trip.dep_time}) meldet eine "
              f"technische Störung am Zug. Finde den Zug über die Zugsuche, ermittle über den "
              f"Wartungsstatus die Fahrzeug-ID und plane eine Wartung vom Typ 'Reparatur' ein, "
              f"fällig am {due}. Nenne die Fahrzeug-ID.")
    refs = [ref_action(1, "wartung_einplanen", fahrzeug_id=trip.vehicle_id,
                       typ="Reparatur", faellig_am=due)]
    asserts = [env_assert("assert_maintenance_exists", fahrzeug_id=trip.vehicle_id, typ="Reparatur")]
    task = build_task(f"action_wartung_suche__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION+SUCHE: gestörten Zug finden, Fahrzeug ermitteln, Reparatur einplanen",
                      injections, refs, asserts, [trip.vehicle_id], ["DB", "ENV_ASSERTION"])
    key = {"kind": "action",
           "expected_tools": ["zuege_suchen", "wartung_status", "wartung_einplanen"],
           "oracle_calls": [oc("zuege_suchen", von=von, nach=nach, produkt=trip.product),
                            oc("wartung_status", kennung=trip.zugnummer),
                            oc("wartung_einplanen", fahrzeug_id=trip.vehicle_id,
                               typ="Reparatur", faellig_am=due)],
           "facts": {"zugnummer": trip.zugnummer, "fahrzeug_id": trip.vehicle_id, "faellig_am": due}}
    return task, key


def t_action_inspektion_bedingt(g: Gen, trip, idx, inject: bool):
    """Conditional write: schedule an inspection ONLY at >=30 min delay; otherwise do nothing.
    The clean branch asserts the absence of the write (assert_value=False)."""
    r = rng(g.seed, "act", "inspektion", trip.trip_id)
    if any(o.type == "Inspektion" and o.vehicle_id == trip.vehicle_id
           for o in g.master.maintenance_orders.values()):
        return None  # a pre-existing Inspektion order would make the no-write assert unsatisfiable
    injections = []
    if inject:
        injections = [inj_call("inject_verspaetung", zugnummer=trip.zugnummer,
                               minuten=r.choice([35, 45, 60]), grund=r.choice(DELAY_CAUSES))]
    tk = _apply(g.fresh(), injections)
    v = tk.verspaetung(trip.zugnummer)
    due = f"2026-07-{r.randint(4, 10):02d} {r.choice(['06:00', '08:00', '22:00'])}"
    ticket = (f"Prüfe die aktuelle Verspätung von {trip.zugnummer}: Hat der Zug 30 Minuten Verspätung "
              f"oder mehr, ermittle über den Wartungsstatus die Fahrzeug-ID und plane eine Wartung vom "
              f"Typ 'Inspektion' ein (fällig am {due}); nenne dann die Fahrzeug-ID. Andernfalls plane "
              f"nichts ein und antworte mit 'keine Inspektion nötig'.")
    if inject:
        refs = [ref_action(1, "wartung_einplanen", fahrzeug_id=trip.vehicle_id,
                           typ="Inspektion", faellig_am=due)]
        asserts = [env_assert("assert_maintenance_exists", fahrzeug_id=trip.vehicle_id, typ="Inspektion")]
        comm = [trip.vehicle_id]
        expected = ["verspaetung", "wartung_status", "wartung_einplanen"]
        oracle = [oc("verspaetung", zugnummer=trip.zugnummer),
                  oc("wartung_status", kennung=trip.zugnummer),
                  oc("wartung_einplanen", fahrzeug_id=trip.vehicle_id, typ="Inspektion", faellig_am=due)]
    else:
        if v["verspaetung_minuten"] >= 30:
            return None  # natural heavy delay -> belongs to the injected branch, skip
        refs = []
        asserts = [env_assert("assert_maintenance_exists", assert_value=False,
                              fahrzeug_id=trip.vehicle_id, typ="Inspektion")]
        comm = ["keine Inspektion nötig"]
        expected = ["verspaetung"]
        oracle = [oc("verspaetung", zugnummer=trip.zugnummer)]
    task = build_task(f"action_inspektion_bedingt__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION bedingt: Inspektion nur bei >=30 Min Verspätung einplanen",
                      injections, refs, asserts, comm, ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "expected_tools": expected, "oracle_calls": oracle,
           "facts": {"verspaetung_min": v["verspaetung_minuten"], "fahrzeug_id": trip.vehicle_id,
                     "schwelle": 30}}
    return task, key


def t_action_ueberfaellig(g: Gen, trip, idx, inject: bool):
    """Selection task: the vehicle has EXACTLY ONE überfällig order among several -> find + start it."""
    tk = g.fresh()
    w = tk.wartung_status(trip.zugnummer)
    over = [o for o in w["wartungsauftraege"] if o["status"] == "überfällig"]
    if len(over) != 1 or len(w["wartungsauftraege"]) < 2:
        return None
    order = over[0]
    ticket = (f"Das Fahrzeug von {trip.zugnummer} hat genau einen überfälligen Wartungsauftrag. Finde "
              f"ihn über den Wartungsstatus und setze ihn auf 'in_Arbeit'. Nenne die Auftrags-ID.")
    refs = [ref_action(1, "wartung_status_setzen", auftrag_id=order["order_id"], status="in_Arbeit")]
    asserts = [env_assert("assert_maintenance_status", auftrag_id=order["order_id"], status="in_Arbeit")]
    task = build_task(f"action_ueberfaellig__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION: den einen überfälligen Auftrag finden und starten",
                      [], refs, asserts, [order["order_id"]], ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "expected_tools": ["wartung_status", "wartung_status_setzen"],
           "oracle_calls": [oc("wartung_status", kennung=trip.zugnummer),
                            oc("wartung_status_setzen", auftrag_id=order["order_id"], status="in_Arbeit")],
           "facts": {"auftrag_id": order["order_id"], "fahrzeug_id": w["fahrzeug_id"]}}
    return task, key


def t_action_wstatus_konflikt(g: Gen, trip, idx, inject: bool):
    """Runtime fault: the named order is abgeschlossen (terminal) -> rejection -> the agent must
    switch to the vehicle's single open order instead."""
    tk = g.fresh()
    w = tk.wartung_status(trip.zugnummer)
    closed = sorted((o for o in w["wartungsauftraege"] if o["status"] == "abgeschlossen"),
                    key=lambda o: o["order_id"])
    open_ = [o for o in w["wartungsauftraege"] if o["status"] in ("geplant", "überfällig")]
    if not closed or len(open_) != 1:
        return None
    ticket = (f"Setze den Wartungsauftrag {closed[0]['order_id']} (Fahrzeug von {trip.zugnummer}) auf "
              f"'in_Arbeit'. Falls der Statuswechsel abgelehnt wird (Endstatus), setze stattdessen den "
              f"offenen Wartungsauftrag des Fahrzeugs auf 'in_Arbeit' und nenne dessen Auftrags-ID.")
    refs = [ref_action(1, "wartung_status_setzen", auftrag_id=open_[0]["order_id"], status="in_Arbeit")]
    asserts = [
        env_assert("assert_maintenance_status", auftrag_id=open_[0]["order_id"], status="in_Arbeit"),
        env_assert("assert_maintenance_status", auftrag_id=closed[0]["order_id"], status="abgeschlossen"),
    ]
    task = build_task(f"action_wstatus_konflikt__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION+REPLAN: Endstatus-Ablehnung -> offenen Auftrag stattdessen starten",
                      [], refs, asserts, [open_[0]["order_id"]], ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "fault": "runtime",
           "expected_tools": ["wartung_status", "wartung_status_setzen"],
           "oracle_calls": [oc("wartung_status", kennung=trip.zugnummer),
                            oc("wartung_status_setzen", auftrag_id=open_[0]["order_id"],
                               status="in_Arbeit")],
           "facts": {"abgeschlossen": closed[0]["order_id"], "offen": open_[0]["order_id"]}}
    return task, key


def t_action_wartung_batch(g: Gen, trip, idx, inject: bool):
    """Multi-write: start BOTH open orders of the vehicle."""
    tk = g.fresh()
    w = tk.wartung_status(trip.zugnummer)
    open2 = sorted((o for o in w["wartungsauftraege"] if o["status"] in ("geplant", "überfällig")),
                   key=lambda o: o["order_id"])
    if len(open2) != 2:
        return None
    o1, o2 = open2
    ticket = (f"Der Werkstatttermin für das Fahrzeug von {trip.zugnummer} wurde vorgezogen: Setze BEIDE "
              f"offenen Wartungsaufträge (Status 'geplant' oder 'überfällig') des Fahrzeugs auf "
              f"'in_Arbeit' und nenne beide Auftrags-IDs.")
    refs = [ref_action(1, "wartung_status_setzen", auftrag_id=o1["order_id"], status="in_Arbeit"),
            ref_action(2, "wartung_status_setzen", auftrag_id=o2["order_id"], status="in_Arbeit")]
    asserts = [env_assert("assert_maintenance_status", auftrag_id=o1["order_id"], status="in_Arbeit"),
               env_assert("assert_maintenance_status", auftrag_id=o2["order_id"], status="in_Arbeit")]
    task = build_task(f"action_wartung_batch__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION: beide offenen Aufträge starten (Multi-Write)",
                      [], refs, asserts, [o1["order_id"], o2["order_id"]], ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "expected_tools": ["wartung_status", "wartung_status_setzen"],
           "oracle_calls": [oc("wartung_status", kennung=trip.zugnummer),
                            oc("wartung_status_setzen", auftrag_id=o1["order_id"], status="in_Arbeit"),
                            oc("wartung_status_setzen", auftrag_id=o2["order_id"], status="in_Arbeit")],
           "facts": {"auftraege": [o1["order_id"], o2["order_id"]]}}
    return task, key


def t_action_gefahrgut(g: Gen, trip, idx, inject: bool):
    """Qualification-driven search: additional Gefahrgut-qualified Lokführer, first hit assigned.
    Pool restricts to products WITHOUT a product-qualification gate (search filters ONE qual)."""
    r = rng(g.seed, "act", "gefahrgut", trip.trip_id)
    emps = g.master.employees
    assigned = {a.emp_id for a in g.master.assignments if a.trip_id == trip.trip_id}
    bases = sorted({e.home_base for e in emps.values()})
    r.shuffle(bases)
    chosen = None
    for base in bases[:15]:
        cands = sorted(eid for eid, e in emps.items()
                       if e.role == "Lokführer" and e.home_base == base
                       and "Gefahrgut" in e.qualifications)
        if cands and cands[0] not in assigned:
            chosen = (base, cands[0])
            break
    if chosen is None:
        return None
    base, first = chosen
    ticket = (f"Für einen Gefahrgut-Sondertransport auf {trip.zugnummer} wird ein zusätzlicher "
              f"Lokführer mit Qualifikation Gefahrgut benötigt. Suche per Mitarbeitersuche Lokführer "
              f"mit Qualifikation Gefahrgut an der Heimatbasis {base} und teile den ersten Treffer "
              f"(kleinste Mitarbeiter-ID) dem Zug als Lokführer zu.")
    refs = [ref_action(1, "crew_zuweisen", zugnummer=trip.zugnummer,
                       mitarbeiter_id=first, rolle="Lokführer")]
    asserts = [env_assert("assert_crew_assigned", zugnummer=trip.zugnummer, mitarbeiter_id=first)]
    task = build_task(f"action_gefahrgut__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION+SUCHE: Gefahrgut-qualifizierten Lokführer finden und zuteilen",
                      [], refs, asserts, [first], ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "expected_tools": ["mitarbeiter_suchen", "crew_zuweisen"],
           "oracle_calls": [oc("mitarbeiter_suchen", rolle="Lokführer",
                               qualifikation="Gefahrgut", heimatbasis=base),
                            oc("crew_zuweisen", zugnummer=trip.zugnummer,
                               mitarbeiter_id=first, rolle="Lokführer")],
           "facts": {"basis": base, "emp_id": first}}
    return task, key


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
    # easy/mid tier (polished wave-1 shapes)
    Spec(t_info_verspaetung, "trips", 40, True),
    Spec(t_info_standort, "trips_pos", 40, False),
    Spec(t_info_ankunft, "trips_pos", 80, True),
    Spec(t_info_wartung, "trips_orders", 40, False),
    Spec(t_info_crew, "trips_lokf", 40, False),
    Spec(t_action_wartung, "trips", 60, False),
    Spec(t_action_crew, "trips_lokf", 60, False),
    Spec(t_action_wartung_status, "trips_orders", 40, False),
    Spec(t_action_ersatz, "trips_lokf", 60, None),
    # hard tier (wave 2) — the 3-4-call templates carry the >=50% multi-tool target,
    # so their n sits above the 1-2-call ones (gate-d calibration, 2026-07-08)
    Spec(t_info_zugsuche_status, "trips_pos_route", 140, True),
    Spec(t_info_verspaetung_suche, "trips", 100, None),
    Spec(t_info_mitarbeiter_suche, "ma_filter_combos", 100, False),
    Spec(t_info_schichtcheck, "trips_lokf", 100, False),
    Spec(t_info_wartung_depot, "wartung_filter_combos", 100, False),
    Spec(t_info_zug_komplett, "trips_komplett", 140, True),
    Spec(t_info_ankunft_suche, "trips_route_page", 140, True),
    Spec(t_action_ersatz_quali, "trips_lokf", 140, None),
    Spec(t_action_crew_doppelt, "trips_lokf", 100, False),
    Spec(t_action_verstaerkung, "trips", 140, False),
    Spec(t_action_wartung_suche, "trips_route_page", 140, None),
    Spec(t_action_inspektion_bedingt, "trips", 100, True, 0.5),
    Spec(t_action_ueberfaellig, "trips_veh_ueberfaellig", 100, False),
    Spec(t_action_wstatus_konflikt, "trips_veh_konflikt", 100, False),
    Spec(t_action_wartung_batch, "trips_veh_2open", 100, False),
    Spec(t_action_gefahrgut, "trips_lokf_no_quali_produkt", 100, False),
]


def main():
    ap = argparse.ArgumentParser(description="Generate db_bahn tasks with built-in answer keys (wave 2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fault-rate", type=float, default=0.35)
    ap.add_argument("--n-bakeoff", type=int, default=25)
    ap.add_argument("--n-heldout", type=int, default=60)
    ap.add_argument("--n-rl", type=int, default=300,
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
            entity = getattr(item, "zugnummer", None) or item.key
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

    total = len(tasks)
    hel_frac, rl_frac = args.n_heldout / total, args.n_rl / total  # targets are approximate
    heldout, rl, sft = [], [], []
    for tpl in sorted(by_tpl):
        ids = by_tpl[tpl]
        n_hel = max(1, round(len(ids) * hel_frac))   # >=1 per template in heldout
        n_rl = max(2, round(len(ids) * rl_frac))     # >=2 per template in rl
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
