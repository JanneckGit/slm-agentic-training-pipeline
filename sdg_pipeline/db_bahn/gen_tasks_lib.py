"""
sdg_pipeline/db_bahn/gen_tasks_lib.py
=====================================
Shared infrastructure for the task generator (split out of gen_tasks.py 2026-07-15,
output verified byte-identical): vocab constants, the Gen eligibility-pool builder and
the task/answer-key building blocks used by every template in gen_templates_easy/hard.py.
"""

import random
import re
from collections import Counter, namedtuple
from pathlib import Path
from typing import Optional

from sdg_pipeline.db_bahn.seed_worldstate import rng
from sdg_pipeline.db_bahn.tau2_domain.data_model import BahnDB
from sdg_pipeline.db_bahn.tau2_domain.tools import BahnTools, MAX_TREFFER, QUALI_PRODUKTE

# DELIBERATE SUBSETS of seed_worldstate.py's world vocab (NOT the same lists): these drive what
# tasks INJECT/TARGET, the seed lists drive what exists in the frozen world. seed's extra entries
# ("Reinigung" maintenance, passenger-flow delay causes) exist in db.json but are never task goals.
# Changing any entry changes the generated task pool (seeded RNG) -> golden hashes break.
DELAY_CAUSES = ["Signalstörung", "Bauarbeiten", "technische Störung am Zug", "Notarzteinsatz"]
MAINT_TYPES = ["Inspektion", "Reparatur", "Radsatztausch", "Softwareupdate"]
QUALS = ["ICE", "IC", "EC", "Nacht", "Gefahrgut"]

MACombo = namedtuple("MACombo", "key rolle qualifikation heimatbasis verfuegbar_um emp_ids")
WOCombo = namedtuple("WOCombo", "key status depot faellig_vor schweregrad order_ids")  # schweregrad: None = ohne Filter
# wave-3 pool items (all carry .key for the (template, entity) dedup in gen_tasks.py)
TripBatch = namedtuple("TripBatch", "key trips")                       # E1/E4: 3 trips per ticket
StationGroup = namedtuple("StationGroup", "key richtung station zugnummern")  # E3: 3-8 zuege_suchen hits
AnschlussPair = namedtuple("AnschlussPair", "key zub ans station_id arr dep puffer")  # E10
VehPair = namedtuple("VehPair", "key vehicle_id trip_a trip_b")        # E7: shared vehicle
NameGroup = namedtuple("NameGroup", "key name emp_ids target_emp_id target_base")     # E12b: ambiguous name
MixPair = namedtuple("MixPair", "key same trip_a trip_b vid_a vid_b")  # T3 (wave 3.5): same/diff vehicle


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
        if kind == "employees":  # wave-2.5: lookup-by-ID template (A1) draws directly from staff
            return sorted(self.master.employees.values(), key=lambda e: e.emp_id)
        # ------------------------------ wave-3 pools ---------------------------------------
        if kind == "trips_batch3":
            # E1/E4: deterministic disjoint 3-trip groups for batch/aggregation tickets
            trips = list(self.pool("trips"))
            order = rng(self.seed, "pool_batch3").sample(range(len(trips)), len(trips))
            return [TripBatch("+".join(sid(trips[j].zugnummer) for j in order[i:i + 3]),
                              tuple(trips[j] for j in order[i:i + 3]))
                    for i in range(0, len(order) - 2, 3)]
        if kind == "station_iter_groups":
            # E3: stations whose full-name von-/nach-filter yields 3-10 zuege_suchen hits.
            # (window widened 3-8 -> 3-10 after the S5 corridor result: iteration depth is the
            # hardest base-4B lever at 6% base yield — harvest every eligible station)
            # Hit counting mirrors the tool exactly (substring on the STATION NAME); both
            # directions are separate pool items (richtung "von" | "nach").
            name_of = {sid_: s.name for sid_, s in self.master.stations.items()}
            out = []
            for st_id in sorted(self.master.stations):
                lo = name_of[st_id].lower()
                for richtung, attr in (("von", "origin_station"), ("nach", "dest_station")):
                    hits = [t for t in self.pool("trips")
                            if lo in name_of.get(getattr(t, attr), getattr(t, attr)).lower()]
                    if 3 <= len(hits) <= 10:
                        hits.sort(key=lambda t: (t.dep_time, t.zugnummer))  # tool result order
                        out.append(StationGroup(f"{richtung}-{sid(name_of[st_id])}", richtung,
                                                name_of[st_id],
                                                tuple(t.zugnummer for t in hits)))
            return out
        if kind == "wo_combos_batch":
            return self._wo_combos_batch()
        if kind == "ma_refine_combos":
            # E8: base combos whose BROAD (rolle, qualifikation) search overflows the result page
            # (>MAX_TREFFER incl. hinweis line) while the refined 4-filter combo lands at 1-3 hits
            broad = Counter()
            for e in self.master.employees.values():
                for q in e.qualifications:
                    broad[(e.role, q)] += 1
            return [c for c in self._ma_combos() if broad[(c.rolle, c.qualifikation)] > MAX_TREFFER]
        if kind == "wo_refine_combos":
            # E8: combos whose BROAD (status) search overflows the page; refined combo 1-3 hits
            status_count = Counter(o.status for o in self.master.maintenance_orders.values())
            return [c for c in self._wo_combos() if status_count[c.status] > MAX_TREFFER]
        if kind == "anschluss_pairs":
            return self._anschluss_pairs()
        if kind == "trips_same_vehicle":
            # E7: 276 vehicles serve >=2 trips — the two smallest zugnummern per vehicle
            by_veh: dict[str, list] = {}
            for t in self.pool("trips"):
                by_veh.setdefault(t.vehicle_id, []).append(t)
            out = []
            for vid in sorted(by_veh):
                ts = sorted(by_veh[vid], key=lambda t: t.zugnummer)
                if len(ts) >= 2:
                    out.append(VehPair(vid, vid, ts[0], ts[1]))
            return out
        if kind == "trips_veh_abgeschlossen":
            # E5c: vehicles with >=1 terminal (abgeschlossen) order -> policy-refusal target
            ok = {v for v, os_ in self._veh_orders().items()
                  if any(o.status == "abgeschlossen" for o in os_)}
            return self._rep_trip_per_vehicle(ok)
        if kind == "employees_common_name":
            # E12: unique names are near-nonexistent (320 name combos for 2140 staff), so the
            # name-search template is ALWAYS the ambiguous variant (search -> refine by base)
            return self._name_groups()
        # ----------------------------- wave-3.5 pools --------------------------------------
        if kind == "trips_pos_lokf":
            # K1: trips with a live position AND an assigned Lokführer (4-facet Lagebericht)
            pos = {p.trip_id for p in self.master.positions}
            lokf = {a.trip_id for a in self.master.assignments if a.role == "Lokführer"}
            return [t for t in self.pool("trips") if t.trip_id in pos and t.trip_id in lokf]
        if kind == "wo_batch_konflikt":
            # K2: (status, depot, faellig_vor[, sev]) combos with EXACTLY 2 open hits, in a depot
            # that also has >=1 terminal (abgeschlossen) order -> 2 succeed, 1 gets rejected
            closed_depots = {o.depot for o in self.master.maintenance_orders.values()
                             if o.status == "abgeschlossen"}
            return [c for c in self._wo_combos_batch()
                    if len(c.order_ids) == 2 and c.depot in closed_depots]
        if kind == "veh_pairs_mixed":
            # T3: same-vehicle pairs (Umlauf) + equally many different-vehicle pairs, so the
            # verdict is genuinely open and both branches of the conditional write occur
            same = [MixPair(f"same-{p.key}", True, p.trip_a, p.trip_b,
                            p.vehicle_id, p.vehicle_id)
                    for p in self.pool("trips_same_vehicle")]
            rep: dict[str, object] = {}
            for t in self.pool("trips"):
                cur = rep.get(t.vehicle_id)
                if cur is None or t.zugnummer < cur.zugnummer:
                    rep[t.vehicle_id] = t
            vids = sorted(rep)
            diff = [MixPair(f"diff-{vids[i]}-{vids[i + 1]}", False,
                            rep[vids[i]], rep[vids[i + 1]], vids[i], vids[i + 1])
                    for i in range(0, len(vids) - 1, 2)]
            return same + diff[:len(same)]
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
        # wave-2.5: 12 times, edge-heavy, all inside real shift coverage 04:00-23:00 (the old
        # "23:30" was dead — shifts end at min(23, start+8)). Edge times keep the 1-3-hit
        # window alive now that 20 depots halve the per-bucket employee density.
        times = ["04:15", "04:45", "05:15", "05:45", "06:15", "06:45",
                 "07:30", "20:45", "21:30", "22:15", "22:45", "23:00"]
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
        """Filter combos (status, depot, faellig_vor[, schweregrad]) with 1-3 matching orders.
        wave-2.5: 12 half-daily cutoffs covering the real due_at range (sim -48h..+120h =
        2026-06-27..07-04; the old 2026-06-25 cutoff was dead, 07-13 duplicated 07-06) plus
        schweregrad as optional 4th dimension — keeps 1-3-hit windows alive at ~4x orders."""
        depots = sorted({o.depot for o in self.master.maintenance_orders.values()})
        cutoffs = ["2026-06-27 12:00", "2026-06-28", "2026-06-28 12:00", "2026-06-29",
                   "2026-06-29 12:00", "2026-06-30", "2026-06-30 12:00", "2026-07-01",
                   "2026-07-01 12:00", "2026-07-02", "2026-07-03", "2026-07-04"]
        combos = []
        for status in ("geplant", "in_Arbeit", "überfällig"):
            for depot in depots:
                for cutoff in cutoffs:
                    for sev in (None, "niedrig", "mittel", "hoch"):
                        hits = tuple(sorted(o.order_id for o in self.master.maintenance_orders.values()
                                            if o.status == status and o.depot == depot
                                            and o.due_at < cutoff
                                            and (sev is None or o.severity == sev)))
                        if 1 <= len(hits) <= 3:
                            combos.append(WOCombo(f"{status}|{depot}|{cutoff}|{sev or 'alle'}",
                                                  status, depot, cutoff, sev, hits))
        return combos

    def _wo_combos_batch(self) -> list:
        """Wave 3 (E2): (status, depot, faellig_vor) combos with 2-3 OPEN orders (geplant/
        überfällig) — batch-write targets. Only wartung_status_setzen batches safely: it is
        commutative and mints no ids (wartung_einplanen/crew_zuweisen derive new ids from
        table length -> order-dependent hashes)."""
        depots = sorted({o.depot for o in self.master.maintenance_orders.values()})
        cutoffs = ["2026-06-27 12:00", "2026-06-28", "2026-06-28 12:00", "2026-06-29",
                   "2026-06-29 12:00", "2026-06-30", "2026-06-30 12:00", "2026-07-01",
                   "2026-07-01 12:00", "2026-07-02", "2026-07-03", "2026-07-04"]
        combos = []
        for status in ("geplant", "überfällig"):
            for depot in depots:
                for cutoff in cutoffs:
                    for sev in (None, "niedrig", "mittel", "hoch"):
                        hits = tuple(sorted(o.order_id for o in self.master.maintenance_orders.values()
                                            if o.status == status and o.depot == depot
                                            and o.due_at < cutoff
                                            and (sev is None or o.severity == sev)))
                        if 2 <= len(hits) <= 3:
                            combos.append(WOCombo(f"batch|{status}|{depot}|{cutoff}|{sev or 'alle'}",
                                                  status, depot, cutoff, sev, hits))
        return combos

    def _anschluss_pairs(self) -> list:
        """Wave 3 (E10): one ordered pair (zubringer -> anschluss) per zubringer trip, sharing a
        station with a PLAN transfer buffer of 8-45 min: >= MIN_UMSTIEG_MIN, so the no-delay case
        is reachable, yet small enough that an injected zubringer delay flips it. The zubringer
        must actually ARRIVE there (not its first stop) and the anschluss must still DEPART
        (not its last stop). Deterministic pick: earliest arrival, then smallest anschluss."""
        first_seq: dict[str, int] = {}
        last_seq: dict[str, int] = {}
        by_station: dict[str, list] = {}
        for s in self.master.schedule:
            first_seq[s.trip_id] = min(first_seq.get(s.trip_id, s.seq), s.seq)
            last_seq[s.trip_id] = max(last_seq.get(s.trip_id, s.seq), s.seq)
            by_station.setdefault(s.station_id, []).append(s)
        best: dict[str, AnschlussPair] = {}
        for st_id in sorted(by_station):
            stops = by_station[st_id]
            for a in stops:
                if a.seq == first_seq[a.trip_id]:
                    continue
                for b in stops:
                    if a.trip_id == b.trip_id or b.seq == last_seq[b.trip_id]:
                        continue
                    puffer = (int(b.dep[:2]) * 60 + int(b.dep[3:5])) \
                        - (int(a.arr[:2]) * 60 + int(a.arr[3:5]))
                    if not (8 <= puffer <= 45):
                        continue
                    zu, an = self.master.trips[a.trip_id], self.master.trips[b.trip_id]
                    cand = AnschlussPair(sid(zu.zugnummer), zu, an, st_id, a.arr, b.dep, puffer)
                    cur = best.get(cand.key)
                    if cur is None or (cand.arr, cand.ans.zugnummer) < (cur.arr, cur.ans.zugnummer):
                        best[cand.key] = cand
        return [best[k] for k in sorted(best)]

    def _name_groups(self) -> list:
        """Wave 3 (E12b): names with 3-14 carriers where ONE home_base pins exactly one of them —
        the ticket names person + base, the agent must search by name and refine."""
        by_name: dict[str, list] = {}
        for eid in sorted(self.master.employees):
            e = self.master.employees[eid]
            by_name.setdefault(e.name, []).append(e)
        out = []
        for name in sorted(by_name):
            emps = by_name[name]
            if not (3 <= len(emps) <= 14):
                continue
            base_cnt = Counter(e.home_base for e in emps)
            target = next((e for e in emps if base_cnt[e.home_base] == 1), None)
            if target is not None:
                out.append(NameGroup(sid(name), name, tuple(e.emp_id for e in emps),
                                     target.emp_id, target.home_base))
        return out

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


def oc(_name: str, **arguments) -> dict:
    """One oracle_calls entry: the exact valid-path call (drives rollout.py's dry-run oracle).
    Positional is _name so tools with a literal `name=` argument (mitarbeiter_suchen, wave 3)
    don't collide."""
    return {"name": _name, "arguments": arguments}


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
# wave 3 (A): ticket-realism helpers. DEDICATED rng streams ("ticket"/"distraktor") — they
# never touch a template's own draw stream, so pre-wave-3 task CONTENT stays byte-stable
# even when a template adds phrasing variants.
# ---------------------------------------------------------------------------------------
# Distractor sentences must contain NO entity-pattern tokens (ids, times, train numbers,
# dates): everything in the ticket lands in the grounding corpus and would weaken the
# anti-hallucination check.
DISTRAKTOREN = [
    "Unabhängig davon: die Kaffeemaschine im Pausenraum ist wieder defekt, aber das klären wir intern.",
    "Der Flurfunk meldet mal wieder alles Mögliche dazu, aber das gehört nicht hierher.",
    "Die Übergabe von der Frühschicht war heute etwas chaotisch, nur damit du es weißt.",
    "Bitte nicht wundern, falls das Ticketsystem diese Anfrage doppelt verschickt hat.",
]


def vary(g: Gen, tname: str, entity: str, variants: list, salt: str = "v"):
    """Pick one ticket-phrasing variant, seeded on (seed, 'ticket', tname, entity, salt)."""
    return rng(g.seed, "ticket", tname, str(entity), salt).choice(variants)


def maybe_distraktor(g: Gen, tname: str, entity: str, p: float = 0.3) -> str:
    """With prob p, one harmless distractor sentence to append to a ticket (else '')."""
    r = rng(g.seed, "distraktor", tname, str(entity))
    roll = r.random()
    return (" " + r.choice(DISTRAKTOREN)) if roll < p else ""

