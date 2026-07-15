"""
sdg_pipeline/db_bahn/gen_templates_hard.py
===========================================
Hard-tier task templates (search without ids, 3-4-call chains, conditional writes,
runtime faults). Split out of gen_tasks.py 2026-07-15 — registry + main() live there.
"""

from sdg_pipeline.db_bahn.seed_worldstate import rng
from sdg_pipeline.db_bahn.gen_tasks_lib import (
    DELAY_CAUSES, QUALS, Gen, MACombo, WOCombo,
    _apply, _grund_comm, build_task, env_assert, inj_call, oc, qual_for, ref_action, sid)
from sdg_pipeline.db_bahn.tau2_domain.tools import BahnTools

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
    args = {"status": combo.status, "depot": combo.depot, "faellig_vor": combo.faellig_vor}
    if combo.schweregrad is not None:
        args["schweregrad"] = combo.schweregrad
    res = tk.wartung_liste(**args)
    rows = res["treffer"]
    if not (1 <= len(rows) <= 3) or {r_["order_id"] for r_ in rows} != set(combo.order_ids):
        return None
    comm = [r_["order_id"] for r_ in rows] + sorted({r_["due_at"][:10] for r_ in rows})
    sev_txt = f" mit Schweregrad '{combo.schweregrad}'" if combo.schweregrad is not None else ""
    ticket = (f"Welche Wartungsaufträge mit Status '{combo.status}'{sev_txt} im Depot {combo.depot} sind "
              f"vor dem {combo.faellig_vor} fällig? Nenne je Auftrag die Auftrags-ID und das Fälligkeitsdatum.")
    task = build_task(f"info_wartung_depot__{sid(combo.key)}__{idx:03d}", ticket,
                      "INFO+SUCHE: Wartungsaufträge flottenweit filtern (Status/Depot/Fälligkeit[/Schweregrad])",
                      [], None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["wartung_liste"],
           "oracle_calls": [oc("wartung_liste", **args)],
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

