"""
sdg_pipeline/db_bahn/gen_templates_wave3.py
===========================================
Wave-3 hardening templates (2026-07-18): the world was too easy — base Qwen3-4B+Think solved
96% of heldout. These templates exploit the world's LATENT hardness (14x duplicate names,
276 shared-vehicle Umläufe, >10-hit search truncation) and add the mechanics the SFT data
never demanded: parallel calls per turn, iteration over search hits, aggregation, refusal,
multi-rejection cascades, data gaps, transfer-time logic and transient tool errors.

Verifier wiring (wave 3, see evaluation/trajectory_reward.py):
  - kind "refusal" -> no_write gate (init-only hash) + optional key["forbidden_tools"]
    (tools the agent must NOT call; checked as comp["no_forbidden"]).
  - NEW oracle_calls convention (wave-3 classes only): scripted REJECTIONS and RETRIES are
    part of the oracle path (cascade: bad1, bad2, search, write; transient: call, call), so
    the dry-run oracle exercises the error loop and expected_calls stays realistic for the
    >=3x efficiency filter. Reference `actions` still contain ONLY valid mutating writes.
  - communicate_info stays substring-based: tickets pin decisive wording where model phrasing
    would otherwise be too free ("kein Grund vermerkt", "dasselbe Fahrzeug").
"""

import re

from sdg_pipeline.db_bahn.seed_worldstate import rng
from sdg_pipeline.db_bahn.gen_tasks_lib import (
    DELAY_CAUSES, MAINT_TYPES, Gen,
    _apply, _grund_comm, build_task, env_assert, inj_call, maybe_distraktor, oc,
    qual_for, ref_action, sid)
from sdg_pipeline.db_bahn.tau2_domain.tools import BahnTools

WRITE_TOOLS = ["wartung_einplanen", "crew_zuweisen", "wartung_status_setzen"]


# ---------------------------------------------------------------------------------------
# E1: batch info — 3 independent status queries in one ticket (parallel-call substrate)
# ---------------------------------------------------------------------------------------
def t_info_batch_verspaetung(g: Gen, batch, idx, inject: bool):
    r = rng(g.seed, "inj", "batch_versp", batch.key)
    targets = r.sample(list(batch.trips), r.choice([1, 2]))
    injections = [inj_call("inject_verspaetung", zugnummer=t.zugnummer,
                           minuten=r.choice([25, 35, 45, 60, 90]), grund=r.choice(DELAY_CAUSES))
                  for t in targets]
    tk = _apply(g.fresh(), injections)
    comm = []
    for t in batch.trips:
        v = tk.verspaetung(t.zugnummer)
        comm += [t.zugnummer] + _grund_comm(v)
    z1, z2, z3 = (t.zugnummer for t in batch.trips)
    ticket = (f"Kurzer Lagecheck für die Leitstelle: Wie ist der aktuelle Verspätungsstand von "
              f"{z1}, {z2} und {z3}? Die drei Abfragen sind unabhängig voneinander. Nenne pro Zug "
              f"die Minuten und den Grund (oder 'pünktlich')."
              + maybe_distraktor(g, "batch_versp", batch.key))
    task = build_task(f"info_batch_verspaetung__{batch.key}__{idx:03d}", ticket,
                      "INFO+BATCH: 3 unabhängige Verspätungsabfragen (Parallel-Substrat)",
                      injections, None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["verspaetung"],
           "oracle_calls": [oc("verspaetung", zugnummer=z) for z in (z1, z2, z3)],
           "facts": {"zuege": [z1, z2, z3]}}
    return task, key


# ---------------------------------------------------------------------------------------
# E2: batch write — find 2-3 open orders via list search, set EVERY one to in_Arbeit
# ---------------------------------------------------------------------------------------
def t_action_batch_liste(g: Gen, combo, idx, inject: bool):
    sev_clause = f" mit Schweregrad '{combo.schweregrad}'" if combo.schweregrad else ""
    ticket = (f"Alle Wartungsaufträge mit Status '{combo.status}' im Depot {combo.depot}"
              f"{sev_clause}, die vor {combo.faellig_vor} fällig sind, werden jetzt bearbeitet. "
              f"Ermittle die betroffenen Aufträge und setze JEDEN davon auf 'in_Arbeit'. "
              f"Nenne am Ende alle betroffenen Auftrags-IDs.")
    search_args = {"status": combo.status, "depot": combo.depot, "faellig_vor": combo.faellig_vor}
    if combo.schweregrad:
        search_args["schweregrad"] = combo.schweregrad
    refs = [ref_action(i + 1, "wartung_status_setzen", auftrag_id=oid, status="in_Arbeit")
            for i, oid in enumerate(combo.order_ids)]
    asserts = [env_assert("assert_maintenance_status", auftrag_id=oid, status="in_Arbeit")
               for oid in combo.order_ids]
    task = build_task(f"action_batch_liste__{sid(combo.key)}__{idx:03d}", ticket,
                      "ACTION+BATCH: Suche -> alle Treffer-Status setzen (2-3 Writes)",
                      [], refs, asserts, list(combo.order_ids), ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "expected_tools": ["wartung_liste", "wartung_status_setzen"],
           "oracle_calls": [oc("wartung_liste", **search_args)]
           + [oc("wartung_status_setzen", auftrag_id=oid, status="in_Arbeit")
              for oid in combo.order_ids],
           "facts": {"order_ids": list(combo.order_ids)}}
    return task, key


# ---------------------------------------------------------------------------------------
# E3: iteration with condition — search 3-8 trains, check each, report only the >=30-min ones
# ---------------------------------------------------------------------------------------
def t_info_iteration_bedingt(g: Gen, grp, idx, inject: bool):
    r = rng(g.seed, "inj", "iter_bedingt", grp.key)
    targets = r.sample(list(grp.zugnummern), min(r.choice([1, 2]), len(grp.zugnummern)))
    injections = [inj_call("inject_verspaetung", zugnummer=z,
                           minuten=r.choice([30, 40, 55, 70]), grund=r.choice(DELAY_CAUSES))
                  for z in targets]
    tk = _apply(g.fresh(), injections)
    comm = []
    for z in grp.zugnummern:
        v = tk.verspaetung(z)
        if v["verspaetung_minuten"] >= 30:
            comm += [z, v["grund"]]
    richtung_txt = "ab" if grp.richtung == "von" else "nach"
    ticket = (f"Verschaffe mir einen Überblick: Suche alle Züge {richtung_txt} {grp.station} "
              f"(Filter '{grp.station}') und finde heraus, welche davon aktuell mindestens 30 "
              f"Minuten Verspätung haben. Nenne für jeden betroffenen Zug die Zugnummer, die "
              f"Minuten und den Grund.")
    task = build_task(f"info_iteration_bedingt__{grp.key}__{idx:03d}", ticket,
                      "INFO+ITERATION: Suche -> je Treffer prüfen -> bedingt melden (4-9 Calls)",
                      injections, None, None, comm, ["COMMUNICATE"])
    search_arg = {grp.richtung: grp.station}
    key = {"kind": "info", "expected_tools": ["zuege_suchen", "verspaetung"],
           "oracle_calls": [oc("zuege_suchen", **search_arg)]
           + [oc("verspaetung", zugnummer=z) for z in grp.zugnummern],
           "facts": {"gruppe": list(grp.zugnummern), "betroffen": comm[::2]}}
    return task, key


# ---------------------------------------------------------------------------------------
# E4: aggregation — which of the 3 trains has the HIGHEST delay (distinct injected minutes)
# ---------------------------------------------------------------------------------------
def t_info_aggregation(g: Gen, batch, idx, inject: bool):
    r = rng(g.seed, "inj", "aggregation", batch.key)
    minutes = r.sample([20, 25, 35, 40, 50, 60, 75, 90], 3)
    injections = [inj_call("inject_verspaetung", zugnummer=t.zugnummer,
                           minuten=m, grund=r.choice(DELAY_CAUSES))
                  for t, m in zip(batch.trips, minutes)]
    tk = _apply(g.fresh(), injections)
    vs = [tk.verspaetung(t.zugnummer) for t in batch.trips]
    winner = max(vs, key=lambda v: v["verspaetung_minuten"])
    z1, z2, z3 = (t.zugnummer for t in batch.trips)
    comm = [winner["zugnummer"], f"{winner['verspaetung_minuten']} minuten", winner["grund"]]
    ticket = (f"Vergleiche die aktuellen Verspätungen von {z1}, {z2} und {z3}: Welcher der drei "
              f"Züge hat gerade die höchste Verspätung? Nenne den Zug, seine Verspätung in "
              f"Minuten und den Grund.")
    task = build_task(f"info_aggregation__{batch.key}__{idx:03d}", ticket,
                      "INFO+AGGREGATION: Maximum über 3 Abfragen bestimmen",
                      injections, None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["verspaetung"],
           "oracle_calls": [oc("verspaetung", zugnummer=z) for z in (z1, z2, z3)],
           "facts": {"sieger": winner["zugnummer"], "minuten": winner["verspaetung_minuten"]}}
    return task, key


# ---------------------------------------------------------------------------------------
# E5a-c: refusal — non-existent entity / impossible request / policy-forbidden write
# ---------------------------------------------------------------------------------------
def t_refusal_nicht_existent(g: Gen, trip, idx, inject: bool):
    digits = re.sub(r"\D", "", trip.zugnummer) or "0"
    fake = f"{trip.product} {int(digits) + 7000}"
    try:
        g.tools._find_trip(fake)
        return None  # collision with a real train — skip
    except ValueError:
        pass
    ticket = (f"Prüfe bitte die aktuelle Verspätung von {fake} und gib Minuten und Grund an. "
              f"Falls das System den Zug nicht kennt, antworte kurz mit '{fake} nicht gefunden' "
              f"und ändere nichts.")
    task = build_task(f"refusal_nicht_existent__{sid(fake)}__{idx:03d}", ticket,
                      "REFUSAL: nicht existierender Zug -> ehrlich melden statt raten",
                      [], None, None, ["nicht gefunden", fake], ["COMMUNICATE"])
    key = {"kind": "refusal", "fault": "runtime", "expected_tools": ["verspaetung"],
           "forbidden_tools": WRITE_TOOLS,
           "oracle_calls": [oc("verspaetung", zugnummer=fake)],  # the failing lookup IS the path
           "facts": {"fake": fake}}
    return task, key


def t_refusal_nicht_machbar(g: Gen, trip, idx, inject: bool):
    """No tool can cancel a train / rebook passengers. Correct: one grounding READ, then a clear
    refusal WITHOUT any write. NOTE comm asymmetry: a hallucinated 'habe storniert' would pass
    no_write, so comm requires the honesty marker 'nicht' alongside the topic word."""
    ticket = (f"Bitte storniere {trip.zugnummer} für heute komplett und informiere die "
              f"betroffenen Reisenden über die Ausfälle. Wirf vorher einen kurzen Blick in den "
              f"Fahrplan des Zuges, damit wir über denselben Zug reden."
              + maybe_distraktor(g, "refusal_machbar", trip.zugnummer))
    task = build_task(f"refusal_nicht_machbar__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "REFUSAL: Auftrag außerhalb der Tool-Fähigkeiten -> belegt ablehnen",
                      [], None, None, [trip.zugnummer, "stornier", "nicht"], ["COMMUNICATE"])
    key = {"kind": "refusal", "expected_tools": ["fahrplan"],
           "forbidden_tools": WRITE_TOOLS,
           "oracle_calls": [oc("fahrplan", zugnummer=trip.zugnummer)],
           "facts": {"zugnummer": trip.zugnummer}}
    return task, key


def t_refusal_policy(g: Gen, trip, idx, inject: bool):
    tk = g.fresh()
    w = tk.wartung_status(trip.zugnummer)
    done = sorted((o for o in w["wartungsauftraege"] if o["status"] == "abgeschlossen"),
                  key=lambda o: o["order_id"])
    if not done:
        return None
    oid = done[0]["order_id"]
    ticket = (f"Der Wartungsauftrag {oid} (Fahrzeug von {trip.zugnummer}) soll wieder geöffnet "
              f"werden — setze ihn zurück auf 'geplant'. Prüfe zuerst seinen aktuellen Status; "
              f"Endstatus-Aufträge dürfen laut Richtlinie nicht geändert werden — in dem Fall "
              f"nichts ändern und kurz begründen.")
    task = build_task(f"refusal_policy__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "REFUSAL: Endstatus-Auftrag -> Check-first, keine Änderung, Begründung",
                      [], None, None, [oid, "abgeschlossen"], ["COMMUNICATE"])
    key = {"kind": "refusal", "expected_tools": ["wartung_status"],
           "forbidden_tools": ["wartung_status_setzen"],
           "oracle_calls": [oc("wartung_status", kennung=trip.zugnummer)],
           "facts": {"auftrag_id": oid}}
    return task, key


# ---------------------------------------------------------------------------------------
# E6: constraint cascade — TWO proposed candidates fail in sequence (role, then quali),
# only the search-first-hit path succeeds. Rejections are part of the oracle path.
# ---------------------------------------------------------------------------------------
def t_action_kaskade(g: Gen, trip, idx, inject: bool):
    qual = qual_for(trip)
    if qual is None:  # quali rejection needs a gated product (ICE/IC/EC)
        return None
    r = rng(g.seed, "act", "kaskade", trip.trip_id)
    tk0 = BahnTools(g.master)
    orig = [c for c in tk0.mitarbeiter_info(trip.zugnummer)["besatzung"] if c["rolle"] == "Lokführer"]
    if not orig:
        return None
    orig_id = orig[0]["mitarbeiter_id"]
    emps = g.master.employees
    assigned = {a.emp_id for a in g.master.assignments if a.trip_id == trip.trip_id}
    bad_rolle = sorted(eid for eid, e in emps.items()
                       if e.role == "Zugbegleiter" and eid not in assigned)
    bad_quali = sorted(eid for eid, e in emps.items()
                       if e.role == "Lokführer" and qual not in e.qualifications
                       and eid not in assigned)
    if not bad_rolle or not bad_quali:
        return None
    bad1, bad2 = emps[r.choice(bad_rolle)], emps[r.choice(bad_quali)]
    base = emps[orig_id].home_base if orig_id in emps else None
    if base is None:
        return None
    cand_valid = sorted(eid for eid, e in emps.items()
                        if e.role == "Lokführer" and e.home_base == base
                        and qual in e.qualifications)
    if not cand_valid:
        return None
    first = cand_valid[0]
    if first in (orig_id, bad1.emp_id, bad2.emp_id) or first in assigned:
        return None
    ersatz = emps[first]
    injections = [inj_call("inject_lokfuehrer_ausfall", zugnummer=trip.zugnummer)]
    # wave 3.5 (soft H1): proposal ORDER stays verbatim (pins the cascade oracle), the search
    # route is replaced by the goal + determinism anchors
    ticket = (f"Der eingeteilte Lokführer von {trip.zugnummer} ist kurzfristig ausgefallen. Die "
              f"Disposition schlägt in dieser Reihenfolge vor: erst {bad1.name} ({bad1.emp_id}), "
              f"sonst {bad2.name} ({bad2.emp_id}). Prüfe die Besatzung und arbeite die Vorschläge "
              f"der Reihe nach ab. Führt beides nicht zum Ziel: sorge für qualifizierten Ersatz — "
              f"ein Lokführer mit Qualifikation {qual} von der Heimatbasis {base}; bei mehreren "
              f"Kandidaten zählt die kleinste Mitarbeiter-ID. Nenne am Ende die zugeteilte "
              f"Mitarbeiter-ID.")
    refs = [ref_action(1, "crew_zuweisen", zugnummer=trip.zugnummer,
                       mitarbeiter_id=ersatz.emp_id, rolle="Lokführer")]
    asserts = [
        env_assert("assert_crew_assigned", zugnummer=trip.zugnummer, mitarbeiter_id=ersatz.emp_id),
        env_assert("assert_crew_assigned", assert_value=False,
                   zugnummer=trip.zugnummer, mitarbeiter_id=bad1.emp_id),
        env_assert("assert_crew_assigned", assert_value=False,
                   zugnummer=trip.zugnummer, mitarbeiter_id=bad2.emp_id),
        env_assert("assert_crew_assigned", assert_value=False,
                   zugnummer=trip.zugnummer, mitarbeiter_id=orig_id),
    ]
    task = build_task(f"action_kaskade__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION+KASKADE: zwei Ablehnungen in Folge -> Suche -> Zuteilung",
                      injections, refs, asserts, [ersatz.emp_id], ["DB", "ENV_ASSERTION"])
    key = {"kind": "action", "fault": "state+runtime",
           "expected_tools": ["mitarbeiter_info", "crew_zuweisen", "mitarbeiter_suchen"],
           "oracle_calls": [
               oc("mitarbeiter_info", zugnummer=trip.zugnummer),
               oc("crew_zuweisen", zugnummer=trip.zugnummer,        # rejected: wrong role
                  mitarbeiter_id=bad1.emp_id, rolle="Lokführer"),
               oc("crew_zuweisen", zugnummer=trip.zugnummer,        # rejected: missing quali
                  mitarbeiter_id=bad2.emp_id, rolle="Lokführer"),
               oc("mitarbeiter_suchen", rolle="Lokführer", qualifikation=qual, heimatbasis=base),
               oc("crew_zuweisen", zugnummer=trip.zugnummer,
                  mitarbeiter_id=ersatz.emp_id, rolle="Lokführer")],
           "facts": {"bad1": bad1.emp_id, "bad2": bad2.emp_id, "ersatz_id": ersatz.emp_id,
                     "ausgefallen": orig_id, "basis": base}}
    return task, key


# ---------------------------------------------------------------------------------------
# E7: Umlauf ambiguity — do two trains share one vehicle? (ticket may claim otherwise)
# ---------------------------------------------------------------------------------------
def t_info_umlauf_fahrzeug(g: Gen, pair, idx, inject: bool):
    r = rng(g.seed, "inj", "umlauf", pair.key)
    za, zb = pair.trip_a.zugnummer, pair.trip_b.zugnummer
    claim = (" Ein Kollege meint, das seien zwei verschiedene Fahrzeuge — prüfe das nach."
             if r.random() < 0.5 else "")
    # wave 3.5 (H5): the verdict dictation is gone — the shared vehicle_id IS the proof and the
    # only comm anchor (documented residual: the verdict wording itself is unchecked)
    ticket = (f"Fahren {za} und {zb} heute mit demselben Fahrzeug?{claim} Nenne die Fahrzeug-ID(s).")
    comm = [pair.vehicle_id]
    task = build_task(f"info_umlauf_fahrzeug__{pair.key}-{sid(za)}__{idx:03d}", ticket,
                      "INFO+UMLAUF: Fahrzeug-Identität zweier Züge klären (Zug- vs. Fahrzeugebene)",
                      [], None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["wartung_status"],
           "oracle_calls": [oc("wartung_status", kennung=za), oc("wartung_status", kennung=zb)],
           "facts": {"vehicle_id": pair.vehicle_id, "zuege": [za, zb]}}
    return task, key


# ---------------------------------------------------------------------------------------
# E8a/b: >10-hit refinement — the COUNT is only visible in the broad (truncated) search,
# the ids only in the refined one -> both calls are load-bearing.
# ---------------------------------------------------------------------------------------
def t_info_ma_verfeinern(g: Gen, combo, idx, inject: bool):
    broad = g.tools.mitarbeiter_suchen(rolle=combo.rolle, qualifikation=combo.qualifikation)
    if broad["treffer_gesamt"] <= 10:
        return None
    ticket = (f"Wie viele {combo.rolle} mit Qualifikation {combo.qualifikation} haben wir "
              f"insgesamt, und welche davon sind an der Heimatbasis {combo.heimatbasis} um "
              f"{combo.verfuegbar_um} verfügbar? Nenne die Gesamtzahl und die Mitarbeiter-IDs "
              f"der Verfügbaren.")
    comm = [str(broad["treffer_gesamt"])] + list(combo.emp_ids)
    task = build_task(f"info_ma_verfeinern__{sid(combo.key)}__{idx:03d}", ticket,
                      "INFO+VERFEINERN: breite Suche (>10, abgeschnitten) -> Filter verfeinern",
                      [], None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["mitarbeiter_suchen"],
           "oracle_calls": [
               oc("mitarbeiter_suchen", rolle=combo.rolle, qualifikation=combo.qualifikation),
               oc("mitarbeiter_suchen", rolle=combo.rolle, qualifikation=combo.qualifikation,
                  heimatbasis=combo.heimatbasis, verfuegbar_um=combo.verfuegbar_um)],
           "facts": {"gesamt": broad["treffer_gesamt"], "verfuegbar": list(combo.emp_ids)}}
    return task, key


def t_info_wo_verfeinern(g: Gen, combo, idx, inject: bool):
    broad = g.tools.wartung_liste(status=combo.status)
    if broad["treffer_gesamt"] <= 10:
        return None
    sev_clause = f" mit Schweregrad '{combo.schweregrad}'" if combo.schweregrad else ""
    ticket = (f"Wie viele Wartungsaufträge stehen flottenweit auf '{combo.status}', und welche "
              f"davon betreffen das Depot {combo.depot}{sev_clause} mit Fälligkeit vor "
              f"{combo.faellig_vor}? Nenne die Gesamtzahl und die Auftrags-IDs.")
    refined_args = {"status": combo.status, "depot": combo.depot, "faellig_vor": combo.faellig_vor}
    if combo.schweregrad:
        refined_args["schweregrad"] = combo.schweregrad
    comm = [str(broad["treffer_gesamt"])] + list(combo.order_ids)
    task = build_task(f"info_wo_verfeinern__{sid(combo.key)}__{idx:03d}", ticket,
                      "INFO+VERFEINERN: Flotten-Suche (>10) -> Depot/Fälligkeit eingrenzen",
                      [], None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["wartung_liste"],
           "oracle_calls": [oc("wartung_liste", status=combo.status),
                            oc("wartung_liste", **refined_args)],
           "facts": {"gesamt": broad["treffer_gesamt"], "auftraege": list(combo.order_ids)}}
    return task, key


# ---------------------------------------------------------------------------------------
# E9: data gap — position removed / delay without recorded cause -> honest gap reporting
# ---------------------------------------------------------------------------------------
def t_info_datenluecke(g: Gen, trip, idx, inject: bool):
    r = rng(g.seed, "inj", "datenluecke", trip.trip_id)
    if r.random() < 0.5:
        injections = [inj_call("inject_standort_unbekannt", zugnummer=trip.zugnummer)]
        ticket = (f"Wo befindet sich {trip.zugnummer} gerade? Falls keine Positionsmeldung "
                  f"vorliegt, antworte wörtlich mit 'keine aktuelle Positionsmeldung'.")
        comm = ["keine aktuelle positionsmeldung"]
        oracle = [oc("zugstandort", zugnummer=trip.zugnummer)]
        tools = ["zugstandort"]
        facts = {"flavor": "standort"}
    else:
        minuten = r.choice([25, 35, 45, 60])
        injections = [inj_call("inject_grund_unbekannt", zugnummer=trip.zugnummer, minuten=minuten)]
        ticket = (f"Prüfe die aktuelle Verspätung von {trip.zugnummer}. Nenne die Minuten; falls "
                  f"kein Grund vermerkt ist, schreibe wörtlich 'kein Grund vermerkt'.")
        comm = [f"{minuten} minuten", "kein grund vermerkt"]
        oracle = [oc("verspaetung", zugnummer=trip.zugnummer)]
        tools = ["verspaetung"]
        facts = {"flavor": "grund", "minuten": minuten}
    task = build_task(f"info_datenluecke__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "INFO+LÜCKE: fehlende Daten ehrlich benennen statt erfinden",
                      injections, None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": tools, "oracle_calls": oracle, "facts": facts}
    return task, key


# ---------------------------------------------------------------------------------------
# E10: transfer-time logic — is the connection still reachable given current delays?
# ---------------------------------------------------------------------------------------
def t_info_anschluss(g: Gen, pair, idx, inject: bool):
    r = rng(g.seed, "inj", "anschluss", pair.key)
    injections = []
    if inject:  # delay the feeder just enough to flip the connection
        minuten = pair.puffer - 5 + r.choice([3, 6, 10, 20])
        injections = [inj_call("inject_verspaetung", zugnummer=pair.zub.zugnummer,
                               minuten=minuten, grund=r.choice(DELAY_CAUSES))]
    tk = _apply(g.fresh(), injections)
    station = tk._station_name(pair.station_id)
    res = tk.anschluss_pruefen(pair.zub.zugnummer, pair.ans.zugnummer, umsteigebahnhof=station)
    rows = res.get("umstiege") or []
    if not rows:
        return None
    row = rows[0]
    if row["anschluss_erreichbar"]:
        comm = [station, row["ankunft_zubringer_effektiv"], row["abfahrt_anschluss_effektiv"],
                f"{row['puffer_minuten']} minuten"]
    else:
        comm = [station, row["ankunft_zubringer_effektiv"], "nicht"]
    ticket = (f"Reisende aus {pair.zub.zugnummer} wollen in {station} auf {pair.ans.zugnummer} "
              f"umsteigen. Prüfe mit der Anschlussprüfung, ob das aktuell klappt. Nenne die "
              f"effektive Ankunft des Zubringers, die effektive Abfahrt des Anschlusses und den "
              f"Puffer in Minuten — und sag klar, ob der Anschluss erreicht wird oder nicht.")
    task = build_task(f"info_anschluss__{pair.key}-{sid(pair.ans.zugnummer)}__{idx:03d}", ticket,
                      "INFO+ZEITLOGIK: Anschluss-Erreichbarkeit inkl. Verspätungen",
                      injections, None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["anschluss_pruefen"],
           "oracle_calls": [oc("verspaetung", zugnummer=pair.zub.zugnummer),
                            oc("anschluss_pruefen", zubringer_zugnummer=pair.zub.zugnummer,
                               anschluss_zugnummer=pair.ans.zugnummer, umsteigebahnhof=station)],
           "facts": {"station": station, "erreichbar": row["anschluss_erreichbar"],
                     "puffer": row["puffer_minuten"]}}
    return task, key


# ---------------------------------------------------------------------------------------
# E11: partially done ticket — one request is ALREADY satisfied -> report, don't re-do
# ---------------------------------------------------------------------------------------
def t_info_teilerledigt(g: Gen, trip, idx, inject: bool):
    r = rng(g.seed, "inj", "teilerledigt", trip.trip_id)
    tk0 = BahnTools(g.master)
    lokf = [c for c in tk0.mitarbeiter_info(trip.zugnummer)["besatzung"] if c["rolle"] == "Lokführer"]
    if not lokf:
        return None
    L = lokf[0]
    injections = []
    if inject:
        injections = [inj_call("inject_verspaetung", zugnummer=trip.zugnummer,
                               minuten=r.choice([25, 35, 45, 60]), grund=r.choice(DELAY_CAUSES))]
    tk = _apply(g.fresh(), injections)
    v = tk.verspaetung(trip.zugnummer)
    comm = [L["mitarbeiter_id"], "bereits"] + _grund_comm(v)
    ticket = (f"Zwei Dinge zu {trip.zugnummer}: Erstens soll {L['name']} "
              f"({L['mitarbeiter_id']}) als Lokführer eingeteilt werden, falls noch nicht "
              f"geschehen. Zweitens brauche ich die aktuelle Verspätung (Minuten und Grund, oder "
              f"'pünktlich'). Prüfe die Besatzung, bevor du etwas änderst; ist die Zuteilung "
              f"schon vorhanden, sag ausdrücklich, dass sie bereits besteht.")
    task = build_task(f"info_teilerledigt__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "INFO+TEILERLEDIGT: erledigten Teil erkennen, nichts doppelt ausführen",
                      injections, None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["mitarbeiter_info", "verspaetung"],
           "oracle_calls": [oc("mitarbeiter_info", zugnummer=trip.zugnummer),
                            oc("verspaetung", zugnummer=trip.zugnummer)],
           "facts": {"lokfuehrer": L["mitarbeiter_id"], "verspaetung": v["verspaetung_minuten"]}}
    return task, key


# ---------------------------------------------------------------------------------------
# E12: name search — ambiguous name (3-14 carriers), refined by home base
# ---------------------------------------------------------------------------------------
def t_info_name_suche(g: Gen, ng, idx, inject: bool):
    hit = g.tools.mitarbeiter_suchen(name=ng.name, heimatbasis=ng.target_base)
    if hit["treffer_gesamt"] != 1 or hit["treffer"][0]["mitarbeiter_id"] != ng.target_emp_id:
        return None  # substring collision with another name — skip
    d = hit["treffer"][0]
    comm = [ng.target_emp_id, d["rolle"]]
    ticket = (f"Mir fehlt die Mitarbeiter-ID von {ng.name} von der Heimatbasis {ng.target_base} — "
              f"bitte heraussuchen. Achtung: der Name kommt mehrfach vor. Nenne die "
              f"Mitarbeiter-ID und die Rolle.")
    task = build_task(f"info_name_suche__{ng.key}__{idx:03d}", ticket,
                      "INFO+NAME: mehrdeutigen Namen suchen und per Heimatbasis auflösen",
                      [], None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["mitarbeiter_suchen"],
           "oracle_calls": [oc("mitarbeiter_suchen", name=ng.name),
                            oc("mitarbeiter_suchen", name=ng.name, heimatbasis=ng.target_base)],
           "facts": {"emp_id": ng.target_emp_id, "rolle": d["rolle"], "traeger": len(ng.emp_ids)}}
    return task, key


# ---------------------------------------------------------------------------------------
# E13: transient tool error — one retryable failure, ONE identical retry, then answer
# ---------------------------------------------------------------------------------------
def t_info_transient(g: Gen, trip, idx, inject: bool):
    tk = g.fresh()  # facts WITHOUT the transient injection — the data itself is unchanged
    s = tk.zugstandort(trip.zugnummer)
    if s.get("status") != "unterwegs" or not s.get("naechster_halt"):
        return None
    injections = [inj_call("inject_transient_stoerung", tool_name="zugstandort", anzahl=1)]
    ticket = f"Wo befindet sich {trip.zugnummer} gerade? Nenne insbesondere den nächsten Halt."
    task = build_task(f"info_transient__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "INFO+TRANSIENT: vorübergehender Dienstfehler -> genau EIN Retry",
                      injections, None, None, [s["naechster_halt"]], ["COMMUNICATE"])
    key = {"kind": "info", "fault": "transient", "expected_tools": ["zugstandort"],
           "oracle_calls": [oc("zugstandort", zugnummer=trip.zugnummer),   # fails (transient)
                            oc("zugstandort", zugnummer=trip.zugnummer)],  # retry succeeds
           "facts": {"naechster_halt": s["naechster_halt"]}}
    return task, key


# =======================================================================================
# Wave 3.5 (2026-07-18): conjunction tickets (K1-K3, multiplicative hardness) and depth
# compositions (T1-T3) — the corridor measurement showed base+think collapses on DEPTH
# (iteration 6%) and multi-part precision, not on single mechanics. Tickets follow the
# soft-H1 style: goals + determinism anchors, no tool-by-tool routes.
# =======================================================================================
def t_action_lagebericht(g: Gen, trip, idx, inject: bool):
    """K1: four concerns in one ticket — delay report, next stop, schedule a maintenance,
    and 'assign L as Lokführer unless already assigned' (L IS assigned: partially done).
    comm is scored on action tasks too, so the info parts gate alongside the DB write."""
    r = rng(g.seed, "act", "lagebericht", trip.trip_id)
    typ = r.choice(MAINT_TYPES)
    if any(o.type == typ and o.vehicle_id == trip.vehicle_id
           for o in g.master.maintenance_orders.values()):
        return None  # keep the exists-assert specific to OUR write
    due = f"2026-07-{r.randint(4, 10):02d} {r.choice(['06:00', '08:00', '22:00'])}"
    injections = []
    if inject:
        injections = [inj_call("inject_verspaetung", zugnummer=trip.zugnummer,
                               minuten=r.choice([25, 35, 45, 60]), grund=r.choice(DELAY_CAUSES))]
    tk = _apply(g.fresh(), injections)
    s = tk.zugstandort(trip.zugnummer)
    if s.get("status") != "unterwegs" or not s.get("naechster_halt"):
        return None
    v = tk.verspaetung(trip.zugnummer)
    lokf = [c for c in tk.mitarbeiter_info(trip.zugnummer)["besatzung"] if c["rolle"] == "Lokführer"]
    if not lokf:
        return None
    L = lokf[0]
    ticket = (f"Lagebericht und zwei Aufträge zu {trip.zugnummer}: (1) Melde die aktuelle "
              f"Verspätung (Minuten und Grund, oder 'pünktlich') und den nächsten Halt. "
              f"(2) Plane für das Fahrzeug von {trip.zugnummer} eine Wartung vom Typ '{typ}' ein, "
              f"fällig am {due}. (3) {L['name']} ({L['mitarbeiter_id']}) soll als Lokführer "
              f"eingeteilt sein, falls noch nicht geschehen — besteht die Zuteilung schon, sag "
              f"ausdrücklich, dass sie bereits besteht.")
    refs = [ref_action(1, "wartung_einplanen", fahrzeug_id=trip.vehicle_id, typ=typ, faellig_am=due)]
    asserts = [env_assert("assert_maintenance_exists", fahrzeug_id=trip.vehicle_id, typ=typ),
               env_assert("assert_crew_assigned", zugnummer=trip.zugnummer,
                          mitarbeiter_id=L["mitarbeiter_id"])]
    comm = _grund_comm(v) + [s["naechster_halt"], L["mitarbeiter_id"], "bereits"]
    task = build_task(f"action_lagebericht__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION+KONJUNKTION: Lagebericht + Write + teilerledigte Zuteilung (4 Anliegen)",
                      injections, refs, asserts, comm, ["DB", "ENV_ASSERTION", "COMMUNICATE"])
    key = {"kind": "action",
           "expected_tools": ["verspaetung", "zugstandort", "mitarbeiter_info",
                              "wartung_status", "wartung_einplanen"],
           "oracle_calls": [oc("verspaetung", zugnummer=trip.zugnummer),
                            oc("zugstandort", zugnummer=trip.zugnummer),
                            oc("mitarbeiter_info", zugnummer=trip.zugnummer),
                            oc("wartung_status", kennung=trip.zugnummer),
                            oc("wartung_einplanen", fahrzeug_id=trip.vehicle_id,
                               typ=typ, faellig_am=due)],
           "facts": {"lokfuehrer": L["mitarbeiter_id"], "typ": typ, "faellig_am": due,
                     "naechster_halt": s["naechster_halt"]}}
    return task, key


def t_action_batch_konflikt(g: Gen, combo, idx, inject: bool):
    """K2: three named orders, one is terminal (abgeschlossen) — set the two open ones,
    recognize and NAME the rejected one instead of forcing it."""
    a, b = combo.order_ids
    closed = sorted(o.order_id for o in g.master.maintenance_orders.values()
                    if o.depot == combo.depot and o.status == "abgeschlossen")
    if not closed:
        return None
    c = closed[0]
    ids = sorted([a, b, c])
    ticket = (f"Die Werkstatt {combo.depot} zieht die Bearbeitung vor: Setze die Wartungsaufträge "
              f"{ids[0]}, {ids[1]} und {ids[2]} auf 'in_Arbeit'. Endstatus-Aufträge dürfen laut "
              f"Richtlinie nicht geändert werden — gehört einer dazu, lass ihn unverändert und "
              f"benenne ihn mit seiner Auftrags-ID.")
    refs = [ref_action(i + 1, "wartung_status_setzen", auftrag_id=oid, status="in_Arbeit")
            for i, oid in enumerate((a, b))]
    asserts = [env_assert("assert_maintenance_status", auftrag_id=a, status="in_Arbeit"),
               env_assert("assert_maintenance_status", auftrag_id=b, status="in_Arbeit"),
               env_assert("assert_maintenance_status", auftrag_id=c, status="abgeschlossen")]
    task = build_task(f"action_batch_konflikt__{sid(combo.key)}__{idx:03d}", ticket,
                      "ACTION+KONJUNKTION: Batch-Write mit Endstatus-Ablehnung (2 setzen, 1 benennen)",
                      [], refs, asserts, [a, b, c, "abgeschlossen"],
                      ["DB", "ENV_ASSERTION", "COMMUNICATE"])
    key = {"kind": "action", "fault": "runtime",
           "expected_tools": ["wartung_status_setzen"],
           "oracle_calls": [oc("wartung_status_setzen", auftrag_id=oid, status="in_Arbeit")
                            for oid in ids],  # the rejected one IS part of the path
           "facts": {"offen": [a, b], "abgeschlossen": c}}
    return task, key


def t_info_batch_phantom(g: Gen, batch, idx, inject: bool):
    """K3: three-train status check, one train does not exist — report the two real ones
    normally and flag the phantom honestly (batch substrate x refusal honesty)."""
    r = rng(g.seed, "inj", "batch_phantom", batch.key)
    t1, t2, t3 = batch.trips
    digits = re.sub(r"\D", "", t3.zugnummer) or "0"
    fake = f"{t3.product} {int(digits) + 7000}"
    try:
        g.tools._find_trip(fake)
        return None  # collision with a real train — skip
    except ValueError:
        pass
    targets = r.sample([t1, t2], r.choice([1, 2]))
    injections = [inj_call("inject_verspaetung", zugnummer=t.zugnummer,
                           minuten=r.choice([25, 35, 45, 60]), grund=r.choice(DELAY_CAUSES))
                  for t in targets]
    tk = _apply(g.fresh(), injections)
    comm = []
    for t in (t1, t2):
        comm += [t.zugnummer] + _grund_comm(tk.verspaetung(t.zugnummer))
    comm += [fake, "nicht gefunden"]
    ticket = (f"Kurzer Lagecheck: Wie stehen {t1.zugnummer}, {fake} und {t2.zugnummer} gerade da? "
              f"Nenne pro Zug die Minuten und den Grund (oder 'pünktlich'). Kennt das System einen "
              f"der Züge nicht, melde ihn im Format '<Zugnummer> nicht gefunden', prüfe die übrigen "
              f"normal und ändere nichts.")
    task = build_task(f"info_batch_phantom__{batch.key}__{idx:03d}", ticket,
                      "INFO+KONJUNKTION: Batch-Lagecheck mit Phantom-Zug (2 melden, 1 ehrlich flaggen)",
                      injections, None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "fault": "state+runtime",
           "expected_tools": ["verspaetung"], "forbidden_tools": WRITE_TOOLS,
           "oracle_calls": [oc("verspaetung", zugnummer=t1.zugnummer),
                            oc("verspaetung", zugnummer=fake),   # fails — it IS the path
                            oc("verspaetung", zugnummer=t2.zugnummer)],
           "facts": {"real": [t1.zugnummer, t2.zugnummer], "fake": fake}}
    return task, key


def t_action_iteration_ersatz(g: Gen, grp, idx, inject: bool):
    """T1 (depth core): one train of a station group lost its Lokführer (injected) — search the
    group, check EVERY crew, identify the affected train, assign the anchored replacement.
    6-13 calls; exactly ONE write (crew ids are minted from table length — order-fragile)."""
    r = rng(g.seed, "act", "iter_ersatz", grp.key)
    lokf_of = {}
    for z in grp.zugnummern:
        trip = g.tools._find_trip(z)
        drivers = [a.emp_id for a in g.master.assignments
                   if a.trip_id == trip.trip_id and a.role == "Lokführer"]
        if not drivers:
            return None  # every member must have a driver, so the gap is UNIQUE post-injection
        lokf_of[z] = (trip, sorted(drivers)[0])
    betroffen_z = r.choice(sorted(grp.zugnummern))
    trip, orig_id = lokf_of[betroffen_z]
    orig = g.master.employees.get(orig_id)
    if orig is None:
        return None
    qual, base = qual_for(trip), orig.home_base
    assigned = {a.emp_id for a in g.master.assignments if a.trip_id == trip.trip_id}
    cand = sorted(eid for eid, e in g.master.employees.items()
                  if e.role == "Lokführer" and e.home_base == base
                  and (qual is None or qual in e.qualifications))
    if not cand or cand[0] == orig_id or cand[0] in assigned:
        return None
    first = cand[0]
    injections = [inj_call("inject_lokfuehrer_ausfall", zugnummer=betroffen_z)]
    qual_txt = f" mit Qualifikation {qual}" if qual else ""
    such_args = {"rolle": "Lokführer", "heimatbasis": base}
    if qual:
        such_args["qualifikation"] = qual
    richtung_txt = "ab" if grp.richtung == "von" else "nach"
    ticket = (f"Auf einem der Züge {richtung_txt} {grp.station} fehlt nach einem kurzfristigen "
              f"Ausfall der Lokführer. Suche alle Züge {richtung_txt} {grp.station} (Filter "
              f"'{grp.station}'), finde heraus, welcher Zug betroffen ist, und sorge dort für "
              f"Ersatz: ein Lokführer{qual_txt} von der Heimatbasis {base}; bei mehreren Kandidaten "
              f"zählt die kleinste Mitarbeiter-ID. Nenne den betroffenen Zug und die zugeteilte "
              f"Mitarbeiter-ID.")
    refs = [ref_action(1, "crew_zuweisen", zugnummer=betroffen_z,
                       mitarbeiter_id=first, rolle="Lokführer")]
    asserts = [env_assert("assert_crew_assigned", zugnummer=betroffen_z, mitarbeiter_id=first),
               env_assert("assert_crew_assigned", assert_value=False,
                          zugnummer=betroffen_z, mitarbeiter_id=orig_id)]
    search_arg = {grp.richtung: grp.station}
    task = build_task(f"action_iteration_ersatz__{grp.key}__{idx:03d}", ticket,
                      "ACTION+TIEFE: Gruppe absuchen, Betroffenen identifizieren, Ersatz zuteilen",
                      injections, refs, asserts, [betroffen_z, first],
                      ["DB", "ENV_ASSERTION", "COMMUNICATE"])
    key = {"kind": "action", "fault": "state",
           "expected_tools": ["zuege_suchen", "mitarbeiter_info", "mitarbeiter_suchen", "crew_zuweisen"],
           "oracle_calls": [oc("zuege_suchen", **search_arg)]
           + [oc("mitarbeiter_info", zugnummer=z) for z in grp.zugnummern]
           + [oc("mitarbeiter_suchen", **such_args),
              oc("crew_zuweisen", zugnummer=betroffen_z, mitarbeiter_id=first, rolle="Lokführer")],
           "facts": {"betroffen": betroffen_z, "ersatz_id": first, "ausgefallen": orig_id,
                     "basis": base}}
    return task, key


def t_action_doppelfault(g: Gen, trip, idx, inject: bool):
    """T2: three simultaneous faults — delay + driver dropout + a transient error on the FIRST
    verspaetung call (retry mandated by policy, NOT hinted in the ticket)."""
    r = rng(g.seed, "act", "doppelfault", trip.trip_id)
    drivers = [a.emp_id for a in g.master.assignments
               if a.trip_id == trip.trip_id and a.role == "Lokführer"]
    if not drivers:
        return None
    orig_id = sorted(drivers)[0]
    orig = g.master.employees.get(orig_id)
    if orig is None:
        return None
    qual, base = qual_for(trip), orig.home_base
    assigned = {a.emp_id for a in g.master.assignments if a.trip_id == trip.trip_id}
    cand = sorted(eid for eid, e in g.master.employees.items()
                  if e.role == "Lokführer" and e.home_base == base
                  and (qual is None or qual in e.qualifications))
    if not cand or cand[0] == orig_id or cand[0] in assigned:
        return None
    first = cand[0]
    state_inj = [inj_call("inject_verspaetung", zugnummer=trip.zugnummer,
                          minuten=r.choice([35, 45, 60, 75]), grund=r.choice(DELAY_CAUSES)),
                 inj_call("inject_lokfuehrer_ausfall", zugnummer=trip.zugnummer)]
    tk = _apply(g.fresh(), state_inj)  # facts WITHOUT the transient — the data is unchanged by it
    v = tk.verspaetung(trip.zugnummer)
    injections = state_inj + [inj_call("inject_transient_stoerung",
                                       tool_name="verspaetung", anzahl=1)]
    qual_txt = f" mit Qualifikation {qual}" if qual else ""
    such_args = {"rolle": "Lokführer", "heimatbasis": base}
    if qual:
        such_args["qualifikation"] = qual
    ticket = (f"{trip.zugnummer} meldet gerade beides: eine größere Verspätung und den "
              f"kurzfristigen Ausfall des eingeteilten Lokführers. Verschaffe dir das Lagebild — "
              f"aktuelle Minuten und Grund — und sorge für qualifizierten Ersatz: ein "
              f"Lokführer{qual_txt} von der Heimatbasis {base}; bei mehreren Kandidaten zählt die "
              f"kleinste Mitarbeiter-ID. Nenne die Minuten, den Grund und die zugeteilte "
              f"Mitarbeiter-ID.")
    refs = [ref_action(1, "crew_zuweisen", zugnummer=trip.zugnummer,
                       mitarbeiter_id=first, rolle="Lokführer")]
    asserts = [env_assert("assert_crew_assigned", zugnummer=trip.zugnummer, mitarbeiter_id=first),
               env_assert("assert_crew_assigned", assert_value=False,
                          zugnummer=trip.zugnummer, mitarbeiter_id=orig_id)]
    comm = [f"{v['verspaetung_minuten']} minuten", v["grund"], first]
    task = build_task(f"action_doppelfault__{sid(trip.zugnummer)}__{idx:03d}", ticket,
                      "ACTION+TIEFE: Verspätung + Ausfall + transienter Fehler (Retry per Policy)",
                      injections, refs, asserts, comm, ["DB", "ENV_ASSERTION", "COMMUNICATE"])
    # bakeoff lesson (0/8 on actions_pass): the ticket states the dropout as FACT, so a crew
    # check is optional efficiency, not a requirement — expected_tools must not demand it
    key = {"kind": "action", "fault": "state+runtime",
           "expected_tools": ["verspaetung", "mitarbeiter_suchen", "crew_zuweisen"],
           "oracle_calls": [oc("verspaetung", zugnummer=trip.zugnummer),   # transient failure
                            oc("verspaetung", zugnummer=trip.zugnummer),   # policy retry
                            oc("mitarbeiter_suchen", **such_args),
                            oc("crew_zuweisen", zugnummer=trip.zugnummer,
                               mitarbeiter_id=first, rolle="Lokführer")],
           "facts": {"minuten": v["verspaetung_minuten"], "ersatz_id": first,
                     "ausgefallen": orig_id, "basis": base}}
    return task, key


def t_action_umlauf_wartung(g: Gen, mp, idx, inject: bool):
    """T3: Umlauf verdict with a real consequence — schedule a repair for the SHARED vehicle,
    or do nothing if the vehicles differ. Both branches occur (mixed pool), guessing loses."""
    r = rng(g.seed, "act", "umlauf_wartung", mp.key)
    if any(o.type == "Reparatur" and o.vehicle_id in (mp.vid_a, mp.vid_b)
           for o in g.master.maintenance_orders.values()):
        return None  # absence assert must be satisfiable / exists assert specific to OUR write
    za, zb = mp.trip_a.zugnummer, mp.trip_b.zugnummer
    due = f"2026-07-{r.randint(4, 10):02d} {r.choice(['06:00', '08:00', '22:00'])}"
    ticket = (f"Für {za} wurde eine Fahrzeugstörung gemeldet. Kläre, ob {zb} heute mit demselben "
              f"Fahrzeug unterwegs ist. Wenn ja, plane für das gemeinsame Fahrzeug eine Wartung "
              f"vom Typ 'Reparatur' ein, fällig am {due}, und nenne die Fahrzeug-ID. Wenn nein, "
              f"plane nichts ein und nenne beide Fahrzeug-IDs.")
    oracle = [oc("wartung_status", kennung=za), oc("wartung_status", kennung=zb)]
    if mp.same:
        refs = [ref_action(1, "wartung_einplanen", fahrzeug_id=mp.vid_a,
                           typ="Reparatur", faellig_am=due)]
        asserts = [env_assert("assert_maintenance_exists", fahrzeug_id=mp.vid_a, typ="Reparatur")]
        comm = [mp.vid_a]
        oracle.append(oc("wartung_einplanen", fahrzeug_id=mp.vid_a,
                         typ="Reparatur", faellig_am=due))
        expected = ["wartung_status", "wartung_einplanen"]
    else:
        refs = []
        asserts = [env_assert("assert_maintenance_exists", assert_value=False,
                              fahrzeug_id=mp.vid_a, typ="Reparatur"),
                   env_assert("assert_maintenance_exists", assert_value=False,
                              fahrzeug_id=mp.vid_b, typ="Reparatur")]
        comm = [mp.vid_a, mp.vid_b]
        expected = ["wartung_status"]
    task = build_task(f"action_umlauf_wartung__{mp.key}__{idx:03d}", ticket,
                      "ACTION+TIEFE: Umlauf-Klärung mit bedingtem Write (beide Zweige real)",
                      [], refs, asserts, comm, ["DB", "ENV_ASSERTION", "COMMUNICATE"])
    key = {"kind": "action", "expected_tools": expected, "oracle_calls": oracle,
           "facts": {"same": mp.same, "vid_a": mp.vid_a, "vid_b": mp.vid_b, "faellig_am": due}}
    return task, key
