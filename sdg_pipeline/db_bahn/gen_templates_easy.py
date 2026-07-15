"""
sdg_pipeline/db_bahn/gen_templates_easy.py
===========================================
Easy/mid-tier task templates (1-2 calls; polished wave-1 set regenerated under the
11-tool domain). Split out of gen_tasks.py 2026-07-15 — registry + main() live there.
"""

from sdg_pipeline.db_bahn.seed_worldstate import rng
from sdg_pipeline.db_bahn.gen_tasks_lib import (
    DELAY_CAUSES, MAINT_TYPES, Gen,
    _apply, _grund_comm, build_task, env_assert, inj_call, oc, qual_for, ref_action, sid)
from sdg_pipeline.db_bahn.tau2_domain.tools import BahnTools

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


def t_info_mitarbeiter(g: Gen, emp, idx, inject: bool):
    """wave-2.5 (A1): lookup-by-ID gold path — the ticket NAMES the emp_id, the correct move is
    mitarbeiter_details (not a blind mitarbeiter_suchen through a 10-row page). Guarantees the
    student sees clean demonstrations of tool #12. comm uses id/role/base/quals — never the
    (ambiguous) name, and never the shift string (en-dash reproduction risk)."""
    tk = g.fresh()
    d = tk.mitarbeiter_details(emp.emp_id)
    comm = [d["mitarbeiter_id"], d["rolle"], d["heimatbasis"]] + list(d["qualifikationen"])
    ticket = (f"Welche Rolle und welche Qualifikationen hat der Mitarbeiter {emp.emp_id}, und was ist "
              f"seine Heimatbasis? Nenne Mitarbeiter-ID, Rolle, Heimatbasis und alle Qualifikationen.")
    task = build_task(f"info_mitarbeiter__{sid(emp.emp_id)}__{idx:03d}", ticket,
                      "INFO: Stammdaten eines BEKANNTEN Mitarbeiters per ID nachschlagen (Lookup statt Suche)",
                      [], None, None, comm, ["COMMUNICATE"])
    key = {"kind": "info", "expected_tools": ["mitarbeiter_details"],
           "oracle_calls": [oc("mitarbeiter_details", mitarbeiter_id=emp.emp_id)],
           "facts": {"details": d}}
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

