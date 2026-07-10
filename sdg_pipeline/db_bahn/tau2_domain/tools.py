"""
sdg_pipeline/db_bahn/tau2_domain/tools.py
=========================================
BahnTools: the executable tool sandbox for the `db_bahn` domain (internal Deutsche-Bahn employee
assistant). READ tools query the frozen world-state; WRITE tools mutate it (re-executed during tau2's
DB-state reward replay). Non-@is_tool `assert_*` methods back task `env_assertions`. German docstrings
become the tool schema the teacher/student sees. Lookups query the DB live so they stay correct after writes.

Wave 2 (2026-07-08): three search READ tools (zuege_suchen / mitarbeiter_suchen / wartung_liste — tasks
without pre-given ids) and business-rule validation on the WRITE tools (role/qualification/duplicate gates,
terminal maintenance status, date format). A rejected WRITE raises a German ValueError which the rollout
harness returns to the agent as a tool-error observation — that is the runtime-fault replan surprise.
Search tools require >=1 filter and page at MAX_TREFFER rows (bounded observations, deterministic order).
"""

import re
from typing import Optional

from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

from sdg_pipeline.db_bahn.tau2_domain.data_model import (
    Assignment, BahnDB, MaintenanceOrder,
)

MAX_TREFFER = 10  # search-tool page size: bounded observations, deterministic first hit
MAINT_STATUSES = ("geplant", "in_Arbeit", "abgeschlossen", "überfällig")
SEVERITIES = ("niedrig", "mittel", "hoch")
# products that carry a matching driver qualification in the seeded world; other products
# (ECE/RJ/EN/…) have no qualification counterpart -> no qualification gate for them
QUALI_PRODUKTE = ("ICE", "IC", "EC")
DUE_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")
HHMM_RE = re.compile(r"^\d{2}:\d{2}$")


class BahnTools(ToolKitBase):
    """Werkzeuge des internen DB-Mitarbeiter-Assistenten (Fahrplan, Verspätung, Standort, Wartung, Personal)."""

    db: BahnDB

    def __init__(self, db: BahnDB) -> None:
        super().__init__(db)

    # --- helpers (not tools) --------------------------------------------------------------
    def _find_trip(self, zugnummer: str):
        z = (zugnummer or "").strip().lower()
        for t in self.db.trips.values():
            if t.zugnummer.lower() == z:
                return t
        raise ValueError(f"Zug '{zugnummer}' nicht gefunden.")

    def _station_name(self, station_id: str) -> str:
        s = self.db.stations.get(station_id)
        return s.name if s else station_id

    def _trip_delay_min(self, trip_id: str):
        ds = [d for d in self.db.delays if d.trip_id == trip_id]
        if not ds:
            return 0, "", "pünktlich"
        worst = max(ds, key=lambda d: d.delay_sec)
        return worst.delay_sec // 60, worst.cause, worst.remark

    # --- READ tools -----------------------------------------------------------------------
    @is_tool(ToolType.READ)
    def fahrplan(self, zugnummer: str) -> dict:
        """
        Gibt den Fahrplan (Halte mit An-/Abfahrtszeiten) eines Zuges zurück.

        Args:
            zugnummer: Die Zugnummer, z. B. "ICE 1562".

        Returns:
            Fahrplan mit Start, Ziel und der Halteliste.
        """
        t = self._find_trip(zugnummer)
        stops = sorted([s for s in self.db.schedule if s.trip_id == t.trip_id], key=lambda s: s.seq)
        return {"zugnummer": t.zugnummer, "produkt": t.product,
                "von": self._station_name(t.origin_station), "nach": self._station_name(t.dest_station),
                "abfahrt": t.dep_time, "ankunft": t.arr_time,
                "halte": [{"station": self._station_name(s.station_id), "an": s.arr, "ab": s.dep} for s in stops]}

    @is_tool(ToolType.READ)
    def verspaetung(self, zugnummer: str) -> dict:
        """
        Gibt die aktuelle Verspätung eines Zuges in Minuten samt Grund zurück.

        Args:
            zugnummer: Die Zugnummer, z. B. "ICE 1562".

        Returns:
            Verspätung in Minuten, Grund und eine Meldung.
        """
        t = self._find_trip(zugnummer)
        mins, cause, remark = self._trip_delay_min(t.trip_id)
        return {"zugnummer": t.zugnummer, "verspaetung_minuten": mins, "grund": cause, "meldung": remark}

    @is_tool(ToolType.READ)
    def zugstandort(self, zugnummer: str) -> dict:
        """
        Gibt den aktuellen (interpolierten) Standort eines fahrenden Zuges zurück.

        Args:
            zugnummer: Die Zugnummer, z. B. "ICE 1562".

        Returns:
            Standort (Koordinaten, Fortschritt, nächster Halt) oder Status "nicht_unterwegs".
        """
        t = self._find_trip(zugnummer)
        p = next((p for p in self.db.positions if p.trip_id == t.trip_id), None)
        if p is None:
            return {"zugnummer": t.zugnummer, "status": "nicht_unterwegs"}
        return {"zugnummer": t.zugnummer, "status": "unterwegs", "lat": p.lat, "lon": p.lon,
                "fortschritt_prozent": p.progress_pct,
                "naechster_halt": self._station_name(p.next_station_id) if p.next_station_id else None,
                "quelle": p.source}

    @is_tool(ToolType.READ)
    def wartung_status(self, kennung: str) -> dict:
        """
        Gibt die Wartungsaufträge für ein Fahrzeug zurück. `kennung` darf eine Zugnummer
        (dann wird das zugeordnete Fahrzeug genommen) oder direkt eine Fahrzeug-ID sein.

        Args:
            kennung: Zugnummer (z. B. "ICE 1562") oder Fahrzeug-ID (z. B. "ICE4-9001").

        Returns:
            Fahrzeug-ID und die Liste der Wartungsaufträge.
        """
        vid = kennung.strip()
        if vid not in self.db.vehicles:
            try:
                vid = self._find_trip(kennung).vehicle_id
            except ValueError:
                raise ValueError(f"Weder Zug noch Fahrzeug '{kennung}' gefunden.")
        orders = [o.model_dump() for o in self.db.maintenance_orders.values() if o.vehicle_id == vid]
        return {"fahrzeug_id": vid, "wartungsauftraege": orders}

    @is_tool(ToolType.READ)
    def mitarbeiter_info(self, zugnummer: str) -> dict:
        """
        Gibt die einem Zug zugeteilte Besatzung (Mitarbeiter + Rollen) zurück.

        Args:
            zugnummer: Die Zugnummer, z. B. "ICE 1562".

        Returns:
            Zugnummer und die Liste der zugeteilten Mitarbeiter.
        """
        t = self._find_trip(zugnummer)
        crew = []
        for a in self.db.assignments:
            if a.trip_id == t.trip_id:
                e = self.db.employees.get(a.emp_id)
                crew.append({"mitarbeiter_id": a.emp_id, "name": e.name if e else a.emp_id,
                             "rolle": a.role, "heimatbasis": e.home_base if e else None})
        return {"zugnummer": t.zugnummer, "besatzung": crew}

    # --- READ search tools (wave 2: tasks WITHOUT pre-given ids) ---------------------------
    def _page(self, rows: list[dict]) -> dict:
        out = {"treffer_gesamt": len(rows), "treffer": rows[:MAX_TREFFER]}
        if len(rows) > MAX_TREFFER:
            out["hinweis"] = f"Nur die ersten {MAX_TREFFER} Treffer angezeigt — Filter enger setzen."
        return out

    @is_tool(ToolType.READ)
    def zuege_suchen(self, von: Optional[str] = None, nach: Optional[str] = None,
                     produkt: Optional[str] = None,
                     min_verspaetung_minuten: Optional[int] = None) -> dict:
        """
        Sucht Züge nach Startbahnhof, Zielbahnhof, Zuggattung und/oder Mindestverspätung.
        Mindestens ein Filter ist erforderlich. Liefert höchstens 10 Treffer (nach Abfahrtszeit
        sortiert); die Zeilen enthalten keine Verspätung — dafür `verspaetung(zugnummer)` aufrufen.

        Args:
            von: Name des Startbahnhofs oder ein Teil davon, z. B. "Berlin Hbf".
            nach: Name des Zielbahnhofs oder ein Teil davon.
            produkt: Zuggattung, z. B. "ICE", "IC", "EC".
            min_verspaetung_minuten: Nur Züge mit mindestens dieser aktuellen Verspätung in Minuten.

        Returns:
            treffer_gesamt und die Liste der Züge (zugnummer, produkt, von, nach, abfahrt).
        """
        if von is None and nach is None and produkt is None and min_verspaetung_minuten is None:
            raise ValueError("Bitte mindestens einen Filter angeben (von, nach, produkt, min_verspaetung_minuten).")
        if min_verspaetung_minuten is not None:
            try:
                min_verspaetung_minuten = int(min_verspaetung_minuten)
            except (TypeError, ValueError):
                raise ValueError("min_verspaetung_minuten muss eine ganze Zahl (Minuten) sein.")
        max_delay: dict[str, int] = {}
        if min_verspaetung_minuten is not None:  # one pass over delays, not trips x delays
            for d in self.db.delays:
                if d.delay_sec > max_delay.get(d.trip_id, -1):
                    max_delay[d.trip_id] = d.delay_sec
        v = (von or "").strip().lower()
        n = (nach or "").strip().lower()
        p = (produkt or "").strip().lower()
        rows = []
        for t in self.db.trips.values():
            if v and v not in self._station_name(t.origin_station).lower():
                continue
            if n and n not in self._station_name(t.dest_station).lower():
                continue
            if p and t.product.lower() != p:
                continue
            if min_verspaetung_minuten is not None and \
                    max_delay.get(t.trip_id, 0) // 60 < min_verspaetung_minuten:
                continue
            rows.append({"zugnummer": t.zugnummer, "produkt": t.product,
                         "von": self._station_name(t.origin_station),
                         "nach": self._station_name(t.dest_station), "abfahrt": t.dep_time})
        rows.sort(key=lambda r: (r["abfahrt"], r["zugnummer"]))
        return self._page(rows)

    @is_tool(ToolType.READ)
    def mitarbeiter_suchen(self, rolle: Optional[str] = None, heimatbasis: Optional[str] = None,
                           qualifikation: Optional[str] = None,
                           verfuegbar_um: Optional[str] = None) -> dict:
        """
        Sucht Mitarbeiter nach Rolle, Heimatbasis, Qualifikation und/oder Schichtverfügbarkeit.
        Mindestens ein Filter ist erforderlich. Liefert höchstens 10 Treffer, aufsteigend nach
        Mitarbeiter-ID sortiert (der erste Treffer ist die kleinste ID).

        Args:
            rolle: z. B. "Lokführer", "Zugbegleiter", "Techniker", "Disponent".
            heimatbasis: Name der Heimatbasis oder ein Teil davon.
            qualifikation: Erforderliche Qualifikation, z. B. "ICE", "IC", "EC", "Nacht", "Gefahrgut".
            verfuegbar_um: Uhrzeit "HH:MM" — nur Mitarbeiter, deren Schicht diese Zeit abdeckt.

        Returns:
            treffer_gesamt und die Liste (mitarbeiter_id, name, rolle, heimatbasis, qualifikationen, schicht).
        """
        if rolle is None and heimatbasis is None and qualifikation is None and verfuegbar_um is None:
            raise ValueError("Bitte mindestens einen Filter angeben (rolle, heimatbasis, qualifikation, verfuegbar_um).")
        if verfuegbar_um is not None:
            verfuegbar_um = verfuegbar_um.strip()
            if not HHMM_RE.match(verfuegbar_um):
                raise ValueError('verfuegbar_um muss das Format "HH:MM" haben, z. B. "06:30".')
        shift_of = {}
        for s in self.db.shifts:
            shift_of.setdefault(s.emp_id, s)
        ro = (rolle or "").strip().lower()
        hb = (heimatbasis or "").strip().lower()
        q = (qualifikation or "").strip().lower()
        rows = []
        for emp_id in sorted(self.db.employees):
            e = self.db.employees[emp_id]
            if ro and e.role.lower() != ro:
                continue
            if hb and hb not in e.home_base.lower():
                continue
            if q and q not in (x.lower() for x in e.qualifications):
                continue
            sh = shift_of.get(emp_id)
            if verfuegbar_um is not None and not (sh and sh.start <= verfuegbar_um <= sh.end):
                continue
            rows.append({"mitarbeiter_id": e.emp_id, "name": e.name, "rolle": e.role,
                         "heimatbasis": e.home_base, "qualifikationen": e.qualifications,
                         "schicht": f"{sh.start}–{sh.end}" if sh else None})
        return self._page(rows)

    @is_tool(ToolType.READ)
    def mitarbeiter_details(self, mitarbeiter_id: str) -> dict:
        """
        Gibt die Stammdaten EINES Mitarbeiters per ID zurück (Rolle, Heimatbasis, Qualifikationen, Schicht).
        Nutze das, um eine BEKANNTE Person zu prüfen (z. B. eine ID aus dem Auftrag) — statt sie über
        mitarbeiter_suchen zu erraten (dessen Trefferliste ist auf 10 gekürzt und eignet sich nicht zum
        Verifizieren einer bestimmten ID).

        Args:
            mitarbeiter_id: Die Mitarbeiter-ID, z. B. "MA-4471".

        Returns:
            Stammdaten (mitarbeiter_id, name, rolle, heimatbasis, qualifikationen, schicht).
        """
        e = self.db.employees.get(mitarbeiter_id.strip())
        if e is None:
            raise ValueError(f"Mitarbeiter '{mitarbeiter_id}' nicht gefunden.")
        sh = next((s for s in self.db.shifts if s.emp_id == e.emp_id), None)
        return {"mitarbeiter_id": e.emp_id, "name": e.name, "rolle": e.role,
                "heimatbasis": e.home_base, "qualifikationen": e.qualifications,
                "schicht": f"{sh.start}–{sh.end}" if sh else None}

    @is_tool(ToolType.READ)
    def wartung_liste(self, status: Optional[str] = None, depot: Optional[str] = None,
                      faellig_vor: Optional[str] = None, schweregrad: Optional[str] = None) -> dict:
        """
        Sucht Wartungsaufträge flottenweit nach Status, Depot, Fälligkeit und/oder Schweregrad.
        Mindestens ein Filter ist erforderlich. Liefert höchstens 10 Treffer, die dringendsten
        (früheste Fälligkeit) zuerst.

        Args:
            status: "geplant", "in_Arbeit", "abgeschlossen" oder "überfällig".
            depot: Depotname oder ein Teil davon.
            faellig_vor: Nur Aufträge fällig vor diesem Zeitpunkt ("YYYY-MM-DD" oder "YYYY-MM-DD HH:MM").
            schweregrad: "niedrig", "mittel" oder "hoch".

        Returns:
            treffer_gesamt und die Liste der Aufträge (order_id, vehicle_id, type, status, depot, due_at, severity).
        """
        if status is None and depot is None and faellig_vor is None and schweregrad is None:
            raise ValueError("Bitte mindestens einen Filter angeben (status, depot, faellig_vor, schweregrad).")
        if status is not None and status not in MAINT_STATUSES:
            raise ValueError(f"Ungültiger Status '{status}'. Erlaubt: {', '.join(MAINT_STATUSES)}.")
        if schweregrad is not None and schweregrad not in SEVERITIES:
            raise ValueError(f"Ungültiger Schweregrad '{schweregrad}'. Erlaubt: {', '.join(SEVERITIES)}.")
        dp = (depot or "").strip().lower()
        rows = []
        for o in self.db.maintenance_orders.values():
            if status is not None and o.status != status:
                continue
            if dp and dp not in o.depot.lower():
                continue
            if faellig_vor is not None and not o.due_at < faellig_vor.strip():
                continue
            if schweregrad is not None and o.severity != schweregrad:
                continue
            rows.append(o.model_dump())
        rows.sort(key=lambda r: (r["due_at"], r["order_id"]))
        return self._page(rows)

    # --- WRITE tools (mutate state -> re-executed during DB-state reward replay) -----------
    @is_tool(ToolType.WRITE)
    def wartung_einplanen(self, fahrzeug_id: str, typ: str, faellig_am: str, depot: Optional[str] = None) -> dict:
        """
        Legt einen neuen Wartungsauftrag für ein Fahrzeug an.

        Args:
            fahrzeug_id: Die Fahrzeug-ID, z. B. "ICE4-9001".
            typ: Art der Wartung (z. B. "Inspektion", "Reparatur", "Radsatztausch").
            faellig_am: Fälligkeitszeitpunkt als "YYYY-MM-DD HH:MM".
            depot: Optionales Depot; sonst das Heimatdepot des Fahrzeugs.

        Returns:
            Den angelegten Wartungsauftrag.
        """
        v = self.db.vehicles.get(fahrzeug_id.strip())
        if v is None:
            raise ValueError(f"Fahrzeug '{fahrzeug_id}' nicht gefunden.")
        if not DUE_AT_RE.match((faellig_am or "").strip()):
            raise ValueError('Ungültiges Format für faellig_am. Erwartet "YYYY-MM-DD HH:MM", z. B. "2026-07-05 06:00".')
        if depot is not None:
            depots = {veh.home_depot for veh in self.db.vehicles.values()}
            if depot.strip() not in depots:
                raise ValueError(f"Depot '{depot}' nicht gefunden.")
        faellig_am = faellig_am.strip()
        oid = f"WO-2026-{1000 + len(self.db.maintenance_orders)}"
        order = MaintenanceOrder(order_id=oid, vehicle_id=v.vehicle_id, type=typ, status="geplant",
                                 depot=depot or v.home_depot, due_at=faellig_am, severity="mittel")
        self.db.maintenance_orders[oid] = order
        return order.model_dump()

    @is_tool(ToolType.WRITE)
    def crew_zuweisen(self, zugnummer: str, mitarbeiter_id: str, rolle: str) -> dict:
        """
        Teilt einen Mitarbeiter einem Zug in einer Rolle zu. Wird abgelehnt, wenn der Mitarbeiter
        dem Zug bereits zugeteilt ist, oder wenn für die Rolle "Lokführer" die Mitarbeiter-Rolle
        bzw. die zur Zuggattung passende Qualifikation (ICE/IC/EC) fehlt.

        Args:
            zugnummer: Die Zugnummer, z. B. "ICE 1562".
            mitarbeiter_id: Die Mitarbeiter-ID, z. B. "MA-4471".
            rolle: Die Rolle (z. B. "Lokführer", "Zugbegleiter").

        Returns:
            Die angelegte Zuteilung.
        """
        t = self._find_trip(zugnummer)
        emp_id = mitarbeiter_id.strip()
        e = self.db.employees.get(emp_id)
        if e is None:
            raise ValueError(f"Mitarbeiter '{mitarbeiter_id}' nicht gefunden.")
        if any(a.trip_id == t.trip_id and a.emp_id == emp_id for a in self.db.assignments):
            raise ValueError(f"Zuweisung abgelehnt: {e.name} ({emp_id}) ist {t.zugnummer} bereits zugeteilt.")
        if rolle.strip() == "Lokführer":
            if e.role != "Lokführer":
                raise ValueError(f"Zuweisung abgelehnt: {e.name} ({emp_id}) ist nicht als Lokführer "
                                 f"qualifiziert (Rolle: {e.role}).")
            if t.product in QUALI_PRODUKTE and t.product not in e.qualifications:
                raise ValueError(f"Zuweisung abgelehnt: {e.name} ({emp_id}) fehlt die Qualifikation "
                                 f"{t.product} für {t.zugnummer}.")
        aid = f"AS-{len(self.db.assignments)}"
        a = Assignment(assignment_id=aid, trip_id=t.trip_id, emp_id=emp_id, role=rolle)
        self.db.assignments.append(a)
        return a.model_dump()

    @is_tool(ToolType.WRITE)
    def wartung_status_setzen(self, auftrag_id: str, status: str) -> dict:
        """
        Setzt den Status eines Wartungsauftrags. "abgeschlossen" ist ein Endstatus —
        abgeschlossene Aufträge können nicht mehr geändert werden.

        Args:
            auftrag_id: Die Auftrags-ID, z. B. "WO-2026-1000".
            status: Neuer Status ("geplant", "in_Arbeit", "abgeschlossen", "überfällig").

        Returns:
            Den aktualisierten Wartungsauftrag.
        """
        o = self.db.maintenance_orders.get(auftrag_id.strip())
        if o is None:
            raise ValueError(f"Wartungsauftrag '{auftrag_id}' nicht gefunden.")
        if status not in MAINT_STATUSES:
            raise ValueError(f"Ungültiger Status '{status}'. Erlaubt: {', '.join(MAINT_STATUSES)}.")
        if o.status == "abgeschlossen":
            raise ValueError(f"Statuswechsel abgelehnt: {o.order_id} ist bereits abgeschlossen (Endstatus).")
        o.status = status
        return o.model_dump()

    # --- fault-injection (NOT tools; only callable via task initialization_actions) --------
    def inject_verspaetung(self, zugnummer: str, minuten: int, grund: str) -> bool:
        """Setzt für alle Halte eines Zuges eine Verspätung (Überraschung für Replan-Tasks)."""
        t = self._find_trip(zugnummer)
        found = False
        for d in self.db.delays:
            if d.trip_id == t.trip_id:
                d.delay_sec = minuten * 60
                d.cause = grund
                d.remark = f"+{minuten} Min: {grund}"
                found = True
        return found

    def inject_lokfuehrer_ausfall(self, zugnummer: str) -> bool:
        """Entfernt die Lokführer-Zuteilung eines Zuges (kurzfristiger Ausfall)."""
        t = self._find_trip(zugnummer)
        before = len(self.db.assignments)
        self.db.assignments = [a for a in self.db.assignments
                               if not (a.trip_id == t.trip_id and a.role == "Lokführer")]
        return len(self.db.assignments) < before

    # --- assertions for task env_assertions (not tools) -----------------------------------
    def assert_maintenance_exists(self, fahrzeug_id: str, typ: str) -> bool:
        """True wenn für das Fahrzeug ein Wartungsauftrag des Typs existiert."""
        return any(o.vehicle_id == fahrzeug_id and o.type == typ for o in self.db.maintenance_orders.values())

    def assert_crew_assigned(self, zugnummer: str, mitarbeiter_id: str) -> bool:
        """True wenn der Mitarbeiter dem Zug zugeteilt ist."""
        try:
            t = self._find_trip(zugnummer)
        except ValueError:
            return False
        return any(a.trip_id == t.trip_id and a.emp_id == mitarbeiter_id for a in self.db.assignments)

    def assert_maintenance_status(self, auftrag_id: str, status: str) -> bool:
        """True wenn der Wartungsauftrag den erwarteten Status hat."""
        o = self.db.maintenance_orders.get(auftrag_id)
        return o is not None and o.status == status
