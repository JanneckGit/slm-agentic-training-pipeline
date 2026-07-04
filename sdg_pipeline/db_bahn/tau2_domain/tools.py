"""
sdg_pipeline/db_bahn/tau2_domain/tools.py
=========================================
BahnTools: the executable tool sandbox for the `db_bahn` domain (internal Deutsche-Bahn employee
assistant). READ tools query the frozen world-state; WRITE tools mutate it (re-executed during tau2's
DB-state reward replay). Non-@is_tool `assert_*` methods back task `env_assertions`. German docstrings
become the tool schema the teacher/student sees. Lookups query the DB live so they stay correct after writes.
"""

from typing import Optional

from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

from sdg_pipeline.db_bahn.tau2_domain.data_model import (
    Assignment, BahnDB, MaintenanceOrder,
)


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
        oid = f"WO-2026-{1000 + len(self.db.maintenance_orders)}"
        order = MaintenanceOrder(order_id=oid, vehicle_id=v.vehicle_id, type=typ, status="geplant",
                                 depot=depot or v.home_depot, due_at=faellig_am, severity="mittel")
        self.db.maintenance_orders[oid] = order
        return order.model_dump()

    @is_tool(ToolType.WRITE)
    def crew_zuweisen(self, zugnummer: str, mitarbeiter_id: str, rolle: str) -> dict:
        """
        Teilt einen Mitarbeiter einem Zug in einer Rolle zu.

        Args:
            zugnummer: Die Zugnummer, z. B. "ICE 1562".
            mitarbeiter_id: Die Mitarbeiter-ID, z. B. "MA-4471".
            rolle: Die Rolle (z. B. "Lokführer", "Zugbegleiter").

        Returns:
            Die angelegte Zuteilung.
        """
        t = self._find_trip(zugnummer)
        if mitarbeiter_id.strip() not in self.db.employees:
            raise ValueError(f"Mitarbeiter '{mitarbeiter_id}' nicht gefunden.")
        aid = f"AS-{len(self.db.assignments)}"
        a = Assignment(assignment_id=aid, trip_id=t.trip_id, emp_id=mitarbeiter_id.strip(), role=rolle)
        self.db.assignments.append(a)
        return a.model_dump()

    @is_tool(ToolType.WRITE)
    def wartung_status_setzen(self, auftrag_id: str, status: str) -> dict:
        """
        Setzt den Status eines Wartungsauftrags.

        Args:
            auftrag_id: Die Auftrags-ID, z. B. "WO-2026-1000".
            status: Neuer Status ("geplant", "in_Arbeit", "abgeschlossen", "überfällig").

        Returns:
            Den aktualisierten Wartungsauftrag.
        """
        o = self.db.maintenance_orders.get(auftrag_id.strip())
        if o is None:
            raise ValueError(f"Wartungsauftrag '{auftrag_id}' nicht gefunden.")
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
