"""
sdg_pipeline/db_bahn/seed_worldstate.py
=======================================
Phase 0 of Plan (B): build a DETERMINISTIC, frozen "Deutsche Bahn world-state" from the
real gtfs.de `de_fv` snapshot + sha256-seeded synthetic generators, and write it as a single
`db.json` (the tau2 `db_bahn` domain loads this into `BahnDB`).

Real tables (from de_fv): stations, lines, trips (+ synthesized Zugnummer), schedule, calendar.
Synthetic, seeded: delays, positions, vehicles, maintenance_orders, employees, shifts, assignments.

Determinism (Plan P0-4/P1-4): every random draw comes from `rng(*keys)` = a random.Random seeded by
sha256("SEED|table|pk") — NEVER Python's builtin hash() (which is per-process salted). The sim-clock is
frozen (SIM_DATE + SIM_NOW): all time-derived values (delays, positions) are precomputed as columns; tools
only read them. Same GTFS snapshot + same --seed ⇒ byte-identical db.json.

Standalone (no tau2 import) so it runs under any Python (sdg image or host).

Usage:
    python sdg_pipeline/db_bahn/seed_worldstate.py \
        --gtfs-dir data/raw/db_sandbox/gtfs_de_fv \
        --out data/raw/db_sandbox/db.json --seed 42

Attribution: schedule/station data © gtfs.de / DELFI e.V., licensed under CC-BY-4.0. Synthetic tables
(delays, positions, vehicles, maintenance, employees, shifts, assignments) are MOCK and clearly labeled.
"""

import argparse
import csv
import hashlib
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

# --- frozen sim-clock (Plan decision, in the de_fv 7-day window) --------------------------
SIM_DATE = "2026-06-29"          # Monday, inside the snapshot's validity window
SIM_NOW = "12:00:00"             # the frozen "now" for positions / running trips
_WEEKDAY_COLS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

DEPOTS = ["München-Pasing", "Berlin Rummelsburg", "Frankfurt Griesheim", "Hamburg-Eidelstedt",
          "Köln Betriebsbahnhof", "Dortmund Bbf", "Leipzig Hbf Süd", "Stuttgart Rosenstein",
          "Nürnberg West", "Hannover Leinhausen",
          # wave-2.5 world enlargement: 10 more depots -> smaller per-depot buckets keep the
          # 1-3-hit filter-combo windows alive at ~4x orders. Names pairwise non-substring
          # (mitarbeiter_suchen/wartung_liste match depot filters as substrings).
          "Dresden-Friedrichstadt", "Bremen Sebaldsbrück", "Karlsruhe Rheinbrücke",
          "Mannheim Rangierbahnhof", "Duisburg-Wedau", "Rostock Seehafen", "Erfurt Nord",
          "Saarbrücken Burbach", "Kiel Meimersdorf", "Augsburg Oberhausen"]
VEHICLE_TYPES = {"ICE": ["ICE 1", "ICE 3", "ICE 4", "ICE 3neo"], "ICE 42": ["ICE 4"],
                 "IC": ["IC 2", "IC 1"], "EC": ["EC"], "ECE": ["ICE 3neo"],
                 "RJ": ["railjet"], "EN": ["Nightjet"]}
ROLES = ["Lokführer", "Zugbegleiter", "Disponent", "Techniker"]
FIRST = ["Anna", "Ben", "Clara", "David", "Eva", "Felix", "Greta", "Hans", "Ines", "Jonas",
         "Katrin", "Lars", "Mia", "Noah", "Olga", "Paul", "Rita", "Sven", "Tina", "Uwe"]
LAST = ["Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker",
        "Hoffmann", "Schäfer", "Koch", "Bauer", "Richter", "Klein", "Wolf", "Schröder"]
MAINT_TYPES = ["Inspektion", "Reparatur", "Reinigung", "Radsatztausch", "Softwareupdate"]
MAINT_STATUS = ["geplant", "in_Arbeit", "abgeschlossen", "überfällig"]
DELAY_CAUSES = ["Bauarbeiten", "Signalstörung", "Verspätung eines vorausfahrenden Zuges",
                "Warten auf Anschlussreisende", "technische Störung am Zug", "Notarzteinsatz"]


def rng(*keys) -> random.Random:
    """Deterministic per-entity RNG: sha256('SEED|k1|k2|...') -> random.Random. NEVER builtin hash()."""
    h = hashlib.sha256("|".join(str(k) for k in keys).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def hhmm_add(t: str, seconds: int) -> str:
    """Add seconds to a GTFS HH:MM:SS (may exceed 24h); return HH:MM."""
    h, m, s = (int(x) for x in t.split(":"))
    total = h * 3600 + m * 60 + s + seconds
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}"


# --- calendar: which services run on SIM_DATE ---------------------------------------------
def active_services(gtfs: Path, sim: date) -> set[str]:
    wd = _WEEKDAY_COLS[sim.weekday()]
    ymd = sim.strftime("%Y%m%d")
    active = set()
    for row in read_csv(gtfs / "calendar.txt"):
        if row.get(wd) == "1" and row["start_date"] <= ymd <= row["end_date"]:
            active.add(row["service_id"])
    cd = gtfs / "calendar_dates.txt"
    if cd.exists():
        for row in read_csv(cd):
            if row["date"] != ymd:
                continue
            if row["exception_type"] == "1":
                active.add(row["service_id"])
            elif row["exception_type"] == "2":
                active.discard(row["service_id"])
    return active


def build_real(gtfs: Path, seed: int, sim: date) -> dict:
    # stations = parent stations (location_type == "1"); real names + coords
    stations, stop_to_station = {}, {}
    for row in read_csv(gtfs / "stops.txt"):
        sid = row["stop_id"]
        parent = row.get("parent_station") or ""
        stop_to_station[sid] = parent if parent else sid
        if row.get("location_type") == "1":
            stations[sid] = {"station_id": sid, "name": row["stop_name"],
                             "lat": float(row["stop_lat"]), "lon": float(row["stop_lon"]),
                             "eva": None}
    # map platform-children coords onto their parent if the parent lacked a row
    for row in read_csv(gtfs / "stops.txt"):
        p = stop_to_station.get(row["stop_id"])
        if p and p not in stations and row.get("stop_lat"):
            stations[p] = {"station_id": p, "name": row["stop_name"],
                           "lat": float(row["stop_lat"]), "lon": float(row["stop_lon"]), "eva": None}

    lines = {r["route_id"]: {"line_id": r["route_id"], "product": r["route_short_name"].split()[0],
                             "short_name": r["route_short_name"]}
             for r in read_csv(gtfs / "routes.txt")}

    active = active_services(gtfs, sim)
    trips = {}
    for r in read_csv(gtfs / "trips.txt"):
        if r["service_id"] not in active:
            continue
        trips[r["trip_id"]] = {"trip_id": r["trip_id"], "line_id": r["route_id"],
                               "service_id": r["service_id"]}
    # schedule (stop_times) for active trips only, mapped to parent stations
    schedule = []
    for r in read_csv(gtfs / "stop_times.txt"):
        tid = r["trip_id"]
        if tid not in trips:
            continue
        st = stop_to_station.get(r["stop_id"], r["stop_id"])
        schedule.append({"trip_id": tid, "seq": int(r["stop_sequence"]), "station_id": st,
                         "arr": (r["arrival_time"] or "")[:5], "dep": (r["departure_time"] or "")[:5],
                         "headsign": r.get("stop_headsign", "")})
    schedule.sort(key=lambda x: (x["trip_id"], x["seq"]))

    # synthesize a deterministic Zugnummer + origin/dest/headsign per trip
    by_trip = {}
    for s in schedule:
        by_trip.setdefault(s["trip_id"], []).append(s)
    used_numbers = set()
    for tid, stops in by_trip.items():
        prod = lines.get(trips[tid]["line_id"], {}).get("product", "IC")
        r = rng(seed, "zugnr", tid)
        base = {"ICE": 1000, "ECE": 1000, "IC": 2000, "EC": 100, "RJ": 60, "EN": 400}.get(prod, 900)
        num = base + r.randint(0, 899)
        while (prod, num) in used_numbers:
            num = base + r.randint(0, 899)
        used_numbers.add((prod, num))
        trips[tid].update({
            "zugnummer": f"{prod} {num}", "product": prod,
            "origin_station": stops[0]["station_id"], "dest_station": stops[-1]["station_id"],
            "headsign": stops[-1].get("headsign") or stations.get(stops[-1]["station_id"], {}).get("name", ""),
            "dep_time": stops[0]["dep"], "arr_time": stops[-1]["arr"],
        })
    return {"stations": list(stations.values()), "lines": list(lines.values()),
            "trips": list(trips.values()), "schedule": schedule, "_by_trip": by_trip}


# --- synthetic seeded tables --------------------------------------------------------------
def build_synthetic(real: dict, seed: int, sim: date) -> dict:
    trips, by_trip, stations = real["trips"], real["_by_trip"], {s["station_id"]: s for s in real["stations"]}
    now_sec = sum(int(x) * f for x, f in zip(SIM_NOW.split(":"), (3600, 60, 1)))

    # vehicles: minted per trip into the product-family pool, then assigned from the pool
    # (wave-2.5: mint prob 0.5 -> 1.0 = one minted vehicle per trip; assignment still draws
    # from the whole pool, so ~half stay trip-less "reserve fleet" -> more order-pattern
    # vehicles for the maintenance templates + realistic depot stock for wartung_liste)
    vehicles, veh_pool = [], {}
    for t in trips:
        prod = t["product"]
        pool = veh_pool.setdefault(prod, [])
        r = rng(seed, "veh", t["trip_id"])
        if not pool or r.random() < 1.0:  # always grow the pool (one vehicle minted per trip)
            vtype = r.choice(VEHICLE_TYPES.get(t.get("short_name", prod), VEHICLE_TYPES.get(prod, ["IC 2"])))
            vid = f"{vtype.replace(' ', '')}-{9000 + len(vehicles)}"
            vehicles.append({"vehicle_id": vid, "type": vtype, "capacity": r.choice([250, 380, 460, 830]),
                             "home_depot": r.choice(DEPOTS)})
            pool.append(vid)
        t["vehicle_id"] = rng(seed, "vehsel", t["trip_id"]).choice(pool)

    # delays (frozen at sim clock): per active trip, a head delay propagated down the stops
    delays = []
    for tid, stops in by_trip.items():
        r = rng(seed, "delay", tid)
        roll = r.random()
        head = 0 if roll < 0.70 else (r.randint(120, 900) if roll < 0.95 else r.randint(900, 5400))
        cause = "" if head == 0 else r.choice(DELAY_CAUSES)
        for s in stops:
            jitter = 0 if head == 0 else r.randint(-60, 120)
            d = max(0, head + jitter)
            delays.append({"trip_id": tid, "seq": s["seq"], "delay_sec": d,
                           "cause": cause, "remark": (f"+{d // 60} Min: {cause}" if d else "pünktlich")})

    # positions: only trips en route at SIM_NOW; interpolate along stop sequence (no shapes.txt)
    def to_sec(hhmm):
        if not hhmm or ":" not in hhmm:
            return None
        h, m = int(hhmm[:2]), int(hhmm[3:5])
        return h * 3600 + m * 60

    positions = []
    for tid, stops in by_trip.items():
        segs = [(s, stations.get(s["station_id"])) for s in stops if stations.get(s["station_id"])]
        if len(segs) < 2:
            continue
        dep0, arrN = to_sec(segs[0][0]["dep"]), to_sec(segs[-1][0]["arr"])
        if dep0 is None or arrN is None or not (dep0 <= now_sec <= arrN):
            continue
        cur = next((i for i in range(len(segs) - 1)
                    if (to_sec(segs[i][0]["dep"]) or 0) <= now_sec <= (to_sec(segs[i + 1][0]["arr"]) or 0)), 0)
        a, b = segs[cur][1], segs[cur + 1][1]
        t0 = to_sec(segs[cur][0]["dep"]) or now_sec
        t1 = to_sec(segs[cur + 1][0]["arr"]) or (t0 + 1)
        frac = max(0.0, min(1.0, (now_sec - t0) / max(1, t1 - t0)))
        positions.append({"trip_id": tid, "lat": round(a["lat"] + frac * (b["lat"] - a["lat"]), 5),
                          "lon": round(a["lon"] + frac * (b["lon"] - a["lon"]), 5),
                          "progress_pct": round(100 * (cur + frac) / (len(segs) - 1), 1),
                          "next_station_id": b["station_id"], "at_station": None,
                          "source": "interpolated_from_schedule (mock)"})

    # maintenance orders: every vehicle carries 1-3 (wave-2.5: was [0,0,1,1,2] on ~548 vehicles;
    # ~1070 vehicles x mean 1.8 orders feed the pattern pools ueberfaellig/konflikt/2open)
    maintenance = []
    for i, v in enumerate(vehicles):
        r = rng(seed, "maint", v["vehicle_id"])
        for _ in range(r.choice([1, 1, 2, 2, 3])):
            due = datetime.combine(sim, datetime.min.time()) + timedelta(hours=r.randint(-48, 120))
            maintenance.append({
                "order_id": f"WO-{2026}-{len(maintenance) + 1000}", "vehicle_id": v["vehicle_id"],
                "type": r.choice(MAINT_TYPES), "status": r.choice(MAINT_STATUS),
                "depot": v["home_depot"], "due_at": due.strftime("%Y-%m-%d %H:%M"),
                "severity": r.choice(["niedrig", "mittel", "hoch"])})

    # employees + shifts
    employees, shifts = [], []
    n_emp = max(200, len(trips) * 2)
    for i in range(n_emp):
        r = rng(seed, "emp", i)
        employees.append({"emp_id": f"MA-{4000 + i}", "name": f"{r.choice(FIRST)} {r.choice(LAST)}",
                          "role": r.choice(ROLES), "home_base": r.choice(DEPOTS),
                          "qualifications": r.sample(["ICE", "IC", "EC", "Nacht", "Gefahrgut"], k=r.randint(1, 3))})
        start = 4 + r.randint(0, 14)
        shifts.append({"shift_id": f"SH-{i}", "emp_id": f"MA-{4000 + i}",
                       "start": f"{start:02d}:00", "end": f"{min(23, start + 8):02d}:00",
                       "base": employees[-1]["home_base"]})

    # assignments: 1 Lokführer + 1-2 Zugbegleiter per trip (seeded from the employee pool)
    lok = [e["emp_id"] for e in employees if e["role"] == "Lokführer"] or [e["emp_id"] for e in employees]
    zub = [e["emp_id"] for e in employees if e["role"] == "Zugbegleiter"] or lok
    assignments = []
    for t in trips:
        r = rng(seed, "assign", t["trip_id"])
        crew = [(r.choice(lok), "Lokführer")] + [(r.choice(zub), "Zugbegleiter") for _ in range(r.randint(1, 2))]
        for emp, role in crew:
            assignments.append({"assignment_id": f"AS-{len(assignments)}", "trip_id": t["trip_id"],
                                "emp_id": emp, "role": role})

    return {"vehicles": vehicles, "delays": delays, "positions": positions,
            "maintenance_orders": maintenance, "employees": employees, "shifts": shifts,
            "assignments": assignments}


def main():
    ap = argparse.ArgumentParser(description="Seed the frozen db_bahn world-state (db.json)")
    ap.add_argument("--gtfs-dir", default="data/raw/db_sandbox/gtfs_de_fv")
    ap.add_argument("--out", default="data/raw/db_sandbox/db.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sim-date", default=SIM_DATE)
    args = ap.parse_args()

    gtfs = Path(args.gtfs_dir)
    sim = date.fromisoformat(args.sim_date)
    real = build_real(gtfs, args.seed, sim)
    synth = build_synthetic(real, args.seed, sim)
    real.pop("_by_trip", None)

    def keyed(rows, k):
        return {r[k]: r for r in rows}

    # entity tables -> dict keyed by pk (tau2 DB idiom + O(1) lookup); relational tables -> list
    db = {"meta": {"seed": args.seed, "sim_date": args.sim_date, "sim_now": SIM_NOW,
                   "source": "gtfs.de de_fv (CC-BY-4.0); synthetic tables are MOCK",
                   "attribution": "Schedule/station data © gtfs.de / DELFI e.V., CC-BY-4.0"},
          "stations": keyed(real["stations"], "station_id"),
          "lines": keyed(real["lines"], "line_id"),
          "trips": keyed(real["trips"], "trip_id"),
          "vehicles": keyed(synth["vehicles"], "vehicle_id"),
          "maintenance_orders": keyed(synth["maintenance_orders"], "order_id"),
          "employees": keyed(synth["employees"], "emp_id"),
          "schedule": real["schedule"], "delays": synth["delays"],
          "positions": synth["positions"], "shifts": synth["shifts"],
          "assignments": synth["assignments"]}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(db, ensure_ascii=False, indent=2, sort_keys=True)
    out.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]

    counts = {k: len(v) for k, v in db.items() if isinstance(v, (list, dict)) and k != "meta"}
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) sha256={digest}")
    for k, v in counts.items():
        print(f"  {k:20s}: {v}")
    (out.parent / "world_manifest.json").write_text(
        json.dumps({"sha256_16": digest, "counts": counts, "meta": db["meta"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
