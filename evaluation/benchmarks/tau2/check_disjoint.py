#!/usr/bin/env python3
"""Disjunktheits-Gate: offizielle tau2-Tasks vs. das AReaL-SFT-Leg.

Prueft, dass KEIN offizieller Task-Text (description/user_scenario/initial_state,
String-Blaetter >= MIN_LEN Zeichen) woertlich im Trainings-Leg vorkommt und dass das
Leg keine telecom-Reste traegt. Zusaetzlich wird der Entity-Overlap (airline-DB-
Kundennamen im Leg-Korpus) als Zahl berichtet — das ist KEIN Fail (AReaL nutzt per
Bauart dieselbe Welt mit neuen Tasks), aber der Grund, warum airline/retail im
Report ein Sternchen tragen und telecom/banking die Held-out-Headlines sind.

Stdlib-only (kein tau2-Import) — laeuft unter jedem Python >= 3.10.
Referenzmessung 2026-08-19 gegen v1.0.1: 0 woertliche Treffer, 0 telecom,
88/393 airline-Namen im Korpus.
"""

import argparse
import json
import re
import sys
from pathlib import Path

MIN_LEN = 30
PROBE_KEYS = ("description", "user_scenario", "initial_state")

_norm_re = re.compile(r"\s+")


def norm(s: str) -> str:
    return _norm_re.sub(" ", s).strip().lower()


def leaves(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from leaves(v)


def run_check(data_dir: Path, leg_path: Path, domains=("airline", "retail")) -> dict:
    """Returns a result dict; result["ok"] is the gate verdict."""
    out = {"leg": str(leg_path), "ok": True, "domains": {}, "telecom_mentions": None,
           "airline_name_overlap": None}
    if not leg_path.exists():
        out["ok"] = None  # skipped — Aufrufer entscheidet (Preflight: WARN)
        out["error"] = f"Leg nicht gefunden: {leg_path}"
        return out

    records = [json.loads(l) for l in leg_path.open()]
    corpus = norm(" ".join(json.dumps(r, ensure_ascii=False) for r in records))
    out["leg_records"] = len(records)
    out["telecom_mentions"] = corpus.count("telecom")
    if out["telecom_mentions"]:
        out["ok"] = False

    for dom in domains:
        tasks_file = data_dir / "domains" / dom / "tasks.json"
        tasks = json.loads(tasks_file.read_text())
        checked, hits, examples = 0, 0, []
        for t in tasks:
            probe = {k: t.get(k) for k in PROBE_KEYS if t.get(k)}
            for s in set(leaves(probe)):
                s_n = norm(s)
                if len(s_n) < MIN_LEN:
                    continue
                checked += 1
                if s_n in corpus:
                    hits += 1
                    if len(examples) < 5:
                        examples.append({"task": t.get("id"), "text": s_n[:90]})
        out["domains"][dom] = {"tasks": len(tasks), "leaves_checked": checked,
                               "verbatim_hits": hits, "examples": examples}
        if hits:
            out["ok"] = False

    # Entity-Spotcheck (informativ, kein Gate)
    db_file = data_dir / "domains" / "airline" / "db.json"
    if db_file.exists():
        db = json.loads(db_file.read_text())
        users = db.get("users") or {}
        uit = users.values() if isinstance(users, dict) else users
        names = sorted({norm(f"{(u.get('name') or {}).get('first_name','')} "
                             f"{(u.get('name') or {}).get('last_name','')}")
                        for u in uit
                        if (u.get("name") or {}).get("first_name")})
        found = [n for n in names if n in corpus]
        out["airline_name_overlap"] = {"db_names": len(names), "in_leg": len(found)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path,
                    default=Path(__import__("os").environ.get("TAU2_DATA_DIR", "")),
                    help="TAU2_DATA_DIR (…/data/tau2)")
    ap.add_argument("--leg", type=Path,
                    default=Path("data/generated/legs/areal_chat.jsonl"))
    a = ap.parse_args()
    if not a.data_dir or not a.data_dir.exists():
        print(f"FAIL: --data-dir/TAU2_DATA_DIR fehlt oder existiert nicht: {a.data_dir}")
        return 1
    # beide Semantiken akzeptieren: …/data (Loader-Konvention) und …/data/tau2
    dd = a.data_dir / "tau2" if (a.data_dir / "tau2" / "domains").is_dir() else a.data_dir
    res = run_check(dd, a.leg)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if res["ok"] is None:
        print("SKIPPED (Leg fehlt)")
        return 0
    print("PASS" if res["ok"] else "FAIL: woertliche Ueberschneidung / telecom-Reste!")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
