"""
data_pipeline/build_sft_mix.py
==============================
Assemble the 3-leg SFT mix from the three converted chat files into ONE shuffled training set + a held-out
val split (never in the gradient) for the overfit detector.

Legs (all already in the unified db_bahn chat format), all under data/generated/legs/:
  - db_bahn : db_traces_chat.jsonl        (verified German DB traces; core, up-weighted by count)
  - AReaL   : areal_chat.jsonl            (tau2 dialogue + policy, 2 domains — the dialogue half)
  - ToolACE : toolace_chat.jsonl          (API-schema breadth + irrelevance)

Filters here:
  - db_bahn: drop the "flail" traces with an identical consecutive tool call (name+args repeated back to
    back) — deterministic 0-false-positive quality signal. ~54 on the wave-3.5 set (13.9k traces): 19
    empty calls (name='' -> "Tool '' not found" -> retried) + 35 back-to-back identical reads (the answer
    does not change, pure redundancy). transient-retries are exempt ('vorübergehend' marker).
  - db_bahn: OPTIONAL model-aware per-template cap (--db-caps, file from derive_db_caps.py): templates
    the base model already solves get downweighted (bands beherrscht/korridor/lernkern -> cap 60/250/
    none, derived from the heldout eval). Inside a capped template the kept subset mirrors the
    (fault, replan, recovery_mode) variant mix — leftover rounding seats go to the SMALLEST strata, so
    rare recovery variants survive by construction. Fail-closed: the caps carry a sha256 of the leg
    they were derived for; a changed leg with stale caps aborts. Rationale: docs/dataset-edits-db.md.
  - Val split: stratified per-source (~val-frac), seeded; the rest is train. One shuffle (seed 42), no blocks.

Output: sft_mix_chat/_val (im Normalfall LABEL-suffigiert via ops/build_sft_data.sh, das danach die
kanonischen Namen data/final/sft_mix_chat.jsonl/_val.jsonl als Symlinks umzeigt) + optionales
--manifest (Modell/Label/Provenance + Counts — traj_sft_pipeline.sh gatet BASE dagegen), plus a
printed stats gate. Die Defaults zeigen auf die kanonischen Namen; sind die bereits Symlinks,
bricht das Skript ab statt durch den Link in eine Label-Datei zu schreiben.
The gate prints BOTH record-% and a trained-char proxy per leg — record share (db_bahn 70 / toolace 21 /
areal 9) overstates the small legs; the gradient follows tokens, where db_bahn dominates and the two
public legs weigh far less than their record count. The char proxy (~80/14/6) tracks that trend
tokenizer-free; the exact token split measured separately is ~db_bahn 85 / toolace 10 / areal 5.

Usage:  PYTHONPATH=. python3 data_pipeline/build_sft_mix.py            # default --val-frac 0.0188 (~370 val)
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter

from data_pipeline.common import write_jsonl

LEGS = {
    "db_bahn": "data/generated/legs/db_traces_chat.jsonl",
    "areal": "data/generated/legs/areal_chat.jsonl",
    "toolace": "data/generated/legs/toolace_chat.jsonl",
}


def is_flail(rec: dict) -> bool:
    """A trace with an identical tool call repeated back-to-back (name + args) — the over-search
    signal. Wave-3 exception: a repeat whose first occurrence got a TRANSIENT error observation
    ('vorübergehend' marker, tools.py TRANSIENT_MSG) is the policy-mandated retry, not a flail."""
    obs = {m.get("tool_call_id"): (m.get("content") or "")
           for m in rec["messages"] if m.get("role") == "tool"}
    calls = [(tc["function"]["name"], json.dumps(tc["function"]["arguments"], sort_keys=True),
              tc.get("id"))
             for m in rec["messages"] if m.get("tool_calls") for tc in m["tool_calls"]]
    return any(calls[i][:2] == calls[i + 1][:2]
               and "vorübergehend" not in obs.get(calls[i][2], "")
               for i in range(len(calls) - 1))


def n_assistant_turns(rec: dict) -> int:
    return sum(1 for m in rec["messages"] if m["role"] == "assistant")


def trained_chars(rec: dict) -> int:
    """Chars in the assistant turns AFTER the last user message — a tokenizer-free proxy for the leg's
    gradient weight (mirrors collator_multiturn.final_turns_only). Record share overstates the small
    legs; this tracks what is actually trained. Chars ≈ tokens/4 (rough, German runs slightly higher)."""
    ms = rec["messages"]
    lastq = max((i for i, m in enumerate(ms) if m["role"] == "user"), default=-1)
    return sum(len(m.get("content") or "") for i, m in enumerate(ms)
               if m["role"] == "assistant" and i > lastq)


def sha256_16(path: str) -> str:
    """Streamed sha256, first 16 hex — same truncation convention as the world manifest
    (seed_worldstate.py); pins a caps file to the exact leg bytes it was derived for."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_caps(path: str) -> dict:
    """Caps file from derive_db_caps.py, validated fail-closed BEFORE any leg is read: caps derived
    from other leg bytes must never silently mis-cap."""
    caps = json.load(open(path))
    prov, templates = caps["provenance"], caps["templates"]
    if not templates:
        raise SystemExit(f"HARD-FAIL: caps file {path} has no templates")
    for t, e in templates.items():
        if not (e.get("cap") is None or (isinstance(e["cap"], int) and e["cap"] >= 0)):
            raise SystemExit(f"HARD-FAIL: caps template {t}: invalid cap {e.get('cap')!r}")
    actual = sha256_16(LEGS["db_bahn"])
    if actual != prov["leg_sha256_16"]:
        raise SystemExit(f"HARD-FAIL: stale caps — leg {LEGS['db_bahn']} ist {actual}, caps abgeleitet "
                         f"fuer {prov['leg_sha256_16']}. Neu ableiten: bash ops/build_sft_data.sh")
    return caps


def variant_stats(recs: list) -> dict:
    """Shares of the within-template variant axes — printed before/after capping so a drift in the
    rare recovery/replan variants is visible in the gate instead of silent."""
    n = len(recs) or 1
    rm = Counter(str(r["_meta"].get("recovery_mode")) for r in recs)
    return {"replan": sum(1 for r in recs if r["_meta"].get("replan")) / n,
            "self_recovery": sum(1 for r in recs if r["_meta"].get("self_recovery")) / n,
            "recovery_mode": {k: v / n for k, v in sorted(rm.items())}}


def apply_caps(recs: list, caps_templates: dict, seed: int) -> tuple:
    """Per-template downsampling to the band cap. The kept subset is a shrunk mirror of the template,
    not a blind draw: records stratify by (fault, replan, recovery_mode); each stratum gets
    floor(proportional) seats and the leftover rounding seats go to the SMALLEST strata first
    (ascending size), so rare variants keep their representation by construction; a seeded shuffle
    then picks inside each stratum. Dedicated Random(seed) — the caps content must not shift the
    shared shuffle stream that cuts the val split of the other legs. Kept records keep file order."""
    rng = random.Random(seed)
    by_tmpl = {}
    for i, r in enumerate(recs):
        by_tmpl.setdefault(r["_meta"]["template"], []).append(i)
    keep, bands, n_capped = set(), {}, 0
    for t in sorted(by_tmpl):
        idxs = by_tmpl[t]
        info = caps_templates.get(t)
        if info is None:
            raise SystemExit(f"HARD-FAIL: template {t} fehlt in der Caps-Datei (handeditiert?)")
        b = bands.setdefault(info.get("band") or "ungedeckelt", {"templates": 0, "before": 0, "after": 0})
        b["templates"] += 1
        b["before"] += len(idxs)
        cap = info.get("cap")
        if cap is None or len(idxs) <= cap:
            keep.update(idxs)
            b["after"] += len(idxs)
            continue
        n_capped += 1
        strata = {}
        for i in idxs:
            m = recs[i]["_meta"]
            strata.setdefault((str(m.get("fault")), bool(m.get("replan")), str(m.get("recovery_mode"))),
                              []).append(i)
        seats = {k: cap * len(v) // len(idxs) for k, v in strata.items()}
        # floor(cap*n_s/n) < n_s (cap < n), so the +1 below never overshoots a stratum, and the
        # leftover count is < #strata, so the slice never wraps
        for k in sorted(strata, key=lambda k: (len(strata[k]), k))[:cap - sum(seats.values())]:
            seats[k] += 1
        for k in sorted(strata):
            pool = list(strata[k])
            rng.shuffle(pool)
            keep.update(pool[:seats[k]])
        b["after"] += cap
    kept = [r for i, r in enumerate(recs) if i in keep]
    return kept, {"bands": bands, "n_capped": n_capped, "n_templates": len(by_tmpl)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.0188)  # ~300 held-out (overfit-diag only; no early-stop)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-train", default="data/final/sft_mix_chat.jsonl")
    ap.add_argument("--out-val", default="data/final/sft_mix_val.jsonl")
    ap.add_argument("--db-caps", default=None,
                    help="caps JSON from derive_db_caps.py — model-aware per-template cap on db_bahn")
    ap.add_argument("--manifest", default=None,
                    help="write a mix manifest JSON (model/label/provenance + counts) to this path")
    args = ap.parse_args()
    for p in (args.out_train, args.out_val, args.manifest):
        if p and os.path.islink(p):
            raise SystemExit(f"HARD-FAIL: {p} ist ein Symlink (kanonischer Name) — nicht durch den "
                             f"Link schreiben. ops/build_sft_data.sh nutzen oder explizite --out-*/"
                             f"--manifest Pfade angeben.")
    rng = random.Random(args.seed)
    caps = load_caps(args.db_caps) if args.db_caps else None

    train, val, stats = [], [], {}
    capinfo = variant_pre = variant_post = None
    for src, path in LEGS.items():
        recs = [json.loads(l) for l in open(path) if l.strip()]
        n_read = len(recs)
        dropped = capped_out = 0
        if src == "db_bahn":
            kept = [r for r in recs if not is_flail(r)]
            dropped = n_read - len(kept)
            recs = kept
            if caps:
                variant_pre = variant_stats(recs)
                recs, capinfo = apply_caps(recs, caps["templates"], args.seed)
                variant_post = variant_stats(recs)
                capped_out = (n_read - dropped) - len(recs)
        for r in recs:                       # normalize a coarse source tag for the leakage/stats gate
            r["_meta"]["mix_source"] = src
        rng.shuffle(recs)                    # per-source shuffle before the val cut (seeded)
        n_val = round(len(recs) * args.val_frac)
        val += recs[:n_val]
        train += recs[n_val:]
        stats[src] = {"in": n_read, "dropped": dropped, "capped": capped_out,
                      "val": n_val, "train": len(recs) - n_val}

    rng.shuffle(train)                       # ONE mixed shuffle (no blocks -> no forgetting)
    rng.shuffle(val)

    for path, data in [(args.out_train, train), (args.out_val, val)]:
        write_jsonl(data, path)

    if args.manifest:                        # identity card of this mix — traj_sft_pipeline gates BASE against it
        prov = caps["provenance"] if caps else None
        man = {"model": prov["model"] if prov else None,
               "label": prov["label"] if prov else None,
               "caps_file": args.db_caps, "caps_provenance": prov,
               "seed": args.seed, "val_frac": args.val_frac,
               "out_train": args.out_train, "out_val": args.out_val,
               "stats": stats, "train": len(train), "val": len(val)}
        with open(args.manifest, "w") as f:   # no timestamp — a re-run stays byte-identical
            json.dump(man, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")

    # --- stats gate ---
    print("=== SFT-MIX STATS ===")
    for src, s in stats.items():
        print(f"  {src:8s} in={s['in']:6d} dropped={s['dropped']:3d} capped={s['capped']:5d}"
              f" -> train {s['train']:6d} / val {s['val']:4d}")
    if capinfo:
        print(f"  db_bahn caps ({args.db_caps}): {capinfo['n_capped']}/{capinfo['n_templates']} Templates gekappt")
        for band, b in sorted(capinfo["bands"].items()):
            print(f"    {band:12s} {b['templates']:2d} tmpl {b['before']:6d} -> {b['after']:6d}")
        rm = " ".join(f"{k} {100*variant_pre['recovery_mode'].get(k, 0):.1f}->{100*variant_post['recovery_mode'].get(k, 0):.1f}%"
                      for k in sorted(set(variant_pre["recovery_mode"]) | set(variant_post["recovery_mode"])))
        print(f"    db-Varianten vor->nach Cap: replan {100*variant_pre['replan']:.1f}->{100*variant_post['replan']:.1f}%"
              f" | self_recovery {100*variant_pre['self_recovery']:.1f}->{100*variant_post['self_recovery']:.1f}% | {rm}")
    print(f"  TOTAL    train {len(train)} / val {len(val)} = {len(train)+len(val)}")
    tr_src = Counter(r["_meta"]["mix_source"] for r in train)
    print(f"  train source-mix (records): {dict(tr_src)} "
          f"({', '.join(f'{k} {100*v/len(train):.0f}%' for k,v in tr_src.items())})")
    tok_src = Counter()                              # trained-char proxy = the real gradient weight
    for r in train:
        tok_src[r["_meta"]["mix_source"]] += trained_chars(r)
    tot_c = sum(tok_src.values()) or 1
    print(f"  train source-mix (trained-char ≈ gradient): "
          f"{', '.join(f'{k} {100*v/tot_c:.0f}%' for k,v in tok_src.items())}")
    mt = sum(1 for r in train if n_assistant_turns(r) >= 3)
    print(f"  train multi-assistant (>=3 turns): {mt} = {100*mt/len(train):.0f}%")
    val_src = Counter(r["_meta"]["mix_source"] for r in val)
    print(f"  val source-mix: {dict(val_src)}")
    # leakage guard: no record object shared (paranoia — different files, but assert disjoint by id)
    assert not (set(map(id, train)) & set(map(id, val))), "HARD-FAIL: train/val overlap"
    print("  train/val disjoint: OK")
    if args.manifest:
        print(f"  manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
