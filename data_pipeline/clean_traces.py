#!/usr/bin/env python3
"""Phase 1 — Cleanup der SDG-thinking-Traces (Option A, 814).

Entfernt das Bestaetigungs-Ritual am Trace-Ende ("Done./Proceeds/[Final Check]/
[Output Generation]") und droppt echt degenerierte / nicht-ausfuehrbare Traces.
Schreibt eine saubere Datei; die Rohdaten bleiben unangetastet.

Pipeline pro Trace:
  1. SQL fuehrt nicht gegen Schema aus            -> drop (sql_exec)
  2. Ritual-Schwanz abschneiden (strip_tail)
  3. nach Strip neu pruefen: trigram-rep>10 ODER uniq<0.30 ODER len<80 ODER hedge>3 -> drop
  4. sonst behalten (mit gestripptem thinking)
"""
import json, re, sqlite3, collections, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/generated/trace_distill.jsonl"
DST = sys.argv[2] if len(sys.argv) > 2 else "data/generated/trace_distill_clean.jsonl"

HEDGE = ("wait", "actually", "hmm", "or maybe", "alternatively",
         "let me reconsider", "on second thought", "or perhaps")
# Ritual-/Filler-Marker (reine Output-Vorbereitung, kein echtes Reasoning)
RIT = re.compile(
    r'\b(Done|Proceeds?|Ready|Matches?|Output matches|No extra text|Final Check|'
    r'Output Generation|Self-Correction|Proceed|No further|Confirmed)\b'
    r'|\[(Output|Final Check|Output Generation|Proceeds?|Done)\]', re.I)

def uniq(t):
    w = re.findall(r"\w+", t.lower()); return len(set(w)) / len(w) if w else 1.0
def hedge(t):
    tl = t.lower(); return sum(tl.count(h) for h in HEDGE)
def trig(t):
    w = re.findall(r"\w+", t.lower())
    if len(w) < 6: return 1
    return max(collections.Counter(tuple(w[i:i+3]) for i in range(len(w)-2)).values())
def sql_ok(schema, q):
    try:
        c = sqlite3.connect(":memory:"); c.executescript(schema); c.execute(q).fetchall(); c.close(); return True
    except Exception: return False

def is_ritual_line(ln):
    s = ln.strip()
    return bool(s) and bool(RIT.search(s)) and len(RIT.sub("", s).strip(" .:-`*()[]\t")) < 8

def strip_tail(t):
    """Cut at the first ritual-only line in the last 2/3, then pop trailing ritual lines."""
    lines = t.split("\n")
    cut = len(lines)
    for i in range(len(lines)//3, len(lines)):
        if is_ritual_line(lines[i]):
            cut = i; break
    kept = lines[:cut]
    while kept and is_ritual_line(kept[-1]):
        kept.pop()
    return "\n".join(kept).rstrip()

rows = [json.loads(l) for l in open(SRC) if l.strip()]
N = len(rows)
out = []
drops = collections.Counter()
shrink = []  # (vorher, nachher) len
for r in rows:
    if not sql_ok(r.get("schema", ""), r.get("sql", "")):
        drops["sql_exec"] += 1; continue
    th0 = r.get("thinking", "")
    th = strip_tail(th0)
    shrink.append((len(th0), len(th)))
    if len(th) < 80:
        drops["too_short"] += 1; continue
    if trig(th) > 10:
        drops["trigram_rep>10"] += 1; continue
    if uniq(th) < 0.30:
        drops["uniq<0.30"] += 1; continue
    if hedge(th) > 3:
        drops["hedge>3"] += 1; continue
    r2 = dict(r); r2["thinking"] = th; r2["_meta_cleaned"] = True
    out.append(r2)

with open(DST, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

def pct(a, p): a = sorted(a); return a[min(len(a)-1, int(p/100*len(a)))]
print(f"=== CLEANUP {SRC} -> {DST} ===")
print(f"rein: {N}   raus: {sum(drops.values())}   sauber: {len(out)} ({100*len(out)/N:.0f}%)")
print("drop-grunde:")
for k, v in drops.most_common(): print(f"   {v:4d}  {k}")
if shrink:
    avg_before = sum(a for a, _ in shrink)/len(shrink)
    avg_after  = sum(b for _, b in shrink)/len(shrink)
    print(f"\nstrip: thinking-len im schnitt {avg_before:.0f} -> {avg_after:.0f} chars")
kth = [r["thinking"] for r in out]
print(f"\nsaubere thinking-laenge: p50={pct([len(x) for x in kth],50)} p90={pct([len(x) for x in kth],90)} p99={pct([len(x) for x in kth],99)} max={max(len(x) for x in kth)} chars")
print(f"saubere trigram-rep:     p90={pct([trig(x) for x in kth],90)} p99={pct([trig(x) for x in kth],99)} max={max(trig(x) for x in kth)}")
print(f"saubere uniq-ratio:      min={min(uniq(x) for x in kth):.2f} p10={pct([uniq(x) for x in kth],10):.2f}")
print(f"saubere hedge:           p90={pct([hedge(x) for x in kth],90)} max={max(hedge(x) for x in kth)}")
print("\nkategorie-balance (complexity):")
for k, v in collections.Counter(r.get("complexity","?") for r in out).most_common():
    print(f"   {v:4d}  {100*v/len(out):4.1f}%  {k}")
