"""
sdg_pipeline/db_bahn/bakeoff_summary.py
=======================================
Aggregate all bake-off trace files (data/generated/sdg/db_traces_bakeoff_dev_*.jsonl) into per-teacher
metrics and print/update the markdown table for docs/teacher-bakeoff.md.

Metrics per teacher (the plan's protocol): verified-yield %, German-quality proxy, replan-rate on
injected tasks, avg turns, tool-call validity, avg wall s/rollout, traces/GPU-h, score =
yield x german per GPU-hour (normalized). German proxy = fraction of final answers that look German
(>=2 German function words or umlauts) — the quick sampled read stays manual.

Plain python3 (PYTHONPATH=. — braucht data_pipeline.common), kein tau2 nötig.
"""

import glob
import json
import re
import sys

from data_pipeline.common import final_answer

GERMAN_WORDS = re.compile(
    r"\b(der|die|das|ist|sind|und|für|wurde|wurden|keine|aktuell|Zug|Verspätung|zugeteilt|Minuten|planmäßig)\b",
    re.IGNORECASE)
UMLAUT = re.compile(r"[äöüÄÖÜß]")


def german_score(text: str) -> bool:
    return len(GERMAN_WORDS.findall(text or "")) >= 2 or bool(UMLAUT.search(text or ""))


def main():
    rows = {}
    for path in sorted(glob.glob("data/generated/sdg/db_traces_bakeoff_dev_*.jsonl")):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = r["teacher"]
            d = rows.setdefault(t, {"n": 0, "ok": 0, "german": 0, "inj": 0, "replan": 0,
                                    "turns": 0.0, "tc_valid": 0.0, "wall": 0.0, "err_finish": 0})
            s = r["score"]
            d["n"] += 1
            d["ok"] += s["score"] == 1.0
            d["german"] += german_score(final_answer(r["messages"]))
            d["turns"] += s["turns_used"]
            d["tc_valid"] += s.get("tool_calls_valid", 0.0)
            d["wall"] += r.get("wall_s", 0.0)
            d["err_finish"] += r.get("finish_reason") not in ("final_answer",)
            if r.get("injected"):
                d["inj"] += 1
                d["replan"] += s.get("replan_occurred", 0.0)

    if not rows:
        print("no bake-off traces found")
        return

    hdr = ("| Teacher | n | verified-yield | German | replan (inj) | avg turns | tc-valid | "
           "s/rollout | traces/GPU-h | score |")
    sep = "|" + "---|" * 10
    lines = [hdr, sep]
    for t, d in sorted(rows.items(), key=lambda kv: -(kv[1]["ok"] / max(1, kv[1]["n"]))):
        n = max(1, d["n"])
        y = d["ok"] / n
        g = d["german"] / n
        wall = d["wall"] / n
        per_h = 3600.0 / wall * y if wall > 0 else 0.0   # verified traces per GPU-hour
        score = y * g * per_h
        replan = f"{d['replan'] / d['inj']:.0%}" if d["inj"] else "–"
        lines.append(f"| {t} | {d['n']} | **{y:.0%}** | {g:.0%} | {replan} | "
                     f"{d['turns'] / n:.1f} | {d['tc_valid'] / n:.0%} | {wall:.0f}s | "
                     f"{per_h:.0f} | {score:.0f} |")
    table = "\n".join(lines)
    print(table)
    if "--write" in sys.argv:
        doc = "docs/teacher-bakeoff.md"
        content = open(doc).read()
        marker = "## Results"
        nxt = "## Winner + validation"
        pre = content.split(marker)[0]
        post = content.split(nxt)[1] if nxt in content else ""
        open(doc, "w").write(pre + marker + "\n\n" + table + "\n\n" + nxt + post)
        print(f"\nwrote table -> {doc}")


if __name__ == "__main__":
    main()
