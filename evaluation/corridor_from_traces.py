"""
evaluation/corridor_from_traces.py
==================================
Turn a k-sample rollout JSONL (sdg_pipeline/db_bahn/rollout.py --k N) into the RL "corridor" file:
per task the number of accepted samples out of k. GRPO learns only from groups with reward
variance — a task the policy solves 0/k or k/k times yields advantage 0 under Dr.GRPO and is pure
compute. The corridor = tasks with --min-ok <= ok <= --max-ok (default 1 .. k-1); that list is the
input for the Stage-2 pool builder.

Acceptance = evaluation/eval_report.py:accepted (score 1.0, not truncated, not degenerate) — the
same gate as the heldout yield, so corridor and eval numbers speak the same language.

Output JSON (provenance pattern as data/generated/eval/db_bahn_caps.json):
  provenance: input file + sha256_16, teacher label, model, k, sampling (gen_params of record 0)
  tasks:      {task_id: {template, kind, injected, k, ok, rate}}
  summary:    {k, n_tasks, hist: {"0": n, ...}, n_corridor, corridor_by_template, incomplete}

Usage:
    python3 evaluation/corridor_from_traces.py \
        --input data/generated/eval/db_traces_rl_train_ep3_corridor_k8.jsonl
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluation.eval_report import accepted  # noqa: E402  (import-free module, safe to pull in)

KEYS_DEFAULT = REPO / "data/raw/db_sandbox/answer_keys.json"


def sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description="per-task ok/k corridor file from a k-sample rollout JSONL")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="default data/generated/eval/rl_corridor_<teacher-label>_k<k>.json")
    ap.add_argument("--keys", type=Path, default=KEYS_DEFAULT, help="answer_keys.json (for `kind`)")
    ap.add_argument("--min-ok", type=int, default=1, help="corridor lower bound (inclusive)")
    ap.add_argument("--max-ok", type=int, default=None, help="corridor upper bound (inclusive; default k-1)")
    args = ap.parse_args()

    keys = json.loads(args.keys.read_text()) if args.keys.exists() else {}
    flags: dict[str, list[bool]] = defaultdict(list)
    info: dict[str, dict] = {}
    seen, dups = set(), 0
    first = None
    for line in open(args.input):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        first = first or r
        tid = r["task_id"]
        key = (tid, r.get("sample_idx"))
        if key in seen:
            dups += 1
            continue                      # a resumed duplicate must not count twice
        seen.add(key)
        flags[tid].append(accepted(r))
        info.setdefault(tid, {"template": r.get("template"),
                              "kind": (keys.get(tid) or {}).get("kind"),
                              "injected": bool(r.get("injected"))})
    if not flags:
        sys.exit(f"no records in {args.input}")

    k = max(len(v) for v in flags.values())
    max_ok = args.max_ok if args.max_ok is not None else k - 1
    tasks, hist, incomplete = {}, Counter(), 0
    corridor_by_tmpl, n_by_tmpl = Counter(), Counter()
    for tid, fl in flags.items():
        if len(fl) < k:
            incomplete += 1               # interrupted run: not comparable, never in the corridor
            continue
        ok = sum(fl)
        hist[ok] += 1
        t = info[tid]["template"]
        n_by_tmpl[t] += 1
        in_corr = args.min_ok <= ok <= max_ok
        corridor_by_tmpl[t] += in_corr
        tasks[tid] = {**info[tid], "k": k, "ok": ok, "rate": round(ok / k, 4), "corridor": in_corr}

    n_corr = sum(1 for t in tasks.values() if t["corridor"])
    label = (first or {}).get("teacher") or "unknown"
    out = args.out or (REPO / "data/generated/eval" / f"rl_corridor_{label}_k{k}.json")
    doc = {
        "provenance": {
            "input": str(args.input), "input_sha256_16": sha256_16(args.input),
            "teacher": label, "model": ((first or {}).get("gen_params") or {}).get("model"),
            "gen_params": (first or {}).get("gen_params"), "split": (first or {}).get("split"),
            "k": k, "corridor_def": {"min_ok": args.min_ok, "max_ok": max_ok},
            "acceptance": "evaluation/eval_report.py:accepted (score==1.0, !truncated, !degenerate)",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "summary": {"k": k, "n_tasks": len(tasks), "hist": {str(i): hist.get(i, 0) for i in range(k + 1)},
                    "n_corridor": n_corr, "incomplete": incomplete, "duplicates_skipped": dups,
                    "corridor_by_template": dict(sorted(corridor_by_tmpl.items()))},
        "tasks": dict(sorted(tasks.items())),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))

    print(f"\n{args.input}  (teacher={label}, split={doc['provenance']['split']}, k={k})")
    print("ok/k histogram:  " + "  ".join(f"{i}:{hist.get(i, 0)}" for i in range(k + 1)))
    print(f"tasks={len(tasks)}  corridor[{args.min_ok}..{max_ok}]={n_corr}  "
          f"always={hist.get(k, 0)}  never={hist.get(0, 0)}")
    if incomplete:
        print(f"WARNING {incomplete} task(s) with fewer than k={k} samples — excluded (interrupted run?)")
    if dups:
        print(f"WARNING {dups} duplicate (task_id, sample_idx) skipped")
    print(f"\n{'Template':34s} {'tasks':>5s} {'corr':>5s} {'mean rate':>9s}")
    rows = []
    for t, n in n_by_tmpl.items():
        rate = sum(v["rate"] for v in tasks.values() if v["template"] == t) / max(1, n)
        rows.append((corridor_by_tmpl[t] / max(1, n), t, n, corridor_by_tmpl[t], rate))
    for _, t, n, c, rate in sorted(rows, reverse=True):
        print(f"{t[:34]:34s} {n:5d} {c:5d} {rate:9.3f}")
    print(f"\nout: {out}")


if __name__ == "__main__":
    main()
