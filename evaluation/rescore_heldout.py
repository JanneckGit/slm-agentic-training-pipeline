"""
evaluation/rescore_heldout.py
=============================
Re-score FINISHED rollout files with the CURRENT verifier (evaluation/trajectory_reward.py) — for verifier
fixes after the fact (first use 12.11.2026: typographic spaces/hyphens, see `_visible` there).

Messages stay exactly as generated; only each record's `score` is recomputed. A file is rewritten only if at
least one score changes:
  - the original is copied to --backup-dir first (an existing backup is never overwritten -> file skipped),
  - unchanged lines are written back byte-for-byte, the new file replaces the old one atomically,
  - <OUT>.meta.json gets a `rescored` entry (date, reason, commit, verifier hash, counts). It sits outside
    `identity`, so the resume check of ops/eval_heldout.sh is unaffected.

    .venv-tau2/bin/python evaluation/rescore_heldout.py --reason "typo-norm" \
        data/generated/eval/db_traces_heldout_eval_*_k8.jsonl
    ... --dry-run          # report what would change, write nothing

Per file it prints records changed, accepted before/after and which verifier components flipped, so an
unexpected side effect of a verifier change is visible before anything is written (use --dry-run first).
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from evaluation.eval_report import accepted  # noqa: E402  (score 1.0, not truncated, not degenerate)

_TASKS: dict = {}
_KEYS: dict = {}


def _init_worker() -> None:
    global _TASKS, _KEYS
    from sdg_pipeline.db_bahn.tau2_domain.environment import DATA_DIR  # tau2 venv only
    _TASKS = {t["id"]: t for t in json.load(open(DATA_DIR / "tasks.json"))}
    _KEYS = json.load(open(DATA_DIR / "answer_keys.json"))


def _rescore(line: str) -> tuple[str, bool, bool, list[str]] | None:
    """-> (line to write, accepted before, accepted after, flipped components); None for blank lines.
    The line comes back untouched when the score did not change."""
    from evaluation.trajectory_reward import score_trajectory
    if not line.strip():
        return None
    r = json.loads(line)
    old = r.get("score") or {}
    acc_before = accepted(r)
    new = score_trajectory(_TASKS[r["task_id"]], r["messages"], _KEYS.get(r["task_id"]))
    if json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True):
        return line, acc_before, acc_before, []
    a, b = old.get("components") or {}, new.get("components") or {}
    flips = [f"{k}:{a.get(k)}->{b.get(k)}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    r["score"] = new
    return json.dumps(r, ensure_ascii=False) + "\n", acc_before, accepted(r), flips or ["aux_only"]


def _provenance() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    verifier = REPO / "evaluation" / "trajectory_reward.py"
    dirty = git("status", "--porcelain", "--", "evaluation/trajectory_reward.py")
    return {"commit": git("rev-parse", "--short", "HEAD") + ("+dirty-verifier" if dirty else ""),
            "verifier_sha256": hashlib.sha256(verifier.read_bytes()).hexdigest()[:16]}


def rescore_file(path: Path, pool: Pool, reason: str, backup_dir: Path, dry_run: bool) -> int:
    lines = path.read_text().splitlines(keepends=True)
    out, changed, flips = [], 0, Counter()
    acc_before = acc_after = 0
    for res in pool.imap(_rescore, lines, chunksize=16):
        if res is None:
            continue
        line, before, after, flipped = res
        out.append(line)
        acc_before += before
        acc_after += after
        if flipped:
            changed += 1
            flips.update(flipped)
    n = sum(1 for ln in lines if ln.strip())
    print(f"{path.name}: {n} Rollouts, {changed} Scores geaendert, akzeptiert {acc_before} -> {acc_after}"
          + (f" | Flips {dict(flips.most_common())}" if flips else ""))
    if not changed or dry_run:
        return 0
    backup = backup_dir / path.name
    if backup.exists():
        print(f"  UEBERSPRUNGEN: Backup {backup} existiert schon — erst klaeren, nichts geschrieben")
        return 1
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".rescore.tmp")
    tmp.write_text("".join(out))
    os.replace(tmp, path)
    meta_p = Path(str(path) + ".meta.json")
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        meta.setdefault("rescored", []).append({
            "at": datetime.now().isoformat(timespec="seconds"), "reason": reason, **_provenance(),
            "changed_records": changed, "accepted_before": acc_before, "accepted_after": acc_after,
            "backup": str(backup.relative_to(REPO)) if backup.is_relative_to(REPO) else str(backup)})
        meta_p.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    else:
        print(f"  Hinweis: keine Sidecar {meta_p.name} — Neubewertung nur durch das Backup belegt")
    print(f"  geschrieben (Original: {backup})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("inputs", nargs="+", help="finished rollout JSONL files")
    ap.add_argument("--reason", required=True, help="short note stored in the sidecar, e.g. 'typo-norm'")
    ap.add_argument("--backup-dir", default=None,
                    help="default: data/archive/rescore_<YYYYMMDD>/ in the repo")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    a = ap.parse_args(argv)
    backup_dir = Path(a.backup_dir) if a.backup_dir else REPO / "data" / "archive" / f"rescore_{datetime.now():%Y%m%d}"
    rc = 0
    with Pool(a.workers, initializer=_init_worker) as pool:
        for p in a.inputs:
            rc |= rescore_file(Path(p).resolve(), pool, a.reason, backup_dir, a.dry_run)
    return rc


if __name__ == "__main__":
    sys.exit(main())
