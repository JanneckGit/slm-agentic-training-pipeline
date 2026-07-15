"""
data_pipeline/validate_areal.py
===============================
Validates the raw AReaL tau2 pull (inclusionAI/AReaL-tau2-data) in data/raw/areal/,
fetched by data_pipeline/prepare_agentic_data.py --dataset areal.

Checks (all JSONL processing is streaming — the 874 MB SFT file is never held in RAM):
  (a) structural  : per-row schema of the per-turn SFT file and the RL task file
  (b) integrity   : row counts + per-domain splits vs the expected values in the config
                    (data.agentic.areal.expected, from the HF card), duplicate ids, sizes
  (c) referential : every RL db_path resolves to a file in tau2_rl_database/; every
                    snapshot parses (json/toml)

Severity model: every check is fail/warn/info. Exit code 1 iff a fail-level check failed.
Report: <root>/validation_report.json + a `validation` stamp on the AReaL-tau2 entry in
data/raw/agentic_manifest.json.

2026-07-15 trim: the one-off acquisition extras (--deep tau2 snapshot loading, telecom-extras
info, metadata/criteria keyset census) were removed after the frozen snapshot passed with
0 fail / 0 warn — the retained core is what a RE-download would need to re-verify.

Usage:
    PYTHONPATH=. .venv-tau2/bin/python data_pipeline/validate_areal.py --config config/pipeline_config.yaml
"""

import argparse
import json
import logging
import sys
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from data_pipeline.common import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DOMAINS = ("airline", "retail", "telecom")
VALID_ROLES = {"system", "user", "assistant", "tool"}
SFT_FILE = "tau2_sft_train.jsonl"
RL_FILE = "tau2_rl_train.jsonl"
DB_DIR = "tau2_rl_database"
# from the HF card (2026-07-08); size drift is warn-level (upstream re-upload signal)
EXPECTED_SIZE_MB = {SFT_FILE: 874.0, RL_FILE: 8.69}
# fallback if the config carries no data.agentic.areal.expected block
EXPECTED_DEFAULT = {
    "sft_rows": 33531,
    "rl_rows": 1982,
    "sft_domains": {"airline": 12842, "retail": 11395, "telecom": 9294},
    "rl_domains": {"airline": 1148, "retail": 563, "telecom": 271},
}


def check(name: str, level: str, passed: bool, detail: str) -> dict:
    assert level in ("fail", "warn", "info")
    return {"name": name, "level": level, "passed": passed, "detail": detail}


def iter_jsonl(path: Path):
    """Yields (line_no, obj_or_None, error_or_None) — streaming, one line at a time."""
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield i, json.loads(line), None
            except json.JSONDecodeError as e:
                yield i, None, str(e)


# SFT metadata carries no explicit domain field; the three domains ship three distinct
# per-domain keysets (verified via census on the 2026-07-08 snapshot):
#   airline -> seed_pattern_task_id · retail -> scenario_id · telecom -> difficulty/num_subtasks
_DOMAIN_MARKER_KEYS = (("seed_pattern_task_id", "airline"), ("scenario_id", "retail"),
                       ("difficulty", "telecom"), ("num_subtasks", "telecom"))
# metadata core shared by all three domain variants
SFT_META_CORE = {"source_dialog_id", "turn_index", "correct", "reward"}


def _derive_domain(row: dict) -> str | None:
    """Best-effort domain of an SFT row: explicit field, per-domain marker key,
    then a substring scan over the metadata blob (dialog ids name the domain)."""
    meta = row.get("metadata") or {}
    for key in ("domain", "source_domain", "env", "task_domain"):
        v = meta.get(key)
        if isinstance(v, str) and v.lower() in DOMAINS:
            return v.lower()
    for key, domain in _DOMAIN_MARKER_KEYS:
        if key in meta:
            return domain
    blob = json.dumps(meta, ensure_ascii=False).lower()
    hits = [d for d in DOMAINS if d in blob]
    return hits[0] if len(hits) == 1 else None


def _tool_calls_of(turn: dict) -> list:
    tc = turn.get("tool_calls")
    return tc if isinstance(tc, list) else []


def _answer_tool_calls_ok(answer: dict) -> tuple[int, int]:
    """Returns (n_calls, n_bad) for answer.tool_calls; accepts OpenAI-style
    {function:{name,arguments}} as well as flat {name,arguments}."""
    n_calls = n_bad = 0
    for tc in _tool_calls_of(answer):
        n_calls += 1
        fn = tc.get("function", tc) if isinstance(tc, dict) else {}
        name = fn.get("name") if isinstance(fn, dict) else None
        args = fn.get("arguments") if isinstance(fn, dict) else None
        ok = isinstance(name, str) and name.strip() != ""
        if ok and isinstance(args, str):
            try:
                json.loads(args)
            except json.JSONDecodeError:
                ok = False
        elif ok and not isinstance(args, (dict, type(None))):
            ok = False
        if not ok:
            n_bad += 1
    return n_calls, n_bad


# ---------------------------------------------------------------------------
# (a)+(b) SFT file — one streaming pass
# ---------------------------------------------------------------------------
def validate_sft(path: Path, expected: dict) -> tuple[list[dict], dict]:
    results: list[dict] = []
    n = 0
    parse_errors: list[str] = []
    missing_top = bad_roles = first_not_system = orphan_tool_msgs = 0
    empty_answers = rows_with_calls = rows_with_thinking = 0
    total_calls = bad_calls = 0
    core_missing = rows_correct = rows_with_correct_label = 0
    domain_counts: Counter = Counter()
    underivable_domain = 0
    dialog_turn_seen: set = set()
    dialog_turn_dupes = 0
    dialog_key = turn_key = None

    for line_no, row, err in iter_jsonl(path):
        if err is not None:
            parse_errors.append(f"line {line_no}: {err}")
            continue
        n += 1

        messages = row.get("messages")
        answer = row.get("answer")
        meta = row.get("metadata")
        if not (isinstance(messages, list) and messages and isinstance(answer, dict) and isinstance(meta, dict)):
            missing_top += 1
            continue

        # roles + tool-message sequencing
        row_bad_role = False
        saw_tool_calls = False
        for m in messages:
            role = m.get("role") if isinstance(m, dict) else None
            if role not in VALID_ROLES:
                row_bad_role = True
            if role == "assistant" and _tool_calls_of(m):
                saw_tool_calls = True
            if role == "tool" and not saw_tool_calls:
                orphan_tool_msgs += 1
        bad_roles += row_bad_role
        if messages[0].get("role") != "system":
            first_not_system += 1

        # answer turn: content/thinking OR tool_calls
        content = answer.get("content") or ""
        thinking = answer.get("thinking") or ""
        n_calls, n_bad = _answer_tool_calls_ok(answer)
        total_calls += n_calls
        bad_calls += n_bad
        if n_calls:
            rows_with_calls += 1
        if str(thinking).strip():
            rows_with_thinking += 1
        if not (str(content).strip() or str(thinking).strip() or n_calls):
            empty_answers += 1

        # core keys + correctness label + domain + (dialog, turn) uniqueness
        if not SFT_META_CORE.issubset(meta.keys()):
            core_missing += 1
        if "correct" in meta:
            rows_with_correct_label += 1
            rows_correct += 1 if meta.get("correct") == 1 else 0
        d = _derive_domain(row)
        if d:
            domain_counts[d] += 1
        else:
            underivable_domain += 1
        if dialog_key is None:
            dialog_key = next((k for k in meta if "dialog" in k.lower()), None)
            turn_key = next((k for k in meta if "turn" in k.lower()), None)
        if dialog_key and turn_key:
            pair = (str(meta.get(dialog_key)), str(meta.get(turn_key)))
            if pair in dialog_turn_seen:
                dialog_turn_dupes += 1
            dialog_turn_seen.add(pair)

    results.append(check("sft_lines_parse", "fail", not parse_errors,
                         f"{len(parse_errors)} unparseable lines" + (f"; first: {parse_errors[0]}" if parse_errors else "")))
    results.append(check("sft_top_level_keys", "fail", missing_top == 0,
                         f"{missing_top}/{n} rows missing messages/answer/metadata"))
    results.append(check("sft_roles_valid", "fail", bad_roles <= 0.001 * max(n, 1),
                         f"{bad_roles}/{n} rows with a role outside {sorted(VALID_ROLES)} (fail >0.1%)"))
    results.append(check("sft_first_msg_system", "warn", first_not_system == 0,
                         f"{first_not_system}/{n} rows whose first message is not system"))
    results.append(check("sft_tool_msg_sequencing", "warn", orphan_tool_msgs == 0,
                         f"{orphan_tool_msgs} tool messages with no earlier assistant tool_calls in-row"))
    results.append(check("sft_answer_nonempty", "fail", empty_answers <= 0.01 * max(n, 1),
                         f"{empty_answers}/{n} answers with neither content/thinking nor tool_calls (fail >1%)"))
    results.append(check("sft_tool_calls_parse", "fail", bad_calls <= 0.01 * max(total_calls, 1),
                         f"{bad_calls}/{total_calls} answer tool_calls with empty name or unparseable arguments (fail >1%)"))
    if bad_calls:
        results.append(check("sft_tool_calls_flawless", "warn", False,
                             f"{bad_calls} imperfect tool_calls (within the 1% budget, still worth a look)"))
    results.append(check("sft_metadata_core_keys", "fail", core_missing <= 0.001 * max(n, 1),
                         f"{core_missing}/{n} rows missing a core key {sorted(SFT_META_CORE)} (fail >0.1%)"))
    correct_rate = rows_correct / max(rows_with_correct_label, 1)
    results.append(check("sft_correct_label", "info", True,
                         f"{rows_with_correct_label}/{n} rows carry a correct-label; correct==1 rate={correct_rate:.3f} "
                         "— the file INCLUDES failed turns; the converter must filter on metadata.correct"))

    exp_rows = expected.get("sft_rows")
    results.append(check("sft_row_count", "fail", n == exp_rows, f"rows={n}, expected={exp_rows}"))
    coverage_ok = underivable_domain <= 0.01 * max(n, 1)
    exp_domains = expected.get("sft_domains", {})
    split_ok = all(domain_counts.get(d, 0) == exp_domains.get(d) for d in DOMAINS)
    results.append(check("sft_domain_split", "fail" if coverage_ok else "warn", split_ok,
                         f"derived={dict(domain_counts)} (underivable={underivable_domain}), expected={exp_domains}"
                         + ("" if coverage_ok else " — >1% underivable, downgraded to warn")))
    if dialog_key and turn_key:
        results.append(check("sft_dialog_turn_unique", "warn", dialog_turn_dupes == 0,
                             f"{dialog_turn_dupes} duplicate ({dialog_key}, {turn_key}) pairs"))
    else:
        results.append(check("sft_dialog_turn_unique", "info", True,
                             "no dialog/turn keys found in metadata — uniqueness check skipped"))
    results.append(check("sft_rates", "info", True,
                         f"tool_call rate={rows_with_calls / max(n, 1):.3f}, thinking rate={rows_with_thinking / max(n, 1):.3f}"
                         " (input for the deferred converter)"))

    counts = {
        "sft_rows": n,
        "sft_domains": dict(domain_counts),
        "sft_underivable_domain": underivable_domain,
        "answer_tool_call_rate": round(rows_with_calls / max(n, 1), 4),
        "thinking_rate": round(rows_with_thinking / max(n, 1), 4),
        "correct_rate": round(correct_rate, 4),
    }
    return results, counts


# ---------------------------------------------------------------------------
# (a)+(b) RL file — one streaming pass
# ---------------------------------------------------------------------------
def validate_rl(path: Path, expected: dict) -> tuple[list[dict], dict, set]:
    results: list[dict] = []
    n = 0
    parse_errors: list[str] = []
    missing_required = criteria_errors = unprefixed = 0
    ids_seen: set = set()
    dupes = 0
    domain_counts: Counter = Counter()
    scenario_missing = 0
    db_paths: set = set()

    for line_no, row, err in iter_jsonl(path):
        if err is not None:
            parse_errors.append(f"line {line_no}: {err}")
            continue
        n += 1

        rid = row.get("id")
        if not (isinstance(rid, str) and rid and row.get("db_path") and row.get("evaluation_criteria") is not None):
            missing_required += 1
            continue
        if rid in ids_seen:
            dupes += 1
        ids_seen.add(rid)
        db_paths.add(str(row["db_path"]))

        domain = next((d for d in DOMAINS if rid.lower().startswith(d)), None)
        if domain is None:  # fallback: user_scenario carries the domain
            us = row.get("user_scenario") or {}
            v = us.get("domain") if isinstance(us, dict) else None
            domain = v.lower() if isinstance(v, str) and v.lower() in DOMAINS else None
        if domain is None:
            unprefixed += 1
        else:
            domain_counts[domain] += 1

        crit = row.get("evaluation_criteria")
        if isinstance(crit, str):
            try:
                json.loads(crit)
            except json.JSONDecodeError:
                criteria_errors += 1

        us = row.get("user_scenario")
        if not (isinstance(us, dict) and us):
            scenario_missing += 1

    results.append(check("rl_lines_parse", "fail", not parse_errors,
                         f"{len(parse_errors)} unparseable lines" + (f"; first: {parse_errors[0]}" if parse_errors else "")))
    results.append(check("rl_required_keys", "fail", missing_required == 0,
                         f"{missing_required}/{n} rows missing id/db_path/evaluation_criteria"))
    results.append(check("rl_ids_unique", "fail", dupes == 0, f"{dupes} duplicate ids"))
    results.append(check("rl_domain_prefix", "fail", unprefixed == 0,
                         f"{unprefixed}/{n} rows with underivable domain (id prefix + user_scenario)"))
    results.append(check("rl_eval_criteria_parse", "fail", criteria_errors == 0,
                         f"{criteria_errors} evaluation_criteria that do not parse as JSON"))
    results.append(check("rl_user_scenario_present", "warn", scenario_missing == 0,
                         f"{scenario_missing}/{n} rows without a non-empty user_scenario dict"))

    exp_rows = expected.get("rl_rows")
    results.append(check("rl_row_count", "fail", n == exp_rows, f"rows={n}, expected={exp_rows}"))
    exp_domains = expected.get("rl_domains", {})
    split_ok = all(domain_counts.get(d, 0) == exp_domains.get(d) for d in DOMAINS)
    results.append(check("rl_domain_split", "fail", split_ok,
                         f"derived={dict(domain_counts)}, expected={exp_domains}"))

    counts = {
        "rl_rows": n,
        "rl_domains": dict(domain_counts),
        "distinct_db_paths": len(db_paths),
    }
    return results, counts, db_paths


# ---------------------------------------------------------------------------
# (c) referential: db_path -> tau2_rl_database/, snapshot parseability
# ---------------------------------------------------------------------------
def validate_db_refs(db_paths: set, root: Path) -> tuple[list[dict], list[dict]]:
    results: list[dict] = []
    db_dir = root / DB_DIR
    snapshot_files = sorted(p for p in db_dir.iterdir() if p.is_file() and not p.name.startswith(".")) \
        if db_dir.is_dir() else []
    results.append(check("db_dir_nonempty", "fail", bool(snapshot_files),
                         f"{len(snapshot_files)} snapshot files in {db_dir}"))

    by_name = {p.name: p for p in snapshot_files}
    unresolved, basename_only, referenced = [], [], set()
    for dp in sorted(db_paths):
        as_is = (root / dp).resolve()
        if as_is.is_file():
            referenced.add(as_is.name)
        elif Path(dp).name in by_name:
            basename_only.append(dp)
            referenced.add(Path(dp).name)
        else:
            unresolved.append(dp)
    results.append(check("db_paths_resolve", "fail", not unresolved,
                         f"{len(unresolved)}/{len(db_paths)} distinct db_paths unresolved"
                         + (f"; first: {unresolved[0]}" if unresolved else "")))
    results.append(check("db_paths_direct", "warn", not basename_only,
                         f"{len(basename_only)} db_paths resolved only via basename match (upstream path prefix differs)"
                         + (f"; e.g. {basename_only[0]}" if basename_only else "")))
    orphans = [p.name for p in snapshot_files if p.name not in referenced]
    results.append(check("db_snapshots_referenced", "warn", not orphans,
                         f"{len(orphans)} snapshot files referenced by no RL task: {orphans}"))

    db_files_report = []
    parse_fail = 0
    for p in snapshot_files:
        ok, msg = True, "ok"
        try:
            if p.suffix == ".json":
                with open(p) as f:
                    json.load(f)
            elif p.suffix == ".toml":
                with open(p, "rb") as f:
                    tomllib.load(f)
            else:
                ok, msg = True, f"skipped (unknown suffix {p.suffix})"
        except Exception as e:  # noqa: BLE001 — any parse failure is the finding itself
            ok, msg = False, f"error: {str(e)[:200]}"
            parse_fail += 1
        db_files_report.append({"file": p.name, "size_mb": round(p.stat().st_size / 1e6, 2),
                                "parse_ok": ok, "detail": msg})
    results.append(check("db_snapshots_parse", "fail", parse_fail == 0,
                         f"{parse_fail}/{len(snapshot_files)} snapshots fail to parse (json/tomllib)"))
    return results, db_files_report


# ---------------------------------------------------------------------------
# report + manifest stamp
# ---------------------------------------------------------------------------
def size_checks(root: Path) -> list[dict]:
    results = []
    for name, exp_mb in EXPECTED_SIZE_MB.items():
        p = root / name
        if not p.is_file():
            results.append(check(f"file_exists:{name}", "fail", False, f"missing: {p}"))
            continue
        mb = p.stat().st_size / 1e6
        results.append(check(f"file_exists:{name}", "fail", True, f"{mb:.1f} MB"))
        results.append(check(f"file_size:{name}", "warn", abs(mb - exp_mb) <= 0.10 * exp_mb,
                             f"{mb:.1f} MB vs expected ~{exp_mb} MB (±10%)"))
    return results


def stamp_manifest(raw_dir: Path, passed: bool, report_path: Path) -> None:
    manifest_path = raw_dir / "agentic_manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    entry = manifest.setdefault("AReaL-tau2", {"name": "AReaL-tau2"})
    entry["validation"] = {
        "passed": passed,
        "report": str(report_path),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"Manifest validation stamp -> {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate the raw AReaL tau2 pull in data/raw/areal/")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--root", default=None, help="AReaL snapshot dir (default: <raw_dir>/areal)")
    parser.add_argument("--report", default=None, help="report path (default: <root>/validation_report.json)")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    root = Path(args.root) if args.root else raw_dir / "areal"
    report_path = Path(args.report) if args.report else root / "validation_report.json"
    areal_cfg = (data_cfg.get("agentic") or {}).get("areal") or {}
    expected = {**EXPECTED_DEFAULT, **(areal_cfg.get("expected") or {})}
    dataset_id = areal_cfg.get("dataset", "inclusionAI/AReaL-tau2-data")

    if not root.is_dir():
        logger.error(f"AReaL root not found: {root} — run prepare_agentic_data.py --dataset areal first")
        sys.exit(1)

    results = size_checks(root)
    counts: dict = {}

    logger.info(f"[SFT] streaming {root / SFT_FILE} …")
    sft_results, sft_counts = validate_sft(root / SFT_FILE, expected)
    results += sft_results
    counts.update(sft_counts)

    logger.info(f"[RL] streaming {root / RL_FILE} …")
    rl_results, rl_counts, db_paths = validate_rl(root / RL_FILE, expected)
    results += rl_results
    counts.update(rl_counts)

    logger.info(f"[DB] referential checks against {root / DB_DIR} …")
    ref_results, db_files_report = validate_db_refs(db_paths, root)
    results += ref_results

    n_fail = sum(1 for r in results if r["level"] == "fail" and not r["passed"])
    n_warn = sum(1 for r in results if r["level"] == "warn" and not r["passed"])
    passed = n_fail == 0

    report = {
        "dataset_id": dataset_id,
        "root": str(root),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": {"passed": passed, "n_fail": n_fail, "n_warn": n_warn, "n_checks": len(results)},
        "counts": counts,
        "checks": results,
        "db_files": db_files_report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    stamp_manifest(raw_dir, passed, report_path)

    logger.info("\n" + "=" * 70)
    for r in results:
        mark = "OK  " if r["passed"] else ("FAIL" if r["level"] == "fail" else "WARN")
        if r["level"] == "info":
            mark = "INFO"
        logger.info(f"  [{mark}] {r['name']}: {r['detail']}")
    logger.info("=" * 70)
    logger.info(f"{'✅ PASS' if passed else '❌ FAIL'} — {n_fail} fail, {n_warn} warn, "
                f"{len(results)} checks. Report: {report_path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
