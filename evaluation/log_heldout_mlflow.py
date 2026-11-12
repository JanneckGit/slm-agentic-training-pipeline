"""
evaluation/log_heldout_mlflow.py
================================
Log a FINISHED heldout run (rollout JSONL) to MLflow after the fact.

For runs on a host without our MLflow store (ops/eval_heldout.sh with MLFLOW=0, e.g. the H100 chain
ops/h100_open_models.sh): computes exactly the summary metrics rollout.py logs at the end of a run (same accept
gate, same counters, same names) — but over the WHOLE file, so a resumed run gets one complete entry instead of
the partial `_resume` entry rollout.py would write. Provenance (host, serve mode, vLLM version) comes from the
sidecars <OUT>.meta.json and <OUT>.serve.json written by ops/eval_heldout.sh.

    .venv-tau2/bin/python evaluation/log_heldout_mlflow.py \
        --input data/generated/eval/db_traces_heldout_eval_<label>_k8.jsonl            # logs run "<label>_k8"
    ... --dry-run                                                                     # print, log nothing

Refuses to log a second run with the same name in the experiment (--force overrides).
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_report import accepted  # noqa: E402  (== rollout.accepted: score 1.0, not truncated, not degenerate)

DEGEN_MAX_THINK_CHARS = 12_000   # same thresholds as rollout.py is_degenerate
DEGEN_MAX_DUP8_RATIO = 0.5
REPO = Path(__file__).resolve().parent.parent


def is_degenerate(rec: dict) -> bool:
    d = rec.get("degen") or {}
    return (d.get("think_ngram_dup_ratio", 0.0) > DEGEN_MAX_DUP8_RATIO
            or d.get("max_think_chars", 0) > DEGEN_MAX_THINK_CHARS)


def load(path: str) -> list[dict]:
    recs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r.pop("messages", None)   # not needed for the summary; keeps memory small
                recs.append(r)
    return recs


def summary(recs: list[dict]) -> tuple[dict, dict]:
    """-> (metrics, per-template yields), mirroring the counters of rollout.py main()."""
    st = Counter()
    tmpl_n, tmpl_ok = Counter(), Counter()
    for r in recs:
        sc = r.get("score") or {}
        ok = accepted(r)
        st["n"] += 1
        st["verified"] += ok
        st["truncated"] += bool(r.get("truncated"))
        st["degen"] += is_degenerate(r)
        st["roleplay_tail"] += (r.get("roleplay_tail_chars") or 0) > 0
        st["replan"] += sc.get("replan_occurred", 0.0)
        st["turns"] += sc.get("turns_used", 0)
        tmpl_n[r["template"]] += 1
        tmpl_ok[r["template"]] += ok
        if ok and sc.get("self_recovery"):
            if r.get("fault", "none") in ("runtime", "state+runtime"):
                st["scripted_rec"] += 1
            else:
                st["emergent_rec"] += 1
    n, v = max(1, st["n"]), max(1, st["verified"])
    metrics = {"verified_yield": st["verified"] / n, "replan_rate": st["replan"] / n,
               "avg_turns": st["turns"] / n, "n_rollouts": st["n"],
               "emergent_recovery_rate": st["emergent_rec"] / v, "scripted_recovery_rate": st["scripted_rec"] / v,
               "truncated_rate": st["truncated"] / n, "degen_rate": st["degen"] / n,
               "roleplay_tail_rate": st["roleplay_tail"] / n}
    yields = {f"yield_{t}": tmpl_ok[t] / max(1, tmpl_n[t]) for t in sorted(tmpl_n)}
    return metrics, yields


def provenance(path: str, recs: list[dict]) -> tuple[dict, str, int]:
    """-> (params, label, k) from the sidecars, falling back to the records (legacy files without sidecar)."""
    meta_p, serve_p = Path(path + ".meta.json"), Path(path + ".serve.json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    ident, info = meta.get("identity") or {}, meta.get("info") or {}
    per_task = Counter(r["task_id"] for r in recs)
    label = ident.get("label") or recs[0].get("teacher")
    k = int(ident.get("k") or max(per_task.values()))
    params = {"split": ident.get("split") or recs[0].get("split"), "teacher": label,
              "model": ident.get("model") or label, "k": k, "n_tasks": len(per_task),
              "branch_on_fail": False, "max_turns": ident.get("max_turns", 15),
              "logged_posthoc": True, "source_file": Path(path).name}
    if info.get("host"):
        params["host"] = info["host"]
    params["serve"] = info.get("serve", "compose")
    if serve_p.exists():
        srv = (json.loads(serve_p.read_text()).get("server") or {})
        if srv.get("vllm_version"):
            params["vllm_version"] = srv["vllm_version"]
    if ident.get("chat_template_kwargs"):
        params["chat_template_kwargs"] = json.dumps(ident["chat_template_kwargs"], sort_keys=True)
    # Sampling + Serve aus der Sidecar-Identitaet: weicht bei Open-Weight-Laeufen je Modell ab (Model-Card)
    for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "max_tokens_per_turn",
                "max_model_len", "gpu_util", "vllm_extra", "tool_mode"):
        if ident.get(key) not in (None, ""):
            params[key] = ident[key]
    if info.get("eval_conc"):
        params["eval_conc"] = info["eval_conc"]
    if meta.get("rescored"):   # evaluation/rescore_heldout.py: scores recomputed after a verifier fix
        last = meta["rescored"][-1]
        params["rescored"] = (f"{last.get('at', '')[:10]} {last.get('reason', '')} "
                              f"(verifier {last.get('verifier_sha256', '')}, {last.get('changed_records', 0)} geaendert)")
    return params, label, k


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--input", required=True, help="rollout JSONL of a finished run")
    ap.add_argument("--experiment", default="db_bahn_traj_eval")
    ap.add_argument("--run-name", default=None, help="default: <label>_k<K> (like ops/eval_heldout.sh)")
    ap.add_argument("--tracking-uri", default=None, help="default: $MLFLOW_TRACKING_URI, else <repo>/mlruns")
    ap.add_argument("--dry-run", action="store_true", help="print metrics/params, log nothing")
    ap.add_argument("--force", action="store_true", help="log even if a run with this name exists")
    a = ap.parse_args(argv)

    recs = load(a.input)
    if not recs:
        print(f"{a.input} ist leer — nichts zu loggen")
        return 2
    metrics, yields = summary(recs)
    params, label, k = provenance(a.input, recs)
    run_name = a.run_name or (f"{label}_k{k}" if k != 1 else label)
    print(f"run '{run_name}' <- {a.input} ({metrics['n_rollouts']} Rollouts, {params['n_tasks']} Tasks, k={k})")
    for key in sorted(metrics):
        print(f"  {key:<24} {metrics[key]:.6g}")
    print(f"  + {len(yields)} yield_<template> | params: {params}")
    if a.dry_run:
        return 0

    import mlflow  # only now: a dry run must work without mlflow
    from mlflow.tracking import MlflowClient
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = a.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or (REPO / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    exp = mlflow.set_experiment(a.experiment)
    dupes = MlflowClient(uri).search_runs([exp.experiment_id],
                                          filter_string=f"tags.mlflow.runName = '{run_name}'")
    if dupes and not a.force:
        print(f"ABBRUCH: Run '{run_name}' existiert schon in {a.experiment} ({dupes[0].info.run_id}) — --force uebersteuert")
        return 1
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics({**metrics, **yields})
    print(f"  [mlflow] logged run '{run_name}' -> {a.experiment} @ {uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
