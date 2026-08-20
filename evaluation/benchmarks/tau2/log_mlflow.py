#!/usr/bin/env python3
"""tau2-bench → MLflow (Muster: bfcl/log_mlflow.py).

Laeuft unter .venv-tau2 (traegt mlflow-skinny; braucht KEIN tau2 — liest nur
run_manifest.json + report_metrics.json, die der Report geschrieben hat).
Run-Name = Label; erneutes Loggen desselben Labels erzeugt bewusst einen NEUEN Run
(Historie bleibt nachvollziehbar). Tracking-URI setzt der Aufrufer
(MLFLOW_TRACKING_URI=file://$REPO/mlruns + MLFLOW_ALLOW_FILE_STORE=true).
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--experiment", default="tau2_eval")
    a = ap.parse_args()

    manifest_path = a.run / "run_manifest.json"
    metrics_path = a.run / "report_metrics.json"
    if not manifest_path.exists() or not metrics_path.exists():
        print("log_mlflow: Manifest oder report_metrics.json fehlt — nichts geloggt.",
              file=sys.stderr)
        return 0
    m = json.loads(manifest_path.read_text())
    rep = json.loads(metrics_path.read_text())
    if not rep.get("domains"):
        print("log_mlflow: keine Domaenen-Metriken — nichts geloggt.", file=sys.stderr)
        return 0

    import mlflow
    mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name=m["label"]):
        n_pass1, weights = 0.0, 0
        for dom, s in rep["domains"].items():
            for k, v in (s.get("pass_hat") or {}).items():
                mlflow.log_metric(f"passhat{k}_{dom}", v)
            if s.get("avg_reward") is not None:
                mlflow.log_metric(f"avg_reward_{dom}", s["avg_reward"])
            for comp, cv in (s.get("components") or {}).items():
                if cv.get("rate") is not None:
                    mlflow.log_metric(f"{comp}_rate_{dom}", cv["rate"])
            mlflow.log_metric(f"think_leaks_{dom}", s.get("think_leaks", 0))
            p1 = (s.get("pass_hat") or {}).get("1")
            if p1 is not None and s.get("tasks"):
                n_pass1 += p1 * s["tasks"]
                weights += s["tasks"]
        if weights:
            mlflow.log_metric("passhat1_overall", n_pass1 / weights)

        params = {
            "model": m["model"],
            "model_alias": m["model_alias"],
            "model_fingerprint": m.get("model_fingerprint"),
            "agent_llm_args": json.dumps({k: v for k, v in m["agent"]["llm_args"].items()
                                          if not k.startswith("api_")}),
            "user_sim_model": m["user_sim"]["model"],
            "user_sim_llm_args": json.dumps({k: v for k, v in m["user_sim"]["llm_args"].items()
                                             if not k.startswith("api_")}),
            "nl_judge": m["nl_judge"]["model_litellm"],
            "num_trials": m["run"]["num_trials"],
            "seed": m["run"]["seed"],
            "max_concurrency": m["run"]["max_concurrency"],
            "domains": ",".join(m["domains"]),
            "tau2_version": m["tau2_version"],
            "tau2_commit": m["tau2_commit"][:12],
            "data_fingerprint": m["data_fingerprint"],
            "repo_commit": m.get("repo_commit"),
        }
        rc = (m["domains"].get("banking_knowledge") or {}).get("retrieval_config")
        if rc:
            params["banking_retrieval_config"] = rc
        mlflow.log_params(params)
        mlflow.log_artifact(str(manifest_path))
        report_txt = a.run / "logs" / "run.log"
        if report_txt.exists():
            mlflow.log_artifact(str(report_txt))
    print(f"== MLflow: Experiment {a.experiment}, Run {m['label']} geloggt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
