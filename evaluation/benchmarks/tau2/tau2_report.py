#!/usr/bin/env python3
"""tau2-bench Report — Tabelle + Gesundheits-Exit-Code (Muster: bfcl_report.py).

Laeuft unter .venv-tau2bench (bewusste Abweichung von der BFCL-Host-python3-Regel:
wir nutzen tau2s OFFIZIELLE pass^k-Implementierung aus tau2.metrics.agent_metrics
statt sie nachzubauen). Liest nur das Label-Verzeichnis, schreibt report_metrics.json
(fuer log_mlflow.py) — sonst nichts.

Exit != 0 bei: geplanter-aber-fehlender Domaene, infra-Fehlern, Think-Leaks
(<think> im content = reasoning-parser fehlte beim Serving → Lauf ungueltig),
oder --gt-check unter --min-reward (Oracle-Eichung gescheitert).

Modi:
  --run <root>                       normaler Report ueber alle Manifest-Domaenen
  --run <root> --baseline <root2>    zusaetzlich Δ pass^1 gegen einen Vergleichslauf
  --gt-check <save.json> --min-reward 0.9    Oracle-Gate (mock, llm_agent_gt)
"""

import argparse
import json
import re
import sys
from pathlib import Path

GERMAN_MARKERS = (" der ", " die ", " und ", " nicht ", " ich ", "ß", " für ", " Ihre ")
CTX = "context_window_exceeded"
INFRA = "infrastructure_error"


def load_results(path: Path):
    """tau2 v1.0.1 legt um --save-to ein VERZEICHNIS an und schreibt eine monolithische
    results.json HINEIN; Results.load auf dem Verzeichnis erwartet dagegen ein
    simulations/-Unterverzeichnis und liefert 0 Sims (Smoke-Befund). Deshalb immer
    auf die innere results.json zeigen."""
    from tau2.data_model.simulation import Results
    if path.is_dir():
        inner = path / "results.json"
        if inner.exists():
            path = inner
    return Results.load(path)


def find_save(root: Path, name: str) -> Path | None:
    for cand in (root / name, root / f"{name}.json"):
        if cand.exists():
            return cand
    return None


def domain_stats(path: Path) -> dict:
    from tau2.metrics.agent_metrics import get_metrics_df, get_tasks_pass_hat_k
    results = load_results(path)
    df, max_k = get_metrics_df(results)
    ph = get_tasks_pass_hat_k(results)
    stats = {
        "tasks": int(df.task_id.nunique()) if not df.empty else 0,
        "simulations": len(results.simulations),
        "max_k": int(max_k),
        "pass_hat": {},
        "avg_reward": float(df.reward.mean()) if not df.empty else None,
        "terminations": {},
        "think_leaks": 0,
        "german_sims": 0,
        "avg_agent_turns": None,
        "avg_duration_s": None,
        "components": {},
    }
    for k in range(1, (max_k or 0) + 1):
        col = f"pass^{k}"
        if col in ph.columns:
            stats["pass_hat"][k] = float(ph[col].mean())

    term, comp = {}, {}
    turns, durs = [], []
    for sim in results.simulations:
        tr = getattr(sim.termination_reason, "value", str(sim.termination_reason))
        term[tr] = term.get(tr, 0) + 1
        if sim.duration:
            durs.append(sim.duration)
        n_agent = 0
        for m in (sim.messages or []):
            role = getattr(m, "role", None)
            content = getattr(m, "content", None) or ""
            if role == "assistant":
                n_agent += 1
            if isinstance(content, str) and "<think>" in content:
                stats["think_leaks"] += 1
                break
        turns.append(n_agent)
        # Sprach-Heuristik ueber die Agent-Antworten dieser Episode
        text = " ".join((getattr(m, "content", None) or "")
                        for m in (sim.messages or [])
                        if getattr(m, "role", None) == "assistant"
                        and isinstance(getattr(m, "content", None), str))
        if sum(text.count(w) for w in GERMAN_MARKERS) >= 3:
            stats["german_sims"] += 1
        ri = sim.reward_info
        if ri is not None:
            def bump(name, ok):
                a, b = comp.get(name, (0, 0))
                comp[name] = (a + int(bool(ok)), b + 1)
            if ri.db_check is not None:
                bump("db", ri.db_check.db_match)
            for c in ri.env_assertions or []:
                bump("env", c.met)
            for c in ri.action_checks or []:
                bump("action", c.action_match)
            for c in ri.communicate_checks or []:
                bump("communicate", c.met)
            for c in ri.nl_assertions or []:
                bump("nl", c.met)
    stats["terminations"] = term
    stats["components"] = {k: {"met": a, "total": b, "rate": a / b if b else None}
                           for k, (a, b) in comp.items()}
    stats["avg_agent_turns"] = round(sum(turns) / len(turns), 1) if turns else None
    stats["avg_duration_s"] = round(sum(durs) / len(durs), 1) if durs else None
    return stats


def fmt_pct(x):
    return f"{100*x:5.1f}" if x is not None else "    –"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--gt-check", type=Path)
    ap.add_argument("--min-reward", type=float, default=0.9)
    a = ap.parse_args()

    # --- Oracle-Gate-Modus: STRUKTURELLE Harness-Signale, nicht ⌀reward=1 ---
    # Der GT-Agent kennt die erwarteten AKTIONEN, aber nicht die COMMUNICATE-
    # Formulierungen, und der User-Sim darf die Welt legal variieren (Smoke 2026-08-19:
    # mock ⌀0.75 bei perfekt funktionierender Harness). Harness-Defekte zeigen sich
    # dagegen strukturell: Action-Checks matchen nicht, Terminations kippen in
    # error/infra/ctx, <think> leakt, oder der User-Sim setzt nie eigene Tool-Calls ab.
    if a.gt_check:
        results = load_results(a.gt_check)
        sims = results.simulations
        problems = []
        n_action = n_action_ok = n_env = n_env_ok = 0
        user_tool_calls = leaks = 0
        rewards = []
        bad_terms = {INFRA, CTX, "agent_error", "user_error", "too_many_errors",
                     "unexpected_error"}
        for s in sims:
            term = getattr(s.termination_reason, "value", str(s.termination_reason))
            if term in bad_terms:
                problems.append(f"{s.task_id}: termination={term}")
            ri = s.reward_info
            rewards.append(ri.reward if ri else 0.0)
            if ri:
                for c in ri.action_checks or []:
                    n_action += 1
                    n_action_ok += int(c.action_match)
                for c in ri.env_assertions or []:
                    n_env += 1
                    n_env_ok += int(c.met)
            for m in (s.messages or []):
                content = getattr(m, "content", None) or ""
                if isinstance(content, str) and "<think>" in content:
                    leaks += 1
                if getattr(m, "role", None) == "user" and (getattr(m, "tool_calls", None) or []):
                    user_tool_calls += 1
        avg = sum(rewards) / len(rewards) if rewards else 0.0
        if not sims:
            problems.append("0 Simulationen im Save-File")
        if n_action and n_action_ok < n_action:
            problems.append(f"action_checks nur {n_action_ok}/{n_action} — Tool-Pfad kaputt?")
        if n_env and n_env_ok < n_env:
            problems.append(f"env_assertions nur {n_env_ok}/{n_env}")
        if leaks:
            problems.append(f"{leaks} Nachrichten mit <think> im content — reasoning-parser fehlt")
        if user_tool_calls == 0:
            problems.append("User-Sim hat KEINEN eigenen Tool-Call abgesetzt — Dual-Control tot?")
        print(f"GT-Oracle: {len(sims)} Sims | ⌀reward={avg:.3f} (informativ) | "
              f"actions {n_action_ok}/{n_action} | env {n_env_ok}/{n_env} | "
              f"user-tool-calls {user_tool_calls} | think-leaks {leaks}")
        if problems:
            for p in problems:
                print(f"  ❌ {p}")
            print("❌ Oracle-Eichung GESCHEITERT — Harness/Serving pruefen, kein Modellbefund.")
            return 1
        print("✅ Oracle-Eichung bestanden (alle strukturellen Signale gruen).")
        return 0

    if not a.run:
        ap.error("--run oder --gt-check angeben")
    manifest = json.loads((a.run / "run_manifest.json").read_text())
    planned = list(manifest["domains"])
    trials = manifest["run"]["num_trials"]

    base = {}
    if a.baseline:
        bm = json.loads((a.baseline / "run_manifest.json").read_text())
        for dom in bm["domains"]:
            f = find_save(a.baseline, dom)
            if f:
                base[dom] = domain_stats(f)

    print(f"\n== tau2-bench {manifest['label']} | model={manifest['model']} | "
          f"user-sim={manifest['user_sim']['model']} | trials={trials}\n")
    hdr = (f"{'Domain':<18} {'tasks':>5} {'sims':>5} "
           + " ".join(f"pass^{k}" for k in range(1, trials + 1))
           + ("  Δp^1" if base else "")
           + "  think-leak  de%  ⌀turns  ⌀dur  terminations")
    print(hdr)
    print("-" * len(hdr))

    rc = 0
    all_stats = {}
    for dom in planned:
        f = find_save(a.run, dom)
        if f is None:
            print(f"{dom:<18}  FEHLT ({dom}/results.json) — geplant laut Manifest")
            rc = 1
            continue
        s = domain_stats(f)
        all_stats[dom] = s
        pk = " ".join(fmt_pct(s["pass_hat"].get(k)) + "%" for k in range(1, trials + 1))
        delta = ""
        if base.get(dom):
            b1, s1 = base[dom]["pass_hat"].get(1), s["pass_hat"].get(1)
            delta = f"  {100*(s1-b1):+5.1f}" if (b1 is not None and s1 is not None) else "     –"
        term = ", ".join(f"{k}:{v}" for k, v in sorted(s["terminations"].items()))
        de = round(100 * s["german_sims"] / s["simulations"], 1) if s["simulations"] else 0
        print(f"{dom:<18} {s['tasks']:>5} {s['simulations']:>5} {pk}{delta}"
              f"  {s['think_leaks']:>10}  {de:>3}  {s['avg_agent_turns']!s:>6}"
              f"  {s['avg_duration_s']!s:>4}  {term}")
        comps = "  ".join(f"{k}={fmt_pct(v['rate']).strip()}% ({v['met']}/{v['total']})"
                          for k, v in sorted(s["components"].items()))
        if comps:
            print(f"{'':<18} └─ {comps}")
        infra = s["terminations"].get(INFRA, 0)
        ctx = s["terminations"].get(CTX, 0)
        if infra:
            print(f"{'':<18} ❌ {infra} infrastructure_error — Harness/Serving pruefen")
            rc = 1
        if ctx:
            print(f"{'':<18} ⚠️  {ctx}× context_window_exceeded — Modellgrenze, "
                  "gesondert interpretieren (nicht als 0 werten)")
        if s["think_leaks"]:
            print(f"{'':<18} ❌ {s['think_leaks']} Sims mit <think> im content — "
                  "reasoning-parser fehlte beim Serving, Lauf UNGUELTIG")
            rc = 1
        if de > 10:
            print(f"{'':<18} ⚠️  {de}% Episoden antworten deutsch (Benchmark ist englisch)")

    # Held-out-Einordnung (Sternchen)
    starred = [d for d in planned if d in ("airline", "retail") and d in all_stats]
    if starred:
        print(f"\n* {', '.join(starred)}: domain-familiar (AReaL-Welt im SFT, Tasks disjunkt) "
              "— telecom/banking_knowledge sind die strikten Held-outs.")

    (a.run / "report_metrics.json").write_text(json.dumps(
        {"domains": all_stats, "trials": trials}, indent=2))
    print(f"\n== report_metrics.json geschrieben | exit={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
