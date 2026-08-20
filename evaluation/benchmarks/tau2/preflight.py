#!/usr/bin/env python3
"""tau2-bench Preflight — CPU-only Gates + run_manifest.json (Muster: bfcl/preflight.py).

Exit 0 = alle Gates gruen (WARNs erlaubt), Exit 1 = mindestens ein FAIL, mit benanntem
Fix. Sammelt ALLE Probleme, bevor gerendert wird. Laeuft unter .venv-tau2bench und
braucht TAU2_DATA_DIR bereits im Env (tau2.utils liest es beim Import).

Schreibt nach --root:  run_manifest.json  +  domains.txt

Die Benchmark-Definition (User-Sim, Sampling, Trials, Seed, Domaenen) kommt aus
benchmark_config.yaml — der User-Sim ist Teil des Messgeraets: abweichende Sim-Config
gegen existierende Laufe unter data/generated/eval/tau2/ ist ein FAIL
(TAU2_ALLOW_SIM_CHANGE=1 stuft auf WARN herab).
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
CONFIG_DEFAULT = Path(__file__).with_name("benchmark_config.yaml")
FINGERPRINT_PIN = Path(__file__).with_name("data_fingerprint.txt")
FORBIDDEN_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")
OFFLINE_RETRIEVAL = {"no_knowledge", "full_kb", "golden_retrieval", "bm25", "bm25_grep",
                     "grep_only"}
AGENT_PORT = 8000
USER_PORT = 8001


class Checks:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def ok(self, name, msg=""):
        self.rows.append((name, "OK", msg))

    def warn(self, name, msg):
        self.rows.append((name, "WARN", msg))

    def fail(self, name, msg):
        self.rows.append((name, "FAIL", msg))

    def render(self) -> bool:
        width = max(len(n) for n, _, _ in self.rows)
        bad = False
        for name, status, msg in self.rows:
            mark = {"OK": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[status]
            print(f"{mark} {name:<{width}}  {msg}")
            bad |= status == "FAIL"
        return not bad


# ---------------------------------------------------------------- helpers ---

def host_path(model: str) -> Path | None:
    """Container-Pfad (/app/data/…) → Host-Pfad; lokale Pfade unveraendert."""
    if model.startswith("/app/"):
        return REPO / model[len("/app/"):]
    p = Path(model)
    return p if p.exists() else None


def hf_snapshot(model: str) -> Path | None:
    """Juengster HF-Cache-Snapshot fuer eine HF-Modell-ID (Sichtweise HF_HOME)."""
    roots = [os.environ.get("HF_HOME"), "/data/hf_cache",
             str(Path.home() / ".cache/huggingface")]
    sub = "models--" + model.replace("/", "--")
    for r in roots:
        if not r:
            continue
        snaps = Path(r) / "hub" / sub / "snapshots"
        if snaps.is_dir():
            cands = sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime)
            if cands:
                return cands[-1]
    return None


def resolve_model_dir(model: str) -> Path | None:
    return host_path(model) or hf_snapshot(model)


def read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def fingerprint_model(d: Path) -> str:
    h = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "generation_config.json"):
        f = d / name
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def fingerprint_data(domains_root: Path, domains: dict) -> str:
    h = hashlib.sha256()
    for dom, spec in sorted(domains.items()):
        ddir = domains_root / dom
        for f in sorted(ddir.glob("*task*.json")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def derive_alias(model: str) -> str:
    """Serving-Alias (--served-model-name): kurz, eindeutig, litellm-tauglich."""
    parts = [p for p in model.rstrip("/").split("/") if p]
    tail = parts[-2:] if (model.startswith("/") or model.startswith("data/")) else parts[-1:]
    slug = re.sub(r"[^a-z0-9.-]+", "-", "-".join(tail).lower()).strip("-")
    return slug or "model"


def autosize_utils(cfg: dict, agent_dir: Path | None, total_cap: float | None = None):
    """GPU-Utils aus der Agent-Modellgroesse berechnen (Auto-Sizing).

    agent_base = (weights + overhead + kv_target) / total_mem
    sim  = clamp(cap − agent_base, sim_min, sim_max)
    agent = cap − sim   (Restluft an den Agenten — KV-Seiten materialisieren lazy)

    Returns (agent_util, sim_util, info) oder (None, None, fehlertext).
    """
    az = cfg["autosize"]
    cap = float(total_cap or az["total_cap"])
    mem = float(az["total_mem_gb"])
    if agent_dir is None:
        return None, None, "Modellverzeichnis nicht aufloesbar — Gewichtsgroesse unbekannt"
    weights_gb = sum(f.stat().st_size for f in agent_dir.glob("*.safetensors")) / 1e9
    if weights_gb < 0.5:
        return None, None, f"keine/zu kleine safetensors in {agent_dir} ({weights_gb:.2f} GB)"
    agent_base = (weights_gb + az["agent_overhead_gb"] + az["agent_kv_target_gb"]) / mem
    sim = max(az["sim_util_min"], min(az["sim_util_max"], cap - agent_base))
    sim = round(sim, 2)
    agent = round(cap - sim, 2)
    info = {"weights_gb": round(weights_gb, 1), "agent_base": round(agent_base, 3),
            "total_cap": cap, "kv_target_gb": az["agent_kv_target_gb"],
            # Sweep 2026-08-20: 4B-Optimum MC12 (17,3/18,5/15,1 Ep/h @ 8/12/16),
            # 8B saturiert frueher (12,3/11,6 @ 8/12) -> Empfehlung nach Gewichtsgroesse
            "recommended_mc": 8 if weights_gb > 12 else 12}
    if agent + 1e-9 < agent_base:
        return None, None, (f"Agent-Modell zu gross fuer Doppel-Serving: braucht util "
                            f"{agent_base:.2f} ({weights_gb:.1f} GB Gewichte + KV-Ziel), "
                            f"aber cap {cap} − sim_min {az['sim_util_min']} laesst nur "
                            f"{agent:.2f} — kv_target senken oder total_cap bewusst anheben")
    return agent, sim, info


def check_generation_config(checks, name, model, expected):
    d = resolve_model_dir(model)
    if d is None:
        checks.warn(f"gen-config {name}",
                    f"Modellverzeichnis fuer {model} nicht aufloesbar — Sampling ungeprueft "
                    "(HF_HOME exportiert?)")
        return None
    gc = read_json(d / "generation_config.json")
    if gc is None:
        checks.fail(f"gen-config {name}", f"{d}/generation_config.json fehlt/unlesbar")
        return None
    diffs = {k: (gc.get(k), v) for k, v in expected.items() if gc.get(k) != v}
    if diffs:
        checks.fail(f"gen-config {name}",
                    f"Abweichung vom erwarteten Rezept {expected}: {diffs} — tau2 sendet "
                    "top_p/top_k NICHT, sie kommen von HIER.")
    else:
        checks.ok(f"gen-config {name}", f"{expected} bestaetigt ({d.name})")
    return d


# ------------------------------------------------------------------- main ---

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Agent-Modell (HF-ID oder /app/data/…-Pfad)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--domains", default="full", help="'full' oder Kommaliste aus der Config")
    ap.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    ap.add_argument("--num-trials", type=int, default=None)
    ap.add_argument("--max-concurrency", type=int, default=None)
    ap.add_argument("--total-cap", type=float, default=None,
                    help="Autosize-Gesamtdeckel uebersteuern (Cap-Leiter-Tests)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="max_steps uebersteuern (Sweep-Protokoll; Benchmark-Default: Config)")
    a = ap.parse_args()

    checks = Checks()
    cfg = yaml.safe_load(a.config.read_text())

    # --- Env-Gates (vor jedem tau2-Import) ---
    present = [k for k in FORBIDDEN_KEYS if os.getenv(k)]
    if present:
        checks.fail("api-keys", f"{present} im Env — verboten. Vergessene gpt-4.1-Defaults "
                                "muessen laut crashen, nie still Geld kosten. unset + retry.")
    else:
        checks.ok("api-keys", "kein Fremd-API-Key im Env")

    data_dir = os.getenv("TAU2_DATA_DIR")
    # tau2s Loader bauen selbst DATA_DIR/"tau2"/"domains"/… → TAU2_DATA_DIR = …/data
    if not data_dir or not (Path(data_dir) / "tau2" / "domains").is_dir():
        checks.fail("data-dir", f"TAU2_DATA_DIR fehlt oder traegt kein tau2/domains/: "
                                f"{data_dir!r} — Fix: export "
                                "TAU2_DATA_DIR=$REPO/data/raw/tau2-bench/data")
        print(); checks.render()
        return 1
    data_dir = Path(data_dir)
    domains_root = data_dir / "tau2" / "domains"

    # --- Pin-Gates: Checkout-Commit + Paketversion ---
    checkout = data_dir.parent  # …/tau2-bench/data → …/tau2-bench
    head = git_head(checkout)
    if head != cfg["tau2_commit"]:
        checks.fail("data-pin", f"Checkout {checkout} steht auf {head}, erwartet "
                                f"{cfg['tau2_commit']} (Tag v{cfg['tau2_version']})")
    else:
        checks.ok("data-pin", f"Checkout = v{cfg['tau2_version']} ({head[:12]})")

    try:
        import tau2  # noqa: F401  (liest TAU2_DATA_DIR beim Import)
        from importlib.metadata import version as pkg_version
        v = pkg_version("tau2")
        if v != cfg["tau2_version"]:
            checks.fail("package", f"tau2=={v} im venv, erwartet {cfg['tau2_version']} — "
                                   "Fix: .venv-tau2bench/bin/pip install ./data/raw/tau2-bench")
        else:
            checks.ok("package", f"tau2=={v}")
    except Exception as e:
        checks.fail("package", f"tau2 nicht importierbar: {e}")
        print(); checks.render()
        return 1

    # --- Domaenen aufloesen + Task-Zaehlungen (ueber die ECHTEN Loader) ---
    all_domains = cfg["domains"]
    if a.domains in ("full", ""):
        selected = dict(all_domains)
    else:
        names = [d.strip() for d in a.domains.split(",") if d.strip()]
        unknown = [d for d in names if d not in all_domains]
        if unknown:
            checks.fail("domains", f"unbekannt: {unknown} (Config kennt {list(all_domains)})")
            selected = {d: all_domains[d] for d in names if d in all_domains}
        else:
            selected = {d: all_domains[d] for d in names}

    from tau2.registry import registry
    for dom, spec in selected.items():
        try:
            loader = registry.get_tasks_loader(spec["task_set"])
            tasks = loader(spec.get("task_split")) if spec.get("task_split") else loader(None)
            n = len(tasks)
            spec["counted_tasks"] = n
            if n != spec["expected_tasks"]:
                checks.fail(f"tasks {dom}", f"{n} geladen, erwartet {spec['expected_tasks']} "
                                            f"(set={spec['task_set']}, split={spec.get('task_split')})")
            else:
                checks.ok(f"tasks {dom}", f"{n} Tasks (split={spec.get('task_split')})")
        except Exception as e:
            checks.fail(f"tasks {dom}", f"Loader-Fehler: {e}")

    # --- Daten-Fingerprint (tau2 check-data prueft nur Existenz — wir pinnen Inhalt) ---
    fp = fingerprint_data(domains_root, all_domains)
    if FINGERPRINT_PIN.exists():
        pinned = FINGERPRINT_PIN.read_text().strip()
        if pinned != fp:
            checks.fail("data-fingerprint", f"{fp} != gepinnt {pinned} — Daten haben sich "
                                            f"geaendert; bewusstes Neu-Pinnen = {FINGERPRINT_PIN} loeschen")
        else:
            checks.ok("data-fingerprint", fp)
    else:
        FINGERPRINT_PIN.write_text(fp + "\n")
        checks.ok("data-fingerprint", f"{fp} (erstmalig gepinnt → {FINGERPRINT_PIN.name})")

    # --- banking: Offline-Retrieval erzwingen + Retrieval-Deps (Laufzeit-Import!) ---
    for dom, spec in selected.items():
        rc = spec.get("retrieval_config")
        if dom == "banking_knowledge":
            if rc not in OFFLINE_RETRIEVAL:
                checks.fail("retrieval", f"banking retrieval_config={rc!r} ist nicht offline "
                                         f"(erlaubt: {sorted(OFFLINE_RETRIEVAL)})")
            else:
                checks.ok("retrieval", f"banking: {rc} (offline)")
            # tau2s Basis-Install deklariert rank_bm25 NICHT; der bm25-Indexer importiert es
            # erst beim Env-Aufbau -> ohne Gate crasht der Lauf zur Laufzeit (Smoke 2026-08-20).
            try:
                import rank_bm25  # noqa: F401
                checks.ok("retrieval-deps", "rank_bm25 importierbar")
            except ImportError:
                checks.fail("retrieval-deps", "rank_bm25 fehlt — Fix: "
                                              ".venv-tau2bench/bin/pip install rank_bm25")

    # --- Sampling-Gates fuer BEIDE Modelle ---
    agent_dir = check_generation_config(checks, "agent", a.model,
                                        cfg["agent"]["expected_generation_config"])
    check_generation_config(checks, "user-sim", cfg["user_sim"]["model"],
                            cfg["user_sim"]["expected_generation_config"])

    # --- Kontext-Gates ---
    mml = int(cfg["agent"]["max_model_len"])
    if agent_dir is not None:
        conf = read_json(agent_dir / "config.json") or {}
        mpe = conf.get("max_position_embeddings")
        if mpe is None:
            checks.fail("context", f"config.json ohne max_position_embeddings ({agent_dir})")
        elif mml > mpe:
            checks.fail("context", f"agent.max_model_len {mml} > mpe {mpe} — vLLM startet nicht")
        elif mml < 16384:
            checks.fail("context", f"agent.max_model_len {mml} < 16384 — telecom-Episoden "
                                   "(Policy + Tools + Verlauf) reissen das still")
        else:
            checks.ok("context", f"agent {mml} ≤ mpe {mpe}")

    # --- Auto-Sizing der GPU-Utils (aus der Agent-Modellgroesse) ---
    agent_util, sim_util, az_info = autosize_utils(cfg, agent_dir, a.total_cap)
    if agent_util is None:
        checks.fail("autosize", str(az_info))
    else:
        checks.ok("autosize", f"agent {agent_util} / sim {sim_util} "
                              f"(cap {az_info['total_cap']}, Gewichte {az_info['weights_gb']} GB, "
                              f"KV-Ziel {az_info['kv_target_gb']} GB)")

    # --- Ein Label = ein Modell (Resume-Schutz) ---
    prev = read_json(a.root / "run_manifest.json")
    if prev and prev.get("model") != a.model:
        checks.fail("label-dir", f"{a.root} traegt bereits Modell {prev.get('model')!r} — "
                                 "neues Modell braucht ein neues Label")
    else:
        checks.ok("label-dir", "kein Modell-Konflikt im Label-Verzeichnis")

    # --- Label-Grammatik (WARN) ---
    alias = derive_alias(a.model)
    first = a.label.split("_")[0]
    if len(first) < 4 or first not in f"{alias} {a.model.lower()}":
        checks.warn("label", f"'{a.label}' folgt nicht <modell>_<stand>[_<zweck>] "
                             f"(erster Block '{first}' passt nicht zu '{alias}')")
    else:
        checks.ok("label", a.label)

    # --- Frozen-Sim-Gate: Serienvergleichbarkeit ---
    sim_def = {"model": cfg["user_sim"]["model"], "llm_args": cfg["user_sim"]["llm_args"],
               "seed": cfg["run"]["seed"]}
    conflicts = []
    for m in (a.root.parent).glob("*/run_manifest.json"):
        old = read_json(m) or {}
        # Manifeste speichern die Args ANGEREICHERT (api_base/api_key) — vor dem
        # Vergleich strippen, sonst kollidiert jedes Label mit sich selbst (Resume).
        old_args = {k: v for k, v in (((old.get("user_sim") or {}).get("llm_args")) or {}).items()
                    if not k.startswith("api_")}
        old_sim = {"model": (old.get("user_sim") or {}).get("model"),
                   "llm_args": old_args,
                   "seed": (old.get("run") or {}).get("seed")}
        if old_sim["model"] and old_sim != sim_def:
            conflicts.append(m.parent.name)
    if conflicts:
        msg = (f"User-Sim-Definition weicht von existierenden Laeufen ab: {conflicts} — "
               "der Sim ist Teil des Messgeraets; TAU2_ALLOW_SIM_CHANGE=1 uebersteuert bewusst")
        if os.getenv("TAU2_ALLOW_SIM_CHANGE") == "1":
            checks.warn("frozen-sim", msg)
        else:
            checks.fail("frozen-sim", msg)
    else:
        checks.ok("frozen-sim", f"{sim_def['model']} @ {sim_def['llm_args']}")

    # --- Disjunktheits-Gate (AReaL-SFT-Leg vs. offizielle Tasks) ---
    sys.path.insert(0, str(Path(__file__).parent))
    import check_disjoint
    dj = check_disjoint.run_check(data_dir / "tau2",
                                  REPO / "data/generated/legs/areal_chat.jsonl")
    if dj["ok"] is None:
        checks.warn("disjoint", dj.get("error", "Leg fehlt — Check uebersprungen"))
    elif dj["ok"]:
        ov = dj.get("airline_name_overlap") or {}
        checks.ok("disjoint", f"0 woertliche Treffer, 0 telecom; Welt-Overlap "
                              f"{ov.get('in_leg')}/{ov.get('db_names')} airline-Namen (bekannt)")
    else:
        checks.fail("disjoint", f"Ueberschneidung mit dem SFT-Leg! {json.dumps(dj['domains'])}")

    # --- Manifest schreiben ---
    def with_endpoint(args: dict, port: int) -> dict:
        return {**args, "api_base": f"http://localhost:{port}/v1", "api_key": "dummy"}

    trials = a.num_trials or cfg["run"]["num_trials"]
    # MC-Prioritaet: expliziter Override (TAU2_MC) > modellabhaengige Sweep-Empfehlung > Config
    mc = a.max_concurrency or (az_info.get("recommended_mc") if isinstance(az_info, dict)
                               else None) or cfg["run"]["max_concurrency"]
    manifest = {
        "label": a.label,
        "model": a.model,
        "model_alias": alias,
        "model_fingerprint": fingerprint_model(agent_dir) if agent_dir else None,
        "agent": {
            "llm": f"openai/{alias}",
            "llm_args": with_endpoint(cfg["agent"]["llm_args"], AGENT_PORT),
            "max_model_len": mml,
            "max_num_seqs": cfg["agent"]["max_num_seqs"],
            "gpu_util": agent_util,
            "tool_call_parser": cfg["agent"]["tool_call_parser"],
            "reasoning_parser": cfg["agent"]["reasoning_parser"],
            "port": AGENT_PORT,
        },
        "user_sim": {
            "model": cfg["user_sim"]["model"],
            "llm": f"openai/{cfg['user_sim']['model']}",
            "llm_args": with_endpoint(cfg["user_sim"]["llm_args"], USER_PORT),
            "max_model_len": cfg["user_sim"]["max_model_len"],
            "max_num_seqs": cfg["user_sim"]["max_num_seqs"],
            "gpu_util": sim_util,
            "extra_args": cfg["user_sim"].get("extra_args", ""),
            "tool_call_parser": cfg["user_sim"]["tool_call_parser"],
            "reasoning_parser": cfg["user_sim"]["reasoning_parser"],
            "port": USER_PORT,
        },
        "nl_judge": {
            "model_litellm": f"openai/{cfg['nl_judge']['model']}",
            "llm_args": with_endpoint(cfg["nl_judge"]["llm_args"], USER_PORT),
        },
        "autosize": az_info if isinstance(az_info, dict) else None,
        "run": {**cfg["run"], "num_trials": trials, "max_concurrency": mc,
                "max_steps": a.max_steps or cfg["run"]["max_steps"]},
        "domains": selected,
        "oracle": cfg.get("oracle"),
        "tau2_version": cfg["tau2_version"],
        "tau2_commit": cfg["tau2_commit"],
        "data_dir": str(data_dir),
        "data_fingerprint": fp,
        "disjoint": {k: v for k, v in dj.items() if k != "domains"} | {
            "verbatim_hits": sum(d["verbatim_hits"] for d in dj.get("domains", {}).values())
            if dj.get("domains") else None},
        "repo_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    print()
    good = checks.render()
    if good:
        a.root.mkdir(parents=True, exist_ok=True)
        (a.root / "run_manifest.json").write_text(json.dumps(manifest, indent=2,
                                                             ensure_ascii=False))
        (a.root / "domains.txt").write_text("\n".join(selected) + "\n")
        print(f"\n== Manifest: {a.root / 'run_manifest.json'}")
        print(f"== Domaenen: {', '.join(selected)} | trials={trials} | conc={mc}")
    else:
        print("\n== PREFLIGHT FAILED — nichts geschrieben.")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
