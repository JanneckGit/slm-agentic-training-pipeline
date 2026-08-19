#!/usr/bin/env python3
"""BFCL preflight — CPU-only gate that fails BEFORE anything is served.

A BFCL full run takes 16-40 h per model. It must not die in hour 9 on a missing import, and it must
certainly not run to completion with silently wrong sampling. This script checks everything that is
checkable without a GPU, resolves the category list, and writes the `run_manifest.json` that later
identifies the run.

Usage (called by ops/eval_bfcl.sh, but runs standalone too):
    .venv-bfcl/bin/python evaluation/benchmarks/bfcl/preflight.py \
        --model Qwen/Qwen3-4B --label qwen3-4b_base \
        --root data/generated/eval/bfcl/qwen3-4b_base --categories full

Exit 0 = all green. Exit 1 = hard failure (the message names the fix).
Writes into --root:  run_manifest.json  and  categories.txt (resolved list, for the shell).
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

REPO = Path(__file__).resolve().parents[3]

# Expected sampling. BFCL sends ONLY temperature + max_tokens; top_p/top_k come from the
# generation_config.json of the SERVED model (vLLM reads it by default). If that deviates, we
# measure a different configuration than every other eval in the repo — and base vs trained stop
# being comparable. Values = the Qwen3 student recipe, identical to ops/eval_heldout.sh.
EXPECTED_GENERATION_CONFIG = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}

# Registry mode. bfcl-eval has no key for the hybrid Qwen3-4B, and borrowing a foreign key means
# borrowing its CONTEXT WINDOW: bfcl derives `max_context_length` from the registry entry's HF
# config, not from the served model. With `Qwen/Qwen3-4B-Instruct-2507-FC` (262,144) the client
# considered every prompt admissible that our 40,960 server rejected -> silent 0 % items.
# Default is therefore to DERIVE a key (`local/<slug>-FC`) and inject an entry whose model_name
# IS the served model (see registry_inject.py). `--model-key` selects a REAL registry key instead
# (no injection) — that is the harness-validation path against official leaderboard rows.
import registry_inject


class Check:
    """Collects results so ALL problems are reported at once instead of one per run."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.failed = False

    def ok(self, name: str, detail: str = "") -> None:
        self.rows.append(("OK", name, detail))

    def warn(self, name: str, detail: str = "") -> None:
        self.rows.append(("WARN", name, detail))

    def fail(self, name: str, detail: str = "") -> None:
        self.rows.append(("FAIL", name, detail))
        self.failed = True

    def render(self) -> None:
        w = max(len(n) for _, n, _ in self.rows)
        for status, name, detail in self.rows:
            mark = {"OK": "  ok  ", "WARN": " warn ", "FAIL": " FAIL "}[status]
            print(f"[{mark}] {name:<{w}}  {detail}")


def host_path(model: str) -> Path | None:
    """Map container paths to host paths.

    ops/eval_bfcl.sh passes the model exactly as vLLM sees it INSIDE THE CONTAINER (/app/data/...).
    The preflight runs on the HOST, where the same directory is <repo>/data/...
    """
    if model.startswith("/app/"):
        p = REPO / model[len("/app/"):]
    else:
        p = Path(model)
    return p if p.exists() else None


def hf_snapshot(model_id: str) -> Path | None:
    """Snapshot directory of an HF model in the local cache (no network)."""
    for base in (os.getenv("HF_HOME"), "/data/hf_cache", Path.home() / ".cache/huggingface"):
        if not base:
            continue
        hub = Path(base) / "hub"
        if not hub.is_dir():
            continue
        cand = hub / ("models--" + model_id.replace("/", "--")) / "snapshots"
        if cand.is_dir():
            snaps = sorted(cand.iterdir())
            if snaps:
                return snaps[-1]
    return None


def harness_snapshot(model_id: str) -> Path | None:
    """Snapshot dir as the HARNESS will see it — HF_HOME, else ~/.cache. Nothing else.

    Deliberately narrower than hf_snapshot(): transformers inside the bfcl venv resolves via
    HF_HOME or the default cache, it does NOT know our /data/hf_cache convention. A check that
    probes more paths than the harness can pass while the actual AutoConfig load fails.
    """
    base = os.getenv("HF_HOME") or (Path.home() / ".cache/huggingface")
    hub = Path(base) / "hub"
    cand = hub / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if cand.is_dir():
        snaps = sorted(cand.iterdir())
        if snaps:
            return snaps[-1]
    return None


def read_mpe(model_dir: Path | None) -> int | None:
    """max_position_embeddings from a model directory's config.json."""
    cfg = model_dir / "config.json" if model_dir else None
    if cfg and cfg.is_file():
        try:
            return int(json.loads(cfg.read_text()).get("max_position_embeddings"))
        except Exception:
            return None
    return None


def fingerprint(model_dir: Path) -> str:
    """Cheap, deterministic checkpoint fingerprint.

    Do not hash the weights (8 GB) — config.json plus the safetensors index identify the checkpoint
    structurally and cost milliseconds.
    """
    h = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "generation_config.json"):
        f = model_dir / name
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def check_deps(c: Check) -> None:
    try:
        import numpy
        c.ok("numpy", numpy.__version__)
    except Exception as e:  # pragma: no cover — no numpy means the venv is broken
        c.fail("numpy", str(e))
        return
    for mod, why in (("faiss", "memory_vector (192 items) drops out without it"),
                     ("sentence_transformers", "encoder for memory_vector"),
                     ("soundfile", "the CLI's qwen_agent import")):
        try:
            __import__(mod)
            c.ok(mod, "importable")
        except Exception as e:
            fix = ""
            if mod == "faiss" and "numpy.distutils" in str(e):
                fix = " -> fix: .venv-bfcl/bin/pip install 'numpy>=2.0'"
            c.fail(mod, f"{why}: {type(e).__name__}{fix}")


def check_registry(c: Check, model_key: str) -> str:
    """-> the name BFCL addresses the model by (= the required --served-model-name alias)."""
    try:
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    except Exception as e:
        c.fail("bfcl-eval", f"not importable: {e}")
        return ""
    if model_key not in MODEL_CONFIG_MAPPING:
        c.fail("registry key", f"'{model_key}' unknown (bfcl models lists the valid ones)")
        return ""
    alias = MODEL_CONFIG_MAPPING[model_key].model_name
    c.ok("registry key", f"{model_key} -> alias '{alias}'")
    return alias


def check_generation_config(c: Check, model: str, strict: bool = True) -> dict:
    """Sampling gate: the served model's top_p/top_k against the student recipe.

    strict=True (our own models): any deviation is a hard FAIL — base vs. trained must sample
    identically. strict=False (real registry key, e.g. the leaderboard harness validation):
    deviations only WARN, the actual values land in the manifest — there the goal is
    leaderboard comparability, not the student recipe.
    """
    flag = c.fail if strict else c.warn
    d = host_path(model) or hf_snapshot(model)
    if d is None:
        c.warn("generation_config", f"no local model directory for '{model}' — sampling unchecked "
                                    "(vLLM pulls it when serving)")
        return {}
    f = d / "generation_config.json"
    if not f.is_file():
        flag("generation_config", f"{f} missing -> vLLM falls back to its own defaults, "
                                  "sampling would not be the student recipe")
        return {}
    got = json.loads(f.read_text())
    bad = {k: (got.get(k), v) for k, v in EXPECTED_GENERATION_CONFIG.items() if got.get(k) != v}
    if bad:
        detail = ", ".join(f"{k}={a!r} instead of {b!r}" for k, (a, b) in bad.items())
        flag("generation_config", f"{detail}  ({f})")
    else:
        c.ok("generation_config", "temp 0.6 / top_p 0.95 / top_k 20 — student recipe")
    return got


def check_registry_context(c: Check, model_name: str, served_mpe: int | None) -> None:
    """THE regression gate for the 262,144 trap: registry ctx must EQUAL served ctx.

    bfcl computes `max_tokens = min(4096, max_context_length - input - 2)` from the config of the
    registry entry's `model_name`, resolved through the HARNESS's cache view. If that differs from
    the served model, the client either believes prompts fit that the server rejects (too big:
    silent 0 % items — the original bug, 262,144 vs 40,960) or clamps output for no reason (too
    small). Both directions are wrong, so: equality, hard.
    """
    p = Path(model_name)
    d = p if p.is_dir() else harness_snapshot(model_name)
    if d is None:
        c.fail("registry-context", f"'{model_name}' not resolvable in the harness cache "
                                   f"(HF_HOME={os.getenv('HF_HOME') or 'unset -> ~/.cache'}) — "
                                   "AutoConfig/AutoTokenizer load would die at generate time. "
                                   "Fix: export HF_HOME=/data/hf_cache")
        return
    registry_mpe = read_mpe(d)
    if registry_mpe is None:
        c.fail("registry-context", f"no max_position_embeddings readable under {d}")
        return
    if served_mpe is not None and registry_mpe != served_mpe:
        c.fail("registry-context", f"registry model carries {registry_mpe}, served model "
                                   f"{served_mpe} — bfcl would budget max_tokens against the "
                                   "WRONG window (the 262,144 trap). Use an injected key or a "
                                   "registry key whose model matches the served context")
        return
    c.ok("registry-context", f"{registry_mpe} == served model — bfcl budgets against the true window")


def check_context(c: Check, limit: int | None, max_model_len: int, cats: list[str]) -> None:
    """Check the serving window from both sides — two runs have already died here.

    UPWARD: `max_position_embeddings` of the SERVED model (`limit`). Qwen3-4B tops out at 40,960;
    vLLM flatly refuses anything larger and the serve step fails (loudly, at least).
    DOWNWARD: BFCL requests max_tokens = min(4096, ctx - input - 2). If the window is too small for
    a prompt, the request does NOT go through and lands as input_token_count 0 in the result —
    silently, as if the model had failed. That is exactly how 4 of 5 multi_turn_long_context
    episodes scored 0 % in the old quick run.

    (The historical third failure mode — bfcl budgeting against the REGISTRY model's 262,144
    instead of the served 40,960 — is now closed by construction: the injected entry's model_name
    IS the served model, and check_registry_context() enforces equality for real keys too.)
    """
    if limit is not None and max_model_len > limit:
        c.fail("max-model-len", f"{max_model_len} > max_position_embeddings={limit} of the served "
                                f"model -> vLLM refuses to start")
        return
    # 23,503 input tokens measured on multi_turn_miss_param; +4,096 max_tokens = ~27,600.
    if any(x.startswith("multi_turn") for x in cats) and max_model_len < 32768:
        c.fail("max-model-len", f"{max_model_len} too tight for multi_turn (measured: miss_param "
                                "reaches 23,503 input tokens, plus 4,096 max_tokens)")
        return
    c.ok("max-model-len", f"{max_model_len}" + (f" (model limit {limit})" if limit else ""))


def resolve_categories(c: Check, spec: str, web_search: bool) -> list[str]:
    from bfcl_eval.constants.category_mapping import (ALL_CATEGORIES, ALL_SCORING_CATEGORIES,
                                                      TEST_COLLECTION_MAPPING)

    if spec == "full":
        # Default = every SCORING category. format_sensitivity is deliberately excluded: bfcl lists
        # it as a NON_SCORING_CATEGORY, it only yields a max-delta/std figure — and it expands to
        # 5,200 entries (200 base items x 26 format configs), more than all scoring categories put
        # together. If you want it anyway: --categories everything.
        cats = list(ALL_SCORING_CATEGORIES)
    elif spec == "everything":
        cats = list(ALL_CATEGORIES)
    elif spec in TEST_COLLECTION_MAPPING:
        cats = list(TEST_COLLECTION_MAPPING[spec])
    else:
        cats = [x.strip() for x in spec.split(",") if x.strip()]
        unknown = [x for x in cats if x not in ALL_CATEGORIES]
        if unknown:
            c.fail("categories", f"unknown: {unknown}")
            return []

    skipped = []
    if not web_search:
        skipped = [x for x in cats if x.startswith("web_search")]
        cats = [x for x in cats if not x.startswith("web_search")]
    elif not os.getenv("SERPAPI_API_KEY"):
        # Deliberately hard: with the category enabled and no key, the run walks into the
        # `while True` retry loop in web_search.py and hangs unattended.
        c.fail("web_search", "BFCL_ENABLE_WEB_SEARCH=1 but SERPAPI_API_KEY is missing")

    if skipped:
        c.warn("web_search", f"skipped ({', '.join(skipped)}) — recorded as 'skipped' in the "
                             "manifest, NOT scored as 0 %")
    return cats


def check_memory_vector_model(c: Check, cats: list[str]) -> None:
    """memory_vector needs an embedding model IN THE HARNESS CACHE — importable is not enough.

    `memory_vector.py` instantiates `SentenceTransformer("all-MiniLM-L6-v2")` at MODULE level:
    the first import tries to DOWNLOAD the model into the effective HF cache. With
    HF_HOME=/data/hf_cache that cache is root-owned -> Permission denied -> every single
    vector item (192) dies instantly as an `infra` error, discovered hours into the run.
    Exactly that killed the 2026-08-18 base run's vector category. Hence: hard gate.
    """
    if "memory_vector" not in cats:
        return
    # Read the model id from the installed package (survives upstream changes); fall back to
    # the known id if the pattern moves.
    model_id = "all-MiniLM-L6-v2"
    try:
        import bfcl_eval
        src = (Path(bfcl_eval.__file__).parent / "eval_checker/multi_turn_eval"
               / "func_source_code/memory_vector.py").read_text()
        m = re.search(r'SentenceTransformer\(\s*"([^"]+)"', src)
        if m:
            model_id = m.group(1)
    except Exception:
        pass
    repo = model_id if "/" in model_id else f"sentence-transformers/{model_id}"
    base = os.getenv("HF_HOME") or (Path.home() / ".cache/huggingface")
    snap_root = Path(base) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots"
    snaps = sorted(snap_root.iterdir()) if snap_root.is_dir() else []
    if not snaps:
        c.fail("memory_vector model",
               f"'{repo}' not in the harness cache ({base}) — the module-level download would "
               f"die on the root-owned cache. Fix: docker compose -f docker/docker-compose.yml "
               f"run --rm -T sdg python3 -c \"from huggingface_hub import snapshot_download; "
               f"snapshot_download('{repo}')\"")
        return
    unreadable = [str(p) for p in snaps[-1].rglob("*") if p.is_file() and not os.access(p, os.R_OK)]
    if unreadable:
        c.fail("memory_vector model",
               f"'{repo}' cached but {len(unreadable)} file(s) not host-readable "
               f"(e.g. {unreadable[0]}) — fix: docker compose … run --rm -T sdg "
               f"chmod -R a+rX /data/hf_cache/hub/models--{repo.replace('/', '--')}")
        return
    c.ok("memory_vector model", f"'{repo}' cached and readable ({base})")


def check_label(c: Check, label: str, model: str) -> None:
    """Label-grammar lint: `<modell>_<stand>[_<zweck>]` — WARN only, labels are a human choice.

    The first `_`-segment must be the model slug (>= 6 chars, re-findable in the model id/path),
    so `ls data/generated/eval/bfcl/` groups a model's runs. `<stand>` names the checkpoint state
    (base | sft-ep3 | grpo-s2-ep1 | harness-validation | ...), the optional `<zweck>` marks
    throwaways (smoke-<x> | probe-<x>) or a deliberate re-measurement (r2 — same label would
    RESUME the existing run instead of measuring anew).
    """
    seg = label.split("_", 1)[0]
    slug = registry_inject.derive_slug(model)
    if len(seg) < 6 or seg.lower() not in f"{slug} {model.lower()}":
        c.warn("label", f"'{label}' does not follow <modell>_<stand>[_<zweck>]: first segment "
                        f"'{seg}' is not a model slug (expected e.g. 'qwen3-4b_...'; derived "
                        f"slug here: '{slug}') — runs of one model will not group in ls")
    else:
        c.ok("label", f"'{label}' follows <modell>_<stand>[_<zweck>]")


def check_format_sensitivity(c: Check, cats: list[str], model_key: str) -> None:
    """format_sensitivity + an FC model = guaranteed zero results.

    `_llm_response_generation.py` filters format_sensitivity test cases out for FC models ("Skip
    format sensitivity test cases for FC models"). Our key ends in -FC, so the category would
    contribute 5,200 planned entries and produce not a single one.
    """
    if any(x.startswith("format_sensitivity") for x in cats) and model_key.endswith("-FC"):
        c.warn("format_sensitivity", f"bfcl skips it for FC models ({model_key}) — the category "
                                     "is guaranteed to produce 0 results")


def inventory(cats: list[str]) -> dict[str, int]:
    """Items per category — via bfcl's OWN loader, not by counting file lines.

    The raw files lie: memory_* pulls in 37 prereq entries (155 -> 192), and format_sensitivity
    expands 200 base items into 5,200 (x26 format configs). Counting lines plans the run wrong by
    a factor of two.
    """
    from bfcl_eval.utils import load_dataset_entry
    out = {}
    for cat in cats:
        try:
            out[cat] = len(load_dataset_entry(cat))
        except Exception:
            out[cat] = 0
    return out


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "?"
    except Exception:
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model path (as vLLM sees it) or HF id")
    ap.add_argument("--label", required=True)
    ap.add_argument("--root", required=True, type=Path, help="BFCL_PROJECT_ROOT of this run")
    ap.add_argument("--categories", default="full", help="'full', a bfcl group, or a comma list")
    ap.add_argument("--model-key", default=None,
                    help="REAL bfcl registry key -> no injection (harness validation against "
                         "official leaderboard rows). Default: derive local/<slug>-FC and inject "
                         "an entry pointing at the served model (see registry_inject.py)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="serving window; default: max_position_embeddings of the served model")
    # Concurrency: measured group-dependent (probe 2026-08-18, 4B on GB10, 8B-verified for fast).
    # fast = non_live+live (short independent items, t90 1.7x over 21); slow = everything else
    # (multi_turn/memory are serial chains — 48 there is 47 % SLOWER plus timeout risk).
    ap.add_argument("--num-threads", type=int, default=None,
                    help="UNIFORM thread count for ALL groups (old behavior); overrides the split")
    ap.add_argument("--num-threads-fast", type=int, default=48,
                    help="non_live+live (measured knee 2026-08-18, 8B-verified)")
    ap.add_argument("--num-threads-slow", type=int, default=21,
                    help="multi_turn+memory+rest (serial chains — do not raise blindly)")
    ap.add_argument("--enable-web-search", action="store_true")
    ap.add_argument("--run-ids", type=Path, help="test_case_ids_to_generate.json (smoke/probe mode)")
    a = ap.parse_args()

    print(f"==== BFCL PREFLIGHT  {a.label}  ({a.model}) ====")
    c = Check()
    check_deps(c)

    # --- registry mode -----------------------------------------------------------------------
    # Injected (default): entry's model_name = the HOST-resolved served model, so bfcl's context
    # budget, the request's `model=` field and the vLLM alias all agree by construction.
    # Real key (--model-key): the stock entry as-is — its model must MATCH the served one, which
    # check_registry_context enforces below.
    model_dir = host_path(a.model) or hf_snapshot(a.model)
    served_mpe = read_mpe(model_dir)
    hp = host_path(a.model)
    source = str(hp.resolve()) if hp else a.model     # checkpoint -> host path, hub model -> HF id
    if a.model_key:
        registry = {"injected": False, "model_key": a.model_key,
                    "model_name": None, "display_name": None, "base_key": None}
    else:
        registry = registry_inject.injection_spec(a.model, source)
        try:
            registry_inject.inject(registry)
        except Exception as e:
            c.fail("registry key", f"injection failed: {e}")
    alias = check_registry(c, registry["model_key"])
    if registry["injected"] is False:
        registry["model_name"] = alias or None

    # Sampling gate is strict only for OUR models: base vs. trained must sample identically.
    # A real registry key (leaderboard validation) records deviations instead of blocking on them.
    gen_cfg = check_generation_config(c, a.model, strict=not a.model_key)
    if alias:
        check_registry_context(c, alias, served_mpe)

    max_model_len = a.max_model_len
    if max_model_len is None:
        max_model_len = served_mpe   # check_context prints/validates the final value below
        if max_model_len is None:
            c.fail("max-model-len", f"not given and not derivable (no local config.json for "
                                    f"'{a.model}') — set VLLM_MAX_MODEL_LEN explicitly")

    # A label's result/ must hold exactly ONE model: bfcl_report globs category-wise across all
    # model dirs, two models under one label would be summed into a single mongrel score.
    expected_dir = registry["model_key"].replace("/", "_")
    result_root = a.root / "result"
    if result_root.is_dir():
        foreign = sorted(p.name for p in result_root.iterdir() if p.is_dir() and p.name != expected_dir)
        if foreign:
            c.fail("result-dir", f"{result_root} already holds foreign model dir(s) {foreign} — "
                                 "bfcl_report would sum two models into one label. Use a fresh "
                                 "label or remove them")

    check_label(c, a.label, a.model)
    cats = resolve_categories(c, a.categories, a.enable_web_search)
    check_format_sensitivity(c, cats, registry["model_key"])

    n_ids = None
    if a.run_ids:
        # In ID mode ONLY what the file lists counts. bfcl's CLI help describes --run-ids as
        # additive to --test-category, an earlier measurement as exclusive. Rather than rely on
        # either reading, --test-category is reduced to the file's keys here — then the run is
        # correct under both interpretations.
        if not a.run_ids.is_file():
            c.fail("run-ids", f"{a.run_ids} not found")
            cats = []
        else:
            wanted = json.loads(a.run_ids.read_text())
            n_ids = sum(len(v) for v in wanted.values())
            cats = [x for x in wanted if x in cats] or list(wanted)
            c.ok("run-ids", f"{n_ids} ids / {len(cats)} categories from {a.run_ids.name}")

    if n_ids is not None:
        inv = {k: len(v) for k, v in wanted.items() if k in cats}
    else:
        inv = inventory(cats) if cats else {}
    total = sum(inv.values())
    if cats:
        c.ok("categories", f"{len(cats)} categories / {total} items")

    if max_model_len is not None:
        check_context(c, served_mpe, max_model_len, cats)
    check_memory_vector_model(c, cats)

    # Fast/slow partition on the FINAL category list (i.e. after any run-ids narrowing).
    # SLOW is defined as "everything not explicitly fast" on purpose: anything unknown or
    # future (web_search, format_sensitivity, new categories after a bfcl upgrade) falls into
    # the conservative bucket automatically — mistakes are built to fail toward safety.
    from bfcl_eval.constants.category_mapping import LIVE_CATEGORY, NON_LIVE_CATEGORY
    fast_set = set(NON_LIVE_CATEGORY) | set(LIVE_CATEGORY)
    cats_fast = [x for x in cats if x in fast_set]
    cats_slow = [x for x in cats if x not in fast_set]
    uniform = a.num_threads is not None
    nt_fast = a.num_threads if uniform else a.num_threads_fast
    nt_slow = a.num_threads if uniform else a.num_threads_slow
    max_seqs = max(nt_fast, nt_slow)
    if cats:
        c.ok("threads", f"fast={nt_fast} ({len(cats_fast)} cats) / slow={nt_slow} "
                        f"({len(cats_slow)} cats)" + ("  — uniform override" if uniform else ""))

    c.render()

    if c.failed:
        print("\n== PREFLIGHT FAILED — nothing served, nothing written.")
        return 1

    manifest = {
        "label": a.label,
        "model": a.model,
        "model_key": registry["model_key"],
        # BFCL sends this name in the `model` field to /v1/completions -> vLLM MUST carry it as
        # --served-model-name, otherwise every single request 404s.
        "model_alias": alias,
        # The registry entry the run uses. injected=True -> run_bfcl.py re-applies it for
        # generate AND evaluate; injected=False -> a stock bfcl key (harness validation).
        "registry": registry,
        "hf_home": os.getenv("HF_HOME"),
        "model_dir_resolved": str(model_dir.resolve()) if model_dir else None,
        "model_fingerprint": fingerprint(model_dir) if model_dir else None,
        "generation_config": {k: gen_cfg.get(k) for k in EXPECTED_GENERATION_CONFIG} if gen_cfg else None,
        "sampling": {"temperature": a.temperature, "note": "top_p/top_k come from generation_config.json"},
        "serving": {"max_model_len": max_model_len, "max_model_len_derived": a.max_model_len is None,
                    # num_threads = legacy key for old readers; equals the serve-side cap
                    "num_threads": max_seqs, "num_threads_fast": nt_fast,
                    "num_threads_slow": nt_slow, "max_num_seqs": max_seqs, "uniform": uniform},
        "categories": cats,
        "categories_skipped": ([x for x in ["web_search_base", "web_search_no_snippet"]]
                               if not a.enable_web_search else []),
        "inventory": inv,
        "items_total": total,
        "run_ids_file": str(a.run_ids) if a.run_ids else None,
        "bfcl_version": __import__("bfcl_eval").__version__ if hasattr(__import__("bfcl_eval"), "__version__")
                        else "2026.3.23",
        "repo_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    a.root.mkdir(parents=True, exist_ok=True)
    (a.root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (a.root / "categories.txt").write_text(",".join(cats) + "\n")
    (a.root / "categories_fast.txt").write_text(",".join(cats_fast) + "\n")
    (a.root / "categories_slow.txt").write_text(",".join(cats_slow) + "\n")
    print(f"\n== PREFLIGHT OK -> {a.root}/run_manifest.json  ({total} items)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
