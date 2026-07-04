"""
data_pipeline/prepare_agentic_data.py
=====================================
Downloads the two public tool-calling / planning datasets that form the first two
legs of the agentic-orchestrator SFT data basis, and writes them as JSONL into
data/raw/, mirroring data_pipeline/prepare_data.py (the Text-to-SQL seed fetcher).

Stage-1 (mixed SFT) data basis has three legs:
  1. ToolACE   (Team-ACE/ToolACE, Apache-2.0)  -> tool-call basics (call -> read -> next)
  2. TaskBench (microsoft/Taskbench, MIT)        -> planning / decomposition (tool-graph)
  3. synthetic DB flows                          -> LATER (domain adaptation; not fetched here)

This script only ACQUIRES the raw data (a faithful copy + source/split tags). Converting
the raw records into the unified chat/training format happens in a later mix step.

Both datasets are public (not gated), so no HF token is required.

Usage:
    python data_pipeline/prepare_agentic_data.py --config config/pipeline_config.yaml --dataset all
    python data_pipeline/prepare_agentic_data.py --dataset toolace --n-samples 200   # quick look
"""

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

import yaml
from datasets import load_dataset
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults – used when config has no `data.agentic` block (script stays runnable
# without editing the config; the dataset IDs are fixed public repos).
# ---------------------------------------------------------------------------
TOOLACE_DEFAULT = {
    "dataset": "Team-ACE/ToolACE",
    "split": "train",
    "license": "Apache-2.0",
}
TASKBENCH_DEFAULT = {
    "dataset": "microsoft/Taskbench",
    "split": "test",  # TaskBench ships its data under a `test` split (no `train`)
    "configs": ["huggingface", "multimedia", "dailylifeapis"],
    "license": "MIT",
    # sidecar files (tool inventory + dependency graph) live at data_<config>/<name>
    "sidecars": ["tool_desc.json", "graph_desc.json"],
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def write_jsonl(rows: list[dict], path: Path) -> Path:
    """One JSON object per line – identical idiom to prepare_data.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _maybe_subsample(rows: list[dict], n_samples: int | None) -> list[dict]:
    """Optional representative subset for a quick 'show' run (seeded)."""
    if n_samples and n_samples < len(rows):
        return random.sample(rows, n_samples)
    return rows


def _preview(row: dict, max_chars: int = 600) -> str:
    """Truncated pretty-print of one record, for the console summary."""
    s = json.dumps(row, ensure_ascii=False, indent=2)
    return s if len(s) <= max_chars else s[:max_chars] + "\n  … (truncated)"


# ---------------------------------------------------------------------------
# ToolACE – ShareGPT-style multi-tool conversations.
#   columns: system (tool defs embedded as JSON in the prompt), conversations [{from,value}]
# ---------------------------------------------------------------------------
def fetch_toolace(cfg: dict, raw_dir: Path, n_samples: int | None) -> dict:
    dataset_id = cfg["dataset"]
    split = cfg["split"]
    logger.info(f"[ToolACE] load_dataset({dataset_id}, split={split}) …")
    ds = load_dataset(dataset_id, split=split)
    logger.info(f"[ToolACE] raw rows: {len(ds)}")

    rows = [{**ex, "source": dataset_id, "split": split} for ex in ds]
    rows = _maybe_subsample(rows, n_samples)

    out = write_jsonl(rows, raw_dir / "toolace" / "toolace.jsonl")
    size_mb = out.stat().st_size / 1e6
    logger.info(f"[ToolACE] wrote {len(rows)} rows -> {out} ({size_mb:.1f} MB)")

    return {
        "name": "ToolACE",
        "dataset_id": dataset_id,
        "license": cfg.get("license", "Apache-2.0"),
        "split": split,
        "rows_written": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "files": [str(out)],
        "size_mb": round(size_mb, 2),
        "teaches": "tool-call basics (call a tool, read the response, call the next)",
        "sample": rows[0] if rows else None,
    }


# ---------------------------------------------------------------------------
# TaskBench – 3 domain configs; each row is a request + a tool-invocation graph
#   (graph columns are JSON-encoded STRINGS). The tool inventory (tool_desc.json)
#   and dependency graph (graph_desc.json) are NOT in the parquet -> fetched
#   separately via hf_hub_download.
# ---------------------------------------------------------------------------
def fetch_taskbench(cfg: dict, raw_dir: Path, n_samples: int | None) -> dict:
    dataset_id = cfg["dataset"]
    split = cfg["split"]
    configs = cfg["configs"]
    sidecars = cfg.get("sidecars", ["tool_desc.json", "graph_desc.json"])
    out_root = raw_dir / "taskbench"

    per_domain = {}
    files = []
    total = 0
    first_sample = None
    columns = []
    for domain in configs:
        logger.info(f"[TaskBench] load_dataset({dataset_id}, {domain}, split={split}) …")
        ds = load_dataset(dataset_id, domain, split=split)
        rows = [{**ex, "source": dataset_id, "domain": domain, "split": split} for ex in ds]
        rows = _maybe_subsample(rows, n_samples)
        if rows and first_sample is None:
            first_sample = rows[0]
            columns = list(rows[0].keys())

        domain_dir = out_root / domain
        data_out = write_jsonl(rows, domain_dir / "data.jsonl")
        files.append(str(data_out))
        total += len(rows)

        # sidecars: data_<domain>/<name> in the repo -> copy into the domain dir
        sidecar_ok = []
        for name in sidecars:
            try:
                cached = hf_hub_download(
                    repo_id=dataset_id,
                    filename=f"data_{domain}/{name}",
                    repo_type="dataset",
                )
                dst = domain_dir / name
                shutil.copyfile(cached, dst)
                files.append(str(dst))
                sidecar_ok.append(name)
            except Exception as e:  # noqa: BLE001 – best-effort, report and continue
                logger.warning(f"[TaskBench] sidecar {domain}/{name} failed: {e}")

        per_domain[domain] = {"rows": len(rows), "sidecars": sidecar_ok}
        logger.info(f"[TaskBench] {domain}: {len(rows)} rows, sidecars={sidecar_ok} -> {domain_dir}")

    return {
        "name": "TaskBench",
        "dataset_id": dataset_id,
        "license": cfg.get("license", "MIT"),
        "split": split,
        "rows_written": total,
        "per_domain": per_domain,
        "columns": columns,
        "files": files,
        "teaches": "planning / decomposition (which steps, which order, which tool + params)",
        "note": "graph columns (tool_nodes/tool_links/…) are JSON-encoded strings; "
                "tool_desc.json + graph_desc.json carry the tool inventory + dependency graph",
        "sample": first_sample,
    }


def _print_summary(summaries: list[dict]) -> None:
    logger.info("\n" + "=" * 70)
    logger.info("AGENTIC SFT DATA BASIS — raw pull summary")
    logger.info("=" * 70)
    for s in summaries:
        logger.info(f"\n### {s['name']}  ({s['dataset_id']}, {s['license']})")
        logger.info(f"  rows written : {s['rows_written']}")
        if "per_domain" in s:
            for dom, info in s["per_domain"].items():
                logger.info(f"    - {dom:14s}: {info['rows']} rows, sidecars={info['sidecars']}")
        logger.info(f"  columns      : {s['columns']}")
        logger.info(f"  teaches      : {s['teaches']}")
        if s.get("sample") is not None:
            logger.info("  sample record:\n  " + _preview(s["sample"]).replace("\n", "\n  "))


def main():
    parser = argparse.ArgumentParser(description="Fetch ToolACE + TaskBench into data/raw (agentic SFT basis)")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--dataset", choices=["toolace", "taskbench", "all"], default="all")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Optional per-dataset cap for a quick 'show' subset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    config = load_config(args.config)
    data_cfg = config["data"]
    agentic_cfg = data_cfg.get("agentic", {})
    raw_dir = Path(data_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    toolace_cfg = {**TOOLACE_DEFAULT, **agentic_cfg.get("toolace", {})}
    taskbench_cfg = {**TASKBENCH_DEFAULT, **agentic_cfg.get("taskbench", {})}

    summaries = []
    if args.dataset in ("toolace", "all"):
        summaries.append(fetch_toolace(toolace_cfg, raw_dir, args.n_samples))
    if args.dataset in ("taskbench", "all"):
        summaries.append(fetch_taskbench(taskbench_cfg, raw_dir, args.n_samples))

    # Manifest (small – counts/paths/columns, no full samples) for quick inspection.
    manifest = {
        s["name"]: {k: v for k, v in s.items() if k != "sample"}
        for s in summaries
    }
    manifest_path = raw_dir / "agentic_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    _print_summary(summaries)
    logger.info(f"\n✅ Raw pull complete. Manifest: {manifest_path}")
    logger.info("Next (later step): convert these raw records into the unified chat/training format.")


if __name__ == "__main__":
    main()
