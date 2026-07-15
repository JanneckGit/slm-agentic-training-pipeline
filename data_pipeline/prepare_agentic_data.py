"""
data_pipeline/prepare_agentic_data.py
=====================================
Downloads the public tool-calling / planning datasets that form the agentic-orchestrator
SFT data basis, and writes them into data/raw/, mirroring data_pipeline/prepare_data.py
(the Text-to-SQL seed fetcher).

Stage-1 (mixed SFT) data basis legs:
  1. ToolACE   (Team-ACE/ToolACE, Apache-2.0)            -> tool-call basics (call -> read -> next)
  2. AReaL     (inclusionAI/AReaL-tau2-data, Apache-2.0) -> tau2-bench flows: per-turn SFT
     (replan/self-correction, airline/retail/telecom) + RL tasks incl. DB snapshots
  3. synthetic DB flows                                  -> generated locally (not fetched here)

(TaskBench was acquired once but deliberately dropped from the mix — task-graph JSON, no
executable tool calls; see docs/agentic-datasets-explained.md.)

This script only ACQUIRES the raw data (a faithful copy; ToolACE gets source/split tags,
AReaL is a byte-identical repo snapshot). Converting the raw records into the unified
chat/training format happens in a later mix step. Validate the AReaL pull with
data_pipeline/validate_areal.py.

All datasets are public (not gated), so no HF token is required.

Usage:
    PYTHONPATH=. python data_pipeline/prepare_agentic_data.py --config config/pipeline_config.yaml --dataset all
    PYTHONPATH=. python data_pipeline/prepare_agentic_data.py --dataset toolace --n-samples 200   # quick look
    PYTHONPATH=. python data_pipeline/prepare_agentic_data.py --dataset areal                     # ~970 MB snapshot
"""

import argparse
import json
import logging
import random
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from data_pipeline.common import load_config, write_jsonl

# NOTE: `datasets` is imported lazily inside fetch_toolace — the tau2 venv
# (.venv-tau2) only ships huggingface_hub, and the AReaL snapshot fetch must run there.

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
AREAL_DEFAULT = {
    "dataset": "inclusionAI/AReaL-tau2-data",
    "license": "Apache-2.0",
    # from the HF card (2026-07-08); validate_areal.py hard-fails on mismatch
    "expected": {
        "sft_rows": 33531,
        "rl_rows": 1982,
        "sft_domains": {"airline": 12842, "retail": 11395, "telecom": 9294},
        "rl_domains": {"airline": 1148, "retail": 563, "telecom": 271},
    },
}


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
    from datasets import load_dataset

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
# AReaL – tau2-bench trajectories (per-turn SFT + RL tasks + DB snapshots).
#   The repo is raw root-level JSONL + a snapshot dir (json/toml) -> full repo
#   snapshot instead of load_dataset (no Arrow round-trip over 874 MB of
#   heterogeneous nested JSON; keeps the copy byte-identical; resumable).
# ---------------------------------------------------------------------------
def fetch_areal(cfg: dict, raw_dir: Path, n_samples: int | None) -> dict:
    dataset_id = cfg["dataset"]
    out_dir = raw_dir / "areal"

    if n_samples:
        logger.info("[AReaL] --n-samples is ignored (file-level repo snapshot, no row fetch)")

    revision = HfApi().dataset_info(dataset_id).sha  # pin for reproducibility
    logger.info(f"[AReaL] snapshot_download({dataset_id}, revision={revision[:12]}…) -> {out_dir}")
    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
        local_dir=out_dir,
    )

    # Dumb post-download gate only (existence + plausible sizes); the real checks
    # live in data_pipeline/validate_areal.py.
    sft_path = out_dir / "tau2_sft_train.jsonl"
    rl_path = out_dir / "tau2_rl_train.jsonl"
    db_dir = out_dir / "tau2_rl_database"
    for p in (sft_path, rl_path):
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f"[AReaL] expected file missing/empty after snapshot: {p}")
    db_files = [p for p in db_dir.iterdir() if p.is_file()] if db_dir.is_dir() else []
    if not db_files:
        raise RuntimeError(f"[AReaL] expected non-empty snapshot dir: {db_dir}")

    sft_mb = sft_path.stat().st_size / 1e6
    rl_mb = rl_path.stat().st_size / 1e6
    db_mb = sum(p.stat().st_size for p in db_files) / 1e6
    logger.info(f"[AReaL] sft={sft_mb:.1f} MB, rl={rl_mb:.1f} MB, db={db_mb:.1f} MB ({len(db_files)} files)")

    with open(rl_path) as f:  # small file – first record as console preview
        sample = json.loads(f.readline())

    return {
        "name": "AReaL-tau2",
        "dataset_id": dataset_id,
        "license": cfg.get("license", "Apache-2.0"),
        "revision": revision,
        "fetch_method": "snapshot_download",
        "rows_written": None,  # snapshot copy – row counts are stamped by validate_areal.py
        "expected_rows": {k: cfg.get("expected", {}).get(k) for k in ("sft_rows", "rl_rows")},
        "columns": list(sample.keys()),
        "files": [str(sft_path), str(rl_path), str(db_dir)],
        "size_mb": {"sft": round(sft_mb, 2), "rl": round(rl_mb, 2), "db": round(db_mb, 2)},
        "teaches": "tau2-bench customer-service flows: replan/self-correction (per-turn SFT with "
                   "CoT + tool_calls) + RL task configs with DB snapshots",
        "note": "PER-TURN format (unlike db_bahn full-episode) — format unification happens at mix "
                "time. Validate with data_pipeline/validate_areal.py.",
        "sample": sample,
    }


def _print_summary(summaries: list[dict]) -> None:
    logger.info("\n" + "=" * 70)
    logger.info("AGENTIC SFT DATA BASIS — raw pull summary")
    logger.info("=" * 70)
    for s in summaries:
        logger.info(f"\n### {s['name']}  ({s['dataset_id']}, {s['license']})")
        rows = s["rows_written"] if s.get("rows_written") is not None else "(snapshot — counts via validate_areal.py)"
        logger.info(f"  rows written : {rows}")
        logger.info(f"  columns      : {s['columns']}")
        logger.info(f"  teaches      : {s['teaches']}")
        if s.get("sample") is not None:
            logger.info("  sample record:\n  " + _preview(s["sample"]).replace("\n", "\n  "))


def main():
    parser = argparse.ArgumentParser(description="Fetch ToolACE + AReaL into data/raw (agentic SFT basis)")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--dataset", choices=["toolace", "areal", "all"], default="all")
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
    areal_cfg = {**AREAL_DEFAULT, **agentic_cfg.get("areal", {})}

    summaries = []
    if args.dataset in ("toolace", "all"):
        summaries.append(fetch_toolace(toolace_cfg, raw_dir, args.n_samples))
    if args.dataset in ("areal", "all"):
        summaries.append(fetch_areal(areal_cfg, raw_dir, args.n_samples))

    # Manifest (small – counts/paths/columns, no full samples) for quick inspection.
    # MERGE into the existing manifest — a partial run (e.g. --dataset toolace) must
    # not clobber the entries of datasets it did not fetch.
    manifest_path = raw_dir / "agentic_manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    for s in summaries:
        entry = {k: v for k, v in s.items() if k != "sample"}
        # keep a validation stamp (validate_areal.py) across a no-op re-fetch — it only
        # goes stale when the pinned upstream revision actually changed
        old = manifest.get(s["name"], {})
        same_revision = entry.get("revision") is not None and old.get("revision") == entry.get("revision")
        if "validation" in old and same_revision:
            entry["validation"] = old["validation"]
        manifest[s["name"]] = entry
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    _print_summary(summaries)
    logger.info(f"\n✅ Raw pull complete. Manifest: {manifest_path}")
    logger.info("Next (later step): convert these raw records into the unified chat/training format.")


if __name__ == "__main__":
    main()
