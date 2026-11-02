"""GRPO run metrics (training_pipeline/grpo_metrics.py): divergence alarm, per-step readers, template
classes, selection with exclusion + manual override, error flags, paired bootstrap — synthetic cases
everywhere, plus a replay of the finished r2 run when its data is present."""
import json
import sys
from pathlib import Path

import pytest

import training_pipeline.grpo_metrics as gm

REPO = Path(__file__).resolve().parents[1]
R2 = REPO / "data/final/grpo/runs/grpo_db_bahn_r2"
EVAL = REPO / "data/generated/eval"


# --- divergence --------------------------------------------------------------------------------------
def _steps(ent, clip=None):
    return {"actor/entropy": dict(enumerate(ent, start=1)),
            "response_length/clip_ratio": dict(enumerate(clip or [0.0] * len(ent), start=1))}


def test_divergence_needs_a_complete_epoch_0():
    d = gm.divergence(_steps([0.4, 0.4]), spe=4)
    assert d["status"] == "no_baseline" and not d["fired"] and d["baseline"] is None


def test_divergence_fires_on_two_consecutive_steps_after_epoch_0():
    ent = [0.4] * 4 + [0.5, 0.9, 0.5, 0.85, 0.9, 0.3]      # baseline 0.4 -> threshold 0.8
    d = gm.divergence(_steps(ent), spe=4)
    assert (d["baseline"], d["threshold"]) == (0.4, 0.8)
    assert d["fired"] and d["rule"] == "entropy" and (d["onset"], d["fired_at"]) == (8, 9)


def test_divergence_ignores_epoch_0_spikes_and_single_steps():
    ent = [0.4, 0.9, 0.9, 0.4] + [0.9, 0.5, 0.9, 0.5]
    d = gm.divergence(_steps(ent), spe=4)
    assert d["status"] == "ok" and not d["fired"]


def test_clip_rule_from_step_1_and_earliest_rule_wins():
    d = gm.divergence(_steps([0.4] * 8, [0.0, 0.06, 0.07] + [0.0] * 5), spe=4)
    assert d["fired"] and d["rule"] == "clip_ratio" and (d["onset"], d["fired_at"]) == (2, 3)
    d = gm.divergence(_steps([0.4] * 4 + [0.9, 0.9, 0.4, 0.4], [0.0] * 5 + [0.1, 0.1, 0.0]), spe=4)
    assert d["rule"] == "entropy" and d["clip_rule"]["fired_at"] == 7


def test_low_entropy_is_reported_not_fired():
    d = gm.divergence(_steps([0.4] * 4 + [0.15, 0.1]), spe=4)
    assert not d["fired"] and d["low_entropy_steps"] == [5, 6]


# --- readers -----------------------------------------------------------------------------------------
def test_mlflow_reader_latest_timestamp_and_skips_deleted(tmp_path):
    exp = tmp_path / "123"
    (exp).mkdir()
    (exp / "meta.yaml").write_text("experiment_id: '123'\nname: db_bahn_grpo\n")
    for rid, stage, lines in (("a", "active", ["10 0.5 1", "20 0.6 2"]), ("b", "active", ["30 0.7 2", "40 0.8 3"]),
                              ("c", "deleted", ["50 9.9 1"])):
        (exp / rid / "tags").mkdir(parents=True)
        (exp / rid / "tags" / "mlflow.runName").write_text("myrun")
        (exp / rid / "meta.yaml").write_text(f"lifecycle_stage: {stage}\n")
        (exp / rid / "metrics" / "actor").mkdir(parents=True)
        (exp / rid / "metrics" / "actor" / "entropy").write_text("\n".join(lines) + "\n")
    got = gm.mlflow_step_metrics("myrun", "db_bahn_grpo", keys=("actor/entropy",), mlruns=tmp_path)
    assert got["actor/entropy"] == {1: 0.5, 2: 0.7, 3: 0.8}
    assert gm.mlflow_step_metrics("other", "db_bahn_grpo", keys=("actor/entropy",), mlruns=tmp_path)["actor/entropy"] == {}


def test_log_reader_handles_numpy_wrappers_and_ansi(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\x1b[36m(TaskRunner)\x1b[0m step:1 - global_seqlen/min:3 - actor/entropy:np.float64(0.41) - "
                   "actor/entropy_loss:9.0 - response_length/clip_ratio:0.0\n"
                   "noise line actor/entropy:5\n"
                   "step:2 - actor/entropy:0.5 - actor/lr:5e-06\n"
                   "step:2 - actor/entropy:0.55 - actor/lr:5e-06\n")
    got = gm.log_step_metrics(log, keys=("actor/entropy", "actor/lr", "response_length/clip_ratio"))
    assert got["actor/entropy"] == {1: 0.41, 2: 0.55} and got["actor/lr"] == {2: 5e-06}
    assert got["response_length/clip_ratio"] == {1: 0.0}


# --- classes, flags, bootstrap -------------------------------------------------------------------------
MAN = {"templates": {"t_e": {"signal": {"echt": 2, "keiner": 18}}, "t_w": {"signal": {"wortlaut": 3, "echt": 1}},
                     "t_g": {"signal": {"wortlaut": 1, "echt": 1}}, "t_k": {"signal": {"keiner": 20}}}}


@pytest.mark.parametrize("t,cls", [("t_e", "echt"), ("t_w", "wortlaut"), ("t_g", "gemischt"), ("t_k", "keiner"),
                                   ("t_x", "nicht_im_pool")])
def test_template_class(t, cls):
    assert gm.template_class(MAN, t) == cls


def test_error_flags(tmp_path):
    (tmp_path / "1.jsonl").write_text(json.dumps({"score": 0, "adapter_error": 1.0, "verifier_error": 0.0}) + "\n"
                                      + json.dumps({"score": 1, "adapter_error": 0.0, "verifier_error": 1.0}) + "\n")
    assert gm.error_flags(tmp_path) == {"rows": 2, "adapter_error": 1.0, "verifier_error": 1.0, "files": 1}
    old = tmp_path / "old"
    old.mkdir()
    (old / "1.jsonl").write_text(json.dumps({"score": 1}) + "\n")
    assert gm.error_flags(old) is None


def test_paired_bootstrap_is_pinned():
    a = gm.paired_bootstrap([1, -1, 0, 2, -3, 1, 0, 0, 1, 2], seed=0)
    assert a == gm.paired_bootstrap([1, -1, 0, 2, -3, 1, 0, 0, 1, 2], seed=0) and a[0] == 3 and a[1] <= 3 <= a[2]
    assert gm.paired_bootstrap([]) == (0, 0, 0)


# --- selection ---------------------------------------------------------------------------------------
def _vals(*rows):
    return [{"step": s, "tasks": 10, "pass_k": p, "mean": m, "rollouts": 80} for s, p, m in rows]


@pytest.fixture
def ckpts(tmp_path):
    d = tmp_path / "ckpt"
    for s in (8, 16, 24, 32, 40):
        (d / f"global_step_{s}").mkdir(parents=True)
    return d


VALS = _vals((0, 0.90, 0.95), (8, 0.91, 0.95), (16, 0.92, 0.96), (24, 0.93, 0.97), (32, 0.95, 0.98), (40, 0.97, 0.99))


def test_select_excludes_from_the_onset(ckpts, tmp_path):
    dec = gm.select(VALS, ckpts, tmp_path / "ad", "r", dry=True)
    assert dec["selected"]["step"] == 40
    dec = gm.select(VALS, ckpts, tmp_path / "ad", "r", dry=True, exclude_from=33)
    assert dec["selected"]["step"] == 32 and dec["excluded_by_divergence"] == [40]
    dec = gm.select(VALS, ckpts, tmp_path / "ad", "r", dry=True, exclude_from=5)
    assert dec["selected"] is None and "before the divergence onset" in dec["reason"]
    assert not (tmp_path / "ad").exists()


def test_pick_step_override_and_its_protection(ckpts, tmp_path):
    run_dir, ad = tmp_path / "run", tmp_path / "ad"
    dec = gm.pick_step(VALS, ckpts, ad, run_dir, "r", 24, "K=8 tail cleaner", "Janneck", onset=None, force=False, dry=False)
    assert (ad / "selected").resolve() == (ckpts / "global_step_24").resolve()
    sel = json.loads((run_dir / "selection.json").read_text())
    assert sel["selected"]["step"] == 24 and sel["reason"] == "K=8 tail cleaner"
    assert sel["override"]["auto_selected_step"] == 40 and sel["override"]["by"] == "Janneck"
    assert gm.existing_override(run_dir) == sel["override"]
    gm.pick_step(VALS, ckpts, ad, run_dir, "r", 16, "second thought", "Janneck", onset=None, force=False, dry=False)
    assert json.loads((run_dir / "selection.json").read_text())["override"]["previous"]["auto_selected_step"] == 40
    with pytest.raises(SystemExit):        # at/after the divergence onset needs --force
        gm.pick_step(VALS, ckpts, ad, run_dir, "r", 40, "x", "J", onset=33, force=False, dry=False)
    with pytest.raises(SystemExit):        # no checkpoint
        gm.pick_step(VALS, ckpts, ad, run_dir, "r", 12, "x", "J", onset=None, force=False, dry=False)


def test_cli_select_refuses_to_overwrite_an_override(ckpts, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    (runs / "rt").mkdir(parents=True)
    sel = {"selected": {"step": 24}, "override": {"date": "2026-10-28", "by": "Janneck", "auto_selected_step": 32}}
    (runs / "rt" / "selection.json").write_text(json.dumps(sel))
    r = {"run": "rt", "profile": "default", "paths": {"runs_root": str(runs), "ckpt_root": str(ckpts.parent),
                                                       "pool_dir": str(tmp_path), "init_model": "x"},
         "pool": {"val_file": "val.parquet", "epoch_size": 128}, "train": {"n": 8, "prompts_per_step": 16, "lr": 2e-5},
         "val": {"n": 8}, "stop": {}, "mlflow": {"project": "none"}}
    monkeypatch.setattr(gm, "load_run", lambda run: r)
    monkeypatch.setattr(sys, "argv", ["grpo_metrics.py", "rt", "--select", "--no-classes"])
    with pytest.raises(SystemExit) as e:
        gm.main()
    assert "manual override" in str(e.value)
    assert json.loads((runs / "rt" / "selection.json").read_text()) == sel


# --- replay of the finished r2 run ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def r2():
    if not (R2 / "resolved_config.json").exists():
        pytest.skip("r2 run data not present")
    return json.loads((R2 / "resolved_config.json").read_text())["resolved"]


def test_r2_divergence_and_lr_replay(r2):
    steps = gm.mlflow_step_metrics("grpo_db_bahn_r2", "db_bahn_grpo")
    if not steps["actor/entropy"]:
        pytest.skip("r2 MLflow metrics not present")
    d = gm.divergence(steps, spe=8)
    assert (d["baseline"], d["threshold"]) == (0.3889, 0.7779)
    assert d["rule"] == "entropy" and (d["onset"], d["fired_at"]) == (33, 34)
    assert d["clip_rule"]["fired_at"] == 38
    assert gm.lr_check(steps, r2["train"])["max_rel_dev"] < 1e-6
    vals = gm.validation_metrics(R2 / "val_dump", r2["val"]["n"])
    ck = REPO / "data/final/grpo/verl_ckpt/grpo_db_bahn_r2"
    assert gm.select(vals, ck, Path("/nonexistent"), "r2", dry=True, exclude_from=d["onset"])["selected"]["step"] == 32


def test_r2_train_classes_replay(r2):
    man = json.loads((REPO / "data/final/grpo/db_bahn_pool/corridor_ep3_k8_20260902/pool_manifest.json").read_text())
    tc = gm.train_classes(R2 / "rollout_dump", gm.task_signals(r2, man), spe=8)
    assert [c["wortlaut"]["fail"] for c in tc] == [94, 91, 79, 48, 98]
    assert [c["echt"]["fail"] for c in tc] == [43, 50, 49, 43, 100]


def test_k8_class_compare_reproduces_r2_finding():
    base, cand = (EVAL / f"db_traces_heldout_eval_after_{x}_k8.jsonl" for x in ("qwen3-4b_ep3", "grpo_db_bahn_r2_step24"))
    if not (base.exists() and cand.exists()):
        pytest.skip("K=8 heldout traces not present")
    from evaluation.k8_class_compare import compare
    man = json.loads((REPO / "data/final/grpo/db_bahn_pool/corridor_ep3_k8_20260902/pool_manifest.json").read_text())
    c = compare(base, cand, man)["classes"]
    assert (c["wortlaut"]["fail_base"], c["wortlaut"]["fail_cand"], c["wortlaut"]["diff"], c["wortlaut"]["ci95"]) == (46, 15, -31, [-49, -15])
    assert (c["echt"]["fail_base"], c["echt"]["fail_cand"], c["echt"]["diff"], c["echt"]["ci95"]) == (29, 47, 18, [4, 32])
