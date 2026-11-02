"""GRPO configuration renderer (training_pipeline/grpo_config.py): the r3 recipe renders the flags it
should, every knob is checked, and older runs re-render exactly the flags they were started with."""
import copy
import json
from pathlib import Path

import pytest

from training_pipeline.grpo_config import (CONF_LOSS_LIB, CONF_LOSS_MODE, flag_checks, flag_map, git_dirty,
                                           load_profile, lr_factor, render_flags, resolved_sha, write_resolved)

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "data/final/grpo/runs"
A = "actor_rollout_ref.actor."


def _bad(r, flags=None):
    return [k for k, v in flag_checks(r, render_flags(r) if flags is None else flags).items() if not v]


@pytest.mark.parametrize("profile,run", [("default", "grpo_test_r3"), ("smoke", None), ("smoke_static", None)])
def test_profiles_pass_all_checks(profile, run):
    r = load_profile(profile, run=run)
    flags = render_flags(r)
    assert _bad(r, flags) == []
    assert len(flag_map(flags)) == len(flags)


def test_r3_recipe_flags():
    fl = flag_map(render_flags(load_profile("default", run="grpo_test_r3")))
    assert fl[A + "use_kl_loss"] == "True" and fl[A + "kl_loss_coef"] == "0.01"
    assert fl["actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"] == "1"
    assert fl[A + "optim.lr_scheduler_type"] == "cosine" and fl[A + "optim.min_lr_ratio"] == "0.1"
    assert fl["trainer.total_training_steps"] == "40"
    assert fl["actor_rollout_ref.rollout.calculate_log_probs"] == "True"
    assert fl["actor_rollout_ref.rollout.logprobs_mode"] == "raw_logprobs"
    assert A + "policy_loss.loss_mode" not in fl and "actor_rollout_ref.model.external_lib" not in fl


def test_confidence_weighting_switch():
    r = load_profile("default", run="grpo_test_r3")
    r["train"]["conf_weight"] = {"enabled": True, "tau": 0.7, "factor": 0.2}
    fl = flag_map(render_flags(r))
    assert fl[A + "policy_loss.loss_mode"] == CONF_LOSS_MODE
    assert fl["actor_rollout_ref.model.external_lib"] == CONF_LOSS_LIB
    assert not any("tau" in k or "factor" in k for k in fl)          # read from resolved_config, not Hydra
    assert _bad(r) == []
    for bad in ({"tau": None}, {"tau": 1.0}, {"tau": 0.7, "factor": 0}, {"tau": 0.7, "factor": 1.5}):
        rb = copy.deepcopy(r)
        rb["train"]["conf_weight"].update(bad)
        assert _bad(rb) == ["conf weighting wired"], bad


@pytest.mark.parametrize("mutate,violation", [
    (lambda f: [x for x in f if "ref.log_prob_micro_batch_size_per_gpu" not in x], "kl -> ref micro-batch"),
    (lambda f: [x.replace("raw_logprobs", "processed_logprobs") for x in f], "log-probs raw"),
    (lambda f: f + ["actor_rollout_ref.actor.policy_loss.loss_mode=conf_weighted"], "conf weighting wired"),
    (lambda f: f + ["trainer.nnodes=1"], "no None / no duplicate keys"),
    (lambda f: f + ["foo.bar=None"], "no None / no duplicate keys"),
])
def test_flag_checks_catch_violations(mutate, violation):
    r = load_profile("default", run="grpo_test_r3")
    assert violation in _bad(r, mutate(render_flags(r)))


@pytest.mark.parametrize("knob,value,violation", [
    ("total_training_steps", 36, "horizon on a checkpoint"),
    ("total_training_steps", 200, "epochs cover horizon"),
    ("test_freq", 4, "save_freq == test_freq"),
])
def test_flag_checks_catch_bad_knobs(knob, value, violation):
    r = load_profile("default", run="grpo_test_r3")
    r["train"][knob] = value
    assert violation in _bad(r)


def test_cosine_without_horizon_is_caught():
    r = load_profile("default", run="grpo_test_r3")
    r["train"]["total_training_steps"] = None
    assert "cosine -> horizon + min ratio" in _bad(r)


def test_lr_factor_matches_verl_closed_form():
    got = [round(lr_factor(k, 4, 40, 0.1), 4) for k in (1, 2, 3, 4, 5, 24, 32, 40)]
    assert got == [0.0, 0.25, 0.5, 0.75, 1.0, 0.5108, 0.2318, 0.1017]
    assert lr_factor(41, 4, 40, 0.1) == pytest.approx(0.1)                    # logged lr after the last step
    assert [lr_factor(k, 4, None, 0, "constant") for k in (1, 5, 40)] == [0.0, 1.0, 1.0]


@pytest.mark.parametrize("run", ["grpo_db_bahn_r2"])
def test_old_run_renders_its_recorded_flags(run):
    f = RUNS / run / "resolved_config.json"
    if not f.exists():
        pytest.skip(f"{f} not present (run data)")
    doc = json.loads(f.read_text())
    assert render_flags(doc["resolved"]) == doc["flags"]
    assert _bad(doc["resolved"], doc["flags"]) == []


def test_r1_differs_only_by_the_later_warmup_flag():
    f = RUNS / "grpo_db_bahn_r1/resolved_config.json"
    if not f.exists():
        pytest.skip(f"{f} not present (run data)")
    doc = json.loads(f.read_text())
    assert set(render_flags(doc["resolved"])) ^ set(doc["flags"]) == {"actor_rollout_ref.actor.optim.lr_warmup_steps=-1"}


def test_write_resolved_records_every_start(tmp_path):
    r = load_profile("default", run="grpo_test_r3")
    path = tmp_path / "grpo_test_r3" / "resolved_config.json"
    assert write_resolved(r, path) == "written"
    assert write_resolved(r, path) == "unchanged"
    starts = [json.loads(x) for x in (path.parent / "starts.jsonl").read_text().splitlines()]
    assert [s["resolved_config"] for s in starts] == ["written", "unchanged"]
    assert "git_dirty" in json.loads(path.read_text())
    r2 = copy.deepcopy(r)
    r2["train"]["kl_coef"] = 0.02
    with pytest.raises(SystemExit):
        write_resolved(r2, path)


@pytest.mark.parametrize("edit", [
    lambda r: r["train"]["conf_weight"].update(tau=0.7),       # not a verl flag (read from resolved_config)
    lambda r: r["stop"].update(entropy_factor=3.0),            # not a verl flag (grpo_metrics)
    lambda r: r.update(image="agentic-grpo:other"),            # runner env, not a flag
    lambda r: r["train"].update(kl_coef=0.02),                 # a flag
])
def test_expect_sha_refuses_any_change_since_the_flag_render(tmp_path, edit):
    r = load_profile("default", run="grpo_test_r3")
    sha = resolved_sha(r)
    assert sha == resolved_sha(load_profile("default", run="grpo_test_r3"))   # deterministic
    edited = copy.deepcopy(r)
    edit(edited)
    path = tmp_path / "grpo_test_r3" / "resolved_config.json"
    with pytest.raises(SystemExit, match="changed since the flags were rendered"):
        write_resolved(edited, path, expect_sha=sha)
    assert not path.parent.exists()                            # refused before anything was written
    assert write_resolved(r, path, expect_sha=sha) == "written"


def test_git_dirty_shape():
    d = git_dirty()
    if d is None:
        pytest.skip("not a git checkout")
    assert isinstance(d["files"], list) and (d["sha256_16"] is None) == (not d["files"])
