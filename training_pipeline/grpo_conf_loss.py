#!/usr/bin/env python3
"""Confidence-weighted GRPO policy loss — the `conf_weighted` loss mode for verl 0.8.0.

In a rollout with negative advantage, every response token gets the same push-down, also the many
tokens the policy produced with high confidence and that were not the mistake. This loss keeps the full
gradient for the uncertain tokens of such a rollout and scales the gradient of its confident tokens down:

    w = factor   where advantage < 0 and p_old(token) > tau      (p_old = exp(old_log_prob), T = 1.0)
    w = 1        everywhere else (all positive-advantage tokens, all uncertain tokens)

The loss is verl's vanilla PPO loss on advantages x w — for w > 0 that is exactly a per-token weight on
the clipped objective (incl. dual clip; clip metrics unchanged). The weights are NOT renormalised under
token-mean, so the update leans to the positive side: watch actor/entropy for collapse.

Wiring (no verl patch): config/grpo_db_bahn.yaml train.conf_weight.enabled renders
    actor_rollout_ref.actor.policy_loss.loss_mode=conf_weighted
    actor_rollout_ref.model.external_lib=training_pipeline.grpo_conf_loss
verl imports `external_lib` in every worker that builds a model config (HFModelConfig.__post_init__),
which registers the loss before the actor first looks it up. tau/factor cannot travel as Hydra keys
(PolicyLossConfig rejects unknown fields): they are read from $GRPO_RUN_DIR/resolved_config.json at import
time, so a bad value fails the worker start instead of a step two hours in.

    python3 training_pipeline/grpo_conf_loss.py --selftest     # torch only (training container, GPU-free)
The verl side (registry + equivalence to vanilla) is the preflight gate `loss` (verl_check).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

LOSS_MODE = "conf_weighted"
METRIC_KEYS = ("actor/conf_downweighted_frac", "actor/conf_neg_token_frac")

_PARAMS: dict | None = None


def _validate(cw: dict) -> dict:
    tau, factor = cw.get("tau"), cw.get("factor", 0.2)
    if not isinstance(tau, (int, float)) or not 0 < tau < 1:
        raise RuntimeError(f"[grpo_conf_loss] conf_weight.tau must be in (0, 1), got {tau!r}")
    if not isinstance(factor, (int, float)) or not 0 < factor <= 1:
        raise RuntimeError(f"[grpo_conf_loss] conf_weight.factor must be in (0, 1], got {factor!r}")
    return {"tau": float(tau), "factor": float(factor)}


def load_params(run_dir: str | None = None) -> dict:
    """tau/factor from <run_dir>/resolved_config.json (resolved.train.conf_weight)."""
    run_dir = run_dir or os.environ.get("GRPO_RUN_DIR")
    if not run_dir:
        raise RuntimeError("[grpo_conf_loss] GRPO_RUN_DIR is not set — cannot read conf_weight")
    f = Path(run_dir) / "resolved_config.json"
    if not f.exists():
        raise RuntimeError(f"[grpo_conf_loss] {f} missing — ops/grpo_run.sh writes it before verl starts")
    cw = json.loads(f.read_text())["resolved"]["train"].get("conf_weight") or {}
    if not cw.get("enabled"):
        raise RuntimeError(f"[grpo_conf_loss] loaded as loss for a run whose conf_weight is not enabled ({f})")
    return _validate(cw)


def override_params(tau: float, factor: float) -> None:
    """Tests and the preflight only."""
    global _PARAMS
    _PARAMS = _validate({"tau": tau, "factor": factor})


def params() -> dict:
    global _PARAMS
    if _PARAMS is None:
        _PARAMS = load_params()
    return _PARAMS


def confidence_weights(old_log_prob, advantages, response_mask, tau: float, factor: float):
    """-> (w, hit): w = factor on hit tokens (advantage < 0, p_old > tau, in the mask), else 1."""
    import torch
    with torch.no_grad():
        mask = response_mask.bool()
        hit = (advantages < 0) & (old_log_prob.exp() > tau) & mask
        w = torch.where(hit, torch.full_like(advantages, factor), torch.ones_like(advantages))
    return w, hit


def conf_metrics(hit, advantages, response_mask) -> dict:
    """Share of response tokens down-weighted, and share with negative advantage (the candidates)."""
    mask = response_mask.bool()
    n = max(int(mask.sum().item()), 1)
    return {METRIC_KEYS[0]: float(hit.sum().item()) / n,
            METRIC_KEYS[1]: float(((advantages < 0) & mask).sum().item()) / n}


# --- verl registration (only where verl is installed: the grpo image) -------------------------------
try:
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla, register_policy_loss
except ImportError:   # tests / training container: the pure-torch part above stays usable
    register_policy_loss = None

if register_policy_loss is not None:
    @register_policy_loss(LOSS_MODE)
    def compute_policy_loss_conf_weighted(old_log_prob, log_prob, advantages, response_mask,
                                          loss_agg_mode: str = "token-mean", config=None, rollout_is_weights=None):
        p = params()
        w, hit = confidence_weights(old_log_prob, advantages, response_mask, p["tau"], p["factor"])
        pg_loss, pg_metrics = compute_policy_loss_vanilla(
            old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages * w, response_mask=response_mask,
            loss_agg_mode=loss_agg_mode, config=config, rollout_is_weights=rollout_is_weights)
        pg_metrics.update(conf_metrics(hit, advantages, response_mask))
        return pg_loss, pg_metrics


def verl_check(tau: float = 0.7, factor: float = 0.2) -> list[str]:
    """Preflight gate body (grpo image, CPU): the loss is registered, factor 1 equals vanilla, and with
    factor < 1 it equals vanilla on hand-weighted advantages — i.e. only advantage<0 & p>tau tokens are
    scaled. Returns the problems ([] = ok)."""
    import torch
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY, compute_policy_loss_vanilla

    problems = []
    fn = POLICY_LOSS_REGISTRY.get(LOSS_MODE)
    if fn is None:
        return [f"{LOSS_MODE} not in POLICY_LOSS_REGISTRY"]
    cfg = OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.2, "clip_ratio_c": 3.0,
                            "global_batch_info": {}})
    g = torch.Generator().manual_seed(0)
    B, T = 6, 40
    old = -torch.rand(B, T, generator=g) * 1.5          # p_old in (0.22, 1]
    new = old + 0.3 * torch.randn(B, T, generator=g)    # ratios incl. clipped ones
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.0, -2.0]).unsqueeze(1).expand(B, T).clone()
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -7:] = False
    kw = dict(old_log_prob=old, log_prob=new, response_mask=mask, loss_agg_mode="token-mean", config=cfg)
    saved = _PARAMS
    try:
        override_params(tau, 1.0)
        l1, m1 = fn(advantages=adv, **kw)
        lv, mv = compute_policy_loss_vanilla(advantages=adv, **kw)
        if not torch.allclose(l1, lv) or any(abs(m1[k] - mv[k]) > 1e-7 for k in mv):
            problems.append("factor 1 differs from vanilla")
        override_params(tau, factor)
        lw, mw = fn(advantages=adv, **kw)
        w_ref = torch.where((adv < 0) & (old.exp() > tau) & mask, torch.tensor(factor), torch.tensor(1.0))
        lr_, mr = compute_policy_loss_vanilla(advantages=adv * w_ref, **kw)
        if not torch.allclose(lw, lr_) or any(abs(mw[k] - mr[k]) > 1e-7 for k in mr):
            problems.append("weighted loss differs from vanilla on hand-weighted advantages")
        if torch.allclose(lw, lv):
            problems.append("weighting had no effect on the test batch")
        if any(k not in mw or not 0 <= mw[k] <= 1 for k in METRIC_KEYS):
            problems.append(f"metrics missing/out of range: { {k: mw.get(k) for k in METRIC_KEYS} }")
    finally:
        globals()["_PARAMS"] = saved
    return problems


def _selftest() -> None:
    import torch
    old = torch.log(torch.tensor([[0.95, 0.5, 0.99, 0.2], [0.95, 0.5, 0.99, 0.2], [0.95, 0.5, 0.99, 0.2]]))
    adv = torch.tensor([[-1.0] * 4, [1.0] * 4, [0.0] * 4])
    mask = torch.tensor([[1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    w, hit = confidence_weights(old, adv, mask, tau=0.9, factor=0.2)
    # row 0 (A<0): 0.95 > 0.9 hit; 0.5 no; 0.99 masked out; 0.2 no — rows 1/2 (A>=0): never
    assert hit.tolist() == [[True, False, False, False], [False] * 4, [False] * 4], hit
    assert torch.allclose(w, torch.tensor([[0.2, 1, 1, 1], [1.0] * 4, [1.0] * 4])), w
    assert not w.requires_grad
    m = conf_metrics(hit, adv, mask)
    assert m == {METRIC_KEYS[0]: 1 / 11, METRIC_KEYS[1]: 3 / 11}, m
    w1, _ = confidence_weights(old, adv, mask, tau=0.9, factor=1.0)
    assert torch.equal(w1, torch.ones_like(adv))
    old.requires_grad_(True)
    w2, _ = confidence_weights(old, adv, mask, tau=0.5, factor=0.2)
    assert not w2.requires_grad
    for bad in ({"tau": None}, {"tau": 1.0}, {"tau": 0.5, "factor": 0.0}, {"tau": 0.5, "factor": 1.5}):
        try:
            _validate(bad)
        except RuntimeError:
            continue
        raise AssertionError(f"_validate accepted {bad}")
    print("grpo_conf_loss selftest OK")


# eager load inside a run: a missing/invalid conf_weight fails the worker start, not a training step
if os.environ.get("GRPO_RUN_DIR") and register_policy_loss is not None:
    _PARAMS = load_params()

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit("usage: grpo_conf_loss.py --selftest")
