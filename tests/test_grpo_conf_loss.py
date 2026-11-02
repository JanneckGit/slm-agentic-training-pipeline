"""Confidence weights of the conf_weighted GRPO loss (training_pipeline/grpo_conf_loss.py) — torch only.
Runs in the training container (ops/run_tests.sh step 2) and the CI collator job; skipped without torch.
The verl side (registry, equality to vanilla on hand-weighted advantages) is the preflight gate `loss`."""
import pytest

torch = pytest.importorskip("torch")

from training_pipeline.grpo_conf_loss import METRIC_KEYS, _selftest, confidence_weights, conf_metrics  # noqa: E402


def test_selftest():
    _selftest()


def test_weights_match_the_definition_on_random_tensors():
    g = torch.Generator().manual_seed(1)
    old = -torch.rand(5, 50, generator=g) * 3
    adv = torch.randn(5, 1, generator=g).expand(5, 50)
    mask = torch.rand(5, 50, generator=g) > 0.2
    w, hit = confidence_weights(old, adv, mask, tau=0.6, factor=0.25)
    ref = torch.ones(5, 50)
    for b in range(5):
        for t in range(50):
            if adv[b, t] < 0 and old[b, t].exp() > 0.6 and mask[b, t]:
                ref[b, t] = 0.25
    assert torch.equal(w, ref) and torch.equal(hit, ref < 1)
    m = conf_metrics(hit, adv, mask)
    assert set(m) == set(METRIC_KEYS) and 0 <= m[METRIC_KEYS[0]] <= m[METRIC_KEYS[1]] <= 1
