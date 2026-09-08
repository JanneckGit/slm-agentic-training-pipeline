"""Sampler math (smoothed weights, low-variance allocation, aging, ledger) — the module's own self-test, stdlib only."""
from training_pipeline.grpo_sampling import _selftest


def test_grpo_sampling_selftest():
    _selftest()
