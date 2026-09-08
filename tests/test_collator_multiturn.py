"""Assistant-only loss-mask golden test — the module's own self-test. Needs torch + transformers and the
Qwen3-4B tokenizer (downloaded on first run): the training container (`ops/run_tests.sh`) or the CI collator job."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
from training_pipeline.collator_multiturn import _selftest  # noqa: E402


def test_collator_multiturn_selftest():
    _selftest()
