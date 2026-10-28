"""REAL-MODEL tests. Skipped by default because they download/load the transformer.
Run on your laptop:   $env:RUN_MODEL_TESTS="1"; python -m pytest tests/test_model_integration.py -q
Sentences are the examples published on the model's Hugging Face card."""
import os

import pytest

from sentiment_engine import TransformersBackend

pytestmark = pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to load the real model")

CASES = [("I absolutely love the new design of this app!", "Positive"), ("The customer service was disappointing.", "Negative"),
         ("Ich bin sehr zufrieden mit dem Kauf.", "Positive"), ("El producto llegó roto y no funciona.", "Negative"),
         ("サービスがひどかったです。", "Negative"), ("هذا المنتج رائع للغاية!", "Positive")]


@pytest.fixture(scope="module")
def backend():
    b = TransformersBackend()
    b.load()
    return b


def test_label_mapping_is_the_expected_three_classes(backend):
    assert sorted(backend.labels) == ["Negative", "Neutral", "Positive"]


@pytest.mark.parametrize("text,expected", CASES)
def test_multilingual_predictions(backend, text, expected):
    r = backend.predict([text])[0]
    assert r["label"] == expected and abs(sum(r["probabilities"].values()) - 1) < 1e-4


def test_long_text_is_truncated_not_crashing(backend):
    assert backend.predict(["great view " * 500])[0]["truncated"] is True


def test_batch_matches_single(backend):
    texts = [c[0] for c in CASES[:3]]
    batch = [r["label"] for r in backend.predict(texts)]
    assert batch == [backend.predict([t])[0]["label"] for t in texts]
