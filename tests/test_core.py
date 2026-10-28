"""Pure-logic tests: preprocessing, label mapping, sampling, metrics, language helpers, load errors."""
import pytest

from evaluation_utils import balance_classes, collect_sample, compute_metrics, group_metrics
from language_utils import SUPPORTED_LANGUAGES, language_name
from preprocessing import normalize_label, prepare_text
from sentiment_engine import ModelManager, describe_load_error, labels_from_config


def test_prepare_text_keeps_scripts_and_cleans_whitespace():
    assert prepare_text("  景色が\n素晴らしい \t 😀\x00 ") == "景色が 素晴らしい 😀"
    assert prepare_text(None) == "" and prepare_text(float("nan")) == ""


def test_labels_from_config_maps_in_index_order():
    assert labels_from_config({0: "negative", 1: "neutral", 2: "positive"}) == ["Negative", "Neutral", "Positive"]
    assert labels_from_config({"0": "Positive", "1": "Negative", "2": "Neutral"}) == ["Positive", "Negative", "Neutral"]


def test_labels_from_config_refuses_to_guess():
    with pytest.raises(ValueError):
        labels_from_config({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_normalize_label_dataset_values():
    assert [normalize_label(x) for x in ("positive", "Neutral", "NEGATIVE", "foo", None)] == ["Positive", "Neutral", "Negative", None, None]


def test_describe_load_error():
    assert "downloaded" in describe_load_error(OSError("Max retries exceeded"))
    assert "pip install" in describe_load_error(ImportError("No module named torch"))


class Boom:
    def load(self):
        raise RuntimeError("bad weights")


def test_model_manager_reports_error_state():
    m = ModelManager(Boom()); m.load_sync()
    assert m.status == "error" and "bad weights" in m.error


def rows():
    data = []
    for i in range(6):
        data.append({"text": f"english review {i}", "label": "positive", "language": "en", "domain": "hotel reviews"})
    data += [{"text": "english review 0", "label": "positive", "language": "en", "domain": "hotel reviews"},   # duplicate
             {"text": "", "label": "positive", "language": "en", "domain": "x"},                             # empty
             {"text": "texto", "label": "weird", "language": "es", "domain": "x"},                           # bad label
             {"text": "hola bueno", "label": "neutral", "language": "es", "domain": "amazon reviews"},
             {"text": "hola malo", "label": "negative", "language": "es", "domain": "hotel reviews"}]
    return data


def test_collect_sample_quota_dedupe_and_invalid():
    s, st = collect_sample(rows(), per_language=3)
    assert sum(r["language"] == "en" for r in s) == 3 and sum(r["language"] == "es" for r in s) == 2
    s2, st2 = collect_sample(rows(), per_language=10)
    assert st2["duplicates_skipped"] == 1 and st2["invalid_skipped"] == 2


def test_collect_sample_domain_filter_and_early_stop():
    s, st = collect_sample(rows(), per_language=10, domain_keywords=["hotel"])
    assert {r["domain"] for r in s} == {"hotel reviews"} and st["domain_skipped"] == 1
    s, st = collect_sample(rows(), per_language=2, expected_languages=["en"])
    assert len(s) == 2 and st["scanned"] < len(rows())


def test_balance_classes():
    s = [{"label": "Positive"}] * 5 + [{"label": "Negative"}] * 2 + [{"label": "Neutral"}] * 3
    b = balance_classes(s)
    assert {l: sum(r["label"] == l for r in b) for l in ("Positive", "Negative", "Neutral")} == {"Positive": 2, "Negative": 2, "Neutral": 2}


def test_compute_metrics_matches_hand_calculation():
    y = ["Positive", "Positive", "Negative", "Negative", "Neutral", "Neutral"]
    p = ["Positive", "Negative", "Negative", "Negative", "Neutral", "Positive"]
    m = compute_metrics(y, p)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"]["Positive"]["precision"] == pytest.approx(.5) and m["per_class"]["Negative"]["recall"] == pytest.approx(1)
    assert m["macro"]["precision"] == pytest.approx((.5 + 2 / 3 + 1) / 3)
    assert m["confusion_matrix"]["labels"] == ["Positive", "Negative", "Neutral"]
    assert m["confusion_matrix"]["matrix"][0] == [1, 1, 0]


def test_group_metrics_flags_low_n():
    items = [{"label": "Positive", "prediction": "Positive", "language": "hi"}] * 3
    g = group_metrics(items, "language", min_n=100)
    assert g["hi"]["n"] == 3 and g["hi"]["reliable"] is False and g["hi"]["accuracy"] == 1


def test_language_tables():
    assert len(SUPPORTED_LANGUAGES) == 16 and language_name("hi") == "Hindi" and language_name(None) == "Unknown"
