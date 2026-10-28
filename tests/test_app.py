"""API tests. They use FakeBackend, a TEST DOUBLE that stands in for the transformer so the Flask
logic (validation, language handling, history, errors) can be tested without downloading a model.
It is never used by the real website. Real-model checks live in test_model_integration.py."""
import io
import json

import pytest

from app import create_app


class FakeBackend:
    model_id = "test-double/not-a-real-model"

    def __init__(self, fail=False):
        self.fail, self.last_text = fail, None

    def load(self):
        if self.fail:
            raise OSError("Max retries exceeded: cannot connect to huggingface.co")

    def predict(self, texts):
        out = []
        for t in texts:
            self.last_text = t
            label = "Positive" if "good" in t else "Negative" if "bad" in t else "Neutral"
            probs = {"Positive": .1, "Negative": .1, "Neutral": .1}
            probs[label] = .8
            out.append({"label": label, "probabilities": probs, "confidence": .8,
                        "n_tokens": 12, "truncated": "LONGREVIEW" in t})
        return out

    def info(self):
        return {"model_id": self.model_id, "device": "cpu", "labels": ["Positive", "Negative", "Neutral"], "max_tokens": 256}


def make(tmp_path, preload="sync", backend=None):
    return create_app({"TESTING": True, "PRELOAD": preload, "BACKEND": backend or FakeBackend(),
                       "MODELS_DIR": tmp_path / "models", "HISTORY_PATH": tmp_path / "hist.csv",
                       "PROFILE_PATH": tmp_path / "profile.json", "UPLOAD_DIR": tmp_path / "up"})


@pytest.fixture
def client(tmp_path):
    return make(tmp_path).test_client()


def post(c, review, **extra):
    return c.post("/api/predict", json={"review": review, **extra})


def test_health_ready(client):
    d = client.get("/api/health").get_json()
    assert d["model_ready"] and d["model_status"] == "ready" and len(d["supported_languages"]) == 16


def test_model_not_loaded_returns_503_not_a_fake_answer(tmp_path):
    r = post(make(tmp_path, preload="off").test_client(), "good place")
    assert r.status_code == 503 and "error" in r.get_json()


def test_failed_download_reports_reason(tmp_path):
    c = make(tmp_path, backend=FakeBackend(fail=True)).test_client()
    assert c.get("/api/health").get_json()["model_status"] == "error"
    r = post(c, "good place")
    assert r.status_code == 503 and "downloaded" in r.get_json()["error"]


def test_predict_with_chosen_language_and_original_text_preserved(tmp_path):
    app = make(tmp_path)
    c = app.test_client()
    raw = "  El hotel era   muy good\n\n y limpio  "
    d = post(c, raw, language="es").get_json()
    assert d["sentiment"] == "Positive" and d["confidence"] == .8
    assert d["language"] == "es" and d["language_source"] == "user" and d["language_supported"] is True
    assert d["review"] == raw                       # original preserved exactly
    assert d["probabilities"]["Positive"] == .8 and set(d["probabilities"]) == {"Positive", "Negative", "Neutral"}


def test_backend_receives_normalised_text(tmp_path):
    backend = FakeBackend()
    c = make(tmp_path, backend=backend).test_client()
    post(c, "  a   bad\n\n trip  ", language="en")
    assert backend.last_text == "a bad trip"


def test_unsupported_language_is_flagged_not_rejected(client):
    d = post(client, "dit is good", language="nl").get_json()
    assert d["language_supported"] is False and any("fine-tuning" in w for w in d["warnings"])


@pytest.mark.parametrize("bad", ["english!!", "e", "1234", "../x"])
def test_invalid_language_code_rejected(client, bad):
    assert post(client, "good", language=bad).status_code == 400


def test_auto_language_degrades_gracefully(client):
    d = post(client, "ok", language="auto").get_json()   # 2 letters: cannot be detected
    assert d["language"] is None and d["language_source"] == "unknown" and d["warnings"]


def test_auto_detection_when_langdetect_installed(client):
    pytest.importorskip("langdetect")
    d = post(client, "El hotel estaba muy limpio y el personal fue amable con nosotros.").get_json()
    assert d["language"] == "es" and d["language_source"] == "detected"


def test_truncation_warning(client):
    d = post(client, "LONGREVIEW good", language="en").get_json()
    assert d["truncated"] and any("only the beginning" in w for w in d["warnings"])


@pytest.mark.parametrize("payload", [{"review": ""}, {"review": "   "}, {"review": 5}, {"other": "x"}])
def test_bad_input_rejected(client, payload):
    assert client.post("/api/predict", json=payload).status_code == 400


def test_non_json_and_too_long_rejected(client):
    assert client.post("/api/predict", data="hello").status_code == 400
    assert post(client, "x" * 5001).status_code == 400


def test_multilingual_unicode_round_trip_and_dashboard(client):
    reviews = [("景色が本当に素晴らしく、good", "ja"), ("होटल bad था", "hi"), ("المكان good", "ar"), ("plain text", "en")]
    for text, lang in reviews:
        assert post(client, text, language=lang).status_code == 200
    dash = client.get("/api/dashboard").get_json()
    assert dash["total"] == 4 and dash["counts"] == {"Positive": 2, "Negative": 1, "Neutral": 1}
    assert dash["languages"] == {"ja": 1, "hi": 1, "ar": 1, "en": 1}
    hist = client.get("/api/history?language=hi").get_json()
    assert hist["matched"] == 1 and hist["items"][0]["review"] == "होटल bad था"
    assert client.get("/api/history?sentiment=Positive&q=%E6%99%AF").get_json()["matched"] == 1
    assert client.get("/api/history?limit=abc").status_code == 400


def test_dashboard_empty_state(client):
    d = client.get("/api/dashboard").get_json()
    assert d["total"] == 0 and d["languages"] == {} and d["average_confidence"] is None


def test_old_format_history_is_preserved_not_corrupted(tmp_path):
    (tmp_path / "hist.csv").write_text("timestamp,review,sentiment,confidence\n2024-01-01,old,Positive,0.9\n")
    c = make(tmp_path).test_client()
    post(c, "good", language="en")
    assert len(list(tmp_path.glob("predictions_old_*.csv"))) == 1      # old file kept, not overwritten
    assert c.get("/api/dashboard").get_json()["total"] == 1


def test_metrics_and_profile_empty_states(client):
    assert client.get("/api/metrics").status_code == 404
    assert client.get("/api/dataset-profile").status_code == 404


def test_metrics_endpoint_reads_run_files(tmp_path):
    (tmp_path / "models").mkdir()
    fixture = {"created_at": "2000-01-01T00:00:00", "split": "test", "model_id": "fixture/only", "overall": {}}  # fixture, not a real result
    (tmp_path / "models" / "evaluation_test_fixture.json").write_text(json.dumps(fixture))
    (tmp_path / "models" / "evaluation_broken.json").write_text("{not json")
    d = make(tmp_path).test_client().get("/api/metrics").get_json()
    assert len(d["runs"]) == 1 and d["runs"][0]["model_id"] == "fixture/only"


def test_upload_rejects_non_csv(client):
    r = client.post("/api/upload-dataset", data={"file": (io.BytesIO(b"x"), "evil.exe")}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_upload_csv_preview_and_summary(client):
    body = "Review,Sentiment\nGreat place,Positive\nBad food,Negative\nOkay visit,Neutral\n,Positive\n".encode()
    d = client.post("/api/upload-dataset", data={"file": (io.BytesIO(body), "d.csv")}, content_type="multipart/form-data").get_json()
    assert d["rows"] == 4 and d["suggested_text_column"] == "Review"
    s = client.post("/api/dataset-summary", json={"upload_id": d["upload_id"], "text_column": "Review", "label_column": "Sentiment"}).get_json()
    assert s["label_distribution"] == {"Positive": 2, "Negative": 1, "Neutral": 1}
    assert client.post("/api/dataset-summary", json={"upload_id": "../x", "text_column": "a"}).status_code == 400


def test_unknown_api_route_is_json_404(client):
    r = client.get("/api/nothing")
    assert r.status_code == 404 and "error" in r.get_json()


def test_all_pages_render(client):
    for path in ["/", "/analyze", "/dashboard", "/dataset", "/about"]:
        assert client.get(path).status_code == 200
