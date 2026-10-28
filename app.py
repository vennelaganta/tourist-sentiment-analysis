"""Flask backend for Tourist Sentiment Analysis (multilingual).

Predictions come from a pretrained multilingual transformer (see sentiment_engine.py).
The model is loaded once in the background - nothing is trained at start-up - and if it is
unavailable the API answers 503 with the reason; it never returns a made-up prediction.
"""
import csv
import glob
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from language_utils import SUPPORTED_LANGUAGES, detect_language, detection_available, language_name
from preprocessing import LABELS, guess_columns, normalize_label, prepare_text
from sentiment_engine import DEFAULT_MODEL_ID, ModelManager, TransformersBackend

BASE = Path(__file__).parent
HISTORY_FIELDS = ["timestamp", "review", "sentiment", "confidence", "language", "language_source", "model"]


def create_app(config=None):
    app = Flask(__name__)
    app.config.update(
        MODELS_DIR=BASE / "models",
        HISTORY_PATH=BASE / "data" / "predictions.csv",
        PROFILE_PATH=BASE / "data" / "dataset_profile.json",
        UPLOAD_DIR=BASE / "data" / "uploads",
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
        MAX_REVIEW_CHARS=5000,
        PRELOAD=os.environ.get("SENTIMENT_PRELOAD", "background"),  # background | sync | off
        BACKEND=None,  # tests inject a test double here
    )
    if config:
        app.config.update(config)

    backend = app.config["BACKEND"] or TransformersBackend(
        os.environ.get("SENTIMENT_MODEL_ID", DEFAULT_MODEL_ID),
        max_tokens=int(os.environ.get("SENTIMENT_MAX_TOKENS", "256")),
        local_only=os.environ.get("SENTIMENT_LOCAL_ONLY", "0") == "1")
    manager = ModelManager(backend)
    lock = threading.Lock()
    if app.config["PRELOAD"] == "sync":
        manager.load_sync()
    elif app.config["PRELOAD"] == "background":
        manager.load_background()

    # ------------------------------------------------------------ helpers
    def err(message, status, **extra):
        return jsonify({"error": message, **extra}), status

    def read_history():
        path = Path(app.config["HISTORY_PATH"])
        if not path.exists():
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def save_prediction(row):
        path = Path(app.config["HISTORY_PATH"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            if path.exists():  # keep an older-format file instead of corrupting it
                with open(path, newline="", encoding="utf-8") as f:
                    header = next(csv.reader(f), [])
                if header != HISTORY_FIELDS:
                    path.rename(path.with_name(f"predictions_old_{datetime.now():%Y%m%d%H%M%S}.csv"))
            new = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)

    def read_upload_df(path):
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc, dtype=str)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode the file as text.")

    # -------------------------------------------------------------- pages
    for rule, endpoint, template in [("/", "index", "index.html"), ("/analyze", "analyze", "analyze.html"),
                                     ("/dashboard", "dashboard", "dashboard.html"),
                                     ("/dataset", "dataset", "dataset.html"), ("/about", "about", "about.html")]:
        app.add_url_rule(rule, endpoint, (lambda t: lambda: render_template(t))(template))

    # ---------------------------------------------------------------- API
    @app.get("/api/health")
    def health():
        info = backend.info() if manager.status == "ready" else {"model_id": getattr(backend, "model_id", None)}
        return jsonify({"status": "ok", "model_status": manager.status, "model_ready": manager.status == "ready",
                        "model_error": manager.error, "model": info,
                        "language_detection": detection_available(),
                        "supported_languages": SUPPORTED_LANGUAGES,
                        "evaluation_available": bool(glob.glob(str(Path(app.config["MODELS_DIR"]) / "evaluation_*.json")))})

    @app.post("/api/model/reload")
    def reload_model():
        manager.load_background()
        return jsonify({"model_status": manager.status})

    @app.post("/api/predict")
    def predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or "review" not in data:
            return err('Send JSON like {"review": "your text", "language": "auto"}.', 400)
        review = data["review"]
        if not isinstance(review, str) or not review.strip():
            return err("Please enter a review before analyzing.", 400)
        if len(review) > app.config["MAX_REVIEW_CHARS"]:
            return err(f"Review is too long (max {app.config['MAX_REVIEW_CHARS']} characters).", 400)
        lang_in = str(data.get("language") or "auto").strip().lower()
        if lang_in != "auto" and not re.fullmatch(r"[a-z]{2,3}", lang_in):
            return err("language must be 'auto' or a 2-3 letter code such as 'en' or 'hi'.", 400)

        if manager.status != "ready":
            if manager.status == "not_loaded":
                manager.load_background()
            msg = ("The multilingual model is still loading (the first start downloads it). Try again in a moment."
                   if manager.status == "loading" else (manager.error or "Model is not available."))
            return err(msg, 503, model_status=manager.status)

        text = prepare_text(review)  # same normalisation as evaluation
        warnings = []
        if lang_in == "auto":
            lang, conf = detect_language(text)
            source = "detected" if lang else "unknown"
            if lang is None:
                warnings.append("Language could not be detected automatically" +
                                ("" if detection_available() else " (langdetect is not installed)") +
                                "; the model still classified the text.")
            elif conf is not None and conf < 0.7:
                warnings.append(f"Language detection is uncertain ({conf:.0%}); choose the language manually if it is wrong.")
        else:
            lang, source, conf = lang_in, "user", None
        supported = None if lang is None else lang in SUPPORTED_LANGUAGES
        if supported is False:
            warnings.append(f"{language_name(lang)} was not among the model's fine-tuning languages. The underlying "
                            "XLM-RoBERTa was pretrained on ~100 languages so it may still work, but treat the result as less reliable.")
        try:
            res = manager.predict([text])[0]
        except Exception as exc:
            return err(f"Prediction failed: {exc}", 500)
        if res.get("truncated"):
            warnings.append(f"The review is longer than the model limit ({backend.info().get('max_tokens')} tokens); only the beginning was analysed.")

        label, conf_score = res["label"], res["confidence"]
        save_prediction({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "review": review,
                         "sentiment": label, "confidence": round(conf_score, 4), "language": lang or "",
                         "language_source": source, "model": backend.info().get("model_id", "")})
        return jsonify({"review": review, "sentiment": label, "confidence": conf_score,
                        "probabilities": res["probabilities"], "language": lang, "language_name": language_name(lang),
                        "language_source": source, "language_confidence": conf, "language_supported": supported,
                        "n_tokens": res.get("n_tokens"), "truncated": bool(res.get("truncated")),
                        "model": backend.info().get("model_id"), "warnings": warnings,
                        "explanation": explain(label, res["probabilities"], lang, supported)})

    @app.get("/api/dashboard")
    def dashboard_stats():
        rows = read_history()
        counts = {l: sum(1 for r in rows if r["sentiment"] == l) for l in LABELS}
        langs = {}
        for r in rows:
            code = r.get("language") or "unknown"
            langs[code] = langs.get(code, 0) + 1
        confs = [float(r["confidence"]) for r in rows if r.get("confidence")]
        return jsonify({"total": len(rows), "counts": counts, "languages": dict(sorted(langs.items(), key=lambda kv: -kv[1])),
                        "language_names": {c: language_name(c if c != "unknown" else None) for c in langs},
                        "average_confidence": sum(confs) / len(confs) if confs else None})

    @app.get("/api/history")
    def history():
        rows = read_history()[::-1]
        q = request.args.get("q", "").strip().lower()
        sentiment = request.args.get("sentiment", "").strip().title()
        language = request.args.get("language", "").strip().lower()
        if sentiment in LABELS:
            rows = [r for r in rows if r["sentiment"] == sentiment]
        if language:
            rows = [r for r in rows if (r.get("language") or "unknown") == language]
        if q:
            rows = [r for r in rows if q in r["review"].lower()]
        try:
            limit = max(1, min(int(request.args.get("limit", 50)), 500))
        except ValueError:
            return err("limit must be a number.", 400)
        return jsonify({"matched": len(rows), "items": rows[:limit]})

    @app.get("/api/metrics")
    def metrics():
        runs = []
        for p in sorted(glob.glob(str(Path(app.config["MODELS_DIR"]) / "evaluation_*.json"))):
            try:
                run = json.loads(Path(p).read_text(encoding="utf-8"))
                run["run_id"] = Path(p).stem
                runs.append(run)
            except (OSError, ValueError):
                continue  # ignore unreadable files rather than fail the dashboard
        if not runs:
            return err("No evaluation results yet. Run evaluate_model.py to generate them.", 404)
        runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)           # newest first
        runs.sort(key=lambda r: r.get("split") != "test")                        # test-split runs first (stable)
        return jsonify({"active_model": backend.info().get("model_id"), "runs": runs})

    @app.get("/api/dataset-profile")
    def dataset_profile():
        path = Path(app.config["PROFILE_PATH"])
        if not path.exists():
            return err("No dataset profile yet. Run: python inspect_dataset.py", 404)
        return jsonify(json.loads(path.read_text(encoding="utf-8")))

    @app.post("/api/upload-dataset")
    def upload_dataset():
        file = request.files.get("file")
        if file is None or not file.filename:
            return err("No file received. Choose a .csv file.", 400)
        name = secure_filename(file.filename)
        if not name.lower().endswith(".csv"):
            return err("Only .csv files are accepted.", 400)
        upload_id = uuid.uuid4().hex  # server-chosen name; the file is only ever read as data
        folder = Path(app.config["UPLOAD_DIR"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{upload_id}.csv"
        file.save(path)
        try:
            df = read_upload_df(path)
            if df.empty or len(df.columns) == 0:
                raise ValueError("The CSV has no rows.")
        except Exception as exc:
            path.unlink(missing_ok=True)
            return err(f"Could not read this CSV: {exc}", 400)
        text_col, label_col = guess_columns(list(df.columns))
        preview = df.head(10).fillna("").astype(str).apply(lambda c: c.str.slice(0, 200))
        return jsonify({
            "upload_id": upload_id, "filename": name, "rows": len(df), "columns": len(df.columns),
            "column_info": [{"name": str(c), "missing": int(df[c].isna().sum())} for c in df.columns],
            "preview": preview.to_dict(orient="records"),
            "suggested_text_column": text_col, "suggested_label_column": label_col})

    @app.post("/api/dataset-summary")
    def dataset_summary():
        data = request.get_json(silent=True) or {}
        upload_id, text_col, label_col = data.get("upload_id", ""), data.get("text_column"), data.get("label_column")
        if not re.fullmatch(r"[0-9a-f]{32}", str(upload_id)):
            return err("Invalid upload id.", 400)
        path = Path(app.config["UPLOAD_DIR"]) / f"{upload_id}.csv"
        if not path.exists():
            return err("Uploaded file not found. Please upload it again.", 404)
        df = read_upload_df(path)
        if text_col not in df.columns:
            return err("Select a valid review-text column.", 400)
        if label_col and label_col not in df.columns:
            return err("The selected label column does not exist.", 400)
        text = df[text_col].dropna().astype(str)
        out = {"text_column": text_col, "usable_reviews": int((text.str.strip() != "").sum()),
               "missing_text": int(df[text_col].isna().sum()), "duplicate_reviews": int(text.duplicated().sum()),
               "avg_words": round(float(text.str.split().str.len().mean()), 1) if len(text) else 0}
        if label_col:
            raw = df[label_col]
            mapped = raw.map(normalize_label)
            out["label_column"] = label_col
            out["label_distribution"] = {k: int(v) for k, v in mapped.value_counts().items()}
            out["unrecognised_labels"] = sorted({str(v) for v in raw[mapped.isna() & raw.notna()]})[:20]
            out["missing_labels"] = int(raw.isna().sum())
        return jsonify(out)

    @app.errorhandler(413)
    def too_large(_):
        return err("File is too large (limit 10 MB).", 413)

    @app.errorhandler(404)
    def not_found(_):
        if request.path.startswith("/api/"):
            return err("Endpoint not found.", 404)
        return render_template("index.html"), 404

    @app.errorhandler(500)
    def server_error(_):
        return err("Unexpected server error.", 500)

    return app


def explain(label, probs, lang, supported):
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    text = f"The multilingual transformer gives {label} the highest probability ({ranked[0][1]:.0%})"
    if len(ranked) > 1:
        text += f", ahead of {ranked[1][0]} ({ranked[1][1]:.0%})"
    text += ". "
    if ranked[0][1] < 0.5:
        text += "The probabilities are close, so the review may express mixed or unclear sentiment. "
    if lang and supported:
        text += f"{language_name(lang)} is one of the languages the model was fine-tuned on."
    return text.strip()


if __name__ == "__main__":
    # debug=False on purpose: Flask's debug reloader would start the model loader twice.
    create_app().run(debug=False, port=5000)
