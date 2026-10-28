"""OPTIONAL classical baseline (TF-IDF + NB/SVM/RF) on a CSV you supply. NOT used by the website.

Usage (from the project folder, venv active):
    python classical_baseline/train_model.py --data data/reviews.csv --text-col Review --label-col Sentiment

Outputs:
    models/sentiment_pipeline.joblib   (TF-IDF + best classifier, one Pipeline object)
    models/metrics.json                (real metrics measured on the held-out test set)
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow running from this subfolder
from preprocessing import LABELS, clean_text, guess_columns, normalize_label, parse_label_map

MODEL_NAMES = ["Naive Bayes", "SVM", "Random Forest"]
BASE = Path(__file__).parent  # outputs go to classical_baseline/output (separate from the website model)


def build_pipeline(name, min_df=2, cv=3):
    """TF-IDF is a step INSIDE the pipeline, so it is fitted on training data only."""
    tfidf = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=min_df,
                            sublinear_tf=True, max_features=50000)
    if name == "Naive Bayes":
        clf = MultinomialNB()
    elif name == "SVM":  # LinearSVC + calibration gives probability estimates
        clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=cv)
    elif name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=42)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def read_csv(path):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the CSV file.")


def evaluate(pipe, x_test, y_test):
    pred = pipe.predict(x_test)
    labels = [l for l in LABELS if l in set(y_test) | set(pred)]
    out = {"accuracy": accuracy_score(y_test, pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_test, pred, average=avg, zero_division=0)
        out[avg] = {"precision": p, "recall": r, "f1": f}
    rep = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
    out["per_class"] = {l: {k: rep[l][k] for k in ("precision", "recall", "f1-score", "support")}
                        for l in labels}
    out["confusion_matrix"] = {"labels": labels,
                               "matrix": confusion_matrix(y_test, pred, labels=labels).tolist()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="Path to labelled CSV")
    ap.add_argument("--text-col", help="Review text column (auto-detected if omitted)")
    ap.add_argument("--label-col", help="Sentiment label column (auto-detected if omitted)")
    ap.add_argument("--label-map", help='Custom label map, e.g. "0=Negative,1=Neutral,2=Positive"')
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = read_csv(args.data)
    g_text, g_label = guess_columns(list(df.columns))
    text_col, label_col = args.text_col or g_text, args.label_col or g_label
    for col in (text_col, label_col):
        if col not in df.columns:
            sys.exit(f"Column {col!r} not found. Available columns: {list(df.columns)}")
    print(f"Using text column '{text_col}' and label column '{label_col}'")

    rows_raw = len(df)
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    custom = parse_label_map(args.label_map)
    df["label"] = df["label"].map(lambda v: normalize_label(v, custom))
    unknown = df["label"].isna().sum()
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).map(clean_text).str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset="text")          # duplicates removed BEFORE splitting
    print(f"Rows: raw={rows_raw}, dropped missing/unmapped={rows_raw - before}"
          f" (unmapped labels={unknown}), duplicates removed={before - len(df)}, kept={len(df)}")
    counts = df["label"].value_counts()
    print("Class distribution:\n" + counts.to_string())
    if len(counts) < 2 or counts.min() < 5:
        sys.exit("Need at least 2 classes with 5+ reviews each. If your labels are 0/1/2 use --label-map.")
    if "Neutral" not in counts:
        print("WARNING: no Neutral reviews found - the model will only predict the classes it saw.")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"].astype(str), df["label"], test_size=args.test_size,
        stratify=df["label"], random_state=args.seed)
    cv = max(2, min(3, int(y_train.value_counts().min())))

    results, fitted = {}, {}
    for name in MODEL_NAMES:
        print(f"\nTraining {name} ...")
        pipe = build_pipeline(name, cv=cv).fit(x_train, y_train)
        results[name], fitted[name] = evaluate(pipe, x_test, y_test), pipe
        print(f"  accuracy={results[name]['accuracy']:.4f}  macro-F1={results[name]['macro']['f1']:.4f}")

    best = max(MODEL_NAMES, key=lambda n: (results[n]["macro"]["f1"], results[n]["accuracy"]))
    print(f"\nBest model by macro-F1: {best}")

    models_dir = BASE / "output"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(fitted[best], models_dir / "sentiment_pipeline.joblib")
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_model": best, "selection_metric": "macro F1 on held-out test set",
        "dataset": {"file": Path(args.data).name, "rows_raw": rows_raw, "rows_used": len(df),
                    "class_counts": {k: int(v) for k, v in counts.items()},
                    "n_train": len(x_train), "n_test": len(x_test), "test_size": args.test_size},
        "models": results,
    }
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved classical_baseline/output/sentiment_pipeline.joblib and metrics.json")
    print(classification_report(y_test, fitted[best].predict(x_test), zero_division=0))


if __name__ == "__main__":
    main()
