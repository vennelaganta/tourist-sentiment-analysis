"""Pure-Python helpers for sampling and scoring (no torch needed, so they are unit-tested)."""
import hashlib
import random
from collections import Counter

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from preprocessing import LABELS, normalize_label, prepare_text

# Languages named on the dataset card (the 17th "Multilingual" bucket has no ISO code).
EXPECTED_LANGUAGES = ["ar", "de", "en", "es", "fr", "hi", "id", "it", "ja", "ko", "ms", "pt", "ru", "tr", "vi", "zh"]
MIN_RELIABLE_N = 100  # heuristic: fewer examples than this -> per-language scores are flagged as unreliable


def text_hash(text):
    return hashlib.md5(prepare_text(text).encode("utf-8")).hexdigest()


def collect_sample(rows, per_language, max_scan=None, domain_keywords=None, expected_languages=None):
    """Take up to `per_language` valid, de-duplicated rows per language from an iterable of dict rows.

    Stops early once every expected language is full. Returns (sample, stats)."""
    counts, seen, sample, filled = Counter(), set(), [], set()
    stats = {"scanned": 0, "invalid_skipped": 0, "duplicates_skipped": 0, "domain_skipped": 0}
    want = set(expected_languages or [])
    for row in rows:
        stats["scanned"] += 1
        if max_scan and stats["scanned"] > max_scan:
            break
        text, label = row.get("text"), normalize_label(row.get("label"))
        if not isinstance(text, str) or not prepare_text(text) or label is None:
            stats["invalid_skipped"] += 1
            continue
        if domain_keywords and not any(k in str(row.get("domain") or "").lower() for k in domain_keywords):
            stats["domain_skipped"] += 1
            continue
        lang = str(row.get("language") or "unknown").lower()
        if counts[lang] >= per_language:
            continue
        h = text_hash(text)
        if h in seen:
            stats["duplicates_skipped"] += 1
            continue
        seen.add(h)
        counts[lang] += 1
        sample.append({"text": text, "label": label, "language": lang, "domain": str(row.get("domain") or "")})
        if counts[lang] >= per_language:
            filled.add(lang)
            if want and want <= filled:
                break
    return sample, stats


def balance_classes(sample, seed=42):
    """Down-sample so every class has as many rows as the rarest class (optional balanced evaluation)."""
    by = {l: [r for r in sample if r["label"] == l] for l in LABELS}
    n = min((len(v) for v in by.values() if v), default=0)
    rng, out = random.Random(seed), []
    for rows in by.values():
        out += rng.sample(rows, n) if n and len(rows) > n else (rows if n else [])
    return out


def compute_metrics(y_true, y_pred):
    labels = [l for l in LABELS if l in set(y_true) | set(y_pred)]
    out = {"accuracy": accuracy_score(y_true, y_pred), "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out[avg] = {"precision": p, "recall": r, "f1": f}
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    out["per_class"] = {l: {"precision": rep[l]["precision"], "recall": rep[l]["recall"],
                            "f1": rep[l]["f1-score"], "support": int(rep[l]["support"])} for l in labels}
    out["confusion_matrix"] = {"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}
    return out


def group_metrics(rows, key, min_n=MIN_RELIABLE_N):
    """rows: dicts with label / prediction / <key>. Returns {group: {n, accuracy, macro_f1, weighted_f1, reliable}}."""
    groups = {}
    for r in rows:
        groups.setdefault(r[key] or "unknown", []).append(r)
    out = {}
    for g, items in sorted(groups.items()):
        y, p = [i["label"] for i in items], [i["prediction"] for i in items]
        _, _, mf, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
        _, _, wf, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
        out[g] = {"n": len(items), "accuracy": accuracy_score(y, p), "macro_f1": mf, "weighted_f1": wf,
                  "reliable": len(items) >= min_n}
    return out
