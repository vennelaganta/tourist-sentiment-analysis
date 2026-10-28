"""Evaluate the multilingual model on a language-stratified SAMPLE of clapAI/MultiLingualSentiment.

Streams the split with the Hugging Face `datasets` library (no full download, no giant CSV).

Workflow that keeps the test split clean:
  1) Model selection on VALIDATION:  python evaluate_model.py --split validation --model-id <candidate>
  2) Final report on TEST (once):    python evaluate_model.py --split test
Results are written to models/evaluation_<split>_<model>_<scope>.json and shown on the Dashboard.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation_utils import (EXPECTED_LANGUAGES, MIN_RELIABLE_N, balance_classes, collect_sample,
                              compute_metrics, group_metrics, text_hash)
from preprocessing import prepare_text
from sentiment_engine import DEFAULT_MODEL_ID, TransformersBackend

DATASET = "clapAI/MultiLingualSentiment"
BASE = Path(__file__).parent


def stream_split(split, seed):
    from datasets import load_dataset
    return load_dataset(DATASET, split=split, streaming=True).shuffle(seed=seed, buffer_size=10_000)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=["validation", "test"], default="validation",
                    help="validation for model selection, test for the final report (default: validation)")
    ap.add_argument("--model-id", default=None, help=f"Hugging Face model id (default {DEFAULT_MODEL_ID})")
    ap.add_argument("--per-language", type=int, default=100, help="reviews sampled per language (default 100)")
    ap.add_argument("--max-scan", type=int, default=0, help="max rows to scan (0 = whole split if needed)")
    ap.add_argument("--domains", default="", help='keep only domains containing these words, e.g. "hotel,restaurant,places"')
    ap.add_argument("--balance-classes", action="store_true", help="down-sample to equal class counts")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--overlap-scan", type=int, default=0,
                    help="optionally stream this many TRAIN rows and count eval texts that also occur there")
    ap.add_argument("--local-only", action="store_true", help="use only the local model cache (offline)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(BASE / "models"))
    args = ap.parse_args()
    keywords = [k.strip().lower() for k in args.domains.split(",") if k.strip()]

    print(f"[1/4] Streaming '{args.split}' split of {DATASET} (no full download) ...")
    try:
        rows = stream_split(args.split, args.seed)
        sample, stats = collect_sample(rows, args.per_language, args.max_scan or None, keywords or None,
                                       None if keywords else EXPECTED_LANGUAGES)
    except ImportError:
        sys.exit("The 'datasets' library is missing. Run: pip install -r requirements.txt")
    except Exception as exc:  # network / hub problems
        sys.exit(f"Could not stream the dataset: {exc}\nCheck your internet connection and retry (see README troubleshooting).")
    if args.balance_classes:
        sample = balance_classes(sample, args.seed)
    if not sample:
        sys.exit("No usable rows were collected (check --domains / --max-scan).")
    print(f"      collected {len(sample)} reviews from {stats['scanned']} scanned rows")

    overlap = None
    if args.overlap_scan > 0:
        print(f"[2/4] Checking overlap against the first {args.overlap_scan} TRAIN rows ...")
        wanted, found = {text_hash(r["text"]) for r in sample}, set()
        for i, row in enumerate(stream_split("train", args.seed)):
            if i >= args.overlap_scan:
                break
            if isinstance(row.get("text"), str) and text_hash(row["text"]) in wanted:
                found.add(text_hash(row["text"]))
        overlap = {"train_rows_scanned": args.overlap_scan, "eval_texts_also_in_scanned_train": len(found),
                   "note": "Lower bound only: just the scanned part of train was checked."}
        print(f"      {len(found)} of {len(wanted)} eval texts also occur in the scanned train rows")
    else:
        print("[2/4] Train-overlap check skipped (use --overlap-scan N to enable)")

    print("[3/4] Loading model and predicting (first run downloads the model) ...")
    backend = TransformersBackend(args.model_id, args.max_tokens, args.batch_size, args.local_only)
    try:
        backend.load()
    except Exception as exc:
        sys.exit(f"Model could not be loaded: {exc}")
    texts = [prepare_text(r["text"]) for r in sample]  # SAME preprocessing as live predictions
    preds, t0 = [], time.time()
    for i in range(0, len(texts), 64):
        preds += backend.predict(texts[i:i + 64])
        print(f"      {min(i + 64, len(texts))}/{len(texts)}", end="\r")
    elapsed = time.time() - t0
    for r, p in zip(sample, preds):
        r["prediction"], r["truncated"] = p["label"], p["truncated"]

    print("\n[4/4] Computing metrics ...")
    y_true, y_pred = [r["label"] for r in sample], [r["prediction"] for r in sample]
    lang_counts = {}
    for r in sample:
        lang_counts[r["language"]] = lang_counts.get(r["language"], 0) + 1
    class_counts = {l: y_true.count(l) for l in sorted(set(y_true))}
    dom_counts = {}
    for r in sample:
        dom_counts[r["domain"]] = dom_counts.get(r["domain"], 0) + 1
    scope = ("domains matching: " + ", ".join(keywords)) if keywords else "all domains"
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": backend.model_id, "split": args.split, "scope": scope,
        "balanced_classes": args.balance_classes,
        "sample": {"n": len(sample), "per_language_quota": args.per_language, **stats,
                   "language_counts": dict(sorted(lang_counts.items())), "class_counts": class_counts,
                   "domain_counts": dict(sorted(dom_counts.items(), key=lambda kv: -kv[1])[:15]),
                   "truncated_reviews": sum(r["truncated"] for r in sample),
                   "min_reliable_n": MIN_RELIABLE_N, "overlap_check": overlap,
                   "missing_expected_languages": [l for l in EXPECTED_LANGUAGES if l not in lang_counts]},
        "overall": compute_metrics(y_true, y_pred),
        "per_language": group_metrics(sample, "language"),
        "per_domain": group_metrics(sample, "domain"),
        "runtime": {**backend.info(), "reviews_per_second": round(len(sample) / elapsed, 2) if elapsed else None},
        "notes": ["Language-stratified sample: overall scores weight every language equally, so they are not "
                  "the same as scores on the natural (English-heavy) test distribution.",
                  "Per-language scores with fewer than %d examples are flagged unreliable." % MIN_RELIABLE_N],
    }
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", backend.model_id.lower()).strip("-")
    suffix = ("_" + re.sub(r"[^a-z0-9]+", "-", "-".join(keywords))) if keywords else ""
    suffix += "_balanced" if args.balance_classes else ""
    path = out_dir / f"evaluation_{args.split}_{slug}{suffix}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    o = result["overall"]
    print(f"\nSaved {path}\nn={len(sample)}  accuracy={o['accuracy']:.4f}  macro-F1={o['macro']['f1']:.4f}  "
          f"weighted-F1={o['weighted']['f1']:.4f}  speed={result['runtime']['reviews_per_second']} reviews/s on {backend.device}")


if __name__ == "__main__":
    main()
