"""Inspect the REAL schema, labels, languages and domains of clapAI/MultiLingualSentiment (streamed sample).

    python inspect_dataset.py            # 20,000-row sample  (~1-2 min)
    python inspect_dataset.py --quick    # 2,000-row sample

Writes data/dataset_profile.json, which the Dataset page displays. Counts come from the sample ONLY.
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATASET = "clapAI/MultiLingualSentiment"
BASE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n_target = 2000 if args.quick else args.sample
    try:
        from datasets import load_dataset, load_dataset_builder
    except ImportError:
        sys.exit("The 'datasets' library is missing. Run: pip install -r requirements.txt")

    profile = {"dataset": DATASET, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        info = load_dataset_builder(DATASET).info
        profile["features"] = {k: str(v) for k, v in (info.features or {}).items()}
        profile["split_sizes"] = {k: v.num_examples for k, v in (info.splits or {}).items()}
    except Exception as exc:
        print(f"(could not read dataset metadata: {exc})")
        profile["features"], profile["split_sizes"] = {}, {}

    print(f"Streaming {n_target} training rows ...")
    try:
        stream = load_dataset(DATASET, split="train", streaming=True).shuffle(seed=args.seed, buffer_size=10_000)
        labels, langs, domains, lengths, examples = Counter(), Counter(), Counter(), [], []
        for i, row in enumerate(stream):
            if i >= n_target:
                break
            labels[row["label"]] += 1
            langs[row["language"]] += 1
            domains[row["domain"]] += 1
            lengths.append(len(row["text"]))
            if len(examples) < 5:
                examples.append({k: (str(v)[:140]) for k, v in row.items()})
    except Exception as exc:
        sys.exit(f"Could not stream the dataset: {exc}\nCheck your internet connection and retry.")

    profile["sample"] = {
        "rows": len(lengths), "note": "Distributions below are from this streamed sample only, not the full dataset.",
        "label_counts": dict(labels.most_common()), "language_counts": dict(langs.most_common()),
        "domain_counts": dict(domains.most_common(30)),
        "text_length_chars": {"median": statistics.median(lengths), "max": max(lengths),
                              "p95": sorted(lengths)[int(0.95 * len(lengths)) - 1]},
        "examples": examples}
    out = BASE / "data" / "dataset_profile.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    print("\nColumns:", list(profile["features"]) or list(examples[0]))
    print("Split sizes:", profile["split_sizes"] or "n/a")
    print("Labels:", profile["sample"]["label_counts"])
    print("Languages:", profile["sample"]["language_counts"])
    print("Domains (top):", list(profile["sample"]["domain_counts"].items())[:10])
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
