"""Combine multiple dev prediction files by specificity bucket.

This lets us stitch together the best-ranked model per query type and evaluate
the resulting routed ensemble without rerunning inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_dev_ndcg import build_specificity_map, evaluate_predictions, print_report


def load_predictions(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def index_predictions(predictions: list[dict]) -> dict[tuple[str, int], dict]:
    indexed: dict[tuple[str, int], dict] = {}
    for row in predictions:
        key = (row["session_id"], int(row["turn_number"]))
        indexed[key] = row
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine dev predictions by specificity bucket.")
    parser.add_argument("--hh", type=str, required=True, help="Prediction file for HH turns.")
    parser.add_argument("--lh", type=str, required=True, help="Prediction file for LH turns.")
    parser.add_argument("--hl", type=str, required=True, help="Prediction file for HL turns.")
    parser.add_argument("--ll", type=str, required=True, help="Prediction file for LL turns.")
    parser.add_argument(
        "--default",
        type=str,
        default=None,
        help="Fallback prediction file when a bucket lookup is missing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Where to write the combined prediction JSON file.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="talkpl-ai/TalkPlayData-Challenge-Dataset",
        help="HF dataset name for specificity lookup and evaluation.",
    )
    args = parser.parse_args()

    hh_predictions = load_predictions(args.hh)
    lh_predictions = load_predictions(args.lh)
    hl_predictions = load_predictions(args.hl)
    ll_predictions = load_predictions(args.ll)
    default_predictions = load_predictions(args.default) if args.default else hh_predictions

    sources = {
        "HH": index_predictions(hh_predictions),
        "LH": index_predictions(lh_predictions),
        "HL": index_predictions(hl_predictions),
        "LL": index_predictions(ll_predictions),
    }
    default_index = index_predictions(default_predictions)

    specificity_map = build_specificity_map(args.dataset_name)
    all_keys = sorted(default_index.keys())
    combined: list[dict] = []
    fallback_counts = {"HH": 0, "LH": 0, "HL": 0, "LL": 0, "DEFAULT": 0}

    for key in all_keys:
        bucket = specificity_map.get(key, "UNKNOWN")
        source = sources.get(bucket)
        row = source.get(key) if source else None
        if row is None:
            row = default_index.get(key)
            fallback_counts["DEFAULT"] += 1
        else:
            fallback_counts[bucket] += 1
        if row is None:
            continue
        combined.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote combined predictions to {output_path}")
    print(f"Bucket counts: {fallback_counts}")

    summary = evaluate_predictions(combined, args.dataset_name)
    print_report(summary)


if __name__ == "__main__":
    main()
