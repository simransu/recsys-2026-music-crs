"""Run multiple dev inference jobs and evaluate each one in one shot.

This is intended for ranking-only experiments. It launches dev inference for a
list of TIDs, then computes overall and specificity-bucket nDCG for each run.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from evaluate_dev_ndcg import evaluate_predictions, print_report


def run_inference(tid: str, batch_size: int, retrieval_batch_size: int, with_generation: bool) -> None:
    cmd = [
        sys.executable,
        "run_inference_devset.py",
        "--tid",
        tid,
        "--batch_size",
        str(batch_size),
        "--retrieval_batch_size",
        str(retrieval_batch_size),
    ]
    if not with_generation:
        cmd.append("--skip_generation")
    subprocess.run(cmd, check=True)


def load_predictions(tid: str) -> list[dict]:
    pred_path = Path("exp/inference/devset") / f"{tid}.json"
    return json.loads(pred_path.read_text(encoding="utf-8"))


def flatten_summary(tid: str, summary: dict[str, object]) -> dict[str, object]:
    overall = summary["overall"]
    by_specificity = summary["by_specificity"]
    by_specificity_recall = summary["by_specificity_recall"]
    row: dict[str, object] = {
        "tid": tid,
        "samples_scored": summary["samples_scored"],
        "samples_missing_ground_truth": summary["samples_missing_ground_truth"],
        "ndcg@1": overall["nDCG@1"],
        "ndcg@10": overall["nDCG@10"],
        "ndcg@20": overall["nDCG@20"],
        "recall@100": overall["recall@100"],
    }
    for bucket in ["HH", "LH", "HL", "LL"]:
        bucket_summary = by_specificity.get(bucket, {})
        bucket_recall = by_specificity_recall.get(bucket, {})
        row[f"{bucket.lower()}_samples"] = bucket_summary.get("samples_scored", 0)
        row[f"{bucket.lower()}_ndcg@1"] = bucket_summary.get("nDCG@1", 0.0)
        row[f"{bucket.lower()}_ndcg@10"] = bucket_summary.get("nDCG@10", 0.0)
        row[f"{bucket.lower()}_ndcg@20"] = bucket_summary.get("nDCG@20", 0.0)
        row[f"{bucket.lower()}_recall@100"] = bucket_recall.get("recall@100", 0.0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple dev inference jobs and evaluate them.")
    parser.add_argument("--tids", nargs="+", required=True, help="One or more config TIDs to run.")
    parser.add_argument("--batch_size", type=int, default=1, help="Generation batch size.")
    parser.add_argument(
        "--retrieval_batch_size",
        type=int,
        default=1,
        help="Retrieval batch size for the dev inference loop.",
    )
    parser.add_argument(
        "--with_generation",
        action="store_true",
        help="Run generation too. Default is ranking-only with skipped generation.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="exp/inference/devset/dev_eval_suite_summary.csv",
        help="Where to write the combined summary CSV.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="talkpl-ai/TalkPlayData-Challenge-Dataset",
        help="HF dataset name for dev evaluation.",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for tid in args.tids:
        print(f"=== Running inference for {tid} ===")
        run_inference(tid, args.batch_size, args.retrieval_batch_size, args.with_generation)
        print(f"=== Evaluating {tid} ===")
        predictions = load_predictions(tid)
        summary = evaluate_predictions(predictions, args.dataset_name)
        print_report(summary)
        rows.append(flatten_summary(tid, summary))

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tid",
        "samples_scored",
        "samples_missing_ground_truth",
        "ndcg@1",
        "ndcg@10",
        "ndcg@20",
        "recall@100",
        "hh_samples",
        "hh_ndcg@1",
        "hh_ndcg@10",
        "hh_ndcg@20",
        "hh_recall@100",
        "lh_samples",
        "lh_ndcg@1",
        "lh_ndcg@10",
        "lh_ndcg@20",
        "lh_recall@100",
        "hl_samples",
        "hl_ndcg@1",
        "hl_ndcg@10",
        "hl_ndcg@20",
        "hl_recall@100",
        "ll_samples",
        "ll_ndcg@1",
        "ll_ndcg@10",
        "ll_ndcg@20",
        "ll_recall@100",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
