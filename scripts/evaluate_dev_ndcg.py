"""Lightweight dev-set nDCG evaluator for Music CRS predictions.

This script mirrors the ranking portion of the official evaluator closely enough
for local hypothesis testing. It ignores generation and diversity metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def ndcg_at_k(predicted: list[str], target: str, k: int) -> float:
    if not predicted or not target:
        return 0.0
    for idx, track_id in enumerate(predicted[:k]):
        if track_id == target:
            return 1.0 / math.log2(idx + 2)
    return 0.0


def recall_at_k(predicted: list[str], target: str, k: int) -> float:
    if not predicted or not target:
        return 0.0
    return 1.0 if target in predicted[:k] else 0.0


def build_ground_truth(dataset_name: str) -> dict[tuple[str, int], str]:
    dataset = load_dataset(dataset_name, split="test")
    ground_truth: dict[tuple[str, int], str] = {}
    for item in dataset:
        session_id = item["session_id"]
        for turn in item["conversations"]:
            if turn.get("role") == "music":
                ground_truth[(session_id, int(turn["turn_number"]))] = str(turn["content"])
    return ground_truth


def build_specificity_map(dataset_name: str) -> dict[tuple[str, int], str]:
    dataset = load_dataset(dataset_name, split="test")
    specificity_map: dict[tuple[str, int], str] = {}
    for item in dataset:
        session_id = item["session_id"]
        conversation_goal = item.get("conversation_goal") or {}
        specificity = (
            str(conversation_goal.get("specificity") or item.get("goal_specificity") or "")
            .strip()
            .upper()
        )
        for turn in item["conversations"]:
            if turn.get("role") == "music":
                specificity_map[(session_id, int(turn["turn_number"]))] = specificity
    return specificity_map


def summarize(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def evaluate_predictions(predictions: list[dict], dataset_name: str) -> dict[str, object]:
    ground_truth = build_ground_truth(dataset_name)
    specificity_map = build_specificity_map(dataset_name)

    per_turn = defaultdict(list)
    per_turn_recall = defaultdict(list)
    per_specificity: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_specificity_recall: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    pool_depths: list[int] = []
    missing = 0
    valid_specificities = {"HH", "HL", "LH", "LL"}
    for row in predictions:
        key = (row["session_id"], int(row["turn_number"]))
        target = ground_truth.get(key)
        if target is None:
            missing += 1
            continue
        specificity = specificity_map.get(key, "") or "UNKNOWN"
        predicted = row.get("predicted_track_ids", [])
        for k in [1, 10, 20]:
            score = ndcg_at_k(predicted, target, k)
            per_turn[k].append(score)
            per_specificity[specificity][k].append(score)
        for rk in [20, 100, 200, 500, 1000]:
            rs = recall_at_k(predicted, target, rk)
            per_turn_recall[rk].append(rs)
            per_specificity_recall[specificity][rk].append(rs)
        pool_depths.append(len(predicted))

    specificity_summary: dict[str, object] = {}
    for bucket in ["HH", "LH", "HL", "LL", "UNKNOWN"]:
        values_by_k = per_specificity.get(bucket, {})
        count = len(values_by_k.get(20, []))
        if bucket not in valid_specificities and count == 0:
            continue
        specificity_summary[bucket] = {
            "samples_scored": count,
            "nDCG@1": summarize(values_by_k.get(1, [])),
            "nDCG@10": summarize(values_by_k.get(10, [])),
            "nDCG@20": summarize(values_by_k.get(20, [])),
        }

    return {
        "samples_scored": len(per_turn[20]),
        "samples_missing_ground_truth": missing,
        "overall": {
            "nDCG@1": summarize(per_turn[1]),
            "nDCG@10": summarize(per_turn[10]),
            "nDCG@20": summarize(per_turn[20]),
            "recall@20": summarize(per_turn_recall[20]),
            "recall@100": summarize(per_turn_recall[100]),
            "recall@200": summarize(per_turn_recall[200]),
            "recall@500": summarize(per_turn_recall[500]),
            "recall@1000": summarize(per_turn_recall[1000]),
        },
        "pool_depth": {
            "mean": sum(pool_depths) / max(len(pool_depths), 1),
            "min": min(pool_depths) if pool_depths else 0,
            "max": max(pool_depths) if pool_depths else 0,
        },
        "by_specificity": specificity_summary,
        "by_specificity_recall": {
            bucket: {
                "samples_scored": len(values_by_k.get(100, [])),
                "recall@20": summarize(values_by_k.get(20, [])),
                "recall@100": summarize(values_by_k.get(100, [])),
                "recall@200": summarize(values_by_k.get(200, [])),
            }
            for bucket, values_by_k in per_specificity_recall.items()
            if bucket in valid_specificities or len(values_by_k.get(100, [])) > 0
        },
    }


def print_report(summary: dict[str, object]) -> None:
    print("=== Dev nDCG ===")
    print(f"samples_scored: {summary['samples_scored']}")
    print(f"samples_missing_ground_truth: {summary['samples_missing_ground_truth']}")
    overall = summary["overall"]
    for k in [1, 10, 20]:
        print(f"nDCG@{k}: {overall[f'nDCG@{k}']:.6f}")
    for rk in [20, 100, 200, 500, 1000]:
        print(f"recall@{rk}: {overall[f'recall@{rk}']:.6f}")
    pd = summary.get("pool_depth", {})
    print(f"pool_depth (candidates per query): mean={pd.get('mean', 0):.1f} "
          f"min={pd.get('min', 0)} max={pd.get('max', 0)}")

    print("=== Dev nDCG by specificity ===")
    by_specificity = summary["by_specificity"]
    for bucket in ["HH", "LH", "HL", "LL", "UNKNOWN"]:
        bucket_summary = by_specificity.get(bucket)
        if not bucket_summary:
            continue
        print(f"[{bucket}] samples_scored: {bucket_summary['samples_scored']}")
        for k in [1, 10, 20]:
            print(f"[{bucket}] nDCG@{k}: {bucket_summary[f'nDCG@{k}']:.6f}")

    print("=== Dev recall@100 by specificity ===")
    by_specificity_recall = summary["by_specificity_recall"]
    for bucket in ["HH", "LH", "HL", "LL", "UNKNOWN"]:
        bucket_summary = by_specificity_recall.get(bucket)
        if not bucket_summary:
            continue
        print(f"[{bucket}] samples_scored: {bucket_summary['samples_scored']}")
        for rk in [20, 100, 200]:
            print(f"[{bucket}] recall@{rk}: {bucket_summary[f'recall@{rk}']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dev-set nDCG for Music CRS predictions.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to prediction JSON file.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="talkpl-ai/TalkPlayData-Challenge-Dataset",
        help="HF dataset name for the dev split.",
    )
    args = parser.parse_args()

    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    print_report(evaluate_predictions(predictions, args.dataset_name))


if __name__ == "__main__":
    main()
