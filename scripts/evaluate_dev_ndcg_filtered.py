"""Dev-set nDCG evaluator filtered by goal_progress_assessment.

Same as evaluate_dev_ndcg.py but only scores turns where the ground truth
track is labeled MOVES_TOWARD_GOAL. Use --mode to switch between:
  - moves   : only MOVES_TOWARD_GOAL turns
  - does_not: only DOES_NOT_MOVE_TOWARD_GOAL turns
  - all     : all turns (same as original script)
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


def build_goal_progress_map(dataset_name: str) -> dict[tuple[str, int], str]:
    """Map (session_id, turn_number) -> goal_progress_assessment for music turns.

    GPA at turn N+1 assesses the recommendation at turn N, so we look up
    assessments[turn_number + 1] for each music turn.
    """
    dataset = load_dataset(dataset_name, split="test")
    progress_map: dict[tuple[str, int], str] = {}
    for item in dataset:
        session_id = item["session_id"]
        assessments = {a["turn_number"]: a["goal_progress_assessment"] for a in item.get("goal_progress_assessments", [])}
        for turn in item["conversations"]:
            if turn.get("role") == "music":
                turn_number = int(turn["turn_number"])
                assessment = assessments.get(turn_number + 1)
                progress_map[(session_id, turn_number)] = assessment or "UNKNOWN"
    return progress_map


def summarize(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def evaluate_predictions(predictions: list[dict], dataset_name: str, mode: str) -> dict[str, object]:
    ground_truth = build_ground_truth(dataset_name)
    specificity_map = build_specificity_map(dataset_name)
    progress_map = build_goal_progress_map(dataset_name)

    per_turn = defaultdict(list)
    per_turn_recall = defaultdict(list)
    per_specificity: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_specificity_recall: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    missing = 0
    skipped = 0
    valid_specificities = {"HH", "HL", "LH", "LL"}

    for row in predictions:
        key = (row["session_id"], int(row["turn_number"]))
        target = ground_truth.get(key)
        if target is None:
            missing += 1
            continue

        progress = progress_map.get(key, "UNKNOWN")
        if mode == "moves" and progress != "MOVES_TOWARD_GOAL":
            skipped += 1
            continue
        if mode == "does_not" and progress != "DOES_NOT_MOVE_TOWARD_GOAL":
            skipped += 1
            continue

        specificity = specificity_map.get(key, "") or "UNKNOWN"
        predicted = row.get("predicted_track_ids", [])
        for k in [1, 10, 20]:
            score = ndcg_at_k(predicted, target, k)
            per_turn[k].append(score)
            per_specificity[specificity][k].append(score)
        recall_score = recall_at_k(predicted, target, 100)
        per_turn_recall[100].append(recall_score)
        per_specificity_recall[specificity][100].append(recall_score)

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
        "samples_skipped_by_filter": skipped,
        "samples_missing_ground_truth": missing,
        "overall": {
            "nDCG@1": summarize(per_turn[1]),
            "nDCG@10": summarize(per_turn[10]),
            "nDCG@20": summarize(per_turn[20]),
            "recall@100": summarize(per_turn_recall[100]),
        },
        "by_specificity": specificity_summary,
        "by_specificity_recall": {
            bucket: {
                "samples_scored": len(values_by_k.get(100, [])),
                "recall@100": summarize(values_by_k.get(100, [])),
            }
            for bucket, values_by_k in per_specificity_recall.items()
            if bucket in valid_specificities or len(values_by_k.get(100, [])) > 0
        },
    }


def print_report(summary: dict[str, object], mode: str) -> None:
    label = {"moves": "MOVES_TOWARD_GOAL only", "does_not": "DOES_NOT_MOVE_TOWARD_GOAL only", "all": "all turns"}[mode]
    print(f"=== Dev nDCG [{label}] ===")
    print(f"samples_scored: {summary['samples_scored']}")
    print(f"samples_skipped_by_filter: {summary['samples_skipped_by_filter']}")
    print(f"samples_missing_ground_truth: {summary['samples_missing_ground_truth']}")
    overall = summary["overall"]
    for k in [1, 10, 20]:
        print(f"nDCG@{k}: {overall[f'nDCG@{k}']:.6f}")
    print(f"recall@100: {overall['recall@100']:.6f}")

    print(f"=== Dev nDCG by specificity [{label}] ===")
    by_specificity = summary["by_specificity"]
    for bucket in ["HH", "LH", "HL", "LL", "UNKNOWN"]:
        bucket_summary = by_specificity.get(bucket)
        if not bucket_summary:
            continue
        print(f"[{bucket}] samples_scored: {bucket_summary['samples_scored']}")
        for k in [1, 10, 20]:
            print(f"[{bucket}] nDCG@{k}: {bucket_summary[f'nDCG@{k}']:.6f}")

    print(f"=== Dev recall@100 by specificity [{label}] ===")
    by_specificity_recall = summary["by_specificity_recall"]
    for bucket in ["HH", "LH", "HL", "LL", "UNKNOWN"]:
        bucket_summary = by_specificity_recall.get(bucket)
        if not bucket_summary:
            continue
        print(f"[{bucket}] samples_scored: {bucket_summary['samples_scored']}")
        print(f"[{bucket}] recall@100: {bucket_summary['recall@100']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filtered dev-set nDCG evaluator.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to prediction JSON file.")
    parser.add_argument("--dataset_name", type=str, default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--mode", type=str, default="moves", choices=["moves", "does_not", "all"],
                        help="Filter: moves=MOVES_TOWARD_GOAL only, does_not=DOES_NOT_MOVE_TOWARD_GOAL only, all=no filter")
    args = parser.parse_args()

    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    print_report(evaluate_predictions(predictions, args.dataset_name, args.mode), args.mode)


if __name__ == "__main__":
    main()
