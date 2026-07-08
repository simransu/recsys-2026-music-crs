"""Analyze where retrieval fails — show what the user asked, what the correct track was, and why it was missed."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--track_metadata", type=str, default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--show_hits", action="store_true", help="Also show successful retrievals")
    parser.add_argument("--top_n", type=int, default=None, help="Only show first N failures")
    args = parser.parse_args()

    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    pred_map = {}
    for row in predictions:
        key = (row["session_id"], int(row["turn_number"]))
        pred_map[key] = row.get("predicted_track_ids", [])

    dataset = load_dataset(args.dataset_name, split="test")
    track_meta_ds = load_dataset(args.track_metadata)
    all_splits = [track_meta_ds[s] for s in track_meta_ds.keys()]
    track_meta = {}
    for split in all_splits:
        for row in split:
            track_meta[row["track_id"]] = row

    all_track_ids = set(track_meta.keys())

    hits = 0
    misses = 0
    miss_details = []
    hit_details = []

    cold_item_misses = 0
    sparse_metadata_misses = 0
    total_rank_when_hit = []

    for item in dataset:
        session_id = item["session_id"]
        conversation_goal = item.get("conversation_goal") or {}
        specificity = str(conversation_goal.get("specificity", "")).strip().upper()
        category = conversation_goal.get("category", "")
        listener_goal = conversation_goal.get("listener_goal", "")

        conversations = item["conversations"]
        user_messages = {}
        context_before = {}
        for turn in conversations:
            tn = int(turn["turn_number"])
            if turn["role"] == "user":
                user_messages[tn] = turn["content"]
            if turn["role"] == "music":
                target_track_id = str(turn["content"])
                key = (session_id, tn)
                predicted = pred_map.get(key, [])
                if not predicted:
                    continue

                prev_turns = [t for t in conversations if int(t["turn_number"]) < tn]
                context_lines = []
                for pt in prev_turns[-4:]:
                    role = pt["role"]
                    content = pt["content"]
                    if role == "music":
                        meta = track_meta.get(content, {})
                        content = f"{meta.get('track_name', '?')} by {meta.get('artist_name', '?')}"
                        role = "rec"
                    context_lines.append(f"  {role}: {str(content)[:120]}")

                target_meta = track_meta.get(target_track_id, {})
                target_name = target_meta.get("track_name", "UNKNOWN")
                target_artist = target_meta.get("artist_name", "UNKNOWN")
                target_album = target_meta.get("album_name", "")
                target_tags = target_meta.get("tag_list", "")
                if isinstance(target_tags, list):
                    target_tags = ", ".join(target_tags)
                target_popularity = target_meta.get("popularity", "")
                is_cold = target_track_id not in all_track_ids

                tag_count = len([t for t in str(target_tags).split(",") if t.strip()]) if target_tags else 0
                has_sparse_meta = tag_count <= 2 and not target_album

                rank = None
                if target_track_id in predicted:
                    rank = predicted.index(target_track_id) + 1

                detail = {
                    "session_id": session_id[:8],
                    "turn": tn,
                    "specificity": specificity,
                    "category": str(category)[:20] if category else "",
                    "listener_goal": str(listener_goal)[:100] if listener_goal else "",
                    "user_query": str(user_messages.get(tn, user_messages.get(tn - 1, "?")))[:150],
                    "target": f"{target_name} by {target_artist}",
                    "target_album": str(target_album)[:50],
                    "target_tags": str(target_tags)[:80],
                    "target_popularity": target_popularity,
                    "is_cold_item": is_cold,
                    "sparse_metadata": has_sparse_meta,
                    "rank": rank,
                    "num_predicted": len(predicted),
                    "context": "\n".join(context_lines),
                }

                if rank is not None:
                    hits += 1
                    total_rank_when_hit.append(rank)
                    hit_details.append(detail)
                else:
                    misses += 1
                    miss_details.append(detail)
                    if is_cold:
                        cold_item_misses += 1
                    if has_sparse_meta:
                        sparse_metadata_misses += 1

    total = hits + misses
    print(f"=== Recall Analysis ===")
    print(f"Total evaluated: {total}")
    print(f"Hits (target in predicted): {hits} ({100*hits/max(total,1):.1f}%)")
    print(f"Misses: {misses} ({100*misses/max(total,1):.1f}%)")
    print(f"Cold item misses: {cold_item_misses}")
    print(f"Sparse metadata misses: {sparse_metadata_misses}")
    if total_rank_when_hit:
        avg_rank = sum(total_rank_when_hit) / len(total_rank_when_hit)
        print(f"Avg rank when hit: {avg_rank:.1f}")
        print(f"Hits in top-20: {sum(1 for r in total_rank_when_hit if r <= 20)}")
        print(f"Hits in top-100: {sum(1 for r in total_rank_when_hit if r <= 100)}")

    miss_by_spec = defaultdict(int)
    for d in miss_details:
        miss_by_spec[d["specificity"]] += 1
    hit_by_spec = defaultdict(int)
    for d in hit_details:
        hit_by_spec[d["specificity"]] += 1
    print(f"\n=== By Specificity ===")
    for spec in ["HH", "HL", "LH", "LL"]:
        h = hit_by_spec.get(spec, 0)
        m = miss_by_spec.get(spec, 0)
        t = h + m
        print(f"[{spec}] hits={h} misses={m} recall={100*h/max(t,1):.1f}%")

    show_count = args.top_n or len(miss_details)
    print(f"\n=== Failed Retrievals (showing {min(show_count, len(miss_details))}/{len(miss_details)}) ===")
    for i, d in enumerate(miss_details[:show_count]):
        print(f"\n--- Miss #{i+1} [{d['specificity']}] session={d['session_id']} turn={d['turn']} ---")
        print(f"User query: {d['user_query']}")
        print(f"Target: {d['target']}")
        print(f"Album: {d['target_album']} | Tags: {d['target_tags']}")
        print(f"Popularity: {d['target_popularity']} | Cold: {d['is_cold_item']} | Sparse: {d['sparse_metadata']}")
        if d['listener_goal']:
            print(f"Listener goal: {d['listener_goal']}")
        if d['context']:
            print(f"Context:\n{d['context']}")

    if args.show_hits:
        print(f"\n=== Successful Retrievals (showing {min(show_count, len(hit_details))}/{len(hit_details)}) ===")
        for i, d in enumerate(hit_details[:show_count]):
            print(f"\n--- Hit #{i+1} [{d['specificity']}] rank={d['rank']} session={d['session_id']} turn={d['turn']} ---")
            print(f"User query: {d['user_query']}")
            print(f"Target: {d['target']}")
            print(f"Tags: {d['target_tags']}")


if __name__ == "__main__":
    main()
