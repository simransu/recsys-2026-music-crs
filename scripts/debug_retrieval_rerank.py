#!/usr/bin/env python3
"""Debug retrieval and reranking candidate flow for Music CRS.

This script is meant to run on the pod when retrieval or reranking appears to
return empty candidate sets. It prints:

- loaded corpus sizes
- retrieval backend sizes
- a small sample of queries built from the evaluation dataset
- retrieval outputs before reranking
- reranker outputs after reranking

Use `--sample_index` to inspect one specific sample, or `--max_samples` to scan
the first N constructed samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from omegaconf import OmegaConf

from mcrs import load_crs_baseline, load_crs_two_tower


def build_retrieval_input(music_crs: Any, session_memory: list[dict[str, Any]], user_id: str | None, user_profile: dict[str, Any] | None, conversation_goal: dict[str, Any] | None) -> str:
    return music_crs._build_retrieval_input(
        session_memory,
        user_id=user_id,
        user_profile=user_profile,
        conversation_goal=conversation_goal,
    )


def build_turn_sample(item: dict[str, Any], music_crs: Any, turn_number: int) -> dict[str, Any]:
    conversations = item["conversations"]
    session_memory: list[dict[str, Any]] = []
    for turn in conversations:
        if turn["turn_number"] >= turn_number:
            continue
        role = turn["role"]
        content = turn["content"]
        if role == "music":
            role = "assistant"
            content = music_crs.item_db.id_to_metadata(content)
        session_memory.append({"role": role, "content": content})

    current_turn = next(turn for turn in conversations if turn["turn_number"] == turn_number)
    return {
        "session_id": item.get("session_id"),
        "turn_number": turn_number,
        "user_id": item.get("user_id"),
        "user_profile": item.get("user_profile"),
        "conversation_goal": item.get("conversation_goal"),
        "user_query": current_turn["content"],
        "session_memory": session_memory,
    }


def build_last_turn_sample(item: dict[str, Any], music_crs: Any) -> dict[str, Any]:
    conversations = item["conversations"]
    session_memory = []
    for turn in conversations[:-1]:
        role = turn["role"]
        content = turn["content"]
        if role == "music":
            role = "assistant"
            content = music_crs.item_db.id_to_metadata(content)
        session_memory.append({"role": role, "content": content})

    last_turn = conversations[-1]
    return {
        "session_id": item.get("session_id"),
        "turn_number": last_turn.get("turn_number"),
        "user_id": item.get("user_id"),
        "user_profile": item.get("user_profile"),
        "conversation_goal": item.get("conversation_goal"),
        "user_query": last_turn["content"],
        "session_memory": session_memory,
    }


def format_preview(text: str, limit: int = 600) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"


def candidate_origin(track_id: str, bm25_ids: set[str], bert_ids: set[str]) -> str:
    in_bm25 = track_id in bm25_ids
    in_bert = track_id in bert_ids
    if in_bm25 and in_bert:
        return "bm25+bert"
    if in_bm25:
        return "bm25"
    if in_bert:
        return "bert"
    return "unknown"


def summarize_rerank_movement(retrieval_items: list[str], ranked_items: list[str], window: int = 20) -> dict[str, Any]:
    retrieval_window = retrieval_items[:window]
    ranked_window = ranked_items[:window]
    retrieval_set = set(retrieval_window)
    ranked_set = set(ranked_window)

    return {
        "top1_same": bool(retrieval_items and ranked_items and retrieval_items[0] == ranked_items[0]),
        "top5_exact_match": retrieval_items[:5] == ranked_items[:5],
        "top20_overlap": len(retrieval_set & ranked_set),
        "top20_jaccard": (len(retrieval_set & ranked_set) / len(retrieval_set | ranked_set)) if (retrieval_set | ranked_set) else 0.0,
        "pos_changed_top20": sum(1 for idx, item in enumerate(retrieval_window) if idx >= len(ranked_window) or ranked_window[idx] != item),
        "retrieval_top1": retrieval_items[0] if retrieval_items else "",
        "ranked_top1": ranked_items[0] if ranked_items else "",
    }


def summarize_source_counts(items: list[str], bm25_ids: set[str], bert_ids: set[str], window: int = 20) -> dict[str, int]:
    counts = {"bm25": 0, "bert": 0, "bm25+bert": 0, "unknown": 0}
    for track_id in items[:window]:
        counts[candidate_origin(track_id, bm25_ids, bert_ids)] += 1
    return counts


def log_stage(message: str) -> None:
    print(f"[debug] {message}", flush=True)


def print_backend_summary(music_crs: Any) -> None:
    print("=== Loaded CRS State ===")
    print(f"item_db size: {len(music_crs.item_db.metadata_dict)}")
    print(f"user_db size: {len(music_crs.user_db.user_profiles)}")
    print(f"retrieval type: {music_crs.retrieval_type}")
    print(f"reranker type: {music_crs.reranker_type}")

    if music_crs.retrieval is None:
        print("retrieval backend: not loaded")
    else:
        print(f"retrieval backend class: {music_crs.retrieval.__class__.__name__}")
        if hasattr(music_crs.retrieval, "track_ids"):
            print(f"retrieval track_ids: {len(music_crs.retrieval.track_ids)}")
        if hasattr(music_crs.retrieval, "embeddings"):
            embeddings = music_crs.retrieval.embeddings
            print(f"retrieval embeddings: {tuple(embeddings.shape)}")
        if hasattr(music_crs.retrieval, "bm25"):
            bm25 = music_crs.retrieval.bm25
            print(f"hybrid.bm25 track_ids: {len(getattr(bm25, 'track_ids', []))}")
        if hasattr(music_crs.retrieval, "bert"):
            bert = music_crs.retrieval.bert
            print(f"hybrid.bert track_ids: {len(getattr(bert, 'track_ids', []))}")
            if hasattr(bert, "embeddings"):
                print(f"hybrid.bert embeddings: {tuple(bert.embeddings.shape)}")

    if music_crs.reranker is None:
        print("reranker backend: not loaded")
    else:
        print(f"reranker backend class: {music_crs.reranker.__class__.__name__}")
        if hasattr(music_crs.reranker, "user_embeddings"):
            print(f"reranker user embeddings: {len(music_crs.reranker.user_embeddings)}")
        if hasattr(music_crs.reranker, "track_embeddings"):
            print(f"reranker track embeddings: {len(music_crs.reranker.track_embeddings)}")
        if hasattr(music_crs.reranker, "scorer"):
            print(f"reranker scorer loaded: {music_crs.reranker.scorer is not None}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug retrieval and reranking candidate flow.")
    parser.add_argument("--tid", type=str, required=True, help="Config task id, for example qwen3_8b_two_tower_devset")
    parser.add_argument("--sample_index", type=int, default=None, help="Inspect a single constructed sample by index.")
    parser.add_argument("--max_samples", type=int, default=12, help="Inspect at most this many constructed samples.")
    parser.add_argument("--batch_size", type=int, default=8, help="How many samples to send through retrieval at once.")
    parser.add_argument("--all_turns", action="store_true", help="For devset-style data, inspect all turns instead of only the final turn.")
    parser.add_argument("--show_query", action="store_true", help="Print the full retrieval query text.")
    parser.add_argument("--retrieval_only", action="store_true", help="Skip reranker loading and only inspect retrieval outputs.")
    parser.add_argument("--dump_json", type=str, default=None, help="Optional path to write per-sample debug output as JSON.")
    args = parser.parse_args()

    log_stage(f"loading config {args.tid}")
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    loader = load_crs_two_tower if config.get("reranker_type", None) == "two_tower" else load_crs_baseline
    log_stage("building CRS object")
    music_crs = loader(
        lm_type=config.lm_type,
        retrieval_type=config.retrieval_type,
        item_db_name=config.item_db_name,
        user_db_name=config.user_db_name,
        track_split_types=config.track_split_types,
        user_split_types=config.user_split_types,
        corpus_types=config.corpus_types,
        cache_dir=config.cache_dir,
        device=config.device,
        retrieval_device=config.get("retrieval_device", None),
        attn_implementation=config.attn_implementation,
        dtype=torch.bfloat16,
        reranker_type=config.get("reranker_type", None),
        reranker_embedding_type=config.get("reranker_embedding_type", "cf-bpr"),
        reranker_checkpoint_path=config.get("reranker_checkpoint_path", "./cache/two_tower_reranker.pt"),
        reranker_device=config.get("reranker_device", "cpu"),
        reranker_projection_dim=config.get("reranker_projection_dim", 256),
        reranker_tower_hidden_dim=config.get("reranker_tower_hidden_dim", 512),
        reranker_dropout=config.get("reranker_dropout", 0.1),
        reranker_temperature=config.get("reranker_temperature", 0.07),
        reranker_alpha=config.get("reranker_alpha", 1.0),
        reranker_beta=config.get("reranker_beta", 0.15),
        reranker_rrf_k=config.get("reranker_rrf_k", 60),
        retrieval_topk=config.get("retrieval_topk", 100),
        rerank_topk=config.get("rerank_topk", 20),
        retrieval_bm25_topk=config.get("retrieval_bm25_topk", 100),
        retrieval_bert_topk=config.get("retrieval_bert_topk", 100),
        retrieval_final_topk=config.get("retrieval_final_topk", 20),
        retrieval_rrf_k=config.get("retrieval_rrf_k", 60),
        retrieval_bm25_weight=config.get("retrieval_bm25_weight", 0.8),
        retrieval_bert_weight=config.get("retrieval_bert_weight", 0.2),
        bm25_field_weights=config.get("bm25_field_weights", None),
        enable_query_rewrite=config.get("enable_query_rewrite", True),
        enable_specificity_routing=config.get("enable_specificity_routing", True),
        enable_user_to_item=config.get("enable_user_to_item", True),
        dense_model_name=config.get("dense_model_name", "bert-base-uncased"),
        dense_query_prefix=config.get("dense_query_prefix", ""),
        dense_doc_prefix=config.get("dense_doc_prefix", ""),
        load_lm=False,
        load_retrieval=False,
        load_reranker=False,
    )

    print_backend_summary(music_crs)

    log_stage(f"loading test dataset {config.test_dataset_name}")
    dataset = load_dataset(config.test_dataset_name, split="test")
    samples: list[dict[str, Any]] = []
    for item in dataset:
        if "conversations" not in item:
            continue
        if args.all_turns:
            for turn_number in range(1, len(item["conversations"]) + 1):
                samples.append(build_turn_sample(item, music_crs, turn_number))
        else:
            samples.append(build_last_turn_sample(item, music_crs))
        if args.sample_index is None and len(samples) >= args.max_samples:
            break

    if not samples:
        print("No samples could be built from the test dataset.")
        return

    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(samples):
            raise IndexError(f"sample_index {args.sample_index} is out of range for {len(samples)} samples")
        samples = [samples[args.sample_index]]
    else:
        samples = samples[: args.max_samples]

    print(f"=== Inspecting {len(samples)} sample(s) ===")

    log_stage("loading retrieval backend")
    music_crs.load_retrieval()
    if config.get("reranker_type", None) and not args.retrieval_only:
        log_stage("loading reranker backend")
        music_crs.load_reranker()

    debug_rows: list[dict[str, Any]] = []
    retrieval_empty = 0
    rerank_empty = 0
    top1_same_count = 0
    top5_exact_match_count = 0
    total_top20_overlap = 0
    total_top20_jaccard = 0.0
    total_pos_changed_top20 = 0
    total_retrieval_source_counts = {"bm25": 0, "bert": 0, "bm25+bert": 0, "unknown": 0}
    total_ranked_source_counts = {"bm25": 0, "bert": 0, "bm25+bert": 0, "unknown": 0}

    try:
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            retrieved = music_crs.batch_retrieval(
                [
                    {
                        "user_query": sample["user_query"],
                        "user_id": sample["user_id"],
                        "session_memory": sample["session_memory"],
                        "user_profile": sample["user_profile"],
                        "conversation_goal": sample["conversation_goal"],
                    }
                    for sample in batch
                ]
            )

            if config.get("reranker_type", None) and not args.retrieval_only:
                reranked = music_crs.batch_rerank(
                    retrieved["user_ids"],
                    retrieved["retrieval_items"],
                    user_profiles=retrieved.get("user_profiles"),
                    query_texts=retrieved.get("retrieval_inputs"),
                    conversation_goals=retrieved.get("conversation_goals"),
                )
            else:
                reranked = {"ranked_items": retrieved["retrieval_items"], "recommend_items": []}

            for idx, sample in enumerate(batch):
                query = retrieved["retrieval_inputs"][idx]
                retrieval_items = retrieved["retrieval_items"][idx]
                ranked_items = reranked["ranked_items"][idx]
                bm25_items: list[str] = []
                bert_items: list[str] = []
                if hasattr(music_crs.retrieval, "bm25"):
                    bm25_items = music_crs.retrieval.bm25.text_to_item_retrieval(query, topk=music_crs.retrieval.bm25_topk)
                if hasattr(music_crs.retrieval, "bert"):
                    bert_items = music_crs.retrieval.bert.text_to_item_retrieval(query, topk=music_crs.retrieval.bert_topk)
                bm25_set = set(bm25_items)
                bert_set = set(bert_items)
                retrieval_is_empty = not retrieval_items
                rerank_is_empty = not ranked_items
                retrieval_empty += int(retrieval_is_empty)
                rerank_empty += int(rerank_is_empty)
                movement = summarize_rerank_movement(retrieval_items, ranked_items, window=20)
                top1_same_count += int(movement["top1_same"])
                top5_exact_match_count += int(movement["top5_exact_match"])
                total_top20_overlap += int(movement["top20_overlap"])
                total_top20_jaccard += float(movement["top20_jaccard"])
                total_pos_changed_top20 += int(movement["pos_changed_top20"])
                retrieval_source_counts = summarize_source_counts(retrieval_items, bm25_set, bert_set, window=20)
                ranked_source_counts = summarize_source_counts(ranked_items, bm25_set, bert_set, window=20)
                for key in total_retrieval_source_counts:
                    total_retrieval_source_counts[key] += retrieval_source_counts[key]
                    total_ranked_source_counts[key] += ranked_source_counts[key]

                row = {
                    "session_id": sample["session_id"],
                    "turn_number": sample["turn_number"],
                    "user_id": sample["user_id"],
                    "query_len": len(query),
                    "query_non_ws_len": len(query.strip()),
                    "retrieval_count": len(retrieval_items),
                    "ranked_count": len(ranked_items),
                    "retrieval_top5": retrieval_items[:5],
                    "ranked_top5": ranked_items[:5],
                    "movement": movement,
                    "retrieval_source_counts": retrieval_source_counts,
                    "ranked_source_counts": ranked_source_counts,
                }
                debug_rows.append(row)

                print("\n--- Sample ---")
                print(f"session_id: {sample['session_id']}")
                print(f"turn_number: {sample['turn_number']}")
                print(f"user_id: {sample['user_id']}")
                print(f"query_len: {row['query_len']} non_ws={row['query_non_ws_len']}")
                if args.show_query:
                    print("query:")
                    print(query)
                else:
                    print("query preview:")
                    print(format_preview(query))
                if bm25_items or bert_items:
                    print(
                        "hybrid source counts: "
                        f"bm25={len(bm25_items)} bert={len(bert_items)} "
                        f"overlap={len(bm25_set & bert_set)} union={len(bm25_set | bert_set)}"
                    )
                print(f"retrieval_count: {row['retrieval_count']}")
                print(f"retrieval_top5: {row['retrieval_top5']}")
                print(f"ranked_count: {row['ranked_count']}")
                print(f"ranked_top5: {row['ranked_top5']}")
                print(
                    "movement: "
                    f"top1_same={movement['top1_same']} "
                    f"top5_exact_match={movement['top5_exact_match']} "
                    f"top20_overlap={movement['top20_overlap']}/20 "
                    f"top20_jaccard={movement['top20_jaccard']:.3f} "
                    f"pos_changed_top20={movement['pos_changed_top20']}"
                )
                print(
                    "source_counts top20: "
                    f"retrieval={retrieval_source_counts} "
                    f"ranked={ranked_source_counts}"
                )
                if bm25_items:
                    print("bm25_top5:")
                    for rank, track_id in enumerate(bm25_items[:5], start=1):
                        print(f"  {rank}. {track_id}")
                if bert_items:
                    print("bert_top5:")
                    for rank, track_id in enumerate(bert_items[:5], start=1):
                        print(f"  {rank}. {track_id}")
                if retrieval_items:
                    print("hybrid_final_top20 provenance:")
                    for rank, track_id in enumerate(retrieval_items[:20], start=1):
                        origin = candidate_origin(track_id, bm25_set, bert_set)
                        print(f"  {rank}. {track_id} [{origin}]")
                if ranked_items:
                    print("reranker_top20 provenance:")
                    for rank, track_id in enumerate(ranked_items[:20], start=1):
                        origin = candidate_origin(track_id, bm25_set, bert_set)
                        print(f"  {rank}. {track_id} [{origin}]")

                if retrieval_is_empty:
                    print("status: retrieval empty")
                if rerank_is_empty:
                    print("status: rerank empty")

        print("\n=== Summary ===")
        print(f"samples_inspected: {len(samples)}")
        print(f"retrieval_empty: {retrieval_empty}")
        print(f"rerank_empty: {rerank_empty}")
        if len(samples) > 0:
            print(f"top1_same_rate: {top1_same_count}/{len(samples)}")
            print(f"top5_exact_match_rate: {top5_exact_match_count}/{len(samples)}")
            print(f"avg_top20_overlap: {total_top20_overlap / len(samples):.2f}")
            print(f"avg_top20_jaccard: {total_top20_jaccard / len(samples):.3f}")
            print(f"avg_pos_changed_top20: {total_pos_changed_top20 / len(samples):.2f}")
            print(f"avg_retrieval_source_counts_top20: {total_retrieval_source_counts}")
            print(f"avg_ranked_source_counts_top20: {total_ranked_source_counts}")
        if args.retrieval_only:
            print("reranker: skipped via --retrieval_only")
        if retrieval_empty == len(samples):
            print("all inspected samples had empty retrieval output")
        if rerank_empty == len(samples):
            print("all inspected samples had empty rerank output")

    finally:
        music_crs.cleanup_retrieval()
        if config.get("reranker_type", None):
            music_crs.cleanup_reranker()

    if args.dump_json:
        dump_path = Path(args.dump_json)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(debug_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote: {dump_path}")


if __name__ == "__main__":
    main()
