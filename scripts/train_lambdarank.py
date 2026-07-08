"""Train LambdaRank model using LightGBM on train data.

Caches retrieval results per-source so adding a new source only requires
running retrieval for that source. Feature extraction is cheap (dict lookups)
and always re-derived from cached retrieval results.

Usage:
    python scripts/train_lambdarank.py --config config/lambdarank_training.yaml
    python scripts/train_lambdarank.py --config config/lambdarank_training.yaml --goal_filter --last_n_turns 1
"""

import os
import json
import argparse
import numpy as np
import torch
import lightgbm as lgb
from datasets import load_dataset
from tqdm import tqdm
from omegaconf import OmegaConf
from mcrs import load_crs_baseline


SOURCE_NAMES = [
    "primary", "bm25", "bert", "bpr", "i2i",
    "i2i_image-siglip2", "i2i_cf-bpr", "i2i_audio-laion_clap",
    "i2i_attributes-qwen3_embedding_0.6b", "i2i_lyrics-qwen3_embedding_0.6b",
    "i2i_metadata-qwen3_embedding_0.6b",
    "train_thought", "cooccur", "qwen3_dense",
    "artist", "album", "entity",
]


def get_retrieval_cache_dir(cache_dir, max_sessions, last_n_turns, goal_filter):
    turns_tag = f"_last{last_n_turns}" if last_n_turns else ""
    goal_tag = "_goalonly" if goal_filter else ""
    tag = f"lambdarank_retrieval_{max_sessions or 'all'}{turns_tag}{goal_tag}"
    return os.path.join(cache_dir, tag)


def save_source_cache(retrieval_cache_dir, source_name, data):
    os.makedirs(retrieval_cache_dir, exist_ok=True)
    path = os.path.join(retrieval_cache_dir, f"{source_name}.json")
    with open(path, "w") as f:
        json.dump(data, f)


def load_source_cache(retrieval_cache_dir, source_name):
    path = os.path.join(retrieval_cache_dir, f"{source_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_query_metadata(retrieval_cache_dir, metadata):
    os.makedirs(retrieval_cache_dir, exist_ok=True)
    path = os.path.join(retrieval_cache_dir, "query_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f)


def load_query_metadata(retrieval_cache_dir):
    path = os.path.join(retrieval_cache_dir, "query_metadata.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main(args):
    config = OmegaConf.load(args.config)
    max_sessions = args.max_sessions or None
    retrieval_cache_dir = get_retrieval_cache_dir(
        config.cache_dir, max_sessions, args.last_n_turns, args.goal_filter,
    )

    # Check which sources already have cached retrieval results
    cached_metadata = load_query_metadata(retrieval_cache_dir)
    cached_sources = {}
    if cached_metadata and not args.no_cache:
        for source_name in SOURCE_NAMES:
            data = load_source_cache(retrieval_cache_dir, source_name)
            if data is not None:
                cached_sources[source_name] = data
        if cached_sources:
            print(f"Loaded cached retrieval for {len(cached_sources)} sources: {list(cached_sources.keys())}")

    # Determine which sources need retrieval
    sources_needed = [s for s in SOURCE_NAMES if s not in cached_sources]

    if sources_needed or not cached_metadata:
        print("Loading CRS model (retrieval only)...")
        music_crs = load_crs_baseline(
            lm_type=config.lm_type,
            retrieval_type=config.retrieval_type,
            item_db_name=config.item_db_name,
            user_db_name=config.user_db_name,
            track_split_types=config.track_split_types,
            user_split_types=config.user_split_types,
            corpus_types=config.corpus_types,
            cache_dir=config.cache_dir,
            device=config.get("retrieval_device", config.device),
            retrieval_device=config.get("retrieval_device", None),
            attn_implementation=config.get("attn_implementation", "sdpa"),
            dtype=torch.bfloat16,
            retrieval_topk=config.get("retrieval_topk", 200),
            retrieval_bm25_topk=config.get("retrieval_bm25_topk", 200),
            retrieval_bert_topk=config.get("retrieval_bert_topk", 200),
            retrieval_rrf_k=config.get("retrieval_rrf_k", 60),
            retrieval_bm25_weight=config.get("retrieval_bm25_weight", 0.5),
            retrieval_bert_weight=config.get("retrieval_bert_weight", 0.3),
            retrieval_bpr_weight=config.get("retrieval_bpr_weight", 0.2),
            retrieval_i2i_weight=config.get("retrieval_i2i_weight", 0.15),
            bm25_field_weights=config.get("bm25_field_weights", None),
            dense_model_name=config.get("dense_model_name", "bert-base-uncased"),
            dense_query_prefix=config.get("dense_query_prefix", ""),
            dense_doc_prefix=config.get("dense_doc_prefix", ""),
            enable_query_rewrite=config.get("enable_query_rewrite", True),
            enable_specificity_routing=False,
            enable_user_to_item=config.get("enable_user_to_item", True),
            enable_seen_track_blocking=False,
            enable_metadata_filtering=config.get("enable_metadata_filtering", False),
            enable_item_to_item=config.get("enable_item_to_item", False),
            enable_llm_query_planning=config.get("enable_llm_query_planning", False),
            metadata_filter_min_pool=config.get("metadata_filter_min_pool", 200),
            llm_query_plan_max_new_tokens=config.get("llm_query_plan_max_new_tokens", 256),
            llm_query_plan_mode=config.get("llm_query_plan_mode", "replace"),
            enable_artist_shortcut=config.get("enable_artist_shortcut", False),
            artist_shortcut_weight=config.get("artist_shortcut_weight", 1.5),
            artist_shortcut_min_count=config.get("artist_shortcut_min_count", 2),
            i2i_embedding_types=config.get("i2i_embedding_types", None),
            i2i_embedding_weights=config.get("i2i_embedding_weights", None),
            enable_train_thought_bm25=config.get("enable_train_thought_bm25", False),
            train_thought_bm25_weight=config.get("train_thought_bm25_weight", 0.4),
            enable_session_cooccurrence=config.get("enable_session_cooccurrence", False),
            session_cooccurrence_weight=config.get("session_cooccurrence_weight", 0.3),
            enable_qwen3_dense=config.get("enable_qwen3_dense", False),
            qwen3_dense_weight=config.get("qwen3_dense_weight", 0.5),
            qwen3_dense_model_name=config.get("qwen3_dense_model_name", "Qwen/Qwen3-Embedding-0.6B"),
            qwen3_dense_embedding_types=config.get("qwen3_dense_embedding_types", None),
            qwen3_dense_embedding_weights=config.get("qwen3_dense_embedding_weights", None),
            qwen3_embedding_query_batch_size=config.get("qwen3_embedding_query_batch_size", 64),
            enable_album_shortcut=config.get("enable_album_shortcut", False),
            album_shortcut_weight=config.get("album_shortcut_weight", 1.0),
            enable_entity_matching=config.get("enable_entity_matching", False),
            entity_matching_weight=config.get("entity_matching_weight", 0.8),
            enable_lambdarank=False,
            load_retrieval=True,
        )
    else:
        # All sources cached — still need music_crs for feature extraction
        print("All retrieval cached. Loading CRS model (metadata only)...")
        music_crs = load_crs_baseline(
            lm_type=config.lm_type,
            retrieval_type=config.retrieval_type,
            item_db_name=config.item_db_name,
            user_db_name=config.user_db_name,
            track_split_types=config.track_split_types,
            user_split_types=config.user_split_types,
            corpus_types=config.corpus_types,
            cache_dir=config.cache_dir,
            device="cpu",
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
            enable_lambdarank=False,
            load_retrieval=False,
        )

    # Prepare query data
    import pandas as pd

    need_batch_data = sources_needed or not cached_metadata or args.no_cache
    if cached_metadata and not args.no_cache and not sources_needed:
        query_metadata_list = cached_metadata
        targets_all = [qm["target_tid"] for qm in query_metadata_list]
        print(f"Loaded {len(query_metadata_list)} cached queries")
    else:
        print("Loading train data...")
        train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
        effective_max = max_sessions or len(train_ds)

        batch_data_all = []
        targets_all = []
        query_metadata_list = []

        print(f"Preparing data from {min(effective_max, len(train_ds))} sessions...")
        for idx, item in enumerate(train_ds):
            if idx >= effective_max:
                break

            session_id = item["session_id"]
            user_id = item.get("user_id", "")
            conversations = item["conversations"]
            conversation_goal = item.get("conversation_goal")
            user_profile = item.get("user_profile")

            df_conv = pd.DataFrame(conversations)
            music_turns = df_conv[df_conv["role"] == "music"]

            if args.goal_filter:
                gpa_map = {g["turn_number"]: g["goal_progress_assessment"]
                           for g in item.get("goal_progress_assessments", [])}
                music_turns = music_turns[
                    music_turns["turn_number"].map(
                        lambda t: gpa_map.get(t + 1) == "MOVES_TOWARD_GOAL"
                    )
                ]

            if args.last_n_turns and len(music_turns) > args.last_n_turns:
                music_turns = music_turns.tail(args.last_n_turns)

            for _, music_row in music_turns.iterrows():
                target_turn = int(music_row["turn_number"])
                target_tid = str(music_row["content"]).strip()

                history = df_conv[df_conv["turn_number"] < target_turn]
                chat_history = []
                for _, row in history.iterrows():
                    role = row["role"]
                    content = row["content"]
                    if role == "music":
                        role = "assistant"
                        content = music_crs.item_db.id_to_metadata(content)
                    chat_history.append({"role": role, "content": content})

                user_turns_at = df_conv[
                    (df_conv["turn_number"] == target_turn) & (df_conv["role"] == "user")
                ]
                if user_turns_at.empty:
                    continue
                user_query = user_turns_at.iloc[0]["content"]

                batch_data_all.append({
                    "user_query": user_query,
                    "user_id": user_id,
                    "session_memory": chat_history,
                    "user_profile": user_profile,
                    "conversation_goal": conversation_goal,
                })
                targets_all.append(target_tid)

                anchors = music_crs._extract_anchor_track_ids(chat_history)
                goal = conversation_goal or {}
                query_metadata_list.append({
                    "target_tid": target_tid,
                    "user_query": user_query,
                    "anchors": anchors,
                    "turn_number": len([t for t in chat_history if t.get("role") == "user"]),
                    "specificity": str(goal.get("specificity", "")).strip().upper(),
                    "category": str(goal.get("category", "")).strip().upper(),
                })

        save_query_metadata(retrieval_cache_dir, query_metadata_list)
        print(f"Total queries to process: {len(batch_data_all)}")

    # Run retrieval for uncached sources
    if sources_needed and need_batch_data:
        print(f"Running retrieval for {len(sources_needed)} sources: {sources_needed}")
        batch_size = args.batch_size

        # Accumulate per-source results across batches
        per_source_results = {s: [] for s in SOURCE_NAMES}

        for batch_start in tqdm(range(0, len(batch_data_all), batch_size), desc="Retrieval"):
            batch_end = min(batch_start + batch_size, len(batch_data_all))
            batch = batch_data_all[batch_start:batch_end]

            try:
                retrieved = music_crs.batch_retrieval(batch)
            except Exception as e:
                print(f"Error in batch {batch_start}: {e}")
                for s in SOURCE_NAMES:
                    per_source_results[s].extend([[] for _ in batch])
                continue

            all_sources = retrieved.get("all_sources", {})

            for s in SOURCE_NAMES:
                items_list = all_sources.get(s, [[] for _ in batch])
                for j in range(len(batch)):
                    items = items_list[j] if j < len(items_list) and items_list[j] else []
                    per_source_results[s].append(items[:200])

        # Save each source's results
        for source_name in SOURCE_NAMES:
            save_source_cache(retrieval_cache_dir, source_name, per_source_results[source_name])
            cached_sources[source_name] = per_source_results[source_name]
        print(f"Saved retrieval results for all {len(SOURCE_NAMES)} sources")

    # Build features from cached retrieval results
    print("\nExtracting features from cached retrieval results...")
    all_features = []
    all_labels = []
    all_groups = []

    num_queries = len(query_metadata_list)
    for qi in tqdm(range(num_queries), desc="Feature extraction"):
        qm = query_metadata_list[qi]
        target_tid = qm["target_tid"]
        user_query = qm["user_query"]
        anchors = qm["anchors"]
        turn_number = qm["turn_number"]
        specificity = ""
        category = ""

        # Build per-source rank maps
        # Exclude sources from feature extraction (candidates still in pool):
        # - train_thought, cooccur: data leakage (built from training data)
        # - primary: circular (RRF fusion that LambdaRank replaces) + carries leakage
        EXCLUDE_FEATURES = {"train_thought", "cooccur", "primary"}
        source_ranks = {}
        for source_name in SOURCE_NAMES:
            if source_name in EXCLUDE_FEATURES:
                continue
            source_data = cached_sources.get(source_name)
            if source_data and qi < len(source_data) and source_data[qi]:
                items = source_data[qi]
                source_ranks[source_name] = {tid: rank + 1 for rank, tid in enumerate(items)}

        # Pool all candidates from all sources
        candidate_set = set()
        for source_name in SOURCE_NAMES:
            source_data = cached_sources.get(source_name)
            if source_data and qi < len(source_data) and source_data[qi]:
                candidate_set.update(source_data[qi][:200])
        if not candidate_set:
            continue
        if target_tid not in candidate_set:
            candidate_set.add(target_tid)

        # Zero out co-occurrence counts during training — data leakage:
        # the graph is built from training data, inflating counts for
        # training examples. At inference the signal is weaker.
        cooccur_counts = None

        group_features = []
        group_labels = []

        for tid in candidate_set:
            feat = music_crs._extract_lambdarank_features(
                tid, source_ranks, user_query, anchors,
                turn_number, specificity, category, cooccur_counts,
            )
            label = 1 if tid == target_tid else 0
            group_features.append(feat)
            group_labels.append(label)

        all_features.extend(group_features)
        all_labels.extend(group_labels)
        all_groups.append(len(group_features))

    print(f"\nTraining data: {len(all_features)} candidates, {len(all_groups)} groups")
    print(f"Positive labels: {sum(all_labels)}, Negative: {len(all_labels) - sum(all_labels)}")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)
    groups = np.array(all_groups, dtype=np.int32)

    train_data = lgb.Dataset(
        X, label=y, group=groups,
        feature_name=music_crs.LAMBDARANK_FEATURE_NAMES,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [1, 5, 10, 20],
        "label_gain": [0, 1],
        "num_leaves": 63,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 1,
        "n_jobs": -1,
    }

    print("\nTraining LambdaRank...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=args.num_rounds,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(50)],
    )

    output_path = args.output or os.path.join(config.cache_dir, "lambdarank_model.txt")
    model.save_model(output_path)
    print(f"\nModel saved to {output_path}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_names = music_crs.LAMBDARANK_FEATURE_NAMES
    print("\nFeature Importance (gain):")
    for name, imp in sorted(zip(feat_names, importance), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_rounds", type=int, default=300)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--last_n_turns", type=int, default=None, help="Only train on last N music turns per session (default: all)")
    parser.add_argument("--goal_filter", action="store_true", help="Only train on MOVES_TOWARD_GOAL turns")
    parser.add_argument("--no_cache", action="store_true", help="Force re-run retrieval even if cache exists")
    args = parser.parse_args()
    main(args)
