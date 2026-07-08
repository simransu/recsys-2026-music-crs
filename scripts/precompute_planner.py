"""Pre-compute and cache LLM query planner results for all training queries.

Run this BEFORE train_lambdarank.py so the planner cache is fully populated
and the training script doesn't need to load the LLM (freeing GPU for retrieval).

Usage:
    python scripts/precompute_planner.py --config config/lambdarank_training.yaml --last_n_turns 1 --goal_filter
"""

import os
import json
import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
from omegaconf import OmegaConf
from mcrs import load_crs_baseline


def main(args):
    config = OmegaConf.load(args.config)

    print("Loading CRS model (LLM only for planner)...")
    music_crs = load_crs_baseline(
        lm_type=config.lm_type,
        retrieval_type=config.retrieval_type,
        item_db_name=config.item_db_name,
        user_db_name=config.user_db_name,
        track_split_types=config.track_split_types,
        user_split_types=config.user_split_types,
        corpus_types=config.corpus_types,
        cache_dir=config.cache_dir,
        device=config.device,
        attn_implementation=config.get("attn_implementation", "sdpa"),
        dtype=torch.bfloat16,
        enable_llm_query_planning=True,
        llm_query_plan_max_new_tokens=config.get("llm_query_plan_max_new_tokens", 256),
        llm_query_plan_mode=config.get("llm_query_plan_mode", "replace"),
        enable_lambdarank=False,
        load_retrieval=False,
    )

    import pandas as pd

    print("Loading train data...")
    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    max_sessions = args.max_sessions or len(train_ds)

    batch_data_all = []
    session_memories_all = []

    print(f"Preparing data from {min(max_sessions, len(train_ds))} sessions...")
    for idx, item in enumerate(train_ds):
        if idx >= max_sessions:
            break

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

            session_memory = chat_history.copy()
            session_memory.append({"role": "user", "content": user_query})

            batch_data_all.append({
                "user_query": user_query,
                "user_id": user_id,
                "session_memory": chat_history,
                "user_profile": user_profile,
                "conversation_goal": conversation_goal,
            })
            session_memories_all.append(session_memory)

    print(f"Total queries to plan: {len(batch_data_all)}")

    # Check how many are already cached
    cache_dir = os.path.join(config.cache_dir, "planner_cache")
    os.makedirs(cache_dir, exist_ok=True)
    already_cached = 0
    for i in range(len(batch_data_all)):
        cache_key = music_crs._plan_cache_key(session_memories_all[i], batch_data_all[i].get("user_id"))
        cache_path = os.path.join(cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_path):
            already_cached += 1
    print(f"Already cached: {already_cached}/{len(batch_data_all)}")

    if already_cached == len(batch_data_all):
        print("All queries already cached. Nothing to do.")
        return

    # Process in batches
    batch_size = args.batch_size
    music_crs.load_lm()
    print(f"LLM loaded. Planning {len(batch_data_all) - already_cached} uncached queries...")

    newly_cached = 0
    for batch_start in tqdm(range(0, len(batch_data_all), batch_size), desc="Planning"):
        batch_end = min(batch_start + batch_size, len(batch_data_all))
        batch_sessions = session_memories_all[batch_start:batch_end]
        batch_data = batch_data_all[batch_start:batch_end]

        results = music_crs._batch_plan_queries(batch_sessions, batch_data)
        for r in results:
            if r:
                newly_cached += 1

    print(f"\nDone. Newly cached: {newly_cached}")
    print(f"Total cached: {already_cached + newly_cached}/{len(batch_data_all)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--last_n_turns", type=int, default=None)
    parser.add_argument("--goal_filter", action="store_true")
    args = parser.parse_args()
    main(args)
