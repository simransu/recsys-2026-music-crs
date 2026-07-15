"""
Batch inference script for Music CRS.
"""

import os
import json
import torch
import argparse
from mcrs import load_crs_baseline
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import pandas as pd
from omegaconf import OmegaConf

def chat_history_parser(conversations, music_crs, target_turn_number):
    """
    Parse conversation history up to a target turn.

    Args:
        conversations (List[Dict]): List of conversation turn dictionaries containing:
            - turn_number: Turn index (1-8)
            - role: Speaker role ('user', 'assistant', 'music')
            - content: Message content or track ID
        music_crs: CRS baseline instance (used to convert track IDs to metadata)
        target_turn_number (int): The turn to predict (history excludes this turn)

    Returns:
        Tuple[List[Dict], str]:
            - chat_history: List of previous messages formatted as [{"role": ..., "content": ...}]
            - user_query: The user query at the target turn
    """
    df_conversation = pd.DataFrame(conversations)
    df_history = df_conversation[df_conversation['turn_number'] < target_turn_number]
    chat_history = []
    for turn_data in df_history.to_dict(orient="records"):
        turn_number = turn_data['turn_number']
        current_role = turn_data['role']
        current_content = turn_data['content']
        if turn_data['role'] == "music":
            current_role = "assistant"
            current_content = music_crs.item_db.id_to_metadata(turn_data['content'])
        chat_history.append({
            "role": current_role,
            "content": current_content
        })
    df_current_turn = df_conversation[df_conversation['turn_number'] == target_turn_number]
    user_query = df_current_turn.iloc[0]['content']
    return chat_history, user_query

def main(args):
    """
    Run batch inference on TalkPlayData-2 test dataset.

    Args:
        args: Namespace object containing:
            - tid (str): Task/configuration identifier
            - batch_size (int): Batch size for inference
            - save_path (str): Output directory (currently unused)

    Returns:
        None. Results are saved to exp/inference/{tid}.json

    Processing:
        - Loads model configuration from config/{tid}.yaml
        - Processes all sessions × 8 turns in batches
        - Tracks progress with tqdm progress bar
        - Saves comprehensive results for evaluation
    """
    os.makedirs("cache", exist_ok=True)
    config = OmegaConf.load(f"config/{args.tid}.yaml")
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
        retrieval_bpr_weight=config.get("retrieval_bpr_weight", 0.2),
        retrieval_i2i_weight=config.get("retrieval_i2i_weight", 0.15),
        bm25_field_weights=config.get("bm25_field_weights", None),
        enable_query_rewrite=config.get("enable_query_rewrite", True),
        enable_specificity_routing=config.get("enable_specificity_routing", True),
        enable_user_to_item=config.get("enable_user_to_item", True),
        enable_seen_track_blocking=config.get("enable_seen_track_blocking", False),
        enable_metadata_filtering=config.get("enable_metadata_filtering", False),
        enable_item_to_item=config.get("enable_item_to_item", False),
        enable_llm_query_planning=config.get("enable_llm_query_planning", False),
        metadata_filter_min_pool=config.get("metadata_filter_min_pool", 50),
        llm_query_plan_max_new_tokens=config.get("llm_query_plan_max_new_tokens", 256),
        llm_query_plan_mode=config.get("llm_query_plan_mode", "replace"),
        dense_model_name=config.get("dense_model_name", "bert-base-uncased"),
        dense_query_prefix=config.get("dense_query_prefix", ""),
        dense_doc_prefix=config.get("dense_doc_prefix", ""),
        specificity_route_map=config.get("specificity_route_map", None),
        load_reranker=False,
        enable_artist_shortcut=config.get("enable_artist_shortcut", False),
        artist_shortcut_weight=config.get("artist_shortcut_weight", 1.5),
        artist_shortcut_min_count=config.get("artist_shortcut_min_count", 2),
        i2i_embedding_types=config.get("i2i_embedding_types", None),
        i2i_embedding_weights=config.get("i2i_embedding_weights", None),
        enable_album_shortcut=config.get("enable_album_shortcut", False),
        album_shortcut_weight=config.get("album_shortcut_weight", 1.0),
        enable_entity_matching=config.get("enable_entity_matching", False),
        entity_matching_weight=config.get("entity_matching_weight", 0.8),
        enable_lambdarank=config.get("enable_lambdarank", False),
        lambdarank_model_path=config.get("lambdarank_model_path", "./cache/lambdarank_model.txt"),
        enable_train_thought_bm25=config.get("enable_train_thought_bm25", False),
        train_thought_bm25_weight=config.get("train_thought_bm25_weight", 0.4),
        enable_session_cooccurrence=config.get("enable_session_cooccurrence", False),
        session_cooccurrence_weight=config.get("session_cooccurrence_weight", 0.3),
        enable_qwen3_dense=config.get("enable_qwen3_dense", False),
        qwen3_dense_weight=config.get("qwen3_dense_weight", 0.5),
        qwen3_dense_model_name=config.get("qwen3_dense_model_name", "Qwen/Qwen3-Embedding-0.6B"),
        qwen3_dense_embedding_types=config.get("qwen3_dense_embedding_types", ["attributes-qwen3_embedding_0.6b"]),
        qwen3_dense_embedding_weights=config.get("qwen3_dense_embedding_weights", None),
    )
    db = load_dataset(config.test_dataset_name, split="test")
    # Prepare all batch data at once
    batch_data, metadata = [], []
    for item in db:
        user_id = item['user_id']
        session_id = item['session_id']
        chat_history = item['conversations'][:-1]
        user_query = item['conversations'][-1]['content']
        turn_number = item['conversations'][-1]['turn_number']
        batch_data.append({
                'user_query': user_query,
                'user_id': user_id,
                'session_memory': chat_history,
                'user_profile': item.get('user_profile'),
                'conversation_goal': item.get('conversation_goal'),
            })
        metadata.append({
            'session_id': session_id,
            'user_id': user_id,
            'turn_number': turn_number
        })
    retrieval_batch_size = args.retrieval_batch_size
    print(f"Running retrieval first with batch_size={retrieval_batch_size}...")
    music_crs.load_retrieval()
    retrieval_results = []
    for i in tqdm(range(0, len(batch_data), retrieval_batch_size), desc="BERT retrieval"):
        batch = batch_data[i:i+retrieval_batch_size]
        batch_metadata = metadata[i:i+retrieval_batch_size]
        retrieved = music_crs.batch_retrieval(batch)
        retrieval_results.append((retrieved, batch_metadata))

    music_crs.cleanup_retrieval()

    if config.get("reranker_type", None):
        print("Running two-tower rerank on retrieval candidates...")
        music_crs.load_reranker()
        reranked_results = []
        for retrieved, batch_metadata in tqdm(retrieval_results, desc="Two-tower rerank"):
            reranked = music_crs.batch_rerank(
                retrieved["user_ids"],
                retrieved["retrieval_items"],
                user_profiles=retrieved.get("user_profiles"),
                query_texts=retrieved.get("retrieval_inputs"),
                conversation_goals=retrieved.get("conversation_goals"),
            )
            retrieved["ranked_items"] = reranked["ranked_items"]
            retrieved["recommend_items"] = reranked["recommend_items"]
            reranked_results.append((retrieved, batch_metadata))
        retrieval_results = reranked_results
        music_crs.cleanup_reranker()
    else:
        for retrieved, _ in retrieval_results:
            retrieved["ranked_items"] = retrieved["retrieval_items"]
            retrieved["recommend_items"] = [music_crs.item_db.id_to_metadata(items[0]) for items in retrieved["retrieval_items"]]

    generation_inputs = []
    for retrieved, batch_metadata in retrieval_results:
        for j in range(len(batch_metadata)):
            generation_inputs.append({
                "sys_prompt": retrieved["sys_prompts"][j],
                "session_memory": retrieved["session_memories"][j],
                "recommend_item": retrieved["recommend_items"][j],
                "ranked_items": retrieved["ranked_items"][j],
                "planned_query": retrieved.get("planned_queries", [{}] * len(batch_metadata))[j],
                "metadata": batch_metadata[j],
            })

    inference_results = []
    if args.skip_generation:
        print("Skipping generation; writing empty predicted_response fields.")
        for item in generation_inputs:
            inference_results.append({
                "session_id": item["metadata"]['session_id'],
                "user_id": item["metadata"]['user_id'],
                "turn_number": item["metadata"]['turn_number'],
                "predicted_track_ids": item["ranked_items"][:20],
                "predicted_response": "",
            })
    else:
        print(f"Running generation second with batch_size={args.batch_size}...")
        music_crs.load_lm()
        for i in tqdm(range(0, len(generation_inputs), args.batch_size), desc="LLM generation"):
            batch = generation_inputs[i:i + args.batch_size]
            responses = music_crs.batch_generation(
                [item["sys_prompt"] for item in batch],
                [item["session_memory"] for item in batch],
                [item["recommend_item"] for item in batch],
                ranked_items=[item["ranked_items"] for item in batch],
                planned_queries=[item.get("planned_query") for item in batch],
            )
            for j, item in enumerate(batch):
                inference_results.append({
                    "session_id": item["metadata"]['session_id'],
                    "user_id": item["metadata"]['user_id'],
                    "turn_number": item["metadata"]['turn_number'],
                    "predicted_track_ids": item["ranked_items"][:20],
                    "predicted_response": responses[j]
                })
        music_crs.cleanup_lm()
    os.makedirs(f"exp/inference/{args.eval_dataset}", exist_ok=True)
    with open(f"exp/inference/{args.eval_dataset}/{args.tid}.json", "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch inference on TalkPlayData-2 test dataset for Music CRS evaluation."
    )
    parser.add_argument(
        "--tid",
        type=str,
        default="llama1b_bm25_blindset_A",
        help="Task identifier matching a config file (e.g., 'llama1b_bm25' loads config/llama1b_bm25.yaml)"
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="blindset_A",
        help="Evaluation dataset name"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of queries to process in parallel. Reduce if encountering GPU memory issues."
    )
    parser.add_argument(
        "--retrieval_batch_size",
        type=int,
        default=32,
        help="Batch size for the retrieval pass. Can usually be larger than generation because BERT is smaller."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./exp/inference",
        help="Base directory for saving results (currently not used, results saved to exp/inference/)"
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip LLM generation and write empty predicted_response fields for faster ranking-only evaluation."
    )
    args = parser.parse_args()
    main(args)
