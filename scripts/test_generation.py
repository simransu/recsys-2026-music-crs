"""Quick test to verify generation quality on a few samples."""
import torch
from mcrs import load_crs_baseline
from datasets import load_dataset
from omegaconf import OmegaConf
import pandas as pd

config = OmegaConf.load("config/qwen3_8b_bm25_blindset_A.yaml")
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
    attn_implementation=config.attn_implementation,
    dtype=torch.bfloat16,
    bm25_field_weights=config.get("bm25_field_weights", None),
    enable_query_rewrite=config.get("enable_query_rewrite", True),
    enable_specificity_routing=False,
    enable_user_to_item=False,
    enable_llm_query_planning=config.get("enable_llm_query_planning", False),
    llm_query_plan_max_new_tokens=config.get("llm_query_plan_max_new_tokens", 256),
    llm_query_plan_mode=config.get("llm_query_plan_mode", "replace"),
)

db = load_dataset(config.test_dataset_name, split="test")
items = list(db)[:2]

music_crs.load_retrieval()
music_crs.load_lm()

for item in items:
    conversations = item['conversations']
    last_turn = conversations[-1]
    user_query = last_turn['content']
    chat_history = conversations[:-1]

    session_memory = [{'role': t['role'] if t['role'] != 'music' else 'assistant',
                       'content': t['content'] if t['role'] != 'music' else music_crs.item_db.id_to_metadata(t['content'])}
                      for t in chat_history]
    music_crs._upload_session_memory(session_memory)
    result = music_crs.chat(
        user_query=user_query,
        user_id=item['user_id'],
    )
    evidence = music_crs._build_recommendation_context(result['ranked_items'], session_memory=music_crs.session_memory)
    print(f"\n=== Session {item['session_id'][:8]} ===")
    print(f"User: {user_query[:120]}")
    print(f"Evidence block:\n{evidence}")
    print(f"Response: {result['response']}")
    print()
