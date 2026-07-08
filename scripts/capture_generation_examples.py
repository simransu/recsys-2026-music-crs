"""
Capture 10 dev examples showing exact inputs and outputs passed to the LLM.
Output: generation_examples.json — one entry per example with:
  - session_id, user_id, final_user_query
  - system_prompt
  - chat_history  (messages before the evidence block)
  - evidence_block  (exact final user message sent to the model)
  - model_response  (raw model output)
  - top_track_id, top_track_metadata
"""
import inspect
import json
import torch
from mcrs import load_crs_baseline
from datasets import load_dataset
from omegaconf import OmegaConf

N_EXAMPLES = 10
OUTPUT_FILE = "generation_examples.json"
CONFIG_PATH = "config/qwen3_8b_bm25_plain_devset.yaml"

config = OmegaConf.load(CONFIG_PATH)

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
    attn_implementation="sdpa",
    dtype=torch.bfloat16,
    bm25_field_weights=config.get("bm25_field_weights", None),
    enable_query_rewrite=config.get("enable_query_rewrite", False),
    enable_specificity_routing=False,
    enable_user_to_item=False,
)

_supports_session_memory = "session_memory" in inspect.signature(
    music_crs._build_recommendation_context
).parameters

db = load_dataset(config.test_dataset_name, split="test")
raw_items = list(db)[:N_EXAMPLES]

# --- Stage 1: retrieval for all items ---
music_crs.load_retrieval()

prepared = []
for item in raw_items:
    conversations = item["conversations"]
    user_query = conversations[-1]["content"]
    chat_history_raw = conversations[:-1]

    session_memory = [
        {
            "role": t["role"] if t["role"] != "music" else "assistant",
            "content": (
                t["content"] if t["role"] != "music"
                else music_crs.item_db.id_to_metadata(t["content"])
            ),
        }
        for t in chat_history_raw
    ]
    session_memory_with_query = session_memory + [{"role": "user", "content": user_query}]
    system_prompt = music_crs._get_system_prompt(item["user_id"])

    retrieval_input = music_crs._build_retrieval_input(session_memory_with_query, item["user_id"])
    retrieval_items, _ = music_crs._retrieve_route_batch(
        [{"user_id": item["user_id"], "conversation_goal": None}],
        [retrieval_input],
    )
    retrieval_items = retrieval_items[0]

    top_track_id = retrieval_items[0] if retrieval_items else None
    top_track_metadata = music_crs.item_db.id_to_metadata(top_track_id) if top_track_id else ""
    if _supports_session_memory:
        evidence_block = music_crs._build_recommendation_context(
            retrieval_items, session_memory=session_memory_with_query
        )
    else:
        evidence_block = music_crs._build_recommendation_context(retrieval_items)

    prepared.append({
        "item": item,
        "user_query": user_query,
        "session_memory": session_memory,
        "system_prompt": system_prompt,
        "evidence_block": evidence_block,
        "top_track_id": top_track_id,
        "top_track_metadata": top_track_metadata,
    })

music_crs.cleanup_retrieval()

# --- Stage 2: generation for all items ---
music_crs.load_lm()

examples = []
for p in prepared:
    model_response = music_crs.lm.response_generation(
        p["system_prompt"],
        p["session_memory"],  # chat history without the evidence block
        p["evidence_block"],  # evidence block is the final user message
    )

    example = {
        "session_id": p["item"]["session_id"],
        "user_id": p["item"]["user_id"],
        "final_user_query": p["user_query"],
        "system_prompt": p["system_prompt"],
        "chat_history": p["session_memory"],
        "evidence_block": p["evidence_block"],
        "model_response": model_response,
        "top_track_id": p["top_track_id"],
        "top_track_metadata": p["top_track_metadata"],
    }
    examples.append(example)

    print(f"\n=== Session {p['item']['session_id'][:8]} ===")
    print(f"User: {p['user_query'][:120]}")
    print(f"Top track: {p['top_track_id']}")
    print(f"Response: {model_response[:300]}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(examples, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(examples)} examples to {OUTPUT_FILE}")
