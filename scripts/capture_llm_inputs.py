"""
Capture exact LLM inputs and outputs for N dev examples, mirroring the
run_inference_blindset.py pipeline (BM25 retrieval → _build_recommendation_context → generation).

Output: llm_input_samples.json
Each entry contains:
  - session_id, user_id, turn_number
  - system_prompt       — exact system message
  - chat_history        — exact message list passed as chat history
  - evidence_block      — exact final user message (output of _build_recommendation_context)
  - model_response      — raw LLM output
  - top_track_id        — top ranked track ID
  - top_track_metadata  — human-readable metadata string for the top track
"""
import json
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from mcrs import load_crs_baseline

N = 10
OUTPUT = "llm_input_samples.json"
CONFIG = "config/qwen3_8b_bm25_plain_devset.yaml"

config = OmegaConf.load(CONFIG)

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

db = load_dataset(config.test_dataset_name, split="test")
raw_items = list(db)[:N]

# Build batch_data in the same format as run_inference_blindset.py
batch_data = []
meta = []
for item in raw_items:
    conversations = item["conversations"]
    chat_history = [
        {
            "role": t["role"] if t["role"] != "music" else "assistant",
            "content": (
                t["content"] if t["role"] != "music"
                else music_crs.item_db.id_to_metadata(t["content"])
            ),
        }
        for t in conversations[:-1]
    ]
    batch_data.append({
        "user_query": conversations[-1]["content"],
        "user_id": item["user_id"],
        "session_memory": chat_history,
        "user_profile": item.get("user_profile"),
        "conversation_goal": item.get("conversation_goal"),
    })
    meta.append({
        "session_id": item["session_id"],
        "user_id": item["user_id"],
        "turn_number": conversations[-1].get("turn_number"),
    })

# --- Stage 1: retrieval (exact same call as run_inference_blindset.py) ---
music_crs.load_retrieval()
retrieved = music_crs.batch_retrieval(batch_data)
music_crs.cleanup_retrieval()

# No reranker — ranked_items = retrieval_items (same as inference script)
ranked_items = retrieved["retrieval_items"]

# Build evidence blocks the same way batch_generation does internally
evidence_blocks = [
    music_crs._build_recommendation_context(items, session_memory=retrieved["session_memories"][i])
    for i, items in enumerate(ranked_items)
]

# --- Stage 2: generation ---
music_crs.load_lm()

samples = []
for i in range(len(batch_data)):
    sys_prompt = retrieved["sys_prompts"][i]
    session_memory = retrieved["session_memories"][i]
    evidence_block = evidence_blocks[i]
    top_track_id = ranked_items[i][0] if ranked_items[i] else None
    top_track_metadata = music_crs.item_db.id_to_metadata(top_track_id) if top_track_id else ""

    response = music_crs.lm.response_generation(sys_prompt, session_memory, evidence_block)

    sample = {
        "session_id": meta[i]["session_id"],
        "user_id": meta[i]["user_id"],
        "turn_number": meta[i]["turn_number"],
        "system_prompt": sys_prompt,
        "chat_history": session_memory,
        "evidence_block": evidence_block,
        "model_response": response,
        "top_track_id": top_track_id,
        "top_track_metadata": top_track_metadata,
    }
    samples.append(sample)

    print(f"\n=== {meta[i]['session_id'][:8]} | turn {meta[i]['turn_number']} ===")
    print(f"User: {session_memory[-1]['content'][:120]}")
    print(f"Track: {top_track_metadata[:80]}")
    print(f"Response: {response[:300]}")

music_crs.cleanup_lm()

with open(OUTPUT, "w") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(samples)} samples to {OUTPUT}")
