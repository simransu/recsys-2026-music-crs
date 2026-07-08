"""
Capture 10 LLM input examples per specificity bucket (HH, HL, LH, LL) from the
dev set. Does NOT run generation — only captures the exact inputs the LLM would
receive (system_prompt, chat_history, evidence_block) for prompt tuning.

Output files (one per bucket):
  llm_samples_HH.json
  llm_samples_HL.json
  llm_samples_LH.json
  llm_samples_LL.json
"""
import inspect
import json
import torch
from collections import defaultdict
from datasets import load_dataset
from omegaconf import OmegaConf
from mcrs import load_crs_baseline

N_PER_BUCKET = 10
BUCKETS = ["HH", "HL", "LH", "LL"]
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
    enable_query_rewrite=False,
    enable_specificity_routing=False,
    enable_user_to_item=False,
)

db = load_dataset(config.test_dataset_name, split="test")

# --- Bucket items by specificity, take first N_PER_BUCKET per bucket ---
bucketed = defaultdict(list)
for item in db:
    conversation_goal = item.get("conversation_goal") or {}
    specificity = str(
        conversation_goal.get("specificity") or item.get("goal_specificity") or ""
    ).strip().upper()
    if specificity in BUCKETS and len(bucketed[specificity]) < N_PER_BUCKET:
        bucketed[specificity].append(item)
    if all(len(bucketed[b]) >= N_PER_BUCKET for b in BUCKETS):
        break

for b in BUCKETS:
    print(f"Bucket {b}: {len(bucketed[b])} items")

# Flatten into a single batch preserving bucket label
all_items = []
for b in BUCKETS:
    for item in bucketed[b]:
        all_items.append((b, item))

# Build batch_data mirroring run_inference_blindset.py
batch_data = []
meta = []
for bucket, item in all_items:
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
        "bucket": bucket,
        "session_id": item["session_id"],
        "user_id": item["user_id"],
        "turn_number": conversations[-1].get("turn_number"),
        "conversation_goal": item.get("conversation_goal"),
    })

# --- Stage 1: retrieval ---
music_crs.load_retrieval()
retrieved = music_crs.batch_retrieval(batch_data)
music_crs.cleanup_retrieval()

ranked_items = retrieved["retrieval_items"]

# Build evidence blocks exactly as batch_generation does
_supports_session_memory = "session_memory" in inspect.signature(
    music_crs._build_recommendation_context
).parameters
evidence_blocks = [
    music_crs._build_recommendation_context(
        items, session_memory=retrieved["session_memories"][i]
    ) if _supports_session_memory else music_crs._build_recommendation_context(items)
    for i, items in enumerate(ranked_items)
]

bucket_samples = defaultdict(list)

for i in range(len(batch_data)):
    sys_prompt = retrieved["sys_prompts"][i]
    session_memory = retrieved["session_memories"][i]
    evidence_block = evidence_blocks[i]
    top_track_id = ranked_items[i][0] if ranked_items[i] else None
    top_track_metadata = music_crs.item_db.id_to_metadata(top_track_id) if top_track_id else ""

    bucket = meta[i]["bucket"]
    sample = {
        "session_id": meta[i]["session_id"],
        "user_id": meta[i]["user_id"],
        "turn_number": meta[i]["turn_number"],
        "specificity": bucket,
        "conversation_goal": meta[i]["conversation_goal"],
        "system_prompt": sys_prompt,
        "chat_history": session_memory,
        "evidence_block": evidence_block,
        "top_track_id": top_track_id,
        "top_track_metadata": top_track_metadata,
    }
    bucket_samples[bucket].append(sample)

    print(f"\n[{bucket}] {meta[i]['session_id'][:8]} | turn {meta[i]['turn_number']}")
    print(f"  User: {session_memory[-1]['content'][:100]}")
    print(f"  Evidence:\n{evidence_block}")

# --- Save one file per bucket ---
for bucket in BUCKETS:
    out_file = f"llm_samples_{bucket}.json"
    with open(out_file, "w") as f:
        json.dump(bucket_samples[bucket], f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(bucket_samples[bucket])} samples to {out_file}")
