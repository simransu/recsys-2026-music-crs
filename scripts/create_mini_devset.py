"""Create a stratified mini dev set (100 rows) preserving specificity distribution."""
import json
import random
from collections import defaultdict
from datasets import load_dataset

random.seed(42)

ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")

buckets = defaultdict(list)
for item in ds:
    cg = item.get("conversation_goal") or {}
    spec = str(cg.get("specificity", "")).strip().upper() or "UNKNOWN"
    buckets[spec].append(item)

target = {"HH": 10, "HL": 31, "LH": 31, "LL": 28}
sampled = []
for spec, count in target.items():
    pool = buckets[spec]
    sampled.extend(random.sample(pool, min(count, len(pool))))

random.shuffle(sampled)
session_ids = [item["session_id"] for item in sampled]
json.dump(session_ids, open("config/mini_devset_session_ids.json", "w"), indent=2)
print(f"Saved {len(session_ids)} session IDs to config/mini_devset_session_ids.json")

from collections import Counter
specs = []
for item in sampled:
    cg = item.get("conversation_goal") or {}
    specs.append(str(cg.get("specificity", "")).strip().upper())
counts = Counter(specs)
for k, v in sorted(counts.items()):
    print(f"  {k}: {v} ({v/len(sampled)*100:.1f}%)")
