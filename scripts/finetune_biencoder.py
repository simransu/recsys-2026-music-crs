"""Fine-tune Qwen3-Embedding-0.6B query encoder against frozen track embeddings.

Strategy:
  - Only the query encoder is trained. Track embeddings are precomputed
    (attributes-qwen3_embedding_0.6b) and frozen.
  - Trained on goal-filtered, last-1-turn data (~13k pairs) matching LambdaRank.
  - Hard negatives from cached retrieval (BM25, BERT, I2I, artist) — tracks
    that our current system retrieves but are NOT the ground truth.
  - Mix: 5 hard negatives + 15 easy (random) negatives per query.

Future expansion:
  - Train on all turns (121k pairs) — requires running retrieval for all turns
    to get hard negatives, or falling back to random for uncached queries.
  - Also retrain LambdaRank on all turns for consistency.
  - Hard negatives from same-artist or same-genre could further improve.

Usage:
    python scripts/finetune_biencoder.py --epochs 5 --batch_size 16 --lr 2e-5
    python scripts/finetune_biencoder.py --epochs 5 --batch_size 16 --lr 2e-5 --goal_filter --last_n_turns 1
"""

import os
import json
import argparse
import random
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, concatenate_datasets
import pandas as pd
from tqdm import tqdm


TASK_INSTRUCTION = (
    "Given a music recommendation conversation, "
    "retrieve tracks that match the listener's request"
)


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    s = str(value)
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return ", ".join(str(v) for v in parsed)
        except Exception:
            pass
    return s


class QueryTrackDataset(Dataset):
    """Each item is (query_text, target_index, hard_neg_indices)."""

    def __init__(self, queries: list[str], target_indices: list[int],
                 hard_negatives: list[list[int]]):
        self.queries = queries
        self.target_indices = target_indices
        self.hard_negatives = hard_negatives

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.target_indices[idx], self.hard_negatives[idx]


def collate_queries(batch):
    queries, targets, hard_negs = zip(*batch)
    return list(queries), list(targets), list(hard_negs)


def load_track_embeddings(
    embedding_type: str = "attributes-qwen3_embedding_0.6b",
    cache_dir: str = "./cache",
) -> tuple[list[str], torch.Tensor]:
    index_dir = os.path.join(cache_dir, "qwen3_dense")
    ids_path = os.path.join(index_dir, "track_ids.json")
    emb_path = os.path.join(index_dir, f"{embedding_type}.pt")

    if os.path.exists(ids_path) and os.path.exists(emb_path):
        print(f"Loading cached track embeddings from {index_dir}")
        track_ids = json.load(open(ids_path))
        track_embs = torch.load(emb_path, map_location="cpu")
    else:
        print("Loading track embeddings from HuggingFace...")
        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
        merged = concatenate_datasets([ds[s] for s in ds.keys()])

        track_ids = []
        embs = []
        for row in merged:
            tid = str(row["track_id"]).strip()
            vec = row.get(embedding_type)
            if vec is None or len(vec) == 0:
                continue
            track_ids.append(tid)
            tensor = torch.tensor(vec, dtype=torch.float32)
            tensor = F.normalize(tensor, dim=0)
            embs.append(tensor)

        track_embs = torch.stack(embs, dim=0)
        os.makedirs(index_dir, exist_ok=True)
        json.dump(track_ids, open(ids_path, "w"))
        torch.save(track_embs, emb_path)

    print(f"Track embeddings: {len(track_ids)} tracks, dim={track_embs.shape[1]}")
    return track_ids, track_embs


def load_hard_negatives(cache_dir: str, last_n_turns: Optional[int], goal_filter: bool):
    """Load cached retrieval results from LambdaRank training as hard negatives."""
    turns_tag = f"_last{last_n_turns}" if last_n_turns else ""
    goal_tag = "_goalonly" if goal_filter else ""
    tag = f"lambdarank_retrieval_all{turns_tag}{goal_tag}"
    retrieval_cache_dir = os.path.join(cache_dir, tag)

    if not os.path.exists(retrieval_cache_dir):
        print(f"No cached retrieval at {retrieval_cache_dir} — using random negatives only")
        return None, None

    sources_to_use = ["bm25", "bert", "i2i", "artist", "album", "entity", "qwen3_dense"]
    source_data = {}
    for source in sources_to_use:
        path = os.path.join(retrieval_cache_dir, f"{source}.json")
        if os.path.exists(path):
            with open(path) as f:
                source_data[source] = json.load(f)

    meta_path = os.path.join(retrieval_cache_dir, "query_metadata.json")
    if not os.path.exists(meta_path):
        print("No query_metadata.json — using random negatives only")
        return None, None

    with open(meta_path) as f:
        query_metadata = json.load(f)

    print(f"Loaded hard negatives from {len(source_data)} sources: {list(source_data.keys())}")
    print(f"  {len(query_metadata)} cached queries")

    # Build per-query hard negative pools
    hard_neg_pools = []
    for qi in range(len(query_metadata)):
        target_tid = query_metadata[qi]["target_tid"]
        pool = set()
        for source_name, data in source_data.items():
            if qi < len(data) and data[qi]:
                pool.update(data[qi][:50])
        pool.discard(target_tid)
        hard_neg_pools.append(list(pool))

    return hard_neg_pools, query_metadata


def build_training_data(
    track_ids: list[str],
    hard_neg_pools: Optional[list[list[str]]],
    cached_query_metadata: Optional[list[dict]],
    max_sessions: Optional[int] = None,
    last_n_turns: Optional[int] = None,
    goal_filter: bool = False,
    num_hard_neg: int = 5,
) -> tuple[list[str], list[int], list[list[int]], dict[str, str]]:
    """Build (query_text, target_index, hard_neg_indices) triples."""
    tid_to_idx = {tid: i for i, tid in enumerate(track_ids)}

    print("Loading training data...")
    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")

    print("Loading track metadata for context building...")
    track_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    track_concat = concatenate_datasets([track_ds[s] for s in track_ds.keys()])
    track_meta_text = {}
    for item in track_concat:
        tid = str(item["track_id"]).strip()
        track_name = _fmt(item.get("track_name", ""))
        artist_name = _fmt(item.get("artist_name", ""))
        tag_list = _fmt(item.get("tag_list", ""))
        doc = f"{track_name} by {artist_name}"
        if tag_list:
            tags_short = ", ".join(tag_list.split(", ")[:8]) if isinstance(tag_list, str) else tag_list
            doc += f". {tags_short}"
        track_meta_text[tid] = doc

    effective_max = max_sessions or len(train_ds)
    queries = []
    target_indices = []
    hard_negatives = []
    cached_qi = 0

    print(f"Building pairs from {min(effective_max, len(train_ds))} sessions...")
    for idx, item in enumerate(tqdm(train_ds, desc="Building pairs")):
        if idx >= effective_max:
            break

        conversations = item["conversations"]
        conversation_goal = item.get("conversation_goal") or {}
        listener_goal = str(conversation_goal.get("listener_goal", ""))

        df_conv = pd.DataFrame(conversations)
        music_turns = df_conv[df_conv["role"] == "music"]

        if goal_filter:
            gpa_map = {g["turn_number"]: g["goal_progress_assessment"]
                       for g in item.get("goal_progress_assessments", [])}
            music_turns = music_turns[
                music_turns["turn_number"].map(
                    lambda t: gpa_map.get(t + 1) == "MOVES_TOWARD_GOAL"
                )
            ]

        if last_n_turns and len(music_turns) > last_n_turns:
            music_turns = music_turns.tail(last_n_turns)

        for _, music_row in music_turns.iterrows():
            target_turn = int(music_row["turn_number"])
            target_tid = str(music_row["content"]).strip()

            if target_tid not in tid_to_idx:
                cached_qi += 1
                continue

            history = df_conv[df_conv["turn_number"] < target_turn]
            user_turns_at = df_conv[
                (df_conv["turn_number"] == target_turn) & (df_conv["role"] == "user")
            ]
            if user_turns_at.empty:
                cached_qi += 1
                continue
            user_query = str(user_turns_at.iloc[0]["content"])

            context_parts = [user_query]
            recent = history.tail(6)
            for _, row in recent.iterrows():
                role = row["role"]
                content = str(row["content"])
                if role == "music" and content in track_meta_text:
                    context_parts.append(f"previously recommended: {track_meta_text[content][:100]}")
                elif role == "user" and content != user_query:
                    context_parts.append(content[:150])

            if listener_goal:
                context_parts.append(listener_goal[:150])

            query_text = " [SEP] ".join(context_parts)[:512]

            # Build hard negatives for this query
            query_hard_neg_indices = []
            if hard_neg_pools is not None and cached_qi < len(hard_neg_pools):
                pool = hard_neg_pools[cached_qi]
                pool_indices = [tid_to_idx[tid] for tid in pool if tid in tid_to_idx]
                if pool_indices:
                    sampled = random.sample(pool_indices, min(num_hard_neg, len(pool_indices)))
                    query_hard_neg_indices = sampled

            queries.append(query_text)
            target_indices.append(tid_to_idx[target_tid])
            hard_negatives.append(query_hard_neg_indices)
            cached_qi += 1

    has_hard = sum(1 for h in hard_negatives if len(h) > 0)
    print(f"Built {len(queries)} training pairs ({has_hard} with hard negatives)")
    return queries, target_indices, hard_negatives, track_meta_text


def train_epoch(
    model, tokenizer, dataloader, optimizer, track_embs,
    device, temperature=0.05, max_length=512,
    num_hard_neg=5, num_easy_neg=15,
):
    model.train()
    total_loss = 0
    num_batches = 0
    num_tracks = track_embs.shape[0]

    for batch_queries, batch_targets, batch_hard_negs in tqdm(dataloader, desc="Training"):
        bs = len(batch_targets)

        enc = tokenizer(
            batch_queries, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        q_emb = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        q_emb = F.normalize(q_emb.float(), p=2, dim=1)

        pos_embs = track_embs[batch_targets].to(device)

        # Collect all hard negatives across batch
        all_hard_indices = set()
        for hn in batch_hard_negs:
            all_hard_indices.update(hn)
        all_hard_indices -= set(batch_targets)
        all_hard_indices = list(all_hard_indices)

        # Sample easy negatives
        target_set = set(batch_targets) | set(all_hard_indices)
        easy_indices = []
        while len(easy_indices) < num_easy_neg:
            candidate = random.randint(0, num_tracks - 1)
            if candidate not in target_set:
                easy_indices.append(candidate)
                target_set.add(candidate)

        # Combine: [batch positives] + [hard negatives] + [easy negatives]
        neg_indices = all_hard_indices + easy_indices
        neg_embs = track_embs[neg_indices].to(device)
        all_doc_embs = torch.cat([pos_embs, neg_embs], dim=0)

        similarity = torch.matmul(q_emb, all_doc_embs.T) / temperature
        labels = torch.arange(bs, device=device)
        loss = F.cross_entropy(similarity, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if num_batches <= 3 or num_batches % 100 == 0:
            print(f"  batch {num_batches}: loss={loss.item():.4f}")
        if torch.isnan(loss):
            print(f"  NaN detected at batch {num_batches}! Stopping.")
            return float("nan")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_recall(model, tokenizer, val_queries, val_target_indices, track_embs, device, max_length=512):
    model.eval()
    hits = {k: 0 for k in [1, 10, 20, 100, 200]}
    total = 0
    batch_size = 64

    for i in range(0, len(val_queries), batch_size):
        batch_q = val_queries[i:i + batch_size]
        batch_targets = val_target_indices[i:i + batch_size]

        enc = tokenizer(batch_q, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        q_emb = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        q_emb = F.normalize(q_emb.float(), p=2, dim=1).cpu()

        scores = torch.matmul(q_emb, track_embs.T)

        for j in range(len(batch_q)):
            target_idx = batch_targets[j]
            _, top_indices = torch.topk(scores[j], k=min(200, scores.shape[1]))
            top_indices = top_indices.tolist()
            total += 1
            for k in hits:
                if target_idx in top_indices[:k]:
                    hits[k] += 1

    if total == 0:
        return {}
    return {f"recall@{k}": v / total for k, v in hits.items()}


def main(args):
    random.seed(42)
    torch.manual_seed(42)

    # Load frozen track embeddings
    track_ids, track_embs = load_track_embeddings(
        embedding_type=args.embedding_type,
        cache_dir=args.cache_dir,
    )

    # Load hard negatives from cached retrieval
    hard_neg_pools, cached_metadata = load_hard_negatives(
        args.cache_dir, args.last_n_turns, args.goal_filter,
    )

    # Build training data (all pairs — no split, eval on mini devset)
    train_queries, train_targets, train_hard_negs, track_meta_text = build_training_data(
        track_ids,
        hard_neg_pools,
        cached_metadata,
        max_sessions=args.max_sessions,
        last_n_turns=args.last_n_turns,
        goal_filter=args.goal_filter,
        num_hard_neg=args.num_hard_neg,
    )

    # Build mini devset eval (held-out test split, no overlap with training)
    val_queries = []
    val_targets = []
    if args.eval_session_ids:
        print(f"Building eval set from {args.eval_session_ids}...")
        eval_ids = set(json.load(open(args.eval_session_ids)))
        eval_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
        tid_to_idx = {tid: i for i, tid in enumerate(track_ids)}
        for item in eval_ds:
            if item["session_id"] not in eval_ids:
                continue
            conversations = item["conversations"]
            conversation_goal = item.get("conversation_goal") or {}
            listener_goal = str(conversation_goal.get("listener_goal", ""))
            df_conv = pd.DataFrame(conversations)
            music_turns = df_conv[df_conv["role"] == "music"].tail(1)
            for _, music_row in music_turns.iterrows():
                target_turn = int(music_row["turn_number"])
                target_tid = str(music_row["content"]).strip()
                if target_tid not in tid_to_idx:
                    continue
                history = df_conv[df_conv["turn_number"] < target_turn]
                user_turns_at = df_conv[
                    (df_conv["turn_number"] == target_turn) & (df_conv["role"] == "user")
                ]
                if user_turns_at.empty:
                    continue
                user_query = str(user_turns_at.iloc[0]["content"])
                context_parts = [user_query]
                recent = history.tail(6)
                for _, row in recent.iterrows():
                    role = row["role"]
                    content = str(row["content"])
                    if role == "music" and content in track_meta_text:
                        context_parts.append(f"previously recommended: {track_meta_text[content][:100]}")
                    elif role == "user" and content != user_query:
                        context_parts.append(content[:150])
                if listener_goal:
                    context_parts.append(listener_goal[:150])
                val_queries.append(" [SEP] ".join(context_parts)[:512])
                val_targets.append(tid_to_idx[target_tid])
        print(f"Eval: {len(val_queries)} queries from mini devset")

    print(f"Train: {len(train_queries)}, Eval: {len(val_queries)}")

    # Load query encoder
    model_name = args.model_name
    print(f"Loading query encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset = QueryTrackDataset(train_queries, train_targets, train_hard_negs)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_queries,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    output_dir = os.path.join(args.output_dir, "finetuned_biencoder")
    os.makedirs(output_dir, exist_ok=True)

    best_recall = 0.0
    for epoch in range(args.epochs):
        avg_loss = train_epoch(
            model, tokenizer, dataloader, optimizer, track_embs,
            device, temperature=args.temperature, max_length=args.max_length,
            num_hard_neg=args.num_hard_neg, num_easy_neg=args.num_easy_neg,
        )
        if avg_loss != avg_loss:  # NaN check
            print("Training diverged. Try lower learning rate or check data.")
            break
        print(f"Epoch {epoch + 1}/{args.epochs} — loss: {avg_loss:.4f}")

        if val_queries:
            recall = evaluate_recall(
                model, tokenizer, val_queries, val_targets, track_embs,
                device, max_length=args.max_length,
            )
            print(f"  Mini devset eval: {recall}")
            r200 = recall.get("recall@200", 0)
            if r200 > best_recall:
                best_recall = r200
                model.save_pretrained(os.path.join(output_dir, "best"))
                tokenizer.save_pretrained(os.path.join(output_dir, "best"))
                print(f"  Saved best model (recall@200={r200:.4f})")
        else:
            model.save_pretrained(os.path.join(output_dir, "best"))
            tokenizer.save_pretrained(os.path.join(output_dir, "best"))

    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"\nFinal model saved to {output_dir}/final")
    if best_recall > 0:
        print(f"Best model saved to {output_dir}/best (recall@20={best_recall:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding_type", type=str, default="attributes-qwen3_embedding_0.6b")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_hard_neg", type=int, default=5)
    parser.add_argument("--num_easy_neg", type=int, default=15)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--last_n_turns", type=int, default=None)
    parser.add_argument("--goal_filter", action="store_true")
    parser.add_argument("--eval_session_ids", type=str, default=None, help="Path to JSON with session IDs for eval (e.g., mini devset)")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output_dir", type=str, default="./cache")
    args = parser.parse_args()
    main(args)
