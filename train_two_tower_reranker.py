import argparse
import hashlib
import json
import os
import random
from contextlib import contextmanager, redirect_stderr, redirect_stdout, nullcontext
import io

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.retrieval_modules import load_retrieval_module
from mcrs.reranker_modules.two_tower import TWO_TOWER_RERANKER, TWO_TOWER_SCORER, TwoTowerFeatureBuilder, FeatureConfig


@contextmanager
def suppress_output():
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        yield


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_loader(model, loader, criterion, device, use_cuda_amp):
    total_loss = 0.0
    total_steps = 0
    if loader is None:
        return float("nan"), float("nan"), float("nan")
    auroc_metric = BinaryAUROC().to(device="cpu")
    aupr_metric = BinaryAveragePrecision().to(device="cpu")
    model.eval()
    with torch.inference_mode():
        for user_vecs, track_vecs, user_feats, item_feats, query_feats, pair_feats, labels in loader:
            user_vecs = user_vecs.to(device)
            track_vecs = track_vecs.to(device)
            user_feats = user_feats.to(device)
            item_feats = item_feats.to(device)
            query_feats = query_feats.to(device)
            pair_feats = pair_feats.to(device)
            labels = labels.to(device)
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_cuda_amp else nullcontext()
            with autocast_ctx:
                logits = model.score_pairs(user_vecs, track_vecs, user_feats, item_feats, query_feats, pair_feats)
                loss = criterion(logits, labels)
            total_loss += loss.item()
            total_steps += 1
            probs = torch.sigmoid(logits.detach()).float().cpu()
            target = labels.detach().float().cpu()
            auroc_metric.update(probs, target.int())
            aupr_metric.update(probs, target.int())
    try:
        auroc = float(auroc_metric.compute().item())
    except Exception:
        auroc = float("nan")
    try:
        aupr = float(aupr_metric.compute().item())
    except Exception:
        aupr = float("nan")
    return (
        total_loss / max(total_steps, 1),
        auroc,
        aupr,
    )


class PairDataset(Dataset):
    def __init__(self, examples, user_embeddings, track_embeddings, feature_builder):
        self.examples = examples
        self.user_embeddings = user_embeddings
        self.track_embeddings = track_embeddings
        self.feature_builder = feature_builder
        self.user_dim = self._infer_embedding_dim(user_embeddings)
        self.track_dim = self._infer_embedding_dim(track_embeddings)

    @staticmethod
    def _infer_embedding_dim(embeddings):
        for value in embeddings.values():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                return value.numel()
        return 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        user_id = example["user_id"]
        track_id = example["track_id"]
        label = example["label"]
        user_profile = example.get("user_profile")
        conversation_goal = example.get("conversation_goal")
        query_text = example.get("query_text")
        candidate_rank = example.get("candidate_rank", 0)
        candidate_pool_size = example.get("candidate_pool_size", 1)
        user_vec = self.user_embeddings.get(user_id)
        if user_vec is None or user_vec.numel() == 0:
            user_vec = torch.zeros(self.user_dim, dtype=torch.float32)
        track_vec = self.track_embeddings.get(track_id)
        if track_vec is None or track_vec.numel() == 0:
            track_vec = torch.zeros(self.track_dim, dtype=torch.float32)
        user_feat = self.feature_builder.build_user_features(user_profile=user_profile, user_id=user_id)
        item_feat = self.feature_builder.build_item_features(track_id)
        query_feat = self.feature_builder.build_query_features(query_text=query_text, conversation_goal=conversation_goal)
        pair_feat = self.feature_builder.build_pair_features(
            user_vec,
            track_vec,
            user_feat,
            item_feat,
            candidate_rank,
            candidate_pool_size,
            query_features=query_feat,
        )
        return (
            user_vec,
            track_vec,
            user_feat,
            item_feat,
            query_feat,
            pair_feat,
            torch.tensor(label, dtype=torch.float32),
        )


def _first_tensor_dim(mapping, name):
    for value in mapping.values():
        if value is not None and getattr(value, "numel", lambda: 0)() > 0:
            return value.numel()
    raise RuntimeError(f"No usable {name} embeddings were loaded.")


def _format_context_block(title, context, preferred_keys=None):
    if not context:
        return ""
    items = []
    if preferred_keys:
        for key in preferred_keys:
            if key in context and context[key] is not None:
                items.append(f"{key}: {context[key]}")
    else:
        for key, value in context.items():
            if value is not None:
                items.append(f"{key}: {value}")
    if not items:
        return ""
    return f"{title}:\n" + "\n".join(items)


def _format_history(history, item_db, user_db, user_id, user_profile=None, conversation_goal=None):
    parts = []
    for turn in history:
        role = turn["role"]
        content = turn["content"]
        if role == "music":
            role = "assistant"
            content = item_db.id_to_metadata(content)
        parts.append(f"{role}: {content}")
    merged_user_profile = dict(user_db.id_to_profile(user_id)) if user_id else {}
    if user_profile:
        merged_user_profile.update(user_profile)
    profile_block = _format_context_block(
        "user_profile",
        merged_user_profile,
        preferred_keys=[
            "user_id",
            "age",
            "age_group",
            "country_code",
            "country_name",
            "gender",
            "preferred_language",
            "preferred_musical_culture",
        ],
    )
    goal_block = _format_context_block(
        "conversation_goal",
        conversation_goal,
        preferred_keys=["category", "specificity", "listener_goal"],
    )
    if profile_block:
        parts.append(profile_block)
    if goal_block:
        parts.append(goal_block)
    query = "\n".join(parts)
    return query if query.strip() else "music recommendation"


def _pair_cache_key(
    dataset_name,
    dataset_split,
    item_db_name,
    user_db_name,
    track_split_types,
    user_split_types,
    corpus_types,
    user_embeddings_dataset_name,
    track_embeddings_dataset_name,
    embedding_type,
    retrieval_type,
    negatives_per_positive,
    hard_negative_pool_size,
    max_examples,
):
    payload = {
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "item_db_name": item_db_name,
        "user_db_name": user_db_name,
        "track_split_types": track_split_types,
        "user_split_types": user_split_types,
        "corpus_types": corpus_types,
        "user_embeddings_dataset_name": user_embeddings_dataset_name,
        "track_embeddings_dataset_name": track_embeddings_dataset_name,
        "embedding_type": embedding_type,
        "retrieval_type": retrieval_type,
        "negatives_per_positive": negatives_per_positive,
        "hard_negative_pool_size": hard_negative_pool_size,
        "max_examples": max_examples,
        "cache_version": 2,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return digest


def build_examples(
    dataset_name,
    dataset_split,
    item_db_name,
    user_db_name,
    track_split_types,
    user_split_types,
    corpus_types,
    user_embeddings,
    track_embeddings,
    retrieval_type,
    cache_dir,
    retrieval_device,
    negatives_per_positive,
    hard_negative_pool_size,
    max_examples=None,
    pair_cache_dir=None,
    pair_cache_key=None,
):
    if pair_cache_dir and pair_cache_key:
        cache_path = os.path.join(pair_cache_dir, f"{pair_cache_key}.pt")
        if os.path.exists(cache_path):
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                if isinstance(cached, dict) and "examples" in cached:
                    print(f"loaded cached train pairs from {cache_path}")
                    return cached["examples"]
            except Exception as exc:
                print(f"failed to load cached train pairs from {cache_path}: {exc}")
    item_db = MusicCatalogDB(item_db_name, track_split_types, corpus_types)
    user_db = UserProfileDB(user_db_name, user_split_types)
    with suppress_output():
        retrieval = load_retrieval_module(retrieval_type, item_db_name, track_split_types, corpus_types, cache_dir, device=retrieval_device)
    track_ids = list(track_embeddings.keys())
    train_db = load_dataset(dataset_name, split=dataset_split)
    examples = []
    seen = set()
    for item in tqdm(train_db, desc="building train pairs"):
        session_id = item.get("session_id")
        user_id = item.get("user_id")
        if user_id not in user_embeddings or user_embeddings[user_id].numel() == 0:
            continue
        user_profile = item.get("user_profile")
        conversation_goal = item.get("conversation_goal")
        history = []
        for turn in item.get("conversations", []):
            if turn.get("role") == "music":
                pos_track_id = turn.get("content")
                if pos_track_id not in track_embeddings or track_embeddings[pos_track_id].numel() == 0:
                    history.append(turn)
                    continue
                key = (user_id, pos_track_id)
                if key not in seen:
                    seen.add(key)
                    examples.append({
                        "session_id": session_id,
                        "user_id": user_id,
                        "track_id": pos_track_id,
                        "label": 1.0,
                        "query_text": _format_history(
                            history,
                            item_db,
                            user_db,
                            user_id,
                            user_profile=user_profile,
                            conversation_goal=conversation_goal,
                        ),
                        "user_profile": user_profile,
                        "conversation_goal": conversation_goal,
                        "candidate_rank": 0,
                        "candidate_pool_size": 1,
                    })
                query = _format_history(
                    history,
                    item_db,
                    user_db,
                    user_id,
                    user_profile=user_profile,
                    conversation_goal=conversation_goal,
                )
                if hard_negative_pool_size > 0:
                    with suppress_output():
                        candidate_pool = retrieval.text_to_item_retrieval(query, topk=hard_negative_pool_size)
                else:
                    candidate_pool = []
                candidate_pool = [
                    track_id
                    for track_id in candidate_pool
                    if track_id != pos_track_id and track_id in track_embeddings and track_embeddings[track_id].numel() > 0
                ]
                if len(candidate_pool) < negatives_per_positive:
                    fallback_pool = [track_id for track_id in track_ids if track_id != pos_track_id and track_id in track_embeddings and track_embeddings[track_id].numel() > 0]
                    random.shuffle(fallback_pool)
                    candidate_pool.extend(fallback_pool)
                candidate_pool = list(dict.fromkeys(candidate_pool))
                random.shuffle(candidate_pool)
                candidate_pool_size = max(len(candidate_pool), 1)
                pos_rank = candidate_pool.index(pos_track_id) if pos_track_id in candidate_pool else candidate_pool_size - 1
                for rank, neg_track_id in enumerate(candidate_pool[:negatives_per_positive]):
                    examples.append({
                        "session_id": session_id,
                        "user_id": user_id,
                        "track_id": neg_track_id,
                        "label": 0.0,
                        "query_text": query,
                        "user_profile": user_profile,
                        "conversation_goal": conversation_goal,
                        "candidate_rank": rank,
                        "candidate_pool_size": candidate_pool_size,
                    })
                examples.append({
                    "session_id": session_id,
                    "user_id": user_id,
                    "track_id": pos_track_id,
                    "label": 1.0,
                    "query_text": query,
                    "user_profile": user_profile,
                    "conversation_goal": conversation_goal,
                    "candidate_rank": pos_rank,
                    "candidate_pool_size": candidate_pool_size,
                })
                if max_examples and len(examples) >= max_examples:
                    retrieval.cleanup()
                    if pair_cache_dir and pair_cache_key:
                        os.makedirs(pair_cache_dir, exist_ok=True)
                        torch.save(
                            {
                                "examples": examples[:max_examples],
                                "pair_cache_key": pair_cache_key,
                                "max_examples": max_examples,
                            },
                            cache_path,
                        )
                        print(f"saved cached train pairs to {cache_path}")
                    return examples[:max_examples]
            history.append(turn)
    retrieval.cleanup()
    if pair_cache_dir and pair_cache_key:
        os.makedirs(pair_cache_dir, exist_ok=True)
        torch.save(
            {
                "examples": examples,
                "pair_cache_key": pair_cache_key,
                "max_examples": max_examples,
            },
            cache_path,
        )
        print(f"saved cached train pairs to {cache_path}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Train the two-tower reranker on challenge embeddings.")
    parser.add_argument("--dataset_name", type=str, default="talkpl-ai/TalkPlayData-Challenge-Dataset")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--valid_split", type=str, default="test")
    parser.add_argument("--item_db_name", type=str, default="talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    parser.add_argument("--user_db_name", type=str, default="talkpl-ai/TalkPlayData-Challenge-User-Metadata")
    parser.add_argument("--track_split_types", nargs="+", default=["all_tracks"])
    parser.add_argument("--user_split_types", nargs="+", default=["all_users"])
    parser.add_argument("--corpus_types", nargs="+", default=["track_name", "artist_name", "album_name", "release_date"])
    parser.add_argument("--user_embeddings_dataset_name", type=str, default="talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    parser.add_argument("--track_embeddings_dataset_name", type=str, default="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    parser.add_argument("--embedding_type", type=str, default="cf-bpr")
    parser.add_argument("--retrieval_type", type=str, default="bm25")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--pair_cache_dir", type=str, default=os.environ.get("MCRS_PAIR_CACHE_DIR", "./cache/pair_cache"))
    parser.add_argument("--retrieval_device", type=str, default="cpu")
    parser.add_argument("--output_path", type=str, default="./cache/two_tower_reranker.pt")
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--tower_hidden_dim", type=int, default=512)
    parser.add_argument("--deep_hidden_dim", type=int, default=512)
    parser.add_argument("--dcn_low_rank_dim", type=int, default=32)
    parser.add_argument("--dcn_num_layers", type=int, default=3)
    parser.add_argument("--dcn_num_experts", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--negatives_per_positive", type=int, default=5)
    parser.add_argument("--hard_negative_pool_size", type=int, default=100)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    pair_cache_key = _pair_cache_key(
        args.dataset_name,
        args.train_split,
        args.item_db_name,
        args.user_db_name,
        args.track_split_types,
        args.user_split_types,
        args.corpus_types,
        args.user_embeddings_dataset_name,
        args.track_embeddings_dataset_name,
        args.embedding_type,
        args.retrieval_type,
        args.negatives_per_positive,
        args.hard_negative_pool_size,
        args.max_examples,
    )
    base_reranker = TWO_TOWER_RERANKER(
        user_embeddings_dataset_name=args.user_embeddings_dataset_name,
        track_embeddings_dataset_name=args.track_embeddings_dataset_name,
        item_db_name=args.item_db_name,
        user_db_name=args.user_db_name,
        track_split_types=args.track_split_types,
        user_split_types=args.user_split_types,
        corpus_types=args.corpus_types,
        embedding_type=args.embedding_type,
        checkpoint_path=args.output_path,
        device="cpu",
        projection_dim=args.projection_dim,
        tower_hidden_dim=args.tower_hidden_dim,
        deep_hidden_dim=args.deep_hidden_dim,
        dcn_low_rank_dim=args.dcn_low_rank_dim,
        dcn_num_layers=args.dcn_num_layers,
        dcn_num_experts=args.dcn_num_experts,
        dropout=args.dropout,
        temperature=args.temperature,
    )
    examples = build_examples(
        args.dataset_name,
        args.train_split,
        args.item_db_name,
        args.user_db_name,
        args.track_split_types,
        args.user_split_types,
        args.corpus_types,
        base_reranker.user_embeddings,
        base_reranker.track_embeddings,
        args.retrieval_type,
        args.cache_dir,
        args.retrieval_device,
        args.negatives_per_positive,
        args.hard_negative_pool_size,
        max_examples=args.max_examples,
        pair_cache_dir=args.pair_cache_dir,
        pair_cache_key=pair_cache_key,
    )
    if not examples:
        raise RuntimeError("No training examples were built.")

    train_examples = examples
    valid_examples = []
    if args.valid_split and args.valid_split != args.train_split:
        valid_pair_cache_key = _pair_cache_key(
            args.dataset_name,
            args.valid_split,
            args.item_db_name,
            args.user_db_name,
            args.track_split_types,
            args.user_split_types,
            args.corpus_types,
            args.user_embeddings_dataset_name,
            args.track_embeddings_dataset_name,
            args.embedding_type,
            args.retrieval_type,
            args.negatives_per_positive,
            args.hard_negative_pool_size,
            args.max_examples,
        )
        valid_examples = build_examples(
            args.dataset_name,
            args.valid_split,
            args.item_db_name,
            args.user_db_name,
            args.track_split_types,
            args.user_split_types,
            args.corpus_types,
            base_reranker.user_embeddings,
            base_reranker.track_embeddings,
            args.retrieval_type,
            args.cache_dir,
            args.retrieval_device,
            args.negatives_per_positive,
            args.hard_negative_pool_size,
            max_examples=args.max_examples,
            pair_cache_dir=args.pair_cache_dir,
            pair_cache_key=valid_pair_cache_key,
        )
    elif len(examples) > 1:
        session_ids = list({example["session_id"] for example in examples})
        random.shuffle(session_ids)
        split_idx = max(1, int(len(session_ids) * 0.95))
        train_session_ids = set(session_ids[:split_idx])
        valid_session_ids = set(session_ids[split_idx:])
        train_examples = [example for example in examples if example["session_id"] in train_session_ids]
        valid_examples = [example for example in examples if example["session_id"] in valid_session_ids]

    train_dataset = PairDataset(train_examples, base_reranker.user_embeddings, base_reranker.track_embeddings, base_reranker.feature_builder)
    valid_dataset = PairDataset(valid_examples, base_reranker.user_embeddings, base_reranker.track_embeddings, base_reranker.feature_builder) if valid_examples else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    ) if valid_dataset else None

    user_dim = _first_tensor_dim(base_reranker.user_embeddings, "user")
    track_dim = _first_tensor_dim(base_reranker.track_embeddings, "track")
    model = TWO_TOWER_SCORER(
        user_dim=user_dim,
        track_dim=track_dim,
        user_feature_dim=base_reranker.feature_builder.user_feature_dim,
        item_feature_dim=base_reranker.feature_builder.item_feature_dim,
        query_feature_dim=base_reranker.feature_builder.query_feature_dim,
        pair_feature_dim=base_reranker.feature_builder.pair_feature_dim,
        projection_dim=args.projection_dim,
        tower_hidden_dim=args.tower_hidden_dim,
        deep_hidden_dim=args.deep_hidden_dim,
        dcn_low_rank_dim=args.dcn_low_rank_dim,
        dcn_num_layers=args.dcn_num_layers,
        dcn_num_experts=args.dcn_num_experts,
        dropout=args.dropout,
        temperature=args.temperature,
    )
    device = resolve_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    use_cuda_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)
    grad_accum = max(args.gradient_accumulation_steps, 1)

    best_valid_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step_idx, (user_vecs, track_vecs, user_feats, item_feats, query_feats, pair_feats, labels) in enumerate(tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}")):
            user_vecs = user_vecs.to(device)
            track_vecs = track_vecs.to(device)
            user_feats = user_feats.to(device)
            item_feats = item_feats.to(device)
            query_feats = query_feats.to(device)
            pair_feats = pair_feats.to(device)
            labels = labels.to(device)
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_cuda_amp else nullcontext()
            with autocast_ctx:
                logits = model.score_pairs(user_vecs, track_vecs, user_feats, item_feats, query_feats, pair_feats)
                loss = criterion(logits, labels) / grad_accum
            scaler.scale(loss).backward()
            if (step_idx + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if len(train_loader) % grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss, train_auroc, train_aupr = evaluate_loader(model, train_loader, criterion, device, use_cuda_amp)
        valid_loss, valid_auroc, valid_aupr = evaluate_loader(model, valid_loader, criterion, device, use_cuda_amp)
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.6f} train_auroc={train_auroc:.6f} train_aupr={train_aupr:.6f} "
            f"valid_loss={valid_loss:.6f} valid_auroc={valid_auroc:.6f} valid_aupr={valid_aupr:.6f}"
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "user_dim": user_dim,
                "track_dim": track_dim,
                "projection_dim": args.projection_dim,
                "tower_hidden_dim": args.tower_hidden_dim,
                "dropout": args.dropout,
                "temperature": args.temperature,
                "user_feature_dim": base_reranker.feature_builder.user_feature_dim,
                "item_feature_dim": base_reranker.feature_builder.item_feature_dim,
                "query_feature_dim": base_reranker.feature_builder.query_feature_dim,
                "pair_feature_dim": base_reranker.feature_builder.pair_feature_dim,
                "deep_hidden_dim": args.deep_hidden_dim,
                "dcn_low_rank_dim": args.dcn_low_rank_dim,
                "dcn_num_layers": args.dcn_num_layers,
                "dcn_num_experts": args.dcn_num_experts,
                "embedding_type": args.embedding_type,
                "dataset_name": args.dataset_name,
            }

    if best_state is None:
        best_state = {
            "model_state_dict": model.state_dict(),
            "user_dim": user_dim,
            "track_dim": track_dim,
            "projection_dim": args.projection_dim,
            "tower_hidden_dim": args.tower_hidden_dim,
            "dropout": args.dropout,
            "temperature": args.temperature,
            "user_feature_dim": base_reranker.feature_builder.user_feature_dim,
            "item_feature_dim": base_reranker.feature_builder.item_feature_dim,
            "query_feature_dim": base_reranker.feature_builder.query_feature_dim,
            "pair_feature_dim": base_reranker.feature_builder.pair_feature_dim,
            "deep_hidden_dim": args.deep_hidden_dim,
            "dcn_low_rank_dim": args.dcn_low_rank_dim,
            "dcn_num_layers": args.dcn_num_layers,
            "dcn_num_experts": args.dcn_num_experts,
            "embedding_type": args.embedding_type,
            "dataset_name": args.dataset_name,
        }

    torch.save(best_state, args.output_path)
    print(f"saved checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
