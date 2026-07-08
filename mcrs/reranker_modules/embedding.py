"""Embedding-based reranker using precomputed user and track vectors."""

from __future__ import annotations

import torch
from datasets import concatenate_datasets, load_dataset
from mcrs.db_item.music_catalog import normalize_entity_id


class EMBEDDING_RERANKER:
    def __init__(
        self,
        user_embeddings_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        track_embeddings_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        embedding_type: str = "cf-bpr",
        alpha: float = 1.0,
        beta: float = 0.15,
        rrf_k: int = 60,
    ) -> None:
        self.user_embeddings_dataset_name = user_embeddings_dataset_name
        self.track_embeddings_dataset_name = track_embeddings_dataset_name
        self.embedding_type = embedding_type
        self.alpha = alpha
        self.beta = beta
        self.rrf_k = rrf_k
        self.user_embeddings = self._load_embeddings(self.user_embeddings_dataset_name, "user_id")
        self.track_embeddings = self._load_embeddings(self.track_embeddings_dataset_name, "track_id")

    def _load_embeddings(self, dataset_name: str, id_field: str) -> dict[str, torch.Tensor]:
        dataset = load_dataset(dataset_name)
        splits = [dataset[split_name] for split_name in dataset.keys()]
        merged = concatenate_datasets(splits) if len(splits) > 1 else splits[0]
        embeddings: dict[str, torch.Tensor] = {}
        for row in merged:
            embedding = row.get(self.embedding_type)
            if embedding is None:
                continue
            tensor = torch.tensor(embedding, dtype=torch.float32)
            norm = torch.linalg.norm(tensor)
            if norm > 0:
                tensor = tensor / norm
            embeddings[normalize_entity_id(row[id_field])] = tensor
        return embeddings

    def rerank(self, user_id: str | None, candidate_track_ids: list[str], topk: int = 20) -> list[str]:
        if not candidate_track_ids:
            return []
        if not self.track_embeddings:
            return candidate_track_ids[:topk]

        with torch.inference_mode():
            user_vec = self.user_embeddings.get(normalize_entity_id(user_id)) if user_id else None
            if user_vec is None:
                user_vec = torch.zeros_like(next(iter(self.track_embeddings.values())))

            scores = []
            for rank, track_id in enumerate(candidate_track_ids):
                track_vec = self.track_embeddings.get(normalize_entity_id(track_id))
                if track_vec is None:
                    track_score = -1.0
                else:
                    track_score = torch.dot(user_vec, track_vec).item()
                rank_bonus = 1.0 / (self.rrf_k + rank + 1)
                scores.append(self.alpha * track_score + self.beta * rank_bonus)

        order = sorted(range(len(candidate_track_ids)), key=lambda idx: scores[idx], reverse=True)
        topk = min(topk, len(candidate_track_ids))
        return [candidate_track_ids[idx] for idx in order[:topk]]

    def batch_rerank(self, user_ids: list[str | None], candidate_track_id_batches: list[list[str]], topk: int = 20) -> list[list[str]]:
        return [
            self.rerank(user_id, candidate_track_ids, topk=topk)
            for user_id, candidate_track_ids in zip(user_ids, candidate_track_id_batches)
        ]

    def cleanup(self) -> None:
        if hasattr(self, "user_embeddings"):
            del self.user_embeddings
        if hasattr(self, "track_embeddings"):
            del self.track_embeddings
