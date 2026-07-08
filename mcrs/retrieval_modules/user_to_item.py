"""User-to-item retrieval over precomputed CF-BPR embeddings."""

from __future__ import annotations

from typing import List

import torch
from datasets import concatenate_datasets, load_dataset

from mcrs.db_item.music_catalog import normalize_entity_id


class USER_TO_ITEM_MODEL:
    def __init__(
        self,
        user_embeddings_dataset_name: str,
        track_embeddings_dataset_name: str,
        item_db_name: str,
        user_db_name: str,
        track_split_types: list[str],
        user_split_types: list[str],
        corpus_types: list[str],
        embedding_type: str = "cf-bpr",
    ) -> None:
        self.user_embeddings_dataset_name = user_embeddings_dataset_name
        self.track_embeddings_dataset_name = track_embeddings_dataset_name
        self.item_db_name = item_db_name
        self.user_db_name = user_db_name
        self.track_split_types = track_split_types
        self.user_split_types = user_split_types
        self.corpus_types = corpus_types
        self.embedding_type = embedding_type
        self.user_embeddings = self._load_embeddings(self.user_embeddings_dataset_name, "user_id")
        self.track_embeddings = self._load_embeddings(self.track_embeddings_dataset_name, "track_id")
        self.track_ids = list(self.track_embeddings.keys())
        self.track_matrix = torch.stack([self.track_embeddings[track_id] for track_id in self.track_ids], dim=0) if self.track_ids else torch.empty(0, 0)

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
            if tensor.numel() == 0:
                continue
            norm = torch.linalg.norm(tensor)
            if norm > 0:
                tensor = tensor / norm
            embeddings[normalize_entity_id(row[id_field])] = tensor
        return embeddings

    def has_user_embedding(self, user_id: str | None) -> bool:
        if not user_id:
            return False
        return normalize_entity_id(user_id) in self.user_embeddings

    def text_to_item_retrieval(self, user_id: str | None, topk: int) -> List[str]:
        if not user_id:
            return []
        user_vec = self.user_embeddings.get(normalize_entity_id(user_id))
        if user_vec is None or not self.track_embeddings:
            return []
        scores = torch.matmul(self.track_matrix, user_vec)
        topk = min(topk, scores.shape[0])
        top_indices = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids[i] for i in top_indices]

    def batch_user_to_item_retrieval(self, user_ids: List[str | None], topk: int) -> List[List[str]]:
        if not self.track_embeddings:
            return [[] for _ in user_ids]
        valid_indices: list[int] = []
        valid_vecs: list[torch.Tensor] = []
        for i, user_id in enumerate(user_ids):
            if not user_id:
                continue
            vec = self.user_embeddings.get(normalize_entity_id(user_id))
            if vec is not None:
                valid_indices.append(i)
                valid_vecs.append(vec)
        results: List[List[str]] = [[] for _ in user_ids]
        if not valid_vecs:
            return results
        user_matrix = torch.stack(valid_vecs, dim=0)
        all_scores = torch.matmul(self.track_matrix, user_matrix.T)
        limit = min(topk, all_scores.shape[0])
        for pos, idx in enumerate(valid_indices):
            top_indices = torch.topk(all_scores[:, pos], k=limit).indices.tolist()
            results[idx] = [self.track_ids[i] for i in top_indices]
        return results

    def cleanup(self) -> None:
        if hasattr(self, "user_embeddings"):
            del self.user_embeddings
        if hasattr(self, "track_embeddings"):
            del self.track_embeddings
        if hasattr(self, "track_ids"):
            del self.track_ids
