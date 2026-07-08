"""Item-to-item retrieval over precomputed track embeddings."""

from __future__ import annotations

from typing import List

import torch
from datasets import concatenate_datasets, load_dataset

from mcrs.db_item.music_catalog import normalize_entity_id


class ITEM_TO_ITEM_MODEL:
    def __init__(
        self,
        track_embeddings_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        embedding_type: str = "attributes-qwen3_embedding_0.6b",
        preloaded_embeddings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.embedding_type = embedding_type
        if preloaded_embeddings is not None:
            self.track_embeddings = preloaded_embeddings
        else:
            self.track_embeddings = self._load_embeddings(track_embeddings_dataset_name)
        self.track_ids = list(self.track_embeddings.keys())
        self.track_matrix = (
            torch.stack([self.track_embeddings[tid] for tid in self.track_ids], dim=0)
            if self.track_ids
            else torch.empty(0, 0)
        )

    def _load_embeddings(self, dataset_name: str) -> dict[str, torch.Tensor]:
        dataset = load_dataset(dataset_name)
        splits = [dataset[s] for s in dataset.keys()]
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
            embeddings[normalize_entity_id(row["track_id"])] = tensor
        return embeddings

    @staticmethod
    def load_multiple(
        embedding_types: list[str],
        dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
    ) -> dict[str, "ITEM_TO_ITEM_MODEL"]:
        dataset = load_dataset(dataset_name)
        splits = [dataset[s] for s in dataset.keys()]
        merged = concatenate_datasets(splits) if len(splits) > 1 else splits[0]
        all_embs: dict[str, dict[str, torch.Tensor]] = {et: {} for et in embedding_types}
        for row in merged:
            tid = normalize_entity_id(row["track_id"])
            for et in embedding_types:
                vec = row.get(et)
                if vec is None:
                    continue
                tensor = torch.tensor(vec, dtype=torch.float32)
                if tensor.numel() == 0:
                    continue
                norm = torch.linalg.norm(tensor)
                if norm > 0:
                    tensor = tensor / norm
                all_embs[et][tid] = tensor
        models = {}
        for et in embedding_types:
            models[et] = ITEM_TO_ITEM_MODEL(
                embedding_type=et,
                preloaded_embeddings=all_embs[et],
            )
            print(f"[i2i] loaded {et}: {len(all_embs[et])} tracks")
        return models

    def has_track_embedding(self, track_id: str) -> bool:
        return normalize_entity_id(track_id) in self.track_embeddings

    def retrieve_similar(self, anchor_track_ids: list[str], topk: int) -> list[str]:
        if not anchor_track_ids or not self.track_embeddings:
            return []
        vecs = []
        for tid in anchor_track_ids:
            vec = self.track_embeddings.get(normalize_entity_id(tid))
            if vec is not None:
                vecs.append(vec)
        if not vecs:
            return []
        query_vec = torch.stack(vecs, dim=0).mean(dim=0)
        norm = torch.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        scores = torch.matmul(self.track_matrix, query_vec)
        limit = min(topk, scores.shape[0])
        top_indices = torch.topk(scores, k=limit).indices.tolist()
        anchor_set = {normalize_entity_id(tid) for tid in anchor_track_ids}
        return [self.track_ids[i] for i in top_indices if self.track_ids[i] not in anchor_set][:topk]

    def batch_retrieve_similar(self, anchor_lists: list[list[str]], topk: int) -> list[list[str]]:
        if not self.track_embeddings:
            return [[] for _ in anchor_lists]
        valid_indices: list[int] = []
        query_vecs: list[torch.Tensor] = []
        anchor_sets: list[set[str]] = []
        for i, anchors in enumerate(anchor_lists):
            if not anchors:
                continue
            vecs = [self.track_embeddings[normalize_entity_id(tid)]
                    for tid in anchors
                    if normalize_entity_id(tid) in self.track_embeddings]
            if not vecs:
                continue
            qv = torch.stack(vecs, dim=0).mean(dim=0)
            norm = torch.linalg.norm(qv)
            if norm > 0:
                qv = qv / norm
            valid_indices.append(i)
            query_vecs.append(qv)
            anchor_sets.append({normalize_entity_id(tid) for tid in anchors})
        results: list[list[str]] = [[] for _ in anchor_lists]
        if not query_vecs:
            return results
        query_matrix = torch.stack(query_vecs, dim=0)
        all_scores = torch.matmul(self.track_matrix, query_matrix.T)
        limit = min(topk, all_scores.shape[0])
        for pos, idx in enumerate(valid_indices):
            top_indices = torch.topk(all_scores[:, pos], k=limit).indices.tolist()
            results[idx] = [self.track_ids[i] for i in top_indices if self.track_ids[i] not in anchor_sets[pos]][:topk]
        return results

    def cleanup(self) -> None:
        if hasattr(self, "track_embeddings"):
            del self.track_embeddings
        if hasattr(self, "track_matrix"):
            del self.track_matrix
        if hasattr(self, "track_ids"):
            del self.track_ids
