"""Multi-source retrieval: BM25 + dense + BPR (warm users), fused via RRF."""

from __future__ import annotations

from typing import List

from .bm25 import BM25_MODEL
from .bert import BERT_MODEL


class MULTI_SOURCE_MODEL:
    def __init__(
        self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        device: str | None = None,
        field_weights: dict[str, int] | None = None,
        dense_model_name: str = "BAAI/bge-small-en-v1.5",
        dense_query_prefix: str = "",
        dense_doc_prefix: str = "",
        bm25_topk: int = 100,
        bert_topk: int = 100,
        final_topk: int = 100,
        rrf_k: int = 60,
        bm25_weight: float = 0.5,
        bert_weight: float = 0.3,
        bpr_weight: float = 0.2,
    ) -> None:
        self.bm25_topk = bm25_topk
        self.bert_topk = bert_topk
        self.final_topk = final_topk
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.bert_weight = bert_weight
        self.bpr_weight = bpr_weight
        self.bm25 = BM25_MODEL(dataset_name, split_types, corpus_types, cache_dir, field_weights=field_weights)
        shared_metadata = self.bm25.metadata_dict
        self.bert = BERT_MODEL(
            dataset_name,
            split_types,
            corpus_types,
            cache_dir,
            model_name=dense_model_name,
            query_prefix=dense_query_prefix,
            doc_prefix=dense_doc_prefix,
            device=device,
            metadata_dict=shared_metadata,
        )

    def _rrf_merge(self, ranked_lists: list[tuple[list[str], float]], limit: int) -> list[str]:
        scores: dict[str, float] = {}
        for items, weight in ranked_lists:
            for rank, item_id in enumerate(items):
                scores[item_id] = scores.get(item_id, 0.0) + weight / (self.rrf_k + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in ranked[:limit]]

    def text_to_item_retrieval(
        self,
        query: str,
        topk: int,
        bpr_items: list[str] | None = None,
        i2i_items: list[str] | None = None,
        i2i_weight: float = 0.15,
        allowed_track_ids: list[str] | None = None,
    ) -> list[str]:
        bm25_items = self.bm25.text_to_item_retrieval(query, self.bm25_topk, allowed_track_ids=allowed_track_ids)
        bert_items = self.bert.text_to_item_retrieval(query, self.bert_topk)
        sources: list[tuple[list[str], float]] = [
            (bm25_items, self.bm25_weight),
            (bert_items, self.bert_weight),
        ]
        if bpr_items:
            sources.append((bpr_items, self.bpr_weight))
        if i2i_items:
            sources.append((i2i_items, i2i_weight))
        return self._rrf_merge(sources, limit=topk)

    def batch_text_to_item_retrieval(
        self,
        queries: list[str],
        topk: int,
        bpr_items_list: list[list[str] | None] | None = None,
        i2i_items_list: list[list[str] | None] | None = None,
        i2i_weights: list[float] | float = 0.15,
        allowed_track_ids_list: list[list[str] | None] | None = None,
        dense_queries: list[str] | None = None,
        bm25_weight_overrides: list[float] | None = None,
        bert_weight_overrides: list[float] | None = None,
    ) -> list[list[str]]:
        if allowed_track_ids_list and any(pool for pool in allowed_track_ids_list):
            bm25_items = [
                self.bm25.text_to_item_retrieval(q, self.bm25_topk, allowed_track_ids=pool)
                for q, pool in zip(queries, allowed_track_ids_list)
            ]
        else:
            bm25_items = self.bm25.batch_text_to_item_retrieval(queries, self.bm25_topk)
        bert_queries = dense_queries if dense_queries is not None else queries
        bert_items = self.bert.batch_text_to_item_retrieval(bert_queries, self.bert_topk)
        if bpr_items_list is None:
            bpr_items_list = [None] * len(queries)
        if i2i_items_list is None:
            i2i_items_list = [None] * len(queries)
        if isinstance(i2i_weights, (int, float)):
            i2i_weights = [float(i2i_weights)] * len(queries)
        results = []
        for i in range(len(queries)):
            bm25_w = bm25_weight_overrides[i] if bm25_weight_overrides else self.bm25_weight
            bert_w = bert_weight_overrides[i] if bert_weight_overrides else self.bert_weight
            sources: list[tuple[list[str], float]] = [
                (bm25_items[i], bm25_w),
                (bert_items[i], bert_w),
            ]
            if bpr_items_list[i]:
                sources.append((bpr_items_list[i], self.bpr_weight))
            if i2i_items_list[i]:
                sources.append((i2i_items_list[i], i2i_weights[i]))
            results.append(self._rrf_merge(sources, limit=topk))
        return results

    def cleanup(self) -> None:
        if hasattr(self, "bm25"):
            self.bm25.cleanup()
        if hasattr(self, "bert"):
            self.bert.cleanup()
