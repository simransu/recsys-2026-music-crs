"""Hybrid retrieval over BM25 and BERT candidates using reciprocal rank fusion."""

from __future__ import annotations

from typing import List

from .bm25 import BM25_MODEL
from .bert import BERT_MODEL


class HYBRID_MODEL:
    def __init__(
        self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        device: str | None = None,
        field_weights: dict[str, int] | None = None,
        dense_model_name: str = "bert-base-uncased",
        dense_query_prefix: str = "",
        dense_doc_prefix: str = "",
        bm25_topk: int = 100,
        bert_topk: int = 100,
        final_topk: int = 20,
        rrf_k: int = 60,
        bm25_weight: float = 0.8,
        bert_weight: float = 0.2,
    ) -> None:
        self.bm25_topk = bm25_topk
        self.bert_topk = bert_topk
        self.final_topk = final_topk
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.bert_weight = bert_weight
        self.bm25 = BM25_MODEL(dataset_name, split_types, corpus_types, cache_dir, field_weights=field_weights)
        self.bert = BERT_MODEL(
            dataset_name,
            split_types,
            corpus_types,
            cache_dir,
            model_name=dense_model_name,
            query_prefix=dense_query_prefix,
            doc_prefix=dense_doc_prefix,
            device=device,
        )

    def _rrf_merge(self, bm25_items: List[str], bert_items: List[str], limit: int) -> List[str]:
        scores: dict[str, float] = {}
        for rank, item_id in enumerate(bm25_items):
            scores[item_id] = scores.get(item_id, 0.0) + self.bm25_weight / (self.rrf_k + rank + 1)
        for rank, item_id in enumerate(bert_items):
            scores[item_id] = scores.get(item_id, 0.0) + self.bert_weight / (self.rrf_k + rank + 1)
        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in ranked_items[:limit]]

    def text_to_item_retrieval(self, query: str, topk: int) -> List[str]:
        bm25_items = self.bm25.text_to_item_retrieval(query, max(topk, self.bm25_topk))
        bert_items = self.bert.text_to_item_retrieval(query, max(topk, self.bert_topk))
        return self._rrf_merge(bm25_items, bert_items, limit=topk)

    def batch_text_to_item_retrieval(self, queries: List[str], topk: int) -> List[List[str]]:
        bm25_items = self.bm25.batch_text_to_item_retrieval(queries, max(topk, self.bm25_topk))
        bert_items = self.bert.batch_text_to_item_retrieval(queries, max(topk, self.bert_topk))
        results = []
        for i in range(len(queries)):
            results.append(self._rrf_merge(bm25_items[i], bert_items[i], limit=topk))
        return results

    def cleanup(self) -> None:
        if hasattr(self, "bm25"):
            self.bm25.cleanup()
        if hasattr(self, "bert"):
            self.bert.cleanup()
