from .bm25 import BM25_MODEL
from .bert import BERT_MODEL
from .multi_source import MULTI_SOURCE_MODEL

def load_retrieval_module(
        retrieval_type: str,
        dataset_name: str,
        track_split_types: list[str],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
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
        bpr_weight: float = 0.2,
    ):
    if retrieval_type == "bm25":
        return BM25_MODEL(dataset_name, track_split_types, corpus_types, cache_dir, field_weights=field_weights)
    elif retrieval_type == "bert":
        return BERT_MODEL(
            dataset_name,
            track_split_types,
            corpus_types,
            cache_dir,
            model_name=dense_model_name,
            query_prefix=dense_query_prefix,
            doc_prefix=dense_doc_prefix,
            device=device,
        )
    elif retrieval_type == "hybrid":
        from .hybrid import HYBRID_MODEL
        return HYBRID_MODEL(
            dataset_name,
            track_split_types,
            corpus_types,
            cache_dir,
            device=device,
            field_weights=field_weights,
            dense_model_name=dense_model_name,
            dense_query_prefix=dense_query_prefix,
            dense_doc_prefix=dense_doc_prefix,
            bm25_topk=bm25_topk,
            bert_topk=bert_topk,
            final_topk=final_topk,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            bert_weight=bert_weight,
        )
    elif retrieval_type == "multi_source":
        return MULTI_SOURCE_MODEL(
            dataset_name,
            track_split_types,
            corpus_types,
            cache_dir,
            device=device,
            field_weights=field_weights,
            dense_model_name=dense_model_name,
            dense_query_prefix=dense_query_prefix,
            dense_doc_prefix=dense_doc_prefix,
            bm25_topk=bm25_topk,
            bert_topk=bert_topk,
            final_topk=final_topk,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            bert_weight=bert_weight,
            bpr_weight=bpr_weight,
        )
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
