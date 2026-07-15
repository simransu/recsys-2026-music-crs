def load_reranker_module(
        reranker_type: str,
        user_embeddings_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        track_embeddings_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
        track_split_types: list[str] | None = None,
        user_split_types: list[str] | None = None,
        corpus_types: list[str] | None = None,
        embedding_type: str = "cf-bpr",
        checkpoint_path: str = "./cache/two_tower_reranker.pt",
        device: str | None = None,
        projection_dim: int = 128,
        tower_hidden_dim: int = 256,
        deep_hidden_dim: int = 256,
        dcn_low_rank_dim: int = 32,
        dcn_num_layers: int = 3,
        dcn_num_experts: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.07,
        alpha: float = 1.0,
        beta: float = 0.15,
        rrf_k: int = 60,
    ):
    if reranker_type == "embedding":
        from .embedding import EMBEDDING_RERANKER
        return EMBEDDING_RERANKER(
            user_embeddings_dataset_name=user_embeddings_dataset_name,
            track_embeddings_dataset_name=track_embeddings_dataset_name,
            embedding_type=embedding_type,
            alpha=alpha,
            beta=beta,
            rrf_k=rrf_k,
        )
    else:
        raise ValueError(f"Unsupported reranker type: {reranker_type}")
