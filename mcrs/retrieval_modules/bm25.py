"""BM25-based retrieval utilities for music track metadata.

This module builds and queries a BM25 index over track metadata fields
loaded from a Hugging Face dataset. The index is cached to disk for
subsequent reuse.
"""
import os
import json
import numpy as np
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import bm25s
from datasets import load_dataset, concatenate_datasets
from mcrs.db_item.music_catalog import format_metadata_value


@contextmanager
def suppress_output():
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        yield


class BM25_MODEL:
    """BM25 retriever over track metadata.
    Builds an index from specified corpus fields (e.g., `track_name`, `artist_name`,`album_name`) and provides text-to-item retrieval.
    """
    def __init__(self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        field_weights: dict[str, int] | None = None,
        metadata_dict: dict[str, dict] | None = None,
    ) -> None:
        """Initialize the BM25 retriever.
        Args:
            dataset_name: Hugging Face dataset name containing track metadata.
            split_types: Dataset splits to load and concatenate.
            corpus_types: Metadata fields to include in the text corpus.
            cache_dir: Directory to cache the BM25 index and artifacts.
            metadata_dict: Pre-loaded metadata dict to avoid reloading.
        """
        self.dataset_name = dataset_name
        self.split_types = split_types
        self.corpus_types = corpus_types
        self.field_weights = field_weights or {
            "track_name": 1,
            "artist_name": 1,
            "album_name": 1,
            "release_date": 1,
            "tag_list": 1,
        }
        weight_signature = "_".join(f"{key}{self.field_weights.get(key, 1)}" for key in sorted(self.field_weights))
        self.corpus_name = "_".join(corpus_types) + f"__{weight_signature}"
        self.cache_dir = cache_dir
        self.metadata_dict = metadata_dict if metadata_dict is not None else self._load_corpus()
        if os.path.exists(f"{self.cache_dir}/bm25/{self.corpus_name}"):
            self.bm25_model, self.track_ids = self._load_bm25(self.corpus_name)
        else:
            self.build_index()
            self.bm25_model, self.track_ids = self._load_bm25(self.corpus_name)
        self.track_id_to_index = {track_id: idx for idx, track_id in enumerate(self.track_ids)}

    def _load_bm25(self, corpus_name: str) -> tuple[bm25s.BM25, list[str]]:
        """Load a cached BM25 index and track id list.
        Args:
            corpus_name: Name of the corpus subdirectory under the cache.
        Returns:
            A tuple of (bm25_model, track_ids).
        """
        bm25 = bm25s.BM25.load(f"{self.cache_dir}/bm25/{corpus_name}", load_corpus=True)
        track_ids = json.load(open(f"{self.cache_dir}/bm25/{corpus_name}/track_ids.json", "r"))
        return bm25, track_ids

    def _load_corpus(self) -> dict[str, dict]:
        """Load and combine metadata splits from the configured dataset.
        Returns:
            A mapping from `track_id` to its metadata dictionary.
        """
        metadata_dataset = load_dataset(self.dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[split_type] for split_type in self.split_types])
        metadata_dict = {item["track_id"]: item for item in metadata_concat_dataset}
        return metadata_dict

    def _stringify_metadata(self, metadata: dict[str, object]) -> str:
        """Convert a metadata dict into a multi-line string for indexing.
        Args:
            metadata: Track metadata with fields listed in `self.corpus_types`.
        Returns:
            A newline-separated string with `field: value` per selected field.
        """
        metadata_str = ""
        for corpus_type in self.corpus_types:
            entity = format_metadata_value(metadata.get(corpus_type, "")).lower()
            repeat_count = self.field_weights.get(corpus_type, 1)
            repeat_count = max(int(repeat_count), 1)
            for _ in range(repeat_count):
                metadata_str += f"{corpus_type}: {entity}\n"
        return metadata_str

    def build_index(self) -> None:
        """Build and persist a BM25 index over the loaded corpus.
        """
        track_ids = list(self.metadata_dict.keys())
        corpus = []
        for track_id in track_ids:
            metadata = self.metadata_dict[track_id]
            metadata_str = self._stringify_metadata(metadata)
            corpus.append(metadata_str)
        with suppress_output():
            corpus_tokens = bm25s.tokenize(corpus)
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)
        os.makedirs(os.path.join(self.cache_dir, "bm25", self.corpus_name), exist_ok=True)
        retriever.save(f"{self.cache_dir}/bm25/{self.corpus_name}", corpus=corpus)
        with open(os.path.join(self.cache_dir, "bm25", self.corpus_name, "track_ids.json"), "w") as f:
            json.dump(track_ids, f, indent=2)

    def _build_weight_mask(self, allowed_track_ids: list[str] | None):
        if not allowed_track_ids:
            return None
        mask = np.zeros(len(self.track_ids), dtype=np.float32)
        for track_id in allowed_track_ids:
            idx = self.track_id_to_index.get(track_id)
            if idx is not None:
                mask[idx] = 1.0
        if mask.sum() <= 0:
            return None
        return mask

    def text_to_item_retrieval(self, query: str, topk: int, allowed_track_ids: list[str] | None = None) -> list[str]:
        """Retrieve top-k track IDs for a natural language query.
        Args:
            query: The user text query to match against the metadata corpus.
            topk: Number of items to retrieve.
            allowed_track_ids: Optional candidate pool restriction.
        Returns:
            A list of track IDs ordered by decreasing BM25 score.
        """
        with suppress_output():
            query_tokens = bm25s.tokenize([query.lower()])
            weight_mask = self._build_weight_mask(allowed_track_ids)
            retrieve_kwargs = {"k": topk, "return_as": "tuple"}
            if weight_mask is not None:
                retrieve_kwargs["weight_mask"] = weight_mask
            doc_scores = self.bm25_model.retrieve(query_tokens, **retrieve_kwargs)
        bm25_results = doc_scores.documents[0]
        return [self.track_ids[item['id']] for item in bm25_results]

    def batch_text_to_item_retrieval(self, queries: list[str], topk: int) -> list[list[str]]:
        """Retrieve top-k track IDs for multiple queries in batch.
        Args:
            queries: List of user text queries to match against the metadata corpus.
            topk: Number of items to retrieve per query.
        Returns:
            A list of lists, where each inner list contains track IDs ordered by decreasing BM25 score.
        """
        with suppress_output():
            query_tokens = bm25s.tokenize([q.lower() for q in queries])
            doc_scores = self.bm25_model.retrieve(query_tokens, k=topk, return_as="tuple")
        results = []
        for i in range(len(queries)):
            bm25_results = doc_scores.documents[i]
            results.append([self.track_ids[item['id']] for item in bm25_results])
        return results

    def cleanup(self) -> None:
        if hasattr(self, "bm25_model"):
            del self.bm25_model
        if hasattr(self, "track_ids"):
            del self.track_ids
