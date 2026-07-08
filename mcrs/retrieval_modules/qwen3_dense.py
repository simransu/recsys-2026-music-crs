"""Dense retrieval using Qwen3-Embedding-0.6B for query encoding against
precomputed track embeddings from the dataset.

The track embeddings (attributes-qwen3_embedding_0.6b) are loaded from
HuggingFace and cached. Queries are encoded at runtime with the same model
using instruction-prefixed last-token pooling, matching the encoding used
to produce the precomputed embeddings.
"""

import os
import json

import torch
import torch.nn.functional as F
from torch import Tensor
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


TASK_INSTRUCTION = (
    "Given a music recommendation conversation, "
    "retrieve tracks that match the listener's request"
)


class Qwen3DenseRetriever:
    """Dense retriever using Qwen3-Embedding-0.6B queries against precomputed track embeddings."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        track_embedding_dataset: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        embedding_types: list[str] | None = None,
        embedding_weights: dict[str, float] | None = None,
        cache_dir: str = "./cache",
        device: str | None = None,
        max_length: int = 512,
        qwen3_embedding_query_batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.track_embedding_dataset = track_embedding_dataset
        self.embedding_types = embedding_types or ["attributes-qwen3_embedding_0.6b"]
        self.embedding_weights = embedding_weights or {et: 1.0 for et in self.embedding_types}
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.query_batch_size = qwen3_embedding_query_batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._log(f"Loading model {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model.to(self.device).eval()

        self._load_track_embeddings()

    @staticmethod
    def _log(msg: str) -> None:
        print(f"[Qwen3Dense] {msg}", flush=True)

    def _load_track_embeddings(self) -> None:
        """Load precomputed track embeddings from HuggingFace dataset, with caching."""
        self.track_ids: list[str] = []
        self.embedding_matrices: dict[str, torch.Tensor] = {}

        index_dir = os.path.join(self.cache_dir, "qwen3_dense")

        all_cached = True
        ids_path = os.path.join(index_dir, "track_ids.json")
        for et in self.embedding_types:
            cache_path = os.path.join(index_dir, f"{et}.pt")
            if not os.path.exists(cache_path) or not os.path.exists(ids_path):
                all_cached = False
                break

        if all_cached:
            self.track_ids = json.load(open(ids_path))
            for et in self.embedding_types:
                cache_path = os.path.join(index_dir, f"{et}.pt")
                self._log(f"Loading cached {et} embeddings")
                self.embedding_matrices[et] = torch.load(cache_path, map_location="cpu")
        else:
            self._log("Loading embeddings from dataset (first run)...")
            ds = load_dataset(self.track_embedding_dataset)
            merged = concatenate_datasets([ds[s] for s in ds.keys()])

            track_ids = []
            embeddings_by_type: dict[str, list[torch.Tensor]] = {
                t: [] for t in self.embedding_types
            }

            for row in merged:
                tid = str(row["track_id"]).strip()
                # Only include tracks that have the primary embedding type
                vec = row.get(self.embedding_types[0])
                if vec is None or len(vec) == 0:
                    continue

                track_ids.append(tid)
                for t in self.embedding_types:
                    v = row.get(t)
                    if v is not None and len(v) > 0:
                        tensor = torch.tensor(v, dtype=torch.float32)
                        tensor = F.normalize(tensor, dim=0)
                    else:
                        dim = embeddings_by_type[t][-1].shape[0] if embeddings_by_type[t] else 1024
                        tensor = torch.zeros(dim)
                    embeddings_by_type[t].append(tensor)

            self.track_ids = track_ids
            os.makedirs(index_dir, exist_ok=True)
            with open(ids_path, "w") as f:
                json.dump(track_ids, f)

            for t in self.embedding_types:
                mat = torch.stack(embeddings_by_type[t], dim=0)
                torch.save(mat, os.path.join(index_dir, f"{t}.pt"))
                self.embedding_matrices[t] = mat
                self._log(f"Cached {t}: {mat.shape}")

        self._log(f"Loaded {len(self.track_ids)} tracks, "
                  f"types: {list(self.embedding_matrices.keys())}")

    def _format_query(self, query: str) -> str:
        return f"Instruct: {TASK_INSTRUCTION}\nQuery:{query}"

    def _encode_queries(self, queries: list[str]) -> torch.Tensor:
        """Encode queries using Qwen3-Embedding-0.6B with instruction prefix and last-token pooling."""
        formatted = [self._format_query(q) for q in queries]
        self.model.eval()
        all_embs = []

        batch_size = self.query_batch_size
        with torch.no_grad():
            for start in range(0, len(formatted), batch_size):
                batch_texts = formatted[start:start + batch_size]
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                embs = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
                embs = F.normalize(embs.float(), p=2, dim=1).cpu()
                all_embs.append(embs)

        return torch.cat(all_embs, dim=0) if len(all_embs) > 1 else all_embs[0]

    def text_to_item_retrieval(self, query: str, topk: int = 200) -> list[str]:
        """Retrieve top-k tracks for a single query."""
        return self.batch_text_to_item_retrieval([query], topk)[0]

    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int = 200
    ) -> list[list[str]]:
        """Retrieve top-k tracks for multiple queries using RRF across embedding types."""
        query_embs = self._encode_queries(queries)
        rrf_k = 60

        if len(self.embedding_types) == 1:
            et = self.embedding_types[0]
            mat = self.embedding_matrices[et]
            scores = torch.matmul(mat, query_embs.T)
            topk_actual = min(topk, scores.shape[0])
            results = []
            for i in range(len(queries)):
                top_indices = torch.topk(scores[:, i], k=topk_actual).indices.tolist()
                results.append([self.track_ids[idx] for idx in top_indices])
            return results

        # Multi-embedding: RRF merge
        results = []
        for i in range(len(queries)):
            rrf_scores: dict[str, float] = {}
            for et in self.embedding_types:
                w = self.embedding_weights.get(et, 1.0)
                mat = self.embedding_matrices[et]
                scores = torch.matmul(mat, query_embs[i])
                topk_actual = min(topk * 2, scores.shape[0])
                top_indices = torch.topk(scores, k=topk_actual).indices.tolist()
                for rank, idx in enumerate(top_indices):
                    tid = self.track_ids[idx]
                    rrf_scores[tid] = rrf_scores.get(tid, 0.0) + w / (rrf_k + rank + 1)

            ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            results.append([tid for tid, _ in ranked[:topk]])
        return results

    def cleanup(self) -> None:
        if hasattr(self, "model"):
            self.model.to("cpu")
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if hasattr(self, "embedding_matrices"):
            del self.embedding_matrices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
