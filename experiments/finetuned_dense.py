"""Dense retrieval using a fine-tuned Qwen3-Embedding query encoder
against precomputed (frozen) track embeddings.

The query encoder is fine-tuned to project queries into the same space
as the precomputed track embeddings from the dataset.
"""

import os
import json

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


TASK_INSTRUCTION = (
    "Given a music recommendation conversation, "
    "retrieve tracks that match the listener's request"
)


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


class FinetunedDenseRetriever:
    def __init__(
        self,
        model_dir: str = "./cache/finetuned_biencoder/best",
        index_dir: str = "./cache/qwen3_dense",
        embedding_type: str = "attributes-qwen3_embedding_0.6b",
        device: str | None = None,
        max_length: int = 512,
        query_batch_size: int = 64,
    ) -> None:
        self.max_length = max_length
        self.query_batch_size = query_batch_size
        self.embedding_type = embedding_type

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._log(f"Loading fine-tuned query encoder from {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
        self.model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16)
        self.model.to(self.device).eval()

        self._load_index(index_dir)

    @staticmethod
    def _log(msg: str) -> None:
        print(f"[FinetunedDense] {msg}", flush=True)

    def _load_index(self, index_dir: str) -> None:
        ids_path = os.path.join(index_dir, "track_ids.json")
        emb_path = os.path.join(index_dir, f"{self.embedding_type}.pt")

        self.track_ids = json.load(open(ids_path))
        self.track_embeddings = torch.load(emb_path, map_location="cpu")
        self._log(f"Loaded index: {len(self.track_ids)} tracks, dim={self.track_embeddings.shape[1]}")

    def _format_query(self, query: str) -> str:
        return f"Instruct: {TASK_INSTRUCTION}\nQuery:{query}"

    @torch.no_grad()
    def _encode_queries(self, queries: list[str]) -> torch.Tensor:
        formatted = [self._format_query(q) for q in queries]
        all_embs = []
        for i in range(0, len(formatted), self.query_batch_size):
            batch = formatted[i:i + self.query_batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            emb = last_token_pool(out.last_hidden_state, enc["attention_mask"])
            emb = F.normalize(emb.float(), p=2, dim=1).cpu()
            all_embs.append(emb)
        return torch.cat(all_embs, dim=0)

    def text_to_item_retrieval(self, query: str, topk: int = 200) -> list[str]:
        return self.batch_text_to_item_retrieval([query], topk)[0]

    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int = 200
    ) -> list[list[str]]:
        query_embs = self._encode_queries(queries)
        scores = torch.matmul(query_embs, self.track_embeddings.T)
        topk_actual = min(topk, scores.shape[1])

        results = []
        for i in range(len(queries)):
            top_indices = torch.topk(scores[i], k=topk_actual).indices.tolist()
            results.append([self.track_ids[idx] for idx in top_indices])
        return results

    def cleanup(self) -> None:
        if hasattr(self, "model"):
            self.model.to("cpu")
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if hasattr(self, "track_embeddings"):
            del self.track_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
