"""BM25 retrieval over train conversation contexts (thought + goal + query).

Indexes only MOVES_TOWARD_GOAL turns from the train split, creating a
mapping from conversational context to track_id. At query time, matches
the current session's query/goal against these contexts to find tracks
that were good recommendations in similar situations.
"""

import os
import json
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io

import bm25s
from datasets import load_dataset


@contextmanager
def suppress_output():
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        yield


class TrainThoughtBM25:
    """BM25 retriever over quality-filtered train conversation contexts."""

    def __init__(
        self,
        train_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Dataset",
        cache_dir: str = "./cache",
    ) -> None:
        self.train_dataset_name = train_dataset_name
        self.cache_dir = cache_dir
        self.index_dir = os.path.join(cache_dir, "bm25", "train_thought")

        if os.path.exists(self.index_dir):
            self.bm25_model, self.doc_track_ids = self._load_index()
        else:
            self._build_index()
            self.bm25_model, self.doc_track_ids = self._load_index()

        self.track_id_set = set(self.doc_track_ids)
        print(f"[TrainThoughtBM25] {len(self.doc_track_ids)} documents, "
              f"{len(self.track_id_set)} unique tracks")

    def _load_index(self):
        bm25 = bm25s.BM25.load(self.index_dir, load_corpus=True)
        doc_track_ids = json.load(
            open(os.path.join(self.index_dir, "doc_track_ids.json"))
        )
        return bm25, doc_track_ids

    def _build_index(self) -> None:
        print("[TrainThoughtBM25] Building index from train data...")
        train_ds = load_dataset(self.train_dataset_name, split="train")

        corpus = []
        doc_track_ids = []

        for item in train_ds:
            gpa_map = {}
            for entry in item.get("goal_progress_assessments", []):
                gpa_map[int(entry["turn_number"])] = entry.get(
                    "goal_progress_assessment"
                )

            goal = item.get("conversation_goal") or {}
            listener_goal = str(goal.get("listener_goal", ""))
            conversations = item["conversations"]

            for t in conversations:
                if t["role"] != "music":
                    continue

                tn = int(t["turn_number"])
                # GPA at turn N+1 assesses the recommendation at turn N
                quality = gpa_map.get(tn + 1)
                if quality != "MOVES_TOWARD_GOAL":
                    continue

                tid = str(t["content"]).strip()
                thought = t.get("thought", "") or ""

                # Get user query at this turn (same turn_number, role=user)
                user_query = ""
                for ct in conversations:
                    if int(ct["turn_number"]) == tn and ct["role"] == "user":
                        user_query = ct["content"]
                        break

                doc_text = f"{thought} {listener_goal} {user_query}".strip()
                if not doc_text:
                    continue

                corpus.append(doc_text.lower())
                doc_track_ids.append(tid)

        print(f"[TrainThoughtBM25] Indexing {len(corpus)} documents...")
        with suppress_output():
            corpus_tokens = bm25s.tokenize(corpus)
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)

        os.makedirs(self.index_dir, exist_ok=True)
        retriever.save(self.index_dir, corpus=corpus)
        with open(os.path.join(self.index_dir, "doc_track_ids.json"), "w") as f:
            json.dump(doc_track_ids, f)
        print(f"[TrainThoughtBM25] Index saved to {self.index_dir}")

    def text_to_item_retrieval(self, query: str, topk: int = 200) -> list[str]:
        """Retrieve unique track IDs for a query against train thought contexts."""
        with suppress_output():
            query_tokens = bm25s.tokenize([query.lower()])
            # Retrieve more docs than topk since multiple docs can map to same track
            fetch_k = min(topk * 5, len(self.doc_track_ids))
            doc_scores = self.bm25_model.retrieve(
                query_tokens, k=fetch_k, return_as="tuple"
            )

        seen = set()
        results = []
        for item in doc_scores.documents[0]:
            tid = self.doc_track_ids[item["id"]]
            if tid not in seen:
                seen.add(tid)
                results.append(tid)
                if len(results) >= topk:
                    break
        return results

    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int = 200
    ) -> list[list[str]]:
        """Batch retrieval for multiple queries."""
        with suppress_output():
            query_tokens = bm25s.tokenize([q.lower() for q in queries])
            fetch_k = min(topk * 5, len(self.doc_track_ids))
            doc_scores = self.bm25_model.retrieve(
                query_tokens, k=fetch_k, return_as="tuple"
            )

        results = []
        for i in range(len(queries)):
            seen = set()
            ranked = []
            for item in doc_scores.documents[i]:
                tid = self.doc_track_ids[item["id"]]
                if tid not in seen:
                    seen.add(tid)
                    ranked.append(tid)
                    if len(ranked) >= topk:
                        break
            results.append(ranked)
        return results

    def cleanup(self) -> None:
        if hasattr(self, "bm25_model"):
            del self.bm25_model
        if hasattr(self, "doc_track_ids"):
            del self.doc_track_ids
