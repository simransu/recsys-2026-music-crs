"""Session co-occurrence retrieval from train data.

Builds a track co-occurrence graph from train sessions, filtered to only
MOVES_TOWARD_GOAL turns. Given anchor tracks from the current session,
retrieves tracks that frequently co-occurred with those anchors in
quality-filtered train sessions.
"""

import os
import json
from collections import defaultdict

from datasets import load_dataset


class SessionCooccurrence:
    """Item-item collaborative filtering based on train session co-occurrence."""

    def __init__(
        self,
        train_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Dataset",
        cache_dir: str = "./cache",
    ) -> None:
        self.train_dataset_name = train_dataset_name
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, "session_cooccurrence.json")

        if os.path.exists(self.cache_path):
            self._load_cache()
        else:
            self._build_graph()
            self._save_cache()

        print(f"[SessionCooccurrence] {len(self.cooccurrence)} tracks with co-occurrence data")

    def _load_cache(self) -> None:
        with open(self.cache_path) as f:
            self.cooccurrence = json.load(f)

    def _save_cache(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cooccurrence, f)

    def _build_graph(self) -> None:
        print("[SessionCooccurrence] Building co-occurrence graph from train...")
        train_ds = load_dataset(self.train_dataset_name, split="train")

        # track_id -> {co_track_id: count}
        cooccur: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for item in train_ds:
            gpa_map = {}
            for entry in item.get("goal_progress_assessments", []):
                gpa_map[int(entry["turn_number"])] = entry.get(
                    "goal_progress_assessment"
                )

            good_tracks = []
            for t in item["conversations"]:
                if t["role"] != "music":
                    continue
                tn = int(t["turn_number"])
                # GPA at turn N+1 assesses the recommendation at turn N
                if gpa_map.get(tn + 1) == "MOVES_TOWARD_GOAL":
                    good_tracks.append(str(t["content"]).strip())

            for i, t1 in enumerate(good_tracks):
                for j, t2 in enumerate(good_tracks):
                    if i != j:
                        cooccur[t1][t2] += 1

        # Convert to regular dict for JSON serialization
        self.cooccurrence = {k: dict(v) for k, v in cooccur.items()}
        print(f"[SessionCooccurrence] Built graph with {len(self.cooccurrence)} tracks")

    def retrieve(
        self,
        anchor_track_ids: list[str],
        topk: int = 200,
        exclude_ids: set[str] | None = None,
    ) -> list[str]:
        """Retrieve tracks that co-occurred with anchors in quality train sessions."""
        if not anchor_track_ids:
            return []

        scores: dict[str, float] = defaultdict(float)
        exclude = set(anchor_track_ids)
        if exclude_ids:
            exclude |= exclude_ids

        for anchor in anchor_track_ids:
            neighbors = self.cooccurrence.get(anchor, {})
            for tid, count in neighbors.items():
                if tid not in exclude:
                    scores[tid] += count

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [tid for tid, _ in ranked[:topk]]

    def batch_retrieve(
        self,
        anchor_lists: list[list[str]],
        topk: int = 200,
        exclude_lists: list[set[str] | None] | None = None,
    ) -> list[list[str]]:
        """Batch retrieval for multiple anchor lists."""
        if exclude_lists is None:
            exclude_lists = [None] * len(anchor_lists)
        return [
            self.retrieve(anchors, topk=topk, exclude_ids=excl)
            for anchors, excl in zip(anchor_lists, exclude_lists)
        ]

    def cleanup(self) -> None:
        if hasattr(self, "cooccurrence"):
            del self.cooccurrence
