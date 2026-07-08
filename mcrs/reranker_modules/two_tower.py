"""Trainable two-tower reranker built on top of precomputed embeddings and structured metadata."""

from __future__ import annotations

import gc
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset

from mcrs.db_item import MusicCatalogDB
from mcrs.db_item.music_catalog import format_metadata_value, normalize_entity_id
from mcrs.db_user import UserProfileDB


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _stable_bucket(text: Any, dim: int, salt: str) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    if dim <= 0:
        return vec
    token = f"{salt}|{text or ''}"
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % dim
    vec[idx] = 1.0
    return vec


def _bucket_sum(values: Any, dim: int, salt: str) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    if dim <= 0 or not values:
        return vec
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    for value in values:
        vec += _stable_bucket(value, dim, salt)
    total = vec.sum()
    if total > 0:
        vec = vec / total
    return vec


def _safe_vector(value: Any, fallback_dim: int) -> torch.Tensor:
    if value is None:
        return torch.zeros(fallback_dim, dtype=torch.float32)
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32)
    tensor = tensor.flatten().to(dtype=torch.float32)
    if tensor.numel() == 0:
        return torch.zeros(fallback_dim, dtype=torch.float32)
    if fallback_dim > 0 and tensor.numel() != fallback_dim:
        if tensor.numel() > fallback_dim:
            tensor = tensor[:fallback_dim]
        else:
            tensor = F.pad(tensor, (0, fallback_dim - tensor.numel()))
    return tensor


def _parse_year_month(release_date: Any) -> tuple[float, float]:
    if not release_date:
        return 0.0, 0.0
    text = str(release_date)
    year = 0.0
    month = 0.0
    try:
        parts = text.split("-")
        if len(parts) >= 1:
            year = float(parts[0])
        if len(parts) >= 2:
            month = float(parts[1])
    except Exception:
        pass
    return year, month


@dataclass
class FeatureConfig:
    user_country_bucket_dim: int = 16
    user_language_bucket_dim: int = 8
    user_culture_bucket_dim: int = 8
    item_tag_bucket_dim: int = 16
    query_bucket_dim: int = 8


class TwoTowerFeatureBuilder:
    def __init__(
        self,
        user_db: UserProfileDB,
        item_db: MusicCatalogDB,
        feature_config: Optional[FeatureConfig] = None,
    ) -> None:
        self.user_db = user_db
        self.item_db = item_db
        self.feature_config = feature_config or FeatureConfig()

    @property
    def user_feature_dim(self) -> int:
        return 1 + 1 + 3 + self.feature_config.user_country_bucket_dim + self.feature_config.user_language_bucket_dim + self.feature_config.user_culture_bucket_dim

    @property
    def item_feature_dim(self) -> int:
        return 8 + self.feature_config.item_tag_bucket_dim

    @property
    def query_feature_dim(self) -> int:
        return 6 + self.feature_config.query_bucket_dim

    @property
    def pair_feature_dim(self) -> int:
        return 4

    def build_user_features(self, user_profile: Optional[dict[str, Any]] = None, user_id: Optional[str] = None) -> torch.Tensor:
        profile = {}
        if user_id:
            try:
                profile.update(self.user_db.id_to_profile(user_id))
            except KeyError:
                pass
        if user_profile:
            profile.update(user_profile)

        age = _safe_float(profile.get("age"), 0.0) / 100.0
        age_group = str(profile.get("age_group", "")).strip()
        age_group_value = 0.0
        if age_group and age_group[0].isdigit():
            age_group_value = _safe_float(age_group.rstrip("s"), 0.0) / 100.0
        gender = str(profile.get("gender", "")).strip().lower()
        gender_vec = torch.tensor(
            [
                1.0 if gender == "male" else 0.0,
                1.0 if gender == "female" else 0.0,
                1.0 if gender not in {"male", "female"} and gender else 0.0,
            ],
            dtype=torch.float32,
        )
        country_bucket = _stable_bucket(profile.get("country_code") or profile.get("country_name"), self.feature_config.user_country_bucket_dim, "country")
        language_bucket = _stable_bucket(profile.get("preferred_language"), self.feature_config.user_language_bucket_dim, "language")
        culture_bucket = _stable_bucket(profile.get("preferred_musical_culture"), self.feature_config.user_culture_bucket_dim, "culture")
        return torch.cat(
            [
                torch.tensor([age, age_group_value], dtype=torch.float32),
                gender_vec,
                country_bucket,
                language_bucket,
                culture_bucket,
            ]
        )

    def build_item_features(self, track_id: str) -> torch.Tensor:
        metadata = self.item_db.metadata_dict.get(track_id, {})
        popularity = _safe_float(metadata.get("popularity"), 0.0)
        duration = _safe_float(metadata.get("duration"), 0.0)
        release_year, release_month = _parse_year_month(metadata.get("release_date"))
        tag_list = metadata.get("tag_list") or []
        tag_count = float(len(tag_list)) if isinstance(tag_list, (list, tuple, set)) else 0.0
        track_name = format_metadata_value(metadata.get("track_name", ""))
        artist_name = format_metadata_value(metadata.get("artist_name", ""))
        album_name = format_metadata_value(metadata.get("album_name", ""))
        numeric = torch.tensor(
            [
                popularity / 100.0,
                torch.log1p(torch.tensor(duration)).item() / 10.0,
                release_year / 3000.0,
                release_month / 12.0,
                tag_count / 50.0,
                len(track_name) / 100.0,
                len(artist_name) / 100.0,
                len(album_name) / 100.0,
            ],
            dtype=torch.float32,
        )
        tag_bucket = _bucket_sum(tag_list, self.feature_config.item_tag_bucket_dim, "tag")
        return torch.cat([numeric, tag_bucket])

    def build_query_features(self, query_text: Optional[str] = None, conversation_goal: Optional[dict[str, Any]] = None) -> torch.Tensor:
        text = query_text or ""
        tokens = text.split()
        unique_tokens = len(set(tokens))
        question_marks = text.count("?")
        lines = text.count("\n") + 1 if text else 0
        goal_text = ""
        if conversation_goal:
            goal_text = str(conversation_goal.get("listener_goal", ""))
        goal_bucket = _stable_bucket(goal_text, self.feature_config.query_bucket_dim, "goal")
        numeric = torch.tensor(
            [
                len(text) / 1000.0,
                len(tokens) / 100.0,
                unique_tokens / 100.0,
                question_marks / 10.0,
                lines / 20.0,
                len(goal_text) / 1000.0,
            ],
            dtype=torch.float32,
        )
        return torch.cat([numeric, goal_bucket])

    def build_pair_features(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        candidate_rank: int,
        candidate_pool_size: int,
        query_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        user_repr = _safe_vector(user_repr, user_repr.numel() or 1)
        item_repr = _safe_vector(item_repr, item_repr.numel() or 1)
        if user_repr.numel() == 0 and item_repr.numel() == 0:
            aligned_dim = 1
        else:
            aligned_dim = min(max(user_repr.numel(), 1), max(item_repr.numel(), 1))
        user_repr = _safe_vector(user_repr, aligned_dim)
        item_repr = _safe_vector(item_repr, aligned_dim)
        if user_repr.numel() == 0 or item_repr.numel() == 0:
            cosine = torch.tensor(0.0, dtype=torch.float32)
            dot = torch.tensor(0.0, dtype=torch.float32)
        else:
            cosine = F.cosine_similarity(user_repr.unsqueeze(0), item_repr.unsqueeze(0), dim=-1).squeeze(0)
        dot = torch.sum(user_repr * item_repr)
        rank_norm = float(candidate_rank) / max(candidate_pool_size - 1, 1) if candidate_pool_size > 0 else 0.0
        inv_rank = 1.0 / (candidate_rank + 1.0)
        return torch.tensor([dot.item(), cosine.item(), rank_norm, inv_rank], dtype=torch.float32)


class CrossNetV2(nn.Module):
    def __init__(self, input_dim: int, low_rank_dim: int = 32, num_layers: int = 3, num_experts: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.low_rank_dim = low_rank_dim
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.dropout = nn.Dropout(dropout)
        self.gates = nn.ModuleList([nn.Linear(input_dim, num_experts) for _ in range(num_layers)])
        self.u_projs = nn.ParameterList([nn.Parameter(torch.randn(num_experts, input_dim, low_rank_dim) * 0.01) for _ in range(num_layers)])
        self.v_projs = nn.ParameterList([nn.Parameter(torch.randn(num_experts, input_dim, low_rank_dim) * 0.01) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for layer_idx in range(self.num_layers):
            gate_logits = self.gates[layer_idx](xl)
            gate_weights = torch.softmax(gate_logits, dim=-1)
            expert_outputs = []
            for expert_idx in range(self.num_experts):
                v = torch.matmul(xl, self.v_projs[layer_idx][expert_idx])
                u = torch.matmul(v, self.u_projs[layer_idx][expert_idx].transpose(0, 1))
                expert_outputs.append(u)
            mixed = torch.zeros_like(x0)
            for expert_idx, expert_out in enumerate(expert_outputs):
                mixed = mixed + gate_weights[:, expert_idx : expert_idx + 1] * expert_out
            xl = x0 * (mixed + self.biases[layer_idx]) + xl
            xl = self.dropout(xl)
        return xl


class TWO_TOWER_SCORER(nn.Module):
    def __init__(
        self,
        user_dim: int,
        track_dim: int,
        user_feature_dim: int = 0,
        item_feature_dim: int = 0,
        query_feature_dim: int = 0,
        pair_feature_dim: int = 0,
        projection_dim: int = 128,
        tower_hidden_dim: int = 256,
        deep_hidden_dim: int = 256,
        dcn_low_rank_dim: int = 32,
        dcn_num_layers: int = 3,
        dcn_num_experts: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.user_dim = user_dim
        self.track_dim = track_dim
        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        self.query_feature_dim = query_feature_dim
        self.pair_feature_dim = pair_feature_dim
        self.projection_dim = projection_dim
        self.tower_hidden_dim = tower_hidden_dim
        self.deep_hidden_dim = deep_hidden_dim
        self.dropout = dropout
        self.temperature = temperature

        self.user_proj = nn.Sequential(
            nn.LayerNorm(user_dim + user_feature_dim),
            nn.Linear(user_dim + user_feature_dim, tower_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tower_hidden_dim, projection_dim),
        )
        self.track_proj = nn.Sequential(
            nn.LayerNorm(track_dim + item_feature_dim),
            nn.Linear(track_dim + item_feature_dim, tower_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tower_hidden_dim, projection_dim),
        )

        self.pair_input_dim = projection_dim * 4 + user_feature_dim + item_feature_dim + query_feature_dim + pair_feature_dim
        self.cross = CrossNetV2(self.pair_input_dim, low_rank_dim=dcn_low_rank_dim, num_layers=dcn_num_layers, num_experts=dcn_num_experts, dropout=dropout)
        self.deep = nn.Sequential(
            nn.LayerNorm(self.pair_input_dim),
            nn.Linear(self.pair_input_dim, deep_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(deep_hidden_dim, deep_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.pair_input_dim + deep_hidden_dim),
            nn.Linear(self.pair_input_dim + deep_hidden_dim, deep_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(deep_hidden_dim, 1),
        )
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def encode_users(self, user_vecs: torch.Tensor, user_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if user_features is None:
            user_features = torch.zeros(user_vecs.size(0), self.user_feature_dim, device=user_vecs.device, dtype=user_vecs.dtype) if self.user_feature_dim > 0 else None
        if user_features is not None and user_features.numel() > 0:
            user_input = torch.cat([user_vecs, user_features], dim=-1)
        else:
            user_input = user_vecs
        return F.normalize(self.user_proj(user_input), dim=-1)

    def encode_tracks(self, track_vecs: torch.Tensor, item_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if item_features is None:
            item_features = torch.zeros(track_vecs.size(0), self.item_feature_dim, device=track_vecs.device, dtype=track_vecs.dtype) if self.item_feature_dim > 0 else None
        if item_features is not None and item_features.numel() > 0:
            track_input = torch.cat([track_vecs, item_features], dim=-1)
        else:
            track_input = track_vecs
        return F.normalize(self.track_proj(track_input), dim=-1)

    def _pair_input(
        self,
        user_repr: torch.Tensor,
        track_repr: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None,
        query_features: Optional[torch.Tensor] = None,
        pair_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pieces = [user_repr, track_repr, user_repr * track_repr, torch.abs(user_repr - track_repr)]
        if user_features is not None and user_features.numel() > 0:
            pieces.append(user_features)
        if item_features is not None and item_features.numel() > 0:
            pieces.append(item_features)
        if query_features is not None and query_features.numel() > 0:
            pieces.append(query_features)
        if pair_features is not None and pair_features.numel() > 0:
            pieces.append(pair_features)
        return torch.cat(pieces, dim=-1)

    def score_pairs(
        self,
        user_vecs: torch.Tensor,
        track_vecs: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None,
        query_features: Optional[torch.Tensor] = None,
        pair_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        user_repr = self.encode_users(user_vecs, user_features)
        track_repr = self.encode_tracks(track_vecs, item_features)
        pair_input = self._pair_input(user_repr, track_repr, user_features, item_features, query_features, pair_features)
        cross_out = self.cross(pair_input)
        deep_out = self.deep(pair_input)
        logit = self.head(torch.cat([cross_out, deep_out], dim=-1)).squeeze(-1)
        return logit * self.logit_scale.exp().clamp(max=100.0)

    def score_matrix(
        self,
        user_vecs: torch.Tensor,
        track_vecs: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None,
        query_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = []
        for idx in range(track_vecs.size(0)):
            track_batch = track_vecs[idx : idx + 1].expand(user_vecs.size(0), -1)
            item_batch = item_features[idx : idx + 1].expand(user_vecs.size(0), -1) if item_features is not None else None
            scores.append(
                self.score_pairs(
                    user_vecs,
                    track_batch,
                    user_features=user_features,
                    item_features=item_batch,
                    query_features=query_features,
                )
            )
        return torch.stack(scores, dim=1)


class TWO_TOWER_RERANKER:
    def __init__(
        self,
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
    ) -> None:
        self.user_embeddings_dataset_name = user_embeddings_dataset_name
        self.track_embeddings_dataset_name = track_embeddings_dataset_name
        self.item_db_name = item_db_name
        self.user_db_name = user_db_name
        self.track_split_types = track_split_types or ["all_tracks"]
        self.user_split_types = user_split_types or ["all_users"]
        self.corpus_types = corpus_types or ["track_name", "artist_name", "album_name"]
        self.embedding_type = embedding_type
        self.checkpoint_path = checkpoint_path
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif str(device).startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device
        self.projection_dim = projection_dim
        self.tower_hidden_dim = tower_hidden_dim
        self.deep_hidden_dim = deep_hidden_dim
        self.dcn_low_rank_dim = dcn_low_rank_dim
        self.dcn_num_layers = dcn_num_layers
        self.dcn_num_experts = dcn_num_experts
        self.dropout = dropout
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.rrf_k = rrf_k
        self.user_embeddings = self._load_embeddings(self.user_embeddings_dataset_name, "user_id")
        self.track_embeddings = self._load_embeddings(self.track_embeddings_dataset_name, "track_id")
        self.user_db = UserProfileDB(self.user_db_name, self.user_split_types)
        self.item_db = MusicCatalogDB(self.item_db_name, self.track_split_types, self.corpus_types)
        self.feature_builder = TwoTowerFeatureBuilder(self.user_db, self.item_db)
        self.user_dim = next(iter(self.user_embeddings.values())).numel() if self.user_embeddings else 0
        self.track_dim = next(iter(self.track_embeddings.values())).numel() if self.track_embeddings else 0
        self.scorer: Optional[TWO_TOWER_SCORER] = None
        self._log_embedding_coverage()
        self._load_checkpoint_if_available()

    def _load_embeddings(self, dataset_name: str, id_field: str) -> dict[str, torch.Tensor]:
        dataset = load_dataset(dataset_name)
        splits = [dataset[split_name] for split_name in dataset.keys()]
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
            embeddings[normalize_entity_id(row[id_field])] = tensor
        return embeddings

    def _log_embedding_coverage(self) -> None:
        if not self.track_embeddings:
            print("Two-tower reranker: no track embeddings were loaded.")
            return
        if not hasattr(self, "item_db") or not self.item_db.metadata_dict:
            return
        metadata_track_ids = {normalize_entity_id(track_id) for track_id in self.item_db.metadata_dict.keys()}
        embedding_track_ids = set(self.track_embeddings.keys())
        missing_track_ids = sorted(metadata_track_ids - embedding_track_ids)
        if missing_track_ids:
            sample = ", ".join(missing_track_ids[:10])
            print(
                f"Two-tower reranker embedding coverage: {len(embedding_track_ids)}/{len(metadata_track_ids)} tracks matched. "
                f"Missing {len(missing_track_ids)} track embeddings. Sample missing ids: {sample}"
            )
        else:
            print(
                f"Two-tower reranker embedding coverage: {len(embedding_track_ids)}/{len(metadata_track_ids)} tracks matched."
            )
        if hasattr(self, "user_db") and getattr(self.user_db, "user_profiles", None):
            metadata_user_ids = {normalize_entity_id(user_id) for user_id in self.user_db.user_profiles.keys()}
            embedding_user_ids = set(self.user_embeddings.keys())
            missing_user_ids = sorted(metadata_user_ids - embedding_user_ids)
            if missing_user_ids:
                sample_users = ", ".join(missing_user_ids[:10])
                print(
                    f"Two-tower reranker user coverage: {len(embedding_user_ids)}/{len(metadata_user_ids)} users matched. "
                    f"Missing {len(missing_user_ids)} user embeddings. Sample missing ids: {sample_users}"
                )
            else:
                print(
                    f"Two-tower reranker user coverage: {len(embedding_user_ids)}/{len(metadata_user_ids)} users matched."
                )

    def _load_checkpoint_if_available(self) -> None:
        if not os.path.exists(self.checkpoint_path):
            return
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model_state = checkpoint.get("model_state_dict")
        if model_state is None:
            return
        self.scorer = TWO_TOWER_SCORER(
            user_dim=checkpoint["user_dim"],
            track_dim=checkpoint["track_dim"],
            user_feature_dim=checkpoint.get("user_feature_dim", self.feature_builder.user_feature_dim),
            item_feature_dim=checkpoint.get("item_feature_dim", self.feature_builder.item_feature_dim),
            query_feature_dim=checkpoint.get("query_feature_dim", self.feature_builder.query_feature_dim),
            pair_feature_dim=checkpoint.get("pair_feature_dim", self.feature_builder.pair_feature_dim),
            projection_dim=checkpoint["projection_dim"],
            tower_hidden_dim=checkpoint.get("tower_hidden_dim", self.tower_hidden_dim),
            deep_hidden_dim=checkpoint.get("deep_hidden_dim", self.deep_hidden_dim),
            dcn_low_rank_dim=checkpoint.get("dcn_low_rank_dim", self.dcn_low_rank_dim),
            dcn_num_layers=checkpoint.get("dcn_num_layers", self.dcn_num_layers),
            dcn_num_experts=checkpoint.get("dcn_num_experts", self.dcn_num_experts),
            dropout=checkpoint.get("dropout", self.dropout),
            temperature=checkpoint.get("temperature", self.temperature),
        )
        self.scorer.load_state_dict(model_state, strict=False)
        self.scorer.to(self.device)
        self.scorer.eval()

    def _default_user_vec(self) -> torch.Tensor:
        dim = self.user_dim or self.track_dim or 1
        return torch.zeros(dim, dtype=torch.float32)

    def _default_track_vec(self) -> torch.Tensor:
        dim = self.track_dim or self.user_dim or 1
        return torch.zeros(dim, dtype=torch.float32)

    @staticmethod
    def _aligned_dot(user_vec: torch.Tensor, track_vec: torch.Tensor) -> float:
        dim = min(user_vec.numel(), track_vec.numel())
        if dim <= 0:
            return -1.0
        return torch.dot(user_vec[:dim], track_vec[:dim]).item()

    def _raw_score(self, user_id: str | None, candidate_track_ids: list[str], topk: int) -> list[str]:
        user_vec = self.user_embeddings.get(normalize_entity_id(user_id)) if user_id else None
        if user_vec is None:
            user_vec = self._default_user_vec()
        user_features = self.feature_builder.build_user_features(user_id=user_id)
        scores = []
        for rank, track_id in enumerate(candidate_track_ids):
            norm_track_id = normalize_entity_id(track_id)
            track_vec = self.track_embeddings.get(norm_track_id)
            if track_vec is None:
                track_score = -1.0
            else:
                track_score = self._aligned_dot(user_vec, track_vec)
            item_features = self.feature_builder.build_item_features(norm_track_id)
            pair_features = self.feature_builder.build_pair_features(
                user_vec,
                track_vec if track_vec is not None else self._default_track_vec(),
                user_features,
                item_features,
                rank,
                len(candidate_track_ids),
            )
            raw_score = track_score + 0.1 * pair_features[0].item() + 0.1 * pair_features[1].item()
            rank_bonus = 1.0 / (self.rrf_k + rank + 1)
            scores.append(self.alpha * raw_score + self.beta * rank_bonus)
        order = sorted(range(len(candidate_track_ids)), key=lambda idx: scores[idx], reverse=True)
        topk = min(topk, len(candidate_track_ids))
        return [candidate_track_ids[idx] for idx in order[:topk]]

    def rerank(
        self,
        user_id: str | None,
        candidate_track_ids: list[str],
        topk: int = 20,
        user_profile: Optional[dict[str, Any]] = None,
        query_text: Optional[str] = None,
        conversation_goal: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        if not candidate_track_ids:
            return []
        if self.scorer is None:
            return self._raw_score(user_id, candidate_track_ids, topk)

        user_vec = self.user_embeddings.get(normalize_entity_id(user_id)) if user_id else None
        if user_vec is None:
            user_vec = self._default_user_vec()
        user_features = self.feature_builder.build_user_features(user_profile=user_profile, user_id=user_id)
        query_features = self.feature_builder.build_query_features(query_text=query_text, conversation_goal=conversation_goal)
        track_vecs = []
        track_features = []
        available_track_ids = []
        missing_track_count = 0
        for track_id in candidate_track_ids:
            norm_track_id = normalize_entity_id(track_id)
            track_vec = self.track_embeddings.get(norm_track_id)
            if track_vec is None:
                track_vec = self._default_track_vec()
                missing_track_count += 1
            available_track_ids.append(norm_track_id)
            track_vecs.append(track_vec)
            track_features.append(self.feature_builder.build_item_features(norm_track_id))
        if missing_track_count:
            print(
                f"Two-tower reranker: {missing_track_count}/{len(candidate_track_ids)} candidates were missing track embeddings; "
                "ranking all retrieved candidates with zero-vector fallback."
            )
        if not available_track_ids:
            return candidate_track_ids[:topk]

        with torch.inference_mode():
            user_tensor = user_vec.unsqueeze(0).to(self.device)
            user_features_tensor = user_features.unsqueeze(0).to(self.device)
            scores = []
            for rank, (track_vec, item_feat) in enumerate(zip(track_vecs, track_features)):
                track_tensor = track_vec.unsqueeze(0).to(self.device)
                item_tensor = item_feat.unsqueeze(0).to(self.device)
                pair_features = self.feature_builder.build_pair_features(
                    user_vec,
                    track_vec,
                    user_features,
                    item_feat,
                    rank,
                    len(available_track_ids),
                    query_features=query_features,
                ).unsqueeze(0).to(self.device)
                query_tensor = query_features.unsqueeze(0).to(self.device)
                logit = self.scorer.score_pairs(
                    user_tensor,
                    track_tensor,
                    user_features=user_features_tensor,
                    item_features=item_tensor,
                    query_features=query_tensor,
                    pair_features=pair_features,
                )
                scores.append(logit.squeeze(0).item())
        blended_scores = []
        for rank, score in enumerate(scores):
            rank_bonus = 1.0 / (self.rrf_k + rank + 1)
            blended_scores.append(self.alpha * score + self.beta * rank_bonus)
        ranked_indices = sorted(range(len(blended_scores)), key=lambda idx: blended_scores[idx], reverse=True)
        topk = min(topk, len(available_track_ids))
        return [available_track_ids[idx] for idx in ranked_indices[:topk]]

    def batch_rerank(
        self,
        user_ids: list[str | None],
        candidate_track_id_batches: list[list[str]],
        topk: int = 20,
        user_profiles: Optional[list[dict[str, Any] | None]] = None,
        query_texts: Optional[list[str | None]] = None,
        conversation_goals: Optional[list[dict[str, Any] | None]] = None,
    ) -> list[list[str]]:
        outputs = []
        for idx, (user_id, candidate_track_ids) in enumerate(zip(user_ids, candidate_track_id_batches)):
            outputs.append(
                self.rerank(
                    user_id,
                    candidate_track_ids,
                    topk=topk,
                    user_profile=user_profiles[idx] if user_profiles is not None else None,
                    query_text=query_texts[idx] if query_texts is not None else None,
                    conversation_goal=conversation_goals[idx] if conversation_goals is not None else None,
                )
            )
        return outputs

    def cleanup(self) -> None:
        if hasattr(self, "scorer") and self.scorer is not None:
            self.scorer.to("cpu")
            del self.scorer
            self.scorer = None
        if hasattr(self, "user_embeddings"):
            del self.user_embeddings
        if hasattr(self, "track_embeddings"):
            del self.track_embeddings
        if hasattr(self, "user_db"):
            del self.user_db
        if hasattr(self, "item_db"):
            del self.item_db
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
