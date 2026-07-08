import os
import gc
import json
import re
import torch
from typing import Optional, Any, List, Dict
from mcrs.db_item import MusicCatalogDB
from mcrs.db_item.music_catalog import format_metadata_value
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module
from mcrs.retrieval_modules import load_retrieval_module
from mcrs.retrieval_modules.user_to_item import USER_TO_ITEM_MODEL
from mcrs.retrieval_modules.item_to_item import ITEM_TO_ITEM_MODEL
from mcrs.retrieval_modules.train_thought_bm25 import TrainThoughtBM25
from mcrs.retrieval_modules.session_cooccurrence import SessionCooccurrence
from mcrs.retrieval_modules.qwen3_dense import Qwen3DenseRetriever
from mcrs.reranker_modules import load_reranker_module

class CRS_BASELINE:
    """
    Conversational Recommender System (CRS) baseline that wires together an LLM module and an item retrieval module over a music catalog and user profiles.
    Attributes:
        cache_dir: Local path for caching artifacts and indices.
        lm_type: Identifier/name for the LLM backend to load.
        retrieval_type: Retrieval backend to use (e.g., "bm25").
        item_db_name: Hugging Face dataset or DB name for item metadata.
        user_db_name: Hugging Face dataset or DB name for user metadata.
        split_types: Dataset split names to load (e.g., ["test_warm", "test_cold"]).
        corpus_types: Item fields used for retrieval (e.g., title, artist, album).
        device: Compute device for the LLM (e.g., "cuda", "cpu").
        dtype: Torch dtype used by the LLM.
        lm: Loaded LLM module used for response generation.
        retrieval: Retrieval module used to fetch candidate items.
        item_db: Item metadata database accessor.
        user_db: User profile database accessor.
        prompts_dir: Directory containing prompt templates.
        role_prompt: Loaded prompt templates keyed by role.
        session_memory: In-memory list of message dicts for the current session.
    """
    def __init__(self,
        lm_type="meta-llama/Llama-3.2-1B-Instruct",
        retrieval_type="bm25",
        item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
        track_split_types: list[str] = ["all_tracks"], # for test
        user_split_types: list[str] = ["all_users"],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir="./cache",
        device="cuda",
        retrieval_device=None,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
        reranker_type=None,
        reranker_embedding_type="cf-bpr",
        reranker_checkpoint_path="./cache/two_tower_reranker.pt",
        reranker_device="cpu",
        reranker_projection_dim=128,
        reranker_tower_hidden_dim=256,
        reranker_deep_hidden_dim=256,
        reranker_dcn_low_rank_dim=32,
        reranker_dcn_num_layers=3,
        reranker_dcn_num_experts=4,
        reranker_dropout=0.1,
        reranker_temperature=0.07,
        reranker_alpha=1.0,
        reranker_beta=0.15,
        reranker_rrf_k=60,
        user_to_item_embedding_type="cf-bpr",
        bm25_field_weights: Optional[Dict[str, int]] = None,
        enable_query_rewrite: bool = True,
        enable_specificity_routing: bool = True,
        enable_user_to_item: bool = True,
        enable_seen_track_blocking: bool = False,
        enable_metadata_filtering: bool = False,
        enable_item_to_item: bool = False,
        enable_llm_query_planning: bool = False,
        metadata_filter_min_pool: int = 50,
        llm_query_plan_max_new_tokens: int = 256,
        llm_query_plan_mode: str = "replace",
        dense_model_name: str = "bert-base-uncased",
        dense_query_prefix: str = "",
        dense_doc_prefix: str = "",
        specificity_route_map: Optional[Dict[str, str]] = None,
        retrieval_topk=100,
        rerank_topk=20,
        retrieval_bm25_topk=100,
        retrieval_bert_topk=100,
        retrieval_final_topk=20,
        retrieval_rrf_k=60,
        retrieval_bm25_weight=0.8,
        retrieval_bert_weight=0.2,
        retrieval_bpr_weight=0.2,
        retrieval_i2i_weight=0.15,
        load_lm=False,
        load_retrieval=False,
        load_reranker=False,
        **kwargs,
    ):
        """Initialize the CRS baseline components.

        Args:
            lm_type: LLM model identifier to load for response generation.
            retrieval_type: Retrieval backend name (e.g., "bm25").
            item_db_name: Dataset/DB name for item metadata.
            user_db_name: Dataset/DB name for user metadata.
            split_types: Dataset split names to load.
            corpus_types: Item metadata fields used for retrieval.
            cache_dir: Local directory for caching artifacts/indices.
            device: Compute device for the LLM (e.g., "cuda", "cpu").
            dtype: Torch dtype for the LLM weights/tensors.
        """
        self.cache_dir = cache_dir
        self.lm_type = lm_type
        self.retrieval_type = retrieval_type
        self.item_db_name = item_db_name
        self.user_db_name = user_db_name
        self.track_split_types = track_split_types
        self.user_split_types = user_split_types
        self.corpus_types = corpus_types
        self.device = device
        self.retrieval_device = retrieval_device if retrieval_device is not None else device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.reranker_type = reranker_type
        self.reranker_embedding_type = reranker_embedding_type
        self.reranker_checkpoint_path = reranker_checkpoint_path
        self.reranker_device = reranker_device
        self.reranker_projection_dim = reranker_projection_dim
        self.reranker_tower_hidden_dim = reranker_tower_hidden_dim
        self.reranker_deep_hidden_dim = reranker_deep_hidden_dim
        self.reranker_dcn_low_rank_dim = reranker_dcn_low_rank_dim
        self.reranker_dcn_num_layers = reranker_dcn_num_layers
        self.reranker_dcn_num_experts = reranker_dcn_num_experts
        self.reranker_dropout = reranker_dropout
        self.reranker_temperature = reranker_temperature
        self.reranker_alpha = reranker_alpha
        self.reranker_beta = reranker_beta
        self.reranker_rrf_k = reranker_rrf_k
        self.user_to_item_embedding_type = user_to_item_embedding_type
        self.bm25_field_weights = bm25_field_weights
        self.enable_query_rewrite = enable_query_rewrite
        self.enable_specificity_routing = enable_specificity_routing
        self.enable_user_to_item = enable_user_to_item
        self.enable_seen_track_blocking = enable_seen_track_blocking
        self.enable_metadata_filtering = enable_metadata_filtering
        self.enable_item_to_item = enable_item_to_item
        self.enable_llm_query_planning = enable_llm_query_planning
        self.metadata_filter_min_pool = int(metadata_filter_min_pool)
        self.llm_query_plan_max_new_tokens = int(llm_query_plan_max_new_tokens)
        self.llm_query_plan_mode = str(llm_query_plan_mode or "replace").strip().lower()
        self.dense_model_name = dense_model_name
        self.dense_query_prefix = dense_query_prefix
        self.dense_doc_prefix = dense_doc_prefix
        self.specificity_route_map = {
            str(key).strip().upper(): str(value).strip().lower()
            for key, value in (specificity_route_map or {}).items()
            if str(key).strip() and str(value).strip()
        }
        self.retrieval_topk = retrieval_topk
        self.rerank_topk = rerank_topk
        self.retrieval_bm25_topk = retrieval_bm25_topk
        self.retrieval_bert_topk = retrieval_bert_topk
        self.retrieval_final_topk = retrieval_final_topk
        self.retrieval_rrf_k = retrieval_rrf_k
        self.retrieval_bm25_weight = retrieval_bm25_weight
        self.retrieval_bert_weight = retrieval_bert_weight
        self.retrieval_bpr_weight = retrieval_bpr_weight
        self.retrieval_i2i_weight = retrieval_i2i_weight
        self.enable_artist_shortcut = kwargs.get("enable_artist_shortcut", False)
        self.artist_shortcut_weight = float(kwargs.get("artist_shortcut_weight", 1.5))
        self.artist_shortcut_min_count = int(kwargs.get("artist_shortcut_min_count", 2))
        self.i2i_embedding_types = kwargs.get("i2i_embedding_types", None)
        self.i2i_embedding_weights = kwargs.get("i2i_embedding_weights", None)
        self.enable_album_shortcut = kwargs.get("enable_album_shortcut", False)
        self.album_shortcut_weight = float(kwargs.get("album_shortcut_weight", 1.0))
        self._album_shortcut_index = None
        self.enable_entity_matching = kwargs.get("enable_entity_matching", False)
        self.entity_matching_weight = float(kwargs.get("entity_matching_weight", 0.8))
        self._entity_index = None
        self.enable_lambdarank = kwargs.get("enable_lambdarank", False)
        self.lambdarank_model_path = kwargs.get("lambdarank_model_path", "./cache/lambdarank_model.txt")
        self._lambdarank_model = None
        self.enable_train_thought_bm25 = kwargs.get("enable_train_thought_bm25", False)
        self.train_thought_bm25_weight = float(kwargs.get("train_thought_bm25_weight", 0.4))
        self.train_thought_bm25 = None
        self.enable_session_cooccurrence = kwargs.get("enable_session_cooccurrence", False)
        self.session_cooccurrence_weight = float(kwargs.get("session_cooccurrence_weight", 0.3))
        self.session_cooccurrence = None
        self.enable_qwen3_dense = kwargs.get("enable_qwen3_dense", False)
        self.qwen3_dense_weight = float(kwargs.get("qwen3_dense_weight", 0.5))
        self.qwen3_dense_model_name = kwargs.get("qwen3_dense_model_name", "Qwen/Qwen3-Embedding-0.6B")
        self.qwen3_dense_embedding_types = kwargs.get("qwen3_dense_embedding_types", ["attributes-qwen3_embedding_0.6b"])
        self.qwen3_dense_embedding_weights = kwargs.get("qwen3_dense_embedding_weights", None)
        self.qwen3_embedding_query_batch_size = int(kwargs.get("qwen3_embedding_query_batch_size", 64))
        self.qwen3_dense = None
        self.item_db = MusicCatalogDB(self.item_db_name, self.track_split_types, self.corpus_types)
        self.user_db = UserProfileDB(self.user_db_name, self.user_split_types)
        self.prompts_dir = os.path.join(os.path.dirname(__file__), "system_prompts")
        self.role_prompt = {
            "role_play": open(f"{self.prompts_dir}/roleplay.txt", "r", encoding="utf-8").read(),
            "personalization": open(f"{self.prompts_dir}/personalization.txt", "r", encoding="utf-8").read(),
            "response_generation": open(f"{self.prompts_dir}/response_generation.txt", "r", encoding="utf-8").read(),
            "response_discovery": open(f"{self.prompts_dir}/template_discovery.txt", "r", encoding="utf-8").read(),
            "response_expert": open(f"{self.prompts_dir}/template_expert.txt", "r", encoding="utf-8").read(),
            "response_conversational": open(f"{self.prompts_dir}/template_conversational.txt", "r", encoding="utf-8").read(),
        }
        self.query_planning_prompt = open(f"{self.prompts_dir}/query_planning.txt", "r", encoding="utf-8").read()
        self.session_memory = []
        self.lm = None
        self.retrieval = None
        self.reranker = None
        self.user_to_item = None
        self.item_to_item = None
        self._metadata_filter_cache = None
        self._last_planned_query = {}
        self._signal_cache: Dict[int, Dict[str, Any]] = {}
        self._artist_shortcut_index = None
        if load_lm:
            self.load_lm()
        if load_retrieval:
            self.load_retrieval()
        if load_reranker and self.reranker_type:
            self.load_reranker()

    def load_lm(self):
        if self.lm is None:
            self.lm = load_lm_module(self.lm_type, self.device, self.attn_implementation, self.dtype)
        return self.lm

    def load_retrieval(self):
        if self.retrieval is None:
            self.retrieval = load_retrieval_module(
                self.retrieval_type,
                self.item_db_name,
                self.track_split_types,
                self.corpus_types,
                self.cache_dir,
                device=self.retrieval_device,
                field_weights=self.bm25_field_weights,
                dense_model_name=self.dense_model_name,
                dense_query_prefix=self.dense_query_prefix,
                dense_doc_prefix=self.dense_doc_prefix,
                bm25_topk=self.retrieval_bm25_topk,
                bert_topk=self.retrieval_bert_topk,
                final_topk=self.retrieval_final_topk,
                rrf_k=self.retrieval_rrf_k,
                bm25_weight=self.retrieval_bm25_weight,
                bert_weight=self.retrieval_bert_weight,
                bpr_weight=self.retrieval_bpr_weight,
            )
        if self.enable_lambdarank:
            self.load_lambdarank()
        return self.retrieval

    def load_reranker(self):
        if self.reranker is None:
            if not self.reranker_type:
                raise RuntimeError("Reranker type is not configured.")
            self.reranker = load_reranker_module(
                self.reranker_type,
                embedding_type=self.reranker_embedding_type,
                checkpoint_path=self.reranker_checkpoint_path,
                device=self.reranker_device,
                item_db_name=self.item_db_name,
                user_db_name=self.user_db_name,
                track_split_types=self.track_split_types,
                user_split_types=self.user_split_types,
                corpus_types=self.corpus_types,
                projection_dim=self.reranker_projection_dim,
                tower_hidden_dim=self.reranker_tower_hidden_dim,
                deep_hidden_dim=self.reranker_deep_hidden_dim,
                dcn_low_rank_dim=self.reranker_dcn_low_rank_dim,
                dcn_num_layers=self.reranker_dcn_num_layers,
                dcn_num_experts=self.reranker_dcn_num_experts,
                dropout=self.reranker_dropout,
                temperature=self.reranker_temperature,
                alpha=self.reranker_alpha,
                beta=self.reranker_beta,
                rrf_k=self.reranker_rrf_k,
            )
        return self.reranker

    def load_user_to_item(self):
        if self.user_to_item is None:
            self.user_to_item = USER_TO_ITEM_MODEL(
                user_embeddings_dataset_name="talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
                track_embeddings_dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                item_db_name=self.item_db_name,
                user_db_name=self.user_db_name,
                track_split_types=self.track_split_types,
                user_split_types=self.user_split_types,
                corpus_types=self.corpus_types,
                embedding_type=self.user_to_item_embedding_type,
            )
        return self.user_to_item

    def cleanup_lm(self) -> None:
        if hasattr(self.lm, "cleanup"):
            self.lm.cleanup()
        self.lm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def cleanup_retrieval(self) -> None:
        if hasattr(self.retrieval, "cleanup"):
            self.retrieval.cleanup()
        self.retrieval = None
        if self.qwen3_dense is not None:
            self.cleanup_qwen3_dense()
        if self.train_thought_bm25 is not None:
            self.cleanup_train_thought_bm25()
        if self.session_cooccurrence is not None:
            self.cleanup_session_cooccurrence()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def cleanup_reranker(self) -> None:
        if hasattr(self.reranker, "cleanup"):
            self.reranker.cleanup()
        self.reranker = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def cleanup_user_to_item(self) -> None:
        if hasattr(self.user_to_item, "cleanup"):
            self.user_to_item.cleanup()
        self.user_to_item = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def load_item_to_item(self):
        if self.item_to_item is None:
            emb_types = self.i2i_embedding_types
            if emb_types and len(emb_types) > 1:
                self.item_to_item = ITEM_TO_ITEM_MODEL.load_multiple(
                    emb_types,
                    dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                )
            else:
                et = emb_types[0] if emb_types else "attributes-qwen3_embedding_0.6b"
                self.item_to_item = ITEM_TO_ITEM_MODEL(
                    track_embeddings_dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                    embedding_type=et,
                )
        return self.item_to_item

    def cleanup_item_to_item(self) -> None:
        if isinstance(self.item_to_item, dict):
            for m in self.item_to_item.values():
                if hasattr(m, "cleanup"):
                    m.cleanup()
        elif hasattr(self.item_to_item, "cleanup"):
            self.item_to_item.cleanup()
        self.item_to_item = None
        self._artist_shortcut_index = None
        gc.collect()

    def load_train_thought_bm25(self):
        if self.train_thought_bm25 is None:
            self.train_thought_bm25 = TrainThoughtBM25(
                train_dataset_name="talkpl-ai/TalkPlayData-Challenge-Dataset",
                cache_dir=self.cache_dir,
            )
        return self.train_thought_bm25

    def cleanup_train_thought_bm25(self) -> None:
        if hasattr(self.train_thought_bm25, "cleanup"):
            self.train_thought_bm25.cleanup()
        self.train_thought_bm25 = None
        gc.collect()

    def load_session_cooccurrence(self):
        if self.session_cooccurrence is None:
            self.session_cooccurrence = SessionCooccurrence(
                train_dataset_name="talkpl-ai/TalkPlayData-Challenge-Dataset",
                cache_dir=self.cache_dir,
            )
        return self.session_cooccurrence

    def cleanup_session_cooccurrence(self) -> None:
        if hasattr(self.session_cooccurrence, "cleanup"):
            self.session_cooccurrence.cleanup()
        self.session_cooccurrence = None
        gc.collect()

    def load_qwen3_dense(self):
        if self.qwen3_dense is None:
            self.qwen3_dense = Qwen3DenseRetriever(
                model_name=self.qwen3_dense_model_name,
                embedding_types=self.qwen3_dense_embedding_types,
                embedding_weights=self.qwen3_dense_embedding_weights,
                cache_dir=self.cache_dir,
                device=self.retrieval_device,
                qwen3_embedding_query_batch_size=self.qwen3_embedding_query_batch_size,
            )
        return self.qwen3_dense

    def cleanup_qwen3_dense(self) -> None:
        if hasattr(self.qwen3_dense, "cleanup"):
            self.qwen3_dense.cleanup()
        self.qwen3_dense = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reset_session_memory(self):
        """Clear all messages stored in the current session memory.
        """
        self.session_memory = []

    def _upload_session_memory(self, chat_history: List[Dict[str, Any]]):
        """Upload the session memory to the database.
        """
        self.session_memory = chat_history

    def _get_system_prompt(
        self,
        user_id: Optional[str] = None,
        conversation_goal: Optional[Dict[str, Any]] = None,
        session_memory: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build the system prompt, optionally personalized with a user profile and conversation goal."""
        template_key = self._select_response_template(session_memory or [], conversation_goal)
        template = self.role_prompt.get(f"response_{template_key}", self.role_prompt["response_generation"])
        system_prompt = self.role_prompt["role_play"] + template
        if user_id:
            user_profile_str = self.user_db.id_to_profile_str(user_id)
            system_prompt += self.role_prompt["personalization"] + '\n' + user_profile_str
        if conversation_goal:
            listener_goal = conversation_goal.get("listener_goal", "")
            specificity = conversation_goal.get("specificity", "")
            if listener_goal or specificity:
                system_prompt += f"\n\nconversation_goal:\n"
                if listener_goal:
                    system_prompt += f"- listener_goal: {listener_goal}\n"
                if specificity:
                    system_prompt += f"- specificity: {specificity}\n"
                system_prompt += "Use the listener_goal to frame why the recommendation fits what the user is trying to achieve."
        return system_prompt

    def _format_context_block(self, title: str, context: Optional[Dict[str, Any]], preferred_keys: Optional[List[str]] = None) -> str:
        if not context:
            return ""
        items = []
        if preferred_keys:
            for key in preferred_keys:
                if key in context and context[key] is not None:
                    items.append(f"{key}: {context[key]}")
        else:
            for key, value in context.items():
                if value is not None:
                    items.append(f"{key}: {value}")
        if not items:
            return ""
        return f"{title}:\n" + "\n".join(items)

    @staticmethod
    def _normalize_text_block(text: str, limit: int = 240) -> str:
        compact = " ".join(str(text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    _FOLLOWUP_KEYWORDS = frozenset({
        "more", "similar", "another", "different", "else", "again",
        "instead", "something like", "less", "other", "try",
    })

    _PIVOT_KEYWORDS = frozenset({
        "completely different", "something else entirely", "change genre",
        "switch genre", "new direction", "nothing like", "opposite",
        "not like", "don't want", "hate",
    })

    def _classify_i2i_intent(self, session_memory: List[Dict[str, Any]]) -> str:
        last_user_msg = ""
        for turn in reversed(session_memory):
            if turn.get("role") == "user":
                last_user_msg = turn.get("content", "").lower()
                break
        prior_assistant_turns = sum(1 for t in session_memory if t.get("role") in {"assistant", "music"})
        if prior_assistant_turns == 0:
            return "fresh"
        if any(kw in last_user_msg for kw in self._PIVOT_KEYWORDS):
            return "pivot"
        if any(kw in last_user_msg for kw in self._FOLLOWUP_KEYWORDS):
            return "followup"
        return "neutral"

    def _select_response_template(
        self,
        session_memory: List[Dict[str, Any]],
        conversation_goal: Optional[Dict[str, Any]] = None,
    ) -> str:
        specificity = self._extract_specificity(conversation_goal)
        last_user_msg = ""
        for turn in reversed(session_memory):
            if turn.get("role") == "user":
                last_user_msg = turn.get("content", "").lower()
                break
        prior_assistant_turns = sum(1 for t in session_memory if t.get("role") == "assistant")
        is_followup = prior_assistant_turns >= 1 and any(kw in last_user_msg for kw in self._FOLLOWUP_KEYWORDS)
        if is_followup:
            return "conversational"
        if specificity == "HH":
            return "expert"
        return "discovery"

    def _extract_specificity(self, conversation_goal: Optional[Dict[str, Any]]) -> str:
        if not conversation_goal:
            return ""
        specificity = str(conversation_goal.get("specificity", "")).strip().upper()
        return specificity

    def _has_warm_user_embedding(self, user_id: Optional[str]) -> bool:
        if not self.enable_user_to_item:
            return False
        if not user_id:
            return False
        if self.user_to_item is None:
            try:
                self.load_user_to_item()
            except Exception:
                return False
        return self.user_to_item.has_user_embedding(user_id) if self.user_to_item is not None else False

    def _build_recent_turns_block(self, session_memory: List[Dict[str, Any]], max_turns: int = 5) -> str:
        if not session_memory:
            return ""
        recent_turns = session_memory[-max_turns:]
        lines = []
        for turn in recent_turns:
            role = turn.get("role", "")
            content = self._normalize_text_block(turn.get("content", ""))
            if content:
                if role in {"assistant", "music"}:
                    track_name = self._extract_field(content, "track_name")
                    artist_name = self._extract_field(content, "artist_name")
                    album_name = self._extract_field(content, "album_name")
                    tag_list = self._extract_field(content, "tag_list")
                    compact_parts = [p for p in [track_name, artist_name, album_name, tag_list] if p]
                    if compact_parts:
                        lines.append(f"{role}_track: " + " | ".join(compact_parts))
                        continue
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _extract_field(text: str, field_name: str) -> str:
        match = re.search(rf"{re.escape(field_name)}:\s*([^,\n]+)", text, flags=re.IGNORECASE)
        if not match:
            return ""
        return match.group(1).strip()

    # --- Signal expansion helpers (used by query planning context) ---

    @staticmethod
    def _dedupe_terms(terms: List[str], limit: int | None = None) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for term in terms:
            compact = " ".join(str(term or "").split()).strip(" ,.;:-_")
            compact = compact.lower()
            if not compact or compact in seen:
                continue
            seen.add(compact)
            deduped.append(compact)
            if limit is not None and len(deduped) >= limit:
                break
        return deduped

    @staticmethod
    def _lexicon_hits(texts: List[str], lexicon: List[str], limit: int | None = None) -> List[str]:
        joined = " ".join(str(text or "").lower() for text in texts if text)
        hits: List[str] = []
        for term in sorted(lexicon, key=len, reverse=True):
            compact = " ".join(term.lower().split())
            if not compact:
                continue
            if " " in compact or "/" in compact or "-" in compact:
                if compact in joined:
                    hits.append(compact)
            elif re.search(rf"\b{re.escape(compact)}\b", joined):
                hits.append(compact)
        return CRS_BASELINE._dedupe_terms(hits, limit=limit)

    @staticmethod
    def _quoted_phrases(text: str) -> List[str]:
        matches = re.findall(r'"([^"]{2,80})"|\'([^\']{2,80})\'', text)
        phrases = [m[0] or m[1] for m in matches if (m[0] or m[1]).strip()]
        return [phrase.strip() for phrase in phrases if phrase.strip()]

    @staticmethod
    def _pattern_matches(text: str, patterns: List[str]) -> List[str]:
        matches: List[str] = []
        for pattern in patterns:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                if isinstance(match, tuple):
                    match = next((part for part in match if part), "")
                compact = " ".join(str(match or "").split()).strip(" ,.;:-_")
                if compact:
                    matches.append(compact)
        return matches

    def _extract_artist_names(self, session_memory: List[Dict[str, Any]]) -> List[str]:
        artists: List[str] = []
        for turn in session_memory[-5:]:
            content = str(turn.get("content", ""))
            if turn.get("role") in {"assistant", "music"}:
                artist = self._extract_field(content, "artist_name")
                if artist:
                    artists.append(artist)
            if turn.get("role") == "user":
                artists.extend(self._pattern_matches(content, [
                    r"\bby\s+([^,.!?;\n]{2,80})",
                    r"\bartist(?: name)?(?: is|:)?\s+([^,.!?;\n]{2,80})",
                ]))
        return self._dedupe_terms(artists, limit=6)

    def _extract_track_titles(self, session_memory: List[Dict[str, Any]]) -> List[str]:
        titles: List[str] = []
        for turn in session_memory[-5:]:
            content = str(turn.get("content", ""))
            if turn.get("role") in {"assistant", "music"}:
                track = self._extract_field(content, "track_name")
                if track:
                    titles.append(track)
            if turn.get("role") == "user":
                titles.extend(self._quoted_phrases(content))
                titles.extend(self._pattern_matches(content, [
                    r"\b(?:song|track|title|called|named)\s+(?:is\s+)?([^,.!?;\n]{2,80})",
                    r"\b(?:the)\s+([A-Za-z0-9][A-Za-z0-9&'./-]*(?:\s+[A-Za-z0-9][A-Za-z0-9&'./-]*){0,6})",
                ]))
        return self._dedupe_terms(titles, limit=6)

    def _extract_album_names(self, session_memory: List[Dict[str, Any]]) -> List[str]:
        albums: List[str] = []
        for turn in session_memory[-5:]:
            content = str(turn.get("content", ""))
            if turn.get("role") in {"assistant", "music"}:
                album = self._extract_field(content, "album_name")
                if album:
                    albums.append(album)
            if turn.get("role") == "user":
                albums.extend(self._pattern_matches(content, [
                    r"\balbum(?: name)?(?: is|:)?\s+([^,.!?;\n]{2,80})",
                    r"\bfrom\s+the\s+album\s+([^,.!?;\n]{2,80})",
                    r"\bfrom\s+([A-Za-z0-9][A-Za-z0-9&'./-]*(?:\s+[A-Za-z0-9][A-Za-z0-9&'./-]*){0,6})",
                ]))
        return self._dedupe_terms(albums, limit=5)

    def _extract_genre_tags(self, session_memory: List[Dict[str, Any]]) -> List[str]:
        texts = [str(turn.get("content", "")) for turn in session_memory[-5:] if turn.get("content")]
        genre_lexicon = [
            "rock", "pop", "hip hop", "hip-hop", "rap", "r&b", "rb", "jazz", "country", "folk",
            "indie", "metal", "punk", "blues", "soul", "electronic", "edm", "house", "techno",
            "trance", "ambient", "classical", "reggae", "funk", "disco", "alternative", "grunge",
            "emo", "lo fi", "lo-fi", "dance pop", "pop rock", "indie rock", "hard rock", "soft rock",
            "punk rock", "singer songwriter", "singer/songwriter", "drum and bass", "trap", "synthpop",
            "progressive rock", "j rock", "j-rock", "k rock", "k-rock", "k pop", "k-pop", "j pop", "j-pop",
            "latin", "ska", "gospel", "psychedelic", "garage rock", "post punk",
        ]
        genres = self._lexicon_hits(texts, genre_lexicon, limit=10)
        for turn in session_memory[-5:]:
            if turn.get("role") in {"assistant", "music"}:
                tag_list = self._extract_field(str(turn.get("content", "")), "tag_list")
                if tag_list:
                    genres.extend([tag.strip().lower() for tag in re.split(r"[,;/|]", tag_list) if tag.strip()])
        return self._dedupe_terms(genres, limit=10)

    def _extract_mood_and_negative_phrases(self, session_memory: List[Dict[str, Any]]) -> tuple[List[str], List[str]]:
        texts = [str(turn.get("content", "")) for turn in session_memory[-5:] if turn.get("content")]
        mood_lexicon = [
            "uplifting", "upbeat", "energetic", "feel good", "feel-good", "mellow", "sad", "happy",
            "angry", "romantic", "dark", "gritty", "acoustic", "atmospheric", "chill", "workout",
            "party", "relaxing", "nostalgic", "bittersweet", "dreamy", "warm", "brooding", "intense",
            "anthem", "driving", "powerful", "laid back", "low key", "high energy", "emotional",
            "melancholic", "nostalgic", "soft", "smooth", "calm", "energetic", "hype", "aggressive",
        ]
        mood_phrases = self._lexicon_hits(texts, mood_lexicon, limit=12)
        negative_constraints: List[str] = []
        for text in texts:
            negative_constraints.extend(self._pattern_matches(text, [
                r"\bnot too\s+([a-z0-9][a-z0-9\s-]{1,40})",
                r"\bnot\s+([a-z0-9][a-z0-9\s-]{1,40})",
                r"\bno\s+([a-z0-9][a-z0-9\s-]{1,40})",
                r"\bwithout\s+([a-z0-9][a-z0-9\s-]{1,40})",
                r"\bavoid\s+([a-z0-9][a-z0-9\s-]{1,40})",
                r"\bdon'?t want\s+([a-z0-9][a-z0-9\s-]{1,40})",
            ]))
        negative_constraints = [f"not {term}" if not term.startswith(("not ", "no ", "without ", "avoid ")) else term for term in negative_constraints]
        return (
            self._dedupe_terms(mood_phrases, limit=12),
            self._dedupe_terms(negative_constraints, limit=8),
        )

    def _extract_conversation_signals(self, session_memory: List[Dict[str, Any]]) -> Dict[str, Any]:
        cache_key = id(session_memory)
        if cache_key in self._signal_cache:
            return self._signal_cache[cache_key]
        artists = self._extract_artist_names(session_memory)
        tracks = self._extract_track_titles(session_memory)
        albums = self._extract_album_names(session_memory)
        genres = self._extract_genre_tags(session_memory)
        moods, negatives = self._extract_mood_and_negative_phrases(session_memory)
        result = {
            "artists": artists,
            "tracks": tracks,
            "albums": albums,
            "genres": genres,
            "moods": moods,
            "negatives": negatives,
        }
        self._signal_cache[cache_key] = result
        if len(self._signal_cache) > 1000:
            self._signal_cache.clear()
        return result

    def _build_signal_expansion_block(self, session_memory: List[Dict[str, Any]], signals: Optional[Dict[str, Any]] = None) -> str:
        if signals is None:
            signals = self._extract_conversation_signals(session_memory)
        artists = signals["artists"]
        tracks = signals["tracks"]
        albums = signals["albums"]
        genres = signals["genres"]
        moods = signals["moods"]
        negatives = signals["negatives"]
        sections = []
        if artists:
            sections.append(f"artist_names: {' '.join([artist for artist in artists for _ in range(2)])}")
        if tracks:
            sections.append(f"track_titles: {' '.join([track for track in tracks for _ in range(2)])}")
        if albums:
            sections.append(f"album_names: {' '.join(albums)}")
        if genres:
            sections.append(f"genre_tags: {' '.join(genres)}")
        if moods:
            sections.append(f"mood_phrases: {' '.join(moods)}")
        if negatives:
            sections.append(f"negative_constraints: {' '.join(negatives)}")
        if not sections:
            return ""
        return "conversation_signal_expansion:\n" + "\n".join(sections)

    # --- Metadata filtering and seen-track blocking ---

    @staticmethod
    def _normalize_match_text(value: object) -> str:
        text = format_metadata_value(value).lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _metadata_filter_records(self) -> list[dict[str, Any]]:
        if self._metadata_filter_cache is not None:
            return self._metadata_filter_cache
        records: list[dict[str, Any]] = []
        for track_id, metadata in self.item_db.metadata_dict.items():
            tag_text = self._normalize_match_text(metadata.get("tag_list", ""))
            release_year = self._extract_release_year(metadata.get("release_date", ""))
            popularity = metadata.get("popularity", 0)
            try:
                popularity_float = float(popularity or 0)
            except Exception:
                popularity_float = 0.0
            records.append({
                "track_id": track_id,
                "track_name": self._normalize_match_text(metadata.get("track_name", "")),
                "artist_name": self._normalize_match_text(metadata.get("artist_name", "")),
                "album_name": self._normalize_match_text(metadata.get("album_name", "")),
                "tag_text": tag_text,
                "release_year": int(release_year) if release_year else None,
                "popularity": popularity_float,
            })
        self._metadata_filter_cache = records
        return records

    @staticmethod
    def _extract_release_year(value: object) -> str:
        match = re.search(r"(19\d{2}|20\d{2})", format_metadata_value(value))
        return match.group(1) if match else ""

    @staticmethod
    def _extract_year_constraints(text: str) -> tuple[int | None, int | None]:
        years = [int(year) for year in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
        if len(years) >= 2:
            return min(years), max(years)
        if len(years) == 1:
            year = years[0]
            if re.search(rf"\b(?:after|since|post)\s+{year}\b", text):
                return year + 1, None
            if re.search(rf"\b(?:before|pre)\s+{year}\b", text):
                return None, year - 1
            return year, year
        decade_match = re.search(r"\b(?:the\s+)?((?:19|20)?\d0)s\b", text)
        if decade_match:
            decade = int(decade_match.group(1))
            if decade < 100:
                decade += 1900 if decade >= 30 else 2000
            return decade, decade + 9
        early_match = re.search(r"\bearly\s+((?:19|20)?\d0)s\b", text)
        if early_match:
            decade = int(early_match.group(1))
            if decade < 100:
                decade += 1900 if decade >= 30 else 2000
            return decade, decade + 3
        late_match = re.search(r"\blate\s+((?:19|20)?\d0)s\b", text)
        if late_match:
            decade = int(late_match.group(1))
            if decade < 100:
                decade += 1900 if decade >= 30 else 2000
            return decade + 6, decade + 9
        return None, None

    @staticmethod
    def _phrase_matches_field(phrase: str, field_text: str) -> bool:
        norm_phrase = CRS_BASELINE._normalize_match_text(phrase)
        if not norm_phrase:
            return False
        return norm_phrase in field_text or field_text in norm_phrase

    def _build_metadata_filter_pool(
        self,
        data: Dict[str, Any],
        session_memory: List[Dict[str, Any]],
        signals: Optional[Dict[str, Any]] = None,
    ) -> list[str] | None:
        if not self.enable_metadata_filtering:
            return None
        conversation_goal = data.get("conversation_goal")
        category = self._extract_category(conversation_goal)
        specificity = self._extract_specificity(conversation_goal)
        text = " ".join(str(turn.get("content", "")) for turn in session_memory if turn.get("content"))
        norm_text = self._normalize_match_text(text)
        if category in {"A", "C", "E"} and specificity not in {"HH", "LH"}:
            return None
        if signals is None:
            signals = self._extract_conversation_signals(session_memory)
        artists = signals["artists"]
        tracks = signals["tracks"]
        albums = signals["albums"]
        genres = signals["genres"]
        moods = signals["moods"]
        negatives = signals["negatives"]
        tag_terms = self._dedupe_terms(genres + moods, limit=12)
        negative_terms = [self._normalize_match_text(term.replace("not ", "")) for term in negatives]
        year_min, year_max = self._extract_year_constraints(norm_text)
        popularity_min: float | None = None
        popularity_max: float | None = None
        if re.search(r"\b(popular|famous|hit|viral|trending|chart|mainstream|radio)\b", norm_text):
            popularity_min = 55.0
        if re.search(r"\b(underground|obscure|not mainstream|less mainstream|deep cut)\b", norm_text):
            popularity_max = 60.0
        has_constraint = any([artists, tracks, albums, tag_terms, year_min is not None, year_max is not None, popularity_min is not None, popularity_max is not None])
        if not has_constraint:
            return None
        target_categories = {"B", "D", "F", "G", "H", "I", "J", "K"}
        if category and category not in target_categories and specificity not in {"HH", "LH"}:
            return None
        candidates: list[str] = []
        for record in self._metadata_filter_records():
            if tracks and not any(self._phrase_matches_field(track, record["track_name"]) for track in tracks):
                continue
            if artists and not any(self._phrase_matches_field(artist, record["artist_name"]) for artist in artists):
                continue
            if albums and not any(self._phrase_matches_field(album, record["album_name"]) for album in albums):
                continue
            release_year = record["release_year"]
            if year_min is not None and (release_year is None or release_year < year_min):
                continue
            if year_max is not None and (release_year is None or release_year > year_max):
                continue
            if popularity_min is not None and record["popularity"] < popularity_min:
                continue
            if popularity_max is not None and record["popularity"] > popularity_max:
                continue
            if tag_terms:
                tag_hits = sum(1 for term in tag_terms if self._normalize_match_text(term) in record["tag_text"])
                if tag_hits == 0 and category in {"D", "G", "I", "J", "K", "F", "B"}:
                    continue
            if negative_terms and any(term and term in record["tag_text"] for term in negative_terms):
                continue
            candidates.append(record["track_id"])
        if len(candidates) < self.metadata_filter_min_pool:
            return None
        return candidates

    def _extract_seen_track_ids(self, session_memory: List[Dict[str, Any]]) -> set[str]:
        if not self.enable_seen_track_blocking:
            return set()
        seen: set[str] = set()
        for turn in session_memory:
            content = str(turn.get("content", "") or "")
            if turn.get("role") == "music":
                norm = content.strip()
                if norm in self.item_db.metadata_dict:
                    seen.add(norm)
            for match in re.findall(r"track_id:\s*([A-Za-z0-9][A-Za-z0-9_-]{6,})", content, flags=re.IGNORECASE):
                if match in self.item_db.metadata_dict:
                    seen.add(match)
            if content.strip() in self.item_db.metadata_dict:
                seen.add(content.strip())
        return seen

    def _extract_anchor_track_ids(self, session_memory: List[Dict[str, Any]], max_anchors: int = 3) -> list[str]:
        anchors: list[str] = []
        for turn in reversed(session_memory):
            content = str(turn.get("content", "") or "")
            role = turn.get("role", "")
            if role in {"music", "assistant"}:
                norm = content.strip()
                if norm in self.item_db.metadata_dict and norm not in anchors:
                    anchors.append(norm)
                for match in re.findall(r"track_id:\s*([A-Za-z0-9][A-Za-z0-9_-]{6,})", content, flags=re.IGNORECASE):
                    if match in self.item_db.metadata_dict and match not in anchors:
                        anchors.append(match)
            if len(anchors) >= max_anchors:
                break
        return anchors

    def _build_artist_shortcut_index(self) -> dict[str, list[str]]:
        if self._artist_shortcut_index is not None:
            return self._artist_shortcut_index
        from collections import defaultdict
        idx: dict[str, list[str]] = defaultdict(list)
        for tid, meta in self.item_db.metadata_dict.items():
            artist = format_metadata_value(meta.get("artist_name", "")).lower().strip()
            if artist:
                idx[artist].append(tid)
        self._artist_shortcut_index = dict(idx)
        return self._artist_shortcut_index

    def _artist_shortcut_retrieve(self, anchor_track_ids: list[str], topk: int = 100) -> list[str]:
        if not self.enable_artist_shortcut or not anchor_track_ids:
            return []
        anchor_artists = []
        for tid in anchor_track_ids[:3]:
            meta = self.item_db.metadata_dict.get(tid, {})
            artist = format_metadata_value(meta.get("artist_name", "")).lower().strip()
            if artist:
                anchor_artists.append(artist)
        if not anchor_artists:
            return []
        from collections import Counter
        counts = Counter(anchor_artists)
        dominant, cnt = counts.most_common(1)[0]
        if cnt < self.artist_shortcut_min_count:
            return []
        idx = self._build_artist_shortcut_index()
        tracks = idx.get(dominant, [])
        seen = set(anchor_track_ids)
        return [t for t in tracks if t not in seen][:topk]

    def _build_album_shortcut_index(self) -> dict[str, list[str]]:
        if self._album_shortcut_index is not None:
            return self._album_shortcut_index
        from collections import defaultdict
        idx: dict[str, list[str]] = defaultdict(list)
        for tid, meta in self.item_db.metadata_dict.items():
            album = format_metadata_value(meta.get("album_name", "")).lower().strip()
            if album:
                idx[album].append(tid)
        self._album_shortcut_index = dict(idx)
        return self._album_shortcut_index

    def _album_shortcut_retrieve(self, anchor_track_ids: list[str], topk: int = 200) -> list[str]:
        if not self.enable_album_shortcut or not anchor_track_ids:
            return []
        idx = self._build_album_shortcut_index()
        seen = set(anchor_track_ids)
        results = []
        for tid in anchor_track_ids[:5]:
            meta = self.item_db.metadata_dict.get(tid, {})
            album = format_metadata_value(meta.get("album_name", "")).lower().strip()
            if album and album in idx:
                for t in idx[album]:
                    if t not in seen:
                        seen.add(t)
                        results.append(t)
        return results[:topk]

    def _build_entity_index(self) -> dict[str, dict[str, list[str]]]:
        if self._entity_index is not None:
            return self._entity_index
        from collections import defaultdict
        track_name_idx: dict[str, list[str]] = defaultdict(list)
        artist_name_idx: dict[str, list[str]] = defaultdict(list)
        album_name_idx: dict[str, list[str]] = defaultdict(list)
        for tid, meta in self.item_db.metadata_dict.items():
            tn = format_metadata_value(meta.get("track_name", "")).lower().strip()
            an = format_metadata_value(meta.get("artist_name", "")).lower().strip()
            al = format_metadata_value(meta.get("album_name", "")).lower().strip()
            if tn and len(tn) > 2:
                track_name_idx[tn].append(tid)
            if an and len(an) > 2:
                artist_name_idx[an].append(tid)
            if al and len(al) > 2:
                album_name_idx[al].append(tid)
        self._entity_index = {
            "track": dict(track_name_idx),
            "artist": dict(artist_name_idx),
            "album": dict(album_name_idx),
        }
        return self._entity_index

    def _entity_matching_retrieve(self, user_query: str, topk: int = 200) -> list[str]:
        if not self.enable_entity_matching or not user_query:
            return []
        idx = self._build_entity_index()
        query_lower = user_query.lower()
        seen = set()
        results = []
        for entity_type in ["track", "artist", "album"]:
            for name, tids in idx[entity_type].items():
                if name in query_lower:
                    for tid in tids:
                        if tid not in seen:
                            seen.add(tid)
                            results.append(tid)
        return results[:topk]

    def _compute_entity_match_features(self, candidate_tid: str, user_query: str) -> dict[str, int]:
        meta = self.item_db.metadata_dict.get(candidate_tid, {})
        query_lower = user_query.lower()
        tn = format_metadata_value(meta.get("track_name", "")).lower().strip()
        an = format_metadata_value(meta.get("artist_name", "")).lower().strip()
        al = format_metadata_value(meta.get("album_name", "")).lower().strip()
        return {
            "track_name_in_query": 1 if tn and len(tn) > 2 and tn in query_lower else 0,
            "artist_name_in_query": 1 if an and len(an) > 2 and an in query_lower else 0,
            "album_name_in_query": 1 if al and len(al) > 2 and al in query_lower else 0,
        }

    def load_lambdarank(self):
        if self._lambdarank_model is None and os.path.exists(self.lambdarank_model_path):
            import lightgbm as lgb
            self._lambdarank_model = lgb.Booster(model_file=self.lambdarank_model_path)
            print(f"[LambdaRank] Loaded model from {self.lambdarank_model_path}")
        return self._lambdarank_model

    I2I_EMBEDDING_TYPES_FOR_FEATURES = [
        "image-siglip2", "cf-bpr", "audio-laion_clap",
        "attributes-qwen3_embedding_0.6b", "lyrics-qwen3_embedding_0.6b",
        "metadata-qwen3_embedding_0.6b",
    ]

    LAMBDARANK_FEATURE_NAMES = [
        "primary_rank", "primary_present",
        "bm25_rank", "bm25_present",
        "bert_rank", "bert_present",
        "bpr_rank", "bpr_present",
        "i2i_rank", "i2i_present",
        *[f for et in I2I_EMBEDDING_TYPES_FOR_FEATURES
          for f in (f"i2i_{et}_rank", f"i2i_{et}_present")],
        "train_thought_rank", "train_thought_present",
        "cooccur_rank", "cooccur_present",
        "qwen3_dense_rank", "qwen3_dense_present",
        "artist_shortcut_present",
        "album_shortcut_present",
        "entity_present",
        "track_name_in_query", "artist_name_in_query", "album_name_in_query",
        "same_artist_as_any_anchor", "same_artist_as_last_anchor",
        "same_album_as_any_anchor", "same_album_as_last_anchor",
        "turn_number", "specificity_encoded", "category_encoded",
        "num_anchors", "cooccur_count_raw",
    ]

    def _extract_lambdarank_features(
        self,
        candidate_tid: str,
        source_ranks: dict[str, dict[str, int]],
        user_query: str,
        anchor_track_ids: list[str],
        turn_number: int,
        specificity: str,
        category: str,
        cooccur_counts: dict[str, int] | None = None,
    ) -> list[float]:
        def get_rank(source_name):
            return source_ranks.get(source_name, {}).get(candidate_tid, 0)
        def get_present(source_name):
            return 1.0 if candidate_tid in source_ranks.get(source_name, {}) else 0.0

        entity_feats = self._compute_entity_match_features(candidate_tid, user_query)

        # Structural features
        meta = self.item_db.metadata_dict.get(candidate_tid, {})
        cand_artist = format_metadata_value(meta.get("artist_name", "")).lower().strip()
        cand_album = format_metadata_value(meta.get("album_name", "")).lower().strip()

        anchor_artists = []
        anchor_albums = []
        for atid in anchor_track_ids:
            ameta = self.item_db.metadata_dict.get(atid, {})
            anchor_artists.append(format_metadata_value(ameta.get("artist_name", "")).lower().strip())
            anchor_albums.append(format_metadata_value(ameta.get("album_name", "")).lower().strip())

        same_artist_any = 1.0 if cand_artist and cand_artist in anchor_artists else 0.0
        same_artist_last = 1.0 if cand_artist and anchor_artists and cand_artist == anchor_artists[0] else 0.0
        same_album_any = 1.0 if cand_album and cand_album in anchor_albums else 0.0
        same_album_last = 1.0 if cand_album and anchor_albums and cand_album == anchor_albums[0] else 0.0

        spec_map = {"HH": 0, "HL": 1, "LH": 2, "LL": 3}
        cat_map = {c: i for i, c in enumerate("ABCDEFGHIJK")}

        cooccur_raw = float(cooccur_counts.get(candidate_tid, 0)) if cooccur_counts else 0.0

        i2i_per_type_feats = []
        for et in self.I2I_EMBEDDING_TYPES_FOR_FEATURES:
            i2i_per_type_feats.extend([
                float(get_rank(f"i2i_{et}")), get_present(f"i2i_{et}"),
            ])

        return [
            float(get_rank("primary")), get_present("primary"),
            float(get_rank("bm25")), get_present("bm25"),
            float(get_rank("bert")), get_present("bert"),
            float(get_rank("bpr")), get_present("bpr"),
            float(get_rank("i2i")), get_present("i2i"),
            *i2i_per_type_feats,
            float(get_rank("train_thought")), get_present("train_thought"),
            float(get_rank("cooccur")), get_present("cooccur"),
            float(get_rank("qwen3_dense")), get_present("qwen3_dense"),
            get_present("artist"),
            get_present("album"),
            get_present("entity"),
            float(entity_feats["track_name_in_query"]),
            float(entity_feats["artist_name_in_query"]),
            float(entity_feats["album_name_in_query"]),
            same_artist_any, same_artist_last,
            same_album_any, same_album_last,
            float(turn_number),
            float(spec_map.get(specificity, -1)),
            float(cat_map.get(category, -1)),
            float(len(anchor_track_ids)),
            cooccur_raw,
        ]

    def _lambdarank_rerank(
        self,
        batch_data: list[dict],
        all_sources: dict[str, list],
        retrieval_inputs: list[str],
        dense_queries: list[str] | None,
        topk: int,
    ) -> list[list[str]]:
        import numpy as np

        results = []
        for i, data in enumerate(batch_data):
            # Pool all candidates
            candidate_set: set[str] = set()
            source_ranks: dict[str, dict[str, int]] = {}
            for source_name, items_list in all_sources.items():
                items = items_list[i]
                if items:
                    rank_map = {tid: rank + 1 for rank, tid in enumerate(items)}
                    source_ranks[source_name] = rank_map
                    candidate_set.update(items)

            if not candidate_set:
                results.append([])
                continue

            # Extract context
            session_mem = data.get("effective_session_memory") or data.get("session_memory") or []
            user_query = ""
            for t in reversed(session_mem):
                if t.get("role") == "user":
                    user_query = t.get("content", "")
                    break

            anchors = self._extract_anchor_track_ids(session_mem)
            goal = data.get("conversation_goal") or {}
            specificity = ""
            category = ""
            turn_number = len([t for t in session_mem if t.get("role") == "user"])

            # Get raw co-occurrence counts
            cooccur_counts = None
            if self.session_cooccurrence is not None and anchors:
                cooccur_counts = {}
                for anchor in anchors:
                    neighbors = self.session_cooccurrence.cooccurrence.get(anchor, {})
                    for tid, count in neighbors.items():
                        cooccur_counts[tid] = cooccur_counts.get(tid, 0) + count

            # Extract features for all candidates
            candidates = list(candidate_set)
            features = []
            for tid in candidates:
                feat = self._extract_lambdarank_features(
                    tid, source_ranks, user_query, anchors,
                    turn_number, specificity, category, cooccur_counts,
                )
                features.append(feat)

            feat_array = np.array(features, dtype=np.float32)
            scores = self._lambdarank_model.predict(feat_array)

            ranked_indices = np.argsort(-scores)
            results.append([candidates[idx] for idx in ranked_indices[:topk]])

        return results

    def _postprocess_retrieval_items(self, data: Dict[str, Any], items: List[str]) -> List[str]:
        seen_track_ids = self._extract_seen_track_ids(data.get("effective_session_memory") or data.get("session_memory") or [])
        if not seen_track_ids:
            return items
        return [item for item in items if item not in seen_track_ids]

    def _extract_category(self, conversation_goal: Optional[Dict[str, Any]]) -> str:
        if not conversation_goal:
            return ""
        category = conversation_goal.get("category", "")
        if isinstance(category, dict):
            category = category.get("code", "")
        return str(category or "").strip().upper()[:1]

    # --- LLM Query Planning ---

    def _build_query_planning_context(
        self,
        session_memory: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        conversation_goal: Optional[Dict[str, Any]] = None,
    ) -> str:
        latest_user_message = ""
        for turn in reversed(session_memory):
            if turn.get("role") == "user":
                latest_user_message = self._normalize_text_block(turn.get("content", ""), limit=400)
                break
        recent_block = self._build_recent_turns_block(session_memory, max_turns=5)
        query_parts = []
        if latest_user_message:
            query_parts.append(f"latest_user_message: {latest_user_message}")
        if recent_block:
            query_parts.append(f"recent_context:\n{recent_block}")
        if user_id:
            query_parts.append(f"user_id: {user_id}")
        merged_user_profile: Optional[Dict[str, Any]] = None
        if user_id:
            merged_user_profile = dict(self.user_db.id_to_profile(user_id))
        if user_profile:
            if merged_user_profile is None:
                merged_user_profile = {}
            merged_user_profile.update(user_profile)
        profile_block = self._format_context_block(
            "user_profile",
            merged_user_profile,
            preferred_keys=[
                "user_id", "age", "age_group", "country_code", "country_name",
                "gender", "preferred_language", "preferred_musical_culture",
            ],
        )
        if profile_block:
            query_parts.append(profile_block)
        goal_block = self._format_context_block(
            "conversation_goal",
            conversation_goal,
            preferred_keys=["category", "specificity", "listener_goal"],
        )
        if goal_block:
            query_parts.append(goal_block)
        signal_block = self._build_signal_expansion_block(session_memory)
        if signal_block:
            query_parts.append(signal_block)
        return "\n".join(query_parts) if query_parts else "music recommendation"

    @staticmethod
    def _normalize_plan_tokens(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            tokens = [token.strip() for token in re.split(r"[,;/|]+", value) if token.strip()]
            return tokens if tokens else [value.strip()] if value.strip() else []
        if isinstance(value, (list, tuple, set)):
            out: list[str] = []
            for item in value:
                out.extend(CRS_BASELINE._normalize_plan_tokens(item))
            return out
        return [str(value).strip()] if str(value).strip() else []

    def _plan_retrieval_query(
        self,
        session_memory: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        conversation_goal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enable_llm_query_planning:
            return {}
        self.load_lm()
        query_context = self._build_query_planning_context(
            session_memory,
            user_id=user_id,
            user_profile=user_profile,
            conversation_goal=conversation_goal,
        )
        planned = self.lm.plan_retrieval_query(
            self.query_planning_prompt,
            query_context,
            max_new_tokens=self.llm_query_plan_max_new_tokens,
        )
        parsed = planned.get("parsed") if isinstance(planned, dict) else {}
        return parsed if isinstance(parsed, dict) else {}

    def _build_planned_query_lines(self, planned_query: Dict[str, Any], llm_query: str = "") -> list[str]:
        lines: list[str] = []
        if llm_query:
            lines.append(f"llm_query: {llm_query}")
        for field_name in [
            "artist_names", "track_titles", "album_names",
            "genre_tags", "mood_phrases", "year_terms",
        ]:
            values = self._normalize_plan_tokens(planned_query.get(field_name))
            if values:
                lines.append(f"{field_name}: {' '.join(values)}")
        return lines

    # --- Retrieval query building ---

    def _build_retrieval_query(
        self,
        session_memory: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        conversation_goal: Optional[Dict[str, Any]] = None,
    ) -> str:
        planned_query = {}
        if self.enable_llm_query_planning:
            planned_query = self._plan_retrieval_query(
                session_memory,
                user_id=user_id,
                user_profile=user_profile,
                conversation_goal=conversation_goal,
            )
            self._last_planned_query = planned_query
        llm_query = str(planned_query.get("bm25_query") or planned_query.get("query") or "").strip() if planned_query else ""
        if llm_query and self.llm_query_plan_mode in {"replace", "only", "llm_only"}:
            planned_lines = self._build_planned_query_lines(planned_query, llm_query=llm_query)
            return "\n".join(planned_lines) if planned_lines else llm_query
        if not self.enable_query_rewrite:
            query_parts = [f"{conversation.get('role')}: {conversation.get('content')}" for conversation in session_memory]
            query_parts = [part for part in query_parts if part.strip()]
            if user_id:
                query_parts.append(f"user_id: {user_id}")
            if planned_query:
                query_parts.extend(self._build_planned_query_lines(planned_query, llm_query=llm_query))
            raw_query = "\n".join(query_parts)
            if raw_query.strip():
                return raw_query
            return "music recommendation"

        latest_user_message = ""
        for turn in reversed(session_memory):
            if turn.get("role") == "user":
                latest_user_message = self._normalize_text_block(turn.get("content", ""), limit=280)
                break
        recent_block = self._build_recent_turns_block(session_memory, max_turns=5)
        query_parts = []
        if latest_user_message:
            query_parts.append(f"intent: {latest_user_message}")
        if recent_block:
            query_parts.append(f"recent_context:\n{recent_block}")
        if user_id:
            query_parts.append(f"user_id: {user_id}")
        merged_user_profile: Optional[Dict[str, Any]] = None
        if user_id:
            merged_user_profile = dict(self.user_db.id_to_profile(user_id))
        if user_profile:
            if merged_user_profile is None:
                merged_user_profile = {}
            merged_user_profile.update(user_profile)
        profile_block = self._format_context_block(
            "user_profile",
            merged_user_profile,
            preferred_keys=[
                "user_id",
                "age",
                "age_group",
                "country_code",
                "country_name",
                "gender",
                "preferred_language",
                "preferred_musical_culture",
            ],
        )
        if profile_block:
            query_parts.append(profile_block)
        goal_block = self._format_context_block(
            "conversation_goal",
            conversation_goal,
            preferred_keys=["category", "specificity", "listener_goal"],
        )
        if goal_block:
            query_parts.append(goal_block)
        if planned_query:
            query_parts.extend(self._build_planned_query_lines(planned_query, llm_query=llm_query))
        return "\n".join(query_parts) if query_parts else "music recommendation"

    def _build_retrieval_input(
        self,
        session_memory: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        conversation_goal: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._build_retrieval_query(
            session_memory,
            user_id=user_id,
            user_profile=user_profile,
            conversation_goal=conversation_goal,
        )

    def _build_dense_query(
        self,
        session_memory: List[Dict[str, Any]],
        conversation_goal: Optional[Dict[str, Any]] = None,
    ) -> str:
        latest_user_message = ""
        for turn in reversed(session_memory):
            if turn.get("role") == "user":
                latest_user_message = self._normalize_text_block(turn.get("content", ""), limit=300)
                break
        parts = []
        if latest_user_message:
            parts.append(latest_user_message)
        if conversation_goal:
            listener_goal = conversation_goal.get("listener_goal", "")
            if listener_goal:
                parts.append(self._normalize_text_block(listener_goal, limit=150))
        for turn in session_memory[-4:]:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role in {"assistant", "music"}:
                track = self._extract_field(content, "track_name")
                artist = self._extract_field(content, "artist_name")
                tags = self._extract_field(content, "tag_list")
                compact = ", ".join(p for p in [track, artist, tags] if p)
                if compact:
                    parts.append(f"previously recommended: {compact}")
            elif role == "user" and content != latest_user_message:
                parts.append(self._normalize_text_block(content, limit=150))
        return " ".join(parts) if parts else "music recommendation"

    def _route_retrieval(
        self,
        user_id: Optional[str],
        conversation_goal: Optional[Dict[str, Any]],
    ) -> str:
        if self.retrieval_type == "multi_source":
            return "multi_source"
        if not self.enable_specificity_routing:
            return "hybrid" if self.retrieval_type == "hybrid" else "bm25"
        specificity = self._extract_specificity(conversation_goal)
        if specificity in self.specificity_route_map:
            return self.specificity_route_map[specificity]
        if self.retrieval_type == "bm25":
            return "bm25"
        if self.retrieval_type == "bert":
            return "bert"
        warm_user = self._has_warm_user_embedding(user_id)
        if specificity == "HH":
            return "bm25"
        if self.enable_user_to_item and warm_user and specificity in {"LL", "LH", "HL"}:
            return "bpr"
        if specificity in {"LH", "LL", "HL"}:
            return "bert"
        return "bm25"

    def _retrieve_multi_source_batch(
        self,
        batch_data: List[Dict[str, Any]],
        retrieval_inputs: List[str],
        dense_queries: List[str] | None = None,
    ) -> tuple[list[list[str]], list[str], dict[str, list] | None]:
        topk = max(self.retrieval_bm25_topk, self.retrieval_topk)
        allowed_pools = [
            self._build_metadata_filter_pool(
                data,
                data.get("effective_session_memory") or data.get("session_memory") or [],
            )
            for data in batch_data
        ]
        bpr_items_list: list[list[str] | None] = [None] * len(batch_data)
        if self.enable_user_to_item:
            self.load_user_to_item()
            user_ids_for_bpr = [data.get("user_id") for data in batch_data]
            bpr_all = self.user_to_item.batch_user_to_item_retrieval(user_ids_for_bpr, topk=topk)
            for i, items in enumerate(bpr_all):
                if items:
                    bpr_items_list[i] = items
        i2i_items_list: list[list[str] | None] = [None] * len(batch_data)
        i2i_per_type: dict[str, list[list[str] | None]] = {}
        i2i_weights: list[float] = [0.0] * len(batch_data)
        i2i_intents: list[str] = [""] * len(batch_data)
        batch_anchors_raw: list[list[str]] = [[] for _ in batch_data]
        if self.enable_item_to_item:
            self.load_item_to_item()
            anchor_lists_for_batch: list[list[str]] = []
            batch_indices_for_i2i: list[int] = []
            for i, data in enumerate(batch_data):
                session_mem = data.get("effective_session_memory") or data.get("session_memory") or []
                anchors = self._extract_anchor_track_ids(session_mem)
                batch_anchors_raw[i] = anchors
                intent = self._classify_i2i_intent(session_mem) if anchors else "fresh"
                i2i_intents[i] = intent
                if not anchors or intent in {"pivot", "fresh"}:
                    continue
                if intent == "followup":
                    i2i_weights[i] = self.retrieval_i2i_weight * 2.0
                    anchor_lists_for_batch.append(anchors[:1])
                else:
                    i2i_weights[i] = self.retrieval_i2i_weight
                    anchor_lists_for_batch.append(anchors)
                batch_indices_for_i2i.append(i)
            if anchor_lists_for_batch and isinstance(self.item_to_item, dict):
                emb_weights = self.i2i_embedding_weights or {}
                default_w = self.retrieval_i2i_weight
                merged_i2i: list[list[tuple[list[str], float]]] = [[] for _ in batch_data]
                for et, model in self.item_to_item.items():
                    et_results = model.batch_retrieve_similar(anchor_lists_for_batch, topk=topk)
                    w = float(emb_weights.get(et, default_w))
                    per_type_list: list[list[str] | None] = [None] * len(batch_data)
                    for pos, idx in enumerate(batch_indices_for_i2i):
                        if et_results[pos]:
                            merged_i2i[idx].append((et_results[pos], w))
                            per_type_list[idx] = et_results[pos]
                    i2i_per_type[et] = per_type_list
                for i in range(len(batch_data)):
                    if not merged_i2i[i]:
                        continue
                    rrf_scores: dict[str, float] = {}
                    rrf_k = self.retrieval_rrf_k
                    for items, w in merged_i2i[i]:
                        for rank, tid in enumerate(items):
                            rrf_scores[tid] = rrf_scores.get(tid, 0.0) + w / (rrf_k + rank + 1)
                    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
                    i2i_items_list[i] = [tid for tid, _ in ranked[:topk]]
                    i2i_weights[i] = 1.0
            elif anchor_lists_for_batch and not isinstance(self.item_to_item, dict):
                i2i_results = self.item_to_item.batch_retrieve_similar(anchor_lists_for_batch, topk=topk)
                for pos, idx in enumerate(batch_indices_for_i2i):
                    i2i_items_list[idx] = i2i_results[pos]
        # Artist shortcut: if anchors share an artist, inject all tracks by that artist
        artist_items_list: list[list[str] | None] = [None] * len(batch_data)
        artist_weights: list[float] = [0.0] * len(batch_data)
        if self.enable_artist_shortcut:
            for i, data in enumerate(batch_data):
                anchors = batch_anchors_raw[i] if self.enable_item_to_item else []
                if not anchors:
                    session_mem = data.get("effective_session_memory") or data.get("session_memory") or []
                    anchors = self._extract_anchor_track_ids(session_mem)
                artist_pool = self._artist_shortcut_retrieve(anchors, topk=topk)
                if artist_pool:
                    artist_items_list[i] = artist_pool
                    artist_weights[i] = self.artist_shortcut_weight
        # Album shortcut: inject all tracks from same album as anchors
        album_items_list: list[list[str] | None] = [None] * len(batch_data)
        if self.enable_album_shortcut:
            for i, data in enumerate(batch_data):
                anchors = batch_anchors_raw[i] if self.enable_item_to_item else []
                if not anchors:
                    session_mem = data.get("effective_session_memory") or data.get("session_memory") or []
                    anchors = self._extract_anchor_track_ids(session_mem)
                album_pool = self._album_shortcut_retrieve(anchors, topk=topk)
                if album_pool:
                    album_items_list[i] = album_pool
        # Entity matching: inject tracks whose name/artist/album appears in user query
        entity_items_list: list[list[str] | None] = [None] * len(batch_data)
        if self.enable_entity_matching:
            for i, data in enumerate(batch_data):
                session_mem = data.get("effective_session_memory") or data.get("session_memory") or []
                user_query = ""
                for t in reversed(session_mem):
                    if t.get("role") == "user":
                        user_query = t.get("content", "")
                        break
                entity_pool = self._entity_matching_retrieve(user_query, topk=topk)
                if entity_pool:
                    entity_items_list[i] = entity_pool
        # Train-thought BM25: retrieve from quality-filtered train contexts
        train_thought_items_list: list[list[str] | None] = [None] * len(batch_data)
        if self.enable_train_thought_bm25:
            self.load_train_thought_bm25()
            train_thought_queries = []
            for i, data in enumerate(batch_data):
                goal = data.get("conversation_goal") or {}
                listener_goal = str(goal.get("listener_goal", ""))
                session_mem = data.get("effective_session_memory") or data.get("session_memory") or []
                user_query = ""
                for t in reversed(session_mem):
                    if t.get("role") == "user":
                        user_query = t.get("content", "")
                        break
                query = f"{user_query} {listener_goal} {retrieval_inputs[i]}".strip()
                train_thought_queries.append(query)
            tt_results = self.train_thought_bm25.batch_text_to_item_retrieval(
                train_thought_queries, topk=topk
            )
            for i, items in enumerate(tt_results):
                if items:
                    train_thought_items_list[i] = items
        # Session co-occurrence: retrieve tracks that co-occurred with anchors in good train sessions
        cooccur_items_list: list[list[str] | None] = [None] * len(batch_data)
        if self.enable_session_cooccurrence:
            self.load_session_cooccurrence()
            anchor_lists = []
            cooccur_indices = []
            for i, data in enumerate(batch_data):
                anchors = batch_anchors_raw[i] if self.enable_item_to_item else []
                if not anchors:
                    session_mem = data.get("effective_session_memory") or data.get("session_memory") or []
                    anchors = self._extract_anchor_track_ids(session_mem)
                if anchors:
                    anchor_lists.append(anchors)
                    cooccur_indices.append(i)
            if anchor_lists:
                cooccur_results = self.session_cooccurrence.batch_retrieve(
                    anchor_lists, topk=topk
                )
                for pos, idx in enumerate(cooccur_indices):
                    if cooccur_results[pos]:
                        cooccur_items_list[idx] = cooccur_results[pos]
        # Qwen3 dense retrieval: encode queries with Qwen3-Embedding-0.6B, match against precomputed track embeddings
        qwen3_dense_items_list: list[list[str] | None] = [None] * len(batch_data)
        if self.enable_qwen3_dense:
            self.load_qwen3_dense()
            qwen3_queries = dense_queries if dense_queries is not None else retrieval_inputs
            qd_results = self.qwen3_dense.batch_text_to_item_retrieval(
                qwen3_queries, topk=topk
            )
            for i, items in enumerate(qd_results):
                if items:
                    qwen3_dense_items_list[i] = items
        bm25_weight_overrides: list[float] = []
        bert_weight_overrides: list[float] = []
        for data in batch_data:
            specificity = self._extract_specificity(data.get("conversation_goal"))
            if specificity == "HH":
                bm25_weight_overrides.append(0.85)
                bert_weight_overrides.append(0.10)
            else:
                bm25_weight_overrides.append(self.retrieval_bm25_weight)
                bert_weight_overrides.append(self.retrieval_bert_weight)
        # Get individual BM25 and BERT results for LambdaRank features
        bm25_items_list: list[list[str]] = []
        bert_items_list: list[list[str]] = []
        if hasattr(self.retrieval, "bm25") and hasattr(self.retrieval, "bert"):
            if allowed_pools and any(pool for pool in allowed_pools):
                bm25_items_list = [
                    self.retrieval.bm25.text_to_item_retrieval(q, topk, allowed_track_ids=pool)
                    for q, pool in zip(retrieval_inputs, allowed_pools)
                ]
            else:
                bm25_items_list = self.retrieval.bm25.batch_text_to_item_retrieval(retrieval_inputs, topk)
            bert_queries = dense_queries if dense_queries is not None else retrieval_inputs
            bert_items_list = self.retrieval.bert.batch_text_to_item_retrieval(bert_queries, topk)
        results = self.retrieval.batch_text_to_item_retrieval(
            retrieval_inputs,
            topk=topk,
            bpr_items_list=bpr_items_list,
            i2i_items_list=i2i_items_list,
            i2i_weights=i2i_weights,
            allowed_track_ids_list=allowed_pools,
            dense_queries=dense_queries,
            bm25_weight_overrides=bm25_weight_overrides,
            bert_weight_overrides=bert_weight_overrides,
        )
        # Collect all source results for merging / LambdaRank
        all_sources = {
            "primary": results,
            "bm25": bm25_items_list if bm25_items_list else [[] for _ in batch_data],
            "bert": bert_items_list if bert_items_list else [[] for _ in batch_data],
            "bpr": bpr_items_list,
            "artist": artist_items_list,
            "album": album_items_list,
            "i2i": i2i_items_list,
            "train_thought": train_thought_items_list,
            "cooccur": cooccur_items_list,
            "qwen3_dense": qwen3_dense_items_list,
            "entity": entity_items_list,
        }
        # Add per-type I2I results for LambdaRank (separate channels)
        for et, per_type_list in i2i_per_type.items():
            all_sources[f"i2i_{et}"] = per_type_list

        if self.enable_lambdarank and self._lambdarank_model is not None:
            results = self._lambdarank_rerank(batch_data, all_sources, retrieval_inputs, dense_queries, topk)
        else:
            # Fallback: Strategy C weighted secondary merge via RRF
            for i, data in enumerate(batch_data):
                has_new = any(all_sources[s][i] for s in all_sources if s != "primary")
                if has_new:
                    rrf_scores: dict[str, float] = {}
                    rrf_k = self.retrieval_rrf_k
                    for rank, tid in enumerate(results[i]):
                        rrf_scores[tid] = rrf_scores.get(tid, 0.0) + 1.0 / (rrf_k + rank + 1)
                    source_weights = {
                        "artist": 0.5, "album": self.album_shortcut_weight,
                        "i2i": 0.2, "train_thought": self.train_thought_bm25_weight,
                        "cooccur": self.session_cooccurrence_weight,
                        "qwen3_dense": self.qwen3_dense_weight,
                        "entity": self.entity_matching_weight,
                    }
                    for source_name, w in source_weights.items():
                        items = all_sources[source_name][i]
                        if items:
                            for rank, tid in enumerate(items):
                                rrf_scores[tid] = rrf_scores.get(tid, 0.0) + w / (rrf_k + rank + 1)
                    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
                    results[i] = [tid for tid, _ in ranked[:topk]]
        for i in range(len(batch_data)):
            results[i] = self._postprocess_retrieval_items(batch_data[i], results[i])
        route_parts = []
        for i in range(len(batch_data)):
            parts = ["multi_source"]
            if bpr_items_list[i]:
                parts.append("bpr")
            if i2i_items_list[i]:
                emb_names = list(self.item_to_item.keys()) if isinstance(self.item_to_item, dict) else ["i2i"]
                parts.append(f"i2i[{'+'.join(emb_names)}]({i2i_intents[i]})")
            if artist_items_list[i]:
                parts.append("artist_shortcut")
            if album_items_list[i]:
                parts.append("album_shortcut")
            if entity_items_list[i]:
                parts.append("entity_match")
            if train_thought_items_list[i]:
                parts.append("train_thought")
            if cooccur_items_list[i]:
                parts.append("cooccur")
            if qwen3_dense_items_list[i]:
                parts.append("qwen3_dense")
            if self.enable_lambdarank and self._lambdarank_model is not None:
                parts.append("lambdarank")
            route_parts.append("+".join(parts))
        return results, route_parts, all_sources

    def _retrieve_route_batch(
        self,
        batch_data: List[Dict[str, Any]],
        retrieval_inputs: List[str],
        dense_queries: List[str] | None = None,
    ) -> tuple[list[list[str]], list[str], dict[str, list] | None]:
        if self.retrieval_type == "multi_source":
            return self._retrieve_multi_source_batch(batch_data, retrieval_inputs, dense_queries=dense_queries)

        routes = [self._route_retrieval(data.get("user_id"), data.get("conversation_goal")) for data in batch_data]
        retrieval_items: list[list[str]] = [[] for _ in batch_data]

        bm25_indices = [idx for idx, route in enumerate(routes) if route == "bm25"]
        bert_indices = [idx for idx, route in enumerate(routes) if route == "bert"]
        bpr_indices = [idx for idx, route in enumerate(routes) if route == "bpr"]
        hybrid_indices = [idx for idx, route in enumerate(routes) if route == "hybrid"]

        if bm25_indices:
            bm25_queries = [retrieval_inputs[idx] for idx in bm25_indices]
            bm25_topk = max(self.retrieval_bm25_topk, self.retrieval_topk)
            bm25_model = self.retrieval.bm25 if hasattr(self.retrieval, "bm25") else self.retrieval
            bm25_allowed_pools = [
                self._build_metadata_filter_pool(
                    batch_data[idx],
                    batch_data[idx].get("effective_session_memory") or batch_data[idx].get("session_memory") or [],
                )
                for idx in bm25_indices
            ]
            if any(pool for pool in bm25_allowed_pools):
                bm25_results = [
                    bm25_model.text_to_item_retrieval(query, topk=bm25_topk, allowed_track_ids=bm25_allowed_pools[pos])
                    for pos, query in enumerate(bm25_queries)
                ]
            elif hasattr(self.retrieval, "bm25"):
                bm25_results = self.retrieval.bm25.batch_text_to_item_retrieval(bm25_queries, topk=bm25_topk)
            else:
                bm25_results = self.retrieval.batch_text_to_item_retrieval(bm25_queries, topk=bm25_topk)
            for pos, idx in enumerate(bm25_indices):
                retrieval_items[idx] = self._postprocess_retrieval_items(batch_data[idx], bm25_results[pos])

        if hybrid_indices:
            hybrid_queries = [retrieval_inputs[idx] for idx in hybrid_indices]
            if hasattr(self.retrieval, "batch_text_to_item_retrieval"):
                hybrid_results = self.retrieval.batch_text_to_item_retrieval(hybrid_queries, topk=self.retrieval_topk)
            else:
                hybrid_results = [self.retrieval.text_to_item_retrieval(q, topk=self.retrieval_topk) for q in hybrid_queries]
            for pos, idx in enumerate(hybrid_indices):
                retrieval_items[idx] = hybrid_results[pos]

        if bert_indices:
            bert_queries = [retrieval_inputs[idx] for idx in bert_indices]
            if hasattr(self.retrieval, "bert"):
                bert_results = self.retrieval.bert.batch_text_to_item_retrieval(bert_queries, topk=max(self.retrieval_bert_topk, self.retrieval_topk))
            else:
                bert_results = [self.retrieval.text_to_item_retrieval(q, topk=self.retrieval_topk) for q in bert_queries]
            for pos, idx in enumerate(bert_indices):
                retrieval_items[idx] = bert_results[pos]

        if bpr_indices:
            if not self.enable_user_to_item:
                for idx in bpr_indices:
                    retrieval_items[idx] = []
                return retrieval_items, routes, None
            self.load_user_to_item()
            warm_user_ids = [batch_data[idx].get("user_id") for idx in bpr_indices]
            bpr_topk = max(self.retrieval_bm25_topk, self.retrieval_topk)
            bpr_results = self.user_to_item.batch_user_to_item_retrieval(warm_user_ids, topk=bpr_topk)
            for pos, idx in enumerate(bpr_indices):
                retrieval_items[idx] = bpr_results[pos]

        return retrieval_items, routes, None

    def _extract_session_likes(self, session_memory: List[Dict[str, Any]]) -> List[str]:
        """Extract tracks the user positively responded to in this session."""
        liked = []
        prev_music_track = None
        for turn in session_memory:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role in {"assistant", "music"}:
                # extract track name if present
                track_name = self._extract_field(content, "track_name")
                artist_name = self._extract_field(content, "artist_name")
                if track_name and artist_name:
                    prev_music_track = f"{track_name} by {artist_name}"
                elif track_name:
                    prev_music_track = track_name
            elif role == "user" and prev_music_track:
                # positive signals
                positive_keywords = ["yes", "perfect", "exactly", "love", "great", "good", "like", "that's it", "more like", "similar"]
                if any(kw in content.lower() for kw in positive_keywords):
                    if prev_music_track not in liked:
                        liked.append(prev_music_track)
                prev_music_track = None
        return liked

    @staticmethod
    def _clean_tags(raw_tags, max_tags: int = 12) -> str:
        """Return a short comma-separated string of meaningful genre/style tags."""
        from mcrs.db_item.music_catalog import format_metadata_value
        import re
        flat = format_metadata_value(raw_tags)
        tags = [t.strip() for t in flat.split(",") if t.strip()]
        seen = set()
        kept = []
        skip_patterns = re.compile(
            r"\d+\s*(of|stars?)\s*\d+|\bseen live\b|\bheard on\b|\bfavorit|\bfavourit"
            r"|\bowned\b|\bbought\b|\bmy \b|\bi \b|\bwe \b",
            re.IGNORECASE,
        )
        for tag in tags:
            if len(tag) > 28:
                continue
            if skip_patterns.search(tag):
                continue
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            kept.append(tag)
            if len(kept) >= max_tags:
                break
        return ", ".join(kept)

    def _build_recommendation_context(self, ranked_items: List[str], max_alternatives: int = 2, session_memory: Optional[List[Dict[str, Any]]] = None, planned_query: Optional[Dict[str, Any]] = None) -> str:
        if not ranked_items:
            return "No recommendation available."

        top_track_id = ranked_items[0]
        top_meta = self.item_db.metadata_dict.get(top_track_id, {})

        title = format_metadata_value(top_meta.get("track_name", "")).title()
        artist = format_metadata_value(top_meta.get("artist_name", "")).title()
        album = format_metadata_value(top_meta.get("album_name", "")).title()
        release_date = format_metadata_value(top_meta.get("release_date", ""))
        year = release_date[:4] if release_date and len(release_date) >= 4 else ""
        style = self._clean_tags(top_meta.get("tag_list", ""))

        lines = []

        # Recommended track
        album_year = f"{album} ({year})" if year else album
        lines.append(f'Recommended track: "{title}" by {artist}')
        if album_year:
            lines.append(f"Album: {album_year}")
        if style:
            lines.append(f"Style: {style}")

        # User intent context from planner
        if planned_query:
            intent_parts = []
            mood = planned_query.get("mood_phrases")
            if mood and isinstance(mood, list) and mood:
                intent_parts.append(f"mood: {', '.join(mood[:4])}")
            genre = planned_query.get("genre_tags")
            if genre and isinstance(genre, list) and genre:
                intent_parts.append(f"genre: {', '.join(genre[:4])}")
            year_terms = planned_query.get("year_terms")
            if year_terms and isinstance(year_terms, list) and year_terms:
                intent_parts.append(f"era: {', '.join(year_terms[:2])}")
            if intent_parts:
                lines.append(f"What the user is looking for: {'; '.join(intent_parts)}")

        # Tracks liked this session
        if session_memory:
            liked = self._extract_session_likes(session_memory)
            if liked:
                lines.append(f"Tracks the user liked this session: {', '.join(liked)}")

        # Alternatives
        alts = []
        for track_id in ranked_items[1: 2]:  # max 1 alternative
            alt_meta = self.item_db.metadata_dict.get(track_id, {})
            alt_title = format_metadata_value(alt_meta.get("track_name", "")).title()
            alt_artist = format_metadata_value(alt_meta.get("artist_name", "")).title()
            alt_album = format_metadata_value(alt_meta.get("album_name", "")).title()
            alt_date = format_metadata_value(alt_meta.get("release_date", ""))
            alt_year = alt_date[:4] if alt_date and len(alt_date) >= 4 else ""
            alt_style = self._clean_tags(alt_meta.get("tag_list", ""), max_tags=6)
            alt_album_year = f"{alt_album} ({alt_year})" if alt_year else alt_album
            alts.append(f'  - "{alt_title}" by {alt_artist} ({alt_album_year}) — {alt_style}')
        if alts:
            lines.append("Alternatives if the top track doesn't fit:")
            lines.extend(alts)

        return "\n".join(lines)

    def chat(self, user_query: str, user_id: Optional[str] = None) -> dict[str, Any]:
        """Run a single CRS turn: retrieve items and generate a response.
        Args:
            user_query: The user's latest message or request.
            user_id: Optional user identifier for personalization.
        Returns:
            A dictionary with keys:
                - user_id: The user identifier (may be None).
                - user_query: Echo of the input query.
                - retrieval_items: List of retrieved item IDs (top candidates).
                - recommend_item: Metadata for the top recommended item.
                - response: The generated assistant response string.
        """
        self.session_memory.append({"role": "user", "content": user_query})
        # stage0. system prompt
        system_prompt = self._get_system_prompt(user_id, session_memory=self.session_memory)
        # stage1. retrieval
        self.load_retrieval()
        self._last_planned_query = {}
        retrieval_input = self._build_retrieval_input(self.session_memory, user_id)
        planned_query = self._last_planned_query
        dense_q = [self._build_dense_query(self.session_memory)] if self.retrieval_type == "multi_source" else None
        retrieval_items, _, _ = self._retrieve_route_batch(
            [{"user_id": user_id, "conversation_goal": None}],
            [retrieval_input],
            dense_queries=dense_q,
        )
        retrieval_items = retrieval_items[0]
        self.cleanup_retrieval()
        ranked_items = retrieval_items
        if self.reranker_type:
            self.load_reranker()
            ranked_items = self.reranker.rerank(user_id, retrieval_items, topk=self.rerank_topk, query_text=retrieval_input)
            self.cleanup_reranker()
        if not ranked_items:
            if retrieval_items:
                print("Reranker returned no candidates for this turn; using retrieval output.")
                ranked_items = retrieval_items[:1]
            else:
                print("Reranker returned no candidates and retrieval also returned nothing.")
                ranked_items = []
        recommend_item = self.item_db.id_to_metadata(ranked_items[0]) if ranked_items else ""
        # stage2. response generation
        self.load_lm()
        response_context = self._build_recommendation_context(ranked_items, session_memory=self.session_memory, planned_query=planned_query)
        response = self.lm.response_generation(system_prompt, self.session_memory, response_context)
        return {
            "user_id": user_id,
            "user_query": user_query,
            "retrieval_items": retrieval_items,
            "ranked_items": ranked_items,
            "recommend_item": recommend_item,
            "response": response,
        }

    def batch_chat(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run multiple CRS turns in batch: retrieve items and generate responses.
        Args:
            batch_data: List of dictionaries, each containing:
                - user_query: The user's latest message or request.
                - user_id: Optional user identifier for personalization.
                - session_memory: List of chat history messages.
        Returns:
            A list of dictionaries, each with keys:
                - user_id: The user identifier (may be None).
                - user_query: Echo of the input query.
                - retrieval_items: List of retrieved item IDs (top candidates).
                - recommend_item: Metadata for the top recommended item.
                - response: The generated assistant response string.
        """
        # Prepare batch inputs
        if self.lm is not None:
            self.lm.cleanup()
            self.lm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        sys_prompts = []
        retrieval_inputs = []
        session_memories = []
        user_ids = []
        planned_queries = []

        for data in batch_data:
            user_query = data['user_query']
            user_id = data.get('user_id')
            user_profile = data.get('user_profile')
            conversation_goal = data.get('conversation_goal')
            session_memory = data['session_memory'].copy()
            session_memory.append({"role": "user", "content": user_query})

            sys_prompts.append(self._get_system_prompt(user_id, conversation_goal=conversation_goal, session_memory=session_memory))
            user_ids.append(user_id)
            self._last_planned_query = {}
            retrieval_inputs.append(
                self._build_retrieval_input(
                    session_memory,
                    user_id,
                    user_profile=user_profile,
                    conversation_goal=conversation_goal,
                )
            )
            planned_queries.append(self._last_planned_query)
            session_memories.append(session_memory)

        # Stage 1: Batch retrieval
        self.load_retrieval()
        dense_queries = None
        if self.retrieval_type == "multi_source":
            dense_queries = [
                self._build_dense_query(session_memories[i], conversation_goal=batch_data[i].get("conversation_goal"))
                for i in range(len(batch_data))
            ]
        batch_retrieval_items, retrieval_routes, batch_all_sources = self._retrieve_route_batch(batch_data, retrieval_inputs, dense_queries=dense_queries)

        self.cleanup_retrieval()
        ranked_items = batch_retrieval_items
        if self.reranker_type:
            self.load_reranker()
            ranked_items = self.reranker.batch_rerank(
                user_ids,
                batch_retrieval_items,
                topk=self.rerank_topk,
                user_profiles=[data.get("user_profile") for data in batch_data],
                query_texts=retrieval_inputs,
                conversation_goals=[data.get("conversation_goal") for data in batch_data],
            )
            self.cleanup_reranker()
        ranked_items = [items if items else (batch_retrieval_items[idx][:1] if idx < len(batch_retrieval_items) and batch_retrieval_items[idx] else []) for idx, items in enumerate(ranked_items)]

        recommend_items = [self._build_recommendation_context(items, session_memory=session_memories[i], planned_query=planned_queries[i] if i < len(planned_queries) else None) for i, items in enumerate(ranked_items)]

        # Stage 2: Batch response generation
        self.load_lm()
        if hasattr(self.lm, 'batch_response_generation'):
            responses = self.lm.batch_response_generation(sys_prompts, session_memories, recommend_items)
        else:
            # Fallback to sequential generation if batch method not available
            responses = [self.lm.response_generation(sys_prompts[i], session_memories[i], recommend_items[i])
                        for i in range(len(batch_data))]

        # Prepare results
        results = []
        for i, data in enumerate(batch_data):
            results.append({
                "user_id": data.get('user_id'),
                "user_query": data['user_query'],
                "retrieval_items": batch_retrieval_items[i],
                "retrieval_route": retrieval_routes[i],
                "ranked_items": ranked_items[i],
                "recommend_item": recommend_items[i],
                "response": responses[i],
            })

        return results

    def _plan_cache_key(self, session_memory: List[Dict[str, Any]], user_id: Optional[str]) -> str:
        import hashlib
        content = json.dumps({"memory": session_memory, "user_id": user_id}, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()

    def _batch_plan_queries(self, session_memories: List[List[Dict[str, Any]]], batch_data: List[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if not self.enable_llm_query_planning:
            return [{} for _ in batch_data]
        cache_dir = os.path.join(self.cache_dir, "planner_cache")
        os.makedirs(cache_dir, exist_ok=True)
        planned: list[Dict[str, Any]] = [{}] * len(batch_data)
        uncached_indices: list[int] = []
        for i in range(len(batch_data)):
            cache_key = self._plan_cache_key(session_memories[i], batch_data[i].get("user_id"))
            cache_path = os.path.join(cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r") as f:
                        planned[i] = json.load(f)
                    continue
                except Exception:
                    pass
            uncached_indices.append(i)
        if uncached_indices:
            print(f"[planner] {len(batch_data) - len(uncached_indices)}/{len(batch_data)} loaded from cache, planning {len(uncached_indices)} remaining")
            self.load_lm()
            for i in uncached_indices:
                result = self._plan_retrieval_query(
                    session_memories[i],
                    user_id=batch_data[i].get("user_id"),
                    user_profile=batch_data[i].get("user_profile"),
                    conversation_goal=batch_data[i].get("conversation_goal"),
                )
                planned[i] = result
                cache_key = self._plan_cache_key(session_memories[i], batch_data[i].get("user_id"))
                cache_path = os.path.join(cache_dir, f"{cache_key}.json")
                try:
                    with open(cache_path, "w") as f:
                        json.dump(result, f)
                except Exception:
                    pass
        else:
            print(f"[planner] {len(batch_data)}/{len(batch_data)} loaded from cache (skip LLM)")
        return planned

    def _build_retrieval_input_from_plan(
        self,
        session_memory: List[Dict[str, Any]],
        planned_query: Dict[str, Any],
        user_id: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        conversation_goal: Optional[Dict[str, Any]] = None,
    ) -> str:
        llm_query = str(planned_query.get("bm25_query") or planned_query.get("query") or "").strip() if planned_query else ""
        if llm_query and self.llm_query_plan_mode in {"replace", "only", "llm_only"}:
            planned_lines = self._build_planned_query_lines(planned_query, llm_query=llm_query)
            return "\n".join(planned_lines) if planned_lines else llm_query
        if not self.enable_query_rewrite:
            query_parts = [f"{c.get('role')}: {c.get('content')}" for c in session_memory]
            query_parts = [p for p in query_parts if p.strip()]
            if user_id:
                query_parts.append(f"user_id: {user_id}")
            if planned_query:
                query_parts.extend(self._build_planned_query_lines(planned_query, llm_query=llm_query))
            raw_query = "\n".join(query_parts)
            return raw_query if raw_query.strip() else "music recommendation"
        latest_user_message = ""
        for turn in reversed(session_memory):
            if turn.get("role") == "user":
                latest_user_message = self._normalize_text_block(turn.get("content", ""), limit=280)
                break
        recent_block = self._build_recent_turns_block(session_memory, max_turns=5)
        query_parts = []
        if latest_user_message:
            query_parts.append(f"intent: {latest_user_message}")
        if recent_block:
            query_parts.append(f"recent_context:\n{recent_block}")
        if user_id:
            query_parts.append(f"user_id: {user_id}")
        merged_user_profile: Optional[Dict[str, Any]] = None
        if user_id:
            merged_user_profile = dict(self.user_db.id_to_profile(user_id))
        if user_profile:
            if merged_user_profile is None:
                merged_user_profile = {}
            merged_user_profile.update(user_profile)
        profile_block = self._format_context_block(
            "user_profile", merged_user_profile,
            preferred_keys=["user_id", "age", "age_group", "country_code", "country_name", "gender", "preferred_language", "preferred_musical_culture"],
        )
        if profile_block:
            query_parts.append(profile_block)
        goal_block = self._format_context_block(
            "conversation_goal", conversation_goal,
            preferred_keys=["category", "specificity", "listener_goal"],
        )
        if goal_block:
            query_parts.append(goal_block)
        if planned_query:
            query_parts.extend(self._build_planned_query_lines(planned_query, llm_query=llm_query))
        return "\n".join(query_parts) if query_parts else "music recommendation"

    def batch_retrieval(self, batch_data: List[Dict[str, Any]]) -> dict[str, Any]:
        if self.retrieval is None:
            raise RuntimeError("Retrieval model is not loaded. Call load_retrieval() before batch_retrieval().")

        sys_prompts = []
        session_memories = []
        user_ids = []

        for data in batch_data:
            user_query = data['user_query']
            user_id = data.get('user_id')
            conversation_goal = data.get('conversation_goal')
            session_memory = data['session_memory'].copy()
            session_memory.append({"role": "user", "content": user_query})
            sys_prompts.append(self._get_system_prompt(user_id, conversation_goal=conversation_goal, session_memory=session_memory))
            session_memories.append(session_memory)
            user_ids.append(user_id)

        planned_queries = self._batch_plan_queries(session_memories, batch_data)

        retrieval_inputs = [
            self._build_retrieval_input_from_plan(
                session_memories[i],
                planned_queries[i],
                user_id=batch_data[i].get("user_id"),
                user_profile=batch_data[i].get("user_profile"),
                conversation_goal=batch_data[i].get("conversation_goal"),
            )
            for i in range(len(batch_data))
        ]

        dense_queries = None
        if self.retrieval_type == "multi_source":
            dense_queries = [
                self._build_dense_query(session_memories[i], conversation_goal=batch_data[i].get("conversation_goal"))
                for i in range(len(batch_data))
            ]

        batch_retrieval_items, retrieval_routes, batch_all_sources = self._retrieve_route_batch(batch_data, retrieval_inputs, dense_queries=dense_queries)

        return {
            "sys_prompts": sys_prompts,
            "session_memories": session_memories,
            "user_ids": user_ids,
            "retrieval_inputs": retrieval_inputs,
            "planned_queries": planned_queries,
            "user_profiles": [data.get("user_profile") for data in batch_data],
            "conversation_goals": [data.get("conversation_goal") for data in batch_data],
            "retrieval_routes": retrieval_routes,
            "retrieval_items": batch_retrieval_items,
            "all_sources": batch_all_sources,
        }

    def batch_rerank(
        self,
        user_ids: list[str | None],
        retrieval_items: list[list[str]],
        user_profiles: list[dict[str, Any] | None] | None = None,
        query_texts: list[str | None] | None = None,
        conversation_goals: list[dict[str, Any] | None] | None = None,
    ) -> dict[str, Any]:
        if self.reranker is None:
            raise RuntimeError("Reranker is not loaded. Call load_reranker() before batch_rerank().")

        empty_retrieval_count = sum(1 for items in retrieval_items if not items)
        if empty_retrieval_count:
            print(
                f"Retrieval returned no candidates for {empty_retrieval_count}/{len(retrieval_items)} items before reranking."
            )

        ranked_items = self.reranker.batch_rerank(
            user_ids,
            retrieval_items,
            topk=self.rerank_topk,
            user_profiles=user_profiles,
            query_texts=query_texts,
            conversation_goals=conversation_goals,
        )
        fallback_count = 0
        normalized_ranked_items = []
        for idx, items in enumerate(ranked_items):
            if items:
                normalized_ranked_items.append(items)
                continue
            fallback_count += 1
            retrieval_fallback = retrieval_items[idx][:1] if idx < len(retrieval_items) and retrieval_items[idx] else []
            if retrieval_fallback:
                normalized_ranked_items.append(retrieval_fallback)
            else:
                normalized_ranked_items.append([])
        if fallback_count:
            print(f"Reranker returned no candidates for {fallback_count}/{len(ranked_items)} items; using retrieval fallback where available.")
        ranked_items = normalized_ranked_items
        recommend_items = [self.item_db.id_to_metadata(items[0]) if items else "" for items in ranked_items]

        return {
            "ranked_items": ranked_items,
            "recommend_items": recommend_items,
        }

    def batch_generation(
        self,
        sys_prompts: list[str],
        session_memories: list[list],
        recommend_items: list[str],
        ranked_items: list[list[str]] | None = None,
        planned_queries: list[dict] | None = None,
    ) -> list[str]:
        if self.lm is None:
            raise RuntimeError("LM is not loaded. Call load_lm() before batch_generation().")

        if ranked_items is not None:
            recommend_items = [
                self._build_recommendation_context(
                    items,
                    session_memory=session_memories[i],
                    planned_query=planned_queries[i] if planned_queries else None,
                )
                for i, items in enumerate(ranked_items)
            ]

        if hasattr(self.lm, 'batch_response_generation'):
            responses = self.lm.batch_response_generation(sys_prompts, session_memories, recommend_items)
        else:
            responses = [self.lm.response_generation(sys_prompts[i], session_memories[i], recommend_items[i])
                        for i in range(len(sys_prompts))]
        return responses

    def cleanup(self) -> None:
        self.cleanup_lm()
        self.cleanup_retrieval()
        self.cleanup_reranker()
        self.cleanup_user_to_item()
        self.cleanup_item_to_item()
