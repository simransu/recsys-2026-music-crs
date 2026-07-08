import json
import re
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    _LMFE_AVAILABLE = True
except ImportError:
    JsonSchemaParser = None
    build_transformers_prefix_allowed_tokens_fn = None
    _LMFE_AVAILABLE = False


class QWEN3_MODEL:
    def __init__(self, model_name="Qwen/Qwen3-8B", device="cuda", attn_implementation="sdpa", dtype=torch.bfloat16):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print(f"[QWEN3_MODEL] loading {model_name} on {device}")
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = dtype if self.device == "cuda" else torch.float32
        self.attn_implementation = attn_implementation
        self.lm, self.tokenizer = self._load_model()
        self.lm.eval()

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        lm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        lm.to(self.device)
        return lm, tokenizer

    def _format_chat_history(self, sys_prompt, chat_history: list, recommend_item: str, enable_thinking: bool = True):
        chat_data = [{"role": "system", "content": sys_prompt}]
        chat_data += chat_history
        chat_data += [{"role": "user", "content": recommend_item}]
        try:
            chat_template = self.tokenizer.apply_chat_template(
                chat_data,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Older transformers versions don't support enable_thinking
            chat_template = self.tokenizer.apply_chat_template(
                chat_data,
                tokenize=False,
                add_generation_prompt=True,
            )
        return chat_template

    def _format_messages(self, messages: list[dict[str, str]], enable_thinking: bool = True) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _extract_after_think(self, text: str) -> str:
        """Return text after </think>. If think block is incomplete, strip it entirely."""
        if "<think>" not in text:
            return text.strip()
        match = re.search(r"</think>\s*", text, flags=re.DOTALL)
        if match:
            return text[match.end():].strip()
        # Incomplete think block — discard everything
        return ""

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, object]:
        cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()
        if not cleaned:
            return {}
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                parsed = json.loads(cleaned)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    _INPUT_MAX = 2048
    _OUTPUT_MAX = 1024
    _CHAT_HISTORY_TURNS = 6
    _PLANNER_JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "bm25_query": {"type": "string"},
            "artist_names": {"type": "array", "items": {"type": "string"}},
            "track_titles": {"type": "array", "items": {"type": "string"}},
            "album_names": {"type": "array", "items": {"type": "string"}},
            "genre_tags": {"type": "array", "items": {"type": "string"}},
            "mood_phrases": {"type": "array", "items": {"type": "string"}},
            "year_terms": {"type": "array", "items": {"type": "string"}},
            "negative_constraints": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "bm25_query", "artist_names", "track_titles", "album_names",
            "genre_tags", "mood_phrases", "year_terms", "negative_constraints",
        ],
        "additionalProperties": False,
    }

    def response_generation(self, sys_prompt: str, chat_history: list, recommend_item: str, max_new_tokens=None, response_format=None):
        if max_new_tokens is None:
            max_new_tokens = self._OUTPUT_MAX
        trimmed_history = chat_history[-self._CHAT_HISTORY_TURNS:] if chat_history else []
        chat_text = self._format_chat_history(sys_prompt, trimmed_history, recommend_item)
        self.tokenizer.truncation_side = "left"
        token_inputs = self.tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=self._INPUT_MAX)
        input_ids = token_inputs.input_ids.to(self.device)
        attention_mask = token_inputs.attention_mask.to(self.device)
        with torch.inference_mode():
            outputs = self.lm.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_text = self.tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        del input_ids, attention_mask, token_inputs, outputs
        if self.device == "cuda":
            torch.cuda.empty_cache()
        result = self._extract_after_think(generated_text)
        if not result:
            # Think block was truncated — retry without thinking to guarantee a response
            chat_text = self._format_chat_history(sys_prompt, trimmed_history, recommend_item, enable_thinking=False)
            token_inputs = self.tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=self._INPUT_MAX)
            input_ids = token_inputs.input_ids.to(self.device)
            attention_mask = token_inputs.attention_mask.to(self.device)
            with torch.inference_mode():
                outputs = self.lm.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            result = self.tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            del input_ids, attention_mask, token_inputs, outputs
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return result

    def batch_response_generation(self, sys_prompts: list[str], chat_histories: list[list], recommend_items: list[str], max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self._OUTPUT_MAX
        formatted_chats = [
            self._format_chat_history(sp, ch[-self._CHAT_HISTORY_TURNS:] if ch else [], ri)
            for sp, ch, ri in zip(sys_prompts, chat_histories, recommend_items)
        ]
        self.tokenizer.truncation_side = "left"
        token_inputs = self.tokenizer(
            formatted_chats,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._INPUT_MAX,
        )
        input_ids = token_inputs.input_ids.to(self.device)
        attention_mask = token_inputs.attention_mask.to(self.device)
        with torch.inference_mode():
            outputs = self.lm.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_texts = self.tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)
        del input_ids, attention_mask, token_inputs, outputs
        if self.device == "cuda":
            torch.cuda.empty_cache()
        results = [self._extract_after_think(t) for t in generated_texts]
        # Retry empty responses individually without thinking
        for idx, result in enumerate(results):
            if not result:
                results[idx] = self.response_generation(
                    sys_prompts[idx],
                    chat_histories[idx],
                    recommend_items[idx],
                    max_new_tokens=max_new_tokens,
                )
        return results

    def plan_retrieval_query(self, sys_prompt: str, query_context: str, max_new_tokens: int = 1024) -> dict[str, object]:
        if not _LMFE_AVAILABLE:
            raise RuntimeError(
                "lm-format-enforcer is not available in this environment, so the Qwen planner cannot produce constrained JSON."
            )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query_context},
        ]
        chat_text = self._format_messages(messages, enable_thinking=False)
        self.tokenizer.truncation_side = "left"
        token_inputs = self.tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=self._INPUT_MAX)
        input_ids = token_inputs.input_ids.to(self.device)
        attention_mask = token_inputs.attention_mask.to(self.device)
        parser = JsonSchemaParser(self._PLANNER_JSON_SCHEMA)
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)
        with torch.inference_mode():
            outputs = self.lm.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_fn,
            )
        raw_text = self.tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        del input_ids, attention_mask, token_inputs, outputs
        if self.device == "cuda":
            torch.cuda.empty_cache()
        parsed = self._extract_json_object(raw_text)
        return {
            "raw_text": raw_text,
            "parsed": parsed,
        }

    def cleanup(self) -> None:
        if hasattr(self, "lm"):
            self.lm.to("cpu")
            del self.lm
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
