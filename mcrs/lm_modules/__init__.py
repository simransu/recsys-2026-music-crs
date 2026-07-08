from .llama import LLAMA_MODEL
from .qwen3 import QWEN3_MODEL

def load_lm_module(lm_type, device, attn_implementation, dtype):
    if not isinstance(lm_type, str) or not lm_type.strip():
        raise ValueError(f"Unsupported LM type: {lm_type}")
    if "qwen3" in lm_type.lower():
        return QWEN3_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
    return LLAMA_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
