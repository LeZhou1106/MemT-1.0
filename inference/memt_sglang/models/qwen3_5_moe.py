"""SGLang registry entry for the Qwen memory model."""

from memt_sglang.runtime.hooks import install_sglang_hooks

install_sglang_hooks()

from memt_sglang.models.qwen.model import MemTQwen3_5MoeForCausalLM

EntryClass = MemTQwen3_5MoeForCausalLM
