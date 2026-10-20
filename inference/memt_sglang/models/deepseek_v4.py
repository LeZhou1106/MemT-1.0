"""SGLang registry entry for the DeepSeek memory model."""

from memt_sglang.runtime.hooks import install_sglang_hooks

install_sglang_hooks()

from memt_sglang.models.deepseek.model import MemTDeepseekV4ForCausalLM

EntryClass = MemTDeepseekV4ForCausalLM
