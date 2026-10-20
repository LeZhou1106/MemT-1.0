"""Register the published DeepSeek schema with native SGLang fields."""

from transformers import (
    AutoConfig,
    AutoTokenizer,
    DeepseekV3Config,
    PreTrainedTokenizerFast,
)

from memt_sglang.models.deepseek.config import native_config_fields, parse_memory_config


class MemTDeepseekV4Config(DeepseekV3Config):
    model_type = "memt_deepseek_v4"
    has_no_defaults_at_init = True

    def __init__(self, **kwargs):
        source = dict(kwargs, model_type=self.model_type)
        memory_config = parse_memory_config(source)
        super().__init__(**native_config_fields(source))
        self.memory_config = memory_config

    def to_dict(self):
        result = super().to_dict()
        result.pop("memory_config", None)
        return result


class MemTDeepseekTokenizer(PreTrainedTokenizerFast):
    """Installed data-only tokenizer for native input_ids serving.

    The client chat format handles the checkpoint's native encoder explicitly.
    Register the published tokenizer name so AutoTokenizer need not import the
    checkpoint's modeling_memt.py merely to tokenize or decode server requests.
    """


def register_deepseek_config():
    AutoConfig.register("memt_deepseek_v4", MemTDeepseekV4Config, exist_ok=True)
    AutoTokenizer.register(
        MemTDeepseekV4Config, fast_tokenizer_class=MemTDeepseekTokenizer, exist_ok=True
    )
