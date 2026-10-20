"""Register the published Qwen schema with native SGLang fields."""

from sglang.srt.configs.qwen3_5 import Qwen3_5MoeTextConfig
from transformers import AutoConfig

from memt_sglang.models.qwen.config import native_config_fields, parse_memory_config


class MemTQwen3_5MoeConfig(Qwen3_5MoeTextConfig):
    model_type = "memt_qwen3_5_moe"
    has_no_defaults_at_init = True

    def __init__(self, backbone_configs=None, memt=None, **kwargs):
        source = dict(
            kwargs,
            model_type=self.model_type,
            backbone_configs=backbone_configs,
            memt=memt,
        )
        memory_config = parse_memory_config(source)
        super().__init__(**native_config_fields(source))
        self.memory_config = memory_config

    def to_dict(self):
        result = super().to_dict()
        result.pop("memory_config", None)
        return result


def register_qwen_config():
    AutoConfig.register("memt_qwen3_5_moe", MemTQwen3_5MoeConfig, exist_ok=True)
