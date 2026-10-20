"""Memory configuration for the released MemT-1.0-Nano checkpoint."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

from memt_sglang.models.common.hf_config import (
    memt_model,
    validate_fixed_fields,
)

# All values are read from config.json. Pinned fields select the memory
# algorithm; the released kernels implement exactly these variants.
FIXED_QWEN_MEMORY_CONFIG = {
    "memory_update_mode": "token",
    "gdn_version": "stable_gdn2",
    "memory_use_short_conv": False,
    "memory_head_mode": "mha",
    "memory_read_scale": "k_inv_sqrt",
    "linear_attention_state_policy": "reset_per_chunk",
    "tail_policy": "commit",
    "output_gate_type": "silu",
    "output_gate_bias": False,
    "output_gate_softcap_enabled": True,
    "stable_gdn2_sqrt_erase_gate": True,
    "backbone_gdn_override": False,
}
QWEN_MEMORY_SIZE_FIELDS = (
    "num_memory_heads",
    "num_kv_heads",
    "chunk_tokens",
    "output_gate_softcap_limit",
)

# SGLang builds the backbone from config.json. Only the serving dtype is pinned;
# memory geometry is read here and checked against the loaded attention layers.
FIXED_QWEN_TEXT_CONFIG = {
    "model_type": "qwen3_5_moe_text",
    "dtype": "bfloat16",
    "quantization_config": None,
}
QWEN_TEXT_SIZE_FIELDS = (
    "hidden_size",
    "num_attention_heads",
    "head_dim",
    "num_hidden_layers",
    "full_attention_interval",
    "rms_norm_eps",
)


@dataclass(frozen=True)
class QwenMemoryConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    layers: tuple[int, ...]
    rms_norm_eps: float
    chunk_tokens: int
    output_gate_softcap: float

    @property
    def state_shapes(self):
        return {
            layer_index: (1, self.num_heads, self.head_dim, self.head_dim)
            for layer_index in self.layers
        }


def normalize_hf_config(config):
    """Preserve the input; model math still undergoes model-specific validation."""
    fields = deepcopy(config if isinstance(config, Mapping) else config.to_dict())
    if memt_model(fields) != "memt-nano":
        raise ValueError("config.json is not a MemT-1.0-Nano configuration")
    if not isinstance(fields.get("memt"), Mapping):
        raise ValueError("MemT release requires only the memt memory object")
    if fields.get("architectures") != ["MemTQwen3_5MoeForCausalLM"]:
        raise ValueError(
            "Expected the published MemTQwen3_5MoeForCausalLM architecture"
        )
    checkpoint = fields.get("memt_checkpoint_id")
    # Recent HF releases omit this descriptive label. The serving identity still
    # includes the exact config SHA256; never invent a checkpoint identifier.
    if "memt_checkpoint_id" in fields and (
        not isinstance(checkpoint, str) or not checkpoint.strip()
    ):
        raise ValueError("memt_checkpoint_id must be a nonempty string when present")
    # Use the installed engine config/model, never checkpoint Python model code.
    fields.pop("auto_map", None)
    return fields


def parse_memory_config(config) -> QwenMemoryConfig:
    fields = normalize_hf_config(config)
    backbone_config = fields.get("backbone_configs", {})
    if backbone_config.get("model_type") != "qwen3_5_moe":
        raise ValueError("MemT-1.0-Nano requires a qwen3_5_moe backbone")
    memory_config = fields.get("memt")
    text_config = backbone_config.get("text_config")
    validate_fixed_fields(
        memory_config,
        FIXED_QWEN_MEMORY_CONFIG,
        section="memt",
        exact=True,
        variable_fields=QWEN_MEMORY_SIZE_FIELDS,
    )
    validate_fixed_fields(
        text_config,
        FIXED_QWEN_TEXT_CONFIG,
        section="text_config",
        variable_fields=QWEN_TEXT_SIZE_FIELDS,
    )
    # memory_head_mode "mha": one memory head and one memory KV head per query head.
    num_heads = text_config["num_attention_heads"]
    if memory_config["num_memory_heads"] != num_heads or (
        memory_config["num_kv_heads"] != num_heads
    ):
        raise ValueError(
            "memt.num_memory_heads and memt.num_kv_heads must equal "
            "text_config.num_attention_heads"
        )
    attention_interval = text_config["full_attention_interval"]
    memory_layers = tuple(
        layer_index
        for layer_index in range(text_config["num_hidden_layers"])
        if (layer_index + 1) % attention_interval == 0
    )
    return QwenMemoryConfig(
        hidden_size=text_config["hidden_size"],
        num_heads=text_config["num_attention_heads"],
        head_dim=text_config["head_dim"],
        layers=memory_layers,
        rms_norm_eps=text_config["rms_norm_eps"],
        chunk_tokens=memory_config["chunk_tokens"],
        output_gate_softcap=memory_config["output_gate_softcap_limit"],
    )


def native_config_fields(config):
    source = deepcopy(normalize_hf_config(config))
    result = deepcopy(source["backbone_configs"]["text_config"])
    rope = result["rope_parameters"]
    result["rope_theta"] = rope["rope_theta"]
    result["partial_rotary_factor"] = rope["partial_rotary_factor"]
    result["rope_scaling"] = deepcopy(rope)
    result.update(
        model_type="memt_qwen3_5_moe",
        architectures=["MemTQwen3_5MoeForCausalLM"],
        memt=source["memt"],
        backbone_configs=source["backbone_configs"],
        torch_dtype="bfloat16",
        mtp_num_hidden_layers=0,
    )
    if "memt_runtime" in source:
        result["memt_runtime"] = source["memt_runtime"]
    if "memt_checkpoint_id" in source:
        result["memt_checkpoint_id"] = source["memt_checkpoint_id"]
    return result
