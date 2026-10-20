"""Released DeepSeek configuration and native engine conversion."""

from copy import deepcopy
from dataclasses import dataclass

from memt_sglang.models.common.hf_config import validate_fixed_fields

# All values are read from config.json and SGLang builds the backbone from it.
# Only the published identity and the unquantized BF16 serving path are pinned;
# memory geometry is checked against the loaded attention layers.
FIXED_DEEPSEEK_CONFIG = {
    "model_type": "memt_deepseek_v4",
    "architectures": ["MemTDeepseekV4ForCausalLM"],
    "dtype": "bfloat16",
    "expert_dtype": "bf16",
    "quantization_config": None,
}
# Read by native_backbone_fields to translate HF names for SGLang.
DEEPSEEK_BACKBONE_FIELDS = (
    "hidden_size",
    "num_attention_heads",
    "layer_types",
    "mlp_layer_types",
    "compress_rates",
    "rope_parameters",
)

# Pinned fields select the memory algorithm; the released kernels implement
# exactly these variants.
FIXED_DEEPSEEK_MEMORY_CONFIG = {
    "gate_latent_norm": False,
    "gdn_version": "stable_gdn2",
    "memory_head_mode": "mha",
    "memory_read_scale": "k_inv_sqrt",
    "memory_update_mode": "token",
    "memory_use_short_conv": False,
    "output_gate_bias": True,
    "output_gate_softcap_enabled": True,
    "output_gate_type": "silu",
    "projection": "grouped_kv_low_rank",
    "stable_gdn2_sqrt_erase_gate": True,
    "tail_policy": "commit",
}
DEEPSEEK_MEMORY_SIZE_FIELDS = (
    "chunk_tokens",
    "gate_rank",
    "head_dim",
    "kv_groups",
    "kv_rank",
    "layers",
    "norm_eps",
    "num_kv_heads",
    "num_memory_heads",
    "output_gate_softcap_limit",
    "q_rank",
)


@dataclass(frozen=True)
class DeepSeekMemoryConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    q_rank: int
    kv_rank: int
    kv_groups: int
    gate_rank: int
    norm_eps: float
    output_softcap: float
    chunk_tokens: int
    layers: tuple[int, ...]

    @property
    def state_shapes(self):
        return {
            layer_index: (1, self.num_heads, self.head_dim, self.head_dim)
            for layer_index in self.layers
        }


def native_backbone_fields(config):
    """Translate HF names without changing the configuration's numerical values."""
    fields = deepcopy(dict(config))
    layer_types = fields["layer_types"]
    mlp_layer_types = fields["mlp_layer_types"]
    rope_config = fields["rope_parameters"]["compress"]
    # Sliding-window layers are uncompressed; config.json gives the other rates.
    compression_ratios = {"sliding_attention": 0, **fields["compress_rates"]}
    for field_name in (
        "memt",
        "auto_map",
        "layer_types",
        "mlp_layer_types",
        "compress_rates",
        "rope_parameters",
        "memt_runtime",
        "memt_checkpoint_id",
    ):
        fields.pop(field_name, None)
    fields.update(
        model_type="deepseek_v4",
        # SGLang 0.5.16 derives V4 KV geometry and defaults from this name.
        # The loader selects the independently registered MemT class separately.
        architectures=["DeepseekV4ForCausalLM"],
        compress_ratios=[compression_ratios[layer_type] for layer_type in layer_types],
        num_hash_layers=mlp_layer_types.count("hash_moe"),
        rope_scaling={
            field_name: rope_config[field_name]
            for field_name in (
                "beta_fast",
                "beta_slow",
                "factor",
                "original_max_position_embeddings",
                "type",
            )
        },
    )
    return fields


def normalize_hf_config(config):
    """Validate the published flat HF config before adapting it to SGLang."""
    validate_fixed_fields(
        config,
        FIXED_DEEPSEEK_CONFIG,
        section="deepseek",
        variable_fields=DEEPSEEK_BACKBONE_FIELDS + ("memt",),
    )
    validate_fixed_fields(
        config["memt"],
        FIXED_DEEPSEEK_MEMORY_CONFIG,
        section="deepseek.memt",
        exact=True,
        variable_fields=DEEPSEEK_MEMORY_SIZE_FIELDS,
    )
    if config["memt"]["num_memory_heads"] != config["num_attention_heads"] or (
        config["memt"]["num_kv_heads"] != config["num_attention_heads"]
    ):
        raise ValueError("Flash memory heads must match the backbone attention heads")
    if "backbone_configs" in config:
        raise ValueError("MemT-1.0-Flash requires the published flat HF configuration")
    normalized = {
        "model_type": "memt_deepseek_v4",
        "architectures": ["MemTDeepseekV4ForCausalLM"],
        "dtype": config["dtype"],
        "backbone_configs": native_backbone_fields(config),
        "memt": deepcopy(config["memt"]),
    }
    for field_name in ("memt_runtime", "memt_checkpoint_id"):
        if field_name in config:
            normalized[field_name] = deepcopy(config[field_name])
    return normalized


def parse_memory_config(config) -> DeepSeekMemoryConfig:
    fields = normalize_hf_config(
        config if isinstance(config, dict) else config.to_dict()
    )
    backbone_config = fields.get("backbone_configs")
    memory_config = fields.get("memt")
    return DeepSeekMemoryConfig(
        hidden_size=backbone_config["hidden_size"],
        num_heads=memory_config["num_memory_heads"],
        head_dim=memory_config["head_dim"],
        q_rank=memory_config["q_rank"],
        kv_rank=memory_config["kv_rank"],
        kv_groups=memory_config["kv_groups"],
        gate_rank=memory_config["gate_rank"],
        norm_eps=memory_config["norm_eps"],
        output_softcap=memory_config["output_gate_softcap_limit"],
        chunk_tokens=memory_config["chunk_tokens"],
        layers=tuple(memory_config["layers"]),
    )


def native_config_fields(config):
    """Return native backbone fields plus validated MemT memory metadata."""
    fields = normalize_hf_config(
        config if isinstance(config, dict) else config.to_dict()
    )
    result = deepcopy(dict(fields["backbone_configs"]))
    result.update(
        model_type="memt_deepseek_v4",
        backbone_configs=deepcopy(fields["backbone_configs"]),
        memt=deepcopy(fields["memt"]),
    )
    for field_name in ("memt_runtime", "memt_checkpoint_id"):
        if field_name in fields:
            result[field_name] = deepcopy(fields[field_name])
    return result
