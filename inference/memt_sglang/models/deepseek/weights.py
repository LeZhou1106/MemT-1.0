"""Published MemT-1.0-Flash weight names and native backbone mapping."""

import re

_MEMORY = re.compile(r"(?:model\.)?layers\.(0|[1-9][0-9]*)\.self_attn\.memt\.(.+)")


def classify_memory(name):
    match = _MEMORY.fullmatch(name)
    if match:
        return int(match[1]), match[2]
    if "memt" in name.split("."):
        raise ValueError(f"Malformed DeepSeek memory weight key: {name}")
    return None


def native_name(name):
    """Map the published HF namespace to SGLang's native checkpoint namespace."""
    if name.startswith("model.hc_head.hc_"):
        return name.replace("model.hc_head.hc_", "model.hc_head_")
    if name.endswith(".self_attn.sinks"):
        return name.removesuffix("sinks") + "attn_sink"
    for source, target in (("attn_hc.", "hc_attn_"), ("ffn_hc.", "hc_ffn_")):
        name = name.replace("." + source, "." + target)
    # HF nests the indexer inside the main compressor; native owns it on attention.
    name = name.replace(".self_attn.compressor.indexer.", ".self_attn.indexer.")
    if ".self_attn.indexer." in name:
        prefix, suffix = name.split(".self_attn.indexer.", 1)
        mapping = {
            "gate_proj.weight": "compressor.wgate.weight",
            "kv_norm.weight": "compressor.norm.weight",
            "kv_proj.weight": "compressor.wkv.weight",
            "position_bias": "compressor.ape",
            "q_b_proj.weight": "wq_b.weight",
            "scorer.weights_proj.weight": "weights_proj.weight",
        }
        if suffix not in mapping:
            raise ValueError(f"Unknown HF Flash indexer tensor: {name}")
        return prefix + ".self_attn.indexer." + mapping[suffix]
    for source, target in (
        ("q_a_proj", "wq_a"),
        ("q_b_proj", "wq_b"),
        ("q_a_norm", "q_norm"),
        ("kv_proj", "wkv"),
        ("o_a_proj", "wo_a"),
        ("o_b_proj", "wo_b"),
    ):
        name = name.replace(".self_attn." + source + ".", ".self_attn." + target + ".")
    for source, target in (
        ("gate_proj.weight", "wgate.weight"),
        ("kv_proj.weight", "wkv.weight"),
        ("kv_norm.weight", "norm.weight"),
        ("position_bias", "ape"),
    ):
        name = name.replace(".compressor." + source, ".compressor." + target)
    return name


def native_weights(weights, config):
    """Unpack expert views without changing values or rewriting checkpoint shards."""
    num_experts, intermediate_size, hidden_size = (
        config.n_routed_experts,
        config.moe_intermediate_size,
        config.hidden_size,
    )
    for name, tensor in weights:
        if name.endswith(".mlp.experts.gate_up_proj"):
            if tuple(tensor.shape) != (num_experts, 2 * intermediate_size, hidden_size):
                raise ValueError(f"Invalid HF packed expert shape: {name}")
            prefix = name.removesuffix("gate_up_proj")
            for expert_index in range(num_experts):
                yield (
                    f"{prefix}{expert_index}.gate_proj.weight",
                    tensor[expert_index, :intermediate_size],
                )
                yield (
                    f"{prefix}{expert_index}.up_proj.weight",
                    tensor[expert_index, intermediate_size:],
                )
        elif name.endswith(".mlp.experts.down_proj"):
            if tuple(tensor.shape) != (num_experts, hidden_size, intermediate_size):
                raise ValueError(f"Invalid HF packed expert shape: {name}")
            prefix = name.removesuffix("down_proj")
            for expert_index in range(num_experts):
                yield f"{prefix}{expert_index}.down_proj.weight", tensor[expert_index]
        else:
            yield native_name(name), tensor
