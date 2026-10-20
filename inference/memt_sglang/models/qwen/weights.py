"""Published Qwen memory keys and streaming native checkpoint conversion."""

import re

_MEMORY = re.compile(
    r"model\.language_model\.layers\.(0|[1-9][0-9]*)\.self_attn\.memt\.(.+)"
)


def classify_memory(name):
    match = _MEMORY.fullmatch(name)
    if match:
        return int(match[1]), match[2]
    if "memt" in name.split("."):
        raise ValueError(f"Malformed Qwen memory weight key: {name}")
    return None


def native_weights(weights):
    seen = set()
    for name, tensor in weights:
        if name.startswith(("model.visual.", "visual.", "mtp.", "model.mtp.")):
            continue
        if name.startswith("model.language_model."):
            name = "model." + name.removeprefix("model.language_model.")
        if name == "model.lm_head.weight":
            name = "lm_head.weight"
        name = name.removeprefix("model.")
        if name in seen:
            raise ValueError(f"Aliased Qwen native weight: {name}")
        seen.add(name)
        yield name, tensor
