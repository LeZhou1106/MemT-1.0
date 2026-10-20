"""Split an HF weight stream using an explicit model-specific key mapping."""

import torch


def load_hf_weights(weights, native_load, blocks, classify):
    """classify(name) returns (layer, parameter suffix), or None for native.

    Blocks must not be registered on the native model until this returns. The
    key mapper owns the published naming contract; this module guesses no keys.
    Native loading may mutate the native model on failure: discard that model.
    """
    expected = {
        (layer, name): tensor
        for layer, block in blocks.items()
        for name, tensor in block.state_dict().items()
    }
    found, seen, exhausted = {}, set(), False

    def native_stream():
        nonlocal exhausted
        for name, tensor in weights:
            if not isinstance(name, str) or name in seen:
                raise ValueError(f"Invalid or duplicate HF weight key: {name!r}")
            seen.add(name)
            key = classify(name)
            if key is None:
                yield name, tensor
                continue
            if key not in expected or key in found:
                raise ValueError(f"Unexpected or aliased memory weight: {name}")
            target = expected[key]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.bfloat16
                or target.dtype != torch.bfloat16
                or tensor.shape != target.shape
            ):
                raise ValueError(f"Memory weight shape/dtype mismatch: {name}")
            found[key] = tensor
        exhausted = True

    result = native_load(native_stream())
    if not exhausted:
        raise ValueError("Native loader did not consume the HF weight stream")
    missing = expected.keys() - found.keys()
    if missing:
        raise ValueError(f"Missing memory weights: {sorted(missing)[:5]}")
    # Validate the whole stream before changing any memory layer.
    for layer, block in blocks.items():
        block.load_state_dict(
            {name: found[layer, name] for name in block.state_dict()}, strict=True
        )
        block.eval()
    return result
