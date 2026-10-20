"""Shared token boundaries and metadata checks at the packed memory entry."""

from itertools import accumulate

import torch


def boundaries(lengths, device):
    return torch.tensor([0, *accumulate(lengths)], device=device, dtype=torch.int32)


def validate_packed_inputs(block, hidden_states, read_states, write_states, lengths):
    """Check metadata once without reading or synchronizing device tensors."""
    if (
        hidden_states.ndim != 2
        or hidden_states.shape[1] != block.hidden_size
        or not hidden_states.is_cuda
        or hidden_states.dtype != torch.bfloat16
    ):
        raise ValueError("Memory expects CUDA BF16 hidden states [tokens, hidden_size]")
    if (
        not lengths
        or any(length <= 0 for length in lengths)
        or sum(lengths) != len(hidden_states)
    ):
        raise ValueError("Packed sample lengths must cover the hidden token rows")
    expected = (len(lengths), block.num_heads, block.head_dim, block.head_dim)
    if any(
        tuple(state.shape) != expected
        or state.dtype != torch.float32
        or state.device != hidden_states.device
        for state in (read_states, write_states)
    ):
        raise ValueError(f"Memory expects FP32 states {expected} on the hidden device")
