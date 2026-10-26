"""Immutable reader tiles cached per CUDA stream, without device-to-host reads."""

from dataclasses import dataclass
from functools import lru_cache

import torch

from memt_sglang.models.common.constants import WY_TILE_TOKENS


@dataclass(frozen=True, slots=True)
class QueryPlan:
    wy_tile_tokens: int
    num_chunk: int
    chunk_tok0: torch.Tensor
    chunk_len: torch.Tensor
    chunk_block_ids: torch.Tensor


@lru_cache(maxsize=64)
def _tile_values(lengths):
    starts, counts, blocks = [], [], []
    offset = 0
    for sample, length in enumerate(lengths):
        for local in range(0, length, WY_TILE_TOKENS):
            starts.append(offset + local)
            counts.append(min(WY_TILE_TOKENS, length - local))
            blocks.append(sample)
        offset += length
    return tuple(starts), tuple(counts), tuple(blocks)


@lru_cache(maxsize=64)
def _device_plan(lengths, stream):
    values = _tile_values(lengths)
    # Retain the creation stream with the immutable buffers. Reuse is confined
    # to that stream, so breakable callbacks never consume another stream's
    # asynchronous initialization. The bounded cache owns the tensor lifetime.
    with torch.cuda.stream(stream):
        tensors = tuple(
            torch.tensor(v, device=stream.device, dtype=torch.int32) for v in values
        )
    return QueryPlan(WY_TILE_TOKENS, len(values[0]), *tensors)


def query_plan(lengths, device):
    lengths = tuple(lengths)
    if not lengths or any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError("Query lengths must be positive CPU integers")
    return _device_plan(lengths, torch.cuda.current_stream(device))
