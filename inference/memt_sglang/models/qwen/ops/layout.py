"""Tile write events and map tokens to per-sample read snapshots."""

from dataclasses import dataclass

import torch

from memt_sglang.models.common.constants import WY_TILE_TOKENS

__all__ = ["WY_TILE_TOKENS", "ChunkPlan", "build_chunk_plan"]


_CACHE_LIMIT = 8


_cache: dict[tuple, tuple] = {}


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """Write-event tiles, read snapshots and per-sample scan offsets."""

    wy_tile_tokens: int
    num_event: int
    num_block: int
    num_sample: int
    num_chunk: int

    total_tokens: int

    event_slice: torch.Tensor

    chunk_indices: torch.Tensor

    sample_chunk_offsets: torch.Tensor

    chunk_block_ids: torch.Tensor

    chunk_is_block_start: torch.Tensor

    chunk_tok0: torch.Tensor

    chunk_len: torch.Tensor

    chunk_commits: torch.Tensor


def _build(
    read_slice: torch.Tensor,
    sample_slice: torch.Tensor,
    wy_tile_tokens: int,
    event_slice: torch.Tensor,
    commit_mask: torch.Tensor | None,
) -> ChunkPlan:
    """Build host metadata and transfer it to the boundary device."""
    device = read_slice.device
    read = read_slice.detach().to("cpu", torch.int64)
    sample = sample_slice.detach().to("cpu", torch.int64)
    event = event_slice.detach().to("cpu", torch.int64)
    total = int(event[-1])

    event_len = event[1:] - event[:-1]  # [N_event]
    n_chunk_per_event = (event_len + wy_tile_tokens - 1) // wy_tile_tokens  # ceil
    n_event = int(event_len.numel())

    event_chunk_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int64), torch.cumsum(n_chunk_per_event, 0))
    )
    num_chunk = int(event_chunk_offsets[-1])

    chunk_event = torch.repeat_interleave(
        torch.arange(n_event, dtype=torch.int64), n_chunk_per_event
    )
    chunk_sub = (
        torch.arange(num_chunk, dtype=torch.int64) - event_chunk_offsets[chunk_event]
    )
    chunk_tok0 = event[chunk_event] + chunk_sub * wy_tile_tokens
    chunk_len = torch.clamp(event[chunk_event + 1] - chunk_tok0, max=wy_tile_tokens)

    sample_chunk_offsets = event_chunk_offsets[torch.searchsorted(event, sample)]

    chunk_block_ids = torch.searchsorted(read, chunk_tok0, right=True) - 1
    chunk_is_block_start = chunk_tok0 == read[chunk_block_ids]

    if commit_mask is None:
        chunk_commits = torch.ones(num_chunk, dtype=torch.bool)
    else:
        commits = commit_mask.detach().to("cpu", torch.bool)
        chunk_commits = commits[chunk_event]

    def on_device(tensor, dtype=torch.int32):
        return tensor.to(device=device, dtype=dtype).contiguous()

    return ChunkPlan(
        wy_tile_tokens=wy_tile_tokens,
        num_event=n_event,
        num_block=int(read.numel()) - 1,
        num_sample=int(sample.numel()) - 1,
        num_chunk=num_chunk,
        total_tokens=total,
        event_slice=on_device(event),
        chunk_indices=on_device(torch.stack((chunk_event, chunk_sub), dim=-1)),
        sample_chunk_offsets=on_device(sample_chunk_offsets),
        chunk_block_ids=on_device(chunk_block_ids),
        chunk_is_block_start=on_device(chunk_is_block_start),
        chunk_tok0=on_device(chunk_tok0),
        chunk_len=on_device(chunk_len),
        chunk_commits=on_device(chunk_commits),
    )


def build_chunk_plan(
    turn_slice: torch.Tensor,
    sample_slice: torch.Tensor,
    wy_tile_tokens: int = WY_TILE_TOKENS,
    *,
    event_slice: torch.Tensor | None = None,
    commit_mask: torch.Tensor | None = None,
) -> ChunkPlan:
    """Cache a tiling plan for these boundary tensor identities."""
    events = turn_slice if event_slice is None else event_slice
    key = (
        id(turn_slice),
        id(sample_slice),
        id(events),
        id(commit_mask),
        wy_tile_tokens,
    )
    hit = _cache.get(key)
    if hit is not None:
        return hit[0]
    plan = _build(turn_slice, sample_slice, wy_tile_tokens, events, commit_mask)
    if len(_cache) >= _CACHE_LIMIT:
        _cache.pop(next(iter(_cache)))
    _cache[key] = (plan, turn_slice, sample_slice, events, commit_mask)
    return plan
