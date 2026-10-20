"""Memory event tiling with a fixed read snapshot per request."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChunkPlan:
    wy_tile_tokens: int
    num_event: int
    num_block: int
    num_sample: int
    num_chunk: int
    total_tokens: int
    event_slice: torch.Tensor
    chunk_indices: torch.Tensor
    sample_chunk_offsets: torch.Tensor
    chunk_tok0: torch.Tensor  # [num_chunk]: first token index in packed input
    chunk_len: torch.Tensor
    chunk_block_ids: torch.Tensor
    chunk_is_block_start: torch.Tensor
    chunk_commits: torch.Tensor


def single_commit_plan(
    length: int, wy_tile_tokens: int, device, *, chunk_tokens
) -> ChunkPlan:
    count = (length + wy_tile_tokens - 1) // wy_tile_tokens
    # Derive bounds from host shape metadata; never copy CUDA tensors to CPU.
    starts = torch.arange(count, dtype=torch.int32) * wy_tile_tokens
    events = [*range(0, length, chunk_tokens), length]
    indices = torch.stack(
        (starts // chunk_tokens, (starts % chunk_tokens) // wy_tile_tokens), -1
    )

    def on_device(value, dtype=torch.int32):
        return torch.as_tensor(value, dtype=dtype).to(device=device).contiguous()

    return ChunkPlan(
        wy_tile_tokens=wy_tile_tokens,
        num_event=len(events) - 1,
        num_block=1,
        num_sample=1,
        num_chunk=count,
        total_tokens=length,
        event_slice=on_device(events),
        chunk_indices=on_device(indices),
        sample_chunk_offsets=on_device([0, count]),
        chunk_tok0=on_device(starts),
        chunk_len=on_device((length - starts).clamp(max=wy_tile_tokens)),
        chunk_block_ids=on_device(torch.zeros_like(starts)),
        chunk_is_block_start=on_device(starts == 0, torch.bool),
        chunk_commits=on_device(torch.ones(count, dtype=torch.bool), torch.bool),
    )
