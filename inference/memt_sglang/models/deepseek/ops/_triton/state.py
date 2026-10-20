"""Scan WY write tiles into FP32 memory snapshots and final states."""

import torch
import triton
import triton.language as tl

from memt_sglang.models.deepseek.ops.layout import ChunkPlan


@triton.jit
def _fwd_h_kernel(
    kg,
    w_wy,
    u_wy,
    gk,
    snaps,
    h0,  # [N_sample, H, K, V] fp32
    ht,  # [N_sample, H, K, V] fp32
    sample_chunk_offsets,  # [N_sample + 1]
    chunk_tok0,  # [NT]
    chunk_len,  # [NT]
    chunk_block_ids,
    chunk_is_block_start,
    chunk_commits,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_H0: tl.constexpr,
    STORE_HT: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v, i_nh = pid % NV, (pid // NV).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_t = tl.arange(0, BT)
    m_k, m_v = o_k < K, o_v < V
    m_kv = m_k[:, None] & m_v[None, :]
    d_kv = o_k[:, None] * V + o_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_H0:
        b_h += tl.load(
            h0 + (i_n * H + i_h).to(tl.int64) * K * V + d_kv, mask=m_kv, other=0.0
        )

    c0 = tl.load(sample_chunk_offsets + i_n).to(tl.int64)
    c1 = tl.load(sample_chunk_offsets + i_n + 1).to(tl.int64)
    for c in range(c0, c1):
        i_block = tl.load(chunk_block_ids + c).to(tl.int64)
        is_block_start = tl.load(chunk_is_block_start + c)
        tok0 = tl.load(chunk_tok0 + c).to(tl.int64)
        clen = tl.load(chunk_len + c)

        tl.store(
            snaps + (i_block * H + i_h) * K * V + d_kv,
            b_h,
            mask=m_kv & (is_block_start != 0),
        )

        row = tok0 + o_t
        m_t = o_t < clen
        m_tk = m_t[:, None] & m_k[None, :]
        m_tv = m_t[:, None] & m_v[None, :]
        d_tk = (row[:, None] * H + i_h) * K + o_k[None, :]
        d_tv = (row[:, None] * H + i_h) * V + o_v[None, :]

        # v_new = u_wy − w_wy @ S
        b_w = tl.load(w_wy + d_tk, mask=m_tk, other=0.0)
        b_u = tl.load(u_wy + d_tv, mask=m_tv, other=0.0)
        b_vn = b_u - tl.dot(b_w, b_h.to(b_w.dtype))

        commit = tl.load(chunk_commits + c) != 0
        # S ← Diag(2^g_last) S + kgᵀ v_new
        b_gl = tl.load(
            gk + ((tok0 + clen - 1) * H + i_h) * K + o_k, mask=m_k, other=0.0
        )
        b_h = tl.where(commit, b_h * tl.exp2(b_gl.to(tl.float32))[:, None], b_h)
        b_k = tl.load(
            kg + (row[None, :] * H + i_h) * K + o_k[:, None],
            mask=m_k[:, None] & m_t[None, :],
            other=0.0,
        )
        b_h = tl.where(commit, b_h + tl.dot(b_k, b_vn.to(b_k.dtype)), b_h)

    if STORE_HT:
        tl.store(ht + (i_n * H + i_h).to(tl.int64) * K * V + d_kv, b_h, mask=m_kv)


def _grid_and_blocks(plan: ChunkPlan, H: int, K: int, V: int, dtype):
    key_block_size = max(16, triton.next_power_of_2(K))
    value_block_size = 16 if K > 256 and dtype != torch.bfloat16 else 32
    stages = 3 if K == 256 and H <= 16 and dtype == torch.bfloat16 else 1

    def grid(meta):
        return (triton.cdiv(V, value_block_size) * plan.num_sample * H,)

    return grid, key_block_size, value_block_size, stages


def memt_gdn2_fwd_h(
    kg: torch.Tensor,
    w_wy: torch.Tensor,
    u_wy: torch.Tensor,
    gk: torch.Tensor,
    plan: ChunkPlan,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
):
    """Scan token updates and return snapshots and candidate states."""
    _, _length, num_heads, key_dim = kg.shape
    value_dim = u_wy.shape[-1]
    grid, key_block_size, value_block_size, stages = _grid_and_blocks(
        plan, num_heads, key_dim, value_dim, kg.dtype
    )

    snaps = kg.new_zeros(
        plan.num_block, num_heads, key_dim, value_dim, dtype=torch.float32
    )
    final_state = (
        kg.new_zeros(
            plan.num_sample, num_heads, key_dim, value_dim, dtype=torch.float32
        )
        if output_final_state
        else None
    )

    _fwd_h_kernel[grid](
        kg=kg,
        w_wy=w_wy,
        u_wy=u_wy,
        gk=gk,
        snaps=snaps,
        h0=initial_state,
        ht=final_state,
        sample_chunk_offsets=plan.sample_chunk_offsets,
        chunk_tok0=plan.chunk_tok0,
        chunk_len=plan.chunk_len,
        chunk_block_ids=plan.chunk_block_ids,
        chunk_is_block_start=plan.chunk_is_block_start,
        chunk_commits=plan.chunk_commits,
        H=num_heads,
        K=key_dim,
        V=value_dim,
        BT=plan.wy_tile_tokens,
        BK=key_block_size,
        BV=value_block_size,
        num_warps=8 if key_dim > 256 else 4,
        num_stages=stages,
        USE_H0=initial_state is not None,
        STORE_HT=output_final_state,
    )
    return snaps, final_state
