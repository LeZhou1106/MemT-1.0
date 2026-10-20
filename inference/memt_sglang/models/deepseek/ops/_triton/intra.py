# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Adapted from flash-linear-attention 0.5.2 (MIT). The original license is in
# memt_sglang/_vendor/fla/FLA_LICENSE; MemT modifications are licensed under
# PolyForm Noncommercial 1.0.0 (see LICENSE).

# MemT specialization removes all query/causal-output work from the writer.
import torch
import triton
import triton.language as tl

from memt_sglang._vendor.fla.ops.utils.op import exp2
from memt_sglang._vendor.fla.utils import IS_TF32_SUPPORTED
from memt_sglang.models.deepseek.ops._triton.wy import recompute_w_u

SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32" if IS_TF32_SUPPORTED else "ieee")


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T", "N"])
def chunk_gdn2_fwd_kernel_intra_token_parallel(
    k,
    g,
    b,
    Akk,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    UNIT_B: tl.constexpr,
):
    i_tg, i_hg = (tl.program_id(0).to(tl.int64), tl.program_id(1))
    if IS_VARLEN:
        i_n = 0
        left, right = (0, N)
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = i_tg // T * T
        i_t = i_tg % T
    if i_t >= T:
        return
    i_c = i_t // BT
    i_s = i_t % BT // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC
    k += bos * H * K
    g += bos * H * K
    b += bos * H * K
    Akk += bos * H * BC
    o_h = tl.arange(0, BH)
    o_k = tl.arange(0, BK)
    m_h = i_hg * BH + o_h < H
    m_k = o_k < K
    p_k = k + i_t * H * K + (i_hg * BH + o_h)[:, None] * K + o_k[None, :]
    p_g = g + i_t * H * K + (i_hg * BH + o_h)[:, None] * K + o_k[None, :]
    p_b = b + i_t * H * K + (i_hg * BH + o_h)[:, None] * K + o_k[None, :]
    b_k = tl.load(p_k, mask=m_h[:, None] & m_k[None, :], other=0.0).to(tl.float32)
    b_g = tl.load(p_g, mask=m_h[:, None] & m_k[None, :], other=0.0).to(tl.float32)
    b_b = (
        1.0
        if UNIT_B
        else tl.load(p_b, mask=m_h[:, None] & m_k[None, :], other=0.0).to(tl.float32)
    )
    b_k = b_k * b_b
    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        p_kj = k + j * H * K + (i_hg * BH + o_h)[:, None] * K + o_k[None, :]
        p_gj = g + j * H * K + (i_hg * BH + o_h)[:, None] * K + o_k[None, :]
        b_kj = tl.load(p_kj, mask=m_h[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_gj = tl.load(p_gj, mask=m_h[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_kgj = b_kj * exp2(b_g - b_gj)
        b_kgj = tl.where(m_k[None, :], b_kgj, 0.0)
        b_Akk = tl.sum(b_k * b_kgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)
        tl.store(
            Akk + i_t * H * BC + (i_hg * BH + o_h) * BC + j - i_ts,
            b_Akk.to(Akk.dtype.element_ty),
            mask=m_h,
        )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_gdn2_fwd_kernel_inter_solve_fused(
    k,
    g,
    b,
    Akkd,
    Akk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    UNIT_B: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
):
    i_t, i_bh = (tl.program_id(0).to(tl.int64), tl.program_id(1))
    i_b, i_h = (i_bh // H, i_bh % H)
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = (i_b * T, i_b * T + T)
    if i_t * BT >= T:
        return
    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC
    o_i = tl.arange(0, BC)
    m_tc0 = i_tc0 + o_i < T
    m_tc1 = i_tc1 + o_i < T
    m_tc2 = i_tc2 + o_i < T
    m_tc3 = i_tc3 + o_i < T
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_k0 = k + (i_tc0 + o_i)[:, None] * (H * K) + o_k[None, :]
        p_g0 = g + (i_tc0 + o_i)[:, None] * (H * K) + o_k[None, :]
        b_k0 = tl.load(p_k0, mask=m_tc0[:, None] & m_k[None, :], other=0.0).to(
            tl.float32
        )
        b_g0 = tl.load(p_g0, mask=m_tc0[:, None] & m_k[None, :], other=0.0).to(
            tl.float32
        )
        if i_tc1 < T:
            p_k1 = k + (i_tc1 + o_i)[:, None] * (H * K) + o_k[None, :]
            p_g1 = g + (i_tc1 + o_i)[:, None] * (H * K) + o_k[None, :]
            p_b1 = b + (i_tc1 + o_i)[:, None] * (H * K) + o_k[None, :]
            b_k1 = tl.load(p_k1, mask=m_tc1[:, None] & m_k[None, :], other=0.0).to(
                tl.float32
            )
            b_g1 = tl.load(p_g1, mask=m_tc1[:, None] & m_k[None, :], other=0.0).to(
                tl.float32
            )
            b_b1 = (
                1.0
                if UNIT_B
                else tl.load(p_b1, mask=m_tc1[:, None] & m_k[None, :], other=0.0).to(
                    tl.float32
                )
            )
            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            b_bk1 = b_b1 * b_k1
            b_Akk10 += tl.dot(b_bk1 * b_gqn, b_kgt)
            if i_tc2 < T:
                p_k2 = k + (i_tc2 + o_i)[:, None] * (H * K) + o_k[None, :]
                p_g2 = g + (i_tc2 + o_i)[:, None] * (H * K) + o_k[None, :]
                p_b2 = b + (i_tc2 + o_i)[:, None] * (H * K) + o_k[None, :]
                b_k2 = tl.load(p_k2, mask=m_tc2[:, None] & m_k[None, :], other=0.0).to(
                    tl.float32
                )
                b_g2 = tl.load(p_g2, mask=m_tc2[:, None] & m_k[None, :], other=0.0).to(
                    tl.float32
                )
                b_b2 = (
                    1.0
                    if UNIT_B
                    else tl.load(
                        p_b2, mask=m_tc2[:, None] & m_k[None, :], other=0.0
                    ).to(tl.float32)
                )
                b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0).to(
                    tl.float32
                )
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_bkg2 = b_b2 * b_k2 * b_gqn2
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
                b_Akk20 += tl.dot(b_bkg2, b_kgt)
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
                b_Akk21 += tl.dot(b_bkg2, b_kgt)
                if i_tc3 < T:
                    p_k3 = k + (i_tc3 + o_i)[:, None] * (H * K) + o_k[None, :]
                    p_g3 = g + (i_tc3 + o_i)[:, None] * (H * K) + o_k[None, :]
                    p_b3 = b + (i_tc3 + o_i)[:, None] * (H * K) + o_k[None, :]
                    b_k3 = tl.load(
                        p_k3, mask=m_tc3[:, None] & m_k[None, :], other=0.0
                    ).to(tl.float32)
                    b_g3 = tl.load(
                        p_g3, mask=m_tc3[:, None] & m_k[None, :], other=0.0
                    ).to(tl.float32)
                    b_b3 = (
                        1.0
                        if UNIT_B
                        else tl.load(
                            p_b3, mask=m_tc3[:, None] & m_k[None, :], other=0.0
                        ).to(tl.float32)
                    )
                    b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0).to(
                        tl.float32
                    )
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_bkg3 = b_b3 * b_k3 * b_gqn3
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                    b_Akk30 += tl.dot(b_bkg3, b_kgt)
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                    b_Akk31 += tl.dot(b_bkg3, b_kgt)
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                    b_Akk32 += tl.dot(b_bkg3, b_kgt)
    if i_tc1 < T:
        pass
    if i_tc2 < T:
        pass
    if i_tc3 < T:
        pass
    p_Akk00 = Akkd + (i_tc0 + o_i)[:, None] * (H * BC) + o_i[None, :]
    p_Akk11 = Akkd + (i_tc1 + o_i)[:, None] * (H * BC) + o_i[None, :]
    p_Akk22 = Akkd + (i_tc2 + o_i)[:, None] * (H * BC) + o_i[None, :]
    p_Akk33 = Akkd + (i_tc3 + o_i)[:, None] * (H * BC) + o_i[None, :]
    b_Ai00 = tl.load(p_Akk00, mask=m_tc0[:, None] & (o_i[None, :] < BC), other=0.0).to(
        tl.float32
    )
    b_Ai11 = tl.load(p_Akk11, mask=m_tc1[:, None] & (o_i[None, :] < BC), other=0.0).to(
        tl.float32
    )
    b_Ai22 = tl.load(p_Akk22, mask=m_tc2[:, None] & (o_i[None, :] < BC), other=0.0).to(
        tl.float32
    )
    b_Ai33 = tl.load(p_Akk33, mask=m_tc3[:, None] & (o_i[None, :] < BC), other=0.0).to(
        tl.float32
    )
    if not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]
        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        b_Ai33 = -tl.where(m_A, b_Ai33, 0)
        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.0)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2 * BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
            b_a22 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.0)
            b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
        for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
            b_a33 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.0)
            b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)
        b_Ai00 += m_I
        b_Ai11 += m_I
        b_Ai22 += m_I
        b_Ai33 += m_I
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    p_Akk00 = Akk + (i_tc0 + o_i)[:, None] * (H * BT) + o_i[None, :]
    p_Akk10 = Akk + (i_tc1 + o_i)[:, None] * (H * BT) + o_i[None, :]
    p_Akk11 = Akk + (i_tc1 + o_i)[:, None] * (H * BT) + (BC + o_i)[None, :]
    p_Akk20 = Akk + (i_tc2 + o_i)[:, None] * (H * BT) + o_i[None, :]
    p_Akk21 = Akk + (i_tc2 + o_i)[:, None] * (H * BT) + (BC + o_i)[None, :]
    p_Akk22 = Akk + (i_tc2 + o_i)[:, None] * (H * BT) + (2 * BC + o_i)[None, :]
    p_Akk30 = Akk + (i_tc3 + o_i)[:, None] * (H * BT) + o_i[None, :]
    p_Akk31 = Akk + (i_tc3 + o_i)[:, None] * (H * BT) + (BC + o_i)[None, :]
    p_Akk32 = Akk + (i_tc3 + o_i)[:, None] * (H * BT) + (2 * BC + o_i)[None, :]
    p_Akk33 = Akk + (i_tc3 + o_i)[:, None] * (H * BT) + (3 * BC + o_i)[None, :]
    tl.store(
        p_Akk00,
        b_Ai00.to(Akk.dtype.element_ty),
        mask=m_tc0[:, None] & (o_i[None, :] < BT),
    )
    tl.store(
        p_Akk10,
        b_Ai10.to(Akk.dtype.element_ty),
        mask=m_tc1[:, None] & (o_i[None, :] < BT),
    )
    tl.store(
        p_Akk11,
        b_Ai11.to(Akk.dtype.element_ty),
        mask=m_tc1[:, None] & ((BC + o_i)[None, :] < BT),
    )
    tl.store(
        p_Akk20,
        b_Ai20.to(Akk.dtype.element_ty),
        mask=m_tc2[:, None] & (o_i[None, :] < BT),
    )
    tl.store(
        p_Akk21,
        b_Ai21.to(Akk.dtype.element_ty),
        mask=m_tc2[:, None] & ((BC + o_i)[None, :] < BT),
    )
    tl.store(
        p_Akk22,
        b_Ai22.to(Akk.dtype.element_ty),
        mask=m_tc2[:, None] & ((2 * BC + o_i)[None, :] < BT),
    )
    tl.store(
        p_Akk30,
        b_Ai30.to(Akk.dtype.element_ty),
        mask=m_tc3[:, None] & (o_i[None, :] < BT),
    )
    tl.store(
        p_Akk31,
        b_Ai31.to(Akk.dtype.element_ty),
        mask=m_tc3[:, None] & ((BC + o_i)[None, :] < BT),
    )
    tl.store(
        p_Akk32,
        b_Ai32.to(Akk.dtype.element_ty),
        mask=m_tc3[:, None] & ((2 * BC + o_i)[None, :] < BT),
    )
    tl.store(
        p_Akk33,
        b_Ai33.to(Akk.dtype.element_ty),
        mask=m_tc3[:, None] & ((3 * BC + o_i)[None, :] < BT),
    )


def write_only_intra(k, v, g, b, w, plan):
    batch, tokens, heads, dim = k.shape
    # The triangular solve always has four sub-blocks; sizes follow the plan.
    token_block_size = plan.wy_tile_tokens
    chunk_block_size = token_block_size // 4
    diagonal = torch.empty(
        batch, tokens, heads, chunk_block_size, device=k.device, dtype=torch.float32
    )
    inverse = torch.zeros(
        batch, tokens, heads, token_block_size, device=k.device, dtype=k.dtype
    )
    head_block_size = 1 if dim > 256 else 2
    chunk_gdn2_fwd_kernel_intra_token_parallel[
        (tokens, triton.cdiv(heads, head_block_size))
    ](
        k,
        g,
        k if b is None else b,
        diagonal,
        plan.event_slice,
        plan.num_event,
        tokens,
        heads,
        dim,
        # SGLang may wrap Triton's Python helpers. Compute this launch-time
        # constant on the host; JIT dependency analysis must not inspect the
        # wrapped helper. The tile size and device arithmetic are unchanged.
        triton.next_power_of_2(dim),
        token_block_size,
        chunk_block_size,
        head_block_size,
        UNIT_B=b is None,
        num_warps=1,
        num_stages=1,
    )
    chunk_gdn2_fwd_kernel_inter_solve_fused[(plan.num_chunk, heads)](
        k=k,
        g=g,
        b=k if b is None else b,
        Akkd=diagonal,
        Akk=inverse,
        cu_seqlens=plan.event_slice,
        chunk_indices=plan.chunk_indices,
        T=tokens,
        H=heads,
        K=dim,
        BT=token_block_size,
        BC=chunk_block_size,
        BK=64,
        USE_SAFE_GATE=False,
        UNIT_B=b is None,
        num_warps=2,
        num_stages=2,
    )
    write_transform, value_transform, scaled_key = recompute_w_u(
        k, v, g, w, inverse, plan
    )
    return write_transform, value_transform, scaled_key, inverse
