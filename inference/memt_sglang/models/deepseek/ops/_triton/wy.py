# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Adapted from flash-linear-attention 0.5.2 (MIT). The original license is in
# memt_sglang/_vendor/fla/FLA_LICENSE; MemT modifications are licensed under
# PolyForm Noncommercial 1.0.0 (see LICENSE).

"""WY operands without materializing a channel-wise unit erase gate."""

import torch
import triton
import triton.language as tl

from memt_sglang._vendor.fla.ops.utils.op import exp2


@triton.jit(do_not_specialize=["T"])
def _unit_wy(
    k,
    v,
    g,
    w_gate,
    A,
    ww,
    uu,
    kg,
    offsets,
    chunks,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr = 64,
    BD: tl.constexpr = 64,
):
    chunk, head = tl.program_id(0).to(tl.int64), tl.program_id(1)
    event = tl.load(chunks + 2 * chunk).to(tl.int32)
    ci = tl.load(chunks + 2 * chunk + 1).to(tl.int64)
    bos = tl.load(offsets + event).to(tl.int32)
    T = tl.load(offsets + event + 1).to(tl.int32) - bos
    t = ci * BT + tl.arange(0, BT)
    a = tl.arange(0, BT)
    pa = A + (bos * H + head) * BT + t[:, None] * H * BT + a[None, :]
    inv = tl.load(pa, mask=t[:, None] < T, other=0)
    for di in range(tl.cdiv(D, BD)):
        d = di * BD + tl.arange(0, BD)
        ix = ((bos + t[:, None]) * H + head) * D + d[None, :]
        valid = (t[:, None] < T) & (d[None, :] < D)
        vv = tl.load(v + ix, mask=valid, other=0)
        gate = tl.load(w_gate + ix, mask=valid, other=0)
        value = (vv * gate).to(vv.dtype)
        u = tl.dot(inv, value)
        tl.store(uu + ix, u.to(uu.dtype.element_ty), mask=valid)
    for di in range(tl.cdiv(D, BD)):
        d = di * BD + tl.arange(0, BD)
        ix = ((bos + t[:, None]) * H + head) * D + d[None, :]
        valid = (t[:, None] < T) & (d[None, :] < D)
        kk = tl.load(k + ix, mask=valid, other=0)
        gg = tl.load(g + ix, mask=valid, other=0).to(tl.float32)
        # Match the original multiplication's FP32 intermediate and BF16 cast.
        key = kk.to(tl.float32) * exp2(gg)
        tail = min(ci * BT + BT, T) - 1
        gn = tl.load(g + ((bos + tail) * H + head) * D + d, mask=d < D, other=0).to(
            tl.float32
        )
        key_tail = kk * tl.where(t[:, None] < T, exp2(gn[None, :] - gg), 0)
        tl.store(kg + ix, key_tail.to(kg.dtype.element_ty), mask=valid)
        w = tl.dot(inv, key.to(kk.dtype))
        tl.store(ww + ix, w.to(ww.dtype.element_ty), mask=valid)


def recompute_w_u(k, v, g, w, inverse, plan):
    write_transform, value_transform, scaled_key = (
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(k),
    )
    # Both token coverage and inverse row stride must use the plan tile.
    _unit_wy[(plan.num_chunk, k.shape[2])](
        k,
        v,
        g,
        w,
        inverse,
        write_transform,
        value_transform,
        scaled_key,
        plan.event_slice,
        plan.chunk_indices,
        k.shape[1],
        k.shape[2],
        k.shape[3],
        BT=plan.wy_tile_tokens,
        num_warps=4,
        num_stages=2,
    )
    return write_transform, value_transform, scaled_key
