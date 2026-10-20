"""Pinned SGLang streaming-session ownership, used only on scheduler control.

The MemT memory slot and native KV slot have the same safe session ID. This
module does not enable radix sharing between different memory states.
"""

from memt_sglang.protocol import SessionError


def _cache(scheduler):
    from sglang.srt.session.streaming_session import StreamingSession

    cache = scheduler.tree_cache
    if isinstance(cache, StreamingSession):
        return cache
    embedded = getattr(cache, "session", None)
    if isinstance(embedded, StreamingSession):
        return embedded
    raise SessionError("KV retention requires --enable-streaming-session")


def open_kv(scheduler, session):
    from sglang.srt.managers.io_struct import OpenSessionReqInput

    _cache(scheduler)
    result = scheduler.session_controller.open(
        OpenSessionReqInput(
            session_id=session, capacity_of_str_len=0, streaming=True, timeout=None
        )
    )
    if not result.success:
        raise SessionError("Native KV session already exists")


def kv_status(scheduler, session):
    native = scheduler.session_controller.get(session)
    if native is None:
        raise SessionError("Native KV session expired")
    slot = _cache(scheduler).slots.get(session)
    return {
        "mode": "native_streaming_session",
        "inflight": native._inflight,
        "resident": bool(slot and slot.is_holding_kv),
        "cached_tokens": slot.kv_committed_len if slot else 0,
        # Observations, not a promise that all SWA historical rows exist.
        "req_pool_idx": slot.req_pool_idx if slot else None,
        "mamba_state_present": bool(slot and slot.mamba_pool_idx is not None),
    }


def close_kv(scheduler, session):
    from sglang.srt.managers.io_struct import CloseSessionReqInput

    if kv_status(scheduler, session)["inflight"]:
        raise SessionError("Wait for native request retirement before KV release")
    scheduler.session_controller.close(CloseSessionReqInput(session_id=session))
    if session in scheduler.session_controller:
        raise SessionError("Native KV close has not completed")


def reset_kv(scheduler, session):
    close_kv(scheduler, session)
    open_kv(scheduler, session)
