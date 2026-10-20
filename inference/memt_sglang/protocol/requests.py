"""Parse MemT request IDs and validate untrusted generation payloads."""

import re
from dataclasses import dataclass

from .constants import MAX_IDENTIFIER_LENGTH

_IDENTIFIER_PATTERN = rf"[A-Za-z0-9_.-]{{1,{MAX_IDENTIFIER_LENGTH}}}\Z"
_IDENTIFIER = re.compile(_IDENTIFIER_PATTERN)


def validate_session_id(session):
    if not isinstance(session, str) or not _IDENTIFIER.fullmatch(session):
        raise ValueError("Invalid MemT session ID")


@dataclass(frozen=True)
class MemoryRequest:
    session: str
    operation: str
    nonce: str
    length: int | None = None

    @classmethod
    def from_rid(cls, rid: str) -> "MemoryRequest":
        if not isinstance(rid, str) or len(parts := rid.split("|")) != 4:
            raise ValueError("Expected memt|session|operation|nonce")
        prefix, session, action, nonce = parts
        if prefix != "memt" or not all(
            _IDENTIFIER.fullmatch(value) for value in (session, nonce)
        ):
            raise ValueError("Invalid MemT session or nonce")
        operation, separator, argument = action.partition(":")
        if operation == "commit":
            if not separator or not re.fullmatch(r"[1-9][0-9]{0,8}", argument):
                raise ValueError("commit requires a positive bounded token count")
            length = int(argument)
            return cls(session, operation, nonce, length=length)
        if operation != "query" or separator:
            raise ValueError("Unknown MemT operation")
        return cls(session, operation, nonce)

    @property
    def rid(self) -> str:
        action = self.operation
        if self.operation == "commit":
            action += f":{self.length}"
        return f"memt|{self.session}|{action}|{self.nonce}"


def is_memt_rid(rid) -> bool:
    # Reserve malformed MemT-prefixed RIDs too: never silently bypass validation.
    return isinstance(rid, str) and rid.startswith("memt")


def validate_payload(
    payload: dict, *, max_prefill_tokens: int | None = None
) -> MemoryRequest | None:
    rid = payload.get("rid")
    if isinstance(rid, list):
        if any(is_memt_rid(item) for item in rid):
            raise ValueError("MemT HTTP requests must be unbatched")
        return None
    if not is_memt_rid(rid):
        return None
    request = MemoryRequest.from_rid(rid)
    tokens = payload.get("input_ids")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int or token < 0 for token in tokens)
        or any(payload.get(key) is not None for key in ("text", "input_embeds"))
    ):
        raise ValueError("MemT requires one nonempty input_ids list")
    if max_prefill_tokens is not None and len(tokens) > max_prefill_tokens:
        raise ValueError(f"MemT input exceeds max_prefill_tokens={max_prefill_tokens}")
    params = payload.get("sampling_params")
    if not isinstance(params, dict) or type(params.get("max_new_tokens")) is not int:
        raise ValueError("MemT requires explicit integer max_new_tokens")
    if params.get("ignore_eos", False) is not False:
        raise ValueError("MemT does not support ignore_eos")
    if (
        params["max_new_tokens"] < 0
        or type(params.get("n", 1)) is not int
        or params.get("n", 1) != 1
    ):
        raise ValueError("MemT requires n=1 and max_new_tokens>=0")
    if request.operation == "commit":
        if params["max_new_tokens"] != 0:
            raise ValueError("commit requires max_new_tokens=0")
        if any(
            payload.get(key)
            for key in (
                "return_logprob",
                "return_hidden_states",
                "return_pooled_hidden_states",
            )
        ):
            raise ValueError("commit provides no output scores or hidden states")
    if request.operation == "commit" and len(tokens) != request.length:
        raise ValueError("commit RID length must equal the actual input token count")
    if type(payload.get("stream", False)) is not bool:
        raise ValueError("stream must be a boolean")
    if payload.get("stream") and request.operation != "query":
        raise ValueError("Only queries support streaming")
    session_params = payload.get("session_params")
    if session_params is not None and (
        request.operation != "query" or session_params != {"id": request.session}
    ):
        raise ValueError("Only server-owned append-only session_params are supported")
    if any(
        payload.get(key) is not None
        for key in ("image_data", "video_data", "audio_data")
    ):
        raise ValueError("MemT supports text token inputs only")
    return request


def validate_boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value
