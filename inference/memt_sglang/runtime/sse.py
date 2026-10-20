"""Bounded incremental SSE framing; no model or HTTP framework dependencies."""

import json

from memt_sglang.runtime.config import MAX_SSE_EVENT_BYTES


class SSEDecoder:
    def __init__(self, limit=MAX_SSE_EVENT_BYTES):
        self.buffer = b""
        self.limit = limit

    def feed(self, data):
        self.buffer += data
        # Native SSE uses LF. Accept CRLF without splitting UTF-8 characters.
        self.buffer = self.buffer.replace(b"\r\n", b"\n")
        while b"\n\n" in self.buffer:
            frame, self.buffer = self.buffer.split(b"\n\n", 1)
            if len(frame) > self.limit:
                raise ValueError("SSE frame exceeds limit")
            lines = [
                line[5:].lstrip(b" ")
                for line in frame.split(b"\n")
                if line.startswith(b"data:")
            ]
            if lines:
                yield b"\n".join(lines).decode("utf-8")
        if len(self.buffer) > self.limit:
            raise ValueError("SSE frame exceeds limit")

    def finish(self):
        if self.buffer.strip():
            raise ValueError("Truncated SSE frame")


def event(value):
    value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return ("data: " + value + "\n\n").encode()
