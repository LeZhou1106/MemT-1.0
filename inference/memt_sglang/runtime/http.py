"""Install explicit session control and native /generate admission."""

import copy

from memt_sglang.runtime.admission import MemoryRequestMiddleware
from memt_sglang.runtime.service import SessionService


def install_http(http_server, identity, *, dp_size):
    snapshot = copy.deepcopy(identity)
    service = SessionService(
        lambda: http_server.get_global_state().tokenizer_manager.memt_control,
        snapshot,
        dp_size=dp_size,
        native_stream_incremental=lambda: bool(
            http_server.get_global_state().tokenizer_manager.incremental_streaming_output
        ),
    )

    async def get_identity():
        return copy.deepcopy(snapshot)

    async def capabilities():
        manager = http_server.get_global_state().tokenizer_manager
        return {
            **service.capabilities(),
            "query_stream": True,
            "stream_format": "cumulative",
            "native_stream_format": "incremental"
            if manager.incremental_streaming_output
            else "cumulative",
            "kv_retention": bool(manager.server_args.enable_streaming_session),
        }

    http_server.app.add_api_route("/memt/identity", get_identity, methods=["GET"])
    http_server.app.add_api_route("/memt/capabilities", capabilities, methods=["GET"])
    http_server.app.add_middleware(MemoryRequestMiddleware, service=service)
    return service
