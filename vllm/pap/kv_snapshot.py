# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocator-safe PAP KV snapshot service outside the EngineCore queue."""

from __future__ import annotations

import asyncio
import json
import os
import socketserver
import struct
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

_HEADER = struct.Struct("!I")
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _read_exact(stream: Any, size: int) -> bytes:
    value = stream.read(size)
    if value is None or len(value) != size:
        raise EOFError("truncated PAP KV snapshot message")
    return value


class _SnapshotRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server = self.server
        if not isinstance(server, _SnapshotUnixServer):
            raise RuntimeError("invalid PAP KV snapshot server")
        try:
            size = _HEADER.unpack(_read_exact(self.rfile, _HEADER.size))[0]
            if not 0 < size <= _MAX_MESSAGE_BYTES:
                raise ValueError("invalid PAP KV snapshot message size")
            request = json.loads(_read_exact(self.rfile, size))
            if not isinstance(request, dict) or not isinstance(
                request.get("payload"), dict
            ):
                raise ValueError("invalid PAP KV snapshot request")
            result = server.callback(
                str(request.get("operation", "")), request["payload"]
            )
            response = {"ok": True, "result": result}
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        payload = json.dumps(response, separators=(",", ":")).encode()
        self.wfile.write(_HEADER.pack(len(payload)))
        self.wfile.write(payload)


class _SnapshotUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(
        self,
        socket_path: str,
        callback: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.callback = callback
        super().__init__(socket_path, _SnapshotRequestHandler)


class PAPKVSnapshotServer:
    """Serve short cache-index leases from an EngineCore-owned side thread."""

    def __init__(
        self,
        socket_path: str,
        callback: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> None:
        path = Path(socket_path)
        if not path.is_absolute():
            raise ValueError("PAP KV snapshot socket path must be absolute")
        if path.exists():
            raise FileExistsError(f"PAP KV snapshot socket already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._server = _SnapshotUnixServer(str(path), callback)
        os.chmod(path, 0o600)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="pap-kv-snapshot",
            daemon=True,
        )
        self._close_lock = threading.Lock()
        self._closed = False
        self._thread.start()

    @property
    def socket_path(self) -> str:
        return str(self._path)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=1.0)
            self._path.unlink(missing_ok=True)


class PAPKVSnapshotClient:
    """Issue one bounded request to the EngineCore snapshot side channel."""

    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path

    async def request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(
            {"operation": operation, "payload": payload},
            separators=(",", ":"),
        ).encode()
        if len(encoded) > _MAX_MESSAGE_BYTES:
            raise ValueError("PAP KV snapshot request is too large")
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        try:
            writer.write(_HEADER.pack(len(encoded)))
            writer.write(encoded)
            await writer.drain()
            size = _HEADER.unpack(await reader.readexactly(_HEADER.size))[0]
            if not 0 < size <= _MAX_MESSAGE_BYTES:
                raise RuntimeError("invalid PAP KV snapshot response size")
            response = json.loads(await reader.readexactly(size))
        finally:
            writer.close()
            await writer.wait_closed()
        if not isinstance(response, dict) or not response.get("ok"):
            error = (
                response.get("error", "invalid response")
                if isinstance(response, dict)
                else "invalid response"
            )
            raise RuntimeError(f"PAP KV snapshot request failed: {error}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("PAP KV snapshot response must be an object")
        return result


__all__ = ["PAPKVSnapshotClient", "PAPKVSnapshotServer"]
