# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-global NVSHMEM world for PAP execution traffic."""

from __future__ import annotations

import math
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from vllm.pap.config import PAPConfigError, read_env_float, read_env_int
from vllm.pap.transport.nvshmem.runtime import (
    PAPNVSHMEMAllocation,
    PAPNVSHMEMError,
    PAPNVSHMEMRuntime,
)

_SIGNAL_KINDS = 4
_SIGNAL_BYTES = 8
_DEFAULT_CONTROL_BYTES = 64 * 1024


@dataclass(frozen=True)
class PAPNVSHMEMWorldConfig:
    """Static PE coordinates shared by every transport in one process."""

    rank: int
    world_size: int
    device_index: int
    buffer_bytes: int
    control_bytes: int
    mode: str
    uid_path: Path | None
    uid_host: str | None = None
    uid_port: int | None = None
    uid_bind_host: str = "0.0.0.0"
    root_rank: int = 0
    init_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.init_timeout_s) or self.init_timeout_s <= 0:
            raise PAPNVSHMEMError(
                "NVSHMEM initialization timeout must be finite and positive"
            )

    @classmethod
    def from_env(
        cls,
        *,
        device_index: int,
        buffer_bytes: int,
    ) -> PAPNVSHMEMWorldConfig:
        """Build and validate a world from launcher metadata."""
        try:
            for name in ("PAP_NVSHMEM_RANK", "PAP_NVSHMEM_WORLD_SIZE"):
                if name not in os.environ:
                    raise KeyError(name)
            rank = read_env_int(os.environ, "PAP_NVSHMEM_RANK", minimum=0)
            world_size = read_env_int(os.environ, "PAP_NVSHMEM_WORLD_SIZE", minimum=2)
            mode = os.environ.get("PAP_NVSHMEM_MODE", "local").lower()
            root_rank = read_env_int(os.environ, "PAP_NVSHMEM_ROOT_RANK", minimum=0)
            control_bytes = read_env_int(
                os.environ,
                "PAP_NVSHMEM_CONTROL_BYTES",
                _DEFAULT_CONTROL_BYTES,
                minimum=9,
            )
            init_timeout_s = read_env_float(
                os.environ, "PAP_NVSHMEM_INIT_TIMEOUT", 30.0
            )
        except (KeyError, PAPConfigError) as exc:
            raise PAPNVSHMEMError(
                f"Invalid NVSHMEM rank/world size/UID file or runtime setting: {exc}"
            ) from exc
        if world_size <= 1 or rank < 0 or rank >= world_size:
            raise PAPNVSHMEMError("PAP NVSHMEM PE coordinates are invalid")
        if root_rank < 0 or root_rank >= world_size:
            raise PAPNVSHMEMError("PAP NVSHMEM root rank is invalid")
        if buffer_bytes <= 0:
            raise PAPNVSHMEMError("PAP NVSHMEM buffer size must be positive")
        if mode not in {"local", "ibgda"}:
            raise PAPNVSHMEMError(
                f"PAP NVSHMEM mode must be local or ibgda, got {mode!r}"
            )
        uid_path_text = os.environ.get("PAP_NVSHMEM_UID_FILE")
        uid_path = Path(uid_path_text) if uid_path_text else None
        uid_host = os.environ.get("PAP_NVSHMEM_UID_HOST") or None
        uid_port_text = os.environ.get("PAP_NVSHMEM_UID_PORT") or None
        if (uid_host is None) != (uid_port_text is None):
            raise PAPNVSHMEMError(
                "PAP NVSHMEM TCP UID bootstrap requires both host and port"
            )
        uid_port = None
        if uid_port_text is not None:
            try:
                uid_port = int(uid_port_text)
            except ValueError as exc:
                raise PAPNVSHMEMError(
                    "PAP NVSHMEM UID port must be an integer"
                ) from exc
            if uid_port <= 0 or uid_port > 65535:
                raise PAPNVSHMEMError("PAP NVSHMEM UID port is invalid")
        if mode == "local" and uid_path is None:
            raise PAPNVSHMEMError("local PAP NVSHMEM requires a UID file")
        if mode == "local" and uid_host is not None:
            raise PAPNVSHMEMError("local PAP NVSHMEM does not accept TCP UID bootstrap")
        if mode == "ibgda" and uid_host is None:
            raise PAPNVSHMEMError(
                "cross-host PAP NVSHMEM requires a TCP UID bootstrap endpoint"
            )
        if mode == "ibgda" and uid_path is not None:
            raise PAPNVSHMEMError("cross-host PAP NVSHMEM does not accept a UID file")
        if uid_path is not None and not uid_path.is_absolute():
            raise PAPNVSHMEMError("PAP NVSHMEM UID path must be absolute")
        return cls(
            rank=rank,
            world_size=world_size,
            device_index=device_index,
            buffer_bytes=buffer_bytes,
            control_bytes=control_bytes,
            mode=mode,
            uid_path=uid_path,
            uid_host=uid_host,
            uid_port=uid_port,
            uid_bind_host=os.environ.get("PAP_NVSHMEM_UID_BIND_HOST", "0.0.0.0"),
            root_rank=root_rank,
            init_timeout_s=init_timeout_s,
        )


class PAPNVSHMEMWorld:
    """Own collective initialization and process-lifetime symmetric buffers."""

    def __init__(
        self,
        config: PAPNVSHMEMWorldConfig,
        *,
        runtime: PAPNVSHMEMRuntime | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime or PAPNVSHMEMRuntime()
        self.data: PAPNVSHMEMAllocation | None = None
        self.staging: PAPNVSHMEMAllocation | None = None
        self.control: PAPNVSHMEMAllocation | None = None
        self.signals: PAPNVSHMEMAllocation | None = None
        self.graph_signals: PAPNVSHMEMAllocation | None = None
        self.graph_epochs: torch.Tensor | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._initialize,
            name=f"pap-nvshmem-init-pe{config.rank}",
            daemon=True,
        )
        self._thread.start()

    @property
    def rank(self) -> int:
        return self.config.rank

    @property
    def world_size(self) -> int:
        return self.config.world_size

    def initialization_status(self) -> dict[str, str]:
        if not self._ready.is_set():
            return {"state": "initializing"}
        if self._error is not None:
            return {"state": "failed", "error": repr(self._error)}
        if any(
            allocation is None
            for allocation in (
                self.data,
                self.control,
                self.signals,
                self.graph_signals,
            )
        ):
            return {"state": "failed", "error": "symmetric buffers unavailable"}
        if self.config.mode == "ibgda" and self.staging is None:
            return {"state": "failed", "error": "symmetric staging unavailable"}
        return {"state": "ready"}

    def wait_ready(self, timeout: float | None = None) -> None:
        """Wait until every PE has initialized the shared layout."""
        wait_timeout = (
            float(timeout) if timeout is not None else self.config.init_timeout_s
        )
        if not math.isfinite(wait_timeout) or wait_timeout < 0:
            raise PAPNVSHMEMError(
                "NVSHMEM wait timeout must be finite and non-negative"
            )
        if not self._ready.wait(wait_timeout):
            raise PAPNVSHMEMError(
                f"timed out initializing NVSHMEM PE {self.config.rank}"
            )
        if self._error is not None:
            raise PAPNVSHMEMError(
                f"NVSHMEM PE {self.config.rank} initialization failed"
            ) from self._error
        if any(
            allocation is None
            for allocation in (
                self.data,
                self.control,
                self.signals,
                self.graph_signals,
            )
        ):
            raise PAPNVSHMEMError("NVSHMEM world published no symmetric buffers")
        if self.config.mode == "ibgda" and self.staging is None:
            raise PAPNVSHMEMError(
                "cross-host NVSHMEM world published no symmetric staging"
            )

    def data_slot_offset(self, source_rank: int) -> int:
        self._validate_rank(source_rank)
        return source_rank * self.config.buffer_bytes

    def signal_offset(self, kind: int, source_rank: int) -> int:
        if kind < 0 or kind >= _SIGNAL_KINDS:
            raise PAPNVSHMEMError(f"invalid NVSHMEM signal kind: {kind}")
        self._validate_rank(source_rank)
        return (kind * self.world_size + source_rank) * _SIGNAL_BYTES

    def control_slot_offset(self, source_rank: int) -> int:
        self._validate_rank(source_rank)
        return source_rank * self.config.control_bytes

    def staging_data_slot_offset(self, peer_rank: int) -> int:
        self._validate_rank(peer_rank)
        return peer_rank * self.config.buffer_bytes

    def staging_control_slot_offset(self, peer_rank: int) -> int:
        self._validate_rank(peer_rank)
        return self.world_size * self.config.buffer_bytes + (
            peer_rank * self.config.control_bytes
        )

    def _initialize(self) -> None:
        try:
            torch.accelerator.set_device_index(self.config.device_index)
            # set_device alone does not materialize a current CUDA context.
            # NVSHMEM UID initialization requires one before get_cucontext.
            torch.empty(
                0,
                dtype=torch.uint8,
                device=torch.device("cuda", self.config.device_index),
            )
            self.runtime.initialize_uid(
                unique_id=self._exchange_unique_id(),
                rank=self.config.rank,
                world_size=self.config.world_size,
                device_index=self.config.device_index,
            )
            self.data = self.runtime.allocate(
                self.config.world_size * self.config.buffer_bytes
            )
            if self.config.mode == "ibgda":
                self.staging = self.runtime.allocate(
                    self.config.world_size
                    * (self.config.buffer_bytes + self.config.control_bytes)
                )
            self.control = self.runtime.allocate(
                self.config.world_size * self.config.control_bytes
            )
            signal_bytes = _SIGNAL_KINDS * self.config.world_size * _SIGNAL_BYTES
            self.signals = self.runtime.allocate(signal_bytes)
            self.graph_signals = self.runtime.allocate(signal_bytes)
            self.graph_epochs = torch.zeros(
                self.config.world_size,
                dtype=torch.uint64,
                device=torch.device("cuda", self.config.device_index),
            )
            self.data.tensor.zero_()
            if self.staging is not None:
                self.staging.tensor.zero_()
            self.control.tensor.zero_()
            self.signals.tensor.zero_()
            self.graph_signals.tensor.zero_()
            torch.accelerator.synchronize(self.config.device_index)
            self.runtime.barrier()
        except BaseException as exc:
            self._error = exc
        finally:
            self._ready.set()

    def _exchange_unique_id(self) -> bytes:
        if self.config.uid_host is not None:
            if self.config.rank == self.config.root_rank:
                return self._publish_unique_id_tcp()
            return self._read_unique_id_tcp()
        uid_path = self.config.uid_path
        if uid_path is None:
            raise PAPNVSHMEMError("PAP NVSHMEM UID file is missing")
        if self.config.rank == self.config.root_rank:
            unique_id = self.runtime.get_unique_id()
            uid_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(uid_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                written = os.write(fd, unique_id)
            finally:
                os.close(fd)
            if written != len(unique_id):
                raise PAPNVSHMEMError("failed to publish the complete NVSHMEM UID")
            return unique_id

        deadline = time.monotonic() + self.config.init_timeout_s
        while True:
            try:
                unique_id = uid_path.read_bytes()
            except FileNotFoundError:
                unique_id = b""
            if len(unique_id) == 128:
                return unique_id
            if unique_id:
                raise PAPNVSHMEMError("NVSHMEM UID file has an invalid size")
            if time.monotonic() >= deadline:
                raise PAPNVSHMEMError("timed out waiting for the NVSHMEM UID")
            time.sleep(0.01)

    def _publish_unique_id_tcp(self) -> bytes:
        unique_id = self.runtime.get_unique_id()
        expected_clients = self.config.world_size - 1
        deadline = time.monotonic() + self.config.init_timeout_s
        seen_ranks: set[int] = set()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            assert self.config.uid_port is not None
            server.bind((self.config.uid_bind_host, self.config.uid_port))
            server.listen(expected_clients)
            while len(seen_ranks) < expected_clients:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PAPNVSHMEMError(
                        "timed out publishing the NVSHMEM UID over TCP"
                    )
                server.settimeout(min(1.0, remaining))
                try:
                    connection, _address = server.accept()
                except TimeoutError:
                    continue
                with connection:
                    connection.settimeout(remaining)
                    rank = struct.unpack("!I", self._recv_exact(connection, 4))[0]
                    if (
                        rank >= self.config.world_size
                        or rank == self.config.root_rank
                        or rank in seen_ranks
                    ):
                        raise PAPNVSHMEMError(
                            f"invalid NVSHMEM TCP bootstrap rank: {rank}"
                        )
                    connection.sendall(unique_id)
                    seen_ranks.add(rank)
        return unique_id

    def _read_unique_id_tcp(self) -> bytes:
        assert self.config.uid_host is not None
        assert self.config.uid_port is not None
        deadline = time.monotonic() + self.config.init_timeout_s
        last_error: OSError | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PAPNVSHMEMError(
                    "timed out reading the NVSHMEM UID over TCP"
                ) from last_error
            try:
                with socket.create_connection(
                    (self.config.uid_host, self.config.uid_port),
                    timeout=min(1.0, remaining),
                ) as connection:
                    connection.sendall(struct.pack("!I", self.config.rank))
                    return self._recv_exact(connection, 128)
            except OSError as exc:
                last_error = exc
                time.sleep(min(0.05, remaining))

    @staticmethod
    def _recv_exact(connection: socket.socket, num_bytes: int) -> bytes:
        chunks: list[bytes] = []
        received = 0
        while received < num_bytes:
            chunk = connection.recv(num_bytes - received)
            if not chunk:
                raise PAPNVSHMEMError("NVSHMEM TCP bootstrap connection closed early")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _validate_rank(self, rank: int) -> None:
        if rank < 0 or rank >= self.world_size:
            raise PAPNVSHMEMError(f"invalid NVSHMEM source rank: {rank}")


_WORLD: PAPNVSHMEMWorld | None = None
_WORLD_LOCK = threading.Lock()


def get_pap_nvshmem_world(
    *,
    device_index: int,
    buffer_bytes: int,
) -> PAPNVSHMEMWorld:
    """Return the process-global world after checking immutable settings."""
    global _WORLD
    config = PAPNVSHMEMWorldConfig.from_env(
        device_index=device_index,
        buffer_bytes=buffer_bytes,
    )
    with _WORLD_LOCK:
        if _WORLD is None:
            _WORLD = PAPNVSHMEMWorld(config)
        elif _WORLD.config != config:
            raise PAPNVSHMEMError("PAP NVSHMEM world configuration changed")
        return _WORLD
