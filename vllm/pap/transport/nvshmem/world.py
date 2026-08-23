# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-global same-host NVSHMEM world for PAP execution traffic."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

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
    uid_path: Path
    root_rank: int = 0

    @classmethod
    def from_env(
        cls,
        *,
        device_index: int,
        buffer_bytes: int,
    ) -> PAPNVSHMEMWorldConfig:
        """Build and validate a same-host world from launcher metadata."""
        try:
            rank = int(os.environ["PAP_NVSHMEM_RANK"])
            world_size = int(os.environ["PAP_NVSHMEM_WORLD_SIZE"])
            uid_path = Path(os.environ["PAP_NVSHMEM_UID_FILE"])
        except (KeyError, ValueError) as exc:
            raise PAPNVSHMEMError(
                "PAP NVSHMEM requires rank, world size, and UID file metadata"
            ) from exc
        root_rank = int(os.environ.get("PAP_NVSHMEM_ROOT_RANK", "0"))
        if world_size <= 1 or rank < 0 or rank >= world_size:
            raise PAPNVSHMEMError("PAP NVSHMEM PE coordinates are invalid")
        if root_rank < 0 or root_rank >= world_size:
            raise PAPNVSHMEMError("PAP NVSHMEM root rank is invalid")
        if buffer_bytes <= 0:
            raise PAPNVSHMEMError("PAP NVSHMEM buffer size must be positive")
        control_bytes = int(
            os.environ.get("PAP_NVSHMEM_CONTROL_BYTES", str(_DEFAULT_CONTROL_BYTES))
        )
        if control_bytes <= 8:
            raise PAPNVSHMEMError("PAP NVSHMEM control record is too small")
        if not uid_path.is_absolute():
            raise PAPNVSHMEMError("PAP NVSHMEM UID path must be absolute")
        return cls(
            rank=rank,
            world_size=world_size,
            device_index=device_index,
            buffer_bytes=buffer_bytes,
            control_bytes=control_bytes,
            uid_path=uid_path,
            root_rank=root_rank,
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

    def wait_ready(self, timeout: float | None = None) -> None:
        """Wait until every local PE has initialized the shared layout."""
        wait_timeout = (
            float(timeout)
            if timeout is not None
            else float(os.environ.get("PAP_NVSHMEM_INIT_TIMEOUT", "30"))
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
        uid_path = self.config.uid_path
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

        deadline = time.monotonic() + float(
            os.environ.get("PAP_NVSHMEM_INIT_TIMEOUT", "30")
        )
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


def reset_pap_nvshmem_world_for_tests() -> None:
    """Clear the singleton when no real NVSHMEM runtime was initialized."""
    global _WORLD
    with _WORLD_LOCK:
        _WORLD = None
