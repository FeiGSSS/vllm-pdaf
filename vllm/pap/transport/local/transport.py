# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-machine stream-ordered CUDA IPC transport.

This transport is a research prototype intended to demonstrate that the
Projection<->Attention per-layer QKV/O transit can be driven well below the
~500 us/layer cost of the NIXL/UCX mailbox stack when both sides run on the
same machine.

Design highlights:

* Each side pre-allocates a CUDA recv buffer on its local GPU and
  exports a CUDA IPC handle (via ``torch.multiprocessing.reductions``).
  Handles are exchanged through the existing PAP control plane (the HTTP
  bind handshake) as a small pickled + base64-encoded metadata blob.
* A single receive buffer carries QKV or attention output serially. CUDA stream
  memory operations publish ready/release generations without a CPU device
  synchronization. A small ``/dev/shm`` doorbell carries QKV descriptors;
  Attention output is descriptorless.
* CUDA stream memory operations are required. Unsupported systems fail closed
  instead of selecting an unvalidated synchronization fallback.
* Receiver spin-wait falls back to ``os.sched_yield`` after a configurable
  number of tight iterations to avoid pinning a core forever.
* Default OFF.  Activate with ``PAP_OFFLOAD_EXEC_TRANSPORT=local_fast``.

This implementation is intentionally lightweight on error-handling for
unusual states such as peer death. It is meant for controlled same-host runs
and implements the ownership-bearing OFFLOAD_EXEC transport contract directly.
"""

from __future__ import annotations

import json
import mmap
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.pap.cuda_stream_memops import (
    probe_stream_mem_ops,
)
from vllm.pap.deferred_cuda_trace import (
    deferred_cuda_trace_enabled,
)
from vllm.pap.protocol import PAPTensorTransport
from vllm.pap.transport.local.endpoint import (
    _doorbell_path,
    _ensure_peer_access,
    _local_hostname,
    _open_or_create_doorbell,
    _pack_cuda_ipc_handle,
    _unpack_cuda_ipc_handle,
)
from vllm.pap.transport.local.io import _PAPLocalFastIOMixin
from vllm.pap.transport.local.protocol import (
    DOORBELL_BYTES,
    DIR_QKV,
    _doorbell_read_header,
    _doorbell_record_offset,
    _doorbell_write,
    _WireMetadata,
)

# ---------------------------------------------------------------------------
# Constants / layout
# ---------------------------------------------------------------------------

# Default recv buffer size.  Same default as the NIXL mailbox path.
DEFAULT_BUFFER_BYTES = 16 * 1024 * 1024

# Doorbell wait behavior. Keep a short tight spin for low-latency handoff, then
# back off so long waits do not burn a CPU core and interfere with the shared
# PA-side control plane / MPS scheduling.
SPIN_TIGHT_ITERS = int(os.environ.get("PAP_LOCAL_FAST_SPIN_ITERS", "2048"))
SPIN_YIELD_ITERS = int(os.environ.get("PAP_LOCAL_FAST_YIELD_ITERS", "64"))
SPIN_SLEEP_US = int(os.environ.get("PAP_LOCAL_FAST_SLEEP_US", "20"))
SPIN_SLEEP_AFTER_US = int(os.environ.get("PAP_LOCAL_FAST_SLEEP_AFTER_US", "50"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _sched_yield() -> None:
    try:
        os.sched_yield()
    except AttributeError:
        time.sleep(0)


# ---------------------------------------------------------------------------
# Per-peer lifecycle state
# ---------------------------------------------------------------------------


@dataclass
class _PeerState:
    """Per-peer state cached on the local transport after bind_peer."""

    peer_tensor: torch.Tensor  # view into peer's recv buffer (local device)
    peer_signal_tensor: torch.Tensor
    peer_doorbell_path: str
    buffer_bytes: int
    peer_doorbell_mm: mmap.mmap | None = None
    peer_doorbell_fd: int | None = None
    # Local-side counters for the *outgoing* directions on this side.
    # Each direction has its own monotonic seq; receiver checks against its
    # own "expected" seq.
    next_qkv_seq: int = 1
    next_output_seq: int = 1
    # Expected incoming seq for each direction.  This is what we wait for.
    expected_qkv_seq: int = 1
    expected_output_seq: int = 1
    last_qkv_seq: int = 0
    last_output_seq: int = 0
    source_refs: dict[int, torch.Tensor] = field(default_factory=dict)
    send_lock: threading.Lock = field(default_factory=threading.Lock)


class PAPLocalFastTransport(_PAPLocalFastIOMixin):
    """Same-machine CUDA IPC + spin-doorbell OFFLOAD_EXEC transport.

    See module docstring for the design rationale and constraints.
    """

    # Declare attributes that the Protocol / call sites introspect.
    transport = PAPTensorTransport.CUDA_IPC
    requires_tcp_trigger = False

    def __init__(
        self,
        *,
        actor_id: str,
        device: torch.device,
        buffer_bytes: int | None = None,
    ) -> None:
        self.actor_id = str(actor_id)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError(
                f"PAPLocalFastTransport requires a CUDA device; got {self.device}"
            )

        self.buffer_bytes = (
            int(buffer_bytes)
            if buffer_bytes is not None
            else int(
                os.environ.get("PAP_LOCAL_FAST_BUFFER_BYTES", str(DEFAULT_BUFFER_BYTES))
            )
        )
        if self.buffer_bytes <= 0:
            raise RuntimeError("PAP local fast buffer_bytes must be positive")

        # Allocate the local recv buffer (1D byte tensor on local GPU).
        with torch.cuda.device(self.device):
            self._recv_buffer = torch.empty(
                self.buffer_bytes, dtype=torch.uint8, device=self.device
            )
            if not probe_stream_mem_ops(self.device):
                raise RuntimeError(
                    "PAP local transport requires CUDA stream memory operations"
                )
            self._signal_buffer = torch.zeros(
                4,
                dtype=torch.int32,
                device=self.device,
            )
        # Pin the underlying storage lifetime: hold a reference to the
        # untyped storage so the IPC handle stays valid until we drop it.
        self._recv_storage = self._recv_buffer.untyped_storage()
        self._signal_storage = self._signal_buffer.untyped_storage()

        # Build / open the local doorbell file.
        self._doorbell_path = _doorbell_path(self.actor_id)
        self._doorbell_fd, self._doorbell_mm = _open_or_create_doorbell(
            self._doorbell_path,
            DOORBELL_BYTES,
        )
        # Zero the doorbell on (re)open.  This is safe because we only ever
        # have one transport per actor_id per machine.
        self._doorbell_mm[:] = b"\x00" * DOORBELL_BYTES

        self._peer: _PeerState | None = None
        self._started = False
        self._bound = False
        self._trace = _env_bool("PAP_OFFLOAD_EXEC_TRACE", False)
        self._deferred_cuda_trace = deferred_cuda_trace_enabled()
        self._step_plan_cache_limit = int(
            os.environ.get("PAP_LOCAL_FAST_STEP_PLAN_CACHE_LIMIT", "256")
        )
        self._sent_step_plans: OrderedDict[str, str] = OrderedDict()
        self._recv_batch_plans: dict[str, dict[str, Any]] = {}
        self._recv_plan_layer_templates: dict[str, tuple[str, str]] = {}
        self._recv_plan_ids_by_key: dict[str, str] = {}
        self._step_plan_builds = 0
        self._step_plan_refs = 0
        self._output_descriptor_elisions = 0
        self._descriptorless_output_receives = 0
        self._binary_qkv_refs = 0
        self._binary_outputs = 0
        self._json_records = 0
        self._stats_reported = False

    # ------------------------------------------------------------------
    # Metadata exchange
    # ------------------------------------------------------------------

    @property
    def local_agent_metadata(self) -> bytes:
        """Serialize local-side IPC handle + doorbell info for the peer."""

        ipc_blob = _pack_cuda_ipc_handle(self._recv_buffer)
        signal_blob = _pack_cuda_ipc_handle(self._signal_buffer)
        payload = {
            "v": 3,
            "actor_id": self.actor_id,
            "hostname": _local_hostname(),
            "device_index": int(self.device.index or 0),
            "buffer_bytes": int(self.buffer_bytes),
            "stream_ordered": True,
            "doorbell_path": self._doorbell_path,
            "ipc_handle": ipc_blob,
            "signal_ipc_handle": signal_blob,
        }
        return json.dumps(payload).encode("utf-8")

    def bind_peer(self, peer_agent_metadata: bytes) -> None:
        """Open the peer's CUDA IPC handle and doorbell file."""

        if self._peer is not None:
            return
        payload = json.loads(peer_agent_metadata.decode("utf-8"))
        if int(payload.get("v", 0)) != 3:
            raise RuntimeError(
                "PAP local transport peers must use the serial-buffer protocol"
            )
        peer_hostname = str(payload.get("hostname", ""))
        if peer_hostname and peer_hostname != _local_hostname():
            raise RuntimeError(
                "PAPLocalFastTransport is same-machine only; peer hostname "
                f"{peer_hostname!r} != local {_local_hostname()!r}. "
                "Set PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox for cross-host."
            )
        peer_buffer_bytes = int(payload.get("buffer_bytes", 0))
        if peer_buffer_bytes <= 0:
            raise RuntimeError("PAP local fast peer has invalid buffer capacity")
        peer_doorbell_path = str(payload["doorbell_path"])
        ipc_blob = str(payload["ipc_handle"])

        # Rebuild the peer's tensor on our local device.  reduce_tensor
        # rebuilds on the original device index; if that is different from
        # ours, the kernel will route via peer access (or staged copy).
        peer_tensor = _unpack_cuda_ipc_handle(ipc_blob)
        peer_device = peer_tensor.device
        if not bool(payload.get("stream_ordered", False)):
            raise RuntimeError(
                "PAP local transport requires a stream-ordered peer"
            )
        peer_signal_tensor = _unpack_cuda_ipc_handle(
            str(payload["signal_ipc_handle"])
        )

        # Enable peer access (local -> peer).  If unsupported, refuse to
        # start: per the spec we do not silently fall back.
        if not _ensure_peer_access(self.device, peer_device):
            raise RuntimeError(
                f"PAPLocalFastTransport: P2P peer access not supported "
                f"between device {self.device.index} and "
                f"device {peer_device.index}; cannot use direct CUDA IPC."
            )

        # Open the peer's doorbell file (read/write shared).
        peer_fd = os.open(peer_doorbell_path, os.O_RDWR)
        peer_mm = mmap.mmap(
            peer_fd,
            DOORBELL_BYTES,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        self._peer = _PeerState(
            peer_tensor=peer_tensor,
            peer_signal_tensor=peer_signal_tensor,
            peer_doorbell_path=peer_doorbell_path,
            buffer_bytes=peer_buffer_bytes,
            peer_doorbell_mm=peer_mm,
            peer_doorbell_fd=peer_fd,
        )
        self._bound = True
        # start() is idempotent and called here for symmetry with the mailbox
        # transport, where bind_peer also starts the endpoint.
        self.start()

    def start(self) -> None:
        """Mark the bound local transport ready."""

        if self._started:
            return
        self._started = True

    def _next_seq(self, peer: _PeerState, direction: int) -> int:
        if direction == DIR_QKV:
            seq = peer.next_qkv_seq
            peer.next_qkv_seq += 1
            return seq
        seq = peer.next_output_seq
        peer.next_output_seq += 1
        return seq

    def _write_doorbell_sync(
        self,
        *,
        peer: _PeerState,
        direction: int,
        seq: int,
        nbytes: int,
        offset: int,
        wire: _WireMetadata,
    ) -> None:
        record_offset = _doorbell_record_offset(direction)
        _doorbell_write(
            peer.peer_doorbell_mm,
            record_offset,
            seq=seq,
            nbytes=nbytes,
            offset=offset,
            metadata=wire.metadata,
            plan_id=wire.plan_id,
            shape=wire.shape,
            layer_index=wire.layer_index,
            dtype_code=wire.dtype_code,
            flags=wire.flags,
        )

    def _wait_control_buffer(
        self,
        *,
        peer: _PeerState,
        direction: int,
        previous_seq: int,
    ) -> None:
        if previous_seq == 0:
            return
        record_offset = _doorbell_record_offset(direction)
        start = time.monotonic()
        iters = 0
        while True:
            ack = _doorbell_read_header(peer.peer_doorbell_mm, record_offset)[4]
            if ack >= previous_seq:
                return
            if time.monotonic() - start >= 30.0:
                raise TimeoutError(
                    "timed out waiting for PAP local fast receive buffer "
                    f"direction={direction} seq={previous_seq}"
                )
            iters += 1
            if iters < SPIN_TIGHT_ITERS:
                continue
            if iters < SPIN_TIGHT_ITERS + SPIN_YIELD_ITERS:
                _sched_yield()
                continue
            waited_us = (time.monotonic() - start) * 1_000_000.0
            if waited_us >= SPIN_SLEEP_AFTER_US:
                time.sleep(max(SPIN_SLEEP_US, 0) / 1_000_000.0)
            else:
                _sched_yield()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_peer(self) -> _PeerState:
        if self._peer is None:
            raise RuntimeError("PAPLocalFastTransport: bind_peer must be called first")
        return self._peer

    def flush(self) -> None:
        if self._peer is None:
            return
        torch.cuda.synchronize(self.device)

    def _report_stats(self) -> None:
        if self._stats_reported:
            return
        self._stats_reported = True
        logger.info(
            "PAP local fast transport stats actor=%s step_plan_builds=%d "
            "step_plan_refs=%d output_descriptor_elisions=%d "
            "descriptorless_output_receives=%d "
            "binary_qkv_refs=%d binary_outputs=%d json_records=%d",
            self.actor_id,
            self._step_plan_builds,
            self._step_plan_refs,
            self._output_descriptor_elisions,
            self._descriptorless_output_receives,
            self._binary_qkv_refs,
            self._binary_outputs,
            self._json_records,
        )

    def close(self) -> None:
        try:
            self.flush()
            self._report_stats()
            if self._peer is not None:
                if self._peer.peer_doorbell_mm is not None:
                    self._peer.peer_doorbell_mm.close()
                if self._peer.peer_doorbell_fd is not None:
                    os.close(self._peer.peer_doorbell_fd)
            if self._doorbell_mm is not None:
                self._doorbell_mm.close()
            if self._doorbell_fd is not None:
                os.close(self._doorbell_fd)
        except Exception:
            pass
        finally:
            self._peer = None
            self._started = False
            self._bound = False
            self._doorbell_fd = None
            self._doorbell_mm = None
            self._recv_storage = None
            self._recv_buffer = None
            self._signal_storage = None
            self._signal_buffer = None

    def __enter__(self) -> PAPLocalFastTransport:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
        return None

    def __del__(self) -> None:
        self.close()
        return None

    def __repr__(self) -> str:
        return (
            f"PAPLocalFastTransport(actor_id={self.actor_id!r}, "
            f"device={self.device!s}, buffer_bytes={self.buffer_bytes})"
        )

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_local_fast_offload_exec_transport(
    *,
    actor_id: str,
    local_rank: int,
    buffer_bytes: int | None = None,
) -> PAPLocalFastTransport:
    """Construct a PAPLocalFastTransport for the given actor / rank."""

    device = torch.device(f"cuda:{int(local_rank)}")
    return PAPLocalFastTransport(
        actor_id=actor_id,
        device=device,
        buffer_bytes=(
            int(buffer_bytes)
            if buffer_bytes is not None
            else int(
                os.environ.get("PAP_LOCAL_FAST_BUFFER_BYTES", str(DEFAULT_BUFFER_BYTES))
            )
        ),
    )


# Late import to avoid pulling logging infrastructure at module import time
# in case the parent process hasn't configured it yet.
import logging  # noqa: E402

logger = logging.getLogger(__name__)
