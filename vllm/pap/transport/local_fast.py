# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP same-machine stream-ordered CUDA IPC fast-path transport.

This transport is a research prototype intended to demonstrate that the
Projection<->Attention per-layer QKV/O transit can be driven well below the
~500 us/layer cost of the NIXL/UCX mailbox stack when both sides run on the
same machine.

Design highlights:

* Each side pre-allocates a CUDA recv buffer on its local GPU and
  exports a CUDA IPC handle (via ``torch.multiprocessing.reductions``).
  Handles are exchanged through the existing PAP control plane (the HTTP
  bind handshake) as a small pickled + base64-encoded metadata blob.
* A two-slot ring carries QKV and attention output in each direction. CUDA
  stream memory operations publish ready/release generations without a CPU
  device synchronization. A small ``/dev/shm`` doorbell carries descriptors.
* Systems without CUDA stream memory operations fall back to the original
  synchronous doorbell path.
* Receiver spin-wait falls back to ``os.sched_yield`` after a configurable
  number of tight iterations to avoid pinning a core forever.
* Default OFF.  Activate with ``PAP_OFFLOAD_EXEC_TRANSPORT=local_fast``.

This implementation is intentionally lightweight on error-handling for unusual
states (slot overrun, peer death).  It is meant for controlled benchmark runs
on a single host.

The transport implements the same public surface as
:class:`PAPNixlMailboxOffloadExecTransport`.  Some methods that are not used
on the fast path (notably ``recv_next_qkv_batch``/``recv_next_qkv_batch_message``
used by the optional mailbox prefetch thread) are stubbed in terms of
``recv_qkv_batch`` so the attention executor's mailbox loop still functions.
"""

from __future__ import annotations

import json
import mmap
import os
import queue
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
from vllm.pap.transport.local_fast_endpoint import (
    _doorbell_path,
    _ensure_peer_access,
    _local_hostname,
    _open_or_create_doorbell,
    _pack_cuda_ipc_handle,
    _unpack_cuda_ipc_handle,
)
from vllm.pap.transport.local_fast_protocol import (
    DIR_QKV,
    _WireMetadata,
    _doorbell_bytes,
    _doorbell_read_header,
    _doorbell_record_offset,
    _doorbell_write,
)
from vllm.pap.transport.local_fast_io import (
    _PAPLocalFastIOMixin,
    _PendingDoorbell,
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
    peer_signal_tensor: torch.Tensor | None
    peer_doorbell_path: str
    slot_count: int
    slot_bytes: int
    doorbell_bytes: int
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
    pending_qkv_seq: int = 0
    pending_output_seq: int = 0
    last_qkv_seq_by_slot: list[int] = field(init=False)
    last_output_seq_by_slot: list[int] = field(init=False)
    source_refs: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    wait_cond: threading.Condition = field(init=False)

    def __post_init__(self) -> None:
        self.wait_cond = threading.Condition(self.send_lock)
        self.last_qkv_seq_by_slot = [0] * self.slot_count
        self.last_output_seq_by_slot = [0] * self.slot_count


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
        self._slot_count = int(os.environ.get("PAP_LOCAL_FAST_SLOT_COUNT", "2"))
        if self._slot_count <= 0:
            raise RuntimeError("PAP_LOCAL_FAST_SLOT_COUNT must be positive")
        self._slot_bytes = self.buffer_bytes // self._slot_count
        if self._slot_bytes <= 0:
            raise RuntimeError("PAP local fast slots exceed the receive buffer")
        self._doorbell_bytes = _doorbell_bytes(self._slot_count)

        # Allocate the local recv buffer (1D byte tensor on local GPU).
        with torch.cuda.device(self.device):
            self._recv_buffer = torch.empty(
                self.buffer_bytes, dtype=torch.uint8, device=self.device
            )
            self._stream_ordered_requested = _env_bool(
                "PAP_LOCAL_FAST_STREAM_ORDERED", True
            )
            self._stream_ordered_available = (
                self._stream_ordered_requested and probe_stream_mem_ops(self.device)
            )
            self._signal_buffer = torch.zeros(
                4 * self._slot_count,
                dtype=torch.int32,
                device=self.device,
            )
        self._stream_ordered = False
        # Pin the underlying storage lifetime: hold a reference to the
        # untyped storage so the IPC handle stays valid until we drop it.
        self._recv_storage = self._recv_buffer.untyped_storage()
        self._signal_storage = self._signal_buffer.untyped_storage()

        # Build / open the local doorbell file.
        self._doorbell_path = _doorbell_path(self.actor_id)
        self._doorbell_fd, self._doorbell_mm = _open_or_create_doorbell(
            self._doorbell_path,
            self._doorbell_bytes,
        )
        # Zero the doorbell on (re)open.  This is safe because we only ever
        # have one transport per actor_id per machine.
        self._doorbell_mm[:] = b"\x00" * self._doorbell_bytes

        self._peer: _PeerState | None = None
        self._started = False
        self._bound = False
        self._trace = _env_bool("PAP_OFFLOAD_EXEC_TRACE", False)
        self._deferred_cuda_trace = deferred_cuda_trace_enabled()
        self._async_doorbell = _env_bool("PAP_LOCAL_FAST_ASYNC_DOORBELL", False)
        self._batch_plan_enabled = _env_bool("PAP_LOCAL_FAST_BATCH_PLAN", True)
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
        self._binary_qkv_refs = 0
        self._binary_outputs = 0
        self._json_records = 0
        self._stats_reported = False
        self._notify_queue: queue.Queue[_PendingDoorbell | None] | None = None
        self._notify_thread: threading.Thread | None = None
        self._notify_error: Exception | None = None
        self._notify_error_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Metadata exchange
    # ------------------------------------------------------------------

    @property
    def local_agent_metadata(self) -> bytes:
        """Serialize local-side IPC handle + doorbell info for the peer."""

        ipc_blob = _pack_cuda_ipc_handle(self._recv_buffer)
        signal_blob = _pack_cuda_ipc_handle(self._signal_buffer)
        payload = {
            "v": 2,
            "actor_id": self.actor_id,
            "hostname": _local_hostname(),
            "device_index": int(self.device.index or 0),
            "buffer_bytes": int(self.buffer_bytes),
            "slot_count": self._slot_count,
            "doorbell_bytes": self._doorbell_bytes,
            "stream_ordered": self._stream_ordered_available,
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
        peer_hostname = str(payload.get("hostname", ""))
        if peer_hostname and peer_hostname != _local_hostname():
            raise RuntimeError(
                "PAPLocalFastTransport is same-machine only; peer hostname "
                f"{peer_hostname!r} != local {_local_hostname()!r}. "
                "Set PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox for cross-host."
            )
        peer_buffer_bytes = int(payload.get("buffer_bytes", 0))
        peer_slot_count = int(payload.get("slot_count", 1))
        if peer_slot_count != self._slot_count:
            raise RuntimeError(
                "PAP local fast peers must use the same slot count: "
                f"local={self._slot_count} peer={peer_slot_count}"
            )
        peer_slot_bytes = peer_buffer_bytes // peer_slot_count
        if peer_slot_bytes <= 0:
            raise RuntimeError("PAP local fast peer has invalid slot capacity")
        peer_doorbell_bytes = int(
            payload.get("doorbell_bytes", _doorbell_bytes(peer_slot_count))
        )
        peer_doorbell_path = str(payload["doorbell_path"])
        ipc_blob = str(payload["ipc_handle"])

        # Rebuild the peer's tensor on our local device.  reduce_tensor
        # rebuilds on the original device index; if that is different from
        # ours, the kernel will route via peer access (or staged copy).
        peer_tensor = _unpack_cuda_ipc_handle(ipc_blob)
        peer_device = peer_tensor.device
        peer_stream_ordered = bool(payload.get("stream_ordered", False))
        self._stream_ordered = bool(
            self._stream_ordered_available and peer_stream_ordered
        )
        peer_signal_tensor = None
        if self._stream_ordered:
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
            peer_doorbell_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        self._peer = _PeerState(
            peer_tensor=peer_tensor,
            peer_signal_tensor=peer_signal_tensor,
            peer_doorbell_path=peer_doorbell_path,
            slot_count=peer_slot_count,
            slot_bytes=peer_slot_bytes,
            doorbell_bytes=peer_doorbell_bytes,
            peer_doorbell_mm=peer_mm,
            peer_doorbell_fd=peer_fd,
        )
        self._bound = True
        # start() is idempotent and called here for symmetry with the mailbox
        # transport, where bind_peer also starts the endpoint.
        self.start()

    def start(self) -> None:
        """Start optional notifier thread; receiver spin stays inline."""

        if self._started:
            return
        self._started = True
        if self._async_doorbell and not self._stream_ordered:
            self._notify_queue = queue.Queue()
            self._notify_thread = threading.Thread(
                target=self._notify_loop,
                name=f"pap-local-fast-notify-{self.actor_id}",
                daemon=True,
            )
            self._notify_thread.start()

    def _set_notify_error(self, exc: Exception) -> None:
        with self._notify_error_lock:
            if self._notify_error is None:
                self._notify_error = exc

    def _raise_if_notify_failed(self) -> None:
        with self._notify_error_lock:
            exc = self._notify_error
        if exc is not None:
            raise RuntimeError("PAP local fast async doorbell notifier failed") from exc

    def _next_seq(self, peer: _PeerState, direction: int) -> int:
        if direction == DIR_QKV:
            seq = peer.next_qkv_seq
            peer.next_qkv_seq += 1
            return seq
        seq = peer.next_output_seq
        peer.next_output_seq += 1
        return seq

    def _set_pending_seq(self, peer: _PeerState, direction: int, seq: int) -> None:
        if direction == DIR_QKV:
            peer.pending_qkv_seq = seq
        else:
            peer.pending_output_seq = seq

    def _clear_pending_seq(self, peer: _PeerState, direction: int, seq: int) -> None:
        if direction == DIR_QKV:
            if peer.pending_qkv_seq == seq:
                peer.pending_qkv_seq = 0
        elif peer.pending_output_seq == seq:
            peer.pending_output_seq = 0
        peer.wait_cond.notify_all()

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
        slot_id = (int(seq) - 1) % peer.slot_count
        record_offset = _doorbell_record_offset(
            direction,
            slot_id,
            peer.slot_count,
        )
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

    def _wait_control_slot(
        self,
        *,
        peer: _PeerState,
        direction: int,
        slot_id: int,
        previous_seq: int,
    ) -> None:
        if previous_seq == 0:
            return
        record_offset = _doorbell_record_offset(
            direction,
            slot_id,
            peer.slot_count,
        )
        start = time.monotonic()
        iters = 0
        while True:
            ack = _doorbell_read_header(peer.peer_doorbell_mm, record_offset)[4]
            if ack >= previous_seq:
                return
            if time.monotonic() - start >= 30.0:
                raise TimeoutError(
                    "timed out waiting for PAP local fast control slot "
                    f"direction={direction} slot={slot_id} seq={previous_seq}"
                )
            iters += 1
            if iters < SPIN_TIGHT_ITERS:
                continue
            _sched_yield()

    def _notify_loop(self) -> None:
        assert self._notify_queue is not None
        while True:
            job = self._notify_queue.get()
            if job is None:
                self._notify_queue.task_done()
                return
            try:
                wait_start = time.perf_counter()
                job.event.synchronize()
                doorbell_start = time.perf_counter()
                peer = self._require_peer()
                self._write_doorbell_sync(
                    peer=peer,
                    direction=job.direction,
                    seq=job.seq,
                    nbytes=job.nbytes,
                    offset=job.offset,
                    wire=job.wire,
                )
                with peer.wait_cond:
                    self._clear_pending_seq(peer, job.direction, job.seq)
                if self._trace:
                    kind = "qkv" if job.direction == DIR_QKV else "output"
                    logger.info(
                        "PAP local fast transport async doorbell trace kind=%s "
                        "layer=%s batch=%s enqueue_ms=%.3f event_wait_ms=%.3f "
                        "doorbell_ms=%.3f seq=%d nbytes=%d",
                        kind,
                        job.descriptor_layer_name,
                        job.descriptor_batch_id,
                        (wait_start - job.enqueue_time) * 1000.0,
                        (doorbell_start - wait_start) * 1000.0,
                        (time.perf_counter() - doorbell_start) * 1000.0,
                        job.seq,
                        job.nbytes,
                    )
            except Exception as exc:
                self._set_notify_error(exc)
                peer = self._peer
                if peer is not None:
                    with peer.wait_cond:
                        peer.wait_cond.notify_all()
                logger.exception(
                    "PAP local fast async doorbell notifier failed actor=%s",
                    self.actor_id,
                )
                return
            finally:
                self._notify_queue.task_done()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_peer(self) -> _PeerState:
        if self._peer is None:
            raise RuntimeError("PAPLocalFastTransport: bind_peer must be called first")
        return self._peer

    def flush(self) -> None:
        peer = self._peer
        if peer is None:
            return
        if self._stream_ordered:
            torch.cuda.synchronize(self.device)
            return
        if not self._async_doorbell:
            return
        with peer.wait_cond:
            while (
                peer.pending_qkv_seq != 0 or peer.pending_output_seq != 0
            ) and self._notify_error is None:
                peer.wait_cond.wait(timeout=0.01)
        self._raise_if_notify_failed()

    def barrier(self) -> None:
        self.flush()

    def wait_idle(self) -> None:
        self.flush()

    def _report_stats(self) -> None:
        if self._stats_reported:
            return
        self._stats_reported = True
        logger.info(
            "PAP local fast transport stats actor=%s step_plan_builds=%d "
            "step_plan_refs=%d output_descriptor_elisions=%d "
            "binary_qkv_refs=%d binary_outputs=%d json_records=%d",
            self.actor_id,
            self._step_plan_builds,
            self._step_plan_refs,
            self._output_descriptor_elisions,
            self._binary_qkv_refs,
            self._binary_outputs,
            self._json_records,
        )

    def close(self) -> None:
        try:
            self.flush()
            self._report_stats()
            if (
                self._notify_queue is not None
                and self._notify_thread is not None
                and self._notify_thread.is_alive()
            ):
                self._notify_queue.put(None)
                self._notify_thread.join(timeout=1.0)
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
            self._notify_queue = None
            self._notify_thread = None
            self._peer = None
            self._started = False
            self._bound = False
            self._doorbell_fd = None
            self._doorbell_mm = None
            self._notify_error = None
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
            f"device={self.device!s}, stream_ordered={self._stream_ordered}, "
            f"slots={self._slot_count})"
        )

    def __getstate__(self) -> dict[str, Any]:
        raise TypeError("PAPLocalFastTransport is not picklable")

    def __setstate__(self, state: dict[str, Any]) -> None:
        raise TypeError("PAPLocalFastTransport is not picklable")

    def debug_state(self) -> dict[str, Any]:
        peer = self._peer
        return {
            "actor_id": self.actor_id,
            "device": str(self.device),
            "async_doorbell": self._async_doorbell,
            "stream_ordered": self._stream_ordered,
            "stream_ordered_available": self._stream_ordered_available,
            "slot_count": self._slot_count,
            "step_plan_builds": self._step_plan_builds,
            "step_plan_refs": self._step_plan_refs,
            "output_descriptor_elisions": self._output_descriptor_elisions,
            "binary_qkv_refs": self._binary_qkv_refs,
            "binary_outputs": self._binary_outputs,
            "json_records": self._json_records,
            "sent_step_plan_cache_entries": len(self._sent_step_plans),
            "recv_step_plan_cache_entries": len(self._recv_batch_plans),
            "started": self._started,
            "bound": self._bound,
            "notify_thread_alive": bool(
                self._notify_thread is not None and self._notify_thread.is_alive()
            ),
            "pending_qkv_seq": int(peer.pending_qkv_seq) if peer is not None else 0,
            "pending_output_seq": int(peer.pending_output_seq)
            if peer is not None
            else 0,
            "notify_error": (
                None if self._notify_error is None else str(self._notify_error)
            ),
        }

    def sender_sync_mode(self) -> str:
        if self._stream_ordered:
            return "stream_ordered_ring"
        return "async_event_notifier" if self._async_doorbell else "stream_synchronize"

    def pending_async(self) -> bool:
        peer = self._peer
        if not self._async_doorbell or peer is None:
            return False
        return bool(peer.pending_qkv_seq != 0 or peer.pending_output_seq != 0)

    def current_notify_error(self) -> str | None:
        return None if self._notify_error is None else str(self._notify_error)

    def notify_queue_size(self) -> int:
        return 0 if self._notify_queue is None else int(self._notify_queue.qsize())

    def assert_ready(self) -> None:
        if self._doorbell_mm is None or self._recv_buffer is None:
            raise RuntimeError("PAPLocalFastTransport is closed")
        if not self._started or not self._bound or self._peer is None:
            raise RuntimeError("PAPLocalFastTransport is not ready")
        self._raise_if_notify_failed()

    def describe(self) -> str:
        return (
            f"actor={self.actor_id} device={self.device} "
            f"stream_ordered={self._stream_ordered} slots={self._slot_count} "
            f"async={self._async_doorbell} started={self._started} "
            f"bound={self._bound}"
        )

    def __str__(self) -> str:
        return repr(self)

    def __bool__(self) -> bool:
        return True

    def __len__(self) -> int:
        return int(self.buffer_bytes)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


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
