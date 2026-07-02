# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP same-machine direct CUDA IPC + spin-wait fast-path transport.

This transport is a research prototype intended to demonstrate that the
Projection<->Attention per-layer QKV/O transit can be driven well below the
~500 us/layer cost of the NIXL/UCX mailbox stack when both sides run on the
same machine.

Design highlights:

* Each side pre-allocates a pinned CUDA recv buffer on its local GPU and
  exports a CUDA IPC handle (via ``torch.multiprocessing.reductions``).
  Handles are exchanged through the existing PAP control plane (the HTTP
  bind handshake) as a small pickled + base64-encoded metadata blob.
* A small ``/dev/shm`` mmap'd "doorbell" file per side carries the monotonic
  sequence number plus payload size/offset for each direction.  Sender writes
  the doorbell only *after* the GPU memcpy is known complete; receiver
  spin-polls the doorbell.
* Default sender behavior uses ``cudaStreamSynchronize`` before ringing the
  doorbell.  An optional env-gated async mode records a CUDA event and lets a
  helper thread wait for completion before writing the doorbell.
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

import base64
import json
import mmap
import os
import pickle
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from math import prod
from typing import Any

import torch

# Local imports
from vllm.pap.data_plane import (
    PAPOffloadExecBatchDescriptor,
    PAPOffloadExecDescriptor,
    PAPTensorTransport,
    _offload_exec_batch_descriptor_from_metadata,
    _offload_exec_batch_descriptor_to_metadata,
)


# ---------------------------------------------------------------------------
# Constants / layout
# ---------------------------------------------------------------------------

DOORBELL_HEADER_BYTES = 32
DOORBELL_RECORD_BYTES = int(
    os.environ.get("PAP_LOCAL_FAST_DOORBELL_RECORD_BYTES", str(64 * 1024))
)
if DOORBELL_RECORD_BYTES < DOORBELL_HEADER_BYTES:
    raise RuntimeError("PAP_LOCAL_FAST_DOORBELL_RECORD_BYTES is too small")
DOORBELL_BYTES = 2 * DOORBELL_RECORD_BYTES
DOORBELL_SEQ_STRUCT = struct.Struct("<QQQQ")

DIR_QKV = 0
DIR_OUTPUT = DOORBELL_RECORD_BYTES

# Default recv buffer size.  Same default as the NIXL mailbox path.
DEFAULT_BUFFER_BYTES = 16 * 1024 * 1024

# Doorbell wait behavior. Keep a short tight spin for low-latency handoff, then
# back off so long waits do not burn a CPU core and interfere with the shared
# PA-side control plane / MPS scheduling.
SPIN_TIGHT_ITERS = int(os.environ.get("PAP_LOCAL_FAST_SPIN_ITERS", "2048"))
SPIN_YIELD_ITERS = int(os.environ.get("PAP_LOCAL_FAST_YIELD_ITERS", "64"))
SPIN_SLEEP_US = int(os.environ.get("PAP_LOCAL_FAST_SLEEP_US", "20"))
SPIN_SLEEP_AFTER_US = int(
    os.environ.get("PAP_LOCAL_FAST_SLEEP_AFTER_US", "50")
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _local_hostname() -> str:
    """Best-effort stable hostname for same-machine detection."""

    candidate = os.environ.get("PAP_LOCAL_FAST_HOSTNAME")
    if candidate:
        return candidate
    try:
        return socket.gethostname() or "unknown"
    except Exception:
        return "unknown"


def _doorbell_path(actor_id: str) -> str:
    base = os.environ.get("PAP_LOCAL_FAST_DOORBELL_DIR", "/dev/shm")
    safe_actor = "".join(c if c.isalnum() or c in "-_" else "_" for c in actor_id)
    return os.path.join(base, f"pap_local_fast_{safe_actor}.db")


def _open_or_create_doorbell(path: str) -> tuple[int, mmap.mmap]:
    """Open (and create if absent) the doorbell file and mmap it."""

    flags = os.O_RDWR | os.O_CREAT
    fd = os.open(path, flags, 0o600)
    try:
        os.ftruncate(fd, DOORBELL_BYTES)
    except OSError:
        os.close(fd)
        raise
    mm = mmap.mmap(fd, DOORBELL_BYTES, flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE)
    # Zero the entire region on creation; subsequent opens inherit the
    # truncation but we re-zero defensively only if the file looks empty.
    return fd, mm


def _doorbell_write(
    mm: mmap.mmap,
    dir_offset: int,
    *,
    seq: int,
    nbytes: int,
    offset: int,
    metadata: dict[str, Any],
) -> None:
    meta = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    if len(meta) > DOORBELL_RECORD_BYTES - DOORBELL_HEADER_BYTES:
        raise RuntimeError(
            "PAP local fast metadata is too large for the doorbell record"
        )
    start = int(dir_offset)
    body_start = start + DOORBELL_HEADER_BYTES
    mm[body_start : body_start + len(meta)] = meta
    header = DOORBELL_SEQ_STRUCT.pack(seq, nbytes, offset, len(meta))
    mm[start : start + DOORBELL_HEADER_BYTES] = header


def _doorbell_read_header(mm: mmap.mmap, dir_offset: int) -> tuple[int, int, int, int]:
    raw = bytes(mm[dir_offset : dir_offset + DOORBELL_HEADER_BYTES])
    seq, nbytes, offset, metadata_len = DOORBELL_SEQ_STRUCT.unpack(raw)
    return int(seq), int(nbytes), int(offset), int(metadata_len)


def _doorbell_read_metadata(
    mm: mmap.mmap,
    dir_offset: int,
    metadata_len: int,
) -> dict[str, Any]:
    if metadata_len < 0 or metadata_len > DOORBELL_RECORD_BYTES - DOORBELL_HEADER_BYTES:
        raise RuntimeError("PAP local fast doorbell metadata length is invalid")
    start = int(dir_offset) + DOORBELL_HEADER_BYTES
    raw = bytes(mm[start : start + metadata_len])
    data = json.loads(raw.decode("utf-8")) if raw else {}
    if not isinstance(data, dict):
        raise RuntimeError("PAP local fast doorbell metadata must be a dict")
    return data


def _pack_cuda_ipc_handle(tensor: torch.Tensor) -> bytes:
    """Return a base64-encoded pickle of reduce_tensor() for ``tensor``.

    ``torch.multiprocessing.reductions.reduce_tensor`` returns a
    ``(rebuild_callable, args_tuple)`` pair that, when invoked, opens the
    exporting process's CUDA IPC handle on the importing side.  The args
    contain the raw IPC handle bytes and metadata.
    """

    from torch.multiprocessing.reductions import reduce_tensor

    rebuild, args = reduce_tensor(tensor)
    return base64.b64encode(pickle.dumps((rebuild, args))).decode("ascii")


def _unpack_cuda_ipc_handle(blob: str) -> torch.Tensor:
    """Rebuild a CUDA tensor from a serialized IPC handle blob."""

    rebuild, args = pickle.loads(base64.b64decode(blob.encode("ascii")))
    return rebuild(*args)


def _ensure_peer_access(local_device: torch.device,
                        peer_device: torch.device) -> bool:
    """Try to enable P2P peer access from local -> peer.

    Returns True if peer access is enabled (or already was), False if it is
    not supported.  When both sides are the same device, returns True without
    doing anything.
    """

    if local_device.index == peer_device.index:
        return True
    if local_device.index is None or peer_device.index is None:
        return False
    try:
        can = torch.cuda.can_device_access_peer(local_device.index,
                                                peer_device.index)
    except Exception:
        can = False
    if not can:
        return False
    try:
        torch.cuda.device(local_device.index)
        torch._C._cuda_enable_peer_access(peer_device.index)
    except RuntimeError as exc:
        # Already-enabled returns "invalid device function" sometimes; treat
        # peer already enabled as success.
        if "peer access is already enabled" not in str(exc).lower():
            raise
    return True


def _sched_yield() -> None:
    try:
        os.sched_yield()
    except AttributeError:
        time.sleep(0)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, str(name), None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"unsupported PAP local fast tensor dtype: {name}")
    return dtype


def _payload_metadata(
    descriptor: PAPOffloadExecBatchDescriptor,
    tensor: torch.Tensor,
) -> dict[str, Any]:
    return {
        "descriptor": _offload_exec_batch_descriptor_to_metadata(descriptor),
        "shape": list(tensor.shape),
        "dtype": _dtype_name(tensor.dtype),
    }


# ---------------------------------------------------------------------------
# Message-like wrapper (duck-typed PAPMailboxMessage)
# ---------------------------------------------------------------------------


@dataclass
class _LocalFastMessage:
    """Minimal duck-typed stand-in for ``PAPMailboxMessage``.

    The mailbox transport returns message objects with ``tensor``,
    ``release()``, and ``kind`` attributes; some call sites use
    ``recv_*_batch_message`` and then call ``.release()`` on the result.
    For the local-fast transport the recv buffer is owned locally and reused
    across batches, so ``release()`` is a no-op.
    """

    msg_id: str
    kind: str
    tensor: torch.Tensor
    metadata: dict[str, Any]
    release_callback: Any = None
    _released: bool = False

    def release(self) -> None:
        self._released = True


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


@dataclass
class _PendingDoorbell:
    direction: int
    seq: int
    nbytes: int
    offset: int
    metadata: dict[str, Any]
    event: torch.cuda.Event
    src_tensor_ref: torch.Tensor
    peer_tensor_ref: torch.Tensor
    enqueue_time: float
    descriptor_layer_name: str
    descriptor_batch_id: str


@dataclass
class _PeerState:
    """Per-peer state cached on the local transport after bind_peer."""

    peer_tensor: torch.Tensor  # view into peer's recv buffer (local device)
    peer_doorbell_path: str
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
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    wait_cond: threading.Condition = field(init=False)

    def __post_init__(self) -> None:
        self.wait_cond = threading.Condition(self.send_lock)


class PAPLocalFastTransport:
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
                "PAPLocalFastTransport requires a CUDA device; got "
                f"{self.device}"
            )

        self.buffer_bytes = (
            int(buffer_bytes)
            if buffer_bytes is not None
            else int(os.environ.get("PAP_LOCAL_FAST_BUFFER_BYTES",
                                    str(DEFAULT_BUFFER_BYTES)))
        )
        if self.buffer_bytes <= 0:
            raise RuntimeError("PAP local fast buffer_bytes must be positive")

        # Allocate the local recv buffer (1D byte tensor on local GPU).
        torch.cuda.device(self.device)
        self._recv_buffer = torch.empty(
            self.buffer_bytes, dtype=torch.uint8, device=self.device
        )
        # Pin the underlying storage lifetime: hold a reference to the
        # untyped storage so the IPC handle stays valid until we drop it.
        self._recv_storage = self._recv_buffer.untyped_storage()

        # Build / open the local doorbell file.
        self._doorbell_path = _doorbell_path(self.actor_id)
        self._doorbell_fd, self._doorbell_mm = _open_or_create_doorbell(
            self._doorbell_path
        )
        # Zero the doorbell on (re)open.  This is safe because we only ever
        # have one transport per actor_id per machine.
        self._doorbell_mm[:] = b"\x00" * DOORBELL_BYTES

        self._peer: _PeerState | None = None
        self._started = False
        self._bound = False
        self._trace = _env_bool("PAP_OFFLOAD_EXEC_TRACE", False)
        self._async_doorbell = _env_bool("PAP_LOCAL_FAST_ASYNC_DOORBELL", False)
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
        payload = {
            "v": 1,
            "actor_id": self.actor_id,
            "hostname": _local_hostname(),
            "device_index": int(self.device.index or 0),
            "buffer_bytes": int(self.buffer_bytes),
            "doorbell_path": self._doorbell_path,
            "ipc_handle": ipc_blob,
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
        peer_device_index = int(payload.get("device_index", 0))
        peer_buffer_bytes = int(payload.get("buffer_bytes", 0))
        peer_doorbell_path = str(payload["doorbell_path"])
        ipc_blob = str(payload["ipc_handle"])

        # Rebuild the peer's tensor on our local device.  reduce_tensor
        # rebuilds on the original device index; if that is different from
        # ours, the kernel will route via peer access (or staged copy).
        peer_tensor = _unpack_cuda_ipc_handle(ipc_blob)
        peer_device = peer_tensor.device

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
            peer_fd, DOORBELL_BYTES, flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        self._peer = _PeerState(
            peer_tensor=peer_tensor,
            peer_doorbell_path=peer_doorbell_path,
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
        if self._async_doorbell:
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
            raise RuntimeError(
                "PAP local fast async doorbell notifier failed"
            ) from exc

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
        metadata: dict[str, Any],
    ) -> None:
        _doorbell_write(
            peer.peer_doorbell_mm,
            direction,
            seq=seq,
            nbytes=nbytes,
            offset=offset,
            metadata=metadata,
        )

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
                    metadata=job.metadata,
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
            raise RuntimeError(
                "PAPLocalFastTransport: bind_peer must be called first"
            )
        return self._peer

    def _send_to_peer(
        self,
        *,
        direction: int,
        descriptor: PAPOffloadExecBatchDescriptor,
        tensor: torch.Tensor,
    ) -> int:
        """Memcpy ``tensor`` into peer's recv buffer and ring the doorbell.

        Returns the slot offset used in the peer's buffer.
        """

        peer = self._require_peer()
        self._raise_if_notify_failed()
        nbytes = int(tensor.numel() * tensor.element_size())
        if nbytes > self.buffer_bytes:
            raise RuntimeError(
                f"PAP local fast payload {nbytes}B exceeds peer buffer "
                f"{self.buffer_bytes}B"
            )
        if nbytes > peer.peer_tensor.numel() * peer.peer_tensor.element_size():
            raise RuntimeError(
                f"PAP local fast payload {nbytes}B exceeds peer IPC buffer "
                f"{peer.peer_tensor.numel() * peer.peer_tensor.element_size()}B"
            )

        # Determine slot offset.  For the prototype we always use offset 0
        # because both directions reuse one buffer and correctness still
        # requires prior sends to finish before the slot is reused.
        offset = 0
        metadata = _payload_metadata(descriptor, tensor)

        with peer.wait_cond:
            if self._async_doorbell:
                while peer.pending_qkv_seq != 0 or peer.pending_output_seq != 0:
                    self._raise_if_notify_failed()
                    peer.wait_cond.wait(timeout=0.01)
            self._raise_if_notify_failed()

            # Copy raw bytes from the source tensor into the peer's uint8 recv
            # buffer.  We re-view both sides as 1-D uint8 of the correct length
            # so dtype/shape mismatches don't matter; the receiver reinterprets
            # the bytes via the carried ``nbytes`` + the descriptor's expected
            # layout.
            src_bytes = tensor.detach().contiguous().view(-1).view(torch.uint8)
            src_bytes = src_bytes.narrow(0, 0, nbytes)
            dst_bytes = peer.peer_tensor.narrow(0, offset, nbytes)

            stream = torch.cuda.current_stream(self.device)
            t_memcpy_start = time.perf_counter()
            dst_bytes.copy_(src_bytes, non_blocking=True)
            t_sync_start = time.perf_counter()
            seq = self._next_seq(peer, direction)

            if self._async_doorbell:
                event = torch.cuda.Event()
                event.record(stream)
                self._set_pending_seq(peer, direction, seq)
                enqueue_time = time.perf_counter()
                assert self._notify_queue is not None
                self._notify_queue.put(
                    _PendingDoorbell(
                        direction=direction,
                        seq=seq,
                        nbytes=nbytes,
                        offset=offset,
                        metadata=metadata,
                        event=event,
                        src_tensor_ref=tensor,
                        peer_tensor_ref=peer.peer_tensor,
                        enqueue_time=enqueue_time,
                        descriptor_layer_name=descriptor.layer_name,
                        descriptor_batch_id=getattr(descriptor, "batch_id", ""),
                    )
                )
                t_done = time.perf_counter()
                sync_ms = 0.0
                enqueue_ms = (t_done - t_sync_start) * 1000.0
                doorbell_ms = 0.0
            else:
                stream.synchronize()
                t_doorbell_start = time.perf_counter()
                self._write_doorbell_sync(
                    peer=peer,
                    direction=direction,
                    seq=seq,
                    nbytes=nbytes,
                    offset=offset,
                    metadata=metadata,
                )
                t_done = time.perf_counter()
                sync_ms = (t_doorbell_start - t_sync_start) * 1000.0
                enqueue_ms = 0.0
                doorbell_ms = (t_done - t_doorbell_start) * 1000.0

        if self._trace:
            kind = "qkv" if direction == DIR_QKV else "output"
            logger.info(
                "PAP local fast transport send trace kind=%s layer=%s "
                "batch=%s memcpy_ms=%.3f sync_ms=%.3f enqueue_ms=%.3f "
                "doorbell_ms=%.3f async=%d nbytes=%d seq=%d",
                kind,
                descriptor.layer_name,
                getattr(descriptor, "batch_id", ""),
                (t_sync_start - t_memcpy_start) * 1000.0,
                sync_ms,
                enqueue_ms,
                doorbell_ms,
                int(self._async_doorbell),
                nbytes,
                seq,
            )
        return offset

    def _recv_from_peer(
        self,
        *,
        direction: int,
    ) -> tuple[int, int, int, dict[str, Any]]:
        """Spin until peer has rung the doorbell for ``direction``."""

        peer = self._require_peer()
        # We read the *local* doorbell (peer wrote into it via mmap).
        mm = self._doorbell_mm
        expected = (
            peer.expected_qkv_seq if direction == DIR_QKV
            else peer.expected_output_seq
        )
        iters = 0
        t_start = time.perf_counter()
        while True:
            seq, nbytes, offset, metadata_len = _doorbell_read_header(mm, direction)
            if seq >= expected and seq != 0:
                break
            iters += 1
            if iters < SPIN_TIGHT_ITERS:
                continue
            if iters < SPIN_TIGHT_ITERS + SPIN_YIELD_ITERS:
                _sched_yield()
                continue
            waited_us = (time.perf_counter() - t_start) * 1_000_000.0
            if waited_us >= SPIN_SLEEP_AFTER_US:
                time.sleep(max(SPIN_SLEEP_US, 0) / 1_000_000.0)
            else:
                _sched_yield()
        metadata = _doorbell_read_metadata(mm, direction, metadata_len)
        # Bump our expectation for the next round.
        if direction == DIR_QKV:
            peer.expected_qkv_seq = expected + 1
        else:
            peer.expected_output_seq = expected + 1
        if self._trace:
            kind = "qkv" if direction == DIR_QKV else "output"
            logger.info(
                "PAP local fast transport recv trace kind=%s "
                "spin_ms=%.3f nbytes=%d offset=%d seq=%d",
                kind,
                (time.perf_counter() - t_start) * 1000.0,
                nbytes,
                offset,
                seq,
            )
        return seq, nbytes, offset, metadata

    def _materialize_recv(
        self,
        *,
        nbytes: int,
        offset: int,
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        shape = tuple(int(dim) for dim in metadata["shape"])
        dtype = _dtype_from_name(str(metadata["dtype"]))
        expected_nbytes = int(prod(shape) * torch.empty((), dtype=dtype).element_size())
        if expected_nbytes != int(nbytes):
            raise RuntimeError(
                f"PAP local fast payload size mismatch: metadata shape={shape} "
                f"dtype={dtype} expects {expected_nbytes} bytes, got {nbytes}"
            )
        view = self._recv_buffer.narrow(0, int(offset), int(nbytes))
        return view.view(dtype).reshape(shape)

    # ------------------------------------------------------------------
    # Public transport API (mirrors PAPNixlMailboxOffloadExecTransport)
    # ------------------------------------------------------------------

    # --- single-descriptor variants (unused on hot path, kept for API) ---

    def send_qkv(
        self,
        descriptor: PAPOffloadExecDescriptor,
        qkv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        batch = PAPOffloadExecBatchDescriptor(
            layer_name=descriptor.layer_name,
            items=(descriptor,),
        )
        self.send_qkv_batch(batch, qkv, remote_address=remote_address)

    def recv_qkv(
        self,
        descriptor: PAPOffloadExecDescriptor,
        *,
        remote_address: str,
    ) -> torch.Tensor:
        batch = PAPOffloadExecBatchDescriptor(
            layer_name=descriptor.layer_name,
            items=(descriptor,),
        )
        return self.recv_qkv_batch(batch, remote_address=remote_address)

    def send_output(
        self,
        descriptor: PAPOffloadExecDescriptor,
        output: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        batch = PAPOffloadExecBatchDescriptor(
            layer_name=descriptor.layer_name,
            items=(descriptor,),
        )
        self.send_output_batch(batch, output, remote_address=remote_address)

    def recv_output(
        self,
        descriptor: PAPOffloadExecDescriptor,
        *,
        remote_address: str,
    ) -> torch.Tensor:
        batch = PAPOffloadExecBatchDescriptor(
            layer_name=descriptor.layer_name,
            items=(descriptor,),
        )
        return self.recv_output_batch(batch, remote_address=remote_address)

    # --- batched variants (the actual hot path) ---

    def send_qkv_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        qkv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        self._send_to_peer(
            direction=DIR_QKV, descriptor=descriptor, tensor=qkv
        )

    def send_qkv_batch_direct(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        qkv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        # In the local-fast path there is no separate "direct payload slot"
        # ceremony; the batched path is already direct.
        self.send_qkv_batch(descriptor, qkv, remote_address=remote_address)

    def recv_qkv_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        remote_address: str,
    ) -> torch.Tensor:
        _seq, nbytes, offset, metadata = self._recv_from_peer(direction=DIR_QKV)
        return self._materialize_recv(
            nbytes=nbytes, offset=offset, metadata=metadata
        )

    def send_output_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        output: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        self._send_to_peer(
            direction=DIR_OUTPUT, descriptor=descriptor, tensor=output
        )

    def recv_output_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        remote_address: str,
    ) -> torch.Tensor:
        _seq, nbytes, offset, metadata = self._recv_from_peer(direction=DIR_OUTPUT)
        return self._materialize_recv(
            nbytes=nbytes, offset=offset, metadata=metadata
        )

    # --- message-style variants (used by attention mailbox loop) ---

    def recv_qkv_batch_message(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        remote_address: str,
    ) -> Any:
        tensor = self.recv_qkv_batch(descriptor, remote_address=remote_address)
        return _LocalFastMessage(
            msg_id=descriptor.qkv_tensor_id,
            kind="attention_task_batch",
            tensor=tensor,
            metadata=_offload_exec_batch_descriptor_to_metadata(descriptor),
        )

    def recv_output_batch_message(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        remote_address: str,
    ) -> Any:
        tensor = self.recv_output_batch(
            descriptor, remote_address=remote_address
        )
        return _LocalFastMessage(
            msg_id=descriptor.output_tensor_id,
            kind="attention_result_batch",
            tensor=tensor,
            metadata=_offload_exec_batch_descriptor_to_metadata(descriptor),
        )

    def recv_next_qkv_batch(
        self,
    ) -> tuple[PAPOffloadExecBatchDescriptor, torch.Tensor]:
        descriptor, message = self.recv_next_qkv_batch_message()
        return descriptor, message.tensor

    def recv_next_qkv_batch_message(
        self,
    ) -> tuple[PAPOffloadExecBatchDescriptor, Any]:
        _seq, nbytes, offset, metadata = self._recv_from_peer(direction=DIR_QKV)
        descriptor_metadata = dict(metadata["descriptor"])
        descriptor = _offload_exec_batch_descriptor_from_metadata(
            descriptor_metadata
        )
        tensor = self._materialize_recv(
            nbytes=nbytes, offset=offset, metadata=metadata
        )
        message = _LocalFastMessage(
            msg_id=descriptor.qkv_tensor_id,
            kind="attention_task_batch",
            tensor=tensor,
            metadata=descriptor_metadata,
        )
        return descriptor, message

    def recv_next_attention_batch_message(
        self,
    ) -> tuple[PAPOffloadExecBatchDescriptor, Any]:
        return self.recv_next_qkv_batch_message()

    # ------------------------------------------------------------------
    # Cleanup (best-effort; daemon process exit will reclaim resources)
    # ------------------------------------------------------------------

    def flush(self) -> None:
        peer = self._peer
        if not self._async_doorbell or peer is None:
            return
        with peer.wait_cond:
            while (
                (peer.pending_qkv_seq != 0 or peer.pending_output_seq != 0)
                and self._notify_error is None
            ):
                peer.wait_cond.wait(timeout=0.01)
        self._raise_if_notify_failed()

    def barrier(self) -> None:
        self.flush()

    def wait_idle(self) -> None:
        self.flush()

    def close(self) -> None:
        try:
            self.flush()
            if self._notify_queue is not None and self._notify_thread is not None:
                if self._notify_thread.is_alive():
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
            f"device={self.device!s}, async_doorbell={self._async_doorbell})"
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
        return (
            "async_event_notifier"
            if self._async_doorbell
            else "stream_synchronize"
        )

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
            else int(os.environ.get("PAP_LOCAL_FAST_BUFFER_BYTES",
                                    str(DEFAULT_BUFFER_BYTES)))
        ),
    )


# Late import to avoid pulling logging infrastructure at module import time
# in case the parent process hasn't configured it yet.
import logging  # noqa: E402

logger = logging.getLogger(__name__)
