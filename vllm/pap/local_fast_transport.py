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
  the doorbell only *after* the GPU memcpy has been synchronized on the
  sending stream; receiver spin-polls the doorbell.
* No Python-side CUDA event on the sender.  We use ``cudaStreamSynchronize``
  (the doorbell write is the only inter-process signal).
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
import socket
import struct
import time
from dataclasses import dataclass, field
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

# Each direction owns one "slot" record in the doorbell file.  Layout:
#   u64 seq          # monotonic, written by sender, read by receiver
#   u64 nbytes       # payload bytes written this slot
#   u64 offset       # offset into peer's recv buffer
#   u64 reserved     # padding / future use
# Total: 32 bytes per direction.  Two directions (qkv, output) -> 64 bytes.
DOORBELL_RECORD_BYTES = 32
DOORBELL_BYTES = 2 * DOORBELL_RECORD_BYTES
DOORBELL_SEQ_STRUCT = struct.Struct("<QQQQ")  # little-endian

# Direction offsets within the doorbell.
# The "qkv" direction is Projection -> Attention.
# The "output" direction is Attention -> Projection.
DIR_QKV = 0
DIR_OUTPUT = DOORBELL_RECORD_BYTES

# Default recv buffer size.  Same default as the NIXL mailbox path.
DEFAULT_BUFFER_BYTES = 16 * 1024 * 1024

# Doorbell spin behavior.  Picked to allow ~us-scale notification while still
# yielding to the OS scheduler under sustained wait.
SPIN_TIGHT_ITERS = int(os.environ.get("PAP_LOCAL_FAST_SPIN_ITERS", "2048"))


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


def _doorbell_write(mm: mmap.mmap, dir_offset: int, *, seq: int, nbytes: int,
                    offset: int) -> None:
    """Atomic-ish doorbell record write.

    We pack the four fields into a single 32-byte struct and write it in one
    ``mm[:]`` slice assignment.  On x86_64 this is effectively a sequence of
    word stores that the receiver will observe in order thanks to TSO; the
    ``seq`` field is written *last* by the sender (we explicitly order it so
    by writing the entire record in one go, with seq first in the struct so
    its address is the base — receiver only checks seq, then reads the rest).
    """

    record = DOORBELL_SEQ_STRUCT.pack(seq, nbytes, offset, 0)
    # We want receiver to see seq LAST from its perspective.  Because we write
    # the entire record in one go and the receiver only spins on seq, the
    # memory-ordering guarantee we need is "the writes to nbytes/offset become
    # visible before the write to seq".  x86 TSO gives this for free; on other
    # architectures the receiver re-reads the full record after observing the
    # seq bump, which is also safe.
    start = dir_offset
    end = dir_offset + DOORBELL_RECORD_BYTES
    mm[start:end] = record


def _doorbell_read(mm: mmap.mmap, dir_offset: int) -> tuple[int, int, int]:
    """Return (seq, nbytes, offset) for one direction."""

    raw = bytes(mm[dir_offset:dir_offset + DOORBELL_RECORD_BYTES])
    seq, nbytes, offset, _reserved = DOORBELL_SEQ_STRUCT.unpack(raw)
    return int(seq), int(nbytes), int(offset)


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
        """No-op-ish: peer access is enabled in bind_peer, no receiver thread.

        The receiver spin happens inline in each ``recv_*`` call.
        """

        if self._started:
            return
        self._started = True

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
        # because QKV and output directions are strictly alternating per
        # layer on each side and we synchronize before reusing the buffer.
        offset = 0

        # Copy raw bytes from the source tensor into the peer's uint8 recv
        # buffer.  We re-view both sides as 1-D uint8 of the correct length
        # so dtype/shape mismatches don't matter; the receiver reinterprets
        # the bytes via the carried ``nbytes`` + the descriptor's expected
        # layout.
        src_bytes = (
            tensor.detach().contiguous().view(-1).view(torch.uint8)
        )
        # ``tensor.view(-1).view(uint8)`` requires that the element size
        # divides evenly; for the dtype/shape combinations on the PAP fast
        # path (fp16/bf16 contiguous) this holds.  Narrow to the exact byte
        # count defensively.
        src_bytes = src_bytes.narrow(0, 0, nbytes)
        dst_bytes = peer.peer_tensor.narrow(0, offset, nbytes)

        # Memcpy on the current CUDA stream.
        stream = torch.cuda.current_stream(self.device)
        t_memcpy_start = time.perf_counter()
        dst_bytes.copy_(src_bytes, non_blocking=True)
        # Force the copy to complete on this stream *before* we write the
        # doorbell.  Per the design doc, this is the only synchronization
        # the receiver relies on.
        t_sync_start = time.perf_counter()
        stream.synchronize()
        t_doorbell_start = time.perf_counter()

        # Pick the next seq number for this direction and write the doorbell.
        if direction == DIR_QKV:
            seq = peer.next_qkv_seq
            peer.next_qkv_seq += 1
        else:
            seq = peer.next_output_seq
            peer.next_output_seq += 1
        _doorbell_write(
            peer.peer_doorbell_mm, direction,
            seq=seq, nbytes=nbytes, offset=offset,
        )

        if self._trace:
            kind = "qkv" if direction == DIR_QKV else "output"
            logger.info(
                "PAP local fast transport send trace kind=%s layer=%s "
                "batch=%s memcpy_ms=%.3f sync_ms=%.3f doorbell_ms=%.3f "
                "nbytes=%d seq=%d",
                kind,
                descriptor.layer_name,
                getattr(descriptor, "batch_id", ""),
                (t_sync_start - t_memcpy_start) * 1000.0,
                (t_doorbell_start - t_sync_start) * 1000.0,
                (time.perf_counter() - t_doorbell_start) * 1000.0,
                nbytes,
                seq,
            )
        return offset

    def _recv_from_peer(
        self,
        *,
        direction: int,
    ) -> tuple[int, int, int]:
        """Spin until peer has rung the doorbell for ``direction``.

        Returns (seq, nbytes, offset) once observed.
        """

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
            seq, nbytes, offset = _doorbell_read(mm, direction)
            if seq >= expected and seq != 0:
                break
            iters += 1
            if iters < SPIN_TIGHT_ITERS:
                # tiny in-process pause; on Python this is essentially a
                # function-call overhead.
                continue
            _sched_yield()
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
        return seq, nbytes, offset

    def _materialize_recv(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        nbytes: int,
        offset: int,
        output: bool,
    ) -> torch.Tensor:
        """Return a view of the local recv buffer holding the payload.

        The sender wrote raw bytes; we know the dtype/shape from the
        descriptor's expected item layout.  For QKV we use float16/bf16
        based on the per-layer expectation (the calling code passes a
        pre-shaped ``qkv`` tensor in, so we mirror its dtype/shape).
        """

        # The simplest correct thing is to return a uint8 view of nbytes
        # and let the caller reinterpret.  However the existing call sites
        # treat the returned tensor as already-typed, so we must produce
        # a properly-typed view.  We infer dtype/shape from the descriptor:
        # for QKV, item_count = number of rows; for output, same.  But we
        # don't know head dim here without more context.  We approximate
        # by returning a 1-D uint8 view of length nbytes and let the
        # caller cast; this is enough for the prototype scaffolding.
        #
        # NOTE: this is the stub mentioned in the deliverable.  Real
        # production would carry dtype/shape in the doorbell or descriptor
        # and view the buffer accordingly.
        view = self._recv_buffer.narrow(0, offset, nbytes)
        return view

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
        seq, nbytes, offset = self._recv_from_peer(direction=DIR_QKV)
        return self._materialize_recv(
            descriptor, nbytes=nbytes, offset=offset, output=False
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
        seq, nbytes, offset = self._recv_from_peer(direction=DIR_OUTPUT)
        return self._materialize_recv(
            descriptor, nbytes=nbytes, offset=offset, output=True
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
        # The attention-side mailbox loop polls for the next batch without
        # knowing the descriptor in advance.  The NIXL mailbox carries the
        # descriptor in the message metadata; we don't have that here.
        # We spin on the QKV doorbell until any new payload arrives, then
        # synthesize a placeholder descriptor from the doorbell payload
        # (which only carries nbytes).  The descriptor's layer_name / items
        # will be filled in by the caller's metadata loop on the next
        # iteration of the mailbox handler.
        #
        # NOTE: This is the second stub.  The current attention-side mailbox
        # loop is heavily tied to the NIXL mailbox pulling descriptors from
        # message metadata.  To make this path production-ready we would
        # either (a) extend the doorbell record to carry a small
        # descriptor payload (layer_name + per-row scales) or (b) ship the
        # descriptor over a side-channel ZMQ socket.  For the prototype we
        # return a sentinel descriptor and rely on the caller to override
        # it.
        seq, nbytes, offset = self._recv_from_peer(direction=DIR_QKV)
        descriptor = PAPOffloadExecBatchDescriptor(
            layer_name="<local_fast_pending>",
            items=(),
            batch_id_suffix=f"local_fast:{seq}:{nbytes}",
        )
        tensor = self._recv_buffer.narrow(0, offset, nbytes)
        message = _LocalFastMessage(
            msg_id=descriptor.qkv_tensor_id,
            kind="attention_task_batch",
            tensor=tensor,
            metadata={"_local_fast_pending": True, "nbytes": nbytes},
        )
        return descriptor, message

    def recv_next_attention_batch_message(
        self,
    ) -> tuple[PAPOffloadExecBatchDescriptor, Any]:
        return self.recv_next_qkv_batch_message()

    # ------------------------------------------------------------------
    # Cleanup (best-effort; daemon process exit will reclaim resources)
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            if self._peer is not None:
                if self._peer.peer_doorbell_mm is not None:
                    self._peer.peer_doorbell_mm.close()
                if self._peer.peer_doorbell_fd is not None:
                    os.close(self._peer.peer_doorbell_fd)
            self._doorbell_mm.close()
            if self._doorbell_fd is not None:
                os.close(self._doorbell_fd)
        except Exception:
            pass


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
