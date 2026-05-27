# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP bidirectional mailbox primitives.

The mailbox runtime models Projection and Attention as symmetric actors:

* an actor puts completed work into its local output pool;
* a backend asynchronously delivers that message to the peer task pool;
* the output entry is removed only after delivery is acknowledged.

The in-process backend is intentionally small and deterministic. The NIXL
backend will use the same actor/pool contract with registered memory and NIXL
notifications.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Condition, Event, RLock, Thread
from typing import Any, Protocol

import msgspec
import torch

logger = logging.getLogger(__name__)
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _nixl_mailbox_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    result = float(default if value in (None, "") else value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _nixl_mailbox_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.lower() in _TRUE_ENV_VALUES


def _nixl_mailbox_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    result = int(default if value in (None, "") else value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _nixl_mailbox_agent_config(config_cls: type | None) -> Any | None:
    if config_cls is None:
        return None
    return config_cls(
        num_threads=_nixl_mailbox_env_int("PAP_NIXL_MAILBOX_NUM_THREADS", 4),
        capture_telemetry=_nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_CAPTURE_TELEMETRY", True
        ),
    )


def _nixl_mailbox_trace_enabled() -> bool:
    value = os.environ.get("PAP_NIXL_MAILBOX_TRACE")
    if value not in (None, ""):
        return value.lower() in _TRUE_ENV_VALUES
    return os.environ.get("PAP_OFFLOAD_EXEC_TRACE", "").lower() in _TRUE_ENV_VALUES


_NIXL_MAILBOX_MSGPACK_PREFIX = b"PAPM1\x00"


def _encode_nixl_mailbox_notification(
    payload: dict[str, Any], *, use_msgpack: bool
) -> bytes:
    if use_msgpack:
        return _NIXL_MAILBOX_MSGPACK_PREFIX + msgspec.msgpack.encode(payload)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _decode_nixl_mailbox_notification(payload: bytes) -> dict[str, Any]:
    if payload.startswith(_NIXL_MAILBOX_MSGPACK_PREFIX):
        data = msgspec.msgpack.decode(payload[len(_NIXL_MAILBOX_MSGPACK_PREFIX) :])
    else:
        data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid PAP NIXL mailbox notification: {data!r}")
    return data


@dataclass(frozen=True)
class PAPNixlMailboxAgentMetadata:
    agent_metadata: bytes
    send_buffer_addr: int
    device_id: int
    slot_count: int
    slot_bytes: int


def _encode_nixl_mailbox_agent_metadata(
    *,
    agent_metadata: bytes,
    send_buffer_addr: int,
    device_id: int,
    slot_count: int,
    slot_bytes: int,
) -> bytes:
    payload = {
        "type": "pap_nixl_mailbox_agent",
        "version": 1,
        "agent_metadata_b64": base64.b64encode(agent_metadata).decode("ascii"),
        "send_buffer_addr": int(send_buffer_addr),
        "device_id": int(device_id),
        "slot_count": int(slot_count),
        "slot_bytes": int(slot_bytes),
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _decode_nixl_mailbox_agent_metadata(
    metadata: bytes,
) -> PAPNixlMailboxAgentMetadata:
    try:
        payload = json.loads(metadata.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return PAPNixlMailboxAgentMetadata(
            agent_metadata=metadata,
            send_buffer_addr=0,
            device_id=0,
            slot_count=1,
            slot_bytes=0,
        )
    if not isinstance(payload, dict) or payload.get("type") != "pap_nixl_mailbox_agent":
        return PAPNixlMailboxAgentMetadata(
            agent_metadata=metadata,
            send_buffer_addr=0,
            device_id=0,
            slot_count=1,
            slot_bytes=0,
        )
    return PAPNixlMailboxAgentMetadata(
        agent_metadata=base64.b64decode(str(payload["agent_metadata_b64"])),
        send_buffer_addr=int(payload["send_buffer_addr"]),
        device_id=int(payload["device_id"]),
        slot_count=int(payload["slot_count"]),
        slot_bytes=int(payload["slot_bytes"]),
    )


@dataclass(frozen=True)
class PAPMailboxMessage:
    """One complete PAP mailbox message.

    The metadata describes the work item. The tensor is the payload associated
    with that work item: QKV for Projection->Attention, or attention output for
    Attention->Projection.
    """

    msg_id: str
    kind: str
    metadata: dict[str, Any]
    tensor: torch.Tensor
    payload_segments: tuple[torch.Tensor, ...] | None = field(
        default=None, repr=False, compare=False
    )
    payload_shape: tuple[int, ...] | None = None
    release_callback: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )
    _released: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.msg_id:
            raise ValueError("PAP mailbox message requires msg_id")
        if not self.kind:
            raise ValueError("PAP mailbox message requires kind")
        if self.payload_segments is not None:
            if not self.payload_segments:
                raise ValueError("PAP mailbox payload_segments must not be empty")
            if self.payload_shape is None:
                raise ValueError("PAP mailbox segmented payload requires payload_shape")
            object.__setattr__(
                self,
                "payload_segments",
                tuple(self.payload_segments),
            )
            object.__setattr__(
                self,
                "payload_shape",
                tuple(int(dim) for dim in self.payload_shape),
            )

    def release(self) -> None:
        callback = self.release_callback
        if callback is None or self._released:
            return
        object.__setattr__(self, "_released", True)
        callback()

    def __del__(self) -> None:
        if self.release_callback is None or self._released:
            return
        try:
            logger.warning(
                "PAP mailbox message %s (%s) was garbage-collected without "
                "release(); releasing its receive slot",
                self.msg_id,
                self.kind,
            )
            self.release()
        except Exception:
            pass


class PAPMailboxBackend(Protocol):
    """Delivery backend used by mailbox actors."""

    def register_actor(self, actor: PAPMailboxActor) -> None: ...

    def deliver(self, sender: PAPMailboxActor, message: PAPMailboxMessage) -> None: ...


class PAPMailboxActor:
    """Local task/output pools for one PAP runtime actor."""

    def __init__(self, actor_id: str, *, backend: PAPMailboxBackend) -> None:
        if not actor_id:
            raise ValueError("actor_id must not be empty")
        self.actor_id = str(actor_id)
        self._backend = backend
        self._peer: PAPMailboxActor | None = None
        self._lock = RLock()
        self._cv = Condition(self._lock)
        self._task_pool: OrderedDict[str, PAPMailboxMessage] = OrderedDict()
        self._output_pool: OrderedDict[str, PAPMailboxMessage] = OrderedDict()
        self._backend.register_actor(self)

    @property
    def output_pool_size(self) -> int:
        with self._lock:
            return len(self._output_pool)

    @property
    def task_pool_size(self) -> int:
        with self._lock:
            return len(self._task_pool)

    @property
    def peer(self) -> PAPMailboxActor:
        if self._peer is None:
            raise RuntimeError(f"PAP mailbox actor {self.actor_id} has no peer")
        return self._peer

    def bind_peer(self, peer: PAPMailboxActor) -> None:
        if peer.actor_id == self.actor_id:
            raise ValueError("PAP mailbox actor cannot bind to itself")
        with self._lock:
            self._peer = peer

    def enqueue_output(self, message: PAPMailboxMessage) -> None:
        """Put a completed message in the output pool and ask backend to send."""

        with self._lock:
            if message.msg_id in self._output_pool:
                raise ValueError(f"duplicate output message id: {message.msg_id}")
            self._output_pool[message.msg_id] = message
        self._backend.deliver(self, message)

    def receive_task(self, message: PAPMailboxMessage) -> None:
        """Receive a peer-delivered message into the local task pool."""

        with self._cv:
            self._task_pool[message.msg_id] = message
            self._cv.notify_all()

    def acknowledge_output(self, msg_id: str) -> None:
        """Remove a delivered message from the output pool."""

        with self._lock:
            self._output_pool.pop(str(msg_id), None)

    def pop_task(
        self,
        msg_id: str | None = None,
        *,
        timeout: float | None = None,
    ) -> PAPMailboxMessage:
        """Pop a task by id, or the oldest task when id is omitted."""

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._cv:
            while True:
                if msg_id is None and self._task_pool:
                    _, message = self._task_pool.popitem(last=False)
                    return message
                if msg_id is not None and msg_id in self._task_pool:
                    return self._task_pool.pop(msg_id)
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"timed out waiting for PAP mailbox task {msg_id}"
                        )
                    self._cv.wait(timeout=remaining)
                else:
                    self._cv.wait()


class InProcessPAPMailboxBackend:
    """Deterministic mailbox backend used by unit tests."""

    def __init__(self, *, auto_ack: bool = True) -> None:
        self._actors: dict[str, PAPMailboxActor] = {}
        self._auto_ack = bool(auto_ack)

    def register_actor(self, actor: PAPMailboxActor) -> None:
        self._actors[actor.actor_id] = actor

    def deliver(self, sender: PAPMailboxActor, message: PAPMailboxMessage) -> None:
        peer = sender.peer
        peer.receive_task(message)
        if self._auto_ack:
            sender.acknowledge_output(message.msg_id)

    def ack(self, actor_id: str, msg_id: str) -> None:
        self._actors[str(actor_id)].acknowledge_output(str(msg_id))


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, str(name), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported PAP mailbox tensor dtype: {name}")
    return dtype


def _agent_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class PAPNixlMailboxEndpoint:
    """Single-peer NIXL mailbox endpoint for PAP actor messages."""

    def __init__(
        self,
        *,
        actor_id: str,
        device: torch.device,
        buffer_bytes: int,
        wrapper_cls: type | None = None,
    ) -> None:
        if buffer_bytes <= 0:
            raise ValueError("buffer_bytes must be positive")
        self.actor_id = str(actor_id)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("PAP NIXL mailbox currently requires a CUDA device")
        self.buffer_bytes = int(buffer_bytes)
        self.device_id = int(self.device.index or 0)
        self.memory_type = "VRAM"
        self._lock = RLock()
        self._cv = Condition(self._lock)
        self._poll_lock = RLock()
        self._closed = Event()
        self._sender_thread: Thread | None = None
        self._receiver_thread: Thread | None = None
        self._send_queue: Queue[PAPMailboxMessage] = Queue()
        self._incoming: OrderedDict[str, PAPMailboxMessage] = OrderedDict()
        self._output_pool: OrderedDict[str, PAPMailboxMessage] = OrderedDict()
        self._send_enqueued_at: dict[str, float] = {}
        self._acked: set[str] = set()
        self._pending_acks: OrderedDict[str, None] = OrderedDict()
        self._slot_count = _nixl_mailbox_env_int("PAP_NIXL_MAILBOX_SLOT_COUNT", 1)
        if self._slot_count <= 0:
            raise ValueError("PAP_NIXL_MAILBOX_SLOT_COUNT must be positive")
        self._slot_bytes = self.buffer_bytes // self._slot_count
        if self._slot_bytes <= 0:
            raise ValueError("PAP_NIXL_MAILBOX_SLOT_COUNT is too large")
        self._recv_slot_count = _nixl_mailbox_env_int(
            "PAP_NIXL_MAILBOX_RECV_SLOT_COUNT", self._slot_count
        )
        if self._recv_slot_count <= 0:
            raise ValueError("PAP_NIXL_MAILBOX_RECV_SLOT_COUNT must be positive")
        self._recv_slot_bytes = self.buffer_bytes // self._recv_slot_count
        if self._recv_slot_bytes <= 0:
            raise ValueError("PAP_NIXL_MAILBOX_RECV_SLOT_COUNT is too large")
        self._next_send_slot = 0
        self._peer_send_buffer_addr: int | None = None
        self._peer_device_id: int | None = None
        self._peer_slot_count = 1
        self._peer_slot_bytes = 0
        self._poll_sleep_seconds = _nixl_mailbox_env_float(
            "PAP_NIXL_MAILBOX_POLL_SECONDS", 0.00001
        )
        self._xfer_poll_sleep_seconds = _nixl_mailbox_env_float(
            "PAP_NIXL_MAILBOX_XFER_POLL_SECONDS", 0.0
        )
        self._trace_enabled = _nixl_mailbox_trace_enabled()
        self._inline_poll_enabled = _nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_INLINE_POLL", False
        )
        self._inline_publish_enabled = _nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_INLINE_PUBLISH", False
        )
        self._slot_protocol_enabled = _nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_SLOT_PROTOCOL", True
        )
        self._zero_copy_recv_enabled = _nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_ZERO_COPY_RECV", True
        )
        self._cache_xfer_dlists_enabled = _nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_CACHE_XFER_DLISTS", True
        )
        self._piggyback_acks_enabled = _nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_PIGGYBACK_ACKS", False
        )
        self._msgpack_notifications_enabled = _nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_MSGPACK_NOTIF", True
        )
        self._async_send_slots_enabled = _nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_ASYNC_SEND_SLOTS", False
        )
        self._xfer_dlist_cache: dict[
            tuple[int, int, int, int, int], tuple[Any, Any]
        ] = {}
        self._send_slot_leases: dict[int, str] = {}
        self._send_slot_by_msg: dict[str, int] = {}
        self._send_slot_wait_seconds = _nixl_mailbox_env_float(
            "PAP_NIXL_MAILBOX_SEND_SLOT_WAIT_SECONDS", 30.0
        )
        self._recv_slot_leases: dict[int, str] = {}
        self._recv_slot_wait_seconds = _nixl_mailbox_env_float(
            "PAP_NIXL_MAILBOX_RECV_SLOT_WAIT_SECONDS", 30.0
        )

        from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

        wrapper_type = NixlWrapper if wrapper_cls is None else wrapper_cls
        if wrapper_type is None:
            raise RuntimeError("NIXL is not available")
        config = _nixl_mailbox_agent_config(nixl_agent_config)
        self._wrapper = wrapper_type(
            f"pap-{self.actor_id}-{uuid.uuid4().hex[:8]}",
            config,
        )
        self._peer_agent_name: str | None = None

        with torch.inference_mode(False), torch.device(self.device):
            self._send_buffer = torch.empty(
                self.buffer_bytes,
                dtype=torch.uint8,
                device=self.device,
            )
            self._recv_buffer = torch.empty(
                self.buffer_bytes,
                dtype=torch.uint8,
                device=self.device,
            )
        self._register_buffers()

    @property
    def local_agent_metadata(self) -> bytes:
        agent_metadata = self._wrapper.get_agent_metadata()
        if not getattr(self, "_slot_protocol_enabled", False):
            return agent_metadata
        return _encode_nixl_mailbox_agent_metadata(
            agent_metadata=agent_metadata,
            send_buffer_addr=self._send_buffer.data_ptr(),
            device_id=self.device_id,
            slot_count=self._slot_count,
            slot_bytes=self._slot_bytes,
        )

    @property
    def output_pool_size(self) -> int:
        with self._lock:
            return len(self._output_pool)

    def _register_buffers(self) -> None:
        descs = self._wrapper.get_reg_descs(
            [
                (
                    self._send_buffer.data_ptr(),
                    self._send_buffer.nbytes,
                    self.device_id,
                    "",
                ),
                (
                    self._recv_buffer.data_ptr(),
                    self._recv_buffer.nbytes,
                    self.device_id,
                    "",
                ),
            ],
            self.memory_type,
        )
        self._wrapper.register_memory(descs, self.memory_type, backends=["UCX"])

    def bind_peer(self, peer_agent_metadata: bytes) -> None:
        metadata = _decode_nixl_mailbox_agent_metadata(peer_agent_metadata)
        self._peer_agent_name = _agent_name(
            self._wrapper.add_remote_agent(metadata.agent_metadata)
        )
        if metadata.send_buffer_addr:
            self._peer_send_buffer_addr = metadata.send_buffer_addr
            self._peer_device_id = metadata.device_id
            self._peer_slot_count = metadata.slot_count
            self._peer_slot_bytes = metadata.slot_bytes

    def start(self) -> None:
        if self._peer_agent_name is None:
            raise RuntimeError("PAP NIXL mailbox endpoint must bind peer first")
        if not self._inline_publish_enabled and self._sender_thread is None:
            self._sender_thread = Thread(
                target=self._sender_loop,
                name=f"pap-nixl-mailbox-send-{self.actor_id}",
                daemon=True,
            )
            self._sender_thread.start()
        if self._receiver_thread is None:
            self._receiver_thread = Thread(
                target=self._receiver_loop,
                name=f"pap-nixl-mailbox-recv-{self.actor_id}",
                daemon=True,
            )
            self._receiver_thread.start()

    def close(self) -> None:
        self._closed.set()
        for thread in (self._sender_thread, self._receiver_thread):
            if thread is not None:
                thread.join(timeout=2)
        self._release_cached_xfer_dlists()

    def _release_cached_xfer_dlists(self) -> None:
        cache = getattr(self, "_xfer_dlist_cache", None)
        if not cache:
            return
        handles = list(cache.values())
        cache.clear()
        for local_h, remote_h in handles:
            self._wrapper.release_dlist_handle(local_h)
            self._wrapper.release_dlist_handle(remote_h)

    def _ensure_send_slot_state(
        self,
    ) -> tuple[Condition, dict[int, str], dict[str, int]]:
        cv = getattr(self, "_cv", None)
        if cv is None:
            lock = getattr(self, "_lock", None)
            if lock is None:
                lock = RLock()
                self._lock = lock
            cv = Condition(lock)
            self._cv = cv
        leases = getattr(self, "_send_slot_leases", None)
        if leases is None:
            leases = {}
            self._send_slot_leases = leases
        by_msg = getattr(self, "_send_slot_by_msg", None)
        if by_msg is None:
            by_msg = {}
            self._send_slot_by_msg = by_msg
        return cv, leases, by_msg

    def _use_send_slot_leases(self) -> bool:
        return bool(
            getattr(self, "_slot_protocol_enabled", False)
            and (
                int(getattr(self, "_slot_count", 1)) > 1
                or getattr(self, "_async_send_slots_enabled", False)
                or getattr(self, "_piggyback_acks_enabled", False)
            )
        )

    def _reserve_send_slot(self, msg_id: str) -> tuple[int, int]:
        wait_seconds = float(getattr(self, "_send_slot_wait_seconds", 30.0))
        deadline = time.monotonic() + wait_seconds
        cv, leases, by_msg = self._ensure_send_slot_state()
        with cv:
            while True:
                start_slot = int(getattr(self, "_next_send_slot", 0))
                for offset in range(self._slot_count):
                    slot_id = (start_slot + offset) % self._slot_count
                    if slot_id not in leases:
                        leases[slot_id] = str(msg_id)
                        by_msg[str(msg_id)] = slot_id
                        self._next_send_slot = (slot_id + 1) % self._slot_count
                        return slot_id, slot_id * self._slot_bytes
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out waiting for PAP NIXL send slot for message {msg_id}"
                    )
                cv.wait(timeout=min(remaining, 0.001))

    def _release_send_slot_for_msg(self, msg_id: str) -> None:
        cv, leases, by_msg = self._ensure_send_slot_state()
        with cv:
            slot_id = by_msg.pop(str(msg_id), None)
            if slot_id is not None and leases.get(slot_id) == str(msg_id):
                del leases[slot_id]
            cv.notify_all()

    def _ack_output(self, msg_id: str) -> None:
        msg_id = str(msg_id)
        with self._cv:
            self._acked.discard(msg_id)
            self._output_pool.pop(msg_id, None)
            self._send_enqueued_at.pop(msg_id, None)
            self._release_send_slot_for_msg(msg_id)
            self._cv.notify_all()

    def _defer_ack(self, msg_id: str) -> None:
        with self._cv:
            self._pending_acks[str(msg_id)] = None
            self._cv.notify_all()

    def _drain_pending_acks(self) -> list[str]:
        if not getattr(self, "_piggyback_acks_enabled", False):
            return []
        with self._cv:
            pending = list(self._pending_acks)
            self._pending_acks.clear()
            return pending

    def _restore_pending_acks(self, msg_ids: list[str]) -> None:
        if not msg_ids:
            return
        with self._cv:
            restored: OrderedDict[str, None] = OrderedDict(
                (str(msg_id), None) for msg_id in msg_ids
            )
            restored.update(self._pending_acks)
            self._pending_acks = restored
            self._cv.notify_all()

    def _send_ack_notification(self, msg_id: str) -> None:
        if self._peer_agent_name is None:
            raise RuntimeError("PAP NIXL mailbox endpoint has no peer")
        ack = {"type": "ack", "msg_id": str(msg_id)}
        self._wrapper.send_notif(
            self._peer_agent_name,
            notif_msg=_encode_nixl_mailbox_notification(
                ack,
                use_msgpack=getattr(self, "_msgpack_notifications_enabled", False),
            ),
        )

    def send(self, message: PAPMailboxMessage) -> None:
        if self._inline_publish_enabled:
            self._send_inline(message)
            return
        with self._lock:
            if message.msg_id in self._output_pool:
                raise ValueError(f"duplicate output message id: {message.msg_id}")
            self._output_pool[message.msg_id] = message
            if self._trace_enabled:
                self._send_enqueued_at[message.msg_id] = time.perf_counter()
        self._send_queue.put(message)

    def _send_inline(self, message: PAPMailboxMessage) -> None:
        trace_enabled = getattr(self, "_trace_enabled", False)
        send_start = time.perf_counter() if trace_enabled else 0.0
        with self._lock:
            if message.msg_id in self._output_pool:
                raise ValueError(f"duplicate output message id: {message.msg_id}")
            pending_msg_ids = tuple(self._output_pool)
        for pending_msg_id in pending_msg_ids:
            self._wait_ack(pending_msg_id)
        with self._lock:
            if message.msg_id in self._output_pool:
                raise ValueError(f"duplicate output message id: {message.msg_id}")
            self._output_pool[message.msg_id] = message
            if trace_enabled:
                self._send_enqueued_at[message.msg_id] = send_start
        publish_stats = self._publish_message(message)
        if trace_enabled:
            now = time.perf_counter()
            logger.info(
                "PAP NIXL mailbox inline send trace actor=%s msg_id=%s "
                "kind=%s nbytes=%d publish_ms=%.3f pack_ms=%.3f "
                "copy_ms=%.3f notify_ms=%.3f total_ms=%.3f",
                self.actor_id,
                message.msg_id,
                message.kind,
                publish_stats["nbytes"],
                publish_stats["publish_ms"],
                publish_stats["pack_ms"],
                publish_stats["copy_ms"],
                publish_stats["notify_ms"],
                (now - send_start) * 1000.0,
            )

    def recv(self, msg_id: str | None = None, *, timeout: float | None = None):
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        trace_enabled = getattr(self, "_trace_enabled", False)
        recv_start = time.perf_counter() if trace_enabled else 0.0

        def trace_recv_wait(message: PAPMailboxMessage) -> None:
            if not trace_enabled:
                return
            logger.info(
                "PAP NIXL mailbox recv wait trace actor=%s msg_id=%s "
                "kind=%s requested_msg_id=%s wait_ms=%.3f",
                self.actor_id,
                message.msg_id,
                message.kind,
                "" if msg_id is None else str(msg_id),
                (time.perf_counter() - recv_start) * 1000.0,
            )

        while True:
            with self._cv:
                if msg_id is None and self._incoming:
                    _, message = self._incoming.popitem(last=False)
                    trace_recv_wait(message)
                    return message
                if msg_id is not None and msg_id in self._incoming:
                    message = self._incoming.pop(msg_id)
                    trace_recv_wait(message)
                    return message
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        f"timed out waiting for PAP NIXL mailbox message {msg_id}"
                    )
                if not self._inline_poll_enabled:
                    self._cv.wait(timeout=remaining)
                    continue
            if self._poll_notifications():
                continue
            with self._cv:
                wait_timeout = self._inline_wait_timeout(remaining)
                self._cv.wait(timeout=wait_timeout)

    def _sender_loop(self) -> None:
        while not self._closed.is_set():
            try:
                message = self._send_queue.get(timeout=0.01)
            except Empty:
                continue
            trace_enabled = self._trace_enabled
            send_start = time.perf_counter() if trace_enabled else 0.0
            with self._lock:
                enqueued_at = self._send_enqueued_at.get(message.msg_id, send_start)
            try:
                publish_stats = self._publish_message(message)
                ack_start = time.perf_counter() if trace_enabled else 0.0
                waits_for_ack = not (
                    getattr(self, "_slot_protocol_enabled", False)
                    and (
                        getattr(self, "_async_send_slots_enabled", False)
                        or getattr(self, "_piggyback_acks_enabled", False)
                    )
                )
                if waits_for_ack:
                    self._wait_ack(message.msg_id)
                if trace_enabled:
                    now = time.perf_counter()
                    ack_wait_ms = (now - ack_start) * 1000.0 if waits_for_ack else 0.0
                    logger.info(
                        "PAP NIXL mailbox send trace actor=%s msg_id=%s kind=%s "
                        "nbytes=%d queue_ms=%.3f publish_ms=%.3f "
                        "pack_ms=%.3f copy_ms=%.3f notify_ms=%.3f "
                        "ack_wait_ms=%.3f "
                        "total_ms=%.3f",
                        self.actor_id,
                        message.msg_id,
                        message.kind,
                        publish_stats["nbytes"],
                        (send_start - enqueued_at) * 1000.0,
                        publish_stats["publish_ms"],
                        publish_stats["pack_ms"],
                        publish_stats["copy_ms"],
                        publish_stats["notify_ms"],
                        ack_wait_ms,
                        (now - enqueued_at) * 1000.0,
                    )
            finally:
                self._send_queue.task_done()

    def _publish_message(self, message: PAPMailboxMessage) -> dict[str, float | int]:
        if self._peer_agent_name is None:
            raise RuntimeError("PAP NIXL mailbox endpoint has no peer")
        trace_enabled = self._trace_enabled
        pack_start = time.perf_counter() if trace_enabled else 0.0
        payload_segments = getattr(message, "payload_segments", None)
        if payload_segments is not None:
            segments = tuple(
                segment.detach().contiguous().to(device=self.device)
                for segment in payload_segments
            )
            dtype = segments[0].dtype
            if any(segment.dtype != dtype for segment in segments):
                raise RuntimeError("PAP NIXL mailbox payload segments must share dtype")
            raw_segments = tuple(
                segment.reshape(-1).view(torch.uint8) for segment in segments
            )
            shape = tuple(int(dim) for dim in message.payload_shape or ())
            if not shape:
                raise RuntimeError("PAP NIXL mailbox segmented payload requires shape")
            nbytes = sum(int(raw_segment.numel()) for raw_segment in raw_segments)
        else:
            tensor = message.tensor.detach().contiguous().to(device=self.device)
            dtype = tensor.dtype
            shape = tuple(int(dim) for dim in tensor.shape)
            raw_segments = (tensor.reshape(-1).view(torch.uint8),)
            nbytes = int(raw_segments[0].numel())
        pack_ms = (time.perf_counter() - pack_start) * 1000.0 if trace_enabled else 0.0
        slot_protocol_enabled = getattr(self, "_slot_protocol_enabled", False)
        slot_capacity = self._slot_bytes if slot_protocol_enabled else self.buffer_bytes
        if nbytes > slot_capacity:
            raise RuntimeError(
                f"PAP NIXL mailbox message {message.msg_id} requires {nbytes} "
                f"bytes, slot has {slot_capacity}"
            )
        use_send_slot_leases = self._use_send_slot_leases()
        if use_send_slot_leases:
            slot_id, slot_offset = self._reserve_send_slot(message.msg_id)
        elif slot_protocol_enabled:
            slot_id = int(getattr(self, "_next_send_slot", 0))
            self._next_send_slot = (slot_id + 1) % self._slot_count
            slot_offset = slot_id * self._slot_bytes
        else:
            slot_id = 0
            slot_offset = 0
        piggybacked_acks: list[str] = []
        try:
            copy_start = time.perf_counter() if trace_enabled else 0.0
            with torch.inference_mode(False):
                write_offset = slot_offset
                for raw_segment in raw_segments:
                    segment_nbytes = int(raw_segment.numel())
                    self._send_buffer[
                        write_offset : write_offset + segment_nbytes
                    ].copy_(raw_segment, non_blocking=True)
                    write_offset += segment_nbytes
                self._synchronize_device_stream()
            copy_ms = (
                (time.perf_counter() - copy_start) * 1000.0 if trace_enabled else 0.0
            )
            payload = {
                "type": "message",
                "msg_id": message.msg_id,
                "kind": message.kind,
                "metadata": message.metadata,
                "shape": list(shape),
                "dtype": _dtype_name(dtype),
                "nbytes": nbytes,
            }
            if slot_protocol_enabled:
                payload["slot_id"] = slot_id
            else:
                payload["addr"] = int(self._send_buffer.data_ptr())
                payload["device_id"] = self.device_id
            piggybacked_acks = self._drain_pending_acks()
            if piggybacked_acks:
                payload["acks"] = piggybacked_acks
            notify_start = time.perf_counter() if trace_enabled else 0.0
            self._wrapper.send_notif(
                self._peer_agent_name,
                notif_msg=_encode_nixl_mailbox_notification(
                    payload,
                    use_msgpack=getattr(self, "_msgpack_notifications_enabled", False),
                ),
            )
            notify_ms = (
                (time.perf_counter() - notify_start) * 1000.0 if trace_enabled else 0.0
            )
        except Exception:
            self._restore_pending_acks(piggybacked_acks)
            if use_send_slot_leases:
                self._release_send_slot_for_msg(message.msg_id)
            raise
        return {
            "nbytes": nbytes,
            "pack_ms": pack_ms,
            "copy_ms": copy_ms,
            "notify_ms": notify_ms,
            "publish_ms": pack_ms + copy_ms + notify_ms,
        }

    def _synchronize_device_stream(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def _wait_ack(self, msg_id: str) -> None:
        deadline = time.monotonic() + 30.0
        while True:
            with self._cv:
                if msg_id not in self._output_pool:
                    self._send_enqueued_at.pop(msg_id, None)
                    self._acked.discard(msg_id)
                    return
                if msg_id in self._acked:
                    self._ack_output(msg_id)
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out waiting for PAP NIXL mailbox ACK {msg_id}"
                    )
                if not self._inline_poll_enabled:
                    self._cv.wait(timeout=min(remaining, 0.01))
                    continue
            if self._poll_notifications():
                continue
            with self._cv:
                self._cv.wait(timeout=self._inline_wait_timeout(remaining))

    def _inline_wait_timeout(self, remaining: float | None) -> float | None:
        wait_timeout = (
            self._poll_sleep_seconds if self._poll_sleep_seconds > 0 else 0.00001
        )
        if remaining is None:
            return wait_timeout
        return min(remaining, wait_timeout)

    def _receiver_loop(self) -> None:
        while not self._closed.is_set():
            did_work = self._poll_notifications()
            if not did_work and self._poll_sleep_seconds > 0:
                time.sleep(self._poll_sleep_seconds)

    def _poll_notifications(self) -> bool:
        poll_lock = getattr(self, "_poll_lock", None)
        if poll_lock is None:
            return self._poll_notifications_unlocked()
        with poll_lock:
            return self._poll_notifications_unlocked()

    def _poll_notifications_unlocked(self) -> bool:
        notifications = self._wrapper.get_new_notifs()
        did_work = False
        for _agent, payloads in notifications.items():
            for payload in payloads:
                did_work = True
                self._handle_notification(payload)
        return did_work

    def _handle_notification(self, payload: bytes) -> None:
        data = _decode_nixl_mailbox_notification(payload)
        for ack_msg_id in data.get("acks") or ():
            self._ack_output(str(ack_msg_id))
        message_type = str(data.get("type"))
        if message_type == "ack":
            self._ack_output(str(data["msg_id"]))
            return
        if message_type != "message":
            raise RuntimeError(f"unknown PAP NIXL mailbox notification: {data}")
        message = self._read_remote_message(data)
        with self._cv:
            self._incoming[message.msg_id] = message
            self._cv.notify_all()
        if getattr(self, "_slot_protocol_enabled", False) and getattr(
            self, "_piggyback_acks_enabled", False
        ):
            self._defer_ack(message.msg_id)
        else:
            self._send_ack_notification(message.msg_id)

    def _remote_payload_location(
        self, data: dict[str, Any], nbytes: int
    ) -> tuple[int, int]:
        if "slot_id" not in data:
            return int(data["addr"]), int(data["device_id"])
        if self._peer_send_buffer_addr is None or self._peer_device_id is None:
            raise RuntimeError(
                "PAP NIXL mailbox slot message received before peer slot metadata"
            )
        slot_id = int(data["slot_id"])
        if slot_id < 0 or slot_id >= self._peer_slot_count:
            raise RuntimeError(f"PAP NIXL mailbox invalid peer slot id: {slot_id}")
        if nbytes > self._peer_slot_bytes:
            raise RuntimeError(
                f"PAP NIXL mailbox incoming message {data['msg_id']} requires "
                f"{nbytes} bytes, peer slot has {self._peer_slot_bytes}"
            )
        return (
            self._peer_send_buffer_addr + slot_id * self._peer_slot_bytes,
            self._peer_device_id,
        )

    def _ensure_recv_slot_state(self) -> tuple[Condition, dict[int, str]]:
        cv = getattr(self, "_cv", None)
        if cv is None:
            lock = getattr(self, "_lock", None)
            if lock is None:
                lock = RLock()
                self._lock = lock
            cv = Condition(lock)
            self._cv = cv
        leases = getattr(self, "_recv_slot_leases", None)
        if leases is None:
            leases = {}
            self._recv_slot_leases = leases
        return cv, leases

    def _recv_slot_candidates(self, data: dict[str, Any]) -> list[int]:
        recv_slot_count = int(getattr(self, "_recv_slot_count", 1))
        if recv_slot_count <= 0:
            raise RuntimeError("PAP NIXL mailbox receive slot count must be positive")
        preferred = int(data.get("slot_id", 0)) % recv_slot_count
        return [
            (preferred + offset) % recv_slot_count for offset in range(recv_slot_count)
        ]

    def _recv_slot_location(self, recv_slot_id: int) -> tuple[int, int]:
        recv_slot_bytes = int(getattr(self, "_recv_slot_bytes", self.buffer_bytes))
        recv_offset = int(recv_slot_id) * recv_slot_bytes
        return recv_offset, int(self._recv_buffer.data_ptr()) + recv_offset

    def _reserve_local_recv_location(
        self, data: dict[str, Any], nbytes: int
    ) -> tuple[int, int, int, bool]:
        recv_slot_bytes = int(getattr(self, "_recv_slot_bytes", self.buffer_bytes))
        if nbytes > recv_slot_bytes:
            raise RuntimeError(
                f"PAP NIXL mailbox incoming message {data['msg_id']} requires "
                f"{nbytes} bytes, receive slot has {recv_slot_bytes}"
            )
        candidates = self._recv_slot_candidates(data)
        if not getattr(self, "_zero_copy_recv_enabled", False):
            recv_slot_id = candidates[0]
            recv_offset, local_addr = self._recv_slot_location(recv_slot_id)
            return recv_slot_id, recv_offset, local_addr, False

        msg_id = str(data["msg_id"])
        wait_seconds = float(getattr(self, "_recv_slot_wait_seconds", 30.0))
        deadline = time.monotonic() + wait_seconds
        cv, leases = self._ensure_recv_slot_state()
        with cv:
            while True:
                for recv_slot_id in candidates:
                    if recv_slot_id not in leases:
                        leases[recv_slot_id] = msg_id
                        recv_offset, local_addr = self._recv_slot_location(recv_slot_id)
                        return recv_slot_id, recv_offset, local_addr, True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out waiting for PAP NIXL receive slot "
                        f"for message {msg_id}"
                    )
                cv.wait(timeout=min(remaining, 0.001))

    def _release_recv_slot(self, recv_slot_id: int, msg_id: str) -> None:
        cv, leases = self._ensure_recv_slot_state()
        with cv:
            if leases.get(int(recv_slot_id)) == str(msg_id):
                del leases[int(recv_slot_id)]
                cv.notify_all()

    def _get_read_dlist_handles(
        self,
        *,
        local_addr: int,
        nbytes: int,
        remote_addr: int,
        remote_device_id: int,
    ) -> tuple[Any, Any, bool]:
        cache_key = (
            local_addr,
            int(nbytes),
            int(self.device_id),
            int(remote_addr),
            int(remote_device_id),
        )
        cache_enabled = getattr(self, "_cache_xfer_dlists_enabled", False)
        if cache_enabled:
            with self._lock:
                cached = self._xfer_dlist_cache.get(cache_key)
            if cached is not None:
                return cached[0], cached[1], False
        local_desc = self._wrapper.get_xfer_descs(
            [(local_addr, nbytes, self.device_id)],
            self.memory_type,
        )
        remote_desc = self._wrapper.get_xfer_descs(
            [(remote_addr, nbytes, remote_device_id)],
            self.memory_type,
        )
        local_h = self._wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", local_desc)
        remote_h = self._wrapper.prep_xfer_dlist(self._peer_agent_name, remote_desc)
        if cache_enabled:
            with self._lock:
                cached = self._xfer_dlist_cache.setdefault(
                    cache_key, (local_h, remote_h)
                )
            if cached != (local_h, remote_h):
                self._wrapper.release_dlist_handle(local_h)
                self._wrapper.release_dlist_handle(remote_h)
            return cached[0], cached[1], False
        return local_h, remote_h, True

    def _read_remote_message(self, data: dict[str, Any]) -> PAPMailboxMessage:
        if self._peer_agent_name is None:
            raise RuntimeError("PAP NIXL mailbox endpoint has no peer")
        nbytes = int(data["nbytes"])
        if nbytes > self.buffer_bytes:
            raise RuntimeError(
                f"PAP NIXL mailbox incoming message {data['msg_id']} requires "
                f"{nbytes} bytes, buffer has {self.buffer_bytes}"
            )
        trace_enabled = self._trace_enabled
        prepare_start = time.perf_counter() if trace_enabled else 0.0
        msg_id = str(data["msg_id"])
        remote_addr, remote_device_id = self._remote_payload_location(data, nbytes)
        recv_slot_id, recv_offset, local_addr, release_recv_slot = (
            self._reserve_local_recv_location(data, nbytes)
        )
        try:
            local_h, remote_h, release_dlists = self._get_read_dlist_handles(
                local_addr=local_addr,
                nbytes=nbytes,
                remote_addr=remote_addr,
                remote_device_id=remote_device_id,
            )
        except Exception:
            if release_recv_slot:
                self._release_recv_slot(recv_slot_id, msg_id)
            raise
        xfer_h = self._wrapper.make_prepped_xfer(
            "READ",
            local_h,
            [0],
            remote_h,
            [0],
        )
        prepare_ms = (
            (time.perf_counter() - prepare_start) * 1000.0 if trace_enabled else 0.0
        )
        transfer_start = time.perf_counter() if trace_enabled else 0.0
        transfer_polls = 0
        try:
            self._wrapper.transfer(xfer_h)
            while True:
                state = self._wrapper.check_xfer_state(xfer_h)
                if state == "DONE":
                    break
                if state != "PROC":
                    raise RuntimeError(f"PAP NIXL transfer failed: {state}")
                transfer_polls += 1
                if self._xfer_poll_sleep_seconds > 0:
                    time.sleep(self._xfer_poll_sleep_seconds)
        except Exception:
            if release_recv_slot:
                self._release_recv_slot(recv_slot_id, msg_id)
            raise
        finally:
            transfer_ms = (
                (time.perf_counter() - transfer_start) * 1000.0
                if trace_enabled
                else 0.0
            )
            self._wrapper.release_xfer_handle(xfer_h)
            if release_dlists:
                self._wrapper.release_dlist_handle(local_h)
                self._wrapper.release_dlist_handle(remote_h)
        dtype = _dtype_from_name(str(data["dtype"]))
        shape = tuple(int(dim) for dim in data["shape"])
        materialize_start = time.perf_counter() if trace_enabled else 0.0
        recv_view = self._recv_buffer[recv_offset : recv_offset + nbytes]
        if self._zero_copy_recv_enabled:
            tensor = recv_view.view(dtype).reshape(shape)
        else:
            with torch.inference_mode(False):
                tensor = torch.empty(shape, dtype=dtype, device=self.device)
                tensor.reshape(-1).view(torch.uint8).copy_(
                    recv_view,
                    non_blocking=True,
                )
                self._synchronize_device_stream()
        materialize_ms = (
            (time.perf_counter() - materialize_start) * 1000.0 if trace_enabled else 0.0
        )
        if trace_enabled:
            logger.info(
                "PAP NIXL mailbox read trace actor=%s msg_id=%s kind=%s "
                "nbytes=%d prepare_ms=%.3f transfer_ms=%.3f "
                "transfer_polls=%d materialize_ms=%.3f total_ms=%.3f",
                self.actor_id,
                msg_id,
                str(data["kind"]),
                nbytes,
                prepare_ms,
                transfer_ms,
                transfer_polls,
                materialize_ms,
                prepare_ms + transfer_ms + materialize_ms,
            )
        release_callback = None
        if release_recv_slot:
            release_callback = lambda recv_slot_id=recv_slot_id, msg_id=msg_id: (
                self._release_recv_slot(recv_slot_id, msg_id)
            )
        return PAPMailboxMessage(
            msg_id=msg_id,
            kind=str(data["kind"]),
            metadata=dict(data.get("metadata") or {}),
            tensor=tensor,
            release_callback=release_callback,
        )
