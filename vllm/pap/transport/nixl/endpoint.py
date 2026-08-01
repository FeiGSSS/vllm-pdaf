# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL-backed PAP bidirectional mailbox endpoint.

Backend-neutral mailbox messages live in
``vllm.pap.transport.nixl.message``. This module owns NIXL agent metadata,
registered tensor slots, notifications, and transfer progress.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from math import prod
from queue import Empty, Queue
from threading import Condition, Event, RLock, Thread
from typing import Any

import msgspec
import torch

from vllm.pap.protocol import PAPOffloadExecTransportClosed
from vllm.pap.transport.nixl.message import (
    PAPMailboxMessage,
    _merge_message_recv_trace,
)

__all__ = [
    "PAPMailboxDirectSendPayload",
    "PAPMailboxMessage",
    "PAPNixlMailboxAgentMetadata",
    "PAPNixlMailboxEndpoint",
]

logger = logging.getLogger(__name__)
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_SLOT_BYTE_ALIGNMENT = 16
NIXL_MAILBOX_ZERO_COPY_RECV_DEFAULT = True


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


def _nixl_mailbox_env_set(name: str, default: str) -> set[str]:
    value = os.environ.get(name)
    raw = default if value in (None, "") else value
    return {item.strip() for item in raw.split(",") if item.strip()}


def _aligned_slot_bytes(*, total_bytes: int, slot_count: int) -> int:
    raw_slot_bytes = int(total_bytes) // int(slot_count)
    return raw_slot_bytes - (raw_slot_bytes % _SLOT_BYTE_ALIGNMENT)


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
    recv_buffer_addr: int = 0
    recv_slot_count: int = 1
    recv_slot_bytes: int = 0


@dataclass(frozen=True)
class PAPMailboxDirectSendPayload:
    tensor: torch.Tensor
    slot_id: int | None


def _encode_nixl_mailbox_agent_metadata(
    *,
    agent_metadata: bytes,
    send_buffer_addr: int,
    device_id: int,
    slot_count: int,
    slot_bytes: int,
    recv_buffer_addr: int,
    recv_slot_count: int,
    recv_slot_bytes: int,
) -> bytes:
    payload = {
        "type": "pap_nixl_mailbox_agent",
        "version": 1,
        "agent_metadata_b64": base64.b64encode(agent_metadata).decode("ascii"),
        "send_buffer_addr": int(send_buffer_addr),
        "device_id": int(device_id),
        "slot_count": int(slot_count),
        "slot_bytes": int(slot_bytes),
        "recv_buffer_addr": int(recv_buffer_addr),
        "recv_slot_count": int(recv_slot_count),
        "recv_slot_bytes": int(recv_slot_bytes),
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
        recv_buffer_addr=int(payload.get("recv_buffer_addr") or 0),
        recv_slot_count=int(payload.get("recv_slot_count") or 1),
        recv_slot_bytes=int(payload.get("recv_slot_bytes") or 0),
    )


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
        self._receive_stopped = Event()
        self._close_lock = RLock()
        self._resources_released = False
        self._sender_thread: Thread | None = None
        self._receiver_thread: Thread | None = None
        self._send_queue: Queue[PAPMailboxMessage] = Queue()
        self._incoming: OrderedDict[str, PAPMailboxMessage] = OrderedDict()
        self._output_pool: OrderedDict[str, PAPMailboxMessage] = OrderedDict()
        self._send_enqueued_at: dict[str, float] = {}
        self._acked: set[str] = set()
        self._pending_recv_releases: OrderedDict[str, int] = OrderedDict()
        self._slot_count = _nixl_mailbox_env_int("PAP_NIXL_MAILBOX_SLOT_COUNT", 1)
        if self._slot_count <= 0:
            raise ValueError("PAP_NIXL_MAILBOX_SLOT_COUNT must be positive")
        self._slot_bytes = _aligned_slot_bytes(
            total_bytes=self.buffer_bytes,
            slot_count=self._slot_count,
        )
        if self._slot_bytes <= 0:
            raise ValueError("PAP_NIXL_MAILBOX_SLOT_COUNT is too large")
        self._recv_slot_count = _nixl_mailbox_env_int(
            "PAP_NIXL_MAILBOX_RECV_SLOT_COUNT", self._slot_count
        )
        if self._recv_slot_count <= 0:
            raise ValueError("PAP_NIXL_MAILBOX_RECV_SLOT_COUNT must be positive")
        self._recv_slot_bytes = _aligned_slot_bytes(
            total_bytes=self.buffer_bytes,
            slot_count=self._recv_slot_count,
        )
        if self._recv_slot_bytes <= 0:
            raise ValueError("PAP_NIXL_MAILBOX_RECV_SLOT_COUNT is too large")
        self._next_send_slot = 0
        self._peer_send_buffer_addr: int | None = None
        self._peer_device_id: int | None = None
        self._peer_slot_count = 1
        self._peer_slot_bytes = 0
        self._peer_recv_buffer_addr: int | None = None
        self._peer_recv_device_id: int | None = None
        self._peer_recv_slot_count = 1
        self._peer_recv_slot_bytes = 0
        self._next_peer_recv_slot = 0
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
            "PAP_NIXL_MAILBOX_INLINE_PUBLISH", True
        )
        self._slot_protocol_enabled = _nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_SLOT_PROTOCOL", True
        )
        self._zero_copy_recv_enabled = _nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_ZERO_COPY_RECV",
            NIXL_MAILBOX_ZERO_COPY_RECV_DEFAULT,
        )
        self._cache_xfer_dlists_enabled = _nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_CACHE_XFER_DLISTS", True
        )
        self._cache_xfer_handles_enabled = _nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_CACHE_XFER_HANDLES", False
        )
        self._cache_write_xfer_handles_enabled = _nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_CACHE_WRITE_XFER_HANDLES", True
        )
        self._push_write_kinds = _nixl_mailbox_env_set(
            "PAP_NIXL_MAILBOX_PUSH_WRITE_KINDS",
            "attention_task_batch,attention_result_batch",
        )
        self._piggyback_recv_releases_enabled = _nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_PIGGYBACK_RECV_RELEASES", True
        )
        self._msgpack_notifications_enabled = _nixl_mailbox_env_bool(
            "PAP_NIXL_MAILBOX_MSGPACK_NOTIF", True
        )
        self._xfer_dlist_cache: dict[
            tuple[int, int, int, int, int], tuple[Any, Any]
        ] = {}
        self._xfer_handle_cache: dict[tuple[int, int, int, int, int], Any] = {}
        self._write_xfer_handle_cache: dict[tuple[int, int, int, int, int], Any] = {}
        self._send_slot_leases: dict[int, str] = {}
        self._send_slot_by_msg: dict[str, int] = {}
        self._send_slot_wait_seconds = _nixl_mailbox_env_float(
            "PAP_NIXL_MAILBOX_SEND_SLOT_WAIT_SECONDS", 30.0
        )
        self._recv_slot_leases: dict[int, str] = {}
        self._recv_slot_wait_seconds = _nixl_mailbox_env_float(
            "PAP_NIXL_MAILBOX_RECV_SLOT_WAIT_SECONDS", 30.0
        )
        self._peer_recv_slot_leases: dict[int, str] = {}
        self._peer_recv_slot_by_msg: dict[str, int] = {}

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
        recv_buffer = getattr(self, "_recv_buffer", None)
        return _encode_nixl_mailbox_agent_metadata(
            agent_metadata=agent_metadata,
            send_buffer_addr=self._send_buffer.data_ptr(),
            device_id=self.device_id,
            slot_count=self._slot_count,
            slot_bytes=self._slot_bytes,
            recv_buffer_addr=0 if recv_buffer is None else recv_buffer.data_ptr(),
            recv_slot_count=int(getattr(self, "_recv_slot_count", 1)),
            recv_slot_bytes=int(getattr(self, "_recv_slot_bytes", 0)),
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
        if metadata.recv_buffer_addr:
            self._peer_recv_buffer_addr = metadata.recv_buffer_addr
            self._peer_recv_device_id = metadata.device_id
            self._peer_recv_slot_count = metadata.recv_slot_count
            self._peer_recv_slot_bytes = metadata.recv_slot_bytes

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

    def stop_receiving(self) -> None:
        """Wake external consumers without stopping NIXL progress threads."""
        with self._cv:
            self._receive_stopped.set()
            self._cv.notify_all()

    def close(self) -> None:
        with self._close_lock:
            if self._resources_released:
                return
            self.stop_receiving()
            with self._cv:
                self._closed.set()
                self._cv.notify_all()
            threads = tuple(
                thread
                for thread in (self._sender_thread, self._receiver_thread)
                if thread is not None
            )
            deadline = time.monotonic() + 2.0
            for thread in threads:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            alive = [thread.name for thread in threads if thread.is_alive()]
            if alive:
                raise RuntimeError(
                    "PAP NIXL mailbox threads did not stop: " + ", ".join(alive)
                )
            self._release_cached_xfer_handles()
            self._release_cached_write_xfer_handles()
            self._release_cached_xfer_dlists()
            self._resources_released = True

    def _release_cached_xfer_handles(self) -> None:
        cache = getattr(self, "_xfer_handle_cache", None)
        if not cache:
            return
        handles = list(cache.values())
        cache.clear()
        for handle in handles:
            self._wrapper.release_xfer_handle(handle)

    def _release_cached_write_xfer_handles(self) -> None:
        cache = getattr(self, "_write_xfer_handle_cache", None)
        if not cache:
            return
        handles = list(cache.values())
        cache.clear()
        for handle in handles:
            self._wrapper.release_xfer_handle(handle)

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
            and int(getattr(self, "_slot_count", 1)) > 1
        )

    def _reserve_send_slot(self, msg_id: str) -> tuple[int, int]:
        wait_seconds = float(getattr(self, "_send_slot_wait_seconds", 30.0))
        deadline = time.monotonic() + wait_seconds
        cv, leases, by_msg = self._ensure_send_slot_state()
        with cv:
            while True:
                if self._closed.is_set():
                    raise PAPOffloadExecTransportClosed("PAP NIXL mailbox is closed")
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

    def _ensure_peer_recv_slot_state(
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
        leases = getattr(self, "_peer_recv_slot_leases", None)
        if leases is None:
            leases = {}
            self._peer_recv_slot_leases = leases
        by_msg = getattr(self, "_peer_recv_slot_by_msg", None)
        if by_msg is None:
            by_msg = {}
            self._peer_recv_slot_by_msg = by_msg
        return cv, leases, by_msg

    def _reserve_peer_recv_slot(self, msg_id: str, nbytes: int) -> tuple[int, int]:
        peer_recv_buffer_addr = getattr(self, "_peer_recv_buffer_addr", None)
        peer_recv_device_id = getattr(self, "_peer_recv_device_id", None)
        if peer_recv_buffer_addr is None or peer_recv_device_id is None:
            raise RuntimeError("PAP NIXL mailbox peer has no receive slots")
        peer_recv_slot_bytes = int(getattr(self, "_peer_recv_slot_bytes", 0))
        if nbytes > peer_recv_slot_bytes:
            raise RuntimeError(
                f"PAP NIXL mailbox push message {msg_id} requires {nbytes} "
                f"bytes, peer receive slot has {peer_recv_slot_bytes}"
            )
        peer_recv_slot_count = int(getattr(self, "_peer_recv_slot_count", 1))
        if peer_recv_slot_count <= 0:
            raise RuntimeError("PAP NIXL mailbox peer receive slot count invalid")
        wait_seconds = float(getattr(self, "_recv_slot_wait_seconds", 30.0))
        deadline = time.monotonic() + wait_seconds
        cv, leases, by_msg = self._ensure_peer_recv_slot_state()
        with cv:
            while True:
                if self._closed.is_set():
                    raise PAPOffloadExecTransportClosed("PAP NIXL mailbox is closed")
                start_slot = int(getattr(self, "_next_peer_recv_slot", 0))
                for offset in range(peer_recv_slot_count):
                    slot_id = (start_slot + offset) % peer_recv_slot_count
                    if slot_id not in leases:
                        leases[slot_id] = str(msg_id)
                        by_msg[str(msg_id)] = slot_id
                        self._next_peer_recv_slot = (slot_id + 1) % peer_recv_slot_count
                        return (
                            slot_id,
                            int(peer_recv_buffer_addr) + slot_id * peer_recv_slot_bytes,
                        )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out waiting for PAP NIXL peer receive slot "
                        f"for message {msg_id}"
                    )
                cv.wait(timeout=min(remaining, 0.001))

    def _release_peer_recv_slot_for_msg(self, msg_id: str) -> None:
        cv, leases, by_msg = self._ensure_peer_recv_slot_state()
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

    def _defer_recv_release(self, msg_id: str, recv_slot_id: int) -> None:
        with self._cv:
            self._pending_recv_releases[str(msg_id)] = int(recv_slot_id)
            self._cv.notify_all()

    def _drain_pending_recv_releases(self) -> list[dict[str, int | str]]:
        if not getattr(self, "_piggyback_recv_releases_enabled", False):
            return []
        with self._cv:
            pending = [
                {"msg_id": msg_id, "recv_slot_id": int(recv_slot_id)}
                for msg_id, recv_slot_id in self._pending_recv_releases.items()
            ]
            self._pending_recv_releases.clear()
            return pending

    def _restore_pending_recv_releases(
        self,
        releases: list[dict[str, int | str]],
    ) -> None:
        if not releases:
            return
        with self._cv:
            restored: OrderedDict[str, int] = OrderedDict(
                (str(release["msg_id"]), int(release["recv_slot_id"]))
                for release in releases
            )
            restored.update(self._pending_recv_releases)
            self._pending_recv_releases = restored
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

    def _send_recv_release_notification(self, msg_id: str, recv_slot_id: int) -> None:
        if self._peer_agent_name is None:
            raise RuntimeError("PAP NIXL mailbox endpoint has no peer")
        release = {
            "type": "recv_release",
            "msg_id": str(msg_id),
            "recv_slot_id": int(recv_slot_id),
        }
        self._wrapper.send_notif(
            self._peer_agent_name,
            notif_msg=_encode_nixl_mailbox_notification(
                release,
                use_msgpack=getattr(self, "_msgpack_notifications_enabled", False),
            ),
        )

    def reserve_direct_send_tensor(
        self,
        msg_id: str,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> PAPMailboxDirectSendPayload:
        shape = tuple(int(dim) for dim in shape)
        if not shape:
            raise ValueError("PAP direct send tensor requires a non-empty shape")
        nbytes = int(prod(shape) * torch.empty((), dtype=dtype).element_size())
        slot_protocol_enabled = getattr(self, "_slot_protocol_enabled", False)
        slot_capacity = self._slot_bytes if slot_protocol_enabled else self.buffer_bytes
        if nbytes > slot_capacity:
            raise RuntimeError(
                f"PAP NIXL mailbox direct payload {msg_id} requires {nbytes} "
                f"bytes, slot has {slot_capacity}"
            )
        if slot_protocol_enabled:
            slot_id, slot_offset = self._reserve_send_slot(str(msg_id))
        else:
            slot_id = None
            slot_offset = 0
        tensor = (
            self._send_buffer[slot_offset : slot_offset + nbytes]
            .view(dtype)
            .reshape(shape)
        )
        return PAPMailboxDirectSendPayload(tensor=tensor, slot_id=slot_id)

    def send(self, message: PAPMailboxMessage) -> None:
        if self._closed.is_set():
            raise PAPOffloadExecTransportClosed("PAP NIXL mailbox is closed")
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
        if publish_stats.get("local_complete"):
            self._ack_output(message.msg_id)
        if trace_enabled:
            now = time.perf_counter()
            logger.info(
                "PAP NIXL mailbox inline send trace actor=%s msg_id=%s "
                "kind=%s nbytes=%d publish_ms=%.3f pack_ms=%.3f "
                "slot_wait_ms=%.3f copy_ms=%.3f "
                "payload_ms=%.3f piggyback_ms=%.3f notify_ms=%.3f "
                "write_ms=%.3f write_prepare_ms=%.3f "
                "write_transfer_ms=%.3f write_polls=%d total_ms=%.3f",
                self.actor_id,
                message.msg_id,
                message.kind,
                publish_stats["nbytes"],
                publish_stats["publish_ms"],
                publish_stats["pack_ms"],
                publish_stats["slot_wait_ms"],
                publish_stats["copy_ms"],
                publish_stats["payload_ms"],
                publish_stats["piggyback_ms"],
                publish_stats["notify_ms"],
                publish_stats["write_ms"],
                publish_stats["write_prepare_ms"],
                publish_stats["write_transfer_ms"],
                publish_stats["write_polls"],
                (now - send_start) * 1000.0,
            )

    def recv(self, msg_id: str | None = None, *, timeout: float | None = None):
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        trace_enabled = getattr(self, "_trace_enabled", False)
        recv_start = time.perf_counter() if trace_enabled else 0.0

        def trace_recv_wait(message: PAPMailboxMessage) -> None:
            if not trace_enabled:
                return
            wait_ms = (time.perf_counter() - recv_start) * 1000.0
            _merge_message_recv_trace(message, {"wait_ms": wait_ms})
            logger.info(
                "PAP NIXL mailbox recv wait trace actor=%s msg_id=%s "
                "kind=%s requested_msg_id=%s wait_ms=%.3f",
                self.actor_id,
                message.msg_id,
                message.kind,
                "" if msg_id is None else str(msg_id),
                wait_ms,
            )

        while True:
            with self._cv:
                if self._receive_stopped.is_set():
                    raise PAPOffloadExecTransportClosed(
                        "PAP NIXL mailbox receive loop stopped"
                    )
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
                local_complete = bool(publish_stats.get("local_complete"))
                if local_complete:
                    self._ack_output(message.msg_id)
                waits_for_ack = not local_complete
                if waits_for_ack:
                    self._wait_ack(message.msg_id)
                if trace_enabled:
                    now = time.perf_counter()
                    ack_wait_ms = (now - ack_start) * 1000.0 if waits_for_ack else 0.0
                    logger.info(
                        "PAP NIXL mailbox send trace actor=%s msg_id=%s kind=%s "
                        "nbytes=%d queue_ms=%.3f publish_ms=%.3f "
                        "pack_ms=%.3f slot_wait_ms=%.3f copy_ms=%.3f "
                        "payload_ms=%.3f piggyback_ms=%.3f notify_ms=%.3f "
                        "write_ms=%.3f write_prepare_ms=%.3f "
                        "write_transfer_ms=%.3f write_polls=%d ack_wait_ms=%.3f "
                        "total_ms=%.3f",
                        self.actor_id,
                        message.msg_id,
                        message.kind,
                        publish_stats["nbytes"],
                        (send_start - enqueued_at) * 1000.0,
                        publish_stats["publish_ms"],
                        publish_stats["pack_ms"],
                        publish_stats["slot_wait_ms"],
                        publish_stats["copy_ms"],
                        publish_stats["payload_ms"],
                        publish_stats["piggyback_ms"],
                        publish_stats["notify_ms"],
                        publish_stats["write_ms"],
                        publish_stats["write_prepare_ms"],
                        publish_stats["write_transfer_ms"],
                        publish_stats["write_polls"],
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
        direct_payload = bool(getattr(message, "direct_payload", False))
        direct_payload_slot_id = getattr(message, "payload_slot_id", None)
        if direct_payload:
            tensor = message.tensor.detach()
            if not tensor.is_contiguous():
                raise RuntimeError("PAP NIXL mailbox direct payload must be contiguous")
            ready_event = getattr(message, "payload_ready_event", None)
            if ready_event is not None:
                ready_event.synchronize()
            dtype = tensor.dtype
            shape = tuple(int(dim) for dim in message.payload_shape or tensor.shape)
            if tuple(tensor.shape) != shape:
                raise RuntimeError("PAP NIXL mailbox direct payload shape mismatch")
            nbytes = int(tensor.numel() * tensor.element_size())
            if getattr(self, "_slot_protocol_enabled", False):
                if direct_payload_slot_id is None:
                    raise RuntimeError(
                        "PAP NIXL mailbox direct payload requires a send slot"
                    )
                slot_start = int(direct_payload_slot_id) * self._slot_bytes
                slot_end = slot_start + self._slot_bytes
                buffer_start = int(self._send_buffer.data_ptr())
                payload_start = int(tensor.data_ptr())
                payload_end = payload_start + nbytes
                expected_start = buffer_start + slot_start
                expected_end = buffer_start + slot_end
                if payload_start < expected_start or payload_end > expected_end:
                    raise RuntimeError(
                        "PAP NIXL mailbox direct payload is outside send slot"
                    )
            raw_segments = ()
        else:
            tensor = message.tensor.detach().contiguous().to(device=self.device)
            dtype = tensor.dtype
            shape = tuple(int(dim) for dim in tensor.shape)
            raw_segments = (tensor.reshape(-1).view(torch.uint8),)
            nbytes = int(raw_segments[0].numel())
        pack_ms = (time.perf_counter() - pack_start) * 1000.0 if trace_enabled else 0.0
        slot_protocol_enabled = getattr(self, "_slot_protocol_enabled", False)
        slot_capacity = self._slot_bytes if slot_protocol_enabled else self.buffer_bytes
        if nbytes > slot_capacity and not direct_payload:
            raise RuntimeError(
                f"PAP NIXL mailbox message {message.msg_id} requires {nbytes} "
                f"bytes, slot has {slot_capacity}"
            )
        if nbytes > self.buffer_bytes:
            raise RuntimeError(
                f"PAP NIXL mailbox message {message.msg_id} requires {nbytes} "
                f"bytes, buffer has {self.buffer_bytes}"
            )
        push_write = self._should_push_write_message(message)
        use_send_slot_leases = (not direct_payload) and self._use_send_slot_leases()
        slot_wait_start = time.perf_counter() if trace_enabled else 0.0
        if use_send_slot_leases:
            slot_id, slot_offset = self._reserve_send_slot(message.msg_id)
        elif slot_protocol_enabled and not direct_payload:
            slot_id = int(getattr(self, "_next_send_slot", 0))
            self._next_send_slot = (slot_id + 1) % self._slot_count
            slot_offset = slot_id * self._slot_bytes
        else:
            slot_id = 0
            slot_offset = 0
        slot_wait_ms = (
            (time.perf_counter() - slot_wait_start) * 1000.0 if trace_enabled else 0.0
        )
        piggybacked_recv_releases: list[dict[str, int | str]] = []
        pushed_recv_slot_id: int | None = None
        write_ms = 0.0
        write_prepare_ms = 0.0
        write_transfer_ms = 0.0
        write_polls = 0
        local_complete = False
        try:
            copy_start = time.perf_counter() if trace_enabled else 0.0
            if not direct_payload:
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
            if direct_payload:
                source_addr = int(tensor.data_ptr())
            else:
                source_addr = int(self._send_buffer.data_ptr()) + int(slot_offset)
            if push_write:
                pushed_recv_slot_id, write_stats = (
                    self._write_payload_to_peer_recv_slot(
                        msg_id=message.msg_id,
                        source_addr=source_addr,
                        nbytes=nbytes,
                    )
                )
                write_ms = float(write_stats["write_ms"])
                write_prepare_ms = float(write_stats["write_prepare_ms"])
                write_transfer_ms = float(write_stats["write_transfer_ms"])
                write_polls = int(write_stats["write_polls"])
                local_complete = True
            payload_start = time.perf_counter() if trace_enabled else 0.0
            payload = {
                "type": "message",
                "msg_id": message.msg_id,
                "kind": message.kind,
                "metadata": message.metadata,
                "shape": list(shape),
                "dtype": _dtype_name(dtype),
                "nbytes": nbytes,
            }
            if pushed_recv_slot_id is not None:
                payload["materialized_recv_slot_id"] = int(pushed_recv_slot_id)
            elif direct_payload and direct_payload_slot_id is not None:
                payload["slot_id"] = int(direct_payload_slot_id)
                payload["direct_payload"] = True
            elif direct_payload:
                payload["addr"] = int(tensor.data_ptr())
                payload["device_id"] = (
                    int(tensor.device.index or 0) if tensor.device.type == "cuda" else 0
                )
                payload["direct_payload"] = True
            elif slot_protocol_enabled:
                payload["slot_id"] = slot_id
            else:
                payload["addr"] = int(self._send_buffer.data_ptr())
                payload["device_id"] = self.device_id
            payload_ms = (
                (time.perf_counter() - payload_start) * 1000.0 if trace_enabled else 0.0
            )
            piggyback_start = time.perf_counter() if trace_enabled else 0.0
            piggybacked_recv_releases = self._drain_pending_recv_releases()
            if piggybacked_recv_releases:
                payload["recv_releases"] = piggybacked_recv_releases
            piggyback_ms = (
                (time.perf_counter() - piggyback_start) * 1000.0
                if trace_enabled
                else 0.0
            )
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
            self._restore_pending_recv_releases(piggybacked_recv_releases)
            if pushed_recv_slot_id is not None:
                self._release_peer_recv_slot_for_msg(message.msg_id)
            if (
                direct_payload and direct_payload_slot_id is not None
            ) or use_send_slot_leases:
                self._release_send_slot_for_msg(message.msg_id)
            raise
        return {
            "nbytes": nbytes,
            "pack_ms": pack_ms,
            "slot_wait_ms": slot_wait_ms,
            "copy_ms": copy_ms,
            "payload_ms": payload_ms,
            "piggyback_ms": piggyback_ms,
            "notify_ms": notify_ms,
            "write_ms": write_ms,
            "write_prepare_ms": write_prepare_ms,
            "write_transfer_ms": write_transfer_ms,
            "write_polls": write_polls,
            "publish_ms": (
                pack_ms
                + slot_wait_ms
                + copy_ms
                + write_ms
                + payload_ms
                + piggyback_ms
                + notify_ms
            ),
            "local_complete": int(local_complete),
        }

    def _synchronize_device_stream(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def _wait_ack(self, msg_id: str) -> None:
        deadline = time.monotonic() + 30.0
        while True:
            with self._cv:
                if self._closed.is_set():
                    raise PAPOffloadExecTransportClosed("PAP NIXL mailbox is closed")
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
                self._closed.wait(self._poll_sleep_seconds)

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
        for release in data.get("recv_releases") or ():
            self._release_peer_recv_slot_for_msg(str(release["msg_id"]))
        message_type = str(data.get("type"))
        if message_type == "ack":
            self._ack_output(str(data["msg_id"]))
            return
        if message_type == "recv_release":
            self._release_peer_recv_slot_for_msg(str(data["msg_id"]))
            return
        if message_type != "message":
            raise RuntimeError(f"unknown PAP NIXL mailbox notification: {data}")
        message = self._read_remote_message(data)
        with self._cv:
            self._incoming[message.msg_id] = message
            self._cv.notify_all()
        if "materialized_recv_slot_id" in data:
            return
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
                if self._closed.is_set():
                    raise PAPOffloadExecTransportClosed("PAP NIXL mailbox is closed")
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

    def _reserve_materialized_recv_location(
        self,
        data: dict[str, Any],
        nbytes: int,
    ) -> tuple[int, int, int, bool]:
        recv_slot_id = int(data["materialized_recv_slot_id"])
        recv_slot_count = int(getattr(self, "_recv_slot_count", 1))
        if recv_slot_id < 0 or recv_slot_id >= recv_slot_count:
            raise RuntimeError(
                f"PAP NIXL mailbox invalid materialized receive slot id: {recv_slot_id}"
            )
        recv_slot_bytes = int(getattr(self, "_recv_slot_bytes", self.buffer_bytes))
        if nbytes > recv_slot_bytes:
            raise RuntimeError(
                f"PAP NIXL mailbox incoming message {data['msg_id']} requires "
                f"{nbytes} bytes, receive slot has {recv_slot_bytes}"
            )
        msg_id = str(data["msg_id"])
        wait_seconds = float(getattr(self, "_recv_slot_wait_seconds", 30.0))
        deadline = time.monotonic() + wait_seconds
        cv, leases = self._ensure_recv_slot_state()
        with cv:
            while True:
                if self._closed.is_set():
                    raise PAPOffloadExecTransportClosed("PAP NIXL mailbox is closed")
                if recv_slot_id not in leases:
                    leases[recv_slot_id] = msg_id
                    recv_offset, local_addr = self._recv_slot_location(recv_slot_id)
                    return recv_slot_id, recv_offset, local_addr, True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out waiting for PAP NIXL materialized receive slot "
                        f"for message {msg_id}"
                    )
                cv.wait(timeout=min(remaining, 0.001))

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

    def _read_xfer_cache_key(
        self,
        *,
        local_addr: int,
        nbytes: int,
        remote_addr: int,
        remote_device_id: int,
    ) -> tuple[int, int, int, int, int]:
        return (
            int(local_addr),
            int(nbytes),
            int(self.device_id),
            int(remote_addr),
            int(remote_device_id),
        )

    def _get_read_xfer_handle(
        self,
        *,
        cache_key: tuple[int, int, int, int, int],
        local_h: Any,
        remote_h: Any,
    ) -> tuple[Any, bool]:
        cache_enabled = bool(
            getattr(self, "_cache_xfer_handles_enabled", False)
            and getattr(self, "_cache_xfer_dlists_enabled", False)
        )
        if cache_enabled:
            with self._lock:
                cached = self._xfer_handle_cache.get(cache_key)
            if cached is not None:
                return cached, False
        xfer_h = self._wrapper.make_prepped_xfer(
            "READ",
            local_h,
            [0],
            remote_h,
            [0],
        )
        if cache_enabled:
            with self._lock:
                cached = self._xfer_handle_cache.setdefault(cache_key, xfer_h)
            if cached != xfer_h:
                self._wrapper.release_xfer_handle(xfer_h)
            return cached, False
        return xfer_h, True

    def _evict_cached_read_xfer_handle(
        self,
        cache_key: tuple[int, int, int, int, int],
        xfer_h: Any,
    ) -> None:
        cache = getattr(self, "_xfer_handle_cache", None)
        if not cache:
            return
        with self._lock:
            if cache.get(cache_key) == xfer_h:
                del cache[cache_key]
                self._wrapper.release_xfer_handle(xfer_h)

    def _get_write_xfer_handle(
        self,
        *,
        cache_key: tuple[int, int, int, int, int],
        local_h: Any,
        remote_h: Any,
    ) -> tuple[Any, bool]:
        cache_enabled = bool(
            getattr(self, "_cache_write_xfer_handles_enabled", False)
            and getattr(self, "_cache_xfer_dlists_enabled", False)
        )
        if cache_enabled:
            with self._lock:
                cached = self._write_xfer_handle_cache.get(cache_key)
            if cached is not None:
                return cached, False
        xfer_h = self._wrapper.make_prepped_xfer(
            "WRITE",
            local_h,
            [0],
            remote_h,
            [0],
        )
        if cache_enabled:
            with self._lock:
                cached = self._write_xfer_handle_cache.setdefault(cache_key, xfer_h)
            if cached != xfer_h:
                self._wrapper.release_xfer_handle(xfer_h)
            return cached, False
        return xfer_h, True

    def _evict_cached_write_xfer_handle(
        self,
        cache_key: tuple[int, int, int, int, int],
        xfer_h: Any,
    ) -> None:
        cache = getattr(self, "_write_xfer_handle_cache", None)
        if not cache:
            return
        with self._lock:
            if cache.get(cache_key) == xfer_h:
                del cache[cache_key]
                self._wrapper.release_xfer_handle(xfer_h)

    def _should_push_write_message(self, message: PAPMailboxMessage) -> bool:
        if message.kind not in getattr(self, "_push_write_kinds", set()):
            return False
        if not getattr(self, "_slot_protocol_enabled", False):
            return False
        return (
            getattr(self, "_peer_recv_buffer_addr", None) is not None
            and getattr(self, "_peer_recv_device_id", None) is not None
            and int(getattr(self, "_peer_recv_slot_bytes", 0)) > 0
        )

    def _write_payload_to_peer_recv_slot(
        self,
        *,
        msg_id: str,
        source_addr: int,
        nbytes: int,
    ) -> tuple[int, dict[str, float | int]]:
        trace_enabled = self._trace_enabled
        write_start = time.perf_counter() if trace_enabled else 0.0
        recv_slot_id, remote_addr = self._reserve_peer_recv_slot(msg_id, nbytes)
        cache_key = self._read_xfer_cache_key(
            local_addr=source_addr,
            nbytes=nbytes,
            remote_addr=remote_addr,
            remote_device_id=int(self._peer_recv_device_id),
        )
        local_h = None
        remote_h = None
        xfer_h = None
        release_dlists = False
        release_xfer = False
        write_polls = 0
        try:
            prepare_start = time.perf_counter() if trace_enabled else 0.0
            local_h, remote_h, release_dlists = self._get_read_dlist_handles(
                local_addr=source_addr,
                nbytes=nbytes,
                remote_addr=remote_addr,
                remote_device_id=int(self._peer_recv_device_id),
            )
            xfer_h, release_xfer = self._get_write_xfer_handle(
                cache_key=cache_key,
                local_h=local_h,
                remote_h=remote_h,
            )
            write_prepare_ms = (
                (time.perf_counter() - prepare_start) * 1000.0 if trace_enabled else 0.0
            )
            transfer_start = time.perf_counter() if trace_enabled else 0.0
            self._wrapper.transfer(xfer_h)
            while True:
                state = self._wrapper.check_xfer_state(xfer_h)
                if state == "DONE":
                    break
                if state != "PROC":
                    self._evict_cached_write_xfer_handle(cache_key, xfer_h)
                    raise RuntimeError(f"PAP NIXL push transfer failed: {state}")
                write_polls += 1
                if self._xfer_poll_sleep_seconds > 0:
                    time.sleep(self._xfer_poll_sleep_seconds)
            write_transfer_ms = (
                (time.perf_counter() - transfer_start) * 1000.0
                if trace_enabled
                else 0.0
            )
        except Exception:
            self._release_peer_recv_slot_for_msg(msg_id)
            raise
        finally:
            if release_xfer and xfer_h is not None:
                self._wrapper.release_xfer_handle(xfer_h)
            if release_dlists and local_h is not None:
                self._wrapper.release_dlist_handle(local_h)
            if release_dlists and remote_h is not None:
                self._wrapper.release_dlist_handle(remote_h)
        write_ms = (
            (time.perf_counter() - write_start) * 1000.0 if trace_enabled else 0.0
        )
        return recv_slot_id, {
            "write_ms": write_ms,
            "write_prepare_ms": write_prepare_ms,
            "write_transfer_ms": write_transfer_ms,
            "write_polls": write_polls,
        }

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
        materialized_recv_slot = "materialized_recv_slot_id" in data
        transfer_polls = 0
        transfer_ms = 0.0
        slot_wait_ms = 0.0
        handle_prepare_ms = 0.0
        if materialized_recv_slot:
            slot_wait_start = time.perf_counter() if trace_enabled else 0.0
            recv_slot_id, recv_offset, local_addr, release_recv_slot = (
                self._reserve_materialized_recv_location(data, nbytes)
            )
            slot_wait_ms = (
                (time.perf_counter() - slot_wait_start) * 1000.0
                if trace_enabled
                else 0.0
            )
            prepare_ms = (
                (time.perf_counter() - prepare_start) * 1000.0 if trace_enabled else 0.0
            )
        else:
            remote_addr, remote_device_id = self._remote_payload_location(data, nbytes)
            slot_wait_start = time.perf_counter() if trace_enabled else 0.0
            recv_slot_id, recv_offset, local_addr, release_recv_slot = (
                self._reserve_local_recv_location(data, nbytes)
            )
            slot_wait_ms = (
                (time.perf_counter() - slot_wait_start) * 1000.0
                if trace_enabled
                else 0.0
            )
            cache_key = self._read_xfer_cache_key(
                local_addr=local_addr,
                nbytes=nbytes,
                remote_addr=remote_addr,
                remote_device_id=remote_device_id,
            )
            local_h = None
            remote_h = None
            release_dlists = False
            try:
                handle_start = time.perf_counter() if trace_enabled else 0.0
                local_h, remote_h, release_dlists = self._get_read_dlist_handles(
                    local_addr=local_addr,
                    nbytes=nbytes,
                    remote_addr=remote_addr,
                    remote_device_id=remote_device_id,
                )
                xfer_h, release_xfer = self._get_read_xfer_handle(
                    cache_key=cache_key,
                    local_h=local_h,
                    remote_h=remote_h,
                )
                handle_prepare_ms = (
                    (time.perf_counter() - handle_start) * 1000.0
                    if trace_enabled
                    else 0.0
                )
            except Exception:
                if release_dlists:
                    self._wrapper.release_dlist_handle(local_h)
                    self._wrapper.release_dlist_handle(remote_h)
                if release_recv_slot:
                    self._release_recv_slot(recv_slot_id, msg_id)
                raise
            prepare_ms = (
                (time.perf_counter() - prepare_start) * 1000.0 if trace_enabled else 0.0
            )
            transfer_start = time.perf_counter() if trace_enabled else 0.0
            try:
                self._wrapper.transfer(xfer_h)
                while True:
                    state = self._wrapper.check_xfer_state(xfer_h)
                    if state == "DONE":
                        break
                    if state != "PROC":
                        self._evict_cached_read_xfer_handle(cache_key, xfer_h)
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
                if release_xfer:
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
        read_total_ms = prepare_ms + transfer_ms + materialize_ms
        recv_trace = None
        if trace_enabled:
            recv_trace = {
                "nbytes": float(nbytes),
                "read_prepare_ms": prepare_ms,
                "read_slot_wait_ms": slot_wait_ms,
                "read_handle_prepare_ms": handle_prepare_ms,
                "transfer_ms": transfer_ms,
                "transfer_polls": float(transfer_polls),
                "materialize_ms": materialize_ms,
                "read_total_ms": read_total_ms,
            }
            logger.info(
                "PAP NIXL mailbox read trace actor=%s msg_id=%s kind=%s "
                "nbytes=%d prepare_ms=%.3f slot_wait_ms=%.3f "
                "handle_prepare_ms=%.3f transfer_ms=%.3f transfer_polls=%d "
                "materialize_ms=%.3f total_ms=%.3f",
                self.actor_id,
                msg_id,
                str(data["kind"]),
                nbytes,
                prepare_ms,
                slot_wait_ms,
                handle_prepare_ms,
                transfer_ms,
                transfer_polls,
                materialize_ms,
                read_total_ms,
            )
        release_callback = None
        if release_recv_slot:
            if materialized_recv_slot:
                if getattr(self, "_piggyback_recv_releases_enabled", False):
                    release_callback = (
                        lambda recv_slot_id=recv_slot_id, msg_id=msg_id: (
                            self._release_recv_slot(recv_slot_id, msg_id),
                            self._defer_recv_release(msg_id, recv_slot_id),
                        )
                    )
                else:
                    release_callback = (
                        lambda recv_slot_id=recv_slot_id, msg_id=msg_id: (
                            self._release_recv_slot(recv_slot_id, msg_id),
                            self._send_recv_release_notification(msg_id, recv_slot_id),
                        )
                    )
            else:
                release_callback = lambda recv_slot_id=recv_slot_id, msg_id=msg_id: (
                    self._release_recv_slot(recv_slot_id, msg_id)
                )
        return PAPMailboxMessage(
            msg_id=msg_id,
            kind=str(data["kind"]),
            metadata=dict(data.get("metadata") or {}),
            tensor=tensor,
            recv_trace=recv_trace,
            release_callback=release_callback,
        )
