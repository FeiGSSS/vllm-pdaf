# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-neutral PAP mailbox messages and actor lifecycle."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Condition, RLock
from typing import Any, Protocol

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PAPMailboxMessage:
    """One complete PAP mailbox message and tensor payload."""

    msg_id: str
    kind: str
    metadata: dict[str, Any]
    tensor: torch.Tensor
    payload_shape: tuple[int, ...] | None = None
    direct_payload: bool = False
    payload_slot_id: int | None = None
    payload_ready_event: Any | None = field(default=None, repr=False, compare=False)
    recv_trace: dict[str, float] | None = field(
        default=None, repr=False, compare=False
    )
    release_callback: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )
    _released: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.msg_id:
            raise ValueError("PAP mailbox message requires msg_id")
        if not self.kind:
            raise ValueError("PAP mailbox message requires kind")
        if self.direct_payload and self.payload_shape is None:
            object.__setattr__(
                self,
                "payload_shape",
                tuple(int(dim) for dim in self.tensor.shape),
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


def _merge_message_recv_trace(
    message: PAPMailboxMessage,
    fields: dict[str, float],
) -> None:
    recv_trace = dict(message.recv_trace or {})
    recv_trace.update((key, float(value)) for key, value in fields.items())
    object.__setattr__(message, "recv_trace", recv_trace)


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
        """Put a completed message in the output pool and request delivery."""

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
