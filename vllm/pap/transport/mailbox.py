# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-neutral PAP mailbox messages."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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
    recv_trace: dict[str, float] | None = field(default=None, repr=False, compare=False)
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
