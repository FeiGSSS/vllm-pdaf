# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conversation-sticky round-robin PA routing."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from vllm.pap.gateway.topology import PAPGroup

logger = logging.getLogger("pap_gateway")


class PAPSessionRoundRobinRouter:
    """Assign new conversations round-robin and keep later turns on that PA."""

    def __init__(
        self,
        groups: Sequence[PAPGroup],
        *,
        max_conversations: int,
    ) -> None:
        if not groups:
            raise ValueError("session round-robin routing requires at least one PA")
        if max_conversations <= 0:
            raise ValueError("session routing conversation limit must be positive")
        self._groups = tuple(groups)
        self._max_conversations = max_conversations
        self._conversations: OrderedDict[str, int] = OrderedDict()
        self._requests: dict[str, str] = {}
        self._next_worker = 0
        self._selections = 0
        self._failures = 0
        self._closed = False

    async def select_group(
        self,
        token_ids: list[int],
        *,
        request_id: str,
        expected_output_tokens: int,
        conversation_id: str = "",
        verified_worker_id: int | None = None,
        verified_prefix_tokens: int = 0,
    ) -> PAPGroup:
        del (
            token_ids,
            expected_output_tokens,
            verified_worker_id,
            verified_prefix_tokens,
        )
        if self._closed:
            raise RuntimeError("session round-robin router is closed")
        if not conversation_id:
            self._failures += 1
            raise ValueError("session round-robin routing requires a conversation ID")
        if request_id in self._requests:
            self._failures += 1
            raise ValueError(f"duplicate session routing request ID: {request_id}")
        worker_id = self._conversations.get(conversation_id)
        new_binding = worker_id is None
        if worker_id is None:
            if len(self._conversations) >= self._max_conversations:
                self._failures += 1
                raise RuntimeError("session routing conversation limit reached")
            worker_id = self._next_worker
            self._next_worker = (self._next_worker + 1) % len(self._groups)
            self._conversations[conversation_id] = worker_id
        else:
            self._conversations.move_to_end(conversation_id)
        self._requests[request_id] = conversation_id
        self._selections += 1
        logger.info(
            "PAP session round-robin placement request_id=%s conversation_id=%s "
            "selected_pa=%d new_binding=%s",
            request_id,
            conversation_id,
            worker_id,
            new_binding,
        )
        return self._groups[worker_id]

    async def mark_prefill_completed(self, request_id: str) -> None:
        del request_id

    def finish_request(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def record_prefill_result(
        self,
        *,
        request_id: str,
        conversation_id: str,
        worker_id: int,
        prompt_tokens: int,
        cached_tokens: int,
        prefill_ms: int,
    ) -> None:
        del (
            request_id,
            conversation_id,
            worker_id,
            prompt_tokens,
            cached_tokens,
            prefill_ms,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "policy": "session_round_robin",
            "selections": self._selections,
            "failures": self._failures,
            "active_reservations": len(self._requests),
            "pending_cleanup": 0,
            "pending_selections": 0,
            "healthy": not self._closed and self._failures == 0,
            "reservation_lifetime": "gateway-request-v1",
            "conversation_bindings": len(self._conversations),
            "max_conversations": self._max_conversations,
            "loads": [],
        }

    async def shutdown(self) -> None:
        self._closed = True
        self._requests.clear()
        self._conversations.clear()
