# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM model-runner bridge for asynchronous PAP sampled tokens."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from vllm.pap.decode_token_client import DecodeTokenClient
from vllm.pap.integration.request import PAPProjectionRequestStore
from vllm.v1.outputs import ModelRunnerOutput


class _DecodeTokenClient(Protocol):
    def publish_batch(self, tokens: Sequence[Mapping[str, object]]) -> None: ...

    def flush_request(self, request_id: str) -> bool: ...

    def forget_request(self, request_id: str) -> None: ...

    def shutdown(self) -> None: ...


def _publish_sampled_tokens(
    output: ModelRunnerOutput,
    *,
    client: _DecodeTokenClient,
    pap_request_ids: frozenset[str],
    session_request_id_by_request: Mapping[str, str],
    attention_endpoint_by_request: Mapping[str, str],
    next_seq_len_by_request: Mapping[str, int],
) -> None:
    notifications: list[dict[str, object]] = []
    for request_id, token_ids in zip(output.req_ids, output.sampled_token_ids):
        if not token_ids or request_id not in pap_request_ids:
            continue
        session_request_id = session_request_id_by_request.get(request_id)
        attention_endpoint = attention_endpoint_by_request.get(request_id)
        next_seq_len = next_seq_len_by_request.get(request_id)
        if (
            session_request_id is None
            or attention_endpoint is None
            or next_seq_len is None
        ):
            raise RuntimeError(
                "PAP asynchronous decode-token delivery is missing routing "
                f"metadata for sampled request {request_id}"
            )
        if len(token_ids) != 1:
            raise RuntimeError(
                "PAP async decode-token handoff requires one sampled token per "
                f"request, got {len(token_ids)} for {request_id}"
            )
        notifications.append(
            {
                "request_id": session_request_id,
                "new_seq_len": next_seq_len,
                "token_id": int(token_ids[0]),
                "endpoint": attention_endpoint,
            }
        )
    client.publish_batch(notifications)


@dataclass(slots=True)
class PAPDecodeTokenBridge:
    """Own the sampled-token client boundary used by the Projection runner."""

    client: _DecodeTokenClient | None = None

    def build_callback(
        self,
        store: PAPProjectionRequestStore,
        *,
        pap_request_ids: frozenset[str],
        request_ids: Sequence[str],
        seq_lens_cpu_upper_bound: Iterable[int],
    ) -> Callable[[ModelRunnerOutput], None] | None:
        """Capture routing state for one asynchronous sampler callback."""
        if not pap_request_ids:
            return None
        normalized_ids = tuple(str(request_id) for request_id in request_ids)
        session_request_id_by_request = {
            request_id: store.prefill_kv_handle_by_request[request_id]
            for request_id in normalized_ids
            if request_id in store.prefill_kv_handle_by_request
        }
        attention_endpoint_by_request = {
            request_id: store.attention_endpoint_by_request[request_id]
            for request_id in normalized_ids
            if request_id in store.attention_endpoint_by_request
        }
        next_seq_len_by_request = {
            request_id: int(seq_len) + 1
            for request_id, seq_len in zip(
                normalized_ids,
                seq_lens_cpu_upper_bound,
            )
        }
        if self.client is None:
            self.client = DecodeTokenClient()
        return functools.partial(
            _publish_sampled_tokens,
            client=self.client,
            pap_request_ids=pap_request_ids,
            session_request_id_by_request=session_request_id_by_request,
            attention_endpoint_by_request=attention_endpoint_by_request,
            next_seq_len_by_request=next_seq_len_by_request,
        )

    def drain_request(
        self,
        store: PAPProjectionRequestStore,
        request_id: str,
    ) -> None:
        """Flush sampled tokens before Projection request state is removed."""
        session_request_id = store.prefill_kv_handle_by_request.get(request_id)
        if self.client is None or session_request_id is None:
            return
        if not self.client.flush_request(session_request_id):
            raise RuntimeError(
                "PAP decode-token delivery failed before request removal: "
                f"{session_request_id}"
            )
        self.client.forget_request(session_request_id)

    def shutdown(self) -> None:
        """Stop the delivery worker if it was created."""
        if self.client is not None:
            self.client.shutdown()
