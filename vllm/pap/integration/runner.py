# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP ownership boundary for vLLM model runners."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger
from vllm.pap.config import reject_removed_pap_flags
from vllm.pap.integration.decode_token import PAPDecodeTokenBridge
from vllm.pap.integration.projection import (
    build_projection_forward_context,
    select_projection_request_ids,
)
from vllm.pap.integration.request import PAPProjectionRequestStore
from vllm.pap.integration.settings import PAPRuntimeSettings
from vllm.pap.topology import (
    PAPProjectionPeerActivity,
    sync_pap_projection_peer_activity,
)

logger = init_logger(__name__)

@dataclass(slots=True)
class PAPModelRunnerAdapter:
    """Own PAP request state and hooks used by a vLLM model runner."""

    globally_enabled: bool
    attention_tcp_endpoint: Any
    block_size: int
    supports_async_sampled_tokens: bool
    projection_kv_unaware: bool
    debug_decision: bool
    critical_trace: bool = False
    store: PAPProjectionRequestStore = field(
        default_factory=PAPProjectionRequestStore
    )
    decode_token_bridge: PAPDecodeTokenBridge = field(
        default_factory=PAPDecodeTokenBridge
    )
    peer_activity: PAPProjectionPeerActivity | None = None

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: Any,
        *,
        supports_async_sampled_tokens: bool,
    ) -> PAPModelRunnerAdapter:
        """Create one adapter from the model-runner composition root."""
        reject_removed_pap_flags(os.environ)
        settings = PAPRuntimeSettings.from_environ()
        kv_transfer_config = vllm_config.kv_transfer_config
        extra = (
            kv_transfer_config.kv_connector_extra_config
            if kv_transfer_config is not None
            else {}
        ) or {}
        attention_tcp_endpoint = (
            kv_transfer_config.get_from_extra_config(
                "pap_attention_tcp_endpoint",
                None,
            )
            if kv_transfer_config is not None
            else None
        )
        return cls(
            globally_enabled=bool(extra.get("pap_enabled", False)),
            attention_tcp_endpoint=attention_tcp_endpoint,
            block_size=int(vllm_config.cache_config.block_size),
            supports_async_sampled_tokens=supports_async_sampled_tokens,
            projection_kv_unaware=settings.projection_kv_unaware,
            debug_decision=settings.debug_decision,
            critical_trace=settings.critical_trace,
        )

    def update_request(
        self,
        request_id: str,
        params: Mapping[str, Any] | None,
    ) -> None:
        """Merge scheduler metadata for one request."""
        if not params:
            if self.debug_decision:
                logger.info(
                    "PAP request update skipped req_id=%s: empty KV params",
                    request_id,
                )
            return
        if self.debug_decision:
            logger.info(
                "PAP request update req_id=%s kv_keys=%s",
                request_id,
                sorted(params.keys()),
            )
        self.store.update(request_id, params)

    def remove_request(self, request_id: str) -> None:
        """Drain sampled tokens before removing Projection request state."""
        if self.supports_async_sampled_tokens:
            self.decode_token_bridge.drain_request(self.store, request_id)
        self.store.remove(request_id)

    def request_ids(self, request_ids: Sequence[str]) -> frozenset[str]:
        """Return PAP-enabled request IDs in one runner batch."""
        return select_projection_request_ids(
            self.store,
            request_ids,
            globally_enabled=self.globally_enabled,
        )

    def build_forward_context(
        self,
        *,
        request_ids: Sequence[str],
        num_scheduled_tokens: Iterable[int],
        num_actual_tokens: int,
        positions: Any,
        seq_lens_cpu_upper_bound: Iterable[int],
        finished_request_ids: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Build the PAP forward context for one model batch."""
        normalized_ids = tuple(str(request_id) for request_id in request_ids)
        pap_enabled = bool(self.request_ids(normalized_ids))
        if pap_enabled and not self.supports_async_sampled_tokens:
            raise RuntimeError(
                "PAP asynchronous decode-token delivery requires the V2 model "
                "runner; set VLLM_USE_V2_MODEL_RUNNER=1"
            )
        context = build_projection_forward_context(
            self.store,
            request_ids=normalized_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            num_actual_tokens=num_actual_tokens,
            positions=positions,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            pap_enabled=pap_enabled,
            attention_tcp_endpoint=self.attention_tcp_endpoint,
            block_size=self.block_size,
            finished_request_ids=finished_request_ids,
        )
        if self.debug_decision:
            logger.info(
                "PAP forward context enabled=%s req_ids=%s tcp_keys=%s "
                "installed=%s",
                context["pap_enabled"],
                context["pap_request_ids"][:4],
                tuple(context["pap_attention_tcp_endpoint_by_request"]),
                tuple(context["pap_attention_kv_installed_by_request"]),
            )
        return context

    def sampled_token_callback(
        self,
        *,
        request_ids: Sequence[str],
        seq_lens_cpu_upper_bound: Iterable[int],
    ) -> Callable[[Any], None] | None:
        """Build the asynchronous sampled-token callback for one batch."""
        pap_request_ids = self.request_ids(request_ids)
        if pap_request_ids and not self.supports_async_sampled_tokens:
            raise RuntimeError(
                "PAP sampled-token callbacks require the V2 model runner"
            )
        return self.decode_token_bridge.build_callback(
            self.store,
            pap_request_ids=pap_request_ids,
            request_ids=request_ids,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        )

    def sync_peer_activity(self, request_ids: Iterable[str]) -> None:
        """Synchronize topology-derived Projection peer membership."""
        self.peer_activity = sync_pap_projection_peer_activity(
            tracker=self.peer_activity,
            request_ids=request_ids,
            endpoint_by_request=self.store.attention_endpoint_by_request,
        )

    def log_prepared_batch(
        self,
        request_ids: Sequence[str],
        num_scheduled_tokens: Mapping[str, int],
    ) -> None:
        """Emit optional request-selection diagnostics."""
        if not self.debug_decision:
            return
        first_ids = tuple(request_ids[:4])
        logger.info(
            "PAP prepare_inputs req_ids=%s endpoint_keys=%s num_tokens=%s",
            first_ids,
            tuple(list(self.store.attention_tcp_endpoint_by_request)[:4]),
            {request_id: num_scheduled_tokens[request_id] for request_id in first_ids},
        )

    def shutdown(self) -> None:
        """Stop PAP-owned background workers."""
        self.decode_token_bridge.shutdown()
