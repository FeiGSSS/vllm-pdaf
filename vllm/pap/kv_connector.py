# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Prefill KV publication through vLLM's connector interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.pap.integration.request import PAPRequestMetadata
from vllm.pap.kv import lease as pap_lease
from vllm.pap.model.prefill import PAPPrefillKVPublisher
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass(frozen=True, slots=True)
class PAPPrefillRequest:
    request_id: str
    num_scheduled_tokens: int
    prefill_kv_handle: str | None
    decode_capacity_tokens: int | None
    import_to_attention: bool
    attention_tcp_endpoint: str | None


@dataclass(slots=True)
class PAPPrefillConnectorMetadata(KVConnectorMetadata):
    requests: tuple[PAPPrefillRequest, ...]
    finished_request_ids: tuple[str, ...]


def _decode_query_len(vllm_config: VllmConfig) -> int:
    speculative = vllm_config.speculative_config
    return 1 if speculative is None else speculative.num_speculative_tokens + 1


class PAPPrefillConnector(KVConnectorBase_V1):
    """Publish paged Prefill KV to PAP Attention over same-host CUDA IPC."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._request_metadata: dict[str, PAPRequestMetadata] = {}
        self._layer_names: tuple[str, ...] = ()
        self._publishers: dict[str, PAPPrefillKVPublisher] = {}
        self._pending_finished: set[str] = set()
        self._decode_query_len = _decode_query_len(vllm_config)
        self._block_size = int(vllm_config.cache_config.block_size)
        self._num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._layer_names = tuple(kv_caches)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        del forward_context, kwargs

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        del kwargs
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, PAPPrefillConnectorMetadata):
            raise TypeError("PAP Prefill connector received incompatible metadata")
        if not metadata.requests:
            return
        if layer_name not in self._layer_names:
            raise RuntimeError(
                f"PAP Prefill connector saw unknown layer {layer_name!r}"
            )
        publisher = self._publishers.get(layer_name)
        if publisher is None:
            publisher = PAPPrefillKVPublisher(
                layer_name=layer_name,
                num_kv_heads=self._num_kv_heads,
                is_last_layer=layer_name == self._layer_names[-1],
                expected_layer_count=len(self._layer_names),
            )
            self._publishers[layer_name] = publisher

        requests = metadata.requests
        publisher.publish_connector_batch(
            request_ids=tuple(request.request_id for request in requests),
            num_scheduled_tokens=tuple(
                request.num_scheduled_tokens for request in requests
            ),
            prefill_kv_handle_by_request={
                request.request_id: request.prefill_kv_handle
                for request in requests
                if request.prefill_kv_handle is not None
            },
            decode_capacity_tokens_by_request={
                request.request_id: request.decode_capacity_tokens
                for request in requests
                if request.decode_capacity_tokens is not None
            },
            import_request_ids={
                request.request_id
                for request in requests
                if request.import_to_attention
            },
            tcp_endpoint_by_request={
                request.request_id: request.attention_tcp_endpoint
                for request in requests
                if request.attention_tcp_endpoint is not None
            },
            attn_metadata=attn_metadata,
            kv_cache=kv_layer,
            block_size=self._block_size,
        )

    def wait_for_save(self) -> None:
        return

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        for req_id in map(str, finished_req_ids):
            if pap_lease.pap_has_active_lease(
                req_id
            ) or pap_lease.pap_was_recently_released(req_id):
                self._pending_finished.add(req_id)
        ready = {
            req_id
            for req_id in self._pending_finished
            if not pap_lease.pap_has_active_lease(req_id)
        }
        if not ready:
            return None, None
        self._pending_finished.difference_update(ready)
        for publisher in self._publishers.values():
            publisher.finish_requests(ready)
        return ready, None

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        del request, num_computed_tokens
        return 0, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        del request, blocks, num_external_tokens

    def on_new_request(self, request: Request) -> None:
        self._request_metadata[request.request_id] = PAPRequestMetadata.from_mapping(
            request.kv_transfer_params
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> PAPPrefillConnectorMetadata:
        num_tokens = scheduler_output.num_scheduled_tokens
        request_ids = sorted(
            num_tokens,
            key=lambda req_id: (
                num_tokens[req_id] != self._decode_query_len,
                num_tokens[req_id],
            ),
        )
        requests = []
        for request_id in request_ids:
            metadata = self._request_metadata.get(request_id, PAPRequestMetadata())
            requests.append(
                PAPPrefillRequest(
                    request_id=request_id,
                    num_scheduled_tokens=num_tokens[request_id],
                    prefill_kv_handle=metadata.prefill_kv_handle,
                    decode_capacity_tokens=metadata.decode_capacity_tokens,
                    import_to_attention=metadata.import_prefill_kv_to_attention,
                    attention_tcp_endpoint=metadata.attention_tcp_endpoint,
                )
            )
        finished = tuple(str(req_id) for req_id in scheduler_output.finished_req_ids)
        return PAPPrefillConnectorMetadata(tuple(requests), finished)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        request_id = request.request_id
        metadata = self._request_metadata.pop(request_id, PAPRequestMetadata())
        if not metadata.import_prefill_kv_to_attention:
            return False, None
        lease_id = pap_lease.pap_active_lease_id(request_id)
        if lease_id is None:
            if request.status == RequestStatus.FINISHED_ABORTED:
                self._pending_finished.discard(request_id)
                for publisher in self._publishers.values():
                    publisher.finish_requests({request_id})
                return False, None
            raise RuntimeError(
                f"PAP Prefill request {request_id} finished without a KV lease"
            )
        params: dict[str, Any] = {
            "remote_num_tokens": int(request.num_computed_tokens),
            "pap_prefill_kv_handle": metadata.prefill_kv_handle,
            "pap_kv_lease_id": lease_id,
        }
        return True, params

    def has_pending_push_work(self) -> bool:
        return bool(self._pending_finished)


__all__ = [
    "PAPPrefillConnector",
    "PAPPrefillConnectorMetadata",
    "PAPPrefillRequest",
]
