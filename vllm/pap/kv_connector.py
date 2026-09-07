# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Prefill KV publication through vLLM's connector interface."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.pap.integration.request import PAPRequestMetadata
from vllm.pap.integration.settings import PAPRuntimeSettings
from vllm.pap.kv import lease as pap_lease
from vllm.pap.model.prefill import PAPPrefillKVPublisher
from vllm.pap.runtime_cuda_context_audit import write_runtime_cuda_context_audit
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


@dataclass(frozen=True, slots=True)
class PAPPrefillRequest:
    request_id: str
    num_scheduled_tokens: int
    prefill_kv_handle: str | None
    decode_capacity_tokens: int | None
    import_to_attention: bool
    attention_tcp_endpoint: str | None
    allocated_block_ids: tuple[int, ...] = ()
    lease_id: str | None = None
    generation: int = 0


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
        self._generations: dict[str, int] = {}
        self._layer_names: tuple[str, ...] = ()
        self._publishers: dict[str, PAPPrefillKVPublisher] = {}
        self._pending_finished: set[str] = set()
        self._control_finished: set[str] = set()
        self._decode_query_len = _decode_query_len(vllm_config)
        self._block_size = int(vllm_config.cache_config.block_size)
        self._default_decode_capacity = (
            PAPRuntimeSettings.from_environ().unified_kv_decode_capacity_tokens
        )
        self._num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._layer_names = tuple(kv_caches)
        write_runtime_cuda_context_audit(role="prefill")

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
            allocated_block_ids_by_request={
                r.request_id: r.allocated_block_ids for r in requests
            },
            decode_capacity_tokens_by_request={
                request.request_id: request.decode_capacity_tokens
                for request in requests
                if request.decode_capacity_tokens is not None
            },
            lease_ids_by_request={r.request_id: r.lease_id for r in requests},
            generations_by_request={r.request_id: r.generation for r in requests},
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
        newly_finished = finished_req_ids - self._control_finished
        self._control_finished.difference_update(finished_req_ids)
        return self._collect_finished(newly_finished)

    def get_finished_for_control(
        self, finished_req_ids: set[str], request_ids: set[str]
    ) -> set[str]:
        """Acknowledge selected releases before the next model iteration."""
        newly_finished = (finished_req_ids & request_ids) - self._control_finished
        ready, _ = self._collect_finished(newly_finished, request_ids)
        ready = ready or set()
        self._control_finished.update(ready & finished_req_ids)
        return ready

    def _collect_finished(
        self, finished_req_ids: set[str], request_ids: set[str] | None = None
    ) -> tuple[set[str] | None, set[str] | None]:
        for req_id in map(str, finished_req_ids):
            if pap_lease.pap_has_active_lease(
                req_id
            ) or pap_lease.pap_was_recently_released(req_id):
                self._pending_finished.add(req_id)
        ready = {
            req_id
            for req_id in self._pending_finished
            if (request_ids is None or req_id in request_ids)
            and not pap_lease.pap_has_active_lease(req_id)
        }
        if not ready:
            return None, None
        self._pending_finished.difference_update(ready)
        for publisher in self._publishers.values():
            publisher.finish_requests(ready)
        for request_id in ready:
            self._request_metadata.pop(request_id, None)
            self._generations.pop(request_id, None)
        return ready, None

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        for request_id in connector_output.finished_sending or ():
            self._request_metadata.pop(request_id, None)
            self._generations.pop(request_id, None)

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
        metadata = PAPRequestMetadata.from_mapping(request.kv_transfer_params)
        if (
            metadata.import_prefill_kv_to_attention
            and metadata.decode_capacity_tokens is None
        ):
            metadata = replace(
                metadata, decode_capacity_tokens=self._default_decode_capacity
            )
        self._request_metadata[request.request_id] = metadata
        self._generations[request.request_id] = 0

    def preempt_request(self, request: Request) -> None:
        """Revoke a Prefill-only mapping before the scheduler recycles its blocks."""
        from vllm.pap.kv.handoff import revoke_prefill_kv

        metadata = self._request_metadata.get(request.request_id)
        if metadata is None or not metadata.import_prefill_kv_to_attention:
            return
        generation = self._generations[request.request_id]
        lease_id = pap_lease.pap_active_lease_id(request.request_id)
        if not metadata.attention_tcp_endpoint or not metadata.prefill_kv_handle:
            raise RuntimeError("PAP preemption lacks Attention ownership endpoint")
        revoke_prefill_kv(
            endpoint=metadata.attention_tcp_endpoint,
            session_handle=metadata.prefill_kv_handle,
            generation=generation,
        )
        if lease_id is not None:
            pap_lease.pap_release_lease(lease_id)
        self._generations[request.request_id] = generation + 1

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        *,
        allocated_blocks: dict[str, tuple[list[int], ...]] | None = None,
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
            owned: tuple[int, ...] = ()
            lease_id = None
            if metadata.import_prefill_kv_to_attention:
                groups = (allocated_blocks or {}).get(request_id)
                if groups is None or len(groups) != 1:
                    raise RuntimeError(
                        "PAP publication requires authoritative KV ownership"
                    )
                owned = tuple(groups[0])
                if not owned or len(set(owned)) != len(owned):
                    raise RuntimeError("PAP scheduler produced invalid owned KV blocks")
                lease_id = pap_lease.pap_active_lease_id(request_id)
                registry = pap_lease.get_global_kv_lease_registry()
                if lease_id is None:
                    lease_id = registry.pin_blocks(
                        request_id=request_id,
                        block_ids=owned,
                        ttl_seconds=0,
                    )
                else:
                    registry.extend_blocks(request_id, owned)
            requests.append(
                PAPPrefillRequest(
                    request_id=request_id,
                    num_scheduled_tokens=num_tokens[request_id],
                    prefill_kv_handle=metadata.prefill_kv_handle,
                    decode_capacity_tokens=metadata.decode_capacity_tokens,
                    import_to_attention=metadata.import_prefill_kv_to_attention,
                    attention_tcp_endpoint=metadata.attention_tcp_endpoint,
                    allocated_block_ids=owned,
                    lease_id=lease_id,
                    generation=self._generations.get(request_id, 0),
                )
            )
        finished = tuple(str(req_id) for req_id in scheduler_output.finished_req_ids)
        return PAPPrefillConnectorMetadata(tuple(requests), finished)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        request_id = request.request_id
        metadata = self._request_metadata.get(request_id, PAPRequestMetadata())
        if not metadata.import_prefill_kv_to_attention:
            self._request_metadata.pop(request_id, None)
            self._generations.pop(request_id, None)
            return False, None
        lease_id = pap_lease.pap_active_lease_id(request_id)
        if lease_id is None:
            if request.status == RequestStatus.FINISHED_ABORTED:
                self._pending_finished.discard(request_id)
                for publisher in self._publishers.values():
                    publisher.finish_requests({request_id})
                self._request_metadata.pop(request_id, None)
                self._generations.pop(request_id, None)
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
