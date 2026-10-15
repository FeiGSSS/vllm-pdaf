# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Prefill KV publication through vLLM's connector interface."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger
from vllm.pap.integration.request import PAPKVCacheMigration, PAPRequestMetadata
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

logger = init_logger(__name__)


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
    kv_migration: PAPKVCacheMigration | None = None
    migration_source_block_ids: tuple[int, ...] = ()
    migration_destination_block_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class PAPPrefillConnectorMetadata(KVConnectorMetadata):
    requests: tuple[PAPPrefillRequest, ...]


def _decode_query_len(vllm_config: VllmConfig) -> int:
    speculative = vllm_config.speculative_config
    return 1 if speculative is None else speculative.num_speculative_tokens + 1


def _contiguous_block_copy_ranges(
    source_block_ids: tuple[int, ...],
    destination_block_ids: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    """Coalesce matching source/destination block runs as start/start/count."""
    if len(source_block_ids) != len(destination_block_ids):
        raise ValueError("PAP KV migration block mapping length mismatch")
    if not source_block_ids:
        return ()
    ranges = []
    source_start = source_block_ids[0]
    destination_start = destination_block_ids[0]
    count = 1
    for source, destination in zip(source_block_ids[1:], destination_block_ids[1:]):
        if source == source_start + count and destination == destination_start + count:
            count += 1
            continue
        ranges.append((source_start, destination_start, count))
        source_start = source
        destination_start = destination
        count = 1
    ranges.append((source_start, destination_start, count))
    return tuple(ranges)


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
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._publishers: dict[str, PAPPrefillKVPublisher] = {}
        self._migration_catalog_id = f"prefill-migration-{os.getpid()}"
        self._migration_catalog: tuple[dict[str, Any], ...] | None = None
        self._migration_start_blocks: dict[str, int] = {}
        self._migration_loads: dict[
            str, tuple[PAPKVCacheMigration, tuple[int, ...], tuple[int, ...]]
        ] = {}
        self._migration_sources: dict[tuple[str, str], torch.Tensor] = {}
        self._migration_requests = 0
        self._migration_blocks = 0
        self._migration_copy_events: list[
            tuple[str, torch.cuda.Event, torch.cuda.Event]
        ] = []
        self._verify_migration = os.environ.get("PAP_KV_MIGRATION_VERIFY", "0") == "1"
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
        self._kv_caches = dict(kv_caches)
        write_runtime_cuda_context_audit(role="prefill")

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        del forward_context, kwargs
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, PAPPrefillConnectorMetadata):
            raise TypeError("PAP Prefill connector received incompatible metadata")
        for request in metadata.requests:
            migration = request.kv_migration
            if migration is None:
                continue
            source_caches = self._open_migration_catalog(migration)
            destination_caches = {
                layer_name: self._kv_caches[layer_name] for layer_name in source_caches
            }
            copy_ranges = _contiguous_block_copy_ranges(
                request.migration_source_block_ids,
                request.migration_destination_block_ids,
            )
            copy_start = torch.cuda.Event(enable_timing=True)
            copy_end = torch.cuda.Event(enable_timing=True)
            copy_start.record()
            for layer_name, source in source_caches.items():
                destination = destination_caches[layer_name]
                for source_start, destination_start, count in copy_ranges:
                    destination_slice = destination[
                        destination_start : destination_start + count
                    ]
                    source_slice = source[source_start : source_start + count]
                    destination_slice.copy_(source_slice, non_blocking=True)
                    if self._verify_migration and not torch.equal(
                        destination_slice, source_slice
                    ):
                        raise RuntimeError(
                            "PAP KV migration verification found mismatched KV"
                        )
            copy_end.record()
            self._migration_copy_events.append(
                (request.request_id, copy_start, copy_end)
            )
            self._migration_requests += 1
            self._migration_blocks += len(request.migration_source_block_ids)
            logger.info(
                "PAP KV migration copied request_id=%s source_pa=%d blocks=%d "
                "tokens=%d ranges=%d",
                request.request_id,
                migration.source_worker_id,
                len(request.migration_source_block_ids),
                len(request.migration_source_block_ids) * self._block_size,
                len(copy_ranges),
            )
            if self._verify_migration:
                logger.info(
                    "PAP KV migration verified request_id=%s blocks=%d layers=%d",
                    request.request_id,
                    len(request.migration_source_block_ids),
                    len(source_caches),
                )

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
        for request_id, start, end in self._migration_copy_events:
            logger.info(
                "PAP KV migration copy timing request_id=%s copy_ms=%.3f",
                request_id,
                start.elapsed_time(end),
            )
        self._migration_copy_events.clear()

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
            self._migration_start_blocks.pop(request_id, None)
            self._migration_loads.pop(request_id, None)
        return ready, None

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        for request_id in connector_output.finished_sending or ():
            self._request_metadata.pop(request_id, None)
            self._generations.pop(request_id, None)
            self._migration_start_blocks.pop(request_id, None)
            self._migration_loads.pop(request_id, None)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        metadata = self._request_metadata.get(request.request_id)
        migration = None if metadata is None else metadata.kv_migration
        if migration is None:
            return 0, False
        if migration.block_size != self._block_size:
            raise ValueError("PAP KV migration block size differs from destination")
        max_prefix = ((request.num_tokens - 1) // self._block_size) * self._block_size
        prefix_tokens = min(migration.prefix_tokens, max_prefix)
        if num_computed_tokens >= prefix_tokens:
            self._migration_start_blocks.pop(request.request_id, None)
            return 0, False
        if num_computed_tokens % self._block_size:
            raise ValueError("PAP local cache match is not block aligned")
        self._migration_start_blocks[request.request_id] = (
            num_computed_tokens // self._block_size
        )
        return prefix_tokens - num_computed_tokens, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens <= 0:
            return
        metadata = self._request_metadata.get(request.request_id)
        migration = None if metadata is None else metadata.kv_migration
        start_block = self._migration_start_blocks.pop(request.request_id, None)
        if migration is None or start_block is None:
            raise RuntimeError("PAP KV migration allocation has no transfer plan")
        if len(blocks.blocks) != 1:
            raise RuntimeError("PAP KV migration requires one KV cache group")
        block_count = num_external_tokens // self._block_size
        source_ids = migration.source_block_ids[start_block : start_block + block_count]
        destination_ids = tuple(
            block.block_id
            for block in blocks.blocks[0][start_block : start_block + block_count]
        )
        if len(source_ids) != block_count or len(destination_ids) != block_count:
            raise RuntimeError("PAP KV migration block allocation is incomplete")
        self._migration_loads[request.request_id] = (
            migration,
            source_ids,
            destination_ids,
        )

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
        self._migration_start_blocks.pop(request.request_id, None)
        self._migration_loads.pop(request.request_id, None)

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
            migration_load = self._migration_loads.pop(request_id, None)
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
                    kv_migration=(
                        None if migration_load is None else migration_load[0]
                    ),
                    migration_source_block_ids=(
                        () if migration_load is None else migration_load[1]
                    ),
                    migration_destination_block_ids=(
                        () if migration_load is None else migration_load[2]
                    ),
                )
            )
        return PAPPrefillConnectorMetadata(tuple(requests))

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        request_id = request.request_id
        metadata = self._request_metadata.get(request_id, PAPRequestMetadata())
        if not metadata.import_prefill_kv_to_attention:
            self._request_metadata.pop(request_id, None)
            self._generations.pop(request_id, None)
            self._migration_start_blocks.pop(request_id, None)
            self._migration_loads.pop(request_id, None)
            return False, None
        lease_id = pap_lease.pap_active_lease_id(request_id)
        if lease_id is None:
            if request.status == RequestStatus.FINISHED_ABORTED:
                self._pending_finished.discard(request_id)
                for publisher in self._publishers.values():
                    publisher.finish_requests({request_id})
                self._request_metadata.pop(request_id, None)
                self._generations.pop(request_id, None)
                self._migration_start_blocks.pop(request_id, None)
                self._migration_loads.pop(request_id, None)
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

    def migration_catalog(self) -> tuple[dict[str, Any], ...]:
        """Return process-lifetime CUDA IPC handles for source KV layers."""
        if not self._kv_caches:
            raise RuntimeError("PAP Prefill KV caches are not registered")
        if self._migration_catalog is None:
            from vllm.pap.kv.handoff import make_cuda_ipc_tensor_handle
            from vllm.v1.attention.backends.utils import get_kv_cache_layout

            self._migration_catalog = tuple(
                {
                    "catalog_id": self._migration_catalog_id,
                    "layer_name": layer_name,
                    "block_size": self._block_size,
                    "num_kv_heads": self._num_kv_heads,
                    "layout": get_kv_cache_layout(),
                    "kv_cache": make_cuda_ipc_tensor_handle(kv_cache).to_dict(),
                }
                for layer_name, kv_cache in self._kv_caches.items()
            )
        return self._migration_catalog

    def _open_migration_catalog(
        self, migration: PAPKVCacheMigration
    ) -> dict[str, torch.Tensor]:
        from vllm.pap.kv.handoff import open_ipc_tensor_handle

        catalogs = {catalog.layer_name: catalog for catalog in migration.catalogs}
        if set(catalogs) != set(self._kv_caches):
            raise RuntimeError("PAP KV migration layer catalog mismatch")
        opened = {}
        for layer_name, destination in self._kv_caches.items():
            catalog = catalogs[layer_name]
            key = (catalog.catalog_id, layer_name)
            source = self._migration_sources.get(key)
            if source is None:
                source = open_ipc_tensor_handle(
                    catalog.kv_cache,
                    allow_remote_device=True,
                )
                self._migration_sources[key] = source
            if (
                source.dtype != destination.dtype
                or source.shape[1:] != destination.shape[1:]
            ):
                raise RuntimeError("PAP KV migration tensor layout mismatch")
            opened[layer_name] = source
        return opened

    def has_pending_push_work(self) -> bool:
        return bool(self._pending_finished)


__all__ = [
    "PAPPrefillConnector",
    "PAPPrefillConnectorMetadata",
    "PAPPrefillRequest",
]
