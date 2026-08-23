# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill-side PAP KV publication adapter."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.pap.attention.client import select_attention_endpoint_for_request
from vllm.pap.kv.handoff import (
    publish_prefill_kv_session_manifest,
    register_prefill_kv_catalog,
)
from vllm.pap.mode import is_pap_request_id
from vllm.pap.model.context import (
    PAPModelForwardBatch,
    pap_endpoint_for_tp_rank,
)
from vllm.pap.protocol import PAPTensorTransport
from vllm.v1.attention.backends.utils import get_kv_cache_layout

logger = init_logger(__name__)


def _pap_unified_kv_export_decode_capacity_tokens() -> int:
    raw = os.environ.get("PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS", "")
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _pap_block_ids_from_block_table(
    *,
    block_table: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> list[int]:
    if block_table.ndim != 2 or int(block_table.shape[0]) != 1:
        raise ValueError("PAP KV import supports one request per block table")
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    num_blocks = (int(seq_len) + int(block_size) - 1) // int(block_size)
    return [
        int(block_id)
        for block_id in block_table[0, :num_blocks]
        .detach()
        .to(device="cpu", dtype=torch.long)
        .tolist()
    ]


def _pap_prune_imported_prefill_kv(
    imported_prefill_kv: set[tuple[Any, ...]],
    finished_request_ids: Iterable[Any],
) -> None:
    finished = {str(request_id) for request_id in finished_request_ids}
    if not finished or not imported_prefill_kv:
        return
    imported_prefill_kv.difference_update(
        import_key
        for import_key in tuple(imported_prefill_kv)
        if import_key[0] in finished
    )


@dataclass(slots=True)
class PAPPrefillKVPublisher:
    """Publish one model layer's static catalog and sealed request manifests."""

    layer_name: str
    num_kv_heads: int
    is_last_layer: bool
    expected_layer_count: int
    catalog_id: str = field(
        default_factory=lambda: (
            os.environ.get("PAP_KV_CATALOG_ID") or f"prefill-{os.getpid()}"
        )
    )
    imported_prefill_kv: set[tuple[Any, ...]] = field(default_factory=set)
    registered_catalog_endpoints: set[str] = field(default_factory=set)
    manifest_ready_events: dict[tuple[str, int], torch.Event] = field(
        default_factory=dict
    )

    def publish_connector_batch(
        self,
        *,
        request_ids: tuple[str, ...],
        num_scheduled_tokens: tuple[int, ...],
        prefill_kv_handle_by_request: dict[str, str],
        decode_capacity_tokens_by_request: dict[str, int],
        import_request_ids: set[str],
        tcp_endpoint_by_request: dict[str, str],
        attn_metadata: Any,
        kv_cache: torch.Tensor,
        block_size: int,
    ) -> None:
        """Publish a batch supplied by the v0.26 KV connector contract."""
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        block_table = getattr(attn_metadata, "block_table", None)
        if seq_lens is None or block_table is None:
            return
        num_reqs = len(request_ids)
        if int(seq_lens.shape[0]) < num_reqs or int(block_table.shape[0]) < num_reqs:
            raise RuntimeError("PAP connector metadata disagrees with attention batch")
        self._publish_manifests(
            request_ids=request_ids,
            num_reqs=num_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            prefill_kv_handle_by_request=prefill_kv_handle_by_request,
            decode_capacity_tokens_by_request=decode_capacity_tokens_by_request,
            import_request_ids=import_request_ids,
            tcp_endpoint_by_request=tcp_endpoint_by_request,
            default_tcp_endpoint=None,
            seq_lens=seq_lens,
            block_table=block_table,
            kv_cache=kv_cache,
            block_size=block_size,
            layout=get_kv_cache_layout(),
        )

    def finish_requests(self, request_ids: Iterable[str]) -> None:
        """Discard connector-side publication bookkeeping for finished requests."""
        finished = tuple(str(request_id) for request_id in request_ids)
        _pap_prune_imported_prefill_kv(self.imported_prefill_kv, finished)
        finished_set = set(finished)
        self.manifest_ready_events = {
            key: event
            for key, event in self.manifest_ready_events.items()
            if key[0] not in finished_set
        }

    def publish(self, attention: Any) -> None:
        """Publish KV state for the current non-Projection forward, if needed."""
        batch = PAPModelForwardBatch.current(self.layer_name)
        if batch is None or not batch.enabled:
            return
        additional_kwargs = batch.additional_kwargs
        finished_request_ids = tuple(
            str(request_id)
            for request_id in additional_kwargs.get("pap_finished_request_ids") or ()
        )
        _pap_prune_imported_prefill_kv(
            self.imported_prefill_kv,
            finished_request_ids,
        )
        if finished_request_ids and self.manifest_ready_events:
            finished = set(finished_request_ids)
            self.manifest_ready_events = {
                key: event
                for key, event in self.manifest_ready_events.items()
                if key[0] not in finished
            }

        if batch.num_reqs <= 0 or len(batch.request_ids) < batch.num_reqs:
            return
        if len(batch.num_scheduled_tokens) < batch.num_reqs:
            return
        prefill_kv_handle_by_request = (
            additional_kwargs.get("pap_prefill_kv_handle_by_request") or {}
        )
        decode_capacity_tokens_by_request = (
            additional_kwargs.get("pap_decode_capacity_tokens_by_request") or {}
        )
        import_request_ids = set(
            additional_kwargs.get("pap_import_prefill_kv_to_attention_by_request") or ()
        )
        tcp_endpoint_by_request = (
            additional_kwargs.get("pap_attention_tcp_endpoint_by_request") or {}
        )
        default_tcp_endpoint = additional_kwargs.get("pap_attention_tcp_endpoint")
        attn_metadata = batch.attention_metadata
        if attn_metadata is None:
            return
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if seq_lens is None or int(seq_lens.shape[0]) < batch.num_reqs:
            return
        block_table = getattr(attn_metadata, "block_table", None)
        if block_table is None or int(block_table.shape[0]) < batch.num_reqs:
            return
        kv_cache = getattr(attention, "kv_cache", None)
        if kv_cache is None:
            return
        block_size = additional_kwargs.get("pap_block_size")
        if block_size is None:
            block_size = getattr(getattr(attention, "impl", None), "block_size", None)
        if block_size is None:
            block_size = getattr(attention, "block_size", None)
        if block_size is None:
            return

        offload_kv_transport = PAPTensorTransport(
            os.environ.get(
                "PAP_OFFLOAD_KV_TRANSPORT",
                PAPTensorTransport.CUDA_IPC.value,
            )
        )
        if offload_kv_transport is not PAPTensorTransport.CUDA_IPC:
            raise RuntimeError("PAP paged Prefill KV export requires cuda_ipc")
        self._publish_manifests(
            request_ids=batch.request_ids,
            num_reqs=batch.num_reqs,
            num_scheduled_tokens=batch.num_scheduled_tokens,
            prefill_kv_handle_by_request=prefill_kv_handle_by_request,
            decode_capacity_tokens_by_request=decode_capacity_tokens_by_request,
            import_request_ids=import_request_ids,
            tcp_endpoint_by_request=tcp_endpoint_by_request,
            default_tcp_endpoint=default_tcp_endpoint,
            seq_lens=seq_lens,
            block_table=block_table,
            kv_cache=kv_cache,
            block_size=int(block_size),
            layout=get_kv_cache_layout(),
        )

    def _publish_manifests(
        self,
        *,
        request_ids: tuple[str, ...],
        num_reqs: int,
        num_scheduled_tokens: tuple[int, ...],
        prefill_kv_handle_by_request: dict[Any, Any],
        decode_capacity_tokens_by_request: dict[Any, Any],
        import_request_ids: set[Any],
        tcp_endpoint_by_request: dict[Any, Any],
        default_tcp_endpoint: Any,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        kv_cache: torch.Tensor,
        block_size: int,
        layout: str,
    ) -> None:
        eligible: list[tuple[int, str, str, str]] = []
        for req_index in range(num_reqs):
            if num_scheduled_tokens[req_index] <= 0:
                continue
            request_id = request_ids[req_index]
            if not is_pap_request_id(request_id):
                continue
            if request_id not in import_request_ids:
                continue
            prefill_kv_handle = prefill_kv_handle_by_request.get(request_id)
            if not prefill_kv_handle:
                continue
            tcp_endpoint = select_attention_endpoint_for_request(
                request_id,
                default_endpoint=default_tcp_endpoint,
                endpoint_by_request=tcp_endpoint_by_request,
            )
            tcp_endpoint = str(pap_endpoint_for_tp_rank(tcp_endpoint))
            eligible.append(
                (req_index, request_id, str(prefill_kv_handle), tcp_endpoint)
            )
        if not eligible:
            return

        for tcp_endpoint in dict.fromkeys(item[3] for item in eligible):
            if tcp_endpoint in self.registered_catalog_endpoints:
                continue
            status = register_prefill_kv_catalog(
                catalog_id=self.catalog_id,
                layer_name=self.layer_name,
                kv_cache=kv_cache,
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                layout=layout,
                tcp_endpoint=tcp_endpoint,
            )
            self.registered_catalog_endpoints.add(tcp_endpoint)
            logger.info(
                "PAP Prefill KV catalog %s catalog_id=%s layer=%s endpoint=%s",
                status,
                self.catalog_id,
                self.layer_name,
                tcp_endpoint,
            )
        if not self.is_last_layer:
            return
        if self.expected_layer_count <= 0:
            raise RuntimeError("sealed Prefill KV handoff requires model layer count")
        if kv_cache.device.type != "cuda":
            raise RuntimeError("sealed Prefill KV handoff requires CUDA KV cache")

        # torch.Event does not implement CUDA IPC handles in PyTorch 2.11.
        ready_event = torch.cuda.Event(interprocess=True)
        ready_event.record(torch.cuda.current_stream(kv_cache.device))
        ready_event_handle = ready_event.ipc_handle()
        seq_lens_cpu = seq_lens.detach().to(device="cpu", dtype=torch.long)
        published = 0
        for req_index, request_id, prefill_kv_handle, tcp_endpoint in eligible:
            prefix_len = int(seq_lens_cpu[req_index].item())
            if prefix_len <= 1:
                continue
            decode_capacity_tokens = decode_capacity_tokens_by_request.get(request_id)
            if decode_capacity_tokens is None:
                decode_capacity_tokens = _pap_unified_kv_export_decode_capacity_tokens()
            decode_capacity_tokens = max(0, int(decode_capacity_tokens))
            import_key = (
                request_id,
                "sealed_manifest",
                prefix_len,
                decode_capacity_tokens,
                prefill_kv_handle,
                tcp_endpoint,
            )
            if import_key in self.imported_prefill_kv:
                continue
            block_seq_len = prefix_len + decode_capacity_tokens
            block_ids = _pap_block_ids_from_block_table(
                block_table=block_table[req_index : req_index + 1],
                seq_len=block_seq_len,
                block_size=block_size,
            )
            publish_prefill_kv_session_manifest(
                request_id=request_id,
                session_handle=prefill_kv_handle,
                catalog_id=self.catalog_id,
                block_ids=block_ids,
                prefix_len=prefix_len,
                block_size=block_size,
                expected_layer_count=self.expected_layer_count,
                ready_event_handle=ready_event_handle,
                tcp_endpoint=tcp_endpoint,
                decode_capacity_tokens=decode_capacity_tokens,
            )
            self.imported_prefill_kv.add(import_key)
            self.manifest_ready_events[(request_id, prefix_len)] = ready_event
            published += 1
        if published:
            eligible_request_ids = {item[1] for item in eligible}
            published_prefixes = [
                key[1]
                for key in self.manifest_ready_events
                if key[0] in eligible_request_ids
            ]
            logger.info(
                "PAP Prefill KV manifests published catalog_id=%s requests=%d "
                "prefix_min=%d prefix_max=%d",
                self.catalog_id,
                published,
                min(published_prefixes),
                max(published_prefixes),
            )


__all__ = [
    "PAPPrefillKVPublisher",
]
