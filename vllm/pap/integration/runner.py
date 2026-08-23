# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP ownership boundary for vLLM model runners."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.config import CUDAGraphMode
from vllm.logger import init_logger
from vllm.pap.config import reject_removed_pap_flags
from vllm.pap.integration.projection import (
    build_projection_forward_context,
    select_projection_request_ids,
)
from vllm.pap.integration.request import PAPProjectionRequestStore
from vllm.pap.integration.settings import PAPRuntimeSettings
from vllm.pap.model.step_graph import (
    PAPProjectionStepGraphManager,
    PAPProjectionStepPreparation,
    prepare_projection_step_graph,
    shutdown_projection_step_graph,
)

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class PAPPreparedModelForward:
    """PAP context and optional whole-step Graph preparation for one batch."""

    additional_kwargs: dict[str, Any]
    step_preparation: PAPProjectionStepPreparation | None


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
    store: PAPProjectionRequestStore = field(default_factory=PAPProjectionRequestStore)
    _step_graph_manager: PAPProjectionStepGraphManager | None = field(
        default=None,
        init=False,
        repr=False,
    )

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
        adapter = cls(
            globally_enabled=bool(extra.get("pap_enabled", False)),
            attention_tcp_endpoint=attention_tcp_endpoint,
            block_size=int(vllm_config.cache_config.block_size),
            supports_async_sampled_tokens=supports_async_sampled_tokens,
            projection_kv_unaware=settings.projection_kv_unaware,
            debug_decision=settings.debug_decision,
            critical_trace=settings.critical_trace,
        )
        if adapter.projection_kv_unaware:
            if not supports_async_sampled_tokens:
                raise RuntimeError("PAP Projection requires the V2 model runner")
            if vllm_config.parallel_config.tensor_parallel_size != 1:
                raise RuntimeError("PAP Projection currently requires TP=1")
            if vllm_config.parallel_config.pipeline_parallel_size != 1:
                raise RuntimeError("PAP Projection currently requires PP=1")
            if vllm_config.parallel_config.decode_context_parallel_size != 1:
                raise RuntimeError("PAP Projection currently requires DCP=1")
            if vllm_config.speculative_config is not None:
                raise RuntimeError("PAP Projection does not support speculative decode")
            if vllm_config.max_concurrent_batches > 2:
                raise RuntimeError(
                    "PAP Projection currently supports at most two in-flight batches"
                )
            adapter._step_graph_manager = PAPProjectionStepGraphManager()
        return adapter

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

    def bootstrap_projection_transport(self, device: torch.device) -> None:
        """Complete the Projection NVSHMEM collective before serving requests."""
        if not self.projection_kv_unaware:
            return
        from vllm.pap.transport.nvshmem.world import get_pap_nvshmem_world

        device_index = device.index
        if device_index is None:
            device_index = torch.accelerator.current_device_index()
        try:
            buffer_bytes = int(os.environ["PAP_NVSHMEM_BUFFER_BYTES"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                "PAP Projection requires PAP_NVSHMEM_BUFFER_BYTES"
            ) from exc
        world = get_pap_nvshmem_world(
            device_index=device_index,
            buffer_bytes=buffer_bytes,
        )
        world.wait_ready()

    def remove_request(self, request_id: str) -> None:
        """Remove Projection request state."""
        self.store.remove(request_id)

    def request_ids(self, request_ids: Sequence[str]) -> frozenset[str]:
        """Return PAP-enabled request IDs in one runner batch."""
        return select_projection_request_ids(
            self.store,
            request_ids,
            globally_enabled=self.globally_enabled,
        )

    def decode_token_seq_lens(
        self,
        request_ids: Sequence[str],
        seq_lens_cpu_upper_bound: Iterable[int],
    ) -> dict[str, int]:
        """Capture frame-local sequence keys for PAP sampled tokens."""
        normalized_ids = tuple(str(request_id) for request_id in request_ids)
        pap_request_ids = self.request_ids(normalized_ids)
        if pap_request_ids and not self.supports_async_sampled_tokens:
            raise RuntimeError(
                "PAP sampled-token delivery requires the V2 model runner"
            )
        return {
            request_id: int(seq_len) + 1
            for request_id, seq_len in zip(
                normalized_ids,
                seq_lens_cpu_upper_bound,
                strict=True,
            )
            if request_id in pap_request_ids
        }

    def group_decode_request_ids(
        self,
        request_ids: Sequence[str],
        num_scheduled_tokens: Mapping[str, int],
    ) -> tuple[str, ...]:
        """Group one-token PAP requests by their Attention peer."""
        normalized_ids = tuple(str(request_id) for request_id in request_ids)
        if not normalized_ids or any(
            int(num_scheduled_tokens[request_id]) != 1 for request_id in normalized_ids
        ):
            return normalized_ids
        if self.request_ids(normalized_ids) != frozenset(normalized_ids):
            return normalized_ids

        groups: dict[str, list[str]] = {}
        for request_id in normalized_ids:
            attention_endpoint = self.store.attention_endpoint_by_request.get(
                request_id
            )
            if not attention_endpoint:
                return normalized_ids
            groups.setdefault(attention_endpoint, []).append(request_id)
        return tuple(
            request_id
            for group_request_ids in groups.values()
            for request_id in group_request_ids
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
        scheduled_tokens = tuple(int(value) for value in num_scheduled_tokens)
        pap_request_ids = self.request_ids(normalized_ids)
        pap_enabled = bool(pap_request_ids)
        if pap_enabled and not self.supports_async_sampled_tokens:
            raise RuntimeError(
                "PAP asynchronous decode-token delivery requires the V2 model "
                "runner; set VLLM_USE_V2_MODEL_RUNNER=1"
            )
        if self.projection_kv_unaware:
            if not pap_enabled:
                raise RuntimeError("PAP Projection received a non-PAP model batch")
            if pap_request_ids != frozenset(normalized_ids):
                raise RuntimeError("PAP Projection does not allow mixed model batches")
            if len(scheduled_tokens) != len(normalized_ids) or any(
                num_tokens != 1 for num_tokens in scheduled_tokens
            ):
                raise RuntimeError("PAP Projection only accepts one-token decode")
            if num_actual_tokens != len(normalized_ids):
                raise RuntimeError("PAP Projection does not allow token padding")
            if not all(
                request_id in self.store.attention_kv_installed_requests
                for request_id in normalized_ids
            ):
                raise RuntimeError("PAP Projection Attention KV is not ready")
        context = build_projection_forward_context(
            self.store,
            request_ids=normalized_ids,
            num_scheduled_tokens=scheduled_tokens,
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
                "PAP forward context enabled=%s req_ids=%s tcp_keys=%s installed=%s",
                context["pap_enabled"],
                context["pap_request_ids"][:4],
                tuple(context["pap_attention_tcp_endpoint_by_request"]),
                tuple(context["pap_attention_kv_installed_by_request"]),
            )
        return context

    def prepare_model_forward(
        self,
        *,
        request_ids: Sequence[str],
        num_scheduled_tokens: Iterable[int],
        num_actual_tokens: int,
        positions: torch.Tensor,
        seq_lens_cpu_upper_bound: Iterable[int],
        finished_request_ids: Iterable[str],
        dtype: torch.dtype,
        native_cudagraph_mode: CUDAGraphMode,
    ) -> PAPPreparedModelForward:
        """Validate and prepare one real model forward before network effects."""
        if not self.projection_kv_unaware:
            return PAPPreparedModelForward({}, None)
        normalized_ids = tuple(str(request_id) for request_id in request_ids)
        if normalized_ids and all(
            request_id.startswith("_warmup_") for request_id in normalized_ids
        ):
            return PAPPreparedModelForward(
                self.build_capture_forward_context({"positions": positions}),
                None,
            )
        if native_cudagraph_mode != CUDAGraphMode.NONE:
            raise RuntimeError(
                "PAP Projection requires native cudagraph_mode=NONE; "
                "the PAP runtime owns the outer whole-step Graph"
            )
        additional_kwargs = self.build_forward_context(
            request_ids=request_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            num_actual_tokens=num_actual_tokens,
            positions=positions,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            finished_request_ids=finished_request_ids,
        )
        preparation = prepare_projection_step_graph(additional_kwargs, dtype)
        if preparation is None:
            raise RuntimeError("PAP Projection whole-step Graph was not prepared")
        return PAPPreparedModelForward(additional_kwargs, preparation)

    def run_model_forward(
        self,
        prepared: PAPPreparedModelForward,
        *,
        model_inputs: Mapping[str, Any],
        forward: Callable[[], Any],
    ) -> Any:
        """Run a normal forward or the PAP-owned whole-step CUDA Graph."""
        preparation = prepared.step_preparation
        if preparation is None:
            return forward()
        manager = self._step_graph_manager
        if manager is None:
            raise RuntimeError("PAP Projection Graph manager is not initialized")
        inputs = tuple(_iter_cuda_tensors(model_inputs))
        return manager.run(preparation, inputs=inputs, forward=forward)

    def build_capture_forward_context(
        self,
        model_inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the complete PAP schema used by Graph capture warmup."""
        positions = model_inputs.get("positions")
        if positions is None:
            return {}
        return build_projection_forward_context(
            self.store,
            request_ids=(),
            num_scheduled_tokens=(),
            num_actual_tokens=int(positions.numel()),
            positions=positions,
            seq_lens_cpu_upper_bound=(),
            pap_enabled=False,
            attention_tcp_endpoint=self.attention_tcp_endpoint,
            block_size=self.block_size,
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
        """Release runner-owned PAP state."""
        if self._step_graph_manager is not None:
            self._step_graph_manager.shutdown()
            self._step_graph_manager = None
        shutdown_projection_step_graph()


def _iter_cuda_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            yield value
        return
    tensors = getattr(value, "tensors", None)
    if isinstance(tensors, Mapping):
        yield from _iter_cuda_tensors(tensors)
        return
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _iter_cuda_tensors(value[key])
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_cuda_tensors(item)
