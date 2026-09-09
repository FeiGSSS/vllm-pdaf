# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamo-backed PA selection for the PAP gateway."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from vllm.pap.gateway.topology import PAPGroup

logger = logging.getLogger("pap_gateway")


class _RoutingTrace:
    def __init__(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._output: TextIO = path.open("x", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def record(self, payload: dict[str, Any]) -> None:
        record = {"timestamp_ns": time.time_ns(), **payload}
        line = json.dumps(record, separators=(",", ":"), sort_keys=True)
        with self._lock:
            self._output.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            self._output.close()


class PAPDynamoRouter:
    """Route every PAP turn through Dynamo's in-process KV selector."""

    def __init__(
        self,
        groups: Sequence[PAPGroup],
        service: Any,
        *,
        model_name: str,
        block_size: int,
    ) -> None:
        self._groups = tuple(groups)
        self._service = service
        self._model_name = model_name
        self._block_size = block_size
        self._reservations: set[str] = set()
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._releasing: set[str] = set()
        self._selections = 0
        self._failures = 0
        self._fence_checks = 0
        self._fence_pins = 0
        self._pending_selections: dict[str, asyncio.Task[PAPGroup]] = {}
        self._closed = False
        self._healthy = True
        trace_output = os.environ.get("PAP_DYNAMO_ROUTING_TRACE_OUTPUT")
        self._trace = _RoutingTrace(trace_output) if trace_output else None

    @classmethod
    async def create(
        cls,
        groups: Sequence[PAPGroup],
        *,
        event_endpoints: Sequence[str],
        replay_endpoints: Sequence[str],
        site_packages: str,
        model_name: str,
        block_size: int,
        max_num_batched_tokens: int,
        total_kv_blocks: int | Sequence[int | None] | None,
        prefill_load_scale: float,
    ) -> PAPDynamoRouter:
        """Create a selector and register each PA as one Dynamo worker."""
        if len(event_endpoints) != len(groups):
            raise ValueError("Dynamo KV event endpoint count must match PA count")
        if len(replay_endpoints) != len(groups):
            raise ValueError("Dynamo KV replay endpoint count must match PA count")
        if prefill_load_scale <= 0:
            raise ValueError("Dynamo Prefill load scale must be positive")
        os.environ.update(
            {
                "DYN_USE_KV_EVENTS": "true",
                "DYN_ROUTER_TRACK_ACTIVE_BLOCKS": "true",
                "DYN_ROUTER_TRACK_PREFILL_TOKENS": "true",
                "DYN_ROUTER_TRACK_OUTPUT_BLOCKS": "false",
                "DYN_ROUTER_ASSUME_KV_REUSE": "true",
                "DYN_ROUTER_PREFILL_LOAD_SCALE": str(prefill_load_scale),
            }
        )
        path = Path(site_packages)
        if not path.is_dir():
            raise RuntimeError(f"Dynamo site-packages does not exist: {path}")
        site_packages_text = str(path)
        if site_packages_text not in sys.path:
            sys.path.append(site_packages_text)

        try:
            from pap_dynamo_router import SelectionService
        except ImportError as exc:
            raise RuntimeError(
                "PAP requires its explicit-owner Dynamo dependency; run "
                "bash benchmarks/pap/scripts/build_pap_dynamo_router.sh"
            ) from exc

        service = SelectionService(indexer_threads=4)
        if service.reservation_lifetime != "explicit-owner-v1":
            service.shutdown()
            raise RuntimeError("PAP requires explicit-owner Dynamo reservations")
        router = cls(groups, service, model_name=model_name, block_size=block_size)
        try:
            worker_capacities: list[int | None]
            if total_kv_blocks is None:
                worker_capacities = [None] * len(groups)
            elif isinstance(total_kv_blocks, int):
                worker_capacities = [total_kv_blocks] * len(groups)
            else:
                worker_capacities = list(total_kv_blocks)
                if len(worker_capacities) != len(groups):
                    raise ValueError("Dynamo KV capacity count must match PA count")
            for worker_id, (
                group,
                event_endpoint,
                replay_endpoint,
                worker_capacity,
            ) in enumerate(
                zip(groups, event_endpoints, replay_endpoints, worker_capacities)
            ):
                worker = {
                    "worker_id": worker_id,
                    "model_name": model_name,
                    "endpoint": group.prefill_base_url,
                    "kv_events_endpoint": event_endpoint,
                    "replay_endpoint": replay_endpoint,
                    "block_size": block_size,
                    "max_num_batched_tokens": max_num_batched_tokens,
                }
                if worker_capacity is not None:
                    worker["total_kv_blocks"] = worker_capacity
                record = await service.upsert_worker(worker)
                if record["lifecycle"] != "schedulable":
                    raise RuntimeError(
                        "Dynamo PA worker is not schedulable: "
                        f"worker={worker_id} record={record}"
                    )
            ready = service.ready()
            if not ready["ready"] or ready["schedulable_workers"] != len(groups):
                raise RuntimeError(f"Dynamo selector is not ready: {ready}")
        except BaseException:
            service.shutdown()
            raise
        return router

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
        """Select a PA and reserve its Prefill and active Decode load."""
        if self._closed or not self._healthy:
            raise RuntimeError("Dynamo router is closed or has failed load accounting")
        if request_id in self._reservations or request_id in self._pending_selections:
            raise ValueError(f"Duplicate Dynamo request ID: {request_id}")
        task = asyncio.create_task(
            self._select_group(
                token_ids,
                request_id,
                expected_output_tokens,
                conversation_id,
                verified_worker_id,
                verified_prefix_tokens,
            ),
            name=f"pap-dynamo-select-{request_id}",
        )
        self._pending_selections[request_id] = task
        task.add_done_callback(lambda _: self._pending_selections.pop(request_id, None))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(
                self._finish_cancelled_selection(task, request_id)
            )
            self._cleanup_tasks.add(cleanup)
            cleanup.add_done_callback(self._cleanup_tasks.discard)
            raise

    async def _finish_cancelled_selection(
        self, task: asyncio.Task[PAPGroup], request_id: str
    ) -> None:
        try:
            await task
        except BaseException:
            return  # Failed bookings are cleaned up by _select_group.
        self.finish_request(request_id)

    async def _select_group(
        self,
        token_ids: list[int],
        request_id: str,
        expected_output_tokens: int,
        conversation_id: str,
        verified_worker_id: int | None,
        verified_prefix_tokens: int,
    ) -> PAPGroup:
        trace = self._trace
        observation: dict[str, Any] | None = None
        overlap_scores: dict[str, Any] | None = None
        if trace is not None or verified_worker_id is not None:
            probe_start_ns = time.perf_counter_ns()
            overlap_scores = await self._service.overlap_scores(
                {"model_name": self._model_name, "token_ids": token_ids}
            )
        if trace is not None:
            loads = self._service.loads(model_name=self._model_name)
            assert overlap_scores is not None
            observation = {
                "event": "selection",
                "request_id": request_id,
                "conversation_id": conversation_id,
                "prompt_tokens": len(token_ids),
                "expected_output_tokens": expected_output_tokens,
                "probe_start_ns": probe_start_ns,
                "probe_done_ns": time.perf_counter_ns(),
                "overlap_scores": overlap_scores,
                "loads_before": loads,
            }
        selection_request: dict[str, Any] = {
            "model_name": self._model_name,
            "token_ids": token_ids,
            "selection_id": request_id,
            "expected_output_tokens": expected_output_tokens,
        }
        fence_applied = False
        if verified_worker_id is not None:
            if not 0 <= verified_worker_id < len(self._groups):
                raise ValueError("verified Dynamo worker is out of range")
            if verified_prefix_tokens <= 0 or verified_prefix_tokens % self._block_size:
                raise ValueError(
                    "verified Dynamo prefix must be positive and block aligned"
                )
            assert overlap_scores is not None
            if int(overlap_scores["block_size"]) != self._block_size:
                raise RuntimeError("Dynamo overlap block size changed")
            indexed_blocks = {
                int(row["worker_id"]): int(row["device_blocks"])
                for row in overlap_scores["workers"]
            }
            if verified_worker_id not in indexed_blocks:
                raise RuntimeError(
                    "verified Dynamo worker is absent from overlap scores"
                )
            verified_blocks = verified_prefix_tokens // self._block_size
            max_indexed_blocks = max(indexed_blocks.values(), default=0)
            self._fence_checks += 1
            if verified_blocks > max_indexed_blocks:
                selection_request["pinned_worker"] = {
                    "worker_id": verified_worker_id,
                    "dp_rank": 0,
                }
                self._fence_pins += 1
                fence_applied = True
            logger.info(
                "PAP Dynamo KV fence request_id=%s source_pa=%d "
                "verified_blocks=%d indexed_source_blocks=%d "
                "max_indexed_blocks=%d pinned=%s",
                request_id,
                verified_worker_id,
                verified_blocks,
                indexed_blocks[verified_worker_id],
                max_indexed_blocks,
                fence_applied,
            )
            if observation is not None:
                observation["kv_fence"] = {
                    "source_worker_id": verified_worker_id,
                    "verified_prefix_tokens": verified_prefix_tokens,
                    "indexed_source_blocks": indexed_blocks[verified_worker_id],
                    "max_indexed_blocks": max_indexed_blocks,
                    "pinned": fence_applied,
                }
        selection_start_ns = time.perf_counter_ns()
        try:
            selected = await self._service.select_and_reserve(selection_request)
            worker_id = int(selected["worker_id"])
            if not 0 <= worker_id < len(self._groups):
                raise RuntimeError(f"Dynamo selected unknown PAP worker {worker_id}")
        except BaseException as exc:
            self._failures += 1
            # A failed response must not orphan a booking that already committed.
            try:
                await self._service.free_reservation(request_id)
            except KeyError:
                pass  # Selection failed before creating the reservation.
            except Exception:
                self._healthy = False
                logger.exception("Failed to unwind Dynamo selection %s", request_id)
            if observation is not None and trace is not None:
                observation.update(
                    {
                        "selection_start_ns": selection_start_ns,
                        "selection_done_ns": time.perf_counter_ns(),
                        "selection_error": repr(exc),
                    }
                )
                trace.record(observation)
            raise
        self._reservations.add(request_id)
        self._selections += 1
        if observation is not None and trace is not None:
            selected_trace = {
                key: value
                for key, value in selected.items()
                if key != "sequence_hashes"
            }
            observation.update(
                {
                    "selection_start_ns": selection_start_ns,
                    "selection_done_ns": time.perf_counter_ns(),
                    "selected": selected_trace,
                }
            )
            trace.record(observation)
        logger.info(
            "PAP Dynamo placement request_id=%s selected_pa=%d "
            "prompt_tokens=%d effective_prefill_tokens=%s overlap=%s",
            request_id,
            worker_id,
            len(token_ids),
            selected.get("effective_prefill_tokens"),
            selected.get("overlap"),
        )
        return self._groups[worker_id]

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
        if self._trace is None:
            return
        self._trace.record(
            {
                "event": "prefill_result",
                "request_id": request_id,
                "conversation_id": conversation_id,
                "worker_id": worker_id,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "prefill_ms": prefill_ms,
            }
        )

    async def mark_prefill_completed(self, request_id: str) -> None:
        """Move an existing reservation from Prefill to Decode load."""
        if request_id in self._reservations:
            await self._service.prefill_complete(request_id)

    def finish_request(self, request_id: str) -> None:
        """Release a reservation outside the response critical path."""
        if request_id not in self._reservations:
            return
        if request_id in self._releasing:
            return
        self._releasing.add(request_id)
        task = asyncio.create_task(
            self._free_reservation(request_id),
            name=f"pap-dynamo-free-{request_id}",
        )
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def _free_reservation(self, request_id: str) -> None:
        try:
            await self._service.free_reservation(request_id)
            self._reservations.discard(request_id)
        except Exception:
            self._failures += 1
            self._healthy = False
            logger.exception(
                "PAP Dynamo reservation release failed request_id=%s",
                request_id,
            )
        finally:
            self._releasing.discard(request_id)

    def stats(self) -> dict[str, Any]:
        """Return selector counters and the latest Dynamo load snapshot."""
        return {
            "enabled": True,
            "selections": self._selections,
            "failures": self._failures,
            "kv_fence_checks": self._fence_checks,
            "kv_fence_pins": self._fence_pins,
            "active_reservations": len(self._reservations),
            "pending_cleanup": len(self._cleanup_tasks),
            "pending_selections": len(self._pending_selections),
            "healthy": self._healthy,
            "reservation_lifetime": "explicit-owner-v1",
            "loads": []
            if self._closed
            else self._service.loads(model_name=self._model_name),
        }

    async def shutdown(self) -> None:
        """Release outstanding reservations and stop the selector."""
        self._closed = True
        self._service.stop_scheduling()
        if self._pending_selections:
            await asyncio.gather(
                *tuple(self._pending_selections.values()), return_exceptions=True
            )
        for request_id in tuple(self._reservations):
            self.finish_request(request_id)
        while self._cleanup_tasks:
            await asyncio.gather(*tuple(self._cleanup_tasks), return_exceptions=True)
        self._service.shutdown()
        if self._trace is not None:
            self._trace.close()
            self._trace = None
