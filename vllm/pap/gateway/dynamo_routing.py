# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamo-backed PA selection for the PAP gateway."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vllm.pap.gateway.topology import PAPGroup

logger = logging.getLogger("pap_gateway")


class PAPDynamoRouter:
    """Route every PAP turn through Dynamo's in-process KV selector."""

    def __init__(
        self,
        groups: Sequence[PAPGroup],
        service: Any,
        *,
        model_name: str,
    ) -> None:
        self._groups = tuple(groups)
        self._service = service
        self._model_name = model_name
        self._reservations: set[str] = set()
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._selections = 0
        self._failures = 0

    @classmethod
    async def create(
        cls,
        groups: Sequence[PAPGroup],
        *,
        event_endpoints: Sequence[str],
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
            from dynamo.llm import SelectionService
        except ImportError as exc:
            raise RuntimeError(
                "Dynamo routing requires ai-dynamo with SelectionService"
            ) from exc

        service = SelectionService(indexer_threads=4)
        router = cls(groups, service, model_name=model_name)
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
            for worker_id, (group, event_endpoint, worker_capacity) in enumerate(
                zip(groups, event_endpoints, worker_capacities)
            ):
                worker = {
                    "worker_id": worker_id,
                    "model_name": model_name,
                    "endpoint": group.prefill_base_url,
                    "kv_events_endpoint": event_endpoint,
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
    ) -> PAPGroup:
        """Select a PA and reserve its Prefill and active Decode load."""
        try:
            selected = await self._service.select_and_reserve(
                {
                    "model_name": self._model_name,
                    "token_ids": token_ids,
                    "selection_id": request_id,
                    "reservation_id": request_id,
                    "expected_output_tokens": expected_output_tokens,
                }
            )
            worker_id = int(selected["worker_id"])
            if not 0 <= worker_id < len(self._groups):
                raise RuntimeError(f"Dynamo selected unknown PAP worker {worker_id}")
        except BaseException:
            self._failures += 1
            raise
        self._reservations.add(request_id)
        self._selections += 1
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

    async def mark_prefill_completed(self, request_id: str) -> None:
        """Move an existing reservation from Prefill to Decode load."""
        if request_id in self._reservations:
            await self._service.prefill_complete(request_id)

    def finish_request(self, request_id: str) -> None:
        """Release a reservation outside the response critical path."""
        if request_id not in self._reservations:
            return
        self._reservations.remove(request_id)
        task = asyncio.create_task(
            self._free_reservation(request_id),
            name=f"pap-dynamo-free-{request_id}",
        )
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def _free_reservation(self, request_id: str) -> None:
        try:
            await self._service.free_reservation(request_id)
        except Exception:
            self._failures += 1
            logger.exception(
                "PAP Dynamo reservation release failed request_id=%s",
                request_id,
            )

    def stats(self) -> dict[str, Any]:
        """Return selector counters and the latest Dynamo load snapshot."""
        return {
            "enabled": True,
            "selections": self._selections,
            "failures": self._failures,
            "active_reservations": len(self._reservations),
            "pending_cleanup": len(self._cleanup_tasks),
            "loads": self._service.loads(model_name=self._model_name),
        }

    async def shutdown(self) -> None:
        """Release outstanding reservations and stop the selector."""
        for request_id in tuple(self._reservations):
            self.finish_request(request_id)
        if self._cleanup_tasks:
            await asyncio.gather(*tuple(self._cleanup_tasks), return_exceptions=True)
        self._service.shutdown()


class PAPDynamoRouterDisabled:
    """No-op health surface when Dynamo routing is disabled."""

    def stats(self) -> dict[str, Any]:
        return {"enabled": False}

    async def mark_prefill_completed(self, request_id: str) -> None:
        return None

    def finish_request(self, request_id: str) -> None:
        return None

    async def shutdown(self) -> None:
        return None
