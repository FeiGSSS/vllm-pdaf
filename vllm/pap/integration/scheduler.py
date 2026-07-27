# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP ownership boundary for vLLM scheduler hooks."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Protocol

from vllm.pap.integration.migration import PAPMigrationJob, PAPMigrationStatus
from vllm.pap.integration.request import PAPRequestMetadata
from vllm.pap.integration.settings import PAPRuntimeSettings
from vllm.pap.lifecycle import lease as pap_lease


class _SchedulerRequest(Protocol):
    request_id: str
    kv_transfer_params: Mapping[str, Any] | None
    num_prompt_tokens: int


@dataclass(frozen=True, slots=True)
class PAPProjectionScheduleState:
    """Scheduler inputs for one metadata-only Projection request."""

    remote_prefix_len: int
    remote_computed_tokens: int
    local_computed_token_offset: int
    allocate_external_computed_blocks: bool = False
    allocate_local_slots: bool = False


@dataclass(slots=True)
class PAPSchedulerAdapter:
    """Translate PAP metadata and lease state for the vLLM scheduler."""

    settings: PAPRuntimeSettings
    migration_jobs: dict[str, PAPMigrationJob] = field(default_factory=dict)
    pending_migration_ids: deque[str] = field(default_factory=deque)

    @classmethod
    def from_environ(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> PAPSchedulerAdapter:
        """Create one scheduler adapter from process environment settings."""
        return cls(PAPRuntimeSettings.from_environ(environ))

    @staticmethod
    def projection_remote_prefix_len(request: _SchedulerRequest) -> int | None:
        """Validate and return the remote prompt prefix length."""
        metadata = PAPRequestMetadata.from_mapping(request.kv_transfer_params)
        if not metadata.projection_kv_unaware:
            return None
        prefix_len = metadata.remote_prefix_len
        if prefix_len is None:
            raise ValueError(
                "PAP KV-unaware Projection request requires pap_remote_prefix_len"
            )
        if prefix_len <= 0:
            raise ValueError(
                "PAP KV-unaware Projection request requires a positive "
                "pap_remote_prefix_len"
            )
        if prefix_len > request.num_prompt_tokens:
            raise ValueError(
                "PAP KV-unaware Projection prefix length cannot exceed prompt length"
            )
        return prefix_len

    @classmethod
    def projection_state(
        cls,
        request: _SchedulerRequest,
    ) -> PAPProjectionScheduleState | None:
        """Build the scheduler view of one Projection request."""
        remote_prefix_len = cls.projection_remote_prefix_len(request)
        if remote_prefix_len is None:
            return None
        remote_computed_tokens = max(remote_prefix_len - 1, 0)
        return PAPProjectionScheduleState(
            remote_prefix_len=remote_prefix_len,
            remote_computed_tokens=remote_computed_tokens,
            local_computed_token_offset=remote_computed_tokens,
        )

    def decode_capacity_tokens(self, request: _SchedulerRequest) -> int:
        """Return the local decode reservation for unified Prefill KV."""
        metadata = PAPRequestMetadata.from_mapping(request.kv_transfer_params)
        if not metadata.import_prefill_kv_to_attention:
            return 0
        if metadata.decode_capacity_tokens is not None:
            return metadata.decode_capacity_tokens
        return self.settings.unified_kv_decode_capacity_tokens

    @staticmethod
    def sweep_expired_leases() -> None:
        """Release expired process-local PAP KV leases."""
        pap_lease.pap_sweep_expired_leases()

    @staticmethod
    def evict_oldest_retained_kv_lease() -> bool:
        """Release one completed-turn lease under KV allocation pressure."""
        return pap_lease.pap_evict_oldest_retained_kv_lease() is not None

    @staticmethod
    def record_kv_export(
        *,
        request_id: str,
        seq_len: int,
        kv_transfer_params: dict[str, Any] | None,
        prefix_token_ids: tuple[int, ...] = (),
        prefix_block_hashes: tuple[bytes, ...] = (),
    ) -> bool:
        """Retain connector metadata beside its active PAP block lease."""
        if not kv_transfer_params:
            return False
        return pap_lease.pap_record_kv_export(
            request_id,
            seq_len,
            kv_transfer_params,
            prefix_token_ids,
            prefix_block_hashes,
        )

    @staticmethod
    def defer_leased_blocks(
        *,
        request_id: str,
        pop_blocks: Callable[[], list[Any]],
        free_blocks: Callable[[Any], None],
    ) -> bool:
        """Transfer block-free ownership to an active PAP lease."""
        if not pap_lease.pap_has_active_lease(request_id):
            return False

        lease_id = pap_lease.pap_active_lease_id(request_id)
        blocks = pop_blocks()
        if lease_id is None:
            free_blocks(reversed(blocks))
            return True

        blocks.reverse()
        pap_lease.pap_stash_deferred_blocks(
            lease_id=lease_id,
            blocks=blocks,
            free_callback=free_blocks,
        )
        return True

    def submit_migration(
        self,
        *,
        request_id: str,
        source_kv_params: dict[str, Any],
        prefix_len: int,
        prefix_token_ids: tuple[int, ...],
        prefix_block_hashes: tuple[bytes, ...],
        decode_capacity_tokens: int,
        session_handle: str,
        attention_tcp_endpoint: str,
    ) -> dict[str, Any]:
        """Enqueue one post-Prefill migration without creating a request."""
        required = (
            "remote_block_ids",
            "remote_engine_id",
            "remote_request_id",
            "remote_host",
            "remote_port",
        )
        missing = [key for key in required if source_kv_params.get(key) is None]
        if missing:
            raise ValueError(f"PAP migration missing NIXL fields: {missing}")
        if prefix_len <= 0:
            raise ValueError("PAP migration prefix_len must be positive")
        if decode_capacity_tokens < 0:
            raise ValueError("PAP migration Decode capacity cannot be negative")
        job = PAPMigrationJob(
            request_id=str(request_id),
            source_kv_params=dict(source_kv_params),
            prefix_len=int(prefix_len),
            prefix_token_ids=tuple(int(token) for token in prefix_token_ids),
            prefix_block_hashes=tuple(
                bytes(block_hash) for block_hash in prefix_block_hashes
            ),
            decode_capacity_tokens=int(decode_capacity_tokens),
            session_handle=str(session_handle),
            attention_tcp_endpoint=str(attention_tcp_endpoint),
        )
        self.migration_jobs[job.job_id] = job
        self.pending_migration_ids.append(job.job_id)
        return job.response()

    def migration_status(self, job_id: str) -> dict[str, Any]:
        """Return the current state of a migration job."""
        job = self.migration_jobs.get(str(job_id))
        if job is None:
            return {
                "job_id": str(job_id),
                "status": "unknown",
                "error": "unknown migration job",
            }
        return job.response()

    def migration_completion(self, job_id: str) -> Future[dict[str, Any]]:
        """Return the completion notification for one migration."""
        job = self.migration_jobs.get(str(job_id))
        if job is None:
            future: Future[dict[str, Any]] = Future()
            future.set_result(
                {
                    "job_id": str(job_id),
                    "status": "unknown",
                    "error": "unknown migration job",
                }
            )
            return future
        return job.completion

    def migration_started(self, job_id: str) -> Future[dict[str, Any]]:
        """Return the notification emitted after target blocks are assigned."""
        job = self.migration_jobs.get(str(job_id))
        if job is None:
            future: Future[dict[str, Any]] = Future()
            future.set_result(
                {
                    "job_id": str(job_id),
                    "status": "unknown",
                    "error": "unknown migration job",
                }
            )
            return future
        return job.started

    @staticmethod
    def _publish_migration_completion(job: PAPMigrationJob) -> None:
        if not job.completion.done():
            job.completion.set_result(job.response())

    @staticmethod
    def _publish_migration_started(job: PAPMigrationJob) -> None:
        if not job.started.done():
            job.started.set_result(job.response())

    @classmethod
    def _publish_migration_failure(cls, job: PAPMigrationJob) -> None:
        cls._publish_migration_started(job)
        cls._publish_migration_completion(job)

    def has_migration_work(self) -> bool:
        """Return whether a migration still needs scheduler or worker progress."""
        return any(
            job.status
            in {
                PAPMigrationStatus.PENDING,
                PAPMigrationStatus.TRANSFERRING,
            }
            for job in self.migration_jobs.values()
        )

    def attach_next_migration(
        self,
        *,
        metadata: Any,
        kv_cache_manager: Any,
        connector: Any,
        reserved_blocks: int,
    ) -> list[dict[str, Any]]:
        """Allocate and attach at most one migration to NIXL metadata."""
        if any(
            job.status is PAPMigrationStatus.TRANSFERRING
            for job in self.migration_jobs.values()
        ):
            return []
        while self.pending_migration_ids:
            job_id = self.pending_migration_ids[0]
            job = self.migration_jobs.get(job_id)
            if job is None or job.status is not PAPMigrationStatus.PENDING:
                self.pending_migration_ids.popleft()
                continue

            blocks = kv_cache_manager.allocate_external_transfer_slots(
                request_id=job.job_id,
                prefix_tokens=job.prefix_len,
                total_capacity_tokens=job.total_capacity_tokens,
                reserved_blocks=reserved_blocks,
            )
            if blocks is None:
                job.status = PAPMigrationStatus.FAILED
                job.error = "insufficient target KV capacity for PAP migration"
                job.completed_at = time.monotonic()
                self.pending_migration_ids.popleft()
                self._publish_migration_failure(job)
                return []
            all_block_ids = tuple(
                tuple(int(block_id) for block_id in group)
                for group in blocks.get_block_ids()
            )
            prefix_block_ids = []
            for group, group_config in zip(
                all_block_ids,
                kv_cache_manager.kv_cache_config.kv_cache_groups,
                strict=True,
            ):
                block_size = group_config.kv_cache_spec.block_size
                prefix_blocks = (job.prefix_len + block_size - 1) // block_size
                prefix_block_ids.append(list(group[:prefix_blocks]))
            local_block_ids = tuple(prefix_block_ids)
            try:
                connector.pap_add_migration_recv(
                    metadata,
                    request_id=job.job_id,
                    local_block_ids=local_block_ids,
                    kv_transfer_params=job.source_kv_params,
                )
                params = connector.pap_build_local_export_params(
                    request_id=job.request_id,
                    block_ids=tuple(list(group) for group in all_block_ids),
                    num_tokens=job.prefix_len,
                )
                pap_lease.pap_pin_blocks(
                    job.request_id,
                    (block_id for group in all_block_ids for block_id in group),
                )
                if not self.record_kv_export(
                    request_id=job.request_id,
                    seq_len=job.prefix_len,
                    kv_transfer_params=params,
                    prefix_token_ids=job.prefix_token_ids,
                    prefix_block_hashes=job.prefix_block_hashes,
                ):
                    raise RuntimeError(
                        "PAP migration could not establish its target KV export"
                    )
            except Exception as exc:
                kv_cache_manager.free_external_transfer_slots(job.job_id)
                lease_id = pap_lease.pap_active_lease_id(job.request_id)
                if lease_id is not None:
                    pap_lease.pap_release_lease(lease_id)
                job.status = PAPMigrationStatus.FAILED
                job.error = str(exc)
                job.completed_at = time.monotonic()
                self.pending_migration_ids.popleft()
                self._publish_migration_failure(job)
                return []

            job.block_ids = all_block_ids
            job.kv_transfer_params = params
            job.status = PAPMigrationStatus.TRANSFERRING
            job.submitted_at = time.monotonic()
            self.pending_migration_ids.popleft()
            self._publish_migration_started(job)
            return [job.manifest()]
        return []

    def finish_migration(
        self,
        *,
        job_id: str,
        kv_cache_manager: Any,
        connector: Any,
    ) -> bool:
        """Seal a worker-published migration and transfer block ownership."""
        job = self.migration_jobs.get(str(job_id))
        if job is None:
            return False
        if job.status is not PAPMigrationStatus.TRANSFERRING:
            return True

        params = job.kv_transfer_params
        if params is None:
            raise RuntimeError("PAP migration has no target KV export")
        if not pap_lease.pap_has_active_lease(job.request_id):
            raise RuntimeError("PAP migration lost its KV lease before completion")

        kv_cache_manager.cache_external_transfer_prefix(
            request_id=job.job_id,
            prefix_token_ids=job.prefix_token_ids,
            prefix_block_hashes=job.prefix_block_hashes,
        )
        blocks = kv_cache_manager.pop_external_transfer_slots(job.job_id)
        lease_id = pap_lease.pap_active_lease_id(job.request_id)
        if lease_id is None:
            kv_cache_manager.block_pool.free_blocks(reversed(blocks))
            raise RuntimeError("PAP migration has no lease for allocated blocks")
        blocks.reverse()
        pap_lease.pap_stash_deferred_blocks(
            lease_id=lease_id,
            blocks=blocks,
            free_callback=kv_cache_manager.block_pool.free_blocks,
        )
        job.kv_transfer_params = params
        job.status = PAPMigrationStatus.READY
        job.completed_at = time.monotonic()
        self._publish_migration_completion(job)
        return True

    def fail_migration(
        self,
        *,
        job_id: str,
        error: str,
        kv_cache_manager: Any,
    ) -> bool:
        """Fail one migration and release its scheduler-owned slots."""
        job = self.migration_jobs.get(str(job_id))
        if job is None:
            return False
        if job.status in {
            PAPMigrationStatus.PENDING,
            PAPMigrationStatus.TRANSFERRING,
        }:
            if job.block_ids:
                kv_cache_manager.free_external_transfer_slots(job.job_id)
            lease_id = pap_lease.pap_active_lease_id(job.request_id)
            if lease_id is not None:
                pap_lease.pap_release_lease(lease_id)
            job.status = PAPMigrationStatus.FAILED
            job.error = str(error)
            job.completed_at = time.monotonic()
            self._publish_migration_failure(job)
        return True
