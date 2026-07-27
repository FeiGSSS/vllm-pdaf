# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    kv_transfer_state,
)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.pap.integration.migration import validate_pap_migration_tp_size
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class KVConnector:
    """KVConnector interface used by GPUModelRunner."""

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        pass

    def post_forward(
        self, finished_req_ids: set[str], wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        return None

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        return EMPTY_MODEL_RUNNER_OUTPUT

    def set_disabled(self, disabled: bool) -> None:
        pass


class ActiveKVConnector(KVConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_caches_dict: dict[str, torch.Tensor],
        cross_layers_kv_cache: torch.Tensor | None = None,
        cross_layers_attn_backend: type["AttentionBackend"] | None = None,
    ):
        self.vllm_config = vllm_config
        self._pap_migrations: dict[str, dict[str, Any]] = {}
        self._pap_static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )
        self._pap_progress_lock = threading.RLock()
        self._pap_progress_thread: threading.Thread | None = None
        self._pap_background_finished_sending: set[str] = set()
        self._pap_background_finished_recving: set[str] = set()
        self._pap_background_invalid_block_ids: set[int] = set()
        self._pap_background_failures: dict[str, str] = {}
        self._pap_background_progress_enabled = (
            vllm_config.parallel_config.tensor_parallel_size == 1
        )
        self.kv_connector = get_kv_transfer_group()
        if cross_layers_kv_cache is None:
            self.kv_connector.register_kv_caches(kv_caches_dict)
        else:
            assert cross_layers_attn_backend is not None
            self.kv_connector.register_cross_layers_kv_cache(
                cross_layers_kv_cache,
                cross_layers_attn_backend,
            )
        self.kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)

        self._disabled = False

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        if self._disabled:
            return

        migration_manifests = scheduler_output.pap_migration_manifests or ()
        if migration_manifests:
            validate_pap_migration_tp_size(
                self.vllm_config.parallel_config.tensor_parallel_size
            )
        with self._pap_progress_lock:
            for raw_migration in migration_manifests:
                migration = dict(raw_migration)
                migration["block_size"] = self.vllm_config.cache_config.block_size
                self._pap_migrations[str(migration["job_id"])] = migration
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            assert kv_connector_metadata is not None
            self.kv_connector.handle_preemptions(kv_connector_metadata)
            self.kv_connector.bind_connector_metadata(kv_connector_metadata)

            # TODO: sort out KV Connectors' use of forward_context
            if is_forward_context_available():
                self.kv_connector.start_load_kv(get_forward_context())
            else:
                with set_forward_context(None, self.vllm_config):
                    self.kv_connector.start_load_kv(get_forward_context())
        self._start_pap_background_progress()

    def _publish_finished_migrations(
        self,
        finished_recving: set[str],
        invalid_block_ids: set[int],
    ) -> dict[str, str]:
        completed_migrations = {
            request_id: self._pap_migrations.pop(request_id)
            for request_id in finished_recving
            if request_id in self._pap_migrations
        }
        if not completed_migrations:
            return {}

        from vllm.pap.model.prefill import publish_completed_migrations

        failures: dict[str, str] = {}
        for request_id, migration in completed_migrations.items():
            migration_blocks = {
                int(block_id) for group in migration["block_ids"] for block_id in group
            }
            if migration_blocks & invalid_block_ids:
                failures[request_id] = "NIXL failed to load migration blocks"
                continue
            try:
                publish_completed_migrations(
                    self._pap_static_forward_context,
                    (migration,),
                )
            except Exception as exc:
                failures[request_id] = str(exc)
        return failures

    def _start_pap_background_progress(self) -> None:
        if not self._pap_background_progress_enabled or not self._pap_migrations:
            return
        thread = self._pap_progress_thread
        if thread is not None and thread.is_alive():
            return
        self._pap_progress_thread = threading.Thread(
            target=self._progress_pap_migrations,
            name="pap-nixl-migration-progress",
            daemon=True,
        )
        self._pap_progress_thread.start()

    def _progress_pap_migrations(self) -> None:
        while True:
            time.sleep(0.001)
            with self._pap_progress_lock:
                if self._disabled or not self._pap_migrations:
                    return
                try:
                    progress_ready = getattr(
                        self.kv_connector,
                        "pap_progress_ready_recvs",
                        None,
                    )
                    if progress_ready is not None:
                        progress_ready()
                    finished_sending, finished_recving = self.kv_connector.get_finished(
                        set()
                    )
                    invalid_block_ids = (
                        self.kv_connector.get_block_ids_with_load_errors()
                    )
                    failures = self._publish_finished_migrations(
                        finished_recving,
                        invalid_block_ids,
                    )
                except Exception as exc:
                    logger.exception("PAP background NIXL migration progress failed")
                    failed = set(self._pap_migrations)
                    self._pap_migrations.clear()
                    self._pap_background_finished_recving.update(failed)
                    self._pap_background_failures.update(
                        {request_id: str(exc) for request_id in failed}
                    )
                    return
                self._pap_background_finished_sending.update(finished_sending)
                self._pap_background_finished_recving.update(finished_recving)
                self._pap_background_invalid_block_ids.update(invalid_block_ids)
                self._pap_background_failures.update(failures)

    def post_forward(
        self, finished_req_ids: set[str], wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        if self._disabled:
            return None

        output = KVConnectorOutput()
        with self._pap_progress_lock:
            if wait_for_save:
                self.kv_connector.wait_for_save()
            finished_sending, finished_recving = self.kv_connector.get_finished(
                finished_req_ids
            )
            invalid_block_ids = self.kv_connector.get_block_ids_with_load_errors()
            failures = self._publish_finished_migrations(
                finished_recving,
                invalid_block_ids,
            )
            finished_sending.update(self._pap_background_finished_sending)
            finished_recving.update(self._pap_background_finished_recving)
            invalid_block_ids.update(self._pap_background_invalid_block_ids)
            failures.update(self._pap_background_failures)
            self._pap_background_finished_sending.clear()
            self._pap_background_finished_recving.clear()
            self._pap_background_invalid_block_ids.clear()
            self._pap_background_failures.clear()

            output.finished_sending = finished_sending
            output.finished_recving = finished_recving
            output.invalid_block_ids = invalid_block_ids
            output.pap_migration_failures = failures or None
            output.kv_connector_stats = self.kv_connector.get_kv_connector_stats()
            output.kv_cache_events = (
                self.kv_connector.get_kv_connector_kv_cache_events()
            )
            output.kv_connector_worker_meta = (
                self.kv_connector.build_connector_worker_meta()
            )
            self.kv_connector.clear_connector_metadata()
        return output

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        if self._disabled:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self.pre_forward(scheduler_output)
        finished_req_ids = scheduler_output.finished_req_ids
        kv_connector_output = self.post_forward(finished_req_ids, wait_for_save=False)
        return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

    def set_disabled(self, disabled: bool) -> None:
        # Ensure that layer-wise connector hooks aren't called when disabled.
        kv_transfer_state._KV_CONNECTOR_AGENT = None if disabled else self.kv_connector
        self._disabled = disabled


NO_OP_KV_CONNECTOR = KVConnector()


def get_kv_connector(
    vllm_config: VllmConfig,
    kv_caches_dict: dict[str, torch.Tensor],
    cross_layers_kv_cache: torch.Tensor | None = None,
    cross_layers_attn_backend: type["AttentionBackend"] | None = None,
) -> KVConnector:
    if not has_kv_transfer_group():
        # No-op connector.
        return NO_OP_KV_CONNECTOR

    return ActiveKVConnector(
        vllm_config,
        kv_caches_dict,
        cross_layers_kv_cache,
        cross_layers_attn_backend,
    )
