from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlConnectorMetadata,
    NixlConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)


def _make_scheduler() -> NixlConnectorScheduler:
    scheduler = object.__new__(NixlConnectorScheduler)
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_save = {}
    scheduler._reqs_need_send = {}
    scheduler._reqs_in_batch = set()
    scheduler._reqs_to_finish_recv = set()
    scheduler._reqs_not_processed = set()
    scheduler._heartbeat_by_engine = {}
    scheduler._last_heartbeat_time = 0.0
    scheduler._heartbeat_interval = 10**9
    scheduler.use_host_buffer = False
    scheduler.is_bidirectional_kv_xfer_enabled = False
    return scheduler


def test_pap_attention_kv_installed_skips_projection_kv_recv() -> None:
    scheduler = _make_scheduler()
    request = SimpleNamespace(
        request_id="id-81",
        kv_transfer_params={
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_block_ids": [[1, 2]],
            "remote_engine_id": "prefill-engine",
            "remote_request_id": "prefill-id-81",
            "remote_host": "127.0.0.1",
            "remote_port": 5559,
            "pap_prefill_kv_handle": "req-81",
            "pap_attention_kv_installed": True,
        },
    )

    scheduler.update_state_after_alloc(
        request,
        blocks=SimpleNamespace(get_unhashed_block_ids_all_groups=lambda: ([7, 8],)),
        num_external_tokens=32,
    )
    metadata = scheduler.build_connector_meta(SimpleNamespace())

    assert isinstance(metadata, NixlConnectorMetadata)
    assert "id-81" not in metadata.reqs_to_recv
    assert metadata.reqs_to_finish_recv == {"id-81"}
    assert request.kv_transfer_params["_remote_blocks_processed"] is True


def test_pap_attention_kv_installed_skips_remote_decode_match() -> None:
    scheduler = _make_scheduler()
    scheduler._has_mamba = False
    scheduler.kv_recompute_threshold = 64
    request = SimpleNamespace(
        request_id="id-82",
        num_prompt_tokens=512,
        kv_transfer_params={
            "do_remote_decode": True,
            "remote_block_ids": [[1, 2, 3, 4]],
            "remote_engine_id": "prefill-engine",
            "remote_request_id": "prefill-id-82",
            "remote_host": "127.0.0.1",
            "remote_port": 5559,
            "remote_num_tokens": 512,
            "pap_attention_kv_installed": True,
        },
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)


def test_remote_decode_match_without_pap_attention_kv_installed() -> None:
    scheduler = _make_scheduler()
    scheduler._has_mamba = False
    scheduler.kv_recompute_threshold = 64
    request = SimpleNamespace(
        request_id="id-83",
        num_prompt_tokens=512,
        kv_transfer_params={
            "do_remote_decode": True,
            "remote_block_ids": [[1, 2, 3, 4]],
            "remote_engine_id": "prefill-engine",
            "remote_request_id": "prefill-id-83",
            "remote_host": "127.0.0.1",
            "remote_port": 5559,
            "remote_num_tokens": 512,
        },
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (512, True)


def test_nixl_worker_reports_pap_immediate_finished_recv() -> None:
    metadata = NixlConnectorMetadata()
    metadata.reqs_to_finish_recv.add("id-81")
    worker = object.__new__(NixlConnectorWorker)
    worker.transfer_topo = object()
    worker._recving_metadata = {}
    worker._recving_transfers = {}
    worker._failed_recv_reqs = SimpleNamespace(empty=lambda: True)
    worker._ready_requests = SimpleNamespace(empty=lambda: True)
    worker._reqs_to_process = set()
    worker._reqs_to_send = {}
    worker.consumer_notification_counts_by_req = {}
    worker.tp_rank = 0
    worker.xfer_stats = SimpleNamespace()
    worker.nixl_wrapper = SimpleNamespace(get_new_notifs=lambda: {})
    worker.use_host_buffer = False
    worker.enable_permute_local_kv = False
    worker.enable_heterogeneous_attn_post_process = False
    worker.use_mla = False

    worker.start_load_kv(metadata)
    finished_sending, finished_recving = worker.get_finished()

    assert finished_sending == set()
    assert finished_recving == {"id-81"}
