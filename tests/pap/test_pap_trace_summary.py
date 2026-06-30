from pathlib import Path

from vllm.pap.trace_summary import summarize_pap_trace_logs


def test_trace_summary_extracts_projection_attention_and_mailbox_stats(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "service_logs"
    log_dir.mkdir()
    (log_dir / "projection_0.log").write_text(
        "PAP OFFLOAD_EXEC projection trace layer=model.layers.1.self_attn.attn "
        "ubatch_id=2 batches=1 calls=1 send_ms=0.030 trigger_ms=0.000 "
        "yield_ms=0.200 recv_ms=0.700 total_ms=0.950 batch_keys=abc "
        "send_done_ns=1000000000 yield_start_ns=1000010000 "
        "yield_end_ns=1005000000 recv_done_ns=1005800000\n"
        "PAP OFFLOAD_EXEC projection timeline "
        "layer=model.layers.1.self_attn.attn ubatch_id=2 batches=1 calls=1 "
        "pre_attn_compute_ms=0.400 send_ms=0.030 trigger_ms=0.000 "
        "yield_ms=0.200 recv_ms=0.700 o_proj_ms=0.300 "
        "remote_total_ms=0.950 self_attn_total_ms=1.750 batch_keys=abc "
        "pre_attn_start_ns=999000000 pre_attn_done_ns=999400000 "
        "send_done_ns=1000000000 yield_start_ns=1000010000 "
        "yield_end_ns=1005000000 recv_done_ns=1005800000 "
        "o_proj_done_ns=1006500000\n"
        "PAP OFFLOAD_EXEC projection layer timeline "
        "layer=model.layers.1.self_attn.attn ubatch_id=2 "
        "input_norm_ms=0.100 self_attn_ms=1.750 "
        "post_attention_layernorm_ms=0.080 mlp_ms=0.600 "
        "layer_total_ms=2.600 layer_start_ns=998900000 "
        "input_norm_done_ns=999000000 self_attn_done_ns=1006500000 "
        "post_norm_done_ns=1006580000 mlp_done_ns=1007180000\n"
        "PAP NIXL mailbox send trace actor=projection msg_id=x "
        "kind=attention_task_batch nbytes=8192 queue_ms=0.020 publish_ms=0.040 "
        "pack_ms=0.006 copy_ms=0.018 notify_ms=0.013 ack_wait_ms=0.250 "
        "total_ms=0.310\n"
        "PAP NIXL mailbox read trace actor=projection msg_id=y "
        "kind=attention_result_batch nbytes=4096 prepare_ms=0.009 transfer_ms=0.070 "
        "transfer_polls=15 materialize_ms=0.009 total_ms=0.088\n"
        "PAP NIXL mailbox recv wait trace actor=projection msg_id=y "
        "kind=attention_result_batch requested_msg_id=y wait_ms=0.410\n"
    )
    (log_dir / "attention_0.log").write_text(
        "PAP OFFLOAD_EXEC attention mailbox batch trace "
        "layer=model.layers.1.self_attn.attn calls=1 recv_qkv_ms=0.820 "
        "compute_ms=0.140 send_output_ms=0.010 total_ms=0.970 "
        "append_kv_ms=0.050 pack_ms=0.030 sdpa_ms=0.040 reshape_ms=0.020 "
        "paged_metadata_ms=0.060 paged_flash_ms=0.070 "
        "shape_lookup_ms=0.011 qkv_split_ms=0.012 query_move_ms=0.013 "
        "query_cat_ms=0.014 append_lock_wait_ms=0.015 "
        "append_prepare_ms=0.016 append_record_ms=0.017 "
        "append_tensor_ms=0.018 append_copy_ms=0.019 "
        "append_state_ms=0.021 "
        "qkv_shape=(1, 4096) output_shape=(1, 2048) batch_key=abc "
        "recv_done_ns=1003900000 compute_done_ns=1004100000 "
        "send_done_ns=1004200000\n"
        "PAP NIXL mailbox send trace actor=attention msg_id=z "
        "kind=attention_result_batch nbytes=4096 queue_ms=0.060 publish_ms=0.043 "
        "pack_ms=0.008 copy_ms=0.019 notify_ms=0.013 ack_wait_ms=0.230 "
        "total_ms=0.350\n"
        "PAP NIXL mailbox read trace actor=attention msg_id=w "
        "kind=attention_task_batch nbytes=8192 prepare_ms=0.009 transfer_ms=0.080 "
        "transfer_polls=10 materialize_ms=0.009 total_ms=0.098\n"
        "PAP NIXL mailbox recv wait trace actor=attention msg_id=w "
        "kind=attention_task_batch requested_msg_id= wait_ms=0.600\n"
    )

    summary = summarize_pap_trace_logs(log_dir)

    assert summary["projection_trace"]["recv_ms"].median == 0.700
    assert summary["projection_trace"]["yield_ms"].median == 0.200
    assert abs(summary["projection_trace"]["gap_ms"].median - 0.020) < 1e-9
    assert summary["projection_trace"]["batches"].median == 1
    assert summary["projection_timeline"]["ubatch_id"].median == 2
    assert summary["projection_timeline"]["pre_attn_compute_ms"].median == 0.400
    assert summary["projection_timeline"]["o_proj_ms"].median == 0.300
    assert summary["projection_timeline"]["self_attn_total_ms"].median == 1.750
    assert summary["projection_layer_timeline"]["input_norm_ms"].median == 0.100
    assert summary["projection_layer_timeline"]["mlp_ms"].median == 0.600
    assert summary["projection_layer_timeline"]["layer_total_ms"].median == 2.600
    assert summary["attention_trace"]["compute_ms"].median == 0.140
    assert summary["attention_trace"]["append_kv_ms"].median == 0.050
    assert summary["attention_trace"]["pack_ms"].median == 0.030
    assert summary["attention_trace"]["sdpa_ms"].median == 0.040
    assert summary["attention_trace"]["reshape_ms"].median == 0.020
    assert summary["attention_trace"]["paged_metadata_ms"].median == 0.060
    assert summary["attention_trace"]["paged_flash_ms"].median == 0.070
    assert summary["attention_trace"]["shape_lookup_ms"].median == 0.011
    assert summary["attention_trace"]["qkv_split_ms"].median == 0.012
    assert summary["attention_trace"]["query_move_ms"].median == 0.013
    assert summary["attention_trace"]["query_cat_ms"].median == 0.014
    assert summary["attention_trace"]["append_lock_wait_ms"].median == 0.015
    assert summary["attention_trace"]["append_prepare_ms"].median == 0.016
    assert summary["attention_trace"]["append_record_ms"].median == 0.017
    assert summary["attention_trace"]["append_tensor_ms"].median == 0.018
    assert summary["attention_trace"]["append_copy_ms"].median == 0.019
    assert summary["attention_trace"]["append_state_ms"].median == 0.021
    assert summary["attention_trace"]["calls"].median == 1
    assert (
        summary["projection_attention_correlation"][
            "attention_path_after_projection_send_ms"
        ].median
        == 4.200
    )
    assert (
        summary["projection_attention_correlation"][
            "projection_resume_after_attention_ready_ms"
        ].median
        == 0.800
    )
    assert (
        summary["projection_attention_correlation"][
            "attention_ready_after_projection_resume_ms"
        ].median
        == 0.0
    )
    assert (
        summary["projection_attention_correlation"][
            "projection_resume_to_recv_done_ms"
        ].median
        == 0.800
    )
    assert summary["mailbox_send"]["projection"]["ack_wait_ms"].median == 0.250
    assert summary["mailbox_send"]["attention"]["total_ms"].median == 0.350
    assert summary["mailbox_read"]["attention"]["transfer_ms"].median == 0.080
    assert summary["mailbox_read"]["projection"]["total_ms"].median == 0.088
    assert (
        summary["mailbox_send_by_kind"]["projection:attention_task_batch"][
            "total_ms"
        ].median
        == 0.310
    )
    assert (
        summary["mailbox_read_by_kind"]["attention:attention_task_batch"][
            "total_ms"
        ].median
        == 0.098
    )
    assert (
        summary["mailbox_wait_by_kind"]["attention:attention_task_batch"][
            "wait_ms"
        ].median
        == 0.600
    )
