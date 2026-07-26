from pathlib import Path

from benchmarks.pap.tooling.trace_summary import summarize_pap_trace_logs


def test_trace_summary_extracts_projection_attention_and_mailbox_stats(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "service_logs"
    log_dir.mkdir()
    (log_dir / "projection_0.log").write_text(
        "PAP OFFLOAD_EXEC projection trace layer=model.layers.1.self_attn.attn "
        "batches=1 calls=1 send_ms=0.030 trigger_ms=0.000 "
        "yield_ms=0.200 recv_ms=0.700 total_ms=0.950 batch_keys=abc "
        "send_done_ns=1000000000 yield_start_ns=1000010000 "
        "yield_end_ns=1005000000 recv_done_ns=1005800000\n"
        "PAP OFFLOAD_EXEC projection fan-in trace "
        "layer=model.layers.1.self_attn.attn peers=3 "
        "first_ready_ms=4.000 last_ready_ms=10.000 "
        "spread_ms=6.000 spread_over_fastest_pct=150.000\n"
        "PAP OFFLOAD_EXEC projection timeline "
        "layer=model.layers.1.self_attn.attn batches=1 calls=1 "
        "pre_attn_compute_ms=0.400 send_ms=0.030 trigger_ms=0.000 "
        "yield_ms=0.200 recv_ms=0.700 o_proj_ms=0.300 "
        "remote_total_ms=0.950 self_attn_total_ms=1.750 batch_keys=abc "
        "pre_attn_start_ns=999000000 pre_attn_done_ns=999400000 "
        "send_done_ns=1000000000 yield_start_ns=1000010000 "
        "yield_end_ns=1005000000 recv_done_ns=1005800000 "
        "o_proj_done_ns=1006500000\n"
        "PAP OFFLOAD_EXEC projection layer timeline "
        "layer=model.layers.1.self_attn.attn "
        "input_norm_ms=0.100 self_attn_ms=1.750 "
        "post_attention_layernorm_ms=0.080 mlp_ms=0.600 "
        "layer_total_ms=2.600 layer_start_ns=998900000 "
        "input_norm_done_ns=999000000 self_attn_done_ns=1006500000 "
        "post_norm_done_ns=1006580000 mlp_done_ns=1007180000\n"
        "PAP OFFLOAD_EXEC projection critical path "
        "layer=model.layers.1.self_attn.attn batch_key=abc calls=1 "
        "input_norm_ms=0.100 qkv_ms=0.400 send_ms=0.030 "
        "recv_ms=0.700 o_proj_ms=0.300 post_norm_ms=0.080 "
        "mlp_ms=0.600 layer_total_ms=2.600 gaps_ms=0.390 "
        "layer_start_ns=998900000 input_norm_done_ns=999000000 "
        "qkv_done_ns=999400000 send_done_ns=1000000000 "
        "recv_done_ns=1005800000 o_proj_done_ns=1006500000 "
        "post_norm_done_ns=1006580000 mlp_done_ns=1007180000\n"
        "PAP OFFLOAD_EXEC projection model forward "
        "num_tokens=1 model_forward_ms=93.600 model_start_ns=900000000 "
        "model_done_ns=993600000\n"
        "PAP OFFLOAD_EXEC projection logits "
        "num_tokens=1 logits_ms=1.400 logits_start_ns=993600000 "
        "logits_done_ns=995000000\n"
        "PAP OFFLOAD_EXEC projection runner forward detail "
        "num_tokens=1 input_prep_ms=0.200 metadata_ms=0.300 "
        "preprocess_ms=0.400 model_forward_ms=90.000 "
        "hidden_slice_ms=0.050 logits_ms=1.200 "
        "postprocess_tail_ms=0.150 total_ms=92.300 "
        "input_prep_start_ns=901000000 input_prep_done_ns=901200000 "
        "metadata_done_ns=901500000 preprocess_done_ns=901900000 "
        "model_forward_done_ns=991900000 hidden_slice_done_ns=991950000 "
        "logits_done_ns=993150000 postprocess_done_ns=993300000\n"
        "PAP OFFLOAD_EXEC projection first output "
        "request_id=abc generated_tokens=1 sched_ms=0.100 "
        "exec_and_sample_ms=93.600 scheduler_update_ms=0.500 "
        "step_to_first_output_ms=94.200 step_start_ns=900000000 "
        "sched_done_ns=900100000 model_done_ns=993700000 "
        "first_output_done_ns=994200000\n"
        "PAP NIXL mailbox send trace actor=projection msg_id=x "
        "kind=attention_task_batch nbytes=8192 queue_ms=0.020 publish_ms=0.040 "
        "pack_ms=0.006 slot_wait_ms=0.005 copy_ms=0.018 payload_ms=0.004 "
        "piggyback_ms=0.002 notify_ms=0.013 write_ms=0.033 "
        "write_prepare_ms=0.011 write_transfer_ms=0.020 write_polls=3 "
        "ack_wait_ms=0.250 total_ms=0.310\n"
        "PAP NIXL mailbox read trace actor=projection msg_id=y "
        "kind=attention_result_batch nbytes=4096 prepare_ms=0.009 "
        "slot_wait_ms=0.002 handle_prepare_ms=0.006 transfer_ms=0.070 "
        "transfer_polls=15 materialize_ms=0.009 total_ms=0.088\n"
        "PAP NIXL mailbox recv wait trace actor=projection msg_id=y "
        "kind=attention_result_batch requested_msg_id=y wait_ms=0.410\n"
        "PAP NIXL mailbox inline send trace actor=inline_projection "
        "msg_id=inline-x "
        "kind=attention_task_inline nbytes=8192 publish_ms=0.050 "
        "pack_ms=0.006 slot_wait_ms=0.005 copy_ms=0.018 "
        "payload_ms=0.004 piggyback_ms=0.002 notify_ms=0.013 "
        "write_ms=0.033 write_prepare_ms=0.011 write_transfer_ms=0.020 "
        "write_polls=3 total_ms=0.060\n"
    )
    (log_dir / "attention_0.log").write_text(
        "PAP OFFLOAD_EXEC attention mailbox batch trace "
        "layer=model.layers.1.self_attn.attn calls=1 recv_qkv_ms=0.820 "
        "compute_ms=0.140 send_output_ms=0.010 total_ms=0.970 "
        "recv_wait_ms=0.600 recv_read_ms=0.098 recv_materialize_ms=0.009 "
        "recv_transfer_ms=0.080 recv_wait_other_ms=0.502 "
        "recv_unaccounted_ms=0.220 "
        "append_kv_ms=0.050 pack_ms=0.030 sdpa_ms=0.040 reshape_ms=0.020 "
        "paged_metadata_ms=0.060 paged_flash_ms=0.070 "
        "shape_lookup_ms=0.011 qkv_split_ms=0.012 query_move_ms=0.013 "
        "query_cat_ms=0.014 append_lock_wait_ms=0.015 "
        "append_prepare_ms=0.016 append_record_ms=0.017 "
        "append_tensor_ms=0.018 append_copy_ms=0.019 "
        "append_state_ms=0.021 metadata_build_ms=0.061 "
        "paged_flash_kernel_ms=0.071 attention_output_reshape_ms=0.022 "
        "compute_unaccounted_ms=0.003 "
        "qkv_shape=(1, 4096) output_shape=(1, 2048) batch_key=abc "
        "recv_done_ns=1003900000 compute_done_ns=1004100000 "
        "send_done_ns=1004200000 recv_start_ns=1002000000 "
        "pre_compute_start_ns=1003900000 pre_compute_done_ns=1004000000 "
        "paged_flash_done_ns=1004070000 reshape_done_ns=1004150000 "
        "send_start_ns=1004150000\n"
        "PAP NIXL mailbox send trace actor=attention msg_id=z "
        "kind=attention_result_batch nbytes=4096 queue_ms=0.060 publish_ms=0.043 "
        "pack_ms=0.008 slot_wait_ms=0.003 copy_ms=0.019 payload_ms=0.002 "
        "piggyback_ms=0.001 notify_ms=0.013 write_ms=0.000 "
        "write_prepare_ms=0.000 write_transfer_ms=0.000 write_polls=0 "
        "ack_wait_ms=0.230 "
        "total_ms=0.350\n"
        "PAP NIXL mailbox read trace actor=attention msg_id=w "
        "kind=attention_task_batch nbytes=8192 prepare_ms=0.009 "
        "slot_wait_ms=0.001 handle_prepare_ms=0.007 transfer_ms=0.080 "
        "transfer_polls=10 materialize_ms=0.009 total_ms=0.098\n"
        "PAP NIXL mailbox recv wait trace actor=attention msg_id=w "
        "kind=attention_task_batch requested_msg_id= wait_ms=0.600\n"
    )

    summary = summarize_pap_trace_logs(log_dir)

    assert summary["projection_trace"]["recv_ms"].median == 0.700
    assert summary["projection_trace"]["yield_ms"].median == 0.200
    assert abs(summary["projection_trace"]["gap_ms"].median - 0.020) < 1e-9
    assert summary["projection_trace"]["batches"].median == 1
    assert summary["projection_fanin"]["peers"].median == 3
    assert summary["projection_fanin"]["spread_ms"].median == 6.0
    assert (
        summary["projection_fanin"]["spread_over_fastest_pct"].median
        == 150.0
    )
    assert "ubatch_id" not in summary["projection_timeline"]
    assert "ubatch_id" not in summary["projection_layer_timeline"]
    assert summary["projection_timeline"]["pre_attn_compute_ms"].median == 0.400
    assert summary["projection_timeline"]["o_proj_ms"].median == 0.300
    assert summary["projection_timeline"]["self_attn_total_ms"].median == 1.750
    assert summary["projection_layer_timeline"]["input_norm_ms"].median == 0.100
    assert summary["projection_layer_timeline"]["mlp_ms"].median == 0.600
    assert summary["projection_layer_timeline"]["layer_total_ms"].median == 2.600
    assert summary["projection_critical_path"]["qkv_ms"].median == 0.400
    assert summary["projection_critical_path"]["recv_ms"].median == 0.700
    assert summary["projection_critical_path"]["gaps_ms"].median == 0.390
    assert summary["projection_model_forward"]["model_forward_ms"].median == 93.600
    assert summary["projection_logits"]["logits_ms"].median == 1.400
    assert summary["projection_runner_forward_detail"]["input_prep_ms"].median == 0.200
    assert summary["projection_runner_forward_detail"]["metadata_ms"].median == 0.300
    assert summary["projection_runner_forward_detail"]["preprocess_ms"].median == 0.400
    assert (
        summary["projection_runner_forward_detail"]["model_forward_ms"].median == 90.000
    )
    assert (
        summary["projection_runner_forward_detail"]["hidden_slice_ms"].median == 0.050
    )
    assert summary["projection_runner_forward_detail"]["logits_ms"].median == 1.200
    assert (
        summary["projection_runner_forward_detail"]["postprocess_tail_ms"].median
        == 0.150
    )
    assert summary["projection_runner_forward_detail"]["total_ms"].median == 92.300
    assert summary["projection_first_output"]["generated_tokens"].median == 1
    assert summary["projection_first_output"]["sched_ms"].median == 0.100
    assert summary["projection_first_output"]["exec_and_sample_ms"].median == 93.600
    assert summary["projection_first_output"]["scheduler_update_ms"].median == 0.500
    assert (
        summary["projection_first_output"]["step_to_first_output_ms"].median == 94.200
    )
    assert summary["attention_trace"]["compute_ms"].median == 0.140
    assert summary["attention_trace"]["append_kv_ms"].median == 0.050
    assert summary["attention_trace"]["recv_wait_ms"].median == 0.600
    assert summary["attention_trace"]["recv_read_ms"].median == 0.098
    assert summary["attention_trace"]["recv_materialize_ms"].median == 0.009
    assert summary["attention_trace"]["recv_transfer_ms"].median == 0.080
    assert summary["attention_trace"]["recv_wait_other_ms"].median == 0.502
    assert summary["attention_trace"]["recv_unaccounted_ms"].median == 0.220
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
    assert summary["attention_trace"]["metadata_build_ms"].median == 0.061
    assert summary["attention_trace"]["paged_flash_kernel_ms"].median == 0.071
    assert summary["attention_trace"]["attention_output_reshape_ms"].median == 0.022
    assert summary["attention_trace"]["compute_unaccounted_ms"].median == 0.003
    assert summary["attention_trace"]["calls"].median == 1
    assert (
        summary["projection_attention_correlation"][
            "projection_send_done_to_attention_recv_start_ms"
        ].median
        == 2.000
    )
    assert (
        summary["projection_attention_correlation"]["attention_recv_ms"].median == 1.900
    )
    assert (
        summary["projection_attention_correlation"]["attention_pre_compute_ms"].median
        == 0.100
    )
    assert (
        summary["projection_attention_correlation"]["attention_compute_ms"].median
        == 0.100
    )
    assert (
        summary["projection_attention_correlation"]["attention_post_compute_ms"].median
        == 0.050
    )
    assert (
        summary["projection_attention_correlation"]["attention_send_ms"].median == 0.050
    )
    assert (
        summary["projection_attention_correlation"][
            "attention_send_done_to_projection_recv_done_ms"
        ].median
        == 1.600
    )
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
    assert summary["mailbox_send"]["projection"]["write_ms"].median == 0.033
    assert summary["mailbox_send"]["projection"]["slot_wait_ms"].median == 0.005
    assert summary["mailbox_send"]["projection"]["payload_ms"].median == 0.004
    assert summary["mailbox_send"]["projection"]["piggyback_ms"].median == 0.002
    assert summary["mailbox_send"]["projection"]["write_prepare_ms"].median == 0.011
    assert summary["mailbox_send"]["projection"]["write_transfer_ms"].median == 0.020
    assert summary["mailbox_send"]["projection"]["write_polls"].median == 3
    assert summary["mailbox_send"]["attention"]["total_ms"].median == 0.350
    assert summary["mailbox_send"]["attention"]["write_polls"].median == 0
    assert summary["mailbox_read"]["attention"]["transfer_ms"].median == 0.080
    assert summary["mailbox_read"]["attention"]["slot_wait_ms"].median == 0.001
    assert summary["mailbox_read"]["attention"]["handle_prepare_ms"].median == 0.007
    assert summary["mailbox_read"]["projection"]["total_ms"].median == 0.088
    assert summary["mailbox_read"]["projection"]["slot_wait_ms"].median == 0.002
    assert summary["mailbox_read"]["projection"]["handle_prepare_ms"].median == 0.006
    assert (
        summary["mailbox_send_by_kind"]["projection:attention_task_batch"][
            "write_ms"
        ].median
        == 0.033
    )
    assert (
        summary["mailbox_send_by_kind"]["projection:attention_task_batch"][
            "total_ms"
        ].median
        == 0.310
    )
    assert (
        summary["mailbox_send_by_kind"]["inline_projection:attention_task_inline"][
            "queue_ms"
        ].median
        == 0.0
    )
    assert (
        summary["mailbox_send_by_kind"]["inline_projection:attention_task_inline"][
            "ack_wait_ms"
        ].median
        == 0.0
    )
    assert (
        summary["mailbox_send_by_kind"]["inline_projection:attention_task_inline"][
            "total_ms"
        ].median
        == 0.060
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


def test_trace_summary_reports_multi_pa_completion_skew(tmp_path: Path) -> None:
    log_dir = tmp_path / "service_logs"
    log_dir.mkdir()
    (log_dir / "projection_0.log").write_text(
        "PAP OFFLOAD_EXEC projection trace layer=model.layers.1.self_attn.attn "
        "batches=3 calls=9 send_ms=0.030 trigger_ms=0.020 "
        "yield_ms=0.000 recv_ms=0.100 total_ms=1.000 batch_keys=a|b|c "
        "route_rows=2|3|4 route_kv_tokens=200|360|600 "
        "send_done_ns=1000000000 yield_start_ns=1000010000 "
        "yield_end_ns=1000020000 recv_done_ns=1013000000\n",
        encoding="utf-8",
    )

    def attention_line(
        key: str,
        calls: int,
        recv_start_ns: int,
        compute_done_ns: int,
        send_done_ns: int,
    ) -> str:
        recv_done_ns = recv_start_ns + 1_000_000
        pre_compute_done_ns = compute_done_ns - 1_000_000
        return (
            "PAP OFFLOAD_EXEC attention mailbox batch trace "
            f"layer=model.layers.1.self_attn.attn calls={calls} "
            "recv_qkv_ms=0.100 compute_ms=0.200 send_output_ms=0.050 "
            "total_ms=1.000 append_kv_ms=0.050 pack_ms=0.030 "
            "sdpa_ms=0.100 reshape_ms=0.020 "
            f"batch_key={key} recv_done_ns={recv_done_ns} "
            f"compute_done_ns={compute_done_ns} send_done_ns={send_done_ns} "
            f"recv_start_ns={recv_start_ns} "
            f"pre_compute_start_ns={recv_done_ns} "
            f"pre_compute_done_ns={pre_compute_done_ns} "
            f"paged_flash_done_ns={compute_done_ns - 500_000} "
            f"reshape_done_ns={compute_done_ns} "
            f"send_start_ns={compute_done_ns}\n"
        )

    (log_dir / "attention_0.log").write_text(
        attention_line("a", 2, 1_001_000_000, 1_005_000_000, 1_006_000_000),
        encoding="utf-8",
    )
    (log_dir / "attention_1.log").write_text(
        attention_line("b", 3, 1_002_000_000, 1_007_000_000, 1_008_000_000),
        encoding="utf-8",
    )
    (log_dir / "attention_2.log").write_text(
        attention_line("c", 4, 1_004_000_000, 1_010_000_000, 1_012_000_000),
        encoding="utf-8",
    )

    summary = summarize_pap_trace_logs(log_dir, include_samples=True)
    correlation = summary["projection_attention_correlation"]
    assert correlation["pa_recv_start_skew_ms"].median == 3.0
    assert correlation["pa_compute_completion_skew_ms"].median == 5.0
    assert correlation["pa_completion_skew_ms"].median == 6.0
    assert correlation["pa_completion_skew_over_fastest_pct"].median == 100.0
    assert abs(correlation["pa_mean_idle_until_slowest_ms"].median - 10 / 3) < 1e-9
    assert correlation["route_rows_range"].median == 2
    assert correlation["route_kv_tokens_range"].median == 400
    assert correlation["slowest_pa_rows"].median == 4
    assert correlation["slowest_pa_kv_tokens"].median == 600
    assert correlation["slowest_pa_has_max_rows"].mean == 1
    assert correlation["slowest_pa_has_max_kv_tokens"].mean == 1
    assert correlation["completion_skew_ms_per_1k_kv_range"].median == 15
    samples = summary["projection_attention_correlation_samples"]
    assert samples["pa_completion_skew_ms"] == [6.0]
    assert samples["pa_completion_skew_over_fastest_pct"] == [100.0]
