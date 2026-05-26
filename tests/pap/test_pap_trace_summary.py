from pathlib import Path

from vllm.pap.trace_summary import summarize_pap_trace_logs


def test_trace_summary_extracts_projection_attention_and_mailbox_stats(tmp_path: Path) -> None:
    log_dir = tmp_path / "service_logs"
    log_dir.mkdir()
    (log_dir / "projection_0.log").write_text(
        "PAP OFFLOAD_EXEC projection trace layer=model.layers.1.self_attn.attn "
        "batches=1 calls=1 send_ms=0.030 trigger_ms=0.000 yield_ms=0.200 "
        "recv_ms=0.700 total_ms=0.950 batch_keys=abc "
        "send_done_ns=1000000000 yield_start_ns=1000010000 "
        "yield_end_ns=1005000000 recv_done_ns=1005800000\n"
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
    assert summary["attention_trace"]["compute_ms"].median == 0.140
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
