from pathlib import Path

from vllm.pap.trace_summary import summarize_pap_trace_logs


def test_trace_summary_extracts_projection_attention_and_mailbox_stats(tmp_path: Path) -> None:
    log_dir = tmp_path / "service_logs"
    log_dir.mkdir()
    (log_dir / "projection_0.log").write_text(
        "PAP OFFLOAD_EXEC projection trace layer=model.layers.1.self_attn.attn "
        "calls=1 send_ms=0.030 trigger_ms=0.000 recv_ms=0.700 total_ms=0.730\n"
        "PAP NIXL mailbox send trace actor=projection msg_id=x "
        "kind=attention_task_batch nbytes=8192 queue_ms=0.020 publish_ms=0.040 "
        "pack_ms=0.006 copy_ms=0.018 notify_ms=0.013 ack_wait_ms=0.250 "
        "total_ms=0.310\n"
        "PAP NIXL mailbox read trace actor=projection msg_id=y "
        "kind=attention_result_batch nbytes=4096 prepare_ms=0.009 transfer_ms=0.070 "
        "transfer_polls=15 materialize_ms=0.009 total_ms=0.088\n"
    )
    (log_dir / "attention_0.log").write_text(
        "PAP OFFLOAD_EXEC attention mailbox batch trace "
        "layer=model.layers.1.self_attn.attn calls=1 recv_qkv_ms=0.820 "
        "compute_ms=0.140 send_output_ms=0.010 total_ms=0.970 "
        "qkv_shape=(1, 4096) output_shape=(1, 2048)\n"
        "PAP NIXL mailbox send trace actor=attention msg_id=z "
        "kind=attention_result_batch nbytes=4096 queue_ms=0.060 publish_ms=0.043 "
        "pack_ms=0.008 copy_ms=0.019 notify_ms=0.013 ack_wait_ms=0.230 "
        "total_ms=0.350\n"
        "PAP NIXL mailbox read trace actor=attention msg_id=w "
        "kind=attention_task_batch nbytes=8192 prepare_ms=0.009 transfer_ms=0.080 "
        "transfer_polls=10 materialize_ms=0.009 total_ms=0.098\n"
    )

    summary = summarize_pap_trace_logs(log_dir)

    assert summary["projection_trace"]["recv_ms"].median == 0.700
    assert summary["attention_trace"]["compute_ms"].median == 0.140
    assert summary["mailbox_send"]["projection"]["ack_wait_ms"].median == 0.250
    assert summary["mailbox_send"]["attention"]["total_ms"].median == 0.350
    assert summary["mailbox_read"]["attention"]["transfer_ms"].median == 0.080
    assert summary["mailbox_read"]["projection"]["total_ms"].median == 0.088
