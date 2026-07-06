import json
from pathlib import Path

from vllm.pap.remote_attention_diagnostics import (
    DiagnosticRow,
    LowerBoundConfig,
    estimate_remote_attention_lower_bound,
    rows_to_markdown,
    summarize_run_directory,
)


def test_estimate_remote_attention_lower_bound_qwen3_8b_batch64() -> None:
    estimate = estimate_remote_attention_lower_bound(
        LowerBoundConfig(
            batch_size=64,
            q_size=4096,
            kv_size=1024,
            output_size=4096,
            dtype_bytes=2,
            p2p_bandwidth_gbps=21.0,
            attention_compute_ms=0.12,
            num_layers=36,
        )
    )

    assert estimate.bytes_per_layer == 1310720
    assert round(estimate.transfer_ms_per_layer, 3) == 0.062
    assert round(estimate.lower_bound_ms_per_layer, 3) == 0.182
    assert round(estimate.lower_bound_ms_per_token, 3) == 6.567


def test_summarize_run_directory_combines_benchmark_trace_and_lower_bound(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "pap_1pa1p_i128_o32_q16_c64_w32"
    service_logs = run_dir / "service_logs"
    service_logs.mkdir(parents=True)
    (run_dir / "1PA1P_i128_o32_q16_c64_w32.json").write_text(
        json.dumps(
            {
                "completed": 256,
                "failed": 0,
                "request_throughput": 6.11,
                "output_throughput": 195.536,
                "median_ttft_ms": 884.675,
                "median_tpot_ms": 294.765,
                "p99_tpot_ms": 306.697,
                "max_concurrent_requests": 64,
            }
        )
    )
    (service_logs / "projection_0.log").write_text(
        "PAP OFFLOAD_EXEC projection timeline "
        "layer=model.layers.1.self_attn.attn batches=1 calls=64 "
        "pre_attn_compute_ms=0.400 send_ms=0.040 trigger_ms=0.000 "
        "yield_ms=0.200 recv_ms=1.010 o_proj_ms=0.300 "
        "remote_total_ms=1.050 self_attn_total_ms=1.750\n"
    )
    (service_logs / "attention_0.log").write_text(
        "PAP OFFLOAD_EXEC attention mailbox batch trace "
        "layer=model.layers.1.self_attn.attn calls=64 recv_qkv_ms=0.820 "
        "compute_ms=0.120 send_output_ms=0.010 total_ms=0.970 "
        "append_kv_ms=0.050 pack_ms=0.030 sdpa_ms=0.040 reshape_ms=0.020 "
        "paged_metadata_ms=0.000 paged_flash_ms=0.000 "
        "qkv_shape=(64, 6144) output_shape=(64, 4096) batch_key=abc "
        "recv_done_ns=1003900000 compute_done_ns=1004100000 "
        "send_done_ns=1004200000\n"
    )

    row = summarize_run_directory(run_dir)

    assert row.topology == "1PA1P"
    assert row.input_len == "128"
    assert row.output_len == "32"
    assert row.qps == "16"
    assert row.max_concurrency == "64"
    assert row.completed == 256
    assert row.failed == 0
    assert row.median_tpot_ms == 294.765
    assert row.projection_remote_total_median_ms == 1.05
    assert row.attention_compute_median_ms == 0.12
    assert round(row.lower_bound_ms_per_layer, 3) == 0.182
    assert round(row.e2e_ms_per_layer, 3) == 8.188
    assert row.fast_path_status["paged_flash"] == "inactive"
    assert row.fast_path_status["attention_batch_calls_median"] == "64.000"


def test_rows_to_markdown_includes_lower_bound_and_fast_path_status() -> None:
    row = DiagnosticRow(
        path="run",
        topology="1PA1P",
        input_len="128",
        output_len="32",
        qps="16",
        max_concurrency="64",
        num_warmups="32",
        completed=256,
        failed=0,
        request_throughput=6.11,
        output_throughput=195.536,
        median_ttft_ms=884.675,
        median_tpot_ms=294.765,
        p99_tpot_ms=306.697,
        projection_remote_total_median_ms=1.05,
        projection_recv_median_ms=1.01,
        attention_compute_median_ms=0.12,
        attention_total_median_ms=0.97,
        lower_bound_ms_per_layer=0.182,
        lower_bound_ms_per_token=6.567,
        e2e_ms_per_layer=8.188,
        fast_path_status={
            "paged_flash": "inactive",
            "fallback": "inactive",
            "attention_batch_calls_median": "64.000",
        },
    )

    markdown = rows_to_markdown([row])

    assert "| topology | input | output | qps | warmup | max conc |" in markdown
    assert "| 1PA1P | 128 | 32 | 16 | 32 | 64 |" in markdown
    assert "0.182" in markdown
    assert "8.188" in markdown
    assert "paged_flash=inactive" in markdown
