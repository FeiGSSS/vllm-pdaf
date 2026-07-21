from pathlib import Path
from types import SimpleNamespace

from examples.disaggregated.disaggregated_serving.disagg_proxy_multiturn import (
    ConversationInstanceRouter,
    _pop_conversation_id,
)

ROOT = Path(__file__).parents[2]
RUNNER = (
    ROOT
    / ".claude/skills/vllm-pap-benchmark/scripts/run_pd_multiturn_load.sh"
)
TOPOLOGY_RUNNER = ROOT / "benchmarks/pap/scripts/run_pd_multiturn_topology.sh"
PAP_RUNNER = ROOT / "benchmarks/pap/scripts/run_pap_workload.sh"
AIPERF_RUNNER = ROOT / "benchmarks/pap/aiperf/run_profile.sh"
CAPACITY_RUNNER = ROOT / "benchmarks/pap/aiperf/run_capacity_matrix.sh"


def test_pd_proxy_accepts_aiperf_session_header() -> None:
    payload = {"model": "qwen"}

    assert _pop_conversation_id(payload, "aiperf-session") == "aiperf-session"


def test_pd_runner_uses_one_official_connector_for_both_modes() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert "NixlConnector" in text
    assert "NixlPushConnector" not in text
    assert "bidirectional_kv_xfer" in text
    assert "oneway" in text
    assert "twoway" in text
    assert "configure_ucx122_runtime" in text
    assert "verify_ucx122_runtime" in text
    assert "disagg_proxy_multiturn.py" in text


def test_pd_runner_records_and_audits_transfer_mode() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert "PD_TRANSFER_MODE=" in text
    assert "BIDIRECTIONAL_KV_XFER=" in text
    assert "--proxy-log" in text
    assert '"nixl-${TRANSFER_MODE}"' in text


def test_pd_runner_captures_default_off_decode_deferred_trace() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert 'PAP_DEFERRED_CUDA_TRACE="${PAP_DEFERRED_CUDA_TRACE:-0}"' in text
    assert "PAP_DEFERRED_TRACE_ROLE=pd_decode" in text
    assert "PAP_DEFERRED_TRACE_OUTPUT=" in text
    assert "capture_pd_decode_deferred_trace" in text
    assert "validate_deferred_trace.py" in text
    assert "pd_decode_deferred_trace.json" in text
    assert '"${output_path}.flush"' in text
    assert text.index("capture_pd_decode_deferred_trace") < text.index(
        '"${PYTHON_BIN}" "${FINALIZER}"'
    )


def test_pd_proxy_balances_and_retains_conversation_owners() -> None:
    clients = [SimpleNamespace(id=index) for index in range(3)]
    router = ConversationInstanceRouter(clients)  # type: ignore[arg-type]

    first = [router.select(f"conv-{index}").id for index in range(6)]
    second = [
        router.select(f"conv-{index}").id for index in reversed(range(6))
    ]

    assert first == [0, 1, 2, 0, 1, 2]
    assert second == [2, 1, 0, 2, 1, 0]
    assert router.snapshot() == {
        "conversations": 6,
        "assignments": [2, 2, 2],
        "requests": [4, 4, 4],
    }


def test_four_gpu_topology_runner_has_bounded_conversation_load() -> None:
    text = TOPOLOGY_RUNNER.read_text(encoding="utf-8")

    assert 'PD_LOAD_TOPOLOGY:-3p1d' in text
    assert 'PREFILL_COUNT + DECODE_COUNT != 4' in text
    assert 'PD_LOAD_REQUEST_TIMEOUT_SECONDS:-180' in text
    assert '--request-timeout-seconds "${REQUEST_TIMEOUT_SECONDS}"' in text
    assert "disagg_proxy_multiturn.py" in text
    assert "ensure_gpus_idle" in text


def test_pd_runner_uses_role_specific_scheduler_limits() -> None:
    text = TOPOLOGY_RUNNER.read_text(encoding="utf-8")

    assert "PD_LOAD_PREFILL_MAX_NUM_BATCHED_TOKENS" in text
    assert "PD_LOAD_PREFILL_MAX_NUM_SEQS" in text
    assert "PD_LOAD_DECODE_MAX_NUM_BATCHED_TOKENS" in text
    assert "PD_LOAD_DECODE_MAX_NUM_SEQS" in text
    assert '--max-num-batched-tokens "${PREFILL_MAX_NUM_BATCHED_TOKENS}"' in text
    assert '--max-num-batched-tokens "${DECODE_MAX_NUM_BATCHED_TOKENS}"' in text


def test_aiperf_capacity_lane_uses_concurrency_without_request_rate() -> None:
    profile_text = AIPERF_RUNNER.read_text(encoding="utf-8")
    capacity_text = CAPACITY_RUNNER.read_text(encoding="utf-8")

    assert 'AIPERF_TIMING_MODE="${AIPERF_TIMING_MODE:-concurrency}"' in profile_text
    assert 'AIPERF_TIMING_MODE=concurrency' in capacity_text
    assert 'PAP_AIPERF_REQUEST_RATE=' in capacity_text
    assert 'PD_AIPERF_REQUEST_RATE=' in capacity_text


def test_capacity_lane_freezes_workload_and_memory_configuration() -> None:
    text = CAPACITY_RUNNER.read_text(encoding="utf-8")

    assert "TURNS=10" in text
    assert "DOCUMENT_TOKENS=8192" in text
    assert "APPEND_TOKENS=512" in text
    assert 'PAP_CAPACITY_OUTPUT_TOKENS:-32' in text
    assert 'multiturn_s${TOTAL_SESSIONS}_8k_plus512_random_o' in text
    assert "THINK_TIME_MS=3000" in text
    assert "TOOL_TIME_MS=1000" in text
    assert "TOOL_EVERY=3" in text
    assert "PAP_CAPACITY_SESSIONS:-32" in text
    assert "PAP_GPU_MEMORY_UTILIZATION=0.76" in text
    assert "PD_GPU_MEMORY_UTILIZATION=0.90" in text
    assert "PREFILL_MAX_NUM_BATCHED_TOKENS=16384" in text
    assert "DECODE_MAX_NUM_BATCHED_TOKENS=64" in text
    assert "MAX_NUM_SEQS=64" in text
    assert "MAX_NUM_PARTIAL_PREFILLS=default_1" in text
    assert "--max-num-partial-prefills" not in text
    assert "PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS=64" in text
    assert "PAP_CAPACITY_PAP_3PA1P_POINTS:-12,20,28,32" in text
    assert "PAP_CAPACITY_PD_1P3D_POINTS:-8" in text
    assert "PAP_CAPACITY_PD_2P2D_POINTS:-10,16,20,24" in text
    assert "PAP_CAPACITY_PD_3P1D_POINTS:-8,14,20" in text
    assert '--sessions "${TOTAL_SESSIONS}"' in text
    assert 'PAP_MULTITURN_LOAD_CONVERSATIONS="${TOTAL_SESSIONS}"' in text
    assert 'PD_LOAD_CONVERSATIONS="${TOTAL_SESSIONS}"' in text
    assert '--sessions "${concurrency}"' not in text


def test_capacity_lane_only_prunes_eligible_slo_failures() -> None:
    text = CAPACITY_RUNNER.read_text(encoding="utf-8")

    assert "point_has_eligible_run=0" in text
    assert "point_has_relaxed_failure=0" in text
    assert "'.correctness.passed'" in text
    assert "Not using ineligible" in text
    assert "PAP_CAPACITY_GPU_IDLE_STABILITY_SECONDS:-15" in text


def test_aiperf_capacity_does_not_reject_load_above_scheduler_batch_size() -> None:
    pap_text = PAP_RUNNER.read_text(encoding="utf-8")
    pd_text = TOPOLOGY_RUNNER.read_text(encoding="utf-8")

    assert (
        'PAP_MULTITURN_ACTIVE_CONVERSATIONS="${PAP_AIPERF_CONCURRENCY}"'
        in pap_text
    )
    assert "active conversations exceed Projection max_num_seqs" not in pap_text
    assert "per-PA conversations exceed Prefill max_num_seqs" not in pap_text
    assert 'ACTIVE_CONVERSATIONS="${PD_AIPERF_CONCURRENCY}"' in pd_text
    assert 'TOTAL_CONVERSATIONS=%q' in pd_text
