from pathlib import Path
from types import SimpleNamespace

from examples.disaggregated.disaggregated_serving.disagg_proxy_multiturn import (
    ConversationPairRouter,
    _pop_conversation_id,
    _select_instance_pair,
)

ROOT = Path(__file__).parents[2]
TOPOLOGY_RUNNER = ROOT / "benchmarks/pap/scripts/run_pd_multiturn_topology.sh"
PAP_RUNNER = ROOT / "benchmarks/pap/scripts/run_pap_workload.sh"
AIPERF_RUNNER = ROOT / "benchmarks/pap/aiperf/run_profile.sh"
CAPACITY_RUNNER = ROOT / "benchmarks/pap/aiperf/run_capacity_matrix.sh"
DP_RUNNER = ROOT / "benchmarks/pap/scripts/run_dp_multiturn.sh"
P17_RUNNER = ROOT / "benchmarks/pap/scripts/run_p17_1pa1p.sh"


def test_pd_proxy_accepts_aiperf_session_header() -> None:
    payload = {"model": "qwen"}

    assert _pop_conversation_id(payload, "aiperf-session") == "aiperf-session"


def test_pd_proxy_balances_and_retains_conversation_pairs() -> None:
    router = ConversationPairRouter(
        [SimpleNamespace(id=index) for index in range(2)],
        [SimpleNamespace(id=index) for index in range(2)],
    )  # type: ignore[arg-type]

    first = [
        tuple(client.id for client in router.select(f"conv-{index}"))
        for index in range(4)
    ]
    second = [
        tuple(client.id for client in router.select(f"conv-{index}"))
        for index in reversed(range(4))
    ]

    assert first == [(0, 0), (1, 1), (0, 1), (1, 0)]
    assert second == list(reversed(first))
    assert router.snapshot() == {
        "prefill": {
            "conversations": 4,
            "assignments": [2, 2],
            "requests": [4, 4],
        },
        "decode": {
            "conversations": 4,
            "assignments": [2, 2],
            "requests": [4, 4],
        },
        "pairs": {
            "conversations": 4,
            "labels": ["p0:d0", "p1:d1", "p0:d1", "p1:d0"],
            "assignments": [1, 1, 1, 1],
            "requests": [2, 2, 2, 2],
        },
    }


def test_pd_proxy_selects_stable_prefill_decode_pairs() -> None:
    state = SimpleNamespace(
        instance_router=ConversationPairRouter(
            [SimpleNamespace(id=index) for index in range(2)],
            [SimpleNamespace(id=index) for index in range(2)],
        ),
    )

    first = {
        conversation_id: tuple(
            client.id
            for client in _select_instance_pair(state, conversation_id)
        )
        for conversation_id in ("conv-0", "conv-1", "conv-2", "conv-3")
    }
    repeated = {
        conversation_id: tuple(
            client.id
            for client in _select_instance_pair(state, conversation_id)
        )
        for conversation_id in ("conv-3", "conv-1", "conv-0", "conv-2")
    }

    assert first == {
        "conv-0": (0, 0),
        "conv-1": (1, 1),
        "conv-2": (0, 1),
        "conv-3": (1, 0),
    }
    assert repeated == first


def test_pd_topology_runner_accepts_arbitrary_positive_gpu_count() -> None:
    text = TOPOLOGY_RUNNER.read_text(encoding="utf-8")

    assert "PD_LOAD_TOPOLOGY:-3p1d" in text
    assert "GPU_COUNT=$((PREFILL_COUNT + DECODE_COUNT))" in text
    assert "PREFILL_COUNT + DECODE_COUNT != 4" not in text
    assert "PD_LOAD_REQUEST_TIMEOUT_SECONDS:-180" in text
    assert 'printf \'CLIENT=%q\\n\' "aiperf"' in text
    assert "PD_LOAD_ROUNDS:-10" in text
    assert "PD_LOAD_CONVERSATIONS:-32" in text
    assert "PD_LOAD_OUTPUT_TOKENS:-32" in text
    assert "pap_pd_multiturn_load_client.py" not in text
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


def test_pap_and_pd_runners_make_piecewise_graph_role_specific() -> None:
    pap_text = PAP_RUNNER.read_text(encoding="utf-8")
    pd_text = TOPOLOGY_RUNNER.read_text(encoding="utf-8")

    assert 'PAP_EXECUTION_MODE="${PAP_EXECUTION_MODE:-eager}"' in pap_text
    assert "PAP_CUDAGRAPH_COMPATIBLE" in pap_text
    assert "PAP_CUDAGRAPH_ROLE=prefill" in pap_text
    assert "PAP_CUDAGRAPH_ROLE=projection" in pap_text
    assert r"\"cudagraph_mode\":\"PIECEWISE\"" in pap_text
    assert "PAP_PREFILL_CUDAGRAPH_CAPTURE_SIZES" in pap_text
    assert "PAP_PROJECTION_CUDAGRAPH_CAPTURE_SIZES" in pap_text

    assert 'EXECUTION_MODE="${PD_LOAD_EXECUTION_MODE:-eager}"' in pd_text
    assert r"\"cudagraph_mode\":\"PIECEWISE\"" in pd_text
    assert "PREFILL_CUDAGRAPH_CAPTURE_SIZES" in pd_text
    assert "DECODE_CUDAGRAPH_CAPTURE_SIZES" in pd_text
    assert "PAP_CUDAGRAPH_COMPATIBLE=0" in pd_text


def test_aiperf_capacity_lane_uses_concurrency_without_request_rate() -> None:
    profile_text = AIPERF_RUNNER.read_text(encoding="utf-8")
    capacity_text = CAPACITY_RUNNER.read_text(encoding="utf-8")

    assert 'AIPERF_TIMING_MODE="${AIPERF_TIMING_MODE:-concurrency}"' in profile_text
    assert "AIPERF_TIMING_MODE=concurrency" in capacity_text
    assert "PAP_AIPERF_REQUEST_RATE=" in capacity_text
    assert "PD_AIPERF_REQUEST_RATE=" in capacity_text


def test_aiperf_is_the_only_performance_testbed() -> None:
    pap_text = PAP_RUNNER.read_text(encoding="utf-8")
    pd_text = TOPOLOGY_RUNNER.read_text(encoding="utf-8")

    assert not P17_RUNNER.exists()
    assert 'case "${PAP_BENCH_CLIENT_MODE}"' not in pap_text
    assert "bench serve" not in pap_text
    assert "multiturn_prefix_cache)" not in pap_text
    assert "pap_pd_multiturn_load_client.py" not in pap_text
    assert 'PAP_BENCH_CLIENT="aiperf"' in pap_text
    assert "PD_LOAD_CLIENT_MODE:-aiperf_multiturn" not in pd_text
    assert "the PD runner is AIPerf-only" in pd_text
    for text in (pap_text, pd_text):
        assert "--document-tokens-median" in text
        assert "--output-tokens-median" in text
        assert "--think-time-ms" in text
        assert "--tool-time-ms" in text


def test_capacity_lane_freezes_workload_and_memory_configuration() -> None:
    text = CAPACITY_RUNNER.read_text(encoding="utf-8")

    assert "PAP_CAPACITY_TURNS:-5" in text
    assert "PAP_CAPACITY_DOCUMENT_TOKENS_MEAN:-8192" in text
    assert "PAP_CAPACITY_APPEND_TOKENS_MEAN:-2200" in text
    assert "PAP_CAPACITY_APPEND_TOKENS_MEDIAN:-800" in text
    assert "PAP_CAPACITY_APPEND_TOKENS_MIN:-4" in text
    assert "PAP_CAPACITY_APPEND_TOKENS_MAX:-4250" in text
    assert "PAP_CAPACITY_OUTPUT_TOKENS:-32" in text
    assert "multiturn_s${TOTAL_SESSIONS}_8k_longtail_random_o" in text
    assert "PAP_CAPACITY_THINK_TIME_MS:-3000" in text
    assert "PAP_CAPACITY_TOOL_TIME_MS:-1000" in text
    assert "PAP_CAPACITY_TOOL_EVERY:-3" in text
    assert "PAP_CAPACITY_SESSIONS:-128" in text
    assert 'DATASET_SESSION_PREFIX="pap-pd-dp-s${TOTAL_SESSIONS}' in text
    assert '--session-prefix "${DATASET_SESSION_PREFIX}"' in text
    assert '--session-prefix "${MATRIX_ID}-session"' not in text
    assert "PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.90" in text
    assert "PAP_PROJECTION_MEMORY_POLICY" in text
    assert "model_weights_x1.20" in text
    assert "PAP_PROJECTION_GPU_MEMORY_UTILIZATION=" not in text
    assert "PAP_GPU_MEMORY_UTILIZATION=" not in text
    assert "PD_GPU_MEMORY_UTILIZATION=0.90" in text
    assert "PAP_CAPACITY_PREFILL_MAX_NUM_BATCHED_TOKENS:-32768" in text
    assert "PAP_CAPACITY_DECODE_MAX_NUM_BATCHED_TOKENS:-256" in text
    assert "PAP_CAPACITY_MAX_NUM_SEQS:-256" in text
    assert "MAX_NUM_PARTIAL_PREFILLS=default_1" in text
    assert "--max-num-partial-prefills" not in text
    assert 'PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS="${OUTPUT_TOKENS_MAX}"' in text
    assert "PAP_CAPACITY_EXECUTION_MODE" in text
    assert 'PAP_EXECUTION_MODE="${EXECUTION_MODE}"' in text
    assert 'PD_LOAD_EXECUTION_MODE="${EXECUTION_MODE}"' in text
    assert "pap_7pa1p,pap_6pa2p,pd_4p4d,pd_6p2d,dp_8" in text
    assert "PAP_CAPACITY_POINTS:-32,64,96,128" in text
    assert '--sessions "${TOTAL_SESSIONS}"' in text
    assert 'PAP_AIPERF_SESSIONS="${TOTAL_SESSIONS}"' in text
    assert 'PD_LOAD_CONVERSATIONS="${TOTAL_SESSIONS}"' in text
    assert 'DP_LOAD_CONVERSATIONS="${TOTAL_SESSIONS}"' in text
    assert '--sessions "${concurrency}"' not in text


def test_capacity_lane_has_native_vllm_dp_baseline() -> None:
    capacity_text = CAPACITY_RUNNER.read_text(encoding="utf-8")
    dp_text = DP_RUNNER.read_text(encoding="utf-8")

    assert 'DP_RUNNER="${ROOT_DIR}/benchmarks/pap/scripts/run_dp_multiturn.sh"' in (
        capacity_text
    )
    assert '--data-parallel-size "${DP_SIZE}"' in dp_text
    assert "VLLM_PORT=" not in dp_text
    assert 'AIPERF_TIMING_MODE=concurrency' in dp_text


def test_pap_launchers_compute_projection_memory_from_model_size() -> None:
    pap_text = PAP_RUNNER.read_text(encoding="utf-8")
    example_text = (ROOT / "examples/pap/launch_pap_nixl.sh").read_text(
        encoding="utf-8"
    )

    for text in (pap_text, example_text):
        assert 'PAP_PREFILL_GPU_MEMORY_UTILIZATION:-0.90' in text
        assert "vllm/pap/model/memory.py" in text
        assert "PROJECTION_GPU_MEMORY_UTILIZATION" in text
        assert "PAP_PROJECTION_GPU_MEMORY_UTILIZATION:-" not in text


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

    assert "PAP_AIPERF_CONCURRENCY > PAP_AIPERF_SESSIONS" in pap_text
    assert "active conversations exceed Projection max_num_seqs" not in pap_text
    assert "per-PA conversations exceed Prefill max_num_seqs" not in pap_text
    assert 'ACTIVE_CONVERSATIONS="${PD_AIPERF_CONCURRENCY}"' in pd_text
    assert "TOTAL_CONVERSATIONS=%q" in pd_text
