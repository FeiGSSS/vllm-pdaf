from pathlib import Path
from types import SimpleNamespace

from examples.disaggregated.disaggregated_serving.disagg_proxy_multiturn import (
    ConversationInstanceRouter,
)


ROOT = Path(__file__).parents[2]
RUNNER = (
    ROOT
    / ".claude/skills/vllm-pap-benchmark/scripts/run_pd_multiturn_load.sh"
)
TOPOLOGY_RUNNER = ROOT / "benchmarks/pap/scripts/run_pd_multiturn_topology.sh"


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
