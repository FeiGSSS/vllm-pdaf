from pathlib import Path

ROOT = Path(__file__).parents[2]
PAP_RUNNER = ROOT / "benchmarks/pap/scripts/run_pap_workload.sh"


def test_static_mps_lifecycle_is_partitioned_and_audited() -> None:
    runner = PAP_RUNNER.read_text(encoding="utf-8")

    assert "nvidia-cuda-mps-control -d -S" in runner
    assert "sm_partition add" in runner
    assert "sm_partition rm" in runner
    assert "CUDA_MPS_SM_PARTITION" in runner
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in runner
    assert "validate_static_partition_visible_sms" in runner
    assert "mps_static_audit_pa_" in runner


def test_conversation_affinity_audit_counts_sessions_for_aiperf() -> None:
    runner = PAP_RUNNER.read_text(encoding="utf-8")

    assert 'client_mode == "aiperf_multiturn"' in runner
