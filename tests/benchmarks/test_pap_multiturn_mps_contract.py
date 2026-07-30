from pathlib import Path

ROOT = Path(__file__).parents[2]
PAP_RUNNER = ROOT / "benchmarks/pap/scripts/run_pap_workload.sh"


def test_static_mps_lifecycle_is_partitioned_and_audited() -> None:
    runner = PAP_RUNNER.read_text(encoding="utf-8")

    assert 'PAP_STATIC_PREFILL_CHUNKS="${PAP_STATIC_PREFILL_CHUNKS:-20}"' in runner
    assert 'PAP_STATIC_ATTENTION_CHUNKS="${PAP_STATIC_ATTENTION_CHUNKS:-3}"' in runner
    assert (
        'PAP_STATIC_PREFILL_EXPECTED_SMS="${PAP_STATIC_PREFILL_EXPECTED_SMS:-80}"'
        in runner
    )
    assert (
        'PAP_STATIC_ATTENTION_EXPECTED_SMS="${PAP_STATIC_ATTENTION_EXPECTED_SMS:-12}"'
        in runner
    )
    assert "nvidia-cuda-mps-control -d -S" in runner
    assert "sm_partition add" in runner
    assert "sm_partition rm" in runner
    assert "CUDA_MPS_SM_PARTITION" in runner
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in runner
    assert "validate_static_partition_visible_sms" in runner
    assert "mps_static_audit_pa_" in runner


def test_pap_uses_its_own_kv_lease_for_long_lived_attention_ownership() -> None:
    runner = PAP_RUNNER.read_text(encoding="utf-8")

    assert (
        'PAP_NIXL_CONNECTOR_LEASE_SECONDS="${PAP_NIXL_CONNECTOR_LEASE_SECONDS:-1}"'
        in runner
    )
    assert r'\"kv_lease_duration\":${PAP_NIXL_CONNECTOR_LEASE_SECONDS}' in runner
    assert 'PAP_KV_LEASE_TTL_SECONDS="${PAP_KV_LEASE_TTL_SECONDS:-300}"' in runner


def test_conversation_affinity_audit_counts_sessions_for_aiperf() -> None:
    runner = PAP_RUNNER.read_text(encoding="utf-8")

    assert 'load_rounds = int(os.environ["PAP_AIPERF_TURNS"])' in runner
    assert 'load_conversations = int(os.environ["PAP_AIPERF_SESSIONS"])' in runner


def test_projection_prepares_metadata_without_microbatch_pipeline() -> None:
    runner = PAP_RUNNER.read_text(encoding="utf-8")

    assert "--no-async-scheduling" not in runner
    assert "PAP_PROJECTION_ASYNC_SCHEDULING" in runner
    assert "ASYNC_SCHEDULING=1" in runner
    assert "SCHEDULER_QUEUE_DEPTH=2" in runner
    assert "PAP_RUNNER_MICROBATCH_COUNT \\" in runner
    assert "audit_projection_scheduling" in runner
    assert "PAP_RUNNER_MICROBATCH_PIPELINE=0" in runner
