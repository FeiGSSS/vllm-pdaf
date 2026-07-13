from pathlib import Path


ROOT = Path(__file__).parents[2]
PAP_WRAPPER = (
    ROOT
    / ".claude/skills/vllm-pap-benchmark/scripts/"
    / "run_pap_multiturn_load.sh"
)
PAP_RUNNER = (
    ROOT
    / ".claude/skills/vllm-pap-benchmark/scripts/"
    / "run_pap_same_pd_workload.sh"
)


def test_pap_only_runner_defaults_to_static_and_keeps_dynamic_profiles() -> None:
    wrapper = PAP_WRAPPER.read_text(encoding="utf-8")
    runner = PAP_RUNNER.read_text(encoding="utf-8")

    assert 'PAP_LOAD_MPS_PROFILE:-baseline_static_64_28' in wrapper
    assert 'baseline_70_30)' in wrapper
    assert 'diagnostic_80_20' in wrapper
    assert 'PAP_BENCH_MPS_PROFILE="${MPS_PROFILE}"' in wrapper
    assert 'PAP_BENCH_MPS_PROFILE:-baseline_70_30' in runner
    assert 'PAP_BENCH_MPS_PROFILE=%q' in runner


def test_pap_only_runner_has_static_mps_baseline_and_diagnostic_alias() -> None:
    wrapper = PAP_WRAPPER.read_text(encoding="utf-8")
    runner = PAP_RUNNER.read_text(encoding="utf-8")

    assert 'PAP_MPS_MODE="${MPS_MODE}"' in wrapper
    assert 'baseline_static_64_28' in wrapper
    assert 'diagnostic_static_64_28' in wrapper
    assert 'PAP_STATIC_PREFILL_CHUNKS="${STATIC_PREFILL_CHUNKS}"' in wrapper
    assert 'PAP_STATIC_ATTENTION_CHUNKS="${STATIC_ATTENTION_CHUNKS}"' in wrapper

    assert 'PAP_MPS_MODE="${PAP_MPS_MODE:-dynamic}"' in runner
    assert 'baseline_static_64_28 | diagnostic_static_64_28)' in runner
    assert 'PAP_STATIC_PREFILL_CHUNKS:-16' in runner
    assert 'PAP_STATIC_ATTENTION_CHUNKS:-7' in runner
    assert 'PAP_MPS_MODE=%q' in runner


def test_static_mps_lifecycle_is_partitioned_and_audited() -> None:
    runner = PAP_RUNNER.read_text(encoding="utf-8")

    assert "nvidia-cuda-mps-control -d -S" in runner
    assert "sm_partition add" in runner
    assert "sm_partition rm" in runner
    assert "CUDA_MPS_SM_PARTITION" in runner
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" in runner
    assert "validate_static_partition_visible_sms" in runner
    assert "mps_static_audit_pa_" in runner


def test_three_lane_runner_does_not_select_diagnostic_mps() -> None:
    orchestrator = (
        ROOT
        / ".claude/skills/vllm-pap-benchmark/scripts/"
        / "run_pd_pap_multiturn_load.sh"
    ).read_text(encoding="utf-8")

    assert "diagnostic_80_20" not in orchestrator
    assert "diagnostic_static_64_28" not in orchestrator
