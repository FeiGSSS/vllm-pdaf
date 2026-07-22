import os
import subprocess
from pathlib import Path

from benchmarks.pap.profile_env import load_profile, runner_environment


ROOT = Path(__file__).parents[2]
P17_PROFILE = ROOT / "benchmarks/pap/profiles/p17_1pa1p.toml"
P17_RUNNER = ROOT / "benchmarks/pap/scripts/run_p17_1pa1p.sh"
PAP_RUNNER = ROOT / "benchmarks/pap/scripts/run_pap_workload.sh"


def test_pap_only_runner_rejects_removed_mps_selectors() -> None:
    result = subprocess.run(
        ["bash", str(P17_RUNNER), "quick", "c1"],
        cwd=ROOT,
        env={**os.environ, "PAP_LOAD_MPS_PROFILE": "legacy"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "PAP_LOAD_MPS_PROFILE was removed" in result.stderr


def test_pap_only_runner_has_one_static_72_20_profile() -> None:
    environment = runner_environment(load_profile(P17_PROFILE))

    assert environment["PAP_STATIC_PREFILL_CHUNKS"] == "18"
    assert environment["PAP_STATIC_ATTENTION_CHUNKS"] == "5"
    assert environment["PAP_STATIC_PREFILL_EXPECTED_SMS"] == "72"
    assert environment["PAP_STATIC_ATTENTION_EXPECTED_SMS"] == "20"


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

    assert 'client_mode in (' in runner
    assert '"multiturn_load",' in runner
    assert '"aiperf_multiturn",' in runner
