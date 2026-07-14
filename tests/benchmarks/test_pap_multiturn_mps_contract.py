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


def test_pap_only_runner_has_one_static_64_28_profile() -> None:
    environment = runner_environment(load_profile(P17_PROFILE))

    assert environment["PAP_STATIC_PREFILL_CHUNKS"] == "16"
    assert environment["PAP_STATIC_ATTENTION_CHUNKS"] == "7"
    assert environment["PAP_STATIC_PREFILL_EXPECTED_SMS"] == "64"
    assert environment["PAP_STATIC_ATTENTION_EXPECTED_SMS"] == "28"


def test_static_mps_lifecycle_is_partitioned_and_audited() -> None:
    runner = PAP_RUNNER.read_text(encoding="utf-8")

    assert "nvidia-cuda-mps-control -d -S" in runner
    assert "sm_partition add" in runner
    assert "sm_partition rm" in runner
    assert "CUDA_MPS_SM_PARTITION" in runner
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in runner
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
