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


def test_pap_only_runner_has_explicit_80_20_diagnostic_profile() -> None:
    wrapper = PAP_WRAPPER.read_text(encoding="utf-8")
    runner = PAP_RUNNER.read_text(encoding="utf-8")

    assert 'PAP_LOAD_MPS_PROFILE:-baseline_70_30' in wrapper
    assert 'diagnostic_80_20' in wrapper
    assert 'PAP_BENCH_MPS_PROFILE="${MPS_PROFILE}"' in wrapper
    assert 'PAP_BENCH_MPS_PROFILE:-baseline_70_30' in runner
    assert 'PAP_BENCH_MPS_PROFILE=%q' in runner


def test_three_lane_runner_does_not_select_diagnostic_mps() -> None:
    orchestrator = (
        ROOT
        / ".claude/skills/vllm-pap-benchmark/scripts/"
        / "run_pd_pap_multiturn_load.sh"
    ).read_text(encoding="utf-8")

    assert "diagnostic_80_20" not in orchestrator
