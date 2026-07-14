import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PAP_SHELL_ENTRYPOINTS = (
    ".claude/skills/vllm-pap-benchmark/scripts/bootstrap_pd_multiturn_reference.sh",
    ".claude/skills/vllm-pap-benchmark/scripts/run_multiturn_north_star.sh",
    ".claude/skills/vllm-pap-benchmark/scripts/run_pap_multiturn_load.sh",
    ".claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh",
    ".claude/skills/vllm-pap-benchmark/scripts/run_pd_same_workload.sh",
    "benchmarks/pap/scripts/run_p17_1pa1p.sh",
    "benchmarks/pap/scripts/run_pap_workload.sh",
    "benchmarks/disagg_benchmarks/run_pap_128_testbed.sh",
    "examples/pap/launch_pap_nixl.sh",
)


@pytest.mark.parametrize("relative_path", PAP_SHELL_ENTRYPOINTS)
def test_pap_shell_entrypoint_has_valid_syntax(relative_path: str) -> None:
    script = ROOT / relative_path
    subprocess.run(
        ["bash", "-n", str(script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
