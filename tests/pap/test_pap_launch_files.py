import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PAP_SHELL_ENTRYPOINTS = (
    "benchmarks/pap/aiperf/run_capacity_matrix.sh",
    "benchmarks/pap/aiperf/run_profile.sh",
    "benchmarks/pap/scripts/run_pap_workload.sh",
    "benchmarks/pap/scripts/run_dp_multiturn.sh",
    "benchmarks/pap/scripts/run_pd_multiturn_topology.sh",
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
