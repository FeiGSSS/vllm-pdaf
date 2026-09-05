# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAP_SHELL_ENTRYPOINTS = (
    "benchmarks/pap/scripts/run_aiperf_profile.sh",
    "benchmarks/pap/scripts/run_pap_workload.sh",
    "benchmarks/pap/scripts/run_dynamo_workload.sh",
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


def test_pap_replay_requires_existing_dataset_before_creating_run(tmp_path):
    run_dir = tmp_path / "run"
    result = subprocess.run(
        ["bash", str(ROOT / "benchmarks/pap/scripts/run_pap_workload.sh")],
        env={
            "PATH": os.environ["PATH"],
            "PAP_ROOT": str(ROOT),
            "PAP_AIPERF_SESSIONS": "1",
            "PAP_AIPERF_EXPECTED_REQUESTS": "2",
            "PAP_AIPERF_INPUT_FILE": str(tmp_path / "missing.jsonl"),
            "RUN_ROOT": str(run_dir),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "existing immutable dataset" in result.stderr
    assert not run_dir.exists()
