# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAP_SHELL_ENTRYPOINTS = (
    "benchmarks/pap/scripts/run_aiperf_profile.sh",
    "benchmarks/pap/scripts/run_pap_workload.sh",
    "benchmarks/pap/scripts/run_dynamo_workload.sh",
    "benchmarks/pap/scripts/build_pap_dynamo_router.sh",
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


@pytest.mark.parametrize(
    "setting,value,error",
    [
        (f"DYNAMO_{role}_ASYNC_SCHEDULING", "unknown", "async scheduling must be")
        for role in ("AGG", "PREFILL", "DECODE")
    ]
    + [
        ("DYNAMO_NAMESPACE", "invalid.namespace", "DYNAMO_NAMESPACE must contain"),
        ("DYNAMO_ROUTER_PREFILL_LOAD_SCALE", "nan", "must be a nonnegative number"),
    ],
)
def test_dynamo_rejects_invalid_configuration_before_startup(
    tmp_path, setting, value, error
):
    run_dir = tmp_path / "run"
    result = subprocess.run(
        ["bash", str(ROOT / "benchmarks/pap/scripts/run_dynamo_workload.sh")],
        env={
            "PATH": os.environ["PATH"],
            "DYNAMO_RUN_ROOT": str(run_dir),
            setting: value,
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert error in result.stderr
    assert not run_dir.exists()


def test_dynamo_replay_rejects_changed_dataset_before_startup(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("[]\n")
    run_dir = tmp_path / "run"
    result = subprocess.run(
        ["bash", str(ROOT / "benchmarks/pap/scripts/run_dynamo_workload.sh")],
        env={
            "PATH": os.environ["PATH"],
            "DYNAMO_RUN_ROOT": str(run_dir),
            "DYNAMO_PYTHON": sys.executable,
            "PAP_PYTHON": sys.executable,
            "AIPERF_BIN": sys.executable,
            "MODEL_PATH": str(tmp_path),
            "DYNAMO_AIPERF_INPUT_FILE": str(dataset),
            "DYNAMO_AIPERF_INPUT_SHA256": "0" * 64,
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "replay dataset checksum mismatch" in result.stderr
    assert not run_dir.exists()


@pytest.mark.parametrize(
    "entrypoint",
    [
        "benchmarks/pap/scripts/run_pap_workload.sh",
        "benchmarks/pap/experiments/e2e/PAP-20260903-AGENTIC-CODE-QPS-MATRIX/run.sh",
    ],
)
def test_pap_runner_rejects_retired_routing_before_startup(tmp_path, entrypoint):
    run_dir = tmp_path / "run"
    result = subprocess.run(
        ["bash", str(ROOT / entrypoint)],
        env={
            "PATH": os.environ["PATH"],
            "PAP_ROUTING_POLICY": "round_robin",
            "RUN_ROOT": str(run_dir),
            "PAP_QPS_SCAN_RUN_ROOT": str(run_dir),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "only Dynamo" in result.stderr
    assert not run_dir.exists()


def test_multiturn_generation_shares_prefixes_without_merging_conversations():
    from benchmarks.pap.datasets.tools.generate_multiturn_dataset import build_records

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text.encode())

        def decode(self, tokens, **kwargs):
            return bytes(tokens).decode()

    records, summary = build_records(
        Tokenizer(),
        "a" * 128,
        sessions=2,
        turns=2,
        document_tokens=32,
        append_tokens=16,
        output_tokens=4,
        session_prefix="test",
    )
    assert records[0]["session_id"] != records[1]["session_id"]
    assert records[0]["turns"] == records[1]["turns"]
    assert all(
        "cache_salt" not in turn["extra"] for row in records for turn in row["turns"]
    )
    assert summary["prefix_cache_policy"] == "shared_across_sessions"


def test_shared_prefix_fixture_derivation_preserves_all_non_salt_fields(tmp_path):
    from benchmarks.pap.datasets.tools.derive_shared_prefix_dataset import derive

    source = ROOT / "benchmarks/pap/datasets/long-context/qwen3-8b-yarn131k"
    destination = tmp_path / "shared"
    manifest = derive(source, destination)
    for name in manifest["files"]:
        original = [
            json.loads(line) for line in (source / name).read_text().splitlines()
        ]
        shared = [
            json.loads(line) for line in (destination / name).read_text().splitlines()
        ]
        for row in original:
            for turn in row["turns"]:
                del turn["extra"]["cache_salt"]
        assert shared == original
    with pytest.raises(FileExistsError):
        derive(source, destination)
