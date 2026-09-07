# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one isolated, audited static-MPS Attention measurement."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--sms", type=int, choices=(12, 20), default=12)
    parser.add_argument("--ncu", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--pat-root", type=Path)
    parser.add_argument("--cache-mode", choices=("warm", "rotating"), default="warm")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[4]
    run = here / "runs" / args.name
    run.mkdir(parents=True, exist_ok=False)
    env = {k: os.environ[k] for k in ("HOME", "USER", "PATH") if k in os.environ}
    env.update(
        {
            "PYTHONPATH": str(root),
            "DISABLE_STREAM": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
            "CUDA_MPS_PIPE_DIRECTORY": tempfile.mkdtemp(prefix="pap-bw-mps-"),
            "CUDA_MPS_LOG_DIRECTORY": str(run / "mps-log"),
        }
    )
    if args.pat_root:
        env["PYTHONPATH"] = str(args.pat_root.resolve()) + os.pathsep + str(root)
    for script in ("run_probe.py", "probe.py"):
        (run / script).write_bytes((here / script).read_bytes())
    for name in ("mps-log",):
        (run / name).mkdir()
    idle = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    (run / "preflight.txt").write_text(idle)
    assert not idle.strip(), "GPUs have active clients"
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    partitions = []
    subprocess.run(["nvidia-cuda-mps-control", "-d", "-S"], env=env, check=True)

    def control(command):
        text = subprocess.check_output(
            ["nvidia-cuda-mps-control"],
            input=command + "\n",
            text=True,
            env=env,
            timeout=15,
        )
        with (run / "mps-control.log").open("a") as log:
            log.write(command + "\n" + text + "\n")
        return text

    try:
        for chunks in (23 - args.sms // 4, args.sms // 4):
            response = control(f"sm_partition add {uuid} {chunks}")
            lines = [
                line.removeprefix("Partition ").removesuffix(" created")
                for line in response.splitlines()
                if line.startswith("Partition ")
            ]
            assert len(lines) == 1 and lines[0].startswith(uuid + "/"), response
            partitions.append(lines[0])
        env["CUDA_MPS_SM_PARTITION"] = partitions[-1]
        control("lspart")
        command = [
            str(root / ".venv/bin/python"),
            str(here / "probe.py"),
            "--expected-sms",
            str(args.sms),
            "--cache-mode",
            args.cache_mode,
            "--output",
            str(run / "measurement.json"),
        ]
        if args.case:
            command += ["--case", args.case]
        if args.ncu:
            command += ["--profile"]
            command = [
                str(args.ncu),
                "--mps",
                "control",
                "--mps-num-clients",
                "1",
                "--mps-timeout",
                "60",
                "--replay-mode",
                "range",
                "--nvtx",
                "--nvtx-include",
                "pap_bandwidth/",
                "--launch-count",
                "1",
                "--clock-control",
                "none",
                "--cache-control",
                "none",
                "--metrics",
                "dram__bytes_read.sum,dram__bytes_write.sum,"
                "gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "-o",
                str(run / "counters"),
                *command,
            ]
        (run / "protocol.json").write_text(
            json.dumps(
                {"env": env, "command": command, "physical_uuid": uuid}, indent=2
            )
        )
        with (run / "worker.log").open("w") as log:
            child = subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300,
            )
        (run / "exit.json").write_text(json.dumps({"exit_code": child.returncode}))
        if child.returncode:
            raise RuntimeError(f"probe failed: {run / 'worker.log'}")
        data = json.loads((run / "measurement.json").read_text())
        assert data["visible_sms"] == args.sms
        assert data["partition"] == partitions[-1]
    finally:
        for partition in reversed(partitions):
            control(f"sm_partition rm {uuid} {partition[len(uuid) + 1 :]}")
        control("quit")
    print(run)


if __name__ == "__main__":
    main()
