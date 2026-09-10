# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One diagnostic run, with synchronous records and an off-host mirror."""

import datetime
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(os.environ.get("PAP_ROOT", Path(__file__).resolve().parents[5])).resolve()
if not (ROOT / "benchmarks/pap/scripts/run_cross_host_dynamo_7p1d_aiperf.sh").is_file():
    raise ValueError("PAP_ROOT does not identify the experiment repository")
RUN = Path(sys.argv[1]).resolve()
GPU_POLLING = os.environ.get("PD_DIAG_GPU_POLLING", "on")
if GPU_POLLING not in ("on", "off"):
    raise ValueError("PD_DIAG_GPU_POLLING must be on or off")
REMOTE = f"L20T:/home/fei/pd-crash-evidence/{RUN.name}/local-mirror/"
STOP = threading.Event()
MIRROR_FAILED = threading.Event()


def command(args, timeout=8):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "out": p.stdout, "err": p.stderr}
    except subprocess.TimeoutExpired:
        return {"rc": 124, "err": "timeout", "out": ""}


def record(name, data):
    data = dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        monotonic=time.monotonic(),
        **data,
    )
    with (RUN / name).open("a") as f:
        f.write(json.dumps(data) + "\n")
        f.flush()
        os.fsync(f.fileno())


def mirror():
    result = command(
        [
            "rsync",
            "-a",
            "--timeout=8",
            "--exclude=cuda_cache/",
            "--exclude=etcd/",
            str(RUN) + "/",
            REMOTE,
        ],
        12,
    )
    record("mirror.jsonl", result)
    return result["rc"] == 0


def mirror_loop():
    failures = 0
    while not STOP.wait(2):
        failures = 0 if mirror() else failures + 1
        if failures >= 3:
            MIRROR_FAILED.set()
        command(["sync", "-f", str(RUN)], 8)


def snapshot():
    pci = {}
    for b in [
        "00:01.1",
        "10:01.1",
        "60:01.1",
        "70:01.1",
        "80:01.1",
        "90:01.1",
        "e0:01.1",
        "f0:01.1",
        "30:01.1",
        "31:00.0",
        "01:00.0",
        "11:00.0",
        "61:00.0",
        "71:00.0",
        "81:00.0",
        "91:00.0",
        "e1:00.0",
        "f1:00.0",
    ]:
        d = Path("/sys/bus/pci/devices/0000:" + b)
        values = {}
        for f in list(d.glob("aer_*")) + [
            d / "current_link_speed",
            d / "current_link_width",
        ]:
            try:
                values[f.name] = f.read_text()
            except OSError as exc:
                values[f.name] = str(exc)
        pci[b] = values
    if GPU_POLLING == "on":
        gpu = command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,pstate,temperature.gpu,power.draw,utilization.gpu,memory.used,pcie.link.gen.current,pcie.link.width.current",
                "--format=csv",
            ]
        )
        processes = command(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv",
            ]
        )
    else:
        gpu = {"rc": 0, "skipped": "PD_DIAG_GPU_POLLING=off"}
        processes = {"rc": 0, "skipped": "PD_DIAG_GPU_POLLING=off"}
    sensors = {}
    files = list(Path("/sys/class/infiniband/ibp49s0/ports/1/counters").glob("*"))
    files += list(Path("/sys/class/infiniband/ibp49s0/ports/1/hw_counters").glob("*"))
    files += list(Path("/sys/class/hwmon").glob("hwmon*/temp*_input"))
    files += [
        Path("/proc/loadavg"),
        Path("/proc/meminfo"),
        Path("/proc/pressure/io"),
        Path("/proc/pressure/memory"),
    ]
    for f in files:
        try:
            sensors[str(f)] = f.read_text()
        except OSError as exc:
            sensors[str(f)] = str(exc)
    record(
        "evidence.jsonl",
        {"pci": pci, "gpu": gpu, "processes": processes, "sensors": sensors},
    )
    totals = {}
    for b, values in pci.items():
        for k, v in values.items():
            if k.startswith("aer_"):
                for line in v.splitlines():
                    fields = line.split()
                    if fields and fields[-1].isdigit():
                        totals[b + "/" + k + "/" + fields[0]] = int(fields[-1])
    return gpu["rc"] or processes["rc"], totals


def main():
    os.chdir(ROOT)
    record(
        "events.jsonl",
        {
            "event": "preflight",
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "cmdline": Path("/proc/cmdline").read_text(),
            "pid": os.getpid(),
            "gpu_polling": GPU_POLLING,
        },
    )
    for host in [None, "L20T"]:
        args = [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
        if host:
            args = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                host,
                " ".join(args),
            ]
        p = command(args)
        record("preflight.jsonl", {"host": host or "L20B", **p})
        if p["rc"] or p["out"].strip():
            raise RuntimeError("GPU occupancy or preflight failure")
    for name, args in {
        "topology.txt": ["nvidia-smi", "topo", "-m"],
        "pci-tree.txt": ["lspci", "-D", "-t"],
        "kernel-before.log": [
            "journalctl",
            "-k",
            "-b",
            "--no-pager",
            "-o",
            "short-iso-precise",
        ],
        "processes-before.txt": ["ps", "-eo", "pid,ppid,pgid,lstart,args"],
    }.items():
        result = command(args)
        (RUN / name).write_text(result["out"] + result["err"])
    rc, baseline = snapshot()
    if rc or not mirror():
        raise RuntimeError("Baseline/mirror failed: refusing workload")
    kernel_file = (RUN / "kernel-live.log").open("a")
    kernel = subprocess.Popen(
        ["journalctl", "-k", "-f", "-n", "0", "-o", "short-iso-precise"],
        stdout=kernel_file,
        stderr=subprocess.STDOUT,
    )
    thread = threading.Thread(target=mirror_loop, daemon=True)
    thread.start()
    log = (RUN / "controller.log").open("a")
    env = dict(
        os.environ,
        DYNAMO_RUN_ID=RUN.name,
        DYNAMO_RUN_ROOT=str(RUN),
        DYNAMO_REMOTE_RUN_ROOT=str(
            ROOT / "benchmarks/pap/experiments/_staging" / (RUN.name + "_remote")
        ),
        PAP_ROOT=str(ROOT),
        PYTHONUNBUFFERED="1",
    )
    runner = subprocess.Popen(
        [
            "bash",
            "-c",
            'set -ae; source "$DYNAMO_RUN_ROOT/requested.env"; exec bash "$DYNAMO_RUN_ROOT/runner_snapshot.sh"',
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (RUN / "runner.pid").write_text(str(runner.pid))
    record("events.jsonl", {"event": "runner_started", "pid": runner.pid})
    started = time.monotonic()
    failure = None
    stopping = None
    while runner.poll() is None:
        rc, totals = snapshot()
        delta = {
            k: v - baseline.get(k, 0)
            for k, v in totals.items()
            if v > baseline.get(k, 0)
        }
        kernel_file.flush()
        os.fsync(kernel_file.fileno())
        log.flush()
        os.fsync(log.fileno())
        if stopping is None and (
            rc or delta or MIRROR_FAILED.is_set() or time.monotonic() - started > 1200
        ):
            failure = {
                "gpu_rc": rc,
                "aer_increases": delta,
                "mirror_failed": MIRROR_FAILED.is_set(),
                "deadline": time.monotonic() - started > 1200,
            }
            record("events.jsonl", {"event": "stop_requested", **failure})
            os.killpg(runner.pid, signal.SIGTERM)
            stopping = time.monotonic()
        if stopping is not None and time.monotonic() - stopping > 90:
            record(
                "events.jsonl",
                {
                    "event": "cleanup_timeout",
                    "warning": "Check worker ownership before cleanup",
                },
            )
            break
        time.sleep(1)
    kernel.terminate()
    kernel.wait(timeout=5)
    kernel_file.close()
    code = runner.poll()
    accepted = code == 0 and failure is None and (RUN / "completion_audit.env").exists()
    status = {
        "state": "FINISHED",
        "accepted": accepted,
        "runner_exit": code,
        "failure": failure,
    }
    (RUN / "status.json").write_text(json.dumps(status, indent=2))
    record("events.jsonl", status)
    STOP.set()
    thread.join(timeout=20)
    command(["sync", "-f", str(RUN)])
    mirror()
    return 0 if accepted else 2


if __name__ == "__main__":
    sys.exit(main())
