# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Repeat the frozen failing workload, stopping on any unsuccessful round."""

import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
CAMPAIGN = Path(sys.argv[1]).resolve()
ROUNDS = 10


def event(**data):
    data["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (CAMPAIGN / "events.jsonl").open("a") as f:
        f.write(json.dumps(data) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(data), flush=True)


def checked(args, timeout=30):
    return subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=timeout
    ).stdout.strip()


def main():
    config = json.loads((CAMPAIGN / "campaign.json").read_text())
    decode_gpus = ",".join(str(g) for g in config["decode_gpus"])
    prefill_gpus = ",".join(str(g) for g in config["prefill_gpus"])
    effective_mapping = checked(
        [
            "bash",
            "-c",
            'set -ae; source "$1"; printf "%s" "$DYNAMO_LOCAL_DECODE_GPUS"',
            "bash",
            str(CAMPAIGN / "requested.env"),
        ]
    )
    if effective_mapping != decode_gpus or config["rounds"] != ROUNDS:
        raise RuntimeError("Campaign manifest and replay configuration differ")
    event(event="campaign_started", pid=os.getpid(), rounds=ROUNDS)
    for index in range(1, ROUNDS + 1):
        run = BASE / "runs" / f"{CAMPAIGN.name}_r{index:02d}"
        run.mkdir()
        for name in (
            "requested.env",
            "runner_snapshot.sh",
            "supervise.py",
            "witness.py",
        ):
            (run / name).write_bytes((CAMPAIGN / name).read_bytes())
        remote = f"/home/fei/pd-crash-evidence/{run.name}"
        (run / "RECOVERY.md").write_text(f"""# Fatal reproduction campaign recovery
Campaign {CAMPAIGN.name}; round {index}/{ROUNDS}; run {run.name}.
User explicitly requested attempts to reproduce whole-host crash with detailed monitoring.
Frozen reference: 20260910_crosshost_4p4d_c30_r2, 4P4D C30 180 requests, local Decode GPUs {decode_gpus}, remote Prefill GPUs {prefill_gpus}. Repeated rounds reinitialize workers; no BIOS/PCI configuration changes.
Remote mirror: L20T:{remote}/local-mirror/. Independent remote witness: L20T:{remote}/witness.jsonl.
If status.json absent, round is interrupted/unknown, never passed. After reboot collect prior-boot journal and BMC SEL before rerun. Prior BMC offset +8h to CST. Prior Fatal locations include Slot14/e1/GPU6 and Slot13/f1/GPU7; do not assign the new event's location without its BMC detail.
Read evidence.jsonl, root-pcie-live.log if user started root watcher, kernel-live.log, events.jsonl, controller.log, startup_audit.json, aiperf/logs/aiperf.log and service_logs. Root watcher is optional and may be absent; Linux AER counters can miss firmware-owned errors. Preserve NUL tails. File tail loss remains possible despite fsync/mirroring.
Remote worker namespace pap-cross-host-{run.name}; verify process identity before cleanup. Never blindly signal PIDs after reboot. Campaign stops on first unsuccessful round; no automatic reboot or power control.
""")
        (BASE / "CURRENT_RUN").write_text(str(run) + "\n")
        (CAMPAIGN / "current-round.txt").write_text(str(run) + "\n")
        checked(
            [
                "ssh",
                "-o",
                "ConnectTimeout=5",
                "L20T",
                f"mkdir -p {shlex.quote(remote + '/local-mirror')}",
            ]
        )
        checked(["rsync", "-a", str(run) + "/", f"L20T:{remote}/local-mirror/"])
        witness_path = remote + "/local-mirror/witness.py"
        launch = f"nohup setsid {shlex.quote(str(ROOT / '.venv/bin/python'))} {shlex.quote(witness_path)} {shlex.quote(remote)} > {shlex.quote(remote + '/witness-launch.log')} 2>&1 < /dev/null & echo $!"
        witness_pid = checked(["ssh", "L20T", launch])
        if not witness_pid.isdigit():
            raise RuntimeError("Invalid witness PID")
        (run / "witness.pid").write_text(witness_pid)
        event(event="round_started", index=index, run=str(run), witness_pid=witness_pid)
        env = dict(os.environ, PAP_ROOT=str(ROOT), PD_DIAG_GPU_POLLING="on")
        with (run / "supervisor.log").open("a") as log:
            result = subprocess.run(
                [str(ROOT / ".venv/bin/python"), str(run / "supervise.py"), str(run)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        status_path = run / "status.json"
        status = (
            json.loads(status_path.read_text())
            if status_path.exists()
            else {"accepted": False, "missing_status": True}
        )
        cleanup_path = run / "cleanup_audit.json"
        cleanup = json.loads(cleanup_path.read_text()) if cleanup_path.exists() else {}
        passed = (
            result.returncode == 0
            and status.get("accepted") is True
            and cleanup.get("status") == "passed"
        )
        event(
            event="round_finished",
            index=index,
            returncode=result.returncode,
            passed=passed,
            status=status,
        )
        if not passed:
            (CAMPAIGN / "status.json").write_text(
                json.dumps(
                    {"state": "STOPPED_ON_FAILURE", "index": index, "run": str(run)},
                    indent=2,
                )
            )
            return 2
        stop = f"if test -r /proc/{witness_pid}/cmdline && tr '\\0' ' ' < /proc/{witness_pid}/cmdline | grep -Fq -- {shlex.quote(witness_path)}; then kill -TERM {witness_pid}; fi"
        checked(["ssh", "L20T", stop])
        # The supervisor log is closed and mirrors stopped before final hashing.
        (run / "SHA256SUMS").write_text(
            "".join(
                hashlib.sha256(f.read_bytes()).hexdigest()
                + "  "
                + str(f.relative_to(run))
                + "\n"
                for f in sorted(run.rglob("*"))
                if f.is_file()
                and f.name != "SHA256SUMS"
                and "cuda_cache" not in f.parts
                and "etcd" not in f.parts
            )
        )
        checked(
            [
                "rsync",
                "-a",
                "--exclude=cuda_cache/",
                "--exclude=etcd/",
                str(run) + "/",
                f"L20T:{remote}/local-mirror/",
            ]
        )
        checked(
            [
                "ssh",
                "L20T",
                f"cd {shlex.quote(remote + '/local-mirror')} && sha256sum --check --quiet SHA256SUMS",
            ]
        )
        time.sleep(5)
    (CAMPAIGN / "status.json").write_text(
        json.dumps({"state": "FINISHED_NO_CRASH", "rounds": ROUNDS}, indent=2)
    )
    event(event="campaign_finished_no_crash", rounds=ROUNDS)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        event(event="campaign_error", error=repr(exc))
        raise
