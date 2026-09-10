# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Recover persistent panic attachments via the readable system journal."""

import argparse
import datetime
import json
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = subprocess.run(
        [
            "journalctl",
            "--no-pager",
            "--all",
            "-o",
            "json",
            "SYSLOG_IDENTIFIER=systemd-pstore",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    (args.output / "journal-export.jsonl").write_text(result.stdout)
    groups = {}
    for line in result.stdout.splitlines():
        entry = json.loads(line)
        match = re.search(
            r" moved to /var/lib/systemd/pstore/(\d+)/(dmesg-erst-\d+)$",
            entry.get("MESSAGE", ""),
        )
        payload = entry.get("FILE")
        if isinstance(payload, list):
            payload = bytes(payload).decode(errors="replace")
        if not match or not isinstance(payload, str):
            continue
        archive, name = match.groups()
        folder = args.output / archive
        folder.mkdir(exist_ok=True)
        (folder / name).write_text(payload)
        groups.setdefault(archive, {})[name] = payload
    if not groups:
        raise RuntimeError(
            "No readable ERST panic attachments; absence is not a clean bill of health"
        )
    summary = []
    for archive, parts in sorted(groups.items()):
        # ERST Part1 has the newest tail; recover text oldest part first.
        combined = "\n".join(parts[name] for name in sorted(parts, reverse=True))
        (args.output / archive / "recovered-dmesg.txt").write_text(combined)
        errors = [
            line
            for line in combined.splitlines()
            if "[Hardware Error]" in line or "Kernel panic -" in line
        ]
        (args.output / archive / "hardware-errors.txt").write_text(
            "\n".join(errors) + "\n"
        )
        summary.append(
            {
                "archive": archive,
                "device_ids": re.findall(r"device_id: (0000:\S+)", combined),
                "aer_uncor_status": re.findall(
                    r"aer_uncor_status: (0x[0-9a-f]+)", combined
                ),
                "panic": re.findall(r"Kernel panic -[^\n]+", combined),
            }
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "captured_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "archives": len(summary),
                "output": str(args.output.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
