# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture reproducibility evidence without copying environment secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def command(*args: str) -> bytes:
    return subprocess.check_output(args, cwd=ROOT, stderr=subprocess.STDOUT)


def capture(destination: Path, model: Path, environments: list[str]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    commands = {
        "git_commit.txt": ("git", "rev-parse", "HEAD"),
        "git_status.txt": ("git", "status", "--short"),
        "source.patch": ("git", "diff", "--binary", "HEAD"),
        "gpu.txt": ("nvidia-smi", "-q"),
        "gpu_topology.txt": ("nvidia-smi", "topo", "-m"),
        "cpu.txt": ("lscpu",),
        "kernel.txt": ("uname", "-a"),
    }
    for filename, args in commands.items():
        (destination / filename).write_bytes(command(*args))

    # Include untracked implementation files and datasets, but not ignored runs,
    # virtual environments or build products. Git deletions stay in source.patch.
    paths = command(
        "git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"
    )
    with tarfile.open(destination / "source.tar.gz", "w:gz") as archive:
        for name in sorted(set(paths.decode().split("\0")) - {""}):
            path = ROOT / name
            if path.exists() or path.is_symlink():
                if destination.resolve() in path.resolve().parents:
                    raise ValueError("snapshot output must be ignored by Git")
                archive.add(path, arcname=name, recursive=False)

    packages_code = (
        "import importlib.metadata as m, json, sys; "
        "print(json.dumps({'python':sys.version, 'executable':sys.executable, "
        "'packages':sorted([{'name':d.metadata['Name'], 'version':d.version} "
        "for d in m.distributions()], key=lambda d:d['name'].lower())}, indent=2))"
    )
    for environment in environments:
        if environment not in {".venv", ".venv-aiperf", ".venv-dynamo"}:
            raise ValueError(f"unknown project environment: {environment}")
        output = command(
            str(ROOT / environment / "bin/python"), "-I", "-c", packages_code
        )
        (destination / f"packages{environment}.json").write_bytes(output)

    model_files = {}
    router_runtime = ROOT / ".local/pap-dynamo-router"
    if router_runtime.is_dir():
        runtime_metadata = destination / "pap_dynamo_router"
        runtime_metadata.mkdir()
        for name in ("build.txt", "pap_dynamo_router.abi3.so"):
            source = router_runtime / name
            # This small CPU-only dependency is not represented by pip freeze.
            shutil.copy2(source, runtime_metadata / name)

    model_metadata = destination / "model_metadata"
    model_metadata.mkdir()
    for path in sorted(model.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        model_files[path.name] = {"bytes": path.stat().st_size, "sha256": digest}
        if path.suffix in {".json", ".txt", ".model", ".jinja"}:
            shutil.copy2(path, model_metadata / path.name)
    (destination / "model_files.json").write_text(
        json.dumps({"model_path": str(model.resolve()), "files": model_files}, indent=2)
        + "\n"
    )
    (destination / "COMPLETE").write_text("captured\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--environments", nargs="+", default=[".venv", ".venv-aiperf"])
    args = parser.parse_args()
    capture(args.destination, args.model, args.environments)


if __name__ == "__main__":
    main()
