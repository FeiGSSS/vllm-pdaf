# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run isolated PAP diagnosis with explicit, retained effective configuration."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def process_audit(run, root):
    rows = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            env = dict(
                item.split("=", 1)
                for item in (proc / "environ").read_text().split("\0")
                if "=" in item
            )
            if env.get("RUN_ID") != run.name:
                continue
            cmd = (proc / "cmdline").read_text().replace("\0", " ").strip()
            rows.append(
                {
                    "pid": int(proc.name),
                    "cmd": cmd,
                    "cwd": str((proc / "cwd").resolve()),
                    "gpu": env.get("CUDA_VISIBLE_DEVICES"),
                    "partition": env.get("CUDA_MPS_SM_PARTITION"),
                }
            )
        except (OSError, ValueError):
            continue
    engines = [r for r in rows if r["cmd"].startswith("VLLM::EngineCore")]
    attention = [r for r in rows if " -m vllm.pap.service " in r["cmd"]]
    servers = [r for r in rows if "/bin/vllm serve " in r["cmd"]]
    assert len(engines) == 8 and len(attention) == 7 and len(servers) == 8, (
        "Wrong live topology"
    )
    assert all(r["cwd"] == str(root) for r in engines + attention + servers)
    for pa in range(7):
        for role in ("prefill", "attention"):
            context = json.loads(
                (run / f"runtime_cuda_context_{role}_{pa}.json").read_text()
            )
            assert any(r["pid"] == context["pid"] for r in engines + attention)
    (run / "process_snapshot.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--pat-root", type=Path)
    parser.add_argument("--prefill-budget", type=int, default=2048)
    parser.add_argument("--attention-chunks", type=int, default=3)
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parent
    root = experiment.parents[4]
    run = experiment / "runs" / args.name
    run.mkdir(parents=True, exist_ok=False)
    dataset = (
        root
        / "benchmarks/pap/datasets/agentic-code"
        / "s60-t3-native32k-stratified-seed42/dataset.jsonl"
    )
    assert (
        hashlib.sha256(dataset.read_bytes()).hexdigest()
        == "5d4357a9a037e9e1a3c25acf402ba06afd1c9e7643678da225b9f5cc63cb257c"
    )
    env = {k: os.environ[k] for k in ("HOME", "USER", "PATH") if k in os.environ}
    env.update(
        {
            "LANG": "C.UTF-8",
            "PYTHONPATH": str(root),
            "PAP_ROOT": str(root),
            "MODEL_PATH": "/data/ssd1/llm-models/Qwen3-8B",
            "PAP_TOPOLOGY": "7pa1p",
            "PAP_PREFILL_GPUS": "0,1,2,3,4,5,6",
            "PAP_PROJECTION_GPUS": "7",
            "PAP_ROUTING_POLICY": "dynamo",
            "PAP_VLLM_DTYPE": "float16",
            "MAX_MODEL_LEN": "32768",
            "PAP_HF_OVERRIDES": "",
            "PAP_PREFILL_MAX_NUM_BATCHED_TOKENS": str(args.prefill_budget),
            "PAP_PREFILL_MAX_NUM_SEQS": "256",
            "PAP_PROJECTION_MAX_NUM_BATCHED_TOKENS": "256",
            "PAP_PROJECTION_MAX_NUM_SEQS": "256",
            "PAP_PREFILL_ASYNC_SCHEDULING": "off",
            "PAP_PROJECTION_ASYNC_SCHEDULING": "on",
            "PAP_PREFILL_GPU_MEMORY_UTILIZATION": "0.90",
            "PAP_BLOCK_SIZE": "16",
            "PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS": "256",
            "PAP_DYNAMO_PREFILL_LOAD_SCALE": "2.0",
            "PAP_AIPERF_INPUT_FILE": str(dataset),
            "PAP_AIPERF_SESSIONS": "60",
            "PAP_AIPERF_EXPECTED_REQUESTS": "180",
            "PAP_AIPERF_CONCURRENCY": "30",
            "PAP_AIPERF_TIMING_MODE": "concurrency",
            "PAP_AIPERF_CUSTOM_DATASET_TYPE": "mooncake-trace",
            "PAP_PREFILL_IPC_PROFILE": str(int(args.trace)),
            "PAP_RUNTIME_CUDA_CONTEXT_AUDIT": "1",
            "PAP_BENCH_STRICT_CORRECTNESS_AUDIT": "1",
            "BENCH_TIMEOUT": "1800",
            "PAP_STATIC_PREFILL_CHUNKS": str(23 - args.attention_chunks),
            "PAP_STATIC_ATTENTION_CHUNKS": str(args.attention_chunks),
            "PAP_STATIC_PREFILL_EXPECTED_SMS": str(4 * (23 - args.attention_chunks)),
            "PAP_STATIC_ATTENTION_EXPECTED_SMS": str(4 * args.attention_chunks),
            "RUN_ROOT": str(run),
            "RUN_ID": args.name,
        }
    )
    if args.pat_root:
        env["PYTHONPATH"] = str(args.pat_root.resolve()) + os.pathsep + str(root)
    if args.trace:
        env.update(
            {
                "PAP_PROJECTION_PA_TRACE_OUTPUT": str(run / "pa_trace.pt"),
                "PAP_PROJECTION_PA_TRACE_RING_STEPS": "8192",
                "PAP_PROJECTION_PA_TRACE_SAMPLES": "8191",
                "PAP_PROJECTION_PA_TRACE_FLUSH_SECONDS": "5",
            }
        )
    (run / "requested.json").write_text(json.dumps(env, indent=2) + "\n")
    (run / "protocol.py").write_bytes(Path(__file__).read_bytes())
    with (run / "preflight.log").open("w") as log:
        subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    assert not (run / "preflight.log").read_text().strip(), "GPUs occupied"
    with (run / "source_import.json").open("w") as log:
        subprocess.run(
            [
                str(root / ".venv/bin/python"),
                "-c",
                "import json,vllm,vllm.pap.integration.engine,prefix_attn; "
                "print(json.dumps({'vllm':vllm.__file__,"
                "'engine':vllm.pap.integration.engine.__file__,"
                "'prefix_attn':prefix_attn.__file__}))",
            ],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    start = time.time()
    collector = None
    if args.trace:
        with (run / "collector.log").open("w") as log:
            collector = subprocess.Popen(
                [
                    str(root / ".venv/bin/python"),
                    str(experiment / "collect.py"),
                    str(run / "prefill_metrics.jsonl"),
                    str(run / "collector.stop"),
                ],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    with (run / "runner.log").open("w") as log:
        child = subprocess.Popen(
            ["bash", "benchmarks/pap/scripts/run_pap_workload.sh"],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        audited = False
        try:
            while child.poll() is None:
                if not audited and (run / "topology_manifest.json").exists():
                    process_audit(run, root)
                    audited = True
                time.sleep(1)
        except BaseException:
            child.terminate()
            child.wait(timeout=120)
            raise
        finally:
            if collector is not None:
                (run / "collector.stop").touch()
                collector.wait(timeout=10)
    (run / "exit.json").write_text(
        json.dumps(
            {
                "exit_code": child.returncode,
                "start": start,
                "end": time.time(),
                "process_audit": audited,
            }
        )
        + "\n"
    )
    print(f"exit={child.returncode} artifacts={run}", flush=True)
    assert audited or child.returncode != 0, "No live topology audit"
    raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
