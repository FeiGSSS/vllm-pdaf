#!/usr/bin/env bash
set -euo pipefail
set -a
. "/home/fei/research/PD/vllm-pap/benchmarks/pap/experiments/microbench/PAP-20260910-GDR-P2P-MINIMAL/runs/control_cpu_60s_r3/source.env"
set +a
exec "/home/fei/research/PD/vllm-pap/.venv-dynamo/bin/python" "/home/fei/research/PD/vllm-pap/benchmarks/pap/experiments/microbench/PAP-20260910-GDR-P2P-MINIMAL/probe_gdr_p2p.py" \
  --role source --agent-name pap-src --gpu 0 --where cuda \
  --chunk-mb 1400 --duration-s 300 \
  --meta-file "/home/fei/research/PD/vllm-pap/benchmarks/pap/experiments/microbench/PAP-20260910-GDR-P2P-MINIMAL/runs/control_cpu_60s_r3/meta.json" \
  --heartbeat-file "/home/fei/research/PD/vllm-pap/benchmarks/pap/experiments/microbench/PAP-20260910-GDR-P2P-MINIMAL/runs/control_cpu_60s_r3/source-heartbeat.jsonl"
