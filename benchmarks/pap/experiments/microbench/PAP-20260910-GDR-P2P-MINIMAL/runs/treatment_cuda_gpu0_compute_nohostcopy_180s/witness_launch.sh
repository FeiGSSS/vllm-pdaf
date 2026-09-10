#!/usr/bin/env bash
set -euo pipefail
exec "/home/fei/research/PD/vllm-pap/.venv-dynamo/bin/python" "/home/fei/research/PD/vllm-pap/benchmarks/pap/experiments/microbench/PAP-20260910-GDR-P2P-MINIMAL/runs/treatment_cuda_gpu0_compute_nohostcopy_180s/witness_liveness.py" \
  --target "10.128.7.24" --ports 22 --out "/home/fei/research/PD/vllm-pap/benchmarks/pap/experiments/microbench/PAP-20260910-GDR-P2P-MINIMAL/runs/treatment_cuda_gpu0_compute_nohostcopy_180s/witness.jsonl" \
  --seconds 480
