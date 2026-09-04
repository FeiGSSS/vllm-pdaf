# Official Dynamo DP8 and 6P2D baselines

This lane isolates the official Dynamo and vLLM distributions from the PAP
development environment. It reuses the project-wide PyPI AIPerf 0.11
environment at `.venv-aiperf` so PAP and Dynamo receive requests from the
same client.

## Environment

```bash
UV_CACHE_DIR=/tmp/uv-cache-dynamo \
  uv venv --python 3.12 --seed .venv-dynamo
UV_CACHE_DIR=/tmp/uv-cache-dynamo \
  uv pip install --python .venv-dynamo/bin/python \
    -r benchmarks/pap/scripts/dynamo-requirements.txt

curl -L --fail -o /tmp/etcd-v3.6.1-linux-amd64.tar.gz \
  https://github.com/etcd-io/etcd/releases/download/v3.6.1/etcd-v3.6.1-linux-amd64.tar.gz
mkdir -p .local/dynamo-etcd-3.6.1
tar -xzf /tmp/etcd-v3.6.1-linux-amd64.tar.gz \
  --strip-components=1 -C .local/dynamo-etcd-3.6.1
```

The runner uses Python safe-path mode. This is mandatory because launching
from the repository root would otherwise import the PAP checkout's `vllm/`
package instead of the official vLLM wheel in `.venv-dynamo`.

`VLLM_USE_FLASHINFER_SAMPLER=0` is shared with the PAP benchmark lane. The
host's default CUDA 12.0 compiler cannot JIT FlashInfer 0.6 sampling kernels
against the official CUDA 13 headers; vLLM's native sampler avoids that
unrelated host-toolkit mismatch.

## Smoke tests

The default baseline uses etcd discovery and KV-aware routing. Select only the
worker architecture:

```bash
DYNAMO_ARCHITECTURE=dp8 DYNAMO_SMOKE_ONLY=1 \
  bash benchmarks/pap/scripts/run_dynamo_workload.sh
DYNAMO_ARCHITECTURE=6p2d DYNAMO_SMOKE_ONLY=1 \
  bash benchmarks/pap/scripts/run_dynamo_workload.sh
```

## AIPerf run

KV-aware routing uses an ephemeral local etcd process for discovery. Dynamo,
vLLM, and etcd all run directly on the host; the runner removes the etcd data
when its run directory is removed.

```bash
bash benchmarks/pap/experiments/e2e/\
PAP-20260903-AGENTIC-CODE-QPS-MATRIX/run.sh
```

Formal runs invoke the shared Dynamo runner through an experiment-local
`run.sh` and `experiment.env`. The shared runner has no default dataset, QPS,
concurrency, warmup, or duration; a non-smoke direct invocation fails closed
unless those settings are supplied explicitly.

Both architectures use Qwen3-8B FP16, TP1, audited piecewise CUDA Graphs,
131K static YaRN, block size 16, and `gpu_memory_utilization=0.90`. DP8 uses
eight aggregated workers; 6P2D uses six Prefill and two Decode workers. The
launcher fails before the workload if any worker enables eager execution or
does not finish Graph capture.

Only 6P2D initializes the project UCX 1.22 same-node runtime and NIXL CUDA IPC
GET zero-copy. Its reliable runtime gate is measured transfer throughput: a
full 6P2D run fails if aggregate KV transfer throughput is below 5 GB/s. DP8
has no KV transfer and records that audit as not applicable.

The benchmark launcher intentionally has no eager switch. Fast-start eager
diagnostics must use a separate ad hoc command and are not valid comparison
runs.
