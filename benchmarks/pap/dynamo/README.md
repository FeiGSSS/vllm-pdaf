# Official Dynamo PD baseline

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
    -r benchmarks/pap/dynamo/requirements.txt

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

## Smoke test

The zero-dependency smoke uses file discovery and Dynamo round-robin routing:

```bash
DYNAMO_PD_TOPOLOGY=1p1d \
DYNAMO_PD_ROUTER_MODE=round-robin \
DYNAMO_PD_DISCOVERY_BACKEND=file \
DYNAMO_PD_SMOKE_ONLY=1 \
  bash benchmarks/pap/scripts/run_dynamo_pd_workload.sh
```

## AIPerf run

KV-aware routing uses an ephemeral local etcd process for discovery. Dynamo,
vLLM, and etcd all run directly on the host; the runner removes the etcd data
when its run directory is removed.

```bash
DYNAMO_PD_TOPOLOGY=6p2d \
DYNAMO_PD_ROUTER_MODE=kv \
DYNAMO_PD_AIPERF_INPUT_FILE=/path/to/multiturn.jsonl \
DYNAMO_PD_AIPERF_SESSIONS=128 \
DYNAMO_PD_AIPERF_CONCURRENCY=32 \
  bash benchmarks/pap/scripts/run_dynamo_pd_workload.sh
```

The worker configuration matches the current PD comparison lane: Qwen3-8B
FP16, TP1, audited piecewise CUDA Graph execution, 32K model length, block
size 16, `gpu_memory_utilization=0.90`, and 2,048 batched tokens for both
Prefill and Decode. The launcher fails before the workload if any worker
enables eager execution or does not finish Graph capture.

The runner fails closed onto the project UCX 1.22 same-node runtime with CUDA
IPC GET zero-copy. It requests NIXL cross-layer block coalescing, although
vLLM 0.26 transfer metrics still report tens of thousands of descriptors for
long requests. The reliable runtime gate is measured transfer throughput:
full runs fail if aggregate KV transfer throughput is below 5 GB/s.

The benchmark launcher intentionally has no eager switch. Fast-start eager
diagnostics must use a separate ad hoc command and are not valid comparison
runs.
