# PAP

This directory contains thin launchers and request examples for the
Prefill-Attention-Projection service. Gateway implementation lives in
`vllm/pap/gateway/`; Attention implementation lives in `vllm/pap/attention/`.
The current release gate is the P17 Qwen3-8B, same-host `1PA1P` path, validated
through the owner-specific `vllm/pap/integration/` boundary. Same-host and
cross-host `xPAyP` implementations remain available but are not revalidated by
this milestone. The validated multi-turn path reuses Prefill-owned KV through
vLLM's native prefix cache; it does not keep an Attention session resident
between turns.

See [`docs/design/pap/`](../../docs/design/pap/README.md) for the canonical
architecture, runtime, and validation boundary.

Roles:

- **Prefill** runs normal vLLM prompt processing and owns prompt paged KV blocks.
- **Attention** is an internal service colocated with Prefill. It opens
  Prefill paged KV through CUDA IPC and appends decode K/V directly to those
  Prefill-owned blocks.
- **Projection** runs the model decode path and sends current-token Q/K/V to
  Attention. It does not receive Prefill prompt KV bytes.

Each Attention session sends decode commits and lease releases back to the
Prefill in its own PA group. Attention creates a separate lazy mailbox
transport for every Projection peer, so PA and Projection counts do not need to
match. The default independent round-robin policy uses all configured nodes;
`PAP_ROUTING_POLICY=projection_affinity` remains available for a static
PA-to-Projection mapping.

Run a local PAP service:

```bash
bash examples/pap/launch_pap_nixl.sh --model /data/ssd1/llm-models/Qwen3-0.6B
```

Useful environment overrides:

- `PAP_TOPOLOGY=1pa1p` for the fastest local experiment.
- `PAP_TOPOLOGY=3pa2p` (or another positive `<x>pa<y>p` value) to configure the
  PA-to-Projection ratio.
- `PAP_PREFILL_GPUS=0,1,2` and `PAP_PROJECTION_GPUS=3,4` to place a `3pa2p`
  topology explicitly.
- `PAP_SERVICE_ONLY=1` to keep services running without the built-in smoke
  request.
- `PAP_SKIP_SMOKE_REQUEST=1` to skip the launcher request.
- `PAP_PROXY_PORT=9000` to choose the OpenAI-compatible proxy port.

Send one request through the proxy:

```bash
.venv/bin/python examples/pap/run_one_request.py \
  --host 127.0.0.1 \
  --port 9000 \
  --model /data/ssd1/llm-models/Qwen3-0.6B \
  --prompt "Summarize the PAP stage-1 data path." \
  --max-tokens 8
```

Runtime logs are written under `examples/pap/logs/`, which is ignored by git.

The self-contained correctness/performance runner accepts the same topology:

```bash
PAP_TOPOLOGY=3pa2p \
PAP_PREFILL_GPUS=0,1,2 \
PAP_PROJECTION_GPUS=3,4 \
bash benchmarks/pap/scripts/run_pap_workload.sh
```

Each run records `topology_manifest.json`, `routing_audit.json`, strict log
audit results, and an all-Attention session-drain result.
