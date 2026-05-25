# PAP Stage-1

This directory contains the current runnable PAP stage-1 experiment. The goal
is a clean single-turn Prefill-Attention-Projection split, not multi-turn KV
reuse.

Roles:

- **Prefill** runs normal vLLM prompt processing and owns prompt paged KV blocks.
- **Attention** is an internal executor colocated with Prefill. It opens
  Prefill paged KV through CUDA IPC for read-side attention computation, and it
  stores decode KV in its own local buffer for the current request.
- **Projection** runs the model decode path and sends current-token Q/K/V to
  Attention. It does not receive Prefill prompt KV bytes.

Stage-1 intentionally excludes request-finalization KV sync, bidirectional KV
pull-back, cross-process KV ownership, prefix attach, and multi-turn cache
reuse. Those directions were removed from this branch until the next
architecture decision is made.

Run a local PAP service:

```bash
bash examples/pap/launch_pap_nixl.sh --model /data/ssd1/llm-models/Qwen3-0.6B
```

Useful environment overrides:

- `PAP_TOPOLOGY=1pa1p` for the fastest local experiment.
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
