# PAP

This directory contains thin launchers and request examples for the
Prefill-Attention-Projection service. Gateway implementation lives in
`vllm/pap/gateway/`; Attention implementation lives in `vllm/pap/attention/`.
`launch_pap_nixl.sh` is a functional NIXL/control-plane smoke launcher. Its
small-model, legacy percentage-MPS defaults are not a performance or regression
configuration. Use the AIPerf runner for the validated same-host
`local_fast`/80-12-SM path.
The current runtime testbed is the eight-GPU AIPerf matrix, validated through
the owner-specific `vllm/pap/integration/` boundary. Same-host `xPAyP` has a
controlled correctness smoke; cross-host `xPAyP` remains available but is not
an active E2E lane. The validated multi-turn path reuses
Prefill-owned KV through vLLM's native prefix cache; it does not keep an
Attention session resident between turns.

See the [current development status](../../docs/design/pap/status.md) and
[`docs/design/pap/`](../../docs/design/pap/README.md) for the canonical
architecture, runtime, and validation boundary.

Roles:

- **Prefill** runs normal vLLM prompt processing and owns prompt paged KV blocks.
- **Attention** is an internal service colocated with Prefill. It opens
  Prefill paged KV through CUDA IPC and appends decode K/V directly to those
  Prefill-owned blocks. The current path runs vLLM's SM-aware Triton
  paged-decode specialization with one workspace shared by all layers in a
  decode step. A payload-free step descriptor lets it prepare QKV-independent
  state before layer-0 QKV arrives.
- **Projection** runs the model decode path and sends current-token Q/K/V to
  Attention. It does not receive Prefill prompt KV bytes. PAP keeps each vLLM
  scheduler batch intact; requests for different PA groups are same-step
  shards, not independently pipelined microbatches. vLLM async scheduling may
  still prepare the next step on the CPU.

Each Attention session sends decode commits and lease releases back to the
Prefill in its own PA group. Attention creates a separate lazy mailbox
transport for every Projection peer, so PA and Projection counts do not need to
match. The default independent round-robin policy uses all configured nodes;
`PAP_ROUTING_POLICY=projection_affinity` remains available for a static
PA-to-Projection mapping. `PAP_ROUTING_POLICY=conversation_affinity` assigns a
new conversation ID to the next PA and pins later turns to that PA.
`PAP_ROUTING_POLICY=attention_load` instead reserves each admitted prompt on
the PA with the fewest committed Attention context tokens. A later turn always
runs Prefill on its retained-history PA. After Prefill reports the exact full
context length, the Gateway selects the Decode PA. If it changes, the target
pulls the complete newly Prefilled KV through bidirectional NIXL and installs
it into its colocated Attention process before Decode starts. The historical
PA remains the Decode target unless moving reduces cross-PA aggregate load
peak by at least `PAP_ATTENTION_LOAD_MIGRATION_MIN_PEAK_GAIN_RATIO` (default
`0.3`). At most `PAP_ATTENTION_LOAD_MIGRATION_MAX_INFLIGHT` migrations are
unresolved at once (default `1`). Migration failure atomically falls Decode
back to the completed Prefill's PA. Completed-turn leases are
pressure-evictable LRU entries, while active Decode leases remain protected.
This load-aware policy is experimental. The general AIPerf topology default
remains `conversation_affinity`; select `attention_load` explicitly. For each
PA, the Gateway admits requests from one Projection source at a time and
switches sources only after that request wave drains. Separate PA groups
continue independently.

Run the functional NIXL smoke service:

```bash
bash examples/pap/launch_pap_nixl.sh --model /data/ssd1/llm-models/Qwen3-0.6B
```

The smoke defaults to NIXL OFFLOAD_EXEC, Qwen3-0.6B, short sequences, and
70/30 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`. These settings test wiring and
failure handling; they are intentionally separate from the AIPerf performance
profile.

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
- `PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.90` is the PA default. Projection has
  no memory-utilization override: the launcher reserves 120% of checkpoint
  weight bytes per TP rank and does not allocate local KV tensors.
- `PAP_EXECUTION_MODE=piecewise` in the benchmark runner to enable the
  validated development path for piecewise CUDA Graph. Eager remains the
  AIPerf default; PAP transport, remote Attention, and KV-publication
  operations stay outside captured regions.

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

The canonical benchmark runner accepts arbitrary positive xPAyP topologies
and always drives them through AIPerf:

```bash
PAP_TOPOLOGY=6pa2p \
PAP_PREFILL_GPUS=0,1,2,3,4,5 \
PAP_PROJECTION_GPUS=6,7 \
bash benchmarks/pap/scripts/run_pap_workload.sh
```

Each run records `topology_manifest.json`, `routing_audit.json`, strict log
audit results, and an all-Attention session-drain result. One-off service smoke
requests belong to `launch_pap_nixl.sh`, not the benchmark runner.

The current eight-GPU capacity comparison is driven by AIPerf. Every matrix
point serves the same 128 conversations and 640 requests: five turns per
conversation, randomized 8K initial input, a broad append distribution sampled
around 1.4K tokens, randomized 16-64-token outputs, and deterministic
think/tool delays. Conversation concurrency limits live sessions while
preserving PA or Prefill ownership across all turns.

PAP compares 7PA1P and 6PA2P. PD uses one-way P→D with 4P4D and 6P2D.
Eight independent fused vLLM replicas with sticky conversation routing form
the dense-model baseline. Run the eager baseline or its
matched piecewise CUDA Graph lane with:

```bash
bash benchmarks/pap/aiperf/run_capacity_matrix.sh

PAP_CAPACITY_EXECUTION_MODE=piecewise \
  bash benchmarks/pap/aiperf/run_capacity_matrix.sh
```

The full default matrix is intentionally long. Use the point-selection and
resume overrides documented in the
[AIPerf testbed](../../benchmarks/pap/aiperf/README.md) when developing a
single topology or concurrency point. The old fixed-length O256/O128 and
12-conversation workloads remain historical evidence, not current defaults.

Current eager and piecewise results are recorded in the
[automatic Projection-memory milestone](../../benchmarks/pap/experiments/PAP-20260722-AIPERF-PROJECTION-AUTO/report.md).
The corrected async scheduling and step-control boundary is validated in the
[step/control-overlap milestone](../../benchmarks/pap/experiments/PAP-20260724-STEP-OVERLAP/report.md).
The rejected no-async treatment is retained as an archived
[scheduler-overlap diagnostic](../../benchmarks/pap/experiments/PAP-20260724-SINGLE-PROJECTION-BATCH/report.md).
