# PAP 4PA4P Qwen3-8B Baseline

Updated: 2026-05-26
Base code checkpoint: `998965825 Add PAP NIXL mailbox handoff checkpoint`

## Purpose

This note records the first Qwen3-8B 4PA4P measurements before trace profiling.
The runs use the PAP NIXL mailbox transport with the default, low-risk protocol
path: Q-first/KV-later, Q-first Projection, and Attention partial overlap are
all disabled.

## Configuration

Common environment:

```bash
MODEL_PATH=/data/ssd1/llm-models/Qwen3-8B
BENCH_NUM_WARMUPS=1
BENCH_TIMEOUT=1800
SERVER_START_TIMEOUT=900
CLUSTER_READY_WAIT_SECONDS=30
PAP_MODE=pap
PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox
PAP_OFFLOAD_EXEC_TRACE=0
PAP_Q_FIRST_KV_LATER=0
PAP_Q_FIRST_PROJECTION=0
PAP_ATTENTION_Q_FIRST_PARTIAL=0
PAP_PREFILL_MPS_PERCENT=30
PAP_ATTENTION_MPS_PERCENT=70
PAP_PROJECTION_GPU_MEMORY_UTILIZATION=0.80
```

Topology and benchmark command:

```bash
bash /home/fei/research/PD/test/baseline/run_benchmark.sh \
  --mode pap \
  --topology 4pa4p \
  --input-lens <input_len> \
  --output-lens <output_len> \
  --qps 32 \
  --num-prompts 300 \
  --model /data/ssd1/llm-models/Qwen3-8B \
  --proxy-port 9000
```

The long-prompt run used `PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.80`. The first
decode-bound attempt at `0.80` OOMed in the Attention executor while
concatenating segmented value tensors, so the completed decode-bound run used
`PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.60` to leave enough memory for the
co-located Attention process on the PA GPUs.

## Results

| Workload | Run directory | Successful | Failed | Duration | Total tok/s | Req/s | Mean TTFT | Mean TPOT | P99 ITL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `i8000/o64/q32` | `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_094753` | 300 | 0 | 363.69 s | 6619.78 | 0.82 | 178690.05 ms | 71.54 ms | 105.27 ms |
| `i1024/o1024/q32` | `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_100613` | 300 | 0 | 687.26 s | 887.20 | 0.44 | 83380.32 ms | 441.70 ms | 586.89 ms |

## Baseline Comparison

The DP/PD comparison baseline lives under
`/home/fei/research/PD/test/baseline/results_summary.md`.

- For `i8000/o64/q32`, 4PA4P is worse than DP8 and all measured PD topologies
  on throughput and TTFT. Its mean TPOT is better than DP8 but still much worse
  than PD 2P6D/4P4D/6P2D.
- For `i1024/o1024/q32`, 4PA4P is worse than DP8 and all measured PD topologies
  on throughput, TTFT, TPOT, and ITL.
- The completed decode-bound result confirms that the bad 4PA4P behavior is not
  just startup transience: the first request waits roughly 7.7 minutes, and
  steady-state TPOT remains hundreds of milliseconds.

## Next Profiling Step

Run the same two workloads with:

```bash
PAP_OFFLOAD_EXEC_TRACE=1
PAP_NIXL_MAILBOX_TRACE=1
```

Then summarize service logs with:

```bash
.venv/bin/python tools/pap_trace_summary.py \
  /home/fei/research/PD/test/baseline/pap/results/runs/<run_id>/service_logs
```

The trace question to answer is whether the bottleneck is dominated by mailbox
transport, per-layer Projection/Attention alternation, or queueing from the
current 4PA4P routing and resource split.
