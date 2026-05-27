# PAP 30B Optimization Handoff

This handoff summarizes the current PD/PAP comparison and the latest PAP
concurrency search. Use it as the starting point for the next PAP optimization
iteration.

## Current Branch State

- Branch: `feature/pap`.
- Latest experiment commits:
  - `1771c2c10 Record PAP low concurrency point`
  - `4fa2e4e44 Record PAP concurrency sweep`
  - `821f21cb6 Record 30B fixed workload PD PAP sweep`
  - `4ea358fc5 Record 30B PD baseline comparison`
  - `a500ba08a Record 30B PAP target-scale wavefront validation`
- Main detailed record:
  [`docs/design/pap-6pa2p-large-workload-20260526.md`](pap-6pa2p-large-workload-20260526.md).

## Fixed Workload

All headline numbers below use the same target workload:

```text
model=/data/ssd1/llm-models/Qwen3-30B-A3B-FP8
dataset=sonnet
input_len=1024
output_len=64
qps=256
num_prompts=2000
gpu_budget=8 L20 GPUs
```

For all PAP runs:

```text
PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox
PAP_OFFLOAD_EXEC_LAYER_WAVEFRONT=1
PAP_OFFLOAD_EXEC_MICROBATCH_COUNT=auto
PAP_OFFLOAD_EXEC_MICROBATCH_AUTO_MIN_BATCH=16
PAP_RUNNER_MICROBATCH_COUNT=0
PAP_PREFILL_MPS_PERCENT=30
PAP_ATTENTION_MPS_PERCENT=70
PAP_PROJECTION_GPU_MEMORY_UTILIZATION=0.95
VLLM_USE_FLASHINFER_SAMPLER=0
```

The `VLLM_USE_FLASHINFER_SAMPLER=0` override matters. Without it, a PAP launch
attempt hit FlashInfer sampler JIT failure:
`BlockAdjacentDifference<...> has no member "FlagHeads"`.

Always unset shell HTTP proxy variables before launching vLLM/PAP:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    ...
```

## Current Best Comparison

| Architecture | Best config | Req/s | Output tok/s | Median TTFT | Median TPOT | P99 TPOT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PD-NIXL throughput best | `4P4D`, `DECODE_MAX_NUM_SEQS=64` | 15.21 | 973.63 | 58.24 s | 26.20 ms | 41.93 ms |
| PAP throughput best | `6PA2P`, `MAX_NUM_SEQS=2000` | 7.82 | 500.65 | 95.25 s | 1201.16 ms | 1578.17 ms |
| PAP latency best found | `6PA2P`, `MAX_NUM_SEQS=320` | 7.68 | 491.62 | 96.34 s | 1153.02 ms | 1352.96 ms |

At this workload, PD-NIXL is still the clear winner:

- PAP throughput best reaches only `51.4%` of PD throughput.
- PAP median TPOT remains around `1.15-1.20 s`, far above PD's `26.20 ms`.
- PAP TTFT is also worse in the best points, around `95-96 s` versus PD
  `58.24 s`.

## PAP Topology Search Summary

Valid PAP topology points:

| Topology | Run root | Output tok/s | Median TTFT | Median TPOT | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `7PA1P` | `/home/fei/research/PD/test/baseline/pap/results/runs/20260527_121356` | 216.07 | 92.94 s | 7592.25 ms | One Projection bottleneck. |
| `6PA2P` | `/home/fei/research/PD/test/baseline/pap/results/runs/20260527_122512` | 500.65 | 95.25 s | 1201.16 ms | Best throughput point. |
| `4PA4P` | `/home/fei/research/PD/test/baseline/pap/results/runs/20260527_124150` | 327.38 | 152.20 s | 1419.97 ms | More Projection nodes did not help. |

Invalid point:

- `5PA3P`, `/home/fei/research/PD/test/baseline/pap/results/runs/20260527_123036`:
  benchmark stayed at `0/2000`; Projection logs reported mailbox ACK and
  receive-slot timeouts.

Takeaway: `6PA2P` is the only useful PAP topology found so far for this
workload. More Projection nodes are not monotonic because the PA side and
mailbox exchange path become limiting.

## PAP Concurrency Search Summary

Within `6PA2P`, we swept `MAX_NUM_SEQS` to test whether higher Projection-side
concurrency unlocks PAP.

| `MAX_NUM_SEQS` | Run root | Output tok/s | Median TPOT | P99 TPOT | Projection `Running` evidence |
| ---: | --- | ---: | ---: | ---: | --- |
| 320 | `/home/fei/research/PD/test/baseline/pap/results/runs/20260527_134611` | 491.62 | 1153.02 ms | 1352.96 ms | p50 `314/314`, max `320/320` |
| 384 | `/home/fei/research/PD/test/baseline/pap/results/runs/20260527_133154` | 498.63 | 1330.89 ms | 1581.83 ms | p50 `368/372`, max `384/384` |
| 448 | `/home/fei/research/PD/test/baseline/pap/results/runs/20260527_133717` | 495.48 | 1581.14 ms | 1778.16 ms | p50 `439/435`, max `448/448` |
| 512 | `/home/fei/research/PD/test/baseline/pap/results/runs/20260527_132011` | 497.82 | 1702.64 ms | 2008.52 ms | p50 `487/488`, max `512/512` |
| 1024 | `/home/fei/research/PD/test/baseline/pap/results/runs/20260527_132539` | 493.36 | 2251.40 ms | 2608.24 ms | p50 `543/556`, max `787/787` |
| 2000 | `/home/fei/research/PD/test/baseline/pap/results/runs/20260527_122512` | 500.65 | 1201.16 ms | 1578.17 ms | p50 `295/296`, max `423/445` |

Concurrency conclusion:

- Scheduler concurrency alone is not the missing knob.
- Best throughput is still `MAX_NUM_SEQS=2000`.
- Best latency found is `MAX_NUM_SEQS=320`, but it gives up a small amount of
  throughput.
- The useful Projection active-request region is roughly `300-450` per
  Projection instance.
- Pushing actual Projection running batch toward `512+` or `787` worsens TPOT.

## Current Bottleneck Hypothesis

PAP can aggregate decode requests on Projection, but the current implementation
does not turn that into an end-to-end win because the Projection/Attention loop
is too expensive.

Observed symptoms:

- Remote Attention exchange dominates TPOT.
- Benchmark progress shows periodic wave stalls even when Projection has enough
  running requests.
- `5PA3P` exposed mailbox ACK / receive-slot instability.
- Whole-model runner microbatch is disabled for Qwen3 MoE FP8 because the fused
  MoE path does not support DBO contexts; the active path is the MoE-specific
  layer-wavefront offload path.

Working conclusion:

> PAP's next win is unlikely to come from larger scheduler concurrency. It needs
> a cheaper batched remote-Attention path.

## Next Development Target

The next concrete development target is the fused batch attention kernel for
PAP remote Attention.

Goal:

- Process the mailbox batch as one fused attention batch instead of handling
  items effectively one request at a time.
- Reduce per-request Python/mailbox/kernel launch overhead.
- Preserve per-request routing back to the correct PA/Projection pair.
- Measure whether median TPOT drops at the current best topology
  (`6PA2P`) and workload.

Recommended first validation matrix after implementing it:

| Model | Topology | Workload | Configs |
| --- | --- | --- | --- |
| Qwen3-30B-A3B-FP8 | `6PA2P` | `1024/64`, qps `256`, prompts `2000` | `MAX_NUM_SEQS=320`, `2000` |
| Qwen3-8B | `6PA2P` | `1024/64`, qps `256`, prompts `1000` | Previous sweet spots `128`, `512` |

Success criteria:

- `0` failed requests.
- No mailbox ACK / receive-slot timeout.
- PAP median TPOT materially below current `1153 ms`.
- PAP output throughput above current `500.65 tok/s`, or a clear TPOT win with
  minimal throughput loss.

## Reproduction Command Template

For PAP `6PA2P`:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    PAP_MODE=pap \
    MODEL_PATH=/data/ssd1/llm-models/Qwen3-30B-A3B-FP8 \
    MAX_MODEL_LEN=2048 \
    MAX_NUM_BATCHED_TOKENS=8192 \
    MAX_NUM_SEQS=<320-or-2000> \
    PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.85 \
    PAP_PROJECTION_GPU_MEMORY_UTILIZATION=0.95 \
    PAP_PREFILL_MPS_PERCENT=30 \
    PAP_ATTENTION_MPS_PERCENT=70 \
    PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox \
    PAP_OFFLOAD_EXEC_LAYER_WAVEFRONT=1 \
    PAP_OFFLOAD_EXEC_MICROBATCH_COUNT=auto \
    PAP_OFFLOAD_EXEC_MICROBATCH_AUTO_MIN_BATCH=16 \
    PAP_RUNNER_MICROBATCH_COUNT=0 \
    BENCH_TIMEOUT=1200 \
    SERVER_START_TIMEOUT=1200 \
    CLUSTER_READY_WAIT_SECONDS=10 \
    bash ../test/baseline/run_benchmark.sh \
      --mode pap \
      --topology 6PA2P \
      --input-lens 1024 \
      --output-lens 64 \
      --qps 256 \
      --num-prompts 2000
```

For PD-NIXL `4P4D`:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    MODEL_PATH=/data/ssd1/llm-models/Qwen3-30B-A3B-FP8 \
    MAX_MODEL_LEN=2048 \
    MAX_NUM_BATCHED_TOKENS=8192 \
    MAX_NUM_SEQS=2000 \
    PREFILL_MAX_NUM_BATCHED_TOKENS=8192 \
    PREFILL_MAX_NUM_SEQS=2000 \
    DECODE_MAX_NUM_BATCHED_TOKENS=8192 \
    DECODE_MAX_NUM_SEQS=64 \
    PREFILL_GPU_MEM_UTIL=0.85 \
    DECODE_GPU_MEM_UTIL=0.7 \
    BENCH_TIMEOUT=3600 \
    SERVER_START_TIMEOUT=1200 \
    CLUSTER_READY_WAIT_SECONDS=10 \
    bash ../test/baseline/run_benchmark.sh \
      --mode nixl_disaggregated \
      --topology 4P4D \
      --input-lens 1024 \
      --output-lens 64 \
      --qps 256 \
      --num-prompts 2000
```

## Practical Notes

- Do not compare a new PAP point against only one arbitrary PD topology. Use
  the current best valid PD point, `4P4D`, unless the workload changes.
- If workload changes, retune both architectures. PD and PAP have different
  topology sensitivity.
- Keep recording actual Projection `Running` requests from service logs; config
  value alone is not the real batch size.
- Treat `qps` as load pressure, not as the actual Projection batch-size knob.
- If a PAP run stalls at `0/2000`, inspect Projection mailbox ACK and receive
  slot logs before rerunning the same configuration.
