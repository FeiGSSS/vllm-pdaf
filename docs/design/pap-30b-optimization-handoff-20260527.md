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

## Optimization TODOs

Use this list as the next iteration backlog. The priority order reflects the
current evidence: PAP is losing to PD because the per-layer
Projection/Attention boundary is too expensive, not because scheduler
concurrency alone is too low.

### P0: Collapse the per-layer handoff cost

Problem:

- PAP pays a Projection-to-Attention-to-Projection boundary at every decode
  layer. The 8B trace measured median Projection offload around `5.6 ms/layer`,
  while raw NIXL reads were only about `0.2 ms`.
- The hot path still includes mailbox scheduling, grouping, ACK handling,
  thread wakeups, Python object work, and Projection DBO yield/resume latency.

TODOs:

- Replace the current mailbox/RPC-style hot path with a persistent slot or ring
  protocol.
- Pre-register QKV and Attention-output buffers at startup; send fixed slot ids,
  layer ids, and batch ids on the hot path.
- Move high-frequency control-path work out of Python and into C++ or a lower
  runtime layer.
- Piggyback ACK/release metadata on the reverse output path, or batch ACKs by
  layer window.
- Add a trace target for this work: first reduce the median per-layer PAP
  boundary below `1 ms`, then evaluate whether `<0.5 ms` is reachable.

### P0: Implement fused batch remote Attention

Problem:

- Mailbox messages are batched, but the current Attention executor effectively
  processes requests one item at a time.
- This keeps per-request metadata handling and kernel launch overhead visible
  and prevents the Attention side from behaving like a real batch operator.

TODOs:

- Build a decode-only fused remote Attention path for mailbox batches.
- Represent each Attention task with batched QKV, block tables, sequence
  lengths, request routing metadata, and output offsets.
- Call one fused paged/ragged Attention kernel per mailbox batch instead of
  looping over requests.
- Scatter the fused output back to the correct Projection slots.
- Validate first on Qwen3 GQA decode-only workloads before adding mixed
  prefill/decode support.

### P1: Make microbatching adaptive instead of fixed 3-way

Problem:

- 3-way hides serial `recv` wait, but it also splits Projection macro batches
  into smaller ubatches and introduces Projection DBO resume latency.
- In the high-QPS 8B run, serial Projection reached median `calls=64`, while
  3-way reduced that to about `21`, hurting Projection arithmetic intensity.

TODOs:

- Add an adaptive microbatch policy: serial or 2-way for small macro batches,
  3-way only when each ubatch stays dense enough.
- Gate ubatching by minimum per-ubatch Projection calls, not only by total
  scheduler batch size.
- Prefer ready ubatches when Attention output has already arrived, so Projection
  does not wait behind unrelated ubatch work.
- Record per-run `macro_calls`, `ubatch_calls`, `A_ready_time`, and
  `P_resume_lag` to drive the policy.
- Re-sweep serial, 2-way, and 3-way after the fused Attention path lands.

### P1: Use a model/workload that can amortize PAP fixed costs

Problem:

- Qwen3-30B-A3B-FP8 has small active compute relative to the total parameter
  count: `hidden_size=2048`, `moe_intermediate_size=768`, `128` experts, and
  `topK=8`.
- The average per-expert token count is roughly batch divided by `16`, so expert
  and Projection GEMMs remain small unless the effective batch is very large.

TODOs:

- Build a simple roofline-style PAP eligibility check:
  `compute_time > communication_time + scheduling_time`.
- Track effective Projection and expert batch density, not only global QPS or
  `MAX_NUM_SEQS`.
- Test a larger-active or coarser-expert MoE model when available.
- Avoid enabling 3-way when it cuts Projection or expert batches below the
  arithmetic-intensity sweet spot.
- Compare PAP on long-context or high-KV-pressure workloads where Attention
  memory pressure creates a real reason to split resources.

### P1: Add a topology-aware fast communication path

Problem:

- AFD systems such as StepMesh assume very low-latency MxN/bipartite
  communication. The current NIXL mailbox path is functional but not yet a
  StepMesh-equivalent fast path.
- Generic collective libraries can also consume SM resources or mismatch the
  PAP communication pattern.

TODOs:

- Add a local fast path for same-node peers using CUDA IPC or direct P2P buffers
  where possible.
- Keep NIXL/RDMA for cross-node cases, but use the persistent slot/ring protocol
  above.
- Place frequent Projection/Attention peers within the same NVLink or PCIe
  locality domain when the topology allows it.
- Record hardware topology and peer placement in every PAP run summary.
- Keep the current mailbox backend as the debug/control implementation, not the
  assumed final performance path.

### P2: Redesign the PD-vs-PAP comparison around PAP's real advantage

Problem:

- PD communicates at the prefill/decode boundary; PAP communicates every decode
  token and every layer. A fixed 8-GPU TPOT comparison therefore favors PD unless
  PAP recovers that cost through higher batch density, lower KV pressure, or
  better resource specialization.

TODOs:

- Compare PD and PAP after independently tuning each architecture's topology,
  concurrency, memory utilization, and batch limits.
- Include SLO-constrained throughput, maximum supported concurrency, cost per
  output token, and maximum context length, not only median TPOT.
- Add workloads where PD decode is KV-capacity or HBM-bandwidth constrained.
- Keep `4P4D` PD-NIXL as the current fixed-workload baseline until the workload
  changes; retune PD whenever the workload changes.
- Treat PAP as a candidate for large-MoE or high-KV-pressure regimes, not as an
  automatic replacement for PD on small-active models.

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

- 2026-05-27 TP=2 bring-up result:
  - Qwen3-32B dense at `/data/ssd1/llm-models/Qwen3-32B` runs through PAP
    `1PA1P` with `PAP_TP_SIZE=2`. The smoke request returned 4 decode tokens
    through the proxy with `system_fingerprint=...-tp2-nohash`.
  - Final verified log directory:
    `/tmp/pap-qwen3-32b-tp2-final-20260527_153716`.
  - Required runtime env on this node:
    `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`. Without these, both PAP TP=2 and a
    plain vLLM Qwen3-32B TP=2 server hang during NCCL initialization before
    model weights load. A minimal PyTorch NCCL all-reduce also hangs without the
    fallback and passes with it, so this is a node/NCCL transport issue rather
    than PAP routing.
  - Smoke command:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    NCCL_P2P_DISABLE=1 \
    NCCL_IB_DISABLE=1 \
    PAP_MODEL_PATH=/data/ssd1/llm-models/Qwen3-32B \
    PAP_TOPOLOGY=1pa1p \
    PAP_TP_SIZE=2 \
    PAP_ENABLE_MPS=0 \
    PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox \
    PAP_OFFLOAD_KV_TRANSPORT=cuda_ipc \
    PAP_MAX_MODEL_LEN=2048 \
    PAP_MAX_NUM_SEQS=2 \
    PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.88 \
    PAP_PROJECTION_GPU_MEMORY_UTILIZATION=0.88 \
    PAP_MAX_TOKENS=4 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    bash examples/pap/launch_pap_nixl.sh
```

- Do not compare a new PAP point against only one arbitrary PD topology. Use
  the current best valid PD point, `4P4D`, unless the workload changes.
- If workload changes, retune both architectures. PD and PAP have different
  topology sensitivity.
- Keep recording actual Projection `Running` requests from service logs; config
  value alone is not the real batch size.
- Treat `qps` as load pressure, not as the actual Projection batch-size knob.
- If a PAP run stalls at `0/2000`, inspect Projection mailbox ACK and receive
  slot logs before rerunning the same configuration.
