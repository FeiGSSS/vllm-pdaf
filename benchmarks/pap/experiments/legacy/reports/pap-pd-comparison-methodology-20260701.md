# PAP vs PD Comparison Methodology

This note defines the next PD-NIXL vs PAP comparison protocol. It addresses the
latest measurement gaps: warmup, TPOT, concurrency, and prefill length.

## Current Baseline Finding

The recent 1P1D / 1PA1P Qwen3-8B profiling shows that pure prefill compute is
not the main PAP-vs-PD gap.

| Path | Workload | Median prefill / pre-projection | Median first-token decode segment | Median TTFT |
| --- | --- | ---: | ---: | ---: |
| PD-NIXL direct, bf16 | input ~127, output 1, qps 16 | 30.8 ms prefill | 75.5 ms decode first chunk | 106.1 ms |
| PAP direct, bf16 | input ~127, output 1, qps 16 | 45.0 ms prefill including IPC | 42.5 ms Projection first chunk | 89.1 ms |

PAP prefill includes synchronous PA-side KV IPC import:

| PAP prefill subcomponent | Median |
| --- | ---: |
| model-side prefill IPC total across 36 layers | 12.6 ms |
| transport total across 36 layers | 10.4 ms |
| attention-side IPC server total across 36 layers | 5.9 ms |
| estimated pure prefill/control, excluding model-side IPC | 32.4 ms |

PAP first-token Projection work is dominated by repeated per-layer remote
attention exchange:

| PAP remote attention trace item | Median |
| --- | ---: |
| per-layer remote total | 1.05 ms |
| per-layer Projection send | 0.04 ms |
| per-layer Projection recv/wait | 1.01 ms |
| per-layer Attention compute | 0.12 ms |
| 36-layer estimated exchange | 37.8 ms |
| ordinary profile Projection first chunk | 42.5 ms |

Interpretation:

- PAP pure prefill compute is close to PD.
- PAP adds about 10-13 ms of synchronous prefill KV IPC today.
- PAP first-token/decode cost is mainly the 36 sequential Projection/Attention
  mailbox round trips.
- Cold-start and first-wave queueing can dominate reported TTFT if warmup is not
  used. TTFT must be reported separately for cold and warmed measurements.

## Problems In The Current Runner

`../test/baseline/run_benchmark.sh` already passes
`--num-warmups "${BENCH_NUM_WARMUPS:-0}"`, but:

- warmup is not exposed as a first-class CLI argument;
- warmup is not written to `run_metadata.json` or `effective_config.env`;
- `--max-concurrency` is not supported;
- result filenames do not encode warmup or concurrency;
- the runner treats `request-rate` as the main load knob, but PAP effective
  Projection batch size must be measured from logs, not inferred from QPS.

The runner should gain:

```text
--num-warmups <N>
--max-concurrency <C|none>
```

and should record both in result metadata. Until that is added, manual
`vllm bench serve` commands are preferred for concurrency sweeps.

## Required Environment

All PD/PAP experiments on this node should use:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    NO_PROXY=127.0.0.1,localhost \
    no_proxy=127.0.0.1,localhost \
    VLLM_BIN=/home/fei/research/PD/vllm-pap/.venv/bin/vllm \
    PYTHON_BIN=/home/fei/research/PD/vllm-pap/.venv/bin/python \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    ...
```

Use `.venv` only. Do not use system `python3`, bare `pip`, or
`/home/fei/research/PD/.uv-base-vllm` for current PAP/PD comparisons.

## Metrics To Report

Every result row must include:

- architecture: PD-NIXL or PAP;
- topology: for example `1P1D`, `4P4D`, `1PA1P`, `6PA2P`;
- model, TP size, dtype;
- input length, output length, request rate, max concurrency, num prompts,
  num warmups;
- completed / failed requests;
- request throughput, output token throughput, total token throughput;
- median and p99 TTFT;
- median, mean, and p99 TPOT;
- max concurrent requests observed by benchmark;
- effective server-side batch evidence:
  - PD: decode worker `KV Transfer metrics`, `Running`/`Waiting` logs;
  - PAP: Projection `Running`, Projection trace `calls`, Attention trace
    `calls`, mailbox wait/read/write summary.

Do not use TPOT from output length 1. Output length 1 is only a TTFT /
first-token test.

## Experiment Matrix

### 1. Warmed Fixed-Workload TPOT

Purpose: compare steady-state decode cost after cold-start and first-wave queue
effects are removed.

| Variable | Value |
| --- | --- |
| model | Qwen3-8B first; Qwen3-32B TP=2 after 8B protocol is stable |
| topologies | PD `1P1D`; PAP `1PA1P` |
| input length | 128 |
| output length | 32 |
| request rate | 16 |
| max concurrency | 64 |
| num prompts | 256 |
| warmups | 32 |

Current non-warmed reference points:

| Architecture | Run | Completed | Median TTFT | Median TPOT | Output tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| PD `1P1D` | `20260701_171300` | 128/128 | 212.5 ms | 24.3 ms | 456.7 |
| PAP `1PA1P` | `20260701_171439` | 128/128 | 1122.2 ms | 96.1 ms | 394.1 |

These are useful as historical references but should not be treated as final
latency numbers because they used no explicit warmup.

### 2. Concurrency Sweep

Purpose: separate scheduler/batch-density effects from raw request rate.

Use burst load with a fixed concurrency cap:

| Variable | Values |
| --- | --- |
| input length | 128 |
| output length | 16 or 32 |
| request rate | `inf` |
| max concurrency | 16, 32, 64, 128 |
| num prompts | `max(4 * max_concurrency, 256)` |
| warmups | 32 |

Interpretation rules:

- If PAP improves with concurrency, check whether Projection `calls` and
  Attention `calls` actually increased.
- If benchmark `max_concurrent_requests` is high but Projection `calls` remains
  small, the bottleneck is routing/scheduling, not offered load.
- If TTFT grows but TPOT improves, report the tradeoff as a throughput/latency
  Pareto point, not as a single winner.

Current no-warmup high-pressure reference:

| Architecture | Workload | Completed | Median TTFT | Median TPOT | Output tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| PD `1P1D` | input 32, output 16, qps 256 | 128/128 | 4521.3 ms | 80.0 ms | 329.3 |
| PAP `1PA1P` best observed | input 32, output 16, qps 256 | 128/128 | 5598.4 ms | 238.8 ms | 405.0 |

The qps-256 runs are saturation/backlog points, not clean steady-state
comparisons, because actual request throughput was far below 256 qps.

### 3. Prefill Length Sweep

Purpose: test whether longer context shifts the balance toward PAP or PD.

| Variable | Values |
| --- | --- |
| input length | 128, 512, 1024, 2048 |
| output length | 16 or 32 |
| request rate | 16 for 8B; retune for 32B |
| max concurrency | 64 |
| num prompts | 256 |
| warmups | 32 |

Expected signals:

- PD pays KV transfer once at the prefill/decode boundary. Longer prefill
  increases PD decode-side KV transfer volume and may increase TTFT.
- PAP currently pays synchronous prefill KV IPC per layer and remote attention
  per decode token per layer. Longer prefill can increase attention-side KV read
  cost, but it does not remove the 36-layer mailbox loop.
- If longer prefill helps PAP, the evidence should show Attention compute/KV
  read becoming large enough to amortize fixed mailbox overhead.

Output length must stay above 1 when comparing TPOT. Use output length 1 only
for first-token decomposition.

### 4. Architecture Retuning

After the 1:1 protocol is stable, compare each architecture at its own best
resource split under the same workload and SLO:

| GPU budget | PD candidates | PAP candidates |
| --- | --- | --- |
| 2 GPUs | `1P1D` | `1PA1P` |
| 4 GPUs | `2P2D`, `3P1D` | `1PA3P`, `3PA1P`, `2PA2P` if supported |
| 8 GPUs | `4P4D`, `6P2D`, `7P1D` | `6PA2P`, `7PA1P`, `4PA4P` |

For every workload change, retune both architectures. Do not compare a tuned PAP
point against a stale PD point, or vice versa.

## Manual Benchmark Template

When a service is already running on `$PORT`:

```bash
/home/fei/research/PD/vllm-pap/.venv/bin/vllm bench serve \
  --backend vllm \
  --model /data/ssd1/llm-models/Qwen3-8B \
  --dataset-name sonnet \
  --dataset-path /home/fei/research/PD/refer_codes/vllm/benchmarks/sonnet_4x.txt \
  --sonnet-input-len 128 \
  --sonnet-output-len 32 \
  --sonnet-prefix-len 50 \
  --num-prompts 256 \
  --num-warmups 32 \
  --request-rate 16 \
  --max-concurrency 64 \
  --port "$PORT" \
  --save-result \
  --result-dir "$RESULT_DIR" \
  --result-filename "ARCH_TOPO_i128_o32_q16_c64_w32.json"
```

Use `--request-rate inf --max-concurrency <C>` for the concurrency sweep.

## Immediate Next Runs

Minimal sufficient next run set:

1. PD `1P1D`, PAP `1PA1P`: input 128, output 32, qps 16, concurrency 64,
   warmups 32, prompts 256.
2. PD `1P1D`, PAP `1PA1P`: input 128, output 32, request-rate `inf`,
   concurrency 16/64/128, warmups 32, prompts 256/256/512.
3. PD `1P1D`, PAP `1PA1P`: input 128/1024/2048, output 16, qps 16,
   concurrency 64, warmups 32, prompts 256.

This is enough to answer:

- how much warmup changes TTFT/TPOT;
- whether TPOT is still dominated by PAP's per-layer remote attention loop;
- whether concurrency improves PAP batch density enough to matter;
- whether longer prefill makes PD KV transfer more expensive relative to PAP.

## Executed Results On 2026-07-01

All runs below used:

- Qwen3-8B, TP=1, bfloat16;
- `.venv` from this repository;
- HTTP proxy variables unset;
- `VLLM_USE_FLASHINFER_SAMPLER=0`;
- `--num-warmups 32`;
- saved JSONs under
  `benchmarks/pap/experiments/legacy/runs/pd_pap_methodology_20260701/`.

### Warmed Fixed-Workload TPOT

| Architecture | Workload | Completed | Median TTFT | Median TPOT | P99 TPOT | Output tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PD `1P1D` | input 128, output 32, qps 16, concurrency 64 | 256/256 | 189.9 ms | 24.9 ms | 26.6 ms | 484.4 |
| PAP `1PA1P` | input 128, output 32, qps 16, concurrency 64 | 256/256 | 884.7 ms | 294.8 ms | 306.7 ms | 195.5 |

PAP is about `11.8x` slower than PD on median TPOT at this warmed fixed-load
point. Warmup removes the cold first request from the measurement, but it does
not remove the steady 36-layer Projection/Attention mailbox loop.

### Concurrency Sweep

These runs use input 128, output 16, request-rate `inf`, warmups 32.

| Architecture | Max concurrency | Prompts | Completed | Median TTFT | Median TPOT | P99 TPOT | Output tok/s | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PD `1P1D` | 16 | 256 | 256/256 | 482.1 ms | 24.4 ms | 24.5 ms | 305.7 | Stable latency point. |
| PD `1P1D` | 64 | 256 | 256/256 | 263.4 ms | 27.8 ms | 28.8 ms | 1263.9 | Best throughput point in this sweep. |
| PD `1P1D` | 128 | 512 | 512/512 | 1650.4 ms | 27.4 ms | 33.5 ms | 720.3 | Higher backlog/tail, TPOT still close to c64. |
| PAP `1PA1P` | 16 | 256 | 256/256 | 653.8 ms | 88.1 ms | 95.8 ms | 127.7 | Valid, but already `3.6x` PD TPOT. |
| PAP `1PA1P` | 64 | 256 | 256/256 | 2424.7 ms | 271.9 ms | 291.9 ms | 165.4 | Required lowering PA prefill memory util to 0.65 after an OOM at 0.80. |
| PAP `1PA1P` | 128 | 512 | failed | - | - | - | - | Attention mailbox thread OOM at 319/512 even with PA prefill memory util 0.65. |

Increasing offered concurrency helped PD throughput while keeping median TPOT
near `25-28 ms`. It did not help PAP: c64 increased output throughput only
slightly over c16 but made median TPOT about `3.1x` worse, and c128 exceeded the
current single-PA memory budget.

### Prefill Length Sweep

The long-context runs use output 16, qps 16, max concurrency 64, warmups 32.
The input-128 rows reuse the warmed fixed-load output-32 results as the
short-context reference; use them for TPOT direction and not for exact
output-length-controlled throughput ratios.

| Architecture | Input length | Completed | Median TTFT | Median TPOT | P99 TPOT | Output tok/s | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PD `1P1D` | 128 | 256/256 | 189.9 ms | 24.9 ms | 26.6 ms | 484.4 | Uses output 32 fixed-load run as the short-context reference. |
| PD `1P1D` | 1024 | 256/256 | 21587.8 ms | 25.5 ms | 32.7 ms | 48.3 | Long prefill/backlog dominates TTFT; decode TPOT remains near 25 ms. |
| PAP `1PA1P` | 128 | 256/256 | 884.7 ms | 294.8 ms | 306.7 ms | 195.5 | Uses output 32 fixed-load run as the short-context reference. |
| PAP `1PA1P` | 1024 | 256/256 | 28095.4 ms | 253.9 ms | 404.8 ms | 30.8 | Required PA prefill memory util 0.55 to leave room for Attention. |

Longer context did not make PAP's median TPOT competitive. It mostly increased
TTFT and reduced throughput for both architectures. The PAP median TPOT stayed
in the same `250-300 ms` range, which is consistent with the fixed per-token,
per-layer remote attention loop still dominating decode.

### Current Interpretation

The current 1:1 8B evidence supports three conclusions:

1. Warmup is necessary for fair TTFT/TPOT reporting, but it does not change the
   main PAP-vs-PD decode gap.
2. Higher concurrency improves PD throughput without materially increasing
   median TPOT. PAP does not get the same benefit because the Projection side
   still pays serialized per-token/per-layer mailbox round trips, and the PA
   GPU also hits memory pressure when prefill and Attention share one device.
3. Longer prefill length mainly shifts time into TTFT/prefill backlog. It does
   not hide the PAP mailbox loop or reduce PAP median TPOT enough to change the
   comparison.

## Phase A Remote-Attention Diagnostics

The Phase A diagnostic loop adds a per-run table that joins benchmark metrics,
PAP trace summaries, and a simple remote-attention lower bound:

```text
T_lb_layer = bytes(QKV + attention_output) / P2P_bandwidth + attention_compute
```

For Qwen3-8B bf16 at batch 64, the default diagnostic assumption is
`q_size=4096`, `kv_size=1024`, `output_size=4096`, `P2P=21 GB/s`, and
`36` layers. This gives a rough lower bound near `0.18 ms/layer`, or
`6.6 ms/token`, before scheduler and queueing effects.

The generated table is stored at:
`benchmarks/pap/experiments/legacy/runs/pd_pap_methodology_20260701/remote_attention_diagnostics.md`.
Use it to decide which existing fast-path flag to test next. If trace columns are
zero for a row, that run lacks the required trace logs and should not be used for
micro-path conclusions.

### Traced Decode Overhead Run On 2026-07-02

Run `20260702_085453` used PAP `1PA1P`, Qwen3-8B, input 128, output 16,
request rate 16, 64 prompts, 8 warmups, and no explicit max-concurrency cap. It
enabled `PAP_OFFLOAD_EXEC_TRACE=1`, `PAP_NIXL_MAILBOX_TRACE=1`, and
`PAP_NIXL_MAILBOX_CAPTURE_TELEMETRY=1`.

Benchmark result: 64/64 completed, median TTFT `475.2 ms`, median TPOT
`151.1 ms`, p99 TPOT `179.3 ms`, output throughput `158.4 tok/s`, peak
concurrency `53`.

Median per-layer trace, with rows above 10 ms filtered as warmup/outliers:

| Component | Median ms/layer | Interpretation |
| --- | ---: | --- |
| Projection pre-attn compute | 0.060 | Local QKV-side work before remote exchange. |
| Projection send task | 0.124 | Publish `attention_task_batch`. |
| Projection recv/wait result | 2.239 | Dominant Projection-side remote wait/read term. |
| Projection remote total | 2.375 | End-to-end remote attention segment on Projection side. |
| Attention recv QKV | 1.610 | Dominated by waiting for task availability. |
| Attention compute | 0.912 | Mostly append+pack, not SDPA. |
| Attention send output | 0.016 | Local send call; mailbox send total is larger. |
| Attention total | 2.578 | Attention-side receive + compute + result send. |
| Benchmark TPOT / 36 layers | 4.196 | E2E decode cost per layer. |

Attention compute median is `0.912 ms/layer`, composed mainly of append KV
`0.270 ms` and pack `0.423 ms`; SDPA itself is only `0.125 ms`. Paged
FlashAttention was inactive in this run (`paged_flash_ms=0`).

Mailbox median timings show that waits dominate raw movement:

| Mailbox path | Median bytes | Median total/wait |
| --- | ---: | ---: |
| Projection send `attention_task_batch` | 159,744 | 0.372 ms total |
| Attention wait for `attention_task_batch` | - | 1.570 ms wait |
| Attention read `attention_task_batch` | 159,744 | 0.010 ms total |
| Attention send `attention_result_batch` | 106,496 | 0.686 ms total |
| Projection wait for `attention_result_batch` | - | 2.117 ms wait |
| Projection read `attention_result_batch` | 106,496 | 0.228 ms total |

The diagnostic lower bound with measured Attention compute is `0.974 ms/layer`,
while benchmark TPOT divided by 36 layers is `4.196 ms/layer` (`4.3x` gap).
Using the earlier pure-attention-compute assumption (`0.12 ms/layer`) gives the
rough theoretical floor of about `0.18 ms/layer`; this run shows the current
implementation already spends most Attention compute time in append/pack before
considering mailbox waits.

Immediate optimization targets from this profile:

1. Reduce per-layer mailbox waits / synchronization, especially Projection result
   wait and Attention task wait.
2. Reduce Attention-side append+pack overhead or activate the paged FlashAttention
   path and verify whether it bypasses enough packing work.
3. Keep measuring batch density: this run had median `calls=13` and p90 `36`, so
   batching exists but the architecture still pays one remote round trip per
   layer.

### Projection Critical-Path Trace On 2026-07-02

Run `20260702_102506` repeated the same PAP `1PA1P`, Qwen3-8B, input 128,
output 16, request rate 16, 64 prompts, 8 warmups workload, adding
`PAP_PROJECTION_CRITICAL_TRACE=1` to trace the Projection-side layer critical
path with one monotonic clock.

Benchmark result: 64/64 completed, median TTFT `513.5 ms`, median TPOT
`171.5 ms`, p99 TPOT `201.9 ms`, output throughput `150.0 tok/s`, peak
concurrency `54`. The extra critical-path logging adds measurable overhead, so
use this run for timing composition rather than as a latency baseline.

Median Projection-side per-layer critical path:

| Component | Median ms/layer |
| --- | ---: |
| input norm | 0.013 |
| QKV + QK norm + RoPE | 0.068 |
| Projection send task | 0.139 |
| Projection recv result | 2.426 |
| O projection | 0.026 |
| post-attn norm | 0.018 |
| MLP | 0.047 |
| traced gaps inside layer | 0.118 |
| layer total | 2.849 |

The component medians sum to the traced layer total, so the previous in-layer
"unattributed" bucket is now mostly resolved. The top-level Projection model
forward median is `106.9 ms`, or `2.97 ms/layer`, close to `36 * 2.849 ms` plus
small model-level overhead.

The remaining benchmark-level gap is outside the Qwen3 layer/model-forward span:
median TPOT / 36 layers is `4.76 ms/layer`, leaving about `1.9 ms/layer` beyond
the traced layer critical path. The likely next targets are scheduler/output-loop
and logits/sampling/output processing instrumentation. `projection_logits` emitted
zero samples in this serving path, so the current hook does not capture logits for
this run.

Remote-attention and mailbox medians are consistent with the earlier traced run:
Projection remote total `2.534 ms/layer`, Attention total `2.862 ms/layer`,
Attention compute `1.079 ms/layer`, Projection result wait `2.285 ms`, and
Attention task wait `1.625 ms`. Attention compute is still dominated by append KV
`0.316 ms` and pack `0.501 ms`; SDPA is `0.141 ms`, and paged FlashAttention is
inactive.
