# PAP NIXL Rank 3 Low-Latency Knob A/B - 2026-07-07

## Setup

- Code base starts after Rank 1 metadata cache and Rank 2 receive breakdown.
- Model: `/data/ssd1/llm-models/Qwen3-8B`
- Topology: `1pa1p`
- Transport: `PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox`
- GPUs: PA/Attention on GPU1, Projection on GPU2
- Workload launcher: `benchmarks/disagg_benchmarks/run_pap_128_testbed.sh`
- Workload: sonnet input length 128, output length 32, 64 prompts,
  request rate 16, 16 warmups
- Trace summary used `max_total_ms=10.0`.

End-to-end TPOT is noisy in this benchmark because `vllm bench serve` is not run
with `--ignore-eos`, so several A/B runs generated fewer than 2048 output
tokens. The keep/revert decision below is based primarily on the per-layer trace
field targeted by Rank 3: `recv_wait_other_ms`.

## Runs

| Variant | Raw run directory |
|---|---|
| Baseline | `/home/fei/research/PD/test/baseline/pap/results/runs/20260707_005951` |
| Telemetry off A | `/home/fei/research/PD/test/baseline/pap/results/runs/20260707_010725` |
| Telemetry off B | `/home/fei/research/PD/test/baseline/pap/results/runs/20260707_011045` |
| Inline poll | `/home/fei/research/PD/test/baseline/pap/results/runs/20260707_010904` |
| Inline publish A | `/home/fei/research/PD/test/baseline/pap/results/runs/20260707_011307` |
| Inline publish B | `/home/fei/research/PD/test/baseline/pap/results/runs/20260707_011441` |
| Inline publish default A | `/home/fei/research/PD/test/baseline/pap/results/runs/20260707_012044` |
| Inline publish default B | `/home/fei/research/PD/test/baseline/pap/results/runs/20260707_012228` |

## Summary Table

| Variant | Output tokens | Median TPOT | `recv_wait_other_ms` | `recv_qkv_ms` | Attention `total_ms` | Projection remote total | Projection recv | Projection send |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | `2048` | `93.705 ms` | `1.393 ms` | `1.515 ms` | `1.996 ms` | `1.567 ms` | `1.402 ms` | `0.142 ms` |
| Telemetry off A | `2023` | `93.071 ms` | `1.326 ms` | `1.436 ms` | `1.903 ms` | `1.470 ms` | `1.311 ms` | `0.143 ms` |
| Telemetry off B | `2016` | `94.596 ms` | `1.373 ms` | `1.496 ms` | `1.970 ms` | `1.532 ms` | `1.357 ms` | `0.145 ms` |
| Inline poll | `2048` | `94.385 ms` | `1.403 ms` | `1.498 ms` | `1.970 ms` | `1.543 ms` | `1.384 ms` | `0.140 ms` |
| Inline publish A | `2005` | `92.305 ms` | `1.131 ms` | `1.251 ms` | `1.958 ms` | `1.473 ms` | `1.024 ms` | `0.441 ms` |
| Inline publish B | `2003` | `93.609 ms` | `1.129 ms` | `1.239 ms` | `1.955 ms` | `1.459 ms` | `1.012 ms` | `0.446 ms` |
| Inline publish default A | `2041` | `96.208 ms` | `1.144 ms` | `1.261 ms` | `1.983 ms` | `1.502 ms` | `1.039 ms` | `0.455 ms` |
| Inline publish default B | `2023` | `90.825 ms` | `1.123 ms` | `1.227 ms` | `1.935 ms` | `1.454 ms` | `1.011 ms` | `0.441 ms` |

## Decisions

### Keep: inline publish default

`PAP_NIXL_MAILBOX_INLINE_PUBLISH=1` consistently reduces the targeted
message-availability wait:

- Baseline `recv_wait_other_ms`: `1.393 ms`
- Inline publish explicit mean across A/B: `1.130 ms`
- Inline publish default mean across A/B: `1.134 ms`

This is an operation-level reduction of about 18-19% for the Rank 3 bottleneck.
Projection-side `recv_ms` also drops from `1.402 ms` to roughly `1.01-1.04 ms`.
Projection `send_ms` rises because publish work moves from the sender thread
onto the caller, but projection remote total still improves in the trace.

The code change is to make `PAP_NIXL_MAILBOX_INLINE_PUBLISH` default to enabled
for NIXL mailbox. It does not affect the local IPC path.

### Reject: inline poll default

`PAP_NIXL_MAILBOX_INLINE_POLL=1` did not improve the target metric:

- Baseline `recv_wait_other_ms`: `1.393 ms`
- Inline poll `recv_wait_other_ms`: `1.403 ms`
- Median TPOT also regressed from `93.705 ms` to `94.385 ms`

No default change.

### Reject for now: telemetry off default

`PAP_NIXL_MAILBOX_CAPTURE_TELEMETRY=0` slightly improved per-layer medians, but
the effect was small and not stable enough end-to-end:

- A: `recv_wait_other_ms=1.326 ms`, median TPOT `93.071 ms`
- B: `recv_wait_other_ms=1.373 ms`, median TPOT `94.596 ms`

No default change in this step.

## Trace Parser Update

Inline publish logs use `PAP NIXL mailbox inline send trace`, while the previous
trace summary only parsed background `send trace` lines. The summary now parses
inline send lines into the same `mailbox_send` and `mailbox_send_by_kind`
groups, with `queue_ms=0` and `ack_wait_ms=0`.
