# PAP 6PA2P Large Workload Matrix Attempt

Date: 2026-05-26

## Purpose

This note records the first large-workload matrix attempt for comparing:

- Native disaggregated `6P2D`
- PAP `6PA2P` serial Projection/Attention
- PAP `6PA2P` runner-level 3-way microbatch pipeline

The intended workload was Qwen3-8B, `num_prompts=600`, input length `1024`,
output length `64`, and offered QPS `32` and `64`. The goal was to increase
scheduler batch pressure enough for the 3-way Projection/Attention overlap to
show a clearer advantage over serial PAP.

## Environment Notes

The interactive shell had HTTP proxy variables set:

```bash
HTTP_PROXY=http://localhost:3128
HTTPS_PROXY=http://localhost:3128
```

All valid benchmark attempts must clear these variables:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    NO_PROXY=127.0.0.1,localhost \
    no_proxy=127.0.0.1,localhost \
    ...
```

The first `6P2D` run without clearing the proxy was invalid: all requests failed
with HTTP 403 and local service logs did not receive benchmark POST traffic.

## Common PAP Configuration

PAP attempts used the following overrides:

```bash
BENCH_TIMEOUT=1800
SERVER_START_TIMEOUT=900
CLUSTER_READY_WAIT_SECONDS=15
PAP_MODE=pap
PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox
PAP_DIRECT_MAILBOX_OUTPUT=1
PAP_OFFLOAD_EXEC_MICROBATCH_COUNT=0
PAP_NIXL_MAILBOX_SLOT_COUNT=8
PAP_NIXL_MAILBOX_RECV_SLOT_COUNT=8
PAP_NIXL_MAILBOX_ASYNC_SEND_SLOTS=4
PAP_Q_FIRST_KV_LATER=0
PAP_Q_FIRST_PROJECTION=0
PAP_ATTENTION_Q_FIRST_PARTIAL=0
PAP_PREFILL_MPS_PERCENT=30
PAP_ATTENTION_MPS_PERCENT=70
PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.60
PAP_PROJECTION_GPU_MEMORY_UTILIZATION=0.80
```

The serial PAP attempt set `PAP_RUNNER_MICROBATCH_COUNT=0`; the pipeline attempt
set `PAP_RUNNER_MICROBATCH_COUNT=3` and
`PAP_RUNNER_MICROBATCH_DECODE_THRESHOLD=12`.

## Results

| Architecture | QPS | Run directory | Result status | Completed | Failed | Mean TTFT | Mean TPOT | P99 TPOT | Notes |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `6P2D` | 32 | `/home/fei/research/PD/test/baseline/disaggregated/results/runs/20260526_145306` | valid | 600 | 0 | 437.15 ms | 37.83 ms | 40.43 ms | Request throughput 28.71 req/s; total throughput 30804.95 tok/s. |
| `6P2D` | 64 | `/home/fei/research/PD/test/baseline/disaggregated/results/runs/20260526_145306` | invalid | 461 | 139 | 3905.25 ms | 40.72 ms | 44.57 ms | Prefill worker OOM; do not use as a performance point. |
| `6PA2P` serial | 32 | `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_145603` | invalid/stuck | 0 | n/a | n/a | n/a | n/a | Benchmark stayed at 0 completed while GPUs became idle; run was terminated. |
| `6PA2P` 3-way | 32 | `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_150828` | invalid/failed | 0 | 600 | 0.00 ms | 0.00 ms | 0.00 ms | Same stuck pattern before termination; after termination benchmark reported incomplete streaming payloads for all requests. |

## Failure Details

The valid `6P2D` qps32 result is the only usable performance point from this
matrix attempt.

The `6P2D` qps64 attempt is invalid because a prefill worker hit CUDA OOM. The
benchmark JSON still contains partial metrics, but they mix 461 successes with
139 failures and must not be used for speedup calculations.

Both PAP `6PA2P` qps32 attempts reached the same failure mode:

- Prefill and Attention workers accepted requests and imported paged KV into
  Attention.
- The proxy posted requests to Projection and saw HTTP 200 headers.
- The benchmark did not receive completed streaming responses.
- GPU utilization dropped back to 0% while Projection EngineCore processes
  remained alive.

For the 3-way run, after terminating the stuck service the benchmark reported
`600` failed requests with `ClientPayloadError: Response payload is not
completed`. This is consistent with the proxy/Projection streaming response
being opened but never completed before shutdown. It is not a valid throughput
measurement.

## Interpretation

The planned large-workload comparison cannot be used yet to show 3-way advantage.
At `num_prompts=600`, input `1024`, output `64`, qps `32`, current PAP `6PA2P`
does not complete either in serial mode or in 3-way runner-microbatch mode.

The likely bottleneck is not simply lack of Projection/Attention overlap. The
evidence points to a PAP serving-path liveness issue after prefill KV import and
Projection streaming response setup. Because both serial and 3-way fail at the
same boundary, further speedup experiments should first reduce or instrument this
failure mode rather than treating the runs as slow-but-valid measurements.

## Recommended Next Steps

- Add PAP streaming diagnostics around `_stream_projection` to log first byte,
  last byte, exceptions, and request completion for each Projection response.
- Enable targeted `PAP_OFFLOAD_EXEC_TRACE=1` on a smaller reproduction to locate
  whether Projection is blocked sending QKV, Attention is blocked computing, or
  Projection is blocked receiving the Attention output.
- Re-run `6P2D` qps64 with lower prefill memory utilization if qps64 remains a
  desired comparison point.
- Rebuild the PAP matrix from a smaller known-good PAP workload, then scale one
  dimension at a time: `num_prompts`, qps, output length, and topology.
