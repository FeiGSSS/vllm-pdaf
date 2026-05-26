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

## Follow-up Root Cause and Fix

A follow-up investigation on the qps32 PAP failure found that the run was not
slow; the Attention mailbox worker crashed early. The failing log was:

```text
Exception in thread pap-offload-exec-mailbox-loop:
KeyError: 'cmpl-bench-df77fec6-2-0-a0acf7f7'
```

The root cause was Projection-side mailbox binding. A Projection process can
batch requests that belong to different PA groups, but the NIXL mailbox
transport was cached as one process-global singleton and bound only to the first
Attention endpoint. Later QKV batches for other PA groups were delivered to the
wrong Attention worker, which had no matching PAP session for those request ids.
The mailbox loop then exited, Projection streams stopped completing, and GPU util
fell to 0%.

The fix is to keep NCCL transport behavior unchanged, but cache NIXL mailbox
transports by Attention endpoint. Each Projection process now creates a separate
mailbox actor per Attention endpoint and binds each actor to its own peer. The
same fix is used by the normal PAP attention path, runner microbatch path, and
Q-first path.

## Valid qps32 Results After Fix

All runs below used Qwen3-8B, `num_prompts=600`, input length `1024`, output
length `64`, offered qps `32`, and the proxy variables were explicitly cleared.

| Architecture | Run directory | Completed | Failed | Duration | Req/s | Output tok/s | Mean TTFT | Mean TPOT | P99 TPOT | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `6P2D` | `/home/fei/research/PD/test/baseline/disaggregated/results/runs/20260526_153852` | 600 | 0 | 20.90 s | 28.71 | 1837.20 | 451.00 ms | 37.93 ms | 40.58 ms | Fresh current-code baseline. |
| `6PA2P` serial | `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_153548` | 600 | 0 | 98.71 s | 6.08 | 389.00 | 32812.35 ms | 269.74 ms | 340.89 ms | `PAP_RUNNER_MICROBATCH_COUNT=0`. |
| `6PA2P` 3-way | `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_153245` | 600 | 0 | 93.81 s | 6.40 | 409.36 | 30644.56 ms | 262.13 ms | 322.07 ms | `PAP_RUNNER_MICROBATCH_COUNT=3`. |

The 3-way runner pipeline is now a valid result and is modestly better than PAP
serial on this workload: request throughput improves from `6.08` to `6.40` req/s
(+5.2%), output throughput improves from `389.00` to `409.36` tok/s (+5.2%), and
mean TPOT drops from `269.74` to `262.13` ms (-2.8%). Native `6P2D` is still much
faster than both PAP variants for this parameter set, so this workload validates
PAP liveness and shows a small 3-way benefit, but it does not yet demonstrate a
large PAP-over-serial advantage.

Verification commands:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_contract.py -q
.venv/bin/python -m pytest \
  tests/pap/test_pap_attention_executor.py::test_run_offload_exec_mailbox_loop_releases_qkv_message \
  tests/pap/test_pap_data_plane.py::test_nixl_mailbox_transport_sends_query_then_kv_batch_messages -q
.venv/bin/python -m py_compile vllm/model_executor/models/qwen3.py
git diff --check -- \
  vllm/model_executor/models/qwen3.py \
  tests/pap/test_pap_contract.py \
  docs/design/pap-6pa2p-large-workload-20260526.md
```

`ruff` was not run because the active `.venv` does not provide the `ruff` module.
