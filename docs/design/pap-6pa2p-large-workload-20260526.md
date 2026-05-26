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

## 6PA2P Trace Critical Path Follow-up

After the valid qps32 runs, a follow-up checked the apparent `0%` GPU utilization
state. At the time of that check there were no active `vllm`, `run_benchmark`,
`launch_service`, or benchmark processes, and GPU memory was already released.
The latest 600-prompt 3-way run had completed normally at `15:34:11`, so the
`0%` utilization snapshot represented post-run idle state, not an in-flight
benchmark.

A running trace benchmark was then sampled with `nvidia-smi`. During the active
decode phase, GPUs 0-5, which host the PA Prefill/Attention groups, were at
`91-95%` utilization, while the Projection GPUs 6-7 were at `33-34%`.

The trace run used the same Qwen3-8B, input length `1024`, output length `64`,
qps `32`, 6PA2P topology, and 3-way runner microbatch configuration as the
600-prompt run, but reduced `num_prompts` to `120` to avoid making trace logging
itself dominate the experiment. Proxy variables were explicitly cleared.

Trace run:

- Run directory:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_155419`
- Result: `120` completed, `0` failed.
- Mean TPOT: `261.06 ms`; median TPOT: `262.46 ms`; p99 TPOT: `272.00 ms`.
- Reference 600-prompt 3-way run mean TPOT: `262.13 ms`.
- Parser:
  `.venv/bin/python tools/pap_trace_summary.py /home/fei/research/PD/test/baseline/pap/results/runs/20260526_155419/service_logs`

Median trace summary, excluding the parser default `>10 ms` warmup/outlier rows:

| Trace point | Median | P99 | Interpretation |
| --- | ---: | ---: | --- |
| Projection offload `total_ms` | 5.584 ms | 7.533 ms | Main per-layer Projection-side offload boundary. |
| Projection `send_ms` | 0.502 ms | 0.932 ms | Model-thread QKV mailbox publish/enqueue path. |
| Projection `recv_ms` | 0.554 ms | 1.366 ms | Model-thread wait/read for Attention output after returning from the DBO yield point. |
| Projection `trigger_ms` | 0.001 ms | 0.002 ms | Effectively absent on the direct mailbox path. |
| Attention mailbox `total_ms` | 2.256 ms | 4.933 ms | Attention worker receives QKV, computes attention, and publishes output. |
| Attention `recv_qkv_ms` | 1.460 ms | 3.430 ms | Wait/read for Projection QKV and scheduling skew. |
| Attention `compute_ms` | 0.807 ms | 1.678 ms | Actual remote attention compute for the mailbox batch. |
| Attention `send_output_ms` | 0.008 ms | 0.017 ms | Attention model-thread output enqueue is negligible. |
| Attention NIXL read `total_ms` | 0.196 ms | 0.358 ms | Raw QKV mailbox transfer is much smaller than the layer boundary. |
| Attention sender `total_ms` | 0.555 ms | 1.074 ms | Sender-thread publish plus ACK wait; not the whole TPOT gap. |

The median Projection offload block alone accounts for about
`5.584 ms * 36 = 201 ms` per token. This aligns with the measured
`261 ms` TPOT: most of the decode time is spent walking 36 sequential layer
boundaries where Projection publishes QKV, yields, waits for Attention output,
and resumes the layer. The remaining roughly `60 ms` comes from non-offloaded
Projection-side work and scheduler/runtime overhead.

The important detail is that Projection `total_ms` is much larger than
`send_ms + recv_ms`. In the current code, the trace timer starts before grouping
and publishing QKV, then Projection can call `dbo_yield()` after sending and
before `recv_output_batch_message()`. That gap is useful for overlap, but it is
still part of the token layer-by-layer critical path. The trace therefore says
the bottleneck is not bare NIXL copy latency. The critical path is the repeated
Projection/Attention handoff plus per-slice compute and scheduling skew across
all 36 transformer layers.

This explains why the 3-way runner pipeline only modestly improves over serial
PAP on this workload. It overlaps some work across runner microbatches, but it
does not remove the per-layer synchronous dependency: layer `N+1` on Projection
cannot proceed until layer `N` remote Attention output has returned for that
microbatch.

## 3-Way Wait/Read Trace Follow-up

A follow-up trace added two missing fields:

- Projection offload trace now records `yield_ms`, the time spent inside
  `dbo_yield()` after Projection publishes QKV and before it starts receiving
  Attention outputs.
- NIXL mailbox trace now records `recv wait trace`, the time from an endpoint
  calling `recv()` until the requested message is available in the local mailbox
  incoming queue. This includes any time waiting for the notification and, when
  the receiver thread has not already materialized the message, the NIXL read.

Run:

- `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_172715`
- Qwen3-8B, `6PA2P`, input `1024`, output `64`, qps `32`,
  `num_prompts=120`, `PAP_RUNNER_MICROBATCH_COUNT=3`.
- Result: `120` completed, `0` failed, mean TPOT `271.29 ms`.
- The extra per-message wait logging adds visible overhead compared with the
  prior `261.06 ms` trace run, so use this run for path decomposition rather
  than absolute TPOT.

Median path decomposition with the parser's default `>10 ms` outlier cutoff:

| Trace point | Median | Meaning |
| --- | ---: | --- |
| Projection `total_ms` | 6.091 ms | One layer's Projection-side offload span. |
| Projection `send_ms` | 0.508 ms | Group by Attention endpoint and enqueue QKV batches. |
| Projection `yield_ms` | 4.829 ms | Cooperative 3-way ubatch switch after sending QKV. |
| Projection `recv_ms` | 0.724 ms | Receive all Attention output batches for this ubatch/layer. |
| Projection residual `gap_ms` | 0.005 ms | Previously hidden time is now explained by `yield_ms`. |
| Projection `batches` | 3 | Typical P0/P1 ubatch fans out to three Attention endpoints. |
| Projection `calls` | 15 | Typical ubatch has about 15 requests total. |
| Attention `calls` | 5 | Each Attention endpoint sees about one third of that ubatch. |
| Attention `recv_qkv_ms` | 1.600 ms | Wait/read next QKV batch. |
| Attention QKV mailbox wait | 1.565 ms | `recv()` wait until QKV message is locally available. |
| Attention QKV NIXL read | 0.192 ms | Actual READ/materialize path for QKV. |
| Attention QKV pure wait estimate | 1.373 ms | `recv wait - read`: Attention is mostly idle waiting for Projection. |
| Attention `compute_ms` | 0.817 ms | Current per-item loop over the `calls` requests. |
| Attention output send enqueue | 0.009 ms | Model-thread output enqueue is negligible. |
| Projection output mailbox wait | 0.002 ms | Projection usually finds output already materialized after yield. |
| Projection output NIXL read | 0.238 ms | Output READ/materialize path, mostly done by the receiver thread. |

This resolves the earlier ambiguity: the old `Projection gap_ms ~= 4.5 ms` was
almost entirely the 3-way `dbo_yield()` interval. During that interval the
current ubatch is sleeping while the Projection worker runs other ubatches and
the Attention workers process previously sent QKV batches.

The “who waits for whom” result is asymmetric:

- Attention waits for Projection. The median Attention QKV `recv()` wait is
  `1.565 ms`, while the actual QKV READ is only `0.192 ms`; the remaining
  roughly `1.37 ms` is waiting for the next QKV message to arrive.
- Projection usually does not wait for Attention at `recv()` time. The median
  Projection output `recv()` wait is `0.002 ms`; the output is usually already
  in the incoming queue because the receiver thread read it during
  `dbo_yield()`.
- Projection still pays the output materialization/read cost indirectly:
  Projection output READ median is `0.238 ms`, but this is not showing up as
  `recv()` blocking because it overlaps with the yield interval.

Attention compute scaling by `calls`:

| Attention calls | Median compute | Compute per call |
| ---: | ---: | ---: |
| 1 | 0.179 ms | 0.179 ms |
| 3 | 0.458 ms | 0.153 ms |
| 5 | 0.747 ms | 0.149 ms |
| 7 | 1.021 ms | 0.146 ms |

The current mailbox message is batched, but Attention compute still loops over
items one request at a time. A fused batch attention kernel would mainly target
this `0.7-1.0 ms` per Attention message compute component. For the median
`calls=5` case, reducing compute from about `0.75-0.82 ms` to `0.3-0.4 ms` would
save roughly `0.4-0.5 ms` on the Attention stage. Across 36 layers, that is a
plausible `15-20 ms/token` TPOT opportunity before secondary queueing effects.

The fused kernel is therefore a reasonable next implementation target, but it
will not by itself remove the Projection `yield_ms` term. The larger system
question after fused Attention is whether faster Attention reduces the
`1.37 ms` QKV wait and the Projection `yield_ms`, or whether Projection-side
ubatch scheduling remains the pacing source.

## Projection Yield Critical Path Follow-up

The previous wait/read trace showed that Projection usually does not block in
`recv()` after returning from `dbo_yield()`, but that was still indirect
evidence. A follow-up trace added stable per-batch trace keys plus monotonic
timestamps on both sides:

- Projection logs each output batch key, QKV send-done time, yield-start time,
  yield-end/resume time, and recv-done time.
- Attention logs the same batch key plus QKV recv-done, compute-done, and output
  send-done time.
- The parser correlates Projection ubatches with the max Attention output
  send-done time across the fanout batches.

Diagnostic run:

- `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_180736`
- Qwen3-8B, `6PA2P`, input `1024`, output `64`, qps `32`,
  `num_prompts=60`, `PAP_RUNNER_MICROBATCH_COUNT=3`.
- Result: `60` completed, `0` failed, mean TPOT `245.45 ms`.
- This run is for critical-path attribution. The added timestamp logging makes
  absolute TPOT less comparable with non-trace runs.

Median correlated timing:

| Trace point | Median | Meaning |
| --- | ---: | --- |
| Projection `total_ms` | 5.764 ms | One Projection-side ubatch offload span. |
| Projection `send_ms` | 0.417 ms | Projection sends QKV fanout. |
| Projection `yield_ms` | 4.560 ms | Current ubatch is suspended in DBO. |
| Projection `recv_ms` | 0.760 ms | Projection resumes and consumes Attention outputs. |
| Attention path after Projection send | 1.139 ms | Max Attention output send-done minus Projection QKV send-done. |
| Projection resume after Attention ready | 3.381 ms | Projection resume time minus max Attention output send-done. |
| Attention ready after Projection resume | 0.000 ms | Cases where Attention was still not ready when Projection resumed. |
| Projection resume to recv done | 0.760 ms | Same interval as Projection `recv_ms`. |

Classification across the diagnostic run:

- Correlated Projection entries: `15048`.
- P-critical entries: `14900` (`99.0%`), where Attention output was ready before
  Projection resumed.
- A-critical entries: `148` (`1.0%`), where Attention output became ready after
  Projection resumed.
- Entries with Attention more than `1 ms` late after Projection resume: `91`
  (`0.6%`).

Therefore the median `yield_ms` is not caused by the remote Attention path. The
remote Attention path is usually complete around `1.1 ms` after Projection sends
QKV, while the Projection ubatch is resumed about `3.4 ms` later. In the current
3-way implementation, the dominant part of `yield_ms` is Projection-side DBO
scheduling/resume latency: the Projection worker is executing other ubatches
before returning to the current ubatch. Remote Attention is only the critical
path for a small tail of cases.

## High-QPS Short-Sequence Follow-up

To test whether larger scheduler batches make 3-way more favorable, a short
sequence/high-QPS workload was run:

- Model: Qwen3-8B.
- Workload: input `128`, output `32`, qps `256`, `num_prompts=1024`.
- Topologies: `6P2D`, `6PA2P` serial, `6PA2P` 3-way.
- PAP runs kept `PAP_OFFLOAD_EXEC_TRACE=1`, so compare PAP serial vs 3-way
  directly; use the 6P2D number as an external baseline, not a trace-equivalent
  PAP comparison.

Benchmark results:

| Architecture | Run root | Success | Mean TPOT | Median TPOT | P99 TPOT | Req/s | Output tok/s | Mean TTFT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `6P2D` | `/home/fei/research/PD/test/baseline/disaggregated/results/runs/20260526_225458` | 1024/1024 | 31.42 ms | 31.48 ms | 34.29 ms | 88.87 | 2843.83 | 4120 ms |
| `6PA2P` serial | `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_225649` | 1024/1024 | 266.26 ms | 256.37 ms | 356.67 ms | 13.67 | 437.30 | 31694 ms |
| `6PA2P` 3-way | `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_225912` | 1024/1024 | 278.57 ms | 303.89 ms | 363.39 ms | 11.84 | 378.84 | 34065 ms |

PAP trace medians:

| Metric | Serial | 3-way | Interpretation |
| --- | ---: | ---: | --- |
| Projection total | 5.628 ms | 6.491 ms | 3-way is slower per ubatch. |
| Projection send | 0.914 ms | 0.480 ms | 3-way sends smaller ubatches. |
| Projection yield | 0.001 ms | 5.114 ms | 3-way adds DBO suspend/resume window. |
| Projection recv | 4.645 ms | 0.827 ms | Serial waits for Attention in recv; 3-way overlaps most of it. |
| Projection calls | 64 | 21 | 3-way cuts the macro batch into smaller Projection GEMMs. |
| Projection fanout batches | 3 | 2 | High-QPS routing does not always fan out to all 3 Attention endpoints. |
| Attention total | 6.685 ms | 3.281 ms | 3-way Attention tasks are smaller. |
| Attention compute | 2.420 ms | 1.214 ms | Smaller attention batch reduces compute per message. |
| Attention calls | 23 | 11 | Attention batch size roughly halves. |
| Attention path after Projection send | 3.563 ms | 2.467 ms | 3-way does make the remote path shorter. |
| Projection resume after Attention ready | 0.000 ms | 2.359 ms | 3-way still waits on P-side resume after A is ready. |
| Attention ready after Projection resume | 3.558 ms | 0.000 ms | Serial is A-critical; 3-way is P-resume-critical. |

This high-QPS experiment does not support the hypothesis that qps `256` and
`1024` prompts are enough to make the current 3-way implementation faster. The
larger macro batch helps serial PAP reach Projection `calls=64`, but 3-way
splits that into median `calls=21`. That smaller ubatch reduces remote Attention
latency, but it also lowers Projection arithmetic intensity and introduces a
median `2.36 ms` P-side resume lag after Attention is already ready. The net
effect is worse TPOT and lower output throughput than serial PAP.

The batch-size hypothesis is still directionally useful, but the current
implementation needs either larger per-ubatch Projection batches, lower
Projection resume latency, or fewer pipeline ways. Under this workload, 2-way is
a more plausible next sweep than 3-way because it may preserve more Projection
batch density while still hiding part of the serial Attention wait.

## High-QPS 2-Way Follow-up

The same short-sequence/high-QPS workload was rerun with
`PAP_RUNNER_MICROBATCH_COUNT=2`.

- Run root: `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_230453`
- Result: `1024` completed, `0` failed.
- Mean TPOT: `264.07 ms`; median TPOT: `276.45 ms`; p99 TPOT: `308.31 ms`.
- Request throughput: `13.34 req/s`; output throughput: `426.90 tok/s`.
- Mean TTFT: `31048 ms`.

Updated benchmark comparison:

| Architecture | Mean TPOT | Median TPOT | P99 TPOT | Req/s | Output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `6P2D` | 31.42 ms | 31.48 ms | 34.29 ms | 88.87 | 2843.83 |
| `6PA2P` serial | 266.26 ms | 256.37 ms | 356.67 ms | 13.67 | 437.30 |
| `6PA2P` 2-way | 264.07 ms | 276.45 ms | 308.31 ms | 13.34 | 426.90 |
| `6PA2P` 3-way | 278.57 ms | 303.89 ms | 363.39 ms | 11.84 | 378.84 |

Updated PAP trace medians:

| Metric | Serial | 2-way | 3-way |
| --- | ---: | ---: | ---: |
| Projection total | 5.628 ms | 5.766 ms | 6.491 ms |
| Projection send | 0.914 ms | 0.776 ms | 0.480 ms |
| Projection yield | 0.001 ms | 3.591 ms | 5.114 ms |
| Projection recv | 4.645 ms | 1.205 ms | 0.827 ms |
| Projection calls | 64 | 32 | 21 |
| Projection fanout batches | 3 | 2 | 2 |
| Attention total | 6.685 ms | 4.105 ms | 3.281 ms |
| Attention compute | 2.420 ms | 1.406 ms | 1.214 ms |
| Attention calls | 23 | 13 | 11 |
| Attention path after Projection send | 3.563 ms | 2.868 ms | 2.467 ms |
| Projection resume after Attention ready | 0.000 ms | 0.677 ms | 2.359 ms |
| Attention ready after Projection resume | 3.558 ms | 0.000 ms | 0.000 ms |

2-way is the best PAP variant by mean TPOT in this noisy high-QPS run, but only
by a small margin over serial. It behaves like the expected compromise:

- It keeps a larger Projection batch than 3-way (`calls=32` vs `21`), so the
  Projection arithmetic-density loss is smaller.
- It hides most of the serial Attention wait: Projection `recv_ms` drops from
  `4.645 ms` to `1.205 ms`.
- It still introduces P-side resume lag after Attention is ready
  (`0.677 ms` median), though much less than 3-way (`2.359 ms`).

The result supports using 2-way as the next tuning baseline. 3-way over-splits
the batch for this workload, while serial leaves the path A-critical. 2-way
mostly removes the A-critical wait without paying as much P-side resume delay,
but the current implementation still needs lower resume overhead or larger
per-ubatch Projection batches to show a robust win.
