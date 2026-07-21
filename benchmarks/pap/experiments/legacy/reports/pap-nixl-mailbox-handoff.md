# PAP NIXL Mailbox Handoff

Updated: 2026-05-28

## Purpose

This document captures the first functional PAP OFFLOAD_EXEC NIXL mailbox
runtime and the first continuation optimization pass. The code proves that
Projection and Attention can exchange QKV and attention output through NIXL
mailbox messages instead of the old OFFLOAD_EXEC TCP trigger, but the current
performance is not the target architecture.

## Current Development State

Ready to hand off as an experimental PAP/NIXL checkpoint:

- Default mailbox path is functionally covered and keeps the best-tested
  low-risk defaults: transfer busy-polling, slot metadata, cached dlists,
  msgpack notifications, safe zero-copy receive, and single-output/empty-output
  allocation cuts.
- Q-first/KV-later protocol support exists but is opt-in. The split-message
  protocol is gated by async send slots plus at least two send slots to avoid
  the single-slot Q/KV deadlock found during benchmarking.
- Q-first Projection compute and Attention partial overlap are implemented as
  opt-in research paths. They validate the paper-aligned ordering and semantics,
  but both regressed TPOT in controlled runs, so their defaults remain off.
- Trace tooling is included for future protocol work:
  `tools/pap_trace_summary.py` summarizes Projection, Attention, mailbox, and
  NIXL timing from service logs.
- Qwen3-32B `3PA1P`, TP=2 is functionally stable through
  `PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox` on the ctx128/bs512/o64 burst
  workload after reserving PA-side memory for the colocated attention
  executors.
- Latest evidence says the remaining TPOT gap is not dominated by raw NIXL READ
  or notification publish latency. The largest measured cost is the per-layer
  Projection/Attention alternation, so the next owner should focus on a
  fast-backend partial attention/LSE path or a deeper model-slice pipeline, not
  another scalar mailbox tweak.

Key opt-in flags to know before continuing:

- `PAP_Q_FIRST_KV_LATER=1`
- `PAP_Q_FIRST_PROJECTION=1`
- `PAP_ATTENTION_Q_FIRST_PARTIAL=1`
- `PAP_NIXL_MAILBOX_ASYNC_SEND_SLOTS=1`
- `PAP_NIXL_MAILBOX_SLOT_COUNT=2`

## Runtime Shape

Projection and Attention are modeled as symmetric mailbox actors:

1. The producer puts a complete work item into its local output pool.
2. A sender thread publishes a NIXL notification describing the payload.
3. The receiver thread polls NIXL notifications.
4. The receiver issues a NIXL READ from the sender registered buffer into its
   own registered receive buffer.
5. The receiver reconstructs the tensor and places it in its task pool.
6. The receiver sends an ACK notification so the sender can clear its output
   pool entry.

Current message kinds:

- `attention_task`: single QKV payload from Projection to Attention.
- `attention_task_batch`: batched QKV payload from Projection to Attention.
- `attention_result`: single attention output payload from Attention to
  Projection.
- `attention_result_batch`: batched attention output payload from Attention to
  Projection.

## Entry Points

- `PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox` selects the mailbox path.
- `PAP_NIXL_MAILBOX_BUFFER_BYTES=16777216` controls the pre-registered mailbox
  buffer size.
- `PAP_NIXL_MAILBOX_POLL_SECONDS=0.00001` controls empty notification poll
  sleep. Set `0` for busy polling during controlled experiments.
- `PAP_NIXL_MAILBOX_XFER_POLL_SECONDS` controls transfer-state poll sleep and
  defaults to `0`, so active NIXL READ completion is busy-polled while empty
  notification polling still sleeps.
- `PAP_NIXL_MAILBOX_SLOT_PROTOCOL=1` is enabled by default. Bind metadata
  now includes the registered send-buffer base address and slot layout, and
  hot-path message notifications carry a fixed `slot_id` instead of a per-message
  remote address.
- `PAP_NIXL_MAILBOX_SLOT_COUNT=1` is the current default. `2` was tested and
  did not improve the sequential decode workload.
- `PAP_NIXL_MAILBOX_RECV_SLOT_COUNT` defaults to the send slot count. Incoming
  slot messages now read into the matching local receive-slot offset, and the
  dlist cache key includes that local slot address.
- `PAP_NIXL_MAILBOX_CACHE_XFER_DLISTS=1` is enabled by default. Receivers cache
  NIXL local/remote dlist side handles per stable slot/nbytes pair while still
  creating and releasing each transfer handle per message.
- `PAP_NIXL_MAILBOX_MSGPACK_NOTIF=1` is enabled by default. Mailbox
  notifications use `msgspec.msgpack` with a small prefix instead of JSON, while
  receivers still auto-detect older JSON notifications. Set `0` to force JSON
  for A/B or compatibility debugging.
- `PAP_NIXL_MAILBOX_ASYNC_SEND_SLOTS=1` lets the sender thread publish the
  next slot-protocol message without waiting for that message's ACK. Send-slot
  leases still prevent overwrite before the peer has read the slot. It remains
  opt-in because the same-period A/B regressed TPOT.
- `PAP_NIXL_MAILBOX_PIGGYBACK_ACKS=1` defers message ACKs and attaches pending
  ACK ids to the next outbound slot-protocol message, reducing steady-state ACK
  notifications from the hot path. It remains opt-in because the same-period
  A/B regressed TPOT.
- `PAP_NIXL_MAILBOX_INLINE_POLL=1` lets blocking recv/ACK waits poll
  notifications inline. It is opt-in because it regressed the warm TPOT A/B.
- `PAP_NIXL_MAILBOX_INLINE_PUBLISH=1` publishes the current message in the
  caller thread and waits for any previous ACK only before reusing the single
  send buffer. It is opt-in because it regressed the warm TPOT A/B.
- `PAP_NIXL_MAILBOX_ZERO_COPY_RECV=1` is enabled by default. It returns a
  tensor view over a leased receive slot instead of allocating/copying into a
  fresh tensor. Attention and Projection release mailbox messages after consuming
  the tensor, so the receiver will not overwrite a live zero-copy view.
- `PAP_NIXL_MAILBOX_RECV_SLOT_WAIT_SECONDS=30.0` bounds how long a zero-copy
  receive waits for a local receive slot when all candidate slots are leased.
- `PAP_NIXL_MAILBOX_SEND_SLOT_WAIT_SECONDS=30.0` bounds how long a
  multi-slot/async/piggyback sender waits for a reusable send slot.
- `PAP_NIXL_MAILBOX_NUM_THREADS=4` and
  `PAP_NIXL_MAILBOX_CAPTURE_TELEMETRY=1` are the current mailbox NIXL agent
  defaults. Lighter `num_threads=0,capture_telemetry=0` and `num_threads=8`
  A/Bs were slower on the warm benchmark.
- `PAP_NIXL_MAILBOX_TRACE=1` enables mailbox-internal publish/read timing. If
  unset, mailbox tracing follows `PAP_OFFLOAD_EXEC_TRACE`.
- For Qwen3-32B TP=2 `3PA1P` on 48 GiB GPUs, use
  `PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.78` or lower. `0.90` leaves almost no
  headroom for the attention executor colocated on each PA GPU and caused an
  attention OOM even though the benchmark client reported HTTP-level
  completions.
- `PAP_Q_FIRST_PROJECTION=1` is an experimental Qwen3-only path that slices the
  unquantized fused QKV projection, sends Q after Q-Proj/RoPE, then computes and
  sends K/V. It remains opt-in because the first controlled run regressed TPOT.
- `PAP_ATTENTION_Q_FIRST_PARTIAL=1` is an experimental Attention-side companion
  for Q-first Projection. The mailbox loop computes previous-token partial
  attention after receiving Q and before waiting for KV, then combines it with
  the current-token partial after KV arrives. It remains opt-in because the
  first CUDA benchmark regressed TPOT.
- Projection creates the mailbox transport from `vllm/model_executor/models/qwen3.py`.
- Attention creates the mailbox transport from `examples/pap/pap_attention_executor.py`.
- Projection binds to Attention through
  `/v1/pap/attention/offload-exec-mailbox/bind`.
- After binding, Projection skips the old TCP trigger and waits for mailbox
  output batches.

## Verification Snapshot

Focused tests:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_trace_summary.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_nixl_mailbox.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_contract.py -q
```

Latest observed result: `143 passed, 16 warnings`.

Compile/checks:

```bash
.venv/bin/python -m py_compile \
  vllm/pap/nixl_mailbox.py \
  vllm/pap/trace_summary.py \
  vllm/pap/remote_attention.py \
  vllm/pap/data_plane.py \
  vllm/pap/shadow_attention.py \
  vllm/model_executor/models/qwen3.py \
  examples/pap/pap_attention_executor.py \
  vllm/v1/worker/gpu/model_runner.py \
  tools/pap_trace_summary.py \
  tests/pap/test_pap_nixl_mailbox.py \
  tests/pap/test_pap_trace_summary.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_contract.py

git diff --check
```

Both completed successfully in the latest run.

## Benchmark Snapshot

### Qwen3-32B TP2 3PA1P stability point

Model: `/data/ssd1/llm-models/Qwen3-32B`
Workload: ctx128/o64/512 prompts, `request-rate=inf`, `max-concurrency=512`
Topology: `3PA1P`, TP=2 for every PA and Projection instance, 8 GPUs total
Transport: `PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox`

| Run | PA GPU mem util | Successful | Failed | Output tokens | Duration | Req/s | Out tok/s | Mean TTFT | Mean TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Invalid OOM run | 0.90 | 512 | 0 | 1731 | 321.28 s | 1.59 | 5.39 | 14462.07 ms | 3045.39 ms |
| Valid run | 0.78 | 512 | 0 | 32768 | 157.29 s | 3.26 | 208.32 | 14282.20 ms | 2224.55 ms |

The `0.90` run is invalid because attention executor logs show
`torch.OutOfMemoryError` on PA GPUs. The valid run left about 5 GiB free on PA
GPUs during decode and generated the full requested 512 * 64 output tokens.

Trace summary from the valid run:

- Projection macro batch `calls`: median 170, mean 155.30, p99 172.
- Projection ubatches per trace: median 3, mean 2.94, p99 3.
- Attention batch `calls`: median 57, mean 52.77, p99 65.
- Attention median timing: recv QKV 2.454 ms, compute 6.057 ms, send output
  0.019 ms, total 8.534 ms.
- Projection median remote-attention trace: send 2.016 ms, yield 17.602 ms,
  recv 1.849 ms, total 21.640 ms.

This confirms that NIXL mailbox is usable as the 3PA1P TP2 correctness baseline,
but it is still far from the PD 3P1D TP2 baseline for the same load
(32.62 s, 15.70 req/s, 1004.51 output tok/s, mean TPOT 210.53 ms).

### Qwen3-0.6B 1PA1P microbenchmarks

Model: `/data/ssd1/llm-models/Qwen3-0.6B`
Workload: `i512/o8/q1/prompts2`
Warmup: `1`

| System | Successful | Failed | Mean TTFT | Mean TPOT | Mean ITL |
| --- | ---: | ---: | ---: | ---: | ---: |
| PAP NIXL mailbox old handoff `1PA1P` | 2 | 0 | 177.52 ms | 56.54 ms | 56.54 ms |
| PAP NIXL mailbox default poll `1PA1P` | 2 | 0 | 102.52 ms | 24.64 ms | 24.64 ms |
| PAP NIXL mailbox transfer busy-poll default `1PA1P` | 2 | 0 | 93.11 ms | 19.91 ms | 19.91 ms |
| PAP NIXL mailbox slot protocol default `1PA1P` | 2 | 0 | 96.63 ms | 20.88 ms | 20.88 ms |
| PAP NIXL mailbox cached dlist default `1PA1P` | 2 | 0 | 101.67 ms | 21.70 ms | 21.70 ms |
| PAP NIXL mailbox recv-slot default rerun A `1PA1P` | 2 | 0 | 651.19 ms | 23.46 ms | 23.46 ms |
| PAP NIXL mailbox recv-slot default rerun B `1PA1P` | 2 | 0 | 641.96 ms | 25.33 ms | 25.33 ms |
| PAP NIXL mailbox recv-slot default rerun C `1PA1P` | 2 | 0 | 644.48 ms | 21.84 ms | 21.84 ms |
| PAP NIXL mailbox explicit 2-slot zero-copy A/B `1PA1P` | 2 | 0 | 645.92 ms | 23.51 ms | 23.51 ms |
| PAP NIXL mailbox explicit 2-slot copy A/B `1PA1P` | 2 | 0 | 650.92 ms | 24.66 ms | 24.66 ms |
| PAP NIXL mailbox leased zero-copy A/B `1PA1P` | 2 | 0 | 646.44 ms | 24.81 ms | 24.81 ms |
| PAP NIXL mailbox safe zero-copy default final `1PA1P` | 2 | 0 | 656.78 ms | 20.17 ms | 20.17 ms |
| PAP NIXL mailbox safe zero-copy explicit A/B `1PA1P` | 2 | 0 | 645.59 ms | 23.15 ms | 23.15 ms |
| PAP NIXL mailbox safe recv-slot2 zero-copy A/B `1PA1P` | 2 | 0 | 634.43 ms | 23.66 ms | 23.66 ms |
| PAP NIXL mailbox same-code copy-path A/B `1PA1P` | 2 | 0 | 656.40 ms | 26.17 ms | 26.17 ms |
| PAP NIXL mailbox ACK piggyback A/B `1PA1P` | 2 | 0 | 638.91 ms | 23.78 ms | 23.78 ms |
| PAP NIXL mailbox msgpack notification A/B `1PA1P` | 2 | 0 | 628.70 ms | 21.96 ms | 21.96 ms |
| PAP NIXL mailbox msgpack notification repeat `1PA1P` | 2 | 0 | 634.97 ms | 20.98 ms | 20.98 ms |
| PAP NIXL mailbox JSON notification same-period control `1PA1P` | 2 | 0 | 639.89 ms | 23.78 ms | 23.78 ms |
| PAP NIXL mailbox msgpack + piggyback A/B `1PA1P` | 2 | 0 | 645.37 ms | 22.19 ms | 22.19 ms |
| PAP NIXL mailbox msgpack default clean rerun `1PA1P` | 2 | 0 | 634.77 ms | 22.85 ms | 22.85 ms |
| PAP NIXL mailbox msgpack 10us default clean rerun `1PA1P` | 2 | 0 | 637.83 ms | 21.94 ms | 21.94 ms |
| PAP NIXL mailbox compact batch metadata `1PA1P` | 2 | 0 | 637.52 ms | 21.83 ms | 21.83 ms |
| PAP NIXL mailbox single-output no-cat clean rerun `1PA1P` | 2 | 0 | 648.07 ms | 21.97 ms | 21.97 ms |
| PAP NIXL mailbox no-cat plus projection empty output `1PA1P` | 2 | 0 | 639.80 ms | 20.78 ms | 20.78 ms |
| PAP NIXL mailbox direct recv output opt-in regression `1PA1P` | 2 | 0 | 635.57 ms | 23.07 ms | 23.07 ms |
| PAP NIXL mailbox direct output gated default rerun `1PA1P` | 2 | 0 | 629.07 ms | 21.82 ms | 21.82 ms |
| PAP NIXL mailbox qkv single-group no-cat default `1PA1P` | 2 | 0 | 633.88 ms | 21.57 ms | 21.57 ms |
| PAP NIXL mailbox segmented QKV default regression `1PA1P` | 2 | 0 | 635.33 ms | 22.58 ms | 22.58 ms |
| PAP NIXL mailbox segmented QKV gated default rerun `1PA1P` | 2 | 0 | 627.53 ms | 22.15 ms | 22.15 ms |
| PAP NIXL mailbox Q-first code default gated rerun `1PA1P` | 2 | 0 | 94.33 ms | 19.43 ms | 19.43 ms |
| PAP NIXL mailbox Q-first opt-in guard fallback `1PA1P` | 2 | 0 | 94.62 ms | 19.61 ms | 19.61 ms |
| PAP NIXL mailbox Q-first async two-slot split `1PA1P` | 2 | 0 | 99.06 ms | 20.44 ms | 20.44 ms |
| PAP NIXL mailbox Q-first Projection default gated rerun `1PA1P` | 2 | 0 | 96.79 ms | 20.56 ms | 20.56 ms |
| PAP NIXL mailbox Q-first Projection async split regression `1PA1P` | 2 | 0 | 108.05 ms | 25.76 ms | 25.76 ms |
| PAP NIXL mailbox Q-first Attention partial regression `1PA1P` | 2 | 0 | 199.03 ms | 64.58 ms | 64.58 ms |
| PAP NIXL mailbox async send-slot A/B `1PA1P` | 2 | 0 | 640.97 ms | 23.98 ms | 23.98 ms |
| PAP NIXL mailbox async send-slot2 A/B `1PA1P` | 2 | 0 | 644.68 ms | 24.19 ms | 24.19 ms |
| PAP NIXL mailbox async default ungated A/B `1PA1P` | 2 | 0 | 652.71 ms | 24.08 ms | 24.08 ms |
| PAP NIXL mailbox send-slot gate default rerun A `1PA1P` | 2 | 0 | 650.54 ms | 26.01 ms | 26.01 ms |
| PAP NIXL mailbox send-slot gate default rerun B `1PA1P` | 2 | 0 | 645.21 ms | 24.96 ms | 24.96 ms |
| PAP NIXL mailbox no dlist cache A/B `1PA1P` | 2 | 0 | 104.87 ms | 22.03 ms | 22.03 ms |
| PAP NIXL mailbox addr payload A/B `1PA1P` | 2 | 0 | 109.04 ms | 23.02 ms | 23.02 ms |
| PAP NIXL mailbox slot count 2 A/B `1PA1P` | 2 | 0 | 108.59 ms | 21.99 ms | 21.99 ms |
| PAP NIXL mailbox inline notification poll A/B `1PA1P` | 2 | 0 | 119.75 ms | 29.21 ms | 29.21 ms |
| PAP NIXL mailbox inline publish A/B `1PA1P` | 2 | 0 | 100.20 ms | 23.03 ms | 23.03 ms |
| PAP NIXL mailbox zero-copy recv A/B `1PA1P` | 2 | 0 | 96.04 ms | 20.48 ms | 20.48 ms |
| PAP NIXL mailbox 10us notification poll A/B `1PA1P` | 2 | 0 | 101.63 ms | 21.64 ms | 21.64 ms |
| PAP NIXL mailbox light agent config A/B `1PA1P` | 2 | 0 | 102.74 ms | 22.00 ms | 22.00 ms |
| PAP NIXL mailbox 8-thread agent A/B `1PA1P` | 2 | 0 | 98.29 ms | 21.93 ms | 21.93 ms |
| PAP NIXL mailbox busy poll `1PA1P` | 2 | 0 | 129.86 ms | 34.18 ms | 34.18 ms |
| PD `1P1D` | 2 | 0 | 34.92 ms | 4.25 ms | 4.25 ms |

Default-poll PAP result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_182825/1PA1P_i512_o8_q1.json`

Transfer busy-poll default PAP result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_185048/1PA1P_i512_o8_q1.json`

Slot protocol default PAP result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_192851/1PA1P_i512_o8_q1.json`

Cached dlist default PAP result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_193742/1PA1P_i512_o8_q1.json`

Recv-slot default rerun A result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_195717/1PA1P_i512_o8_q1.json`

Recv-slot default rerun B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_195820/1PA1P_i512_o8_q1.json`

Recv-slot default rerun C result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_195913/1PA1P_i512_o8_q1.json`

Explicit recv-slot 2-slot zero-copy A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_200059/1PA1P_i512_o8_q1.json`

Explicit recv-slot 2-slot copy A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_200158/1PA1P_i512_o8_q1.json`

Leased zero-copy A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_201537/1PA1P_i512_o8_q1.json`

Safe zero-copy default final result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_201649/1PA1P_i512_o8_q1.json`

Safe zero-copy explicit A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_201243/1PA1P_i512_o8_q1.json`

Safe recv-slot2 zero-copy A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_201139/1PA1P_i512_o8_q1.json`

Same-code copy-path A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_201347/1PA1P_i512_o8_q1.json`

ACK piggyback A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_203816/1PA1P_i512_o8_q1.json`

Msgpack notification A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_204809/1PA1P_i512_o8_q1.json`

JSON notification same-period control result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_204909/1PA1P_i512_o8_q1.json`

Msgpack + piggyback A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_205007/1PA1P_i512_o8_q1.json`

Invalid default msgpack port-collision run, do not use:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_205411/1PA1P_i512_o8_q1.json`

Msgpack default clean rerun result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_205922/1PA1P_i512_o8_q1.json`

Msgpack 10us default clean rerun result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_211202/1PA1P_i512_o8_q1.json`

Compact batch metadata result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_211336/1PA1P_i512_o8_q1.json`

Single-output no-cat clean rerun result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_071848/1PA1P_i512_o8_q1.json`

No-cat plus projection empty output result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_072154/1PA1P_i512_o8_q1.json`

Direct recv output opt-in regression result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_073208/1PA1P_i512_o8_q1.json`

Direct output gated default rerun result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_073517/1PA1P_i512_o8_q1.json`

QKV single-group no-cat default result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_073841/1PA1P_i512_o8_q1.json`

Segmented QKV default regression result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_075055/1PA1P_i512_o8_q1.json`

Segmented QKV gated default rerun result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_075445/1PA1P_i512_o8_q1.json`

Q-first code default gated rerun result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_080908/1PA1P_i512_o8_q1.json`

Q-first opt-in guard fallback result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_081619/1PA1P_i512_o8_q1.json`

Q-first async two-slot split protocol result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_081733/1PA1P_i512_o8_q1.json`

Q-first Projection default gated rerun result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_083142/1PA1P_i512_o8_q1.json`

Q-first Projection async split regression result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_083322/1PA1P_i512_o8_q1.json`

Q-first Attention partial regression result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_085456/1PA1P_i512_o8_q1.json`

Default trace-guided lower-level protocol audit result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_085859/1PA1P_i512_o8_q1.json`

Async send-slot A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_204016/1PA1P_i512_o8_q1.json`

Async send-slot2 A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_202557/1PA1P_i512_o8_q1.json`

Async default ungated A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_202652/1PA1P_i512_o8_q1.json`

Send-slot gate default rerun A result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_203116/1PA1P_i512_o8_q1.json`

Send-slot gate default rerun B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_203628/1PA1P_i512_o8_q1.json`

Invalid proxied rerun, kept only as a pitfall:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_195444/1PA1P_i512_o8_q1.json`

No dlist cache A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_193912/1PA1P_i512_o8_q1.json`

Addr payload A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_192540/1PA1P_i512_o8_q1.json`

Slot count 2 A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_192956/1PA1P_i512_o8_q1.json`

Inline notification poll A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_184627/1PA1P_i512_o8_q1.json`

Inline publish A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_185729/1PA1P_i512_o8_q1.json`

Zero-copy receive A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_190232/1PA1P_i512_o8_q1.json`

10us notification poll A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_190331/1PA1P_i512_o8_q1.json`

Light agent config A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_190655/1PA1P_i512_o8_q1.json`

8-thread agent A/B result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_190856/1PA1P_i512_o8_q1.json`

Busy-poll PAP result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_182934/1PA1P_i512_o8_q1.json`

Old PAP result file:
`/home/fei/research/PD/test/baseline/pap/results/runs/20260525_174449/1PA1P_i512_o8_q1.json`

PD result file:
`/home/fei/research/PD/test/baseline/disaggregated/results/runs/20260525_174550/1P1D_i512_o8_q1.json`

## 2026-05-25 Continuation Update

Added first-pass mailbox hot-path instrumentation and removed the fixed 500us
mailbox polling sleeps from the default path:

- Empty notification polling now sleeps `PAP_NIXL_MAILBOX_POLL_SECONDS`, default
  `0.00005`, only when no notifications were handled.
- Transfer-state polling now sleeps `PAP_NIXL_MAILBOX_XFER_POLL_SECONDS`, default
  `0`, only while NIXL reports `PROC`. This avoids adding a 50us sleep to each
  active READ state check without busy-spinning the idle notification poll loop.
- Mailbox trace logs split queue, pack, send-buffer copy, notification, ACK
  wait, descriptor prep, transfer wait, and receive materialization.
- Trace timing calls are gated behind trace mode so the default hot path only
  carries the lower polling latency.
- Projection trace logging now reads `offload_exec_batches[0][3]`, the batch
  descriptor, instead of the endpoint string at index `2`. This fixes profiling
  with `PAP_OFFLOAD_EXEC_TRACE=1`.
- The default 50us notification polling setting reduced the handoff TPOT from
  `56.54 ms` to `24.64 ms` on the same warm benchmark. Full notification busy
  polling was worse on this workload, with `34.18 ms` TPOT.
- Attention-side mailbox batch tracing showed median attention compute around
  `0.143 ms`, median mailbox receive materialization around `0.035 ms`, and
  median per-direction send ACK waits around `0.38-0.40 ms`. The mailbox chain,
  not the attention kernel, remains the main TPOT cost.
- Inline notification polling was tested and left opt-in: it regressed TPOT to
  `29.21 ms`, while disabling it returned to `24.32 ms`.
- Making transfer-state polling busy by default improved the same warm benchmark
  to `19.91 ms` TPOT.

## 2026-05-25 Follow-up A/B Results

The next continuation tested several narrower mailbox hot-path hypotheses after
transfer busy-polling became the best default:

- Inline publish removed the sender queue for the current message, while still
  waiting for a previous ACK before reusing the single send buffer. This was
  functionally correct in focused tests but regressed warm TPOT to `23.03 ms`,
  so it remains opt-in.
- Zero-copy receive returned a view over the registered receive buffer and
  avoided the receive materialization allocation/copy. It passed focused tests
  but measured `20.48 ms` TPOT, not better than the `19.91 ms` best default.
  It should be revisited only with a real multi-slot receive ring.
- Reducing empty notification poll sleep from 50us to 10us measured `21.64 ms`
  TPOT, and full busy polling had already measured `34.18 ms`; keep the 50us
  default.
- Replacing the tuned mailbox NIXL agent config with the NIXL library defaults
  (`num_threads=0`, telemetry disabled) measured `22.00 ms`, while 8 threads
  measured `21.93 ms`; keep `num_threads=4,capture_telemetry=1` as the current
  default.

These A/Bs strengthen the conclusion that the next meaningful improvement is
not another scalar polling/copy tweak. The remaining architecture work should
move to a persistent slot/ring protocol that can safely combine zero-copy receive
views, fixed slot metadata, and ACK reuse/piggybacking.

## 2026-05-25 Slot Protocol Update

This continuation implemented the first persistent-slot protocol slice:

- The mailbox bind metadata can now wrap the NIXL agent metadata with the
  sender registered-buffer base address, device id, slot count, and slot size.
- Updated peers decode this metadata at bind time and can resolve incoming
  `slot_id` payloads to remote NIXL READ addresses without carrying an `addr`
  field on every message notification. Raw NIXL metadata remains accepted for
  compatibility.
- `PAP_NIXL_MAILBOX_SLOT_PROTOCOL` is now enabled by default because the
  same-period A/B measured `21.51 ms` TPOT with slot metadata versus `23.02 ms`
  with the legacy addr payload. The final default run measured `20.88 ms`.
- `PAP_NIXL_MAILBOX_SLOT_COUNT=2` measured `21.99 ms`, so multi-slot buffering
  is not yet a default. The likely reason is that the decode loop is still
  sequential and the sender thread still waits for ACK per message.

This is not the full ring-buffer architecture yet, but it establishes the
protocol anchor required for the next step: per-slot receive buffers, cached
per-slot transfer descriptors/handles where NIXL allows, and ACK reuse or
piggybacking.

## 2026-05-25 Cached Dlist Update

The follow-up after fixed slot metadata caches NIXL dlist side handles on the
receive path:

- Receivers now cache local and remote dlist handles by stable tuple
  `(local_recv_addr, nbytes, local_device_id, remote_slot_addr, remote_device_id)`.
- Each message still creates and releases its own transfer handle with
  `make_prepped_xfer`, matching the existing NIXL connector usage pattern. Only
  the descriptor/list side handles are reused.
- Cached dlist handles are released when the endpoint closes. Focused tests
  cover cache reuse and cleanup.
- Same-period A/B measured `21.70 ms` TPOT with cache enabled versus `22.03 ms`
  with `PAP_NIXL_MAILBOX_CACHE_XFER_DLISTS=0`, so the cache remains default-on.

This is another incremental ring-buffer prerequisite: once per-slot receive
buffers are introduced, the cache key can move from one shared recv buffer to
per-slot recv descriptors, and then zero-copy receive can be made safe by
construction.


## 2026-05-25 Per-Slot Receive Update

This continuation implemented the local receive-buffer side of the slot protocol:

- Endpoints now derive `PAP_NIXL_MAILBOX_RECV_SLOT_COUNT` from
  `PAP_NIXL_MAILBOX_SLOT_COUNT` by default and compute a fixed receive-slot size.
- Incoming slot messages read into `recv_buffer + slot_id * recv_slot_bytes`
  modulo the local receive slot count, instead of always using the start of the
  shared receive buffer.
- The cached dlist key now includes the local receive-slot address, so each
  `(local slot, remote slot, nbytes)` pair can reuse the right descriptor/list
  handles without aliasing another receive slot.
- Zero-copy materialization now returns a tensor view over the selected receive
  slot slice, not always `_recv_buffer[:nbytes]`.
- Focused tests cover the slot-local descriptor address and zero-copy tensor view.

Same-period A/Bs were noisy and should not be over-read. With `NO_PROXY` set for
local requests, three default reruns after the receive-slot implementation
measured `23.46 ms`, `25.33 ms`, and `21.84 ms` TPOT. Explicit
`PAP_NIXL_MAILBOX_SLOT_COUNT=2 PAP_NIXL_MAILBOX_RECV_SLOT_COUNT=2` measured
`24.66 ms` with copy materialization and `23.51 ms` with
`PAP_NIXL_MAILBOX_ZERO_COPY_RECV=1`. The invalid `20260525_195444` run returned
`Forbidden` because `HTTP_PROXY/HTTPS_PROXY` captured localhost traffic; keep the
existing `NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost`
requirement.

Conclusion: per-slot receive addressing is a necessary correctness prerequisite,
but not a standalone TPOT breakthrough. Zero-copy remains opt-in because the
current task pool does not own receive-slot lifetimes, and the explicit 2-slot
zero-copy run did not beat the best transfer-busy-poll baseline. The next
architectural step is a real slot lease/release protocol: ACK should mean the
sender slot was read, while the receiver slot must not be reused until the queued
tensor view is consumed.


## 2026-05-25 Receive-Slot Lease Update

This continuation made zero-copy receive safe enough to become the mailbox
default on the current PAP hot path:

- `PAPMailboxMessage` now carries an optional release callback and exposes
  idempotent `message.release()`.
- Zero-copy mailbox reads reserve a free local receive slot before issuing the
  NIXL READ. If the preferred slot is still leased, the receiver scans the local
  slot ring and waits up to `PAP_NIXL_MAILBOX_RECV_SLOT_WAIT_SECONDS`, default
  `30.0`, instead of overwriting a queued tensor view. Failed transfers release
  their reserved slot before propagating the error.
- Attention-side mailbox consumption now receives the message object and releases
  QKV after attention compute. Projection receives the output message object and
  releases it after copying the batch into the model output buffer.
- `PAP_NIXL_MAILBOX_ZERO_COPY_RECV` is now default-on because the same-code
  A/B measured `20.17 ms` TPOT with safe zero-copy versus `26.17 ms` with the
  copy materialization path.
- Focused tests cover free-slot selection, idempotent release, transfer-failure
  release, Attention-loop release, and the Projection contract.

Latest controlled runs with `NO_PROXY=127.0.0.1,localhost`:

- Default safe zero-copy: `20.17 ms` TPOT, result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260525_201649/1PA1P_i512_o8_q1.json`.
- Same-code copy path: `26.17 ms` TPOT, result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260525_201347/1PA1P_i512_o8_q1.json`.
- Safe zero-copy with `PAP_NIXL_MAILBOX_RECV_SLOT_COUNT=2`: `23.66 ms` TPOT,
  result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260525_201139/1PA1P_i512_o8_q1.json`.

The gain is meaningful relative to the adjacent copy-path run, but PAP remains
far from PD (`4.25 ms` TPOT). The next bottleneck is still the per-layer
message/ACK/JSON/READ loop; the next architectural cut should batch or pipeline
multiple layer messages through a persistent ring rather than continuing with
scalar per-message tweaks.

## 2026-05-25 Send-Slot Lease Update

This continuation made sender slot ownership explicit but kept the new behavior
off the default single-slot sequential path:

- ACK handling now clears the sender output pool and releases any leased send
  slot through the same helper. Piggybacked ACKs use the same cleanup path before
  processing the incoming message.
- Multi-slot, async, and piggyback publishing reserve the first free send
  slot instead of rotating blindly into a slot whose previous message has not
  been ACKed. The release path
  is covered for direct ACKs and piggybacked ACKs.
- The sender loop may skip immediate ACK waits only when
  `PAP_NIXL_MAILBOX_ASYNC_SEND_SLOTS=1` or piggyback ACK mode is enabled. The
  default `PAP_NIXL_MAILBOX_SLOT_COUNT=1` synchronous path avoids send-slot lease
  bookkeeping.
- Focused tests cover free-slot selection, ACK release, piggyback cleanup, async
  sender-loop behavior, and the single-slot synchronous non-lease path.

The measured result was not a win for the current sequential decode loop.
`PAP_NIXL_MAILBOX_SLOT_COUNT=2 PAP_NIXL_MAILBOX_ASYNC_SEND_SLOTS=1` measured
`24.19 ms` TPOT, and the single-slot default reruns after gating measured
`26.01 ms` and `24.96 ms`. Treat send-slot leasing as a correctness prerequisite
for a future pipelined/ring design, not as the next default TPOT optimization.

## 2026-05-25 ACK Piggyback / Async Send Update

This continuation tested two ways to remove ACK waiting/notification work from
steady-state decode without changing mailbox tensor ownership:

- `PAP_NIXL_MAILBOX_PIGGYBACK_ACKS=1` keeps ACK semantics but queues received
  message ids locally and attaches them to the next outbound slot-protocol
  message as an `acks` list. In the normal Projection->Attention->Projection
  decode loop, the Attention output can ACK the QKV read, and the next QKV can
  ACK the previous output read. Focused tests cover ACK deferral, ACK drain into
  the next message, inbound piggyback ACK processing, and the sender-loop rule
  that piggyback mode cannot wait for immediate ACK notifications.
- `PAP_NIXL_MAILBOX_ASYNC_SEND_SLOTS=1` lets the sender thread publish without
  waiting for the message ACK while send-slot leases still prevent buffer reuse
  before the peer has read that slot.

Both were functionally correct in focused tests, but neither improved the warm
`1PA1P i512/o8/q1/prompts2` benchmark:

- ACK piggyback: `23.78 ms` TPOT, result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260525_203816/1PA1P_i512_o8_q1.json`.
- Async send slots: `23.98 ms` TPOT, result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260525_204016/1PA1P_i512_o8_q1.json`.

Keep both paths opt-in. This result suggests the dominant residual cost is not
just the extra ACK notification or sender-thread ACK wait; the larger per-layer
Python/message/READ sequencing remains the next target.

## 2026-05-25 Msgpack Notification Update

This continuation replaced hot-path mailbox notification JSON with an
auto-detected msgpack codec and made it default-on:

- `_encode_nixl_mailbox_notification()` emits JSON or msgpack. Msgpack payloads
  carry a `PAPM1` prefix so `_decode_nixl_mailbox_notification()` can still
  accept older JSON notifications without an out-of-band protocol flag.
- `PAP_NIXL_MAILBOX_MSGPACK_NOTIF=1` is now the default. Tests cover msgpack
  round-trip, sender-side publish, ACK handling from msgpack notifications, and
  the default contract.
- Same-period A/B: msgpack measured `21.96 ms` TPOT versus `23.78 ms` for the
  JSON control. A clean default rerun after flipping the default measured
  `22.85 ms` TPOT.
- An additional controlled msgpack run measured `20.98 ms` TPOT, result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260525_205228/1PA1P_i512_o8_q1.json`.
- Combining msgpack with `PAP_NIXL_MAILBOX_PIGGYBACK_ACKS=1` measured
  `22.19 ms` TPOT, so piggyback remains opt-in.

This is a modest win and does not change the main conclusion: PAP remains far
from PD (`4.25 ms` TPOT), and the next large cut still needs to collapse the
per-layer Python/message/READ sequencing rather than only changing codecs.

## 2026-05-26 Decode Hot-Path Allocation Update

This continuation kept the msgpack + 10us mailbox defaults and removed two
per-layer allocations/copies that are common in qps1 decode:

- Attention executor now sends the only computed output tensor directly when an
  OFFLOAD_EXEC batch has one item, instead of always building a new
  `torch.cat(outputs, dim=0)` batch tensor. The mailbox loop and TCP-trigger
  batch helper share the same helper.
- Projection now allocates the PAP attention output buffer with
  `torch.empty_like(query)` when every scheduled request is routed to remote
  Attention. It keeps `torch.zeros_like(query)` only for the partial-offload
  fallback where some rows may remain intentionally zero.
- The OFFLOAD_EXEC batch descriptor metadata already uses compact v2 arrays
  (`v/l/r/s/a`) and remains backward-compatible with legacy `items` metadata.

Verification after these changes: focused PAP suite `119 passed, 16 warnings`.

Controlled `1PA1P i512/o8/q1/prompts2` results:

- Msgpack + 10us default before these allocation cuts: `21.94 ms` TPOT, result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260525_211202/1PA1P_i512_o8_q1.json`.
- Compact metadata run: `21.83 ms` TPOT, result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260525_211336/1PA1P_i512_o8_q1.json`.
- Single-output no-cat only: `21.97 ms` TPOT, result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_071848/1PA1P_i512_o8_q1.json`.
- Single-output no-cat plus Projection empty output: `20.78 ms` TPOT, result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_072154/1PA1P_i512_o8_q1.json`.

The useful gain came from avoiding Projection-side zero fill on the fully
offloaded decode path; single-output no-cat is still worth keeping because it
removes a per-layer allocation even though the isolated benchmark was within
noise.

A follow-up tested returning the mailbox receive tensor directly into `o_proj`
and deferring message release until after `o_proj`. That path is functionally
covered but regressed to `23.07 ms` TPOT, so it is gated behind the opt-in
`PAP_DIRECT_MAILBOX_OUTPUT=1` flag and is not a default. The likely cause is
that the borrowed receive-buffer view is a worse input to the projection GEMM
than the copied/owned output buffer.

Projection now also avoids the second `torch.cat([single_qkv])` when a qps1
OFFLOAD_EXEC group has one QKV item. The default rerun after gating direct
output and adding QKV single-group no-cat measured `21.57 ms` TPOT.

PAP is still far from PD (`4.25 ms` TPOT). The next material step remains
changing the per-layer synchronous Projection-Attention-Projection handoff, not
only shrinking individual messages.

## 2026-05-26 Paper-Guided Low-Level Protocol Experiment

The referenced Lamina paper points to two relevant directions for PAP:

- Section 4.1/FHBN: reduce per-layer GPU-aware networking control overhead and
  avoid CPU involvement on the critical path.
- Section 4.2.2: split decode attention into previous-token and current-token
  parts, send Q as soon as Q-Proj is ready, then send K/V later, hiding part of
  communication behind K/V and remaining model-slice work.

This stage tested a small current-code approximation of the lower-level protocol
idea: NIXL mailbox messages can now carry `payload_segments`, allowing Projection
to pass Q/K/V segments without constructing an intermediate packed QKV tensor.
The endpoint writes these segments sequentially into the registered send slot and
publishes one normal mailbox notification. Focused tests cover the data-plane API
and endpoint segment copy behavior.

Result: enabling segmented QKV by default regressed the same `1PA1P i512/o8/q1`
benchmark to `22.58 ms` TPOT, result
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_075055/1PA1P_i512_o8_q1.json`.
The likely reason is that three separate GPU copies into the send slot are worse
than one packed tensor copy at this payload size. The path is therefore gated
behind `PAP_SEGMENTED_QKV=1`, and the default rerun measured `22.15 ms` TPOT,
result
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_075445/1PA1P_i512_o8_q1.json`.

Conclusion: simple segmented copy is not the FHBN-like protocol win. The next
stage should implement a queue-first / Q-first, KV-later pipeline at the model
slice level: send Q as soon as Q-Proj/RoPE produces it, let Attention start the
previous-token partial attention, then send K/V and combine with the current-token
contribution. That is the paper-aligned path most likely to reduce TPOT rather
than moving copies around inside the existing synchronous round trip.

## Paper-Guided Q-First/KV-Later Protocol Stage

This continuation added the first explicit Q-first/KV-later mailbox protocol,
gated behind `PAP_Q_FIRST_KV_LATER=1`:

- `PAPOffloadExecBatchDescriptor` now exposes `query_tensor_id` and
  `kv_tensor_id` alongside the existing packed `qkv_tensor_id`.
- `PAPNixlMailboxOffloadExecTransport` can send `attention_query_batch` and
  `attention_kv_batch` messages, and Attention-side consumption can reassemble
  those two messages into the existing packed-QKV compute path.
- Qwen3 Projection uses the split send path only when the transport reports
  `supports_query_first_kv_later`; otherwise `PAP_Q_FIRST_KV_LATER=1` safely
  falls back to the existing packed-QKV send.

A first opt-in benchmark without that guard reproduced a deadlock: the default
single-slot synchronous mailbox sender waits for the Q ACK before publishing KV,
while Attention waits for the KV message after receiving Q. The guard now requires
`PAP_NIXL_MAILBOX_ASYNC_SEND_SLOTS=1` and `PAP_NIXL_MAILBOX_SLOT_COUNT>=2` before
the split protocol can activate. Focused regression tests cover this capability
gate and the query/KV reassembly path.

Verification after these changes:

- `.venv/bin/python -m py_compile vllm/pap/data_plane.py vllm/model_executor/models/qwen3.py tests/pap/test_pap_data_plane.py tests/pap/test_pap_contract.py`
- `.venv/bin/python -m pytest tests/pap/test_pap_nixl_mailbox.py tests/pap/test_pap_data_plane.py tests/pap/test_pap_attention_executor.py tests/pap/test_pap_contract.py -q`

Latest observed result: `123 passed, 16 warnings`.

Controlled `1PA1P i512/o8/q1/prompts2` results:

- Default gated code path: `19.43 ms` TPOT, result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_080908/1PA1P_i512_o8_q1.json`.
- `PAP_Q_FIRST_KV_LATER=1` without async two-slot capability: safe fallback,
  `19.61 ms` TPOT, result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_081619/1PA1P_i512_o8_q1.json`.
- Actual split-message protocol with `PAP_NIXL_MAILBOX_ASYNC_SEND_SLOTS=1` and
  `PAP_NIXL_MAILBOX_SLOT_COUNT=2`: `20.44 ms` TPOT, result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_081733/1PA1P_i512_o8_q1.json`.

Conclusion: split messages alone are not yet a TPOT win because Qwen3 still uses
a fused `QKVParallelLinear`, so Q/K/V are all computed before `_compute_pap_attention`,
and Attention still waits for KV before doing the same packed-QKV computation.
The next paper-aligned stage must split the Projection compute itself: run Q-Proj
and RoPE early, send Q immediately, then run K/V-Proj and send KV. After that,
Attention needs a partial previous-token attention path so the early Q arrival
can overlap useful Attention work with Projection-side K/V computation.

## Paper-Guided Q-First Projection Compute Stage

This continuation implemented that first Projection-side split as an explicit
opt-in path behind `PAP_Q_FIRST_PROJECTION=1`:

- Qwen3 checks that PAP decode routing is active, the transport supports the
  Q-first/KV-later mailbox protocol, and the fused QKV projection is an
  unquantized sliceable linear before changing the compute path.
- The path computes the Q slice first, applies Q norm and Q-only RoPE, sends an
  `attention_query_batch`, then computes the K/V slice, applies K norm and
  K-only RoPE, and calls `_compute_pap_attention(..., query_already_sent=True)`
  so only the KV follow-up is sent.
- Unsupported transports fall back before doing the extra Q-only GEMM, avoiding
  accidental overhead when the split mailbox protocol cannot run.

Controlled `1PA1P i512/o8/q1/prompts2` results:

- Default gated rerun with the code present but `PAP_Q_FIRST_PROJECTION=0`:
  `20.56 ms` TPOT, result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_083142/1PA1P_i512_o8_q1.json`.
- Actual split Projection compute with `PAP_Q_FIRST_PROJECTION=1`,
  `PAP_NIXL_MAILBOX_ASYNC_SEND_SLOTS=1`, and
  `PAP_NIXL_MAILBOX_SLOT_COUNT=2`: `25.76 ms` TPOT, result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_083322/1PA1P_i512_o8_q1.json`.

Conclusion: the paper-aligned ordering is now represented in code, but it is
not a performance win yet. The likely causes are two smaller GEMMs instead of
one fused QKV GEMM, PyTorch-native Q/K RoPE calls, two mailbox messages, and no
Attention-side previous-token partial computation to consume early Q while
Projection is still computing K/V. Keep `PAP_Q_FIRST_PROJECTION` off by default.
The next meaningful stage should add the Attention-side partial path or move the
split projection/send sequence into a lower-level fused path where Q can arrive
early without paying Python/GEMM fragmentation overhead.

## Paper-Guided Q-First Attention Partial Stage

This continuation added the missing Attention-side consumer for the Q-first
protocol, gated behind `PAP_ATTENTION_Q_FIRST_PARTIAL=1`:

- The NIXL mailbox transport can now expose an `attention_query_batch` message
  immediately through `recv_next_attention_batch_message()` and later receive
  the matching `attention_kv_batch` through `recv_kv_batch_message()`. The
  older `recv_next_qkv_batch_message()` reassembly path remains available for
  compatibility.
- `remote_attention.py` now has a numerically stable partial attention state:
  per-query/head max score, exp denominator, and weighted value numerator.
  Combining previous-token and current-token states matches full attention in
  focused tests.
- The Attention mailbox loop can compute previous-token partial attention as
  soon as Q arrives, then wait for KV, append the current KV, compute the
  current-token partial, combine the two states, and send the normal output.

Verification after these changes:

- `.venv/bin/python -m py_compile vllm/pap/remote_attention.py vllm/pap/data_plane.py examples/pap/pap_attention_executor.py tests/pap/test_pap_remote_attention.py tests/pap/test_pap_data_plane.py tests/pap/test_pap_attention_executor.py`
- `.venv/bin/python -m pytest tests/pap/test_pap_remote_attention.py tests/pap/test_pap_nixl_mailbox.py tests/pap/test_pap_data_plane.py tests/pap/test_pap_attention_executor.py tests/pap/test_pap_contract.py -q`

Latest observed result: `142 passed, 16 warnings`.

Controlled `1PA1P i512/o8/q1/prompts2` result with
`PAP_Q_FIRST_PROJECTION=1`, `PAP_ATTENTION_Q_FIRST_PARTIAL=1`,
`PAP_NIXL_MAILBOX_ASYNC_SEND_SLOTS=1`, and `PAP_NIXL_MAILBOX_SLOT_COUNT=2`:
`64.58 ms` TPOT, result
`/home/fei/research/PD/test/baseline/pap/results/runs/20260526_085456/1PA1P_i512_o8_q1.json`.

Conclusion: the paper's partial-overlap semantics are now represented and tested,
but the current implementation is slower than the packed SDPA path. The root
cause is that partial-state attention needs the softmax denominator/LSE, so this
prototype uses explicit score/numerator/denominator einsum kernels instead of the
CUDA SDPA path used by the packed fallback. Keep
`PAP_ATTENTION_Q_FIRST_PARTIAL` off by default. The next viable version of this
idea needs a lower-level attention kernel that returns partial output plus LSE,
or a fused Projection/send/Attention kernel path; implementing the overlap only
in Python is the wrong performance level.

## Trace-Guided Lower-Level Protocol Audit

After the Q-first partial experiment regressed, this continuation returned to
the paper's lower-level protocol point and measured the current default mailbox
path with `PAP_OFFLOAD_EXEC_TRACE=1` and both Q-first flags disabled. It also
added `vllm/pap/trace_summary.py` plus `tools/pap_trace_summary.py` so future
protocol experiments can be compared from the service logs instead of manually
reading thousands of per-layer trace lines.

Controlled default trace run:

- Command shape: `PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox`,
  `PAP_OFFLOAD_EXEC_TRACE=1`, `PAP_Q_FIRST_PROJECTION=0`,
  `PAP_ATTENTION_Q_FIRST_PARTIAL=0`, `1PA1P i512/o8/q1/prompts2`.
- Benchmark result: `25.00 ms` TPOT, result
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260526_085859/1PA1P_i512_o8_q1.json`.
- Trace parser command:
  `.venv/bin/python tools/pap_trace_summary.py /home/fei/research/PD/test/baseline/pap/results/runs/20260526_085859/service_logs`.

Median trace findings after excluding >10ms warmup/outliers:

| Trace point | Median | Interpretation |
| --- | ---: | --- |
| Projection `send_ms` | 0.028 ms | enqueue/send call is not the bottleneck |
| Projection `recv_ms` | 0.732 ms | waits for Attention output and opposing slice work |
| Attention `recv_qkv_ms` | 0.819 ms | waits for next Projection QKV and opposing slice work |
| Attention `compute_ms` | 0.144 ms | attention kernel itself is small |
| Attention NIXL READ total | 0.097 ms | lower-level READ path is already sub-0.1ms |
| Projection NIXL READ total | 0.087 ms | output READ path is already sub-0.1ms |
| Projection mailbox publish | 0.038 ms | notification/copy publish is small |
| Attention mailbox publish | 0.040 ms | notification/copy publish is small |
| Projection sender ACK wait | 0.252 ms | visible in sender thread, but not the model-thread send call |
| Attention sender ACK wait | 0.230 ms | visible in sender thread, but not the attention-loop enqueue call |

Conclusion: the current NIXL mailbox data movement is not the full TPOT gap. The
sub-0.1ms READ path and ~0.04ms publish path are already much smaller than the
~0.8-1.0ms per-layer alternation between Projection and Attention. This explains
why scalar protocol tweaks such as msgpack, piggyback ACKs, inline polling, and
segmented copies move TPOT only modestly or regress. To approach PD TPOT, the
next low-level implementation must reduce the sequential slice boundary itself:
either a kernel-level Q-first partial attention path that preserves the fast
attention backend/LSE, or a more invasive model-slice pipeline that can overlap
Projection work with Attention work without falling back to Python-level einsum.

## Current Bottleneck Hypothesis

The current mailbox is functionally correct but too expensive because it is
still a per-layer synchronous RPC loop:

```text
Projection layer N sends QKV -> Attention reads QKV -> Attention computes ->
Attention sends output -> Projection waits -> Projection layer N+1 starts
```

For Qwen3-0.6B this repeats across 28 layers. The latest transfer busy-poll
default run is `19.91 ms / 28`, about `0.71 ms/layer`, which is still far above
the measured attention-kernel cost and still much slower than PD.

Likely contributors:

- Python queue/thread wakeups.
- Codec/metadata work per message.
- ACK notification per message.
- NIXL descriptor/handle prep per message.
- NIXL READ setup per message.
- Extra device copies into and out of mailbox buffers.
- No persistent cross-layer ring protocol.
- No cross-request pipelining beyond whatever batch happens at one layer.

## Next Engineering Tasks

1. Replace the current message runtime with a persistent slot/ring-buffer design:
   - pre-register buffers once,
   - use fixed slot ids instead of JSON tensor descriptions where possible,
   - batch or pipeline ACK/release metadata with cross-layer work items,
   - avoid descriptor/handle prep in the hot path if NIXL allows cached handles,
   - avoid per-message output tensor allocation/copy when a receive slot can be
     handed directly to the consumer,
   - keep a fallback timeout/error path for failed peer delivery.
2. Compare against both PD and the previous PAP batch TCP-trigger path. The
   NIXL path must beat the old TCP-trigger path before it is worth deeper
   integration.

## Operational Notes

- Always run benchmark requests with
  `NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost`.
- Always include warmup before reporting latency.
- Use `.venv/bin/python`; do not use system `python3` or bare `pip`.
- Clean up orphan vLLM engine processes before rerunning GPU benchmarks.
