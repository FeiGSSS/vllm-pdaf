# PAP deferred QKV fan-out submission

> Controlled 7PA1P development validation. Raw AIPerf and trace artifacts
> remain machine-local under `experiments/_staging/`.

Date: 2026-07-27

## Decision

Reject and roll back deferred collective submission.

The original path was already concurrent: as soon as one PA shard was ready,
Projection asynchronously enqueued it on that peer's independent CUDA stream
and continued constructing the next shard. Collecting every shard before
iterating over the same asynchronous sends did not create new GPU concurrency,
reduce copies, or reduce doorbells. It only delayed the earliest peer.

The first implementation added a common GPU release gate after all host-side
submissions. That treatment was rejected and removed before commit: it made
the returns more tightly grouped but delayed the barrier completion and
reduced end-to-end throughput.

The later ungated deferred-submission treatment showed no stable benefit and
was also rolled back. Fan-in tracing introduced during the experiment remains
because it is independent diagnostic instrumentation.

## Workload

- Qwen3-8B FP16, eager execution, 7PA1P on eight NVIDIA L20 GPUs.
- 128 conversations, five turns, 640 requests, concurrency 32.
- Long-tail randomized multi-turn input, randomized 16-64-token output.
- `max_model_len=32768`, PA memory utilization 0.90.
- Prefill/Projection `max_num_seqs=256`.
- Prefill `max_num_batched_tokens=32768`.
- Projection `max_num_batched_tokens=256`.
- Attention-load routing with the validated sparse-migration settings.

## Results

All trace-off runs completed 640/640 requests with zero AIPerf errors and
drained all seven Attention sessions.

| Treatment | Repetitions | TTFT avg | ITL avg | Request throughput |
| --- | ---: | ---: | ---: | ---: |
| Previous sparse-routing baseline | 2 | 1,665.19 ms | 49.80 ms | 5.225 req/s |
| Rejected common GPU gate | 2 | 1,684.98 ms | 50.67 ms | 5.065 req/s |
| Deferred submission, independent streams | 1 | 1,651.09 ms | 48.47 ms | 5.220 req/s |

The common gate regressed mean ITL by 1.7% and throughput by 3.1% relative to
the prior two-run baseline. Deferred submission without the gate restored
throughput within 0.1% of the baseline, but its single-run ITL difference is
not stable evidence. It was removed because it added no concurrency absent
from the original immediate asynchronous submission path.

The gate trace explains the negative result:

| Per-layer fan-in metric | Before gate | Common gate | Change |
| --- | ---: | ---: | ---: |
| First PA ready, median | 0.250 ms | 0.352 ms | +40.8% |
| Last PA ready, median | 0.635 ms | 0.685 ms | +7.9% |
| Last-minus-first spread, median | 0.339 ms | 0.308 ms | -9.1% |
| Last-minus-first spread, P90 | 0.883 ms | 0.850 ms | -3.7% |

The gate synchronized start times, but barrier latency depends on the last
return rather than the width of the return distribution. Delaying the fast PA
therefore cannot solve load-driven Attention skew.

## Rollback validation

After post-Prefill Decode placement was implemented, immediate asynchronous
submission was restored and tested twice on the same S128/C32 workload. Both
runs completed 640/640 requests with zero errors, zero migration misses, passed
all runtime audits, and drained every Attention session.

| Metric | Deferred submission | Immediate run 1 | Immediate run 2 |
| --- | ---: | ---: | ---: |
| Request throughput | 5.312 req/s | 5.028 req/s | 5.089 req/s |
| Mean TTFT | 1,566.33 ms | 1,731.50 ms | 1,679.77 ms |
| Mean ITL | 49.03 ms | 54.14 ms | 50.09 ms |
| Successful migrations | 5 | 4 | 5 |

The first rollback repetition was substantially slower; the second was close
to the older immediate-submit baseline. These non-interleaved runs do not
prove equivalence, nor do they prove that deferred submission is stably
faster. The rollback is an architectural decision: deferred submission did
not introduce new GPU concurrency and had no established repeatable benefit.
The observed variance is retained rather than presenting the rollback as a
performance improvement.

## Validation

```text
.venv/bin/python -m pytest \
  tests/pap/test_pap_local_fast_transport.py \
  tests/pap/test_pap_qwen3_async_send.py \
  tests/pap/test_pap_trace_summary.py -q

Rollback source: 18 passed, 2 skipped
```

`py_compile` and `git diff --check` passed for the changed source and tests.
The repository virtual environment does not contain the `ruff` module, so the
attempted focused ruff check was unavailable and is not counted as passing.

Raw artifacts:

```text
benchmarks/pap/experiments/_staging/scheduling/
  20260727_collective_fanout/
    comparison/
    runs/
      7pa1p_collective_fanout_trace_s32_c32/
      7pa1p_collective_fanout_formal_s128_c32/
      7pa1p_collective_fanout_formal_s128_c32_rep2/
      7pa1p_collective_streams_final_s128_c32/
  20260727_fanout_rollback/runs/
    immediate_async_s128_c32/
    immediate_async_s128_c32_rep2/
```

The first three run directories preserve the rejected gate treatment as
negative evidence. The final directory preserves the later rolled-back
deferred-submission treatment. Current source uses immediate per-peer
asynchronous submission.
