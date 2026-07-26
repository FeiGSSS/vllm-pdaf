# PAP collective QKV fan-out

> Controlled 7PA1P development validation. Raw AIPerf and trace artifacts
> remain machine-local under `experiments/_staging/`.

Date: 2026-07-27

## Decision

Accept the collective fan-out organization, but do not use a common GPU start
gate.

Projection now finishes constructing the complete set of per-PA QKV shards
before it submits that set. Each local-fast transport continues to enqueue its
copy on an independent peer CUDA stream. There is no cross-peer stream
dependency, so the GPU transfers may overlap while the existing per-peer
buffer, doorbell, and backpressure semantics remain unchanged.

The first implementation added a common GPU release gate after all host-side
submissions. That treatment was rejected and removed before commit: it made
the returns more tightly grouped but delayed the barrier completion and
reduced end-to-end throughput.

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
| Final independent-stream collective | 1 | 1,651.09 ms | 48.47 ms | 5.220 req/s |

The common gate regressed mean ITL by 1.7% and throughput by 3.1% relative to
the prior two-run baseline. The final implementation restored throughput
within 0.1% of the baseline and produced a 2.7% lower mean ITL in its
validation run. Treat the latter as no-regression evidence, not a stable
performance gain, because the final treatment has one repetition.

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

## Validation

```text
.venv/bin/python -m pytest \
  tests/pap/test_pap_local_fast_transport.py \
  tests/pap/test_pap_qwen3_async_send.py \
  tests/pap/test_pap_trace_summary.py -q

19 passed, 2 skipped
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
```

The first three run directories preserve the rejected gate treatment as
negative evidence. Only the final run corresponds to the committed source.
