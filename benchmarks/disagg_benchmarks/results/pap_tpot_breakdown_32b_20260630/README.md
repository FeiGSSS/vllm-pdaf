# PAP 32B TPOT Breakdown - 2026-06-30

## Goal

Break down the 4s-level PAP TPOT observed on Qwen3-32B TP=2 and identify the
dominant root cause.

## Baseline: ctx32, output16, batch512

Source:
`../pap_after_nccl_removal_32b_20260630/1pa1p_tp2_nixl_ctx32_bs512`

- Model: Qwen3-32B
- Topology: 1PA1P
- TP size: 2
- Transport: NIXL mailbox
- Prompts: 512
- Input length: 32
- Output length: 16
- Completed: 512/512
- Output throughput: 66.76 tok/s
- Median TTFT: 17012.18 ms
- Median TPOT: 4446.34 ms

Trace accounting, using include-outliers summary:

- Projection trace median calls per layer batch: 94
- Projection trace mean calls per layer batch: 91.02
- Projection remote-attention total median: 17.336 ms
- Projection remote-attention total mean: 20.798 ms
- Attention compute median: 4.112 ms
- Attention total median: 5.850 ms
- Projection send median: 2.678 ms
- Projection recv median: 13.793 ms
- Projection/Attention correlation median:
  - attention path after projection send: 13.632 ms
  - projection resume to recv done: 14.291 ms

Wave estimate:

```text
full-batch waves ~= 512 / 94 = 5.45
estimated TPOT ~= 64 layers * 17.336 ms * 5.45 = 6043 ms
```

This overestimates the benchmark median because the projection engine is not at
512 running requests for the whole run. Projection engine logs show nonzero
running request counts such as 282, 334, 459, 510, 230, and 178.

The engine-side ratio is consistent with the benchmark:

```text
median(Running / generation_throughput) ~= 4.86 s/token
benchmark median TPOT ~= 4.45 s/token
```

## D1-lite: ctx1, output16, batch512

Directory: `d1_ctx1_o16_bs512`

This run was intended to remove prefill cost and isolate decode. It did not
complete.

Failure evidence:

- Attention mailbox loop raised:
  `RuntimeError: prefill KV must be imported before stateful decode attention`
- Projection engine then logged:
  `No available shared memory broadcast block found in 60 seconds`
- Prefill emitted 1024 KV lease expiration warnings:
  `Releasing expired KV blocks ... retrieved by 0 remote worker(s)`

Interpretation:

`input_len=1` exposes a KV-import readiness race. Decode can reach stateful
remote attention before the attention-side prefill KV import is installed. This
is a correctness/readiness issue in the PAP handoff path, so ctx1 is not a valid
TPOT breakdown workload until the KV import gate is fixed.

The same 1024 prefill KV lease expiration warnings also appear in the successful
ctx32 baseline, so the warning itself is not sufficient to predict failure. The
hard failure is the attention-side stateful decode before KV import.

## D2: ctx32, output16, batch128

Directory: `d2_ctx32_o16_bs128`

- Model: Qwen3-32B
- Topology: 1PA1P
- TP size: 2
- Transport: NIXL mailbox
- Prompts: 128
- Input length: 32
- Output length: 16
- Completed: 128/128
- Output throughput: 71.91 tok/s
- Median TTFT: 6742.30 ms
- Median TPOT: 1444.06 ms

Trace accounting, using include-outliers summary:

- Projection trace median calls per layer batch: 42
- Projection trace mean calls per layer batch: 36.57
- Projection remote-attention total median: 7.213 ms
- Projection remote-attention total mean: 7.442 ms
- Attention compute median: 1.264 ms
- Attention total median: 1.887 ms
- Projection send median: 1.785 ms
- Projection recv median: 4.619 ms
- Projection/Attention correlation median:
  - attention path after projection send: 4.701 ms
  - projection resume to recv done: 5.022 ms

Wave estimate:

```text
waves ~= 128 / 42 = 3.05
estimated TPOT ~= 64 layers * 7.213 ms * 3.05 = 1407 ms
benchmark median TPOT ~= 1444 ms
```

This closes almost exactly.

Engine-side ratio also matches:

```text
median(Running / generation_throughput) ~= 1.67 s/token
benchmark median TPOT ~= 1.44 s/token
```

## Root Cause

The 4s TPOT is primarily caused by low Projection-side decode token throughput,
not by client/proxy timing artifacts and not by one isolated slow kernel.

The dominant mechanism is:

```text
TPOT ~= effective_running_requests
        / projection_generation_throughput

projection_generation_throughput is low because:

one output token
  -> multiple Projection decode waves
  -> each wave must traverse 64 layers
  -> each layer has a remote Attention dependency
  -> current layer path is dominated by Projection waiting for Attention result
```

In the 512-concurrency run, the effective running batch is hundreds of requests,
but each per-layer remote-attention batch only carries around 90 calls at the
median. That implies roughly 4-6 waves per output token. Multiplying this by
64 sequential layers and the 17 ms median remote-attention layer path produces
the observed multi-second TPOT.

In the 128-concurrency run, the same accounting predicts the measured TPOT:
3.05 waves * 64 layers * 7.213 ms/layer = 1.407 s, while the measured median
TPOT is 1.444 s.

## Secondary Findings

- The `ctx1` workload exposes a KV import readiness race and should not be used
  as a decode-only benchmark until the attention-side KV readiness gate is fixed.
- Prefill KV lease expiration warnings appear even in successful runs; they are
  useful diagnostics but not alone the root cause of the 4s TPOT.
- PAP trace rows without phase labels make prefill/decode attribution harder.
  The next profiling improvement should add phase/decode-step IDs to Projection
  and Attention trace records.

## Next Optimization Targets

1. Increase per-layer remote-attention batch size, or reduce decode wave count.
2. Fuse/batch attention-side processing so one remote attention call carries
   larger effective work.
3. Add strict attention KV import readiness before Projection decode dispatch.
4. Add phase and decode-step trace fields to make future TPOT accounting direct.
