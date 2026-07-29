# PAP research loop L11: invalidated boundary refinement

Date: 2026-07-29

## Question and decision

L11 attempted to refine the L10 long-input capacity boundary with PAP
7PA1P C21/C23/C25 and PD 6P2D C22/C24/C26. The loop is invalidated as a
performance comparison because the registered correctness condition failed at
PAP C25. Its performance rows must not be combined into a tuned PAP-versus-PD
claim.

The failure exposed an asynchronous sampled-token ownership bug. Projection
published a GPU output before the Scheduler had accepted it, while the
sequence key could be derived from mutable Scheduler state already advanced by
the next asynchronous step. This produced 161 request errors at C25, including
decode-commit sequence mismatches.

## Diagnostic result and fix

Commit `4a3e36820` preserves the exact GPU-frame sequence key in
`ModelRunnerOutput`, but moves token publication to the Scheduler after output
acceptance. The ModelRunner no longer performs token-delivery network I/O, and
the Scheduler does not synchronously drain the delivery worker.

Two independent full C25 diagnostics on the fix each completed 640/640
requests and passed correctness, decode-token join, routing, and zero-session
drain:

| Run | Req/s | Mean TTFT | Mean ITL | Matched | Mismatch | Duplicate | Pending |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frame-key R1 | 6.432 | 1797.01 ms | 44.75 ms | 9879 | 0 | 0 | 0 |
| frame-key R2 | 6.457 | 1762.22 ms | 40.84 ms | 9879 | 0 | 0 | 0 |

The throughput difference between repeats is 0.39%. These runs are
implementation validation collected from a dirty tracked worktree and are not
paper evidence.

## Partial observations

The valid pre-failure points preserve the earlier direction: PAP C21 passes
only Relaxed at 5.858 good req/s, while PD C22 passes Standard and Relaxed at
7.673 and 7.723 good req/s. That observation is not promoted because L11's
pre-registered experiment failed correctness and spans the superseded runtime.

## Provenance

- Invalid comparison bundle:
  `benchmarks/pap/experiments/_staging/capacity/20260729_l11_longinput_boundary_refine/`
- Diagnostic bundles:
  `20260729_l11_frame_key_c25_r1` and
  `20260729_l11_frame_key_c25_r2`
- Dataset SHA-256:
  `ae2adf59908bfa7bb6b2ac4cc5d122fdd82d07da11d55361ef87c19f495e6ed5`
- Correctness-fix commit: `4a3e36820`
- Targeted verification: 34 tests passed; Ruff passed.

## Successor

L11 does not answer whether PAP's larger distributed KV pool creates a
long-context advantage. L12 replaces the incremental boundary refinement with
a mechanism-directed four-round workload whose context approaches the model
limit and whose concurrency points bracket the capacity predicted from actual
startup KV-token budgets.
