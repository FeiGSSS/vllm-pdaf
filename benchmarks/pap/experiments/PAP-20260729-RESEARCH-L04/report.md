# PAP research loop L04: accurately matched topology frontier

Date: 2026-07-29

## Question and decision

L04 localized a 7PA1P operating point against the valid 6PA2P C32 control
from L03. At C21, 7PA1P is within 0.59% of the control's achieved request
throughput. It reduces mean TTFT by 61.9%, increases mean ITL by 9.33%, has
0.20% lower standard goodput, and has 0.19% higher relaxed goodput.

All pre-registered C04 thresholds pass. This establishes an observed
standard/relaxed frontier point: at the same request rate and goodput, 7PA1P
exchanges substantial Prefill/queueing headroom for a modest Decode-token
latency penalty. It does not establish strict-SLO stability because one
7PA1P repetition has a 94.84% strict-good fraction, just below the 95%
admission threshold.

## Identity

- Model: Qwen3-8B, FP16, eager.
- Hardware: eight NVIDIA L20 GPUs.
- Treatment: 7PA1P C21, two repetitions.
- Control: 6PA2P C32, two valid L03 repetitions.
- Load: 128 conversations, five turns, 640 requests per repetition.
- Dataset SHA-256:
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.
- Treatment runtime code: clean commit `13c786e78`.
- Control runtime code: clean commit `642cd0f18`.
- The commits differ only in research documentation and AIPerf summary code;
  the serving runtime is identical.
- Final isolated-point summary fix: commit `734223485`.
- Tracing: disabled.
- Routing: conversation affinity; zero migrations.

Machine-local treatment artifacts:

```text
benchmarks/pap/experiments/_staging/capacity/
  20260729_l04_7pa1p_c21_r2/
```

The control remains under the L03 artifact directory.

## Correctness

Both treatment repetitions completed 640/640 requests. Their shared service
run completed 1,280 requests and passed correctness, routing, decode-token
join, Projection scheduling, static-MPS, session-drain, and lease release
audits. Conversation placement is balanced 37/37/37/37/36/36/36 across the
seven PA nodes.

## Per-repetition treatment

| Rep | Throughput | TTFT avg | ITL avg | Strict | Standard | Relaxed |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | 9.757 req/s | 625.1 ms | 30.78 ms | 97.03%, pass | 99.22%, pass | 100.00%, pass |
| 2 | 9.674 req/s | 622.8 ms | 31.90 ms | 94.84%, fail | 97.19%, pass | 98.59%, pass |

The second repetition's strict failure is a Decode tail, not a throughput or
correctness failure: ITL p95/p99 are 52.74/103.60 ms while TTFT p99 is only
1.98 seconds.

## Mean comparison

| Metric | 7PA1P C21 | 6PA2P C32 | 7PA1P relative |
| --- | ---: | ---: | ---: |
| Request throughput | 9.716 req/s | 9.659 req/s | +0.59% |
| TTFT average | 623.9 ms | 1637.2 ms | -61.9% |
| ITL average | 31.34 ms | 28.67 ms | +9.33% |
| Strict good fraction | 95.94% | 97.50% | -1.56 pp |
| Standard good fraction | 98.20% | 98.98% | -0.78 pp |
| Relaxed good fraction | 99.30% | 99.69% | -0.39 pp |
| Strict goodput | 9.322 req/s | 9.418 req/s | -1.03% |
| Standard goodput | 9.542 req/s | 9.561 req/s | -0.20% |
| Relaxed goodput | 9.648 req/s | 9.629 req/s | +0.19% |

The result clarifies the topology trade-off rather than selecting a universal
winner. 7PA1P has large TTFT headroom but a small mean-ITL deficit and an
unstable strict tail. This suggests testing whether the fixed 72/20-SM
Prefill/Attention partition is Pareto-suboptimal before adding a new
scheduling mechanism.

## Infrastructure correction

The first automatic summary incorrectly reported this valid repeated run as
incomplete. `run_isolated_architecture_points` called the legacy single-root
summarizer directly, bypassing the repetition-aware path added in
`15b5b90af`. Commit `734223485` routes both new and resumed isolated points
through the common sweep summarizer. Resuming the existing matrix then
generated two correct summaries without rerunning GPU work.

## Decision and next loop

C04 is supported within its pre-registered standard/relaxed conditions and
recorded at `observed` claim maturity. It is not paper-ready: the control was
reused rather than run contemporaneously, and strict stability remains
unresolved.

L05 will test a resource-allocation explanation before designing a complex
scheduler. The 7PA1P C21 treatment has 61.9% TTFT headroom but 9.33% worse
mean ITL. Repartitioning each PA from 18/5 MPS chunks (72/20 SMs) to 16/7
chunks (64/28 SMs) should move resources from Prefill to the measured
Attention bottleneck.

