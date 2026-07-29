# PAP research loop L03: iso-throughput topology comparison

Date: 2026-07-29

## Question and decision

L03 tested whether 7PA1P C20 offers comparable ITL and SLO goodput to
6PA2P C32 after controlling the achieved request throughput.

The two topologies are within 2.55% in mean request throughput. Relative to
6PA2P, 7PA1P reduces mean TTFT by 60.5% and increases mean ITL by 6.64%.
Those outcomes satisfy the registered throughput, TTFT, and ITL thresholds.
However, 7PA1P has 2.78% lower standard goodput and 2.70% lower relaxed
goodput. C03 is therefore falsified as written.

The goodput difference does not reveal a new SLO-tail failure: both topologies
pass all three tiers in both repetitions, and their mean good-request
fractions differ by at most 0.31 percentage points. The difference primarily
reflects the remaining 2.55% throughput mismatch. L04 will localize the
7PA1P operating point at C21 before making a topology-frontier claim.

## Identity

- Model: Qwen3-8B, FP16, eager.
- Hardware: eight NVIDIA L20 GPUs.
- Topologies: 7PA1P C20 and 6PA2P C32.
- Load: 128 conversations, five turns, 640 requests per repetition.
- Repetitions: two per topology without restarting between repetitions.
- Dataset SHA-256:
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.
- Runtime code: clean commit `642cd0f18`.
- Repetition-aware summary code: commit `15b5b90af`.
- Tracing: disabled.
- Routing: conversation affinity; zero migrations.

Machine-local artifacts:

```text
benchmarks/pap/experiments/_staging/capacity/
  20260729_l03_iso_throughput_r2/
```

The matrix configuration, immutable dataset and manifest, service logs,
correctness audits, AIPerf profiles, and four compact capacity summaries are
preserved in that directory.

## Correctness

Every repetition completed 640/640 requests. The shared two-repetition service
runs each completed 1,280 requests and passed correctness, routing,
decode-token join, Projection scheduling, static-MPS, session-drain, and lease
release audits.

7PA1P distributed 256 conversations across its seven PA nodes as
37/37/37/37/36/36/36. 6PA2P distributed them as 43/43/43/43/42/42 and routed
640 requests to each Projection process. Neither topology migrated KV state.

## Per-repetition result

| Topology | Rep | Throughput | TTFT avg | ITL avg | Strict goodput | Standard goodput | Relaxed goodput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7PA1P C20 | 1 | 9.464 req/s | 642.5 ms | 30.13 ms | 9.302 | 9.361 | 9.420 |
| 7PA1P C20 | 2 | 9.362 req/s | 651.5 ms | 31.00 ms | 8.996 | 9.230 | 9.318 |
| 6PA2P C32 | 1 | 9.511 req/s | 1690.3 ms | 28.88 ms | 9.214 | 9.392 | 9.452 |
| 6PA2P C32 | 2 | 9.807 req/s | 1584.0 ms | 28.45 ms | 9.623 | 9.730 | 9.807 |

All twelve topology/repetition/SLO outcomes pass the required 95%
good-request fraction.

## Mean comparison

| Metric | 7PA1P C20 | 6PA2P C32 | 7PA1P relative |
| --- | ---: | ---: | ---: |
| Request throughput | 9.413 req/s | 9.659 req/s | -2.55% |
| TTFT average | 647.0 ms | 1637.2 ms | -60.5% |
| ITL average | 30.57 ms | 28.67 ms | +6.64% |
| Strict good fraction | 97.19% | 97.50% | -0.31 pp |
| Standard good fraction | 98.75% | 98.98% | -0.23 pp |
| Relaxed good fraction | 99.53% | 99.69% | -0.16 pp |
| Strict goodput | 9.149 req/s | 9.418 req/s | -2.86% |
| Standard goodput | 9.295 req/s | 9.561 req/s | -2.78% |
| Relaxed goodput | 9.369 req/s | 9.629 req/s | -2.70% |

The result replaces the misleading fixed-C32 interpretation. Increasing the
PA-to-Projection ratio moves the system to a different batching point:
7PA1P can trade a small ITL increase for much shorter queueing and TTFT. It
does not yet establish higher capacity or goodput at matched throughput.

## Infrastructure correction

AIPerf places repeated profiles under
`profile_runs/run_0001`, `run_0002`, and so on. The original PAP capacity
summarizer only inspected the single-run root and therefore marked this valid
matrix incomplete. Commit `15b5b90af` makes the matrix runner summarize every
repetition independently while validating shared service audits against the
combined request count. A focused unit test covers this behavior.

The correction only reads and summarizes existing artifacts; it does not
change the serving runtime or measured profiles.

## Decision and next loop

Do not claim a 7PA1P goodput advantage from C20. The falsification is narrow:
the target throughput interval was too broad for an exact goodput comparison.

L04 will run two clean 7PA1P C21 repetitions on the same dataset. It will
reuse the valid 6PA2P C32 L03 control. If mean achieved throughput is not
within 1.5%, one adjacent concurrency point may be chosen according to the
direction of the mismatch, without changing the SLOs or outcome thresholds.

