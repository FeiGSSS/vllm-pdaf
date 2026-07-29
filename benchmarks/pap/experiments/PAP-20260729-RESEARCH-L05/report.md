# PAP research loop L05: PA resource repartitioning

Date: 2026-07-29

## Question and decision

L05 tested whether the fixed 18/5-chunk PA partition was responsible for the
remaining 7PA1P ITL penalty at the matched C04 operating point. The treatment
moved two chunks from Prefill to Attention, changing every PA from 72/20 to
64/28 visible SMs.

C05 is falsified. Relative to the repeated 18/5 baseline, 16/7 changes mean
ITL by +0.45%, reduces request throughput by 4.98%, increases mean TTFT by
18.2%, and reduces standard/relaxed goodput by 5.06%/4.98%. More visible
Attention SMs do not improve the end-to-end token cycle at this point, while
the lost Prefill resources enter the critical path.

Keep 18/5 as the canonical allocation for this workload. Do not build an
adaptive PA resource partitioner from the C04 ITL gap.

## Identity

- Model: Qwen3-8B, FP16, eager.
- Hardware: eight NVIDIA L20 GPUs.
- Topology and load: 7PA1P C21; 128 conversations, five turns, 640 requests
  per repetition.
- Treatment: 16 Prefill / 7 Attention MPS chunks, 64/28 visible SMs.
- Baseline: L04 18/5 chunks, 72/20 visible SMs.
- Repetitions: two per allocation.
- Dataset SHA-256:
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.
- Treatment runtime and benchmark code: clean commit `513350e18`.
- Config-aware summary code: commit `57926aac2`.
- Tracing: disabled.
- Routing: conversation affinity; zero migrations.

Machine-local treatment artifacts:

```text
benchmarks/pap/experiments/_staging/capacity/
  20260729_l05_7pa1p_c21_mps16_7_r2/
```

## Correctness and treatment integrity

Both repetitions completed 640/640 requests. Correctness, routing,
decode-token join, Projection scheduling, lease release, and session drain
audits pass.

Each of the seven PA audit files records:

```text
PREFILL_CHUNKS=16
ATTENTION_CHUNKS=7
PREFILL_VISIBLE_SMS=64
ATTENTION_VISIBLE_SMS=28
```

The MPS server reports all 23 chunks assigned. The result is therefore a real
resource treatment, not a configuration-only change.

The initial summary marked these audits invalid because it hard-coded the
canonical 72/20 values. Commit `57926aac2` validates audits against the
fail-closed `effective_config.env` allocation instead. It also checks that all
23 L20 chunks are assigned and each chunk exposes four SMs. Resummarization
then validates both repetitions without rerunning the service.

## Result

| Metric | 16/7 treatment | 18/5 baseline | Treatment change |
| --- | ---: | ---: | ---: |
| Request throughput | 9.232 req/s | 9.716 req/s | -4.98% |
| TTFT average | 737.2 ms | 623.9 ms | +18.2% |
| ITL average | 31.48 ms | 31.34 ms | +0.45% |
| Strict good fraction | 94.30% | 95.94% | -1.64 pp |
| Standard good fraction | 98.13% | 98.20% | -0.08 pp |
| Relaxed good fraction | 99.30% | 99.30% | equal |
| Strict goodput | 8.706 req/s | 9.322 req/s | -6.61% |
| Standard goodput | 9.059 req/s | 9.542 req/s | -5.06% |
| Relaxed goodput | 9.168 req/s | 9.648 req/s | -4.98% |

Both treatment repetitions pass standard and relaxed SLOs and fail strict.
The average good-request fractions remain similar to the baseline, so the
goodput loss follows the lower service throughput rather than a new tail
failure.

Against the reused 6PA2P C32 control, the treatment still has 55.0% lower
mean TTFT and 9.82% higher mean ITL, but request throughput is 4.42% lower and
standard/relaxed goodput is 5.25%/4.79% lower. It fails the registered C05
throughput and goodput conditions as well as the required 5% ITL improvement.

## Interpretation

The L02 remote-Attention span was the largest measured forward component, but
that did not imply that more visible SMs would improve it. The previous paged
Attention probes already showed that memory-bandwidth utilization depends on
kernel grid and occupancy, not only the MPS SM count. At this C21 shape, the
additional eight Attention SMs do not shorten request-level ITL, while taking
eight SMs from each Prefill process reduces the cadence at which sessions
enter and leave Decode.

The benchmark override remains useful for controlled sensitivity experiments.
Its production default remains 18/5 and every non-default allocation is
recorded and audited.

## Decision and next loop

Reject C05 and stop the adaptive-MPS direction for this workload. L06 will
audit the current corrected-NIXL PAP/PD/DP capacity evidence. This is necessary
because the reduced O16 workload reverses older PAP-over-PD results: before
new implementation work, the paper must state exactly which current
advantages remain and which claims were artifacts of an outdated PD baseline
or workload.

