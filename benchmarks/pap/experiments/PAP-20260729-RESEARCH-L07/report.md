# PAP research loop L07: clean three-way boundary confirmation

Date: 2026-07-29

## Question and decision

L07 repeated the eight preselected PAP, PD, and fused-DP capacity boundaries
on clean committed code. All 16 repetitions completed 640/640 requests,
passed correctness validation, used the same dataset, and restarted the
service between points.

C07 is falsified. Corrected PD exceeds PAP goodput by more than the registered
15% in every SLO tier, but PAP exceeds fused DP only under the strict SLO. The
registered standard-SLO PAP-over-DP margin is absent.

The honest current result is therefore:

- PAP is not the goodput winner over corrected PD on this O16 workload.
- PAP has a strong strict-SLO advantage over fused DP.
- Fused DP overtakes PAP under the standard and relaxed SLOs.
- 7PA1P C34 is repeat-stable only under the relaxed SLO; one of its two
  repetitions fails the standard tier.

## Conservative repeated result

Throughput and goodput use the lower repetition. Tail latency uses the higher
repetition. A point is SLO-eligible only when both repetitions pass.

| SLO | Best PAP | Best PD | Fused DP | PAP vs PD | PAP vs DP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict | 6PA2P C32, 9.445 good req/s | 6P2D C31, 13.932 | DP C8, 5.985 | -32.2% | +57.8% |
| Standard | 6PA2P C32, 9.659 | 6P2D C44, 15.305 | DP C18, 10.994 | -36.9% | -12.1% |
| Relaxed | 7PA1P C34, 11.654 | 6P2D C48, 15.405 | DP C28, 14.083 | -24.4% | -17.2% |

The raw repeated-point boundaries are:

| Architecture | C | Min req/s | Mean req/s | Worst TTFT p95 | Worst ITL p95 | Strict | Standard | Relaxed |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| PAP 6PA2P | 32 | 9.781 | 9.878 | 4106.33 ms | 33.30 ms | pass | pass | pass |
| PAP 7PA1P | 34 | 12.010 | 12.282 | 3409.25 ms | 82.03 ms | fail | fail | pass |
| PD 6P2D | 31 | 14.153 | 14.412 | 2630.58 ms | 40.60 ms | pass | pass | pass |
| PD 6P2D | 44 | 15.722 | 16.567 | 6713.55 ms | 54.34 ms | fail | pass | pass |
| PD 6P2D | 48 | 16.031 | 16.195 | 7901.00 ms | 82.86 ms | fail | fail | pass |
| Fused DP | 8 | 6.119 | 6.753 | 693.57 ms | 23.91 ms | pass | pass | pass |
| Fused DP | 18 | 11.459 | 13.185 | 898.59 ms | 69.56 ms | fail | pass | pass |
| Fused DP | 28 | 14.703 | 18.499 | 1208.96 ms | 93.62 ms | fail | fail | pass |

## Provenance

- Runtime commit: `163c2dfa1`.
- Tracked worktree: clean for every repetition.
- Model: Qwen3-8B, FP16, eager.
- Hardware: eight NVIDIA L20 GPUs.
- Dataset: 128 conversations, five turns, 640 requests.
- Dataset SHA-256:
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.
- Output distribution: mean 16, median 15, range 8--32 tokens.
- Initial input distribution: mean 4096, median 4000, range 2048--5632
  requested tokens.
- Append distribution: mean 1100, median 400, range 4--2125 requested tokens;
  sampled mean 697.338.
- Think/tool delays: 1000/300 ms, with a tool event every third turn.
- PD/PAP same-node runtime: UCX 1.22, GET zcopy enabled, protocol emulation
  disabled.
- Repetitions: two, with service restart between points.

Repository-local raw bundle:

```text
benchmarks/pap/experiments/_staging/capacity/
  20260729_l07_three_way_clean_boundaries_r2/
```

Summary regeneration:

```bash
.venv/bin/python benchmarks/pap/aiperf/summarize_capacity_matrix.py \
  benchmarks/pap/experiments/_staging/capacity/\
20260729_l07_three_way_clean_boundaries_r2
```

## Interpretation and limitation

This loop confirms that the unfavorable PD comparison was not an artifact of
the dirty July 28 worktree. It does not prove that these architectures have
been globally optimized for every workload: the points were preselected from
the preceding broad scan rather than searched again in L07.

The result also contradicts earlier PAP-favorable O32 experiments. Several
workload dimensions changed together between those studies, including output
length, prompt length, and think/tool delays. The next loop must isolate one
dimension before changing the scheduler or transport.

## Next loop

L08 changes only the output-length distribution from O16 to O32 while
preserving input text, session order, and think/tool delays. It compares fixed
PAP 7PA1P C34 and PD 6P2D C48 points as a diagnostic, not as a new tuned
capacity claim.
