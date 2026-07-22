# PAP versus PD: piecewise CUDA Graph comparison

## Scope

This four-GPU development scan compares PAP and PD with vLLM piecewise CUDA
Graphs after the source-audited eager capacity baseline.

- Piecewise CUDA Graph implementation: `265018942`
- Projection capture fix and experiment commit: `d6d590bb5`
- Tracked worktree at launch: clean
- AIPerf: 0.11.0
- Model and hardware: Qwen3-8B FP16 on four NVIDIA L20 GPUs
- Work per point: 32 conversations, ten turns, 320 requests
- Timing: conversation concurrency with delays
  `0,3,3,1,3,3,1,3,3,1` seconds
- Length seed: 42; output mean/median/range: 32/30/16-64 tokens
- Capacity settings: `max_num_seqs=64`, Prefill
  `max_num_batched_tokens=16384`, Decode/Projection
  `max_num_batched_tokens=64`, and default
  `max_num_partial_prefills=1`
- Execution mode: piecewise for both PAP and PD

The eager and Graph dataset files have different hashes because their
matrix-specific session IDs and cache salts differ. After normalizing only
those two identity fields, every turn's text, requested output length, delay,
and generation setting is identical. The scheduler and memory settings are
the same as the
[audited eager baseline](../PAP-20260721-AIPERF-AUDITED-CAPACITY/report.md).

This is one repetition at six informative matched points, not a release-level
three-repetition claim. The known-bad PD 1P3D transfer-overload point and
higher already-failing eager points were not repeated.

## CUDA Graph boundary and validity

PAP captures graph-safe model compute while keeping remote Attention,
OFFLOAD_EXEC transport, and Prefill KV publication behind graph-unsafe opaque
boundaries. Prefill captures scheduled-token sizes
`1,2,4,8,16,32,64,128`; Projection and PD Decode capture
`1,2,4,8,12,16,20,24,28,32`. These sizes select replay or fallback and do not
limit request admission.

All six points completed 320 of 320 requests, passed routing and output
validation, and reported zero conversation migrations and zero client errors.
The matrix therefore contains 1,920 valid request records.

The capture audit found `Graph capturing finished` in all 24 model-service
logs. Every PAP Projection captured ten sizes after explicitly skipping only
its inapplicable local-Attention kernel warmup; all Prefill services captured
eight sizes. Nine PAP Prefill logs also contain one empty-subgraph warning,
consistent with the deliberately graph-excluded KV publisher split boundary;
the same workers completed their graph-safe captures. Two API logs contain
`EngineDeadError` only after the launcher initiated SIGTERM cleanup, after all
requests had completed, so they are shutdown noise rather than run failures.

An earlier C8 diagnostic at `265018942` completed correctly but Projection
returned before `capture_model()`. It is preserved locally but excluded from
all tables and conclusions.

## Piecewise results

| Architecture | Topology | C | TTFT p95 ms | ITL p95 ms | Req/s | Strict | Standard | Relaxed |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| PAP | 3PA1P | 8 | 2,106.09 | 35.47 | 2.024 | pass | pass | pass |
| PAP | 3PA1P | 12 | 2,803.84 | 41.06 | 2.556 | pass | pass | pass |
| PAP | 3PA1P | 20 | 11,230.53 | 52.40 | 2.876 | fail | fail | pass |
| PD | 2P2D | 10 | 8,475.24 | 32.63 | 1.833 | fail | pass | pass |
| PD | 2P2D | 16 | 13,920.06 | 34.61 | 2.658 | fail | fail | pass |
| PD | 3P1D | 8 | 4,224.66 | 33.27 | 1.904 | pass | pass | pass |

The tested concurrency envelope is:

| SLO | PAP 3PA1P | Best PD | PAP difference |
| --- | ---: | ---: | ---: |
| Strict | C12 | C8, 3P1D | +4 / +50% |
| Standard | C12 | C10, 2P2D | +2 / +20% |
| Relaxed | C20 | C16, 2P2D | +4 / +25% |

Only complete and correct configurations with at least 95% good requests are
eligible for goodput:

| SLO | PAP best | PD best | PAP versus PD |
| --- | ---: | ---: | ---: |
| Strict | 2.460 req/s, C12 | 1.833 req/s, 3P1D C8 | +34.2% |
| Standard | 2.556 req/s, C12 | 1.904 req/s, 3P1D C8 | +34.2% |
| Relaxed | 2.777 req/s, C20 | 2.591 req/s, 2P2D C16 | +7.2% |

## Matched eager comparison

Negative latency deltas and positive throughput deltas are improvements.

| Configuration | TTFT eager → Graph | Delta | ITL eager → Graph | Delta | Req/s eager → Graph | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PAP 3PA1P C8 | 2,024.84 → 2,106.09 | +4.0% | 37.21 → 35.47 | -4.7% | 1.994 → 2.024 | +1.5% |
| PAP 3PA1P C12 | 3,552.38 → 2,803.84 | -21.1% | 41.32 → 41.06 | -0.6% | 2.492 → 2.556 | +2.6% |
| PAP 3PA1P C20 | 10,068.56 → 11,230.53 | +11.5% | 49.22 → 52.40 | +6.5% | 2.715 → 2.876 | +5.9% |
| PD 2P2D C10 | 7,431.79 → 8,475.24 | +14.0% | 31.87 → 32.63 | +2.4% | 1.801 → 1.833 | +1.8% |
| PD 2P2D C16 | 11,570.79 → 13,920.06 | +20.3% | 33.67 → 34.61 | +2.8% | 2.813 → 2.658 | -5.5% |
| PD 3P1D C8 | 4,565.87 → 4,224.66 | -7.5% | 34.70 → 33.27 | -4.1% | 1.835 → 1.904 | +3.8% |

CUDA Graph is not a uniform per-point speedup. PAP C12 gains substantially
and moves from standard-only to strict-compliant, while PAP C20 exchanges
higher throughput for worse tail latency. PD is similarly mixed. The
architecture conclusion should therefore use SLO-qualified goodput, not a
claim that Graph always lowers latency.

## Conclusion

With both architectures using piecewise CUDA Graphs, PAP retains the larger
tested concurrency envelope and leads the best PD configuration in all three
goodput tiers: +34.2% strict, +34.2% standard, and +7.2% relaxed. Compared with
the eager baseline, the previous -1.3% relaxed-goodput gap becomes a measured
+7.2% lead in this single repetition.

This establishes a correct development baseline for PAP piecewise CUDA Graph,
but the mixed matched-point deltas mean a later three-repetition run is needed
before treating the exact percentages as a release claim.

The complete 47 MiB valid evidence bundle is preserved locally under
[`20260721_d6d590bb5_aiperf_piecewise_o32_s32`](runs/20260721_d6d590bb5_aiperf_piecewise_o32_s32/raw/capacity_results.md).
The excluded 13 MiB diagnostic is under
[`20260721_265018942_projection_capture_skipped_diagnostic`](runs/20260721_265018942_projection_capture_skipped_diagnostic/raw/capacity_results.md).
