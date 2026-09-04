# PAP expanded Attention latency matrix

Experiment ID: `PAP-20260824-ATTENTION-LATENCY-SURFACE`.

This experiment evaluates the current PAP implementation. It is not a vLLM
v0.26 porting checkpoint.

## Scope

This checkpoint contains direct measurements only. No regression, nearest
neighbor model, interpolation, or other latency fitting was performed.

The run is diagnostic because the source tree was dirty. Every shard records
the source commit and dirty state in its own `run.env`; the merged artifact
retains the source-shard paths and hashes.

## Matrix

- Hardware: eight NVIDIA L20 GPUs, each exposed as an independent 12-SM MPS
  partition.
- Model shape: Qwen3-8B FP16, 32 query heads, 8 KV heads, head dimension 128,
  and KV block size 16.
- Unique workloads: 374.
- Candidate configurations per workload: 19.
- Total measurements: 7,106.
- Completed measurements: 7,106.
- Numerical comparisons passed: 7,106.
- Largest absolute output error: `3.814697265625e-6`.

The 374 workloads cover:

- request counts 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, and 64;
- mean per-request contexts 512, 1K, 2K, 4K, 8K, 12K, 16K, 24K, and 32K,
  subject to a 256K total-context limit;
- 86 equal, 72 bimodal, 72 one-heavy, 72 Zipf-0.8, and 72 Zipf-1.2 vectors;
- actual total contexts from 512 through 262,144 tokens;
- actual per-request contexts from 176 through 32,768 tokens.

Invalid or duplicate vectors caused by the total and per-request capacity
bounds are removed before sharding.

The practical candidate set retains production auto, 12 PAP grouped-query
configurations, and six vLLM split configurations. The previous 31-candidate
scan showed that `block_h=1/2` was consistently dominated, so the expanded
matrix does not repeat those configurations at every workload.

## Production-auto coverage

Production auto is the fixed `s8/h4/w4/g1/n32` low-SM specialization. Its
direct gap from the best of the 19 measured candidates is:

| Bound | Workloads within bound | Fraction |
| ---: | ---: | ---: |
| 1% | 68 / 374 | 18.18% |
| 2% | 115 / 374 | 30.75% |
| 3% | 150 / 374 | 40.11% |
| 5% | 239 / 374 | 63.90% |
| 10% | 323 / 374 | 86.36% |
| 15% | 365 / 374 | 97.59% |

- mean gap: 4.92%;
- median gap: 4.05%;
- 90th-percentile gap: 11.05%;
- 95th-percentile gap: 13.58%;
- maximum gap: 32.50%.

The expanded matrix therefore rejects the earlier, sparse-matrix implication
that production auto is always within 15%. It remains a strong central
default, but the fixed eight-way split over-partitions some many-request,
short-context batches.

The largest direct production gaps are:

| Requests | Total | Distribution | Longest request | Best config | Best (ms) | Auto (ms) | Gap |
| ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |
| 48 | 24K | bimodal | 704 | `s2/h4/w4/g1/n32` | 0.1891 | 0.2505 | 32.50% |
| 64 | 32K | bimodal | 704 | `s1/h4/w4/g1/n32` | 0.2644 | 0.3408 | 28.89% |
| 32 | 16K | bimodal | 704 | `s2/h4/w4/g1/n32` | 0.1277 | 0.1618 | 26.71% |
| 24 | 12K | bimodal | 704 | `s2/h4/w4/g1/n32` | 0.0985 | 0.1231 | 25.04% |

The most frequent measured winner is `s8/h4/w4/g2/n32` at 136 workloads.
Split 32 wins 58, split 4 wins 50, split 16 wins 42, and split 2 wins 27.
Production auto and its explicitly forced equivalent have separate repeated
timings, so their nominally identical wins are split between two IDs.

## Distribution effect

At fixed request count and total context, one-heavy distributions can be much
slower than equal lengths even after selecting the best measured kernel. The
largest observed penalties are 52.44% for 48 requests totaling 24K and 52.26%
for 64 requests totaling 32K. These are direct measurements, not fitted
values. KV-block rounding remains recorded in the flat table so that its
contribution can be separated later.

## Cross-GPU calibration

After the matrix, all eight GPUs repeated the same four production-auto
anchors. The maximum card-to-card latency span was:

| Anchor | Latency range (ms) | Maximum span |
| --- | ---: | ---: |
| B1, total 1K | 0.02464–0.02519 | 2.25% |
| B4, total 128K | 1.00686–1.00968 | 0.28% |
| B16, total 128K | 0.96629–0.96823 | 0.20% |
| B64, total 128K | 1.01140–1.01433 | 0.29% |

The short anchor is dominated by the approximately 25-microsecond launch
floor; its absolute card-to-card span is only 0.00055 ms. For substantive
Attention workloads, the measured cross-GPU span is below 0.3%.

## Artifacts

Output root:
`/tmp/pap-attention-latency-surface-v026-expanded-374`

- `timing/result.json`: merged 7,106-row raw matrix and sample arrays;
- `shards/shard_*/timing/result.json`: raw per-GPU measurements;
- `calibration/gpu_*/timing/result.json`: repeated cross-GPU anchors;
- `tables/latency_table.csv`: flat matrix;
- `tables/best_by_shape.csv`: measured winner for each workload;
- `tables/distribution_comparison.csv`: fixed-count/fixed-total distributions;
- `tables/config_summary.csv`: direct per-configuration gap summary;
- `tables/matrix_summary.json`: compact direct-measurement summary;
- `tables/artifact_manifest.json`: merged-source and table hashes.
