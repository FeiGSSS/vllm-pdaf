# PAP Attention latency matrix pilot

Experiment ID: `PAP-20260824-ATTENTION-LATENCY-SURFACE`.

This pilot evaluates the current PAP implementation. It is not a vLLM v0.26
porting checkpoint.

## Scope

This checkpoint contains direct GPU measurements only. It does not fit,
interpolate, or evaluate a latency model.

The current run is diagnostic rather than paper-ready because its source tree
was dirty. The complete provenance, raw matrix hash, and table hashes are in
`artifact_manifest.json` under the output directory.

## Protocol

- Hardware: one NVIDIA L20 exposed as a 12-SM MPS partition.
- Model shape: Qwen3-8B FP16, 32 query heads, 8 KV heads, head dimension 128,
  and KV block size 16.
- Workloads: 68 distinct per-request context vectors after deduplication.
- Request count: 1, 2, 4, 8, 16, 32, or 64.
- Total context: 512 through 262,144 tokens.
- Per-request context observed in this matrix: 437 through 32,768 tokens
  (the generator enforces a 128-token lower bound).
- Distributions: equal, bimodal, one-heavy, and Zipf-like.
- Fixed-total slices: 32,768, 65,536, 131,072, and 262,144 tokens.
- Candidates: 31 measured configurations:
    - current production auto selection;
    - 24 PAP grouped-query configurations covering splits 1/2/4/8/16/32,
    head blocks 1/2/4, and controlled one-factor scans of warps 2/4/8,
    stages 1/2/3, and token blocks 16/32/64;
    - vLLM paged-decode with splits 1/2/4/8/16/32.
- Timing: 20 warmups followed by seven CUDA-event samples, each averaging 50
  kernel invocations.
- Correctness: each candidate is compared with production auto using FP16
  `rtol=atol=0.02`.

The measured matrix has 68 × 31 = 2,108 rows. All 2,108 rows completed and all
2,108 passed the numerical comparison.

## Direct observations

### Fixed total context, equal request lengths

| Total context | Fastest request count | Fastest (ms) | Slowest request count | Slowest (ms) | Span |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 8 | 0.2479 | 1 | 0.2633 | 6.21% |
| 65,536 | 8 | 0.4769 | 64 | 0.4992 | 4.67% |
| 131,072 | 8 | 0.9331 | 64 | 0.9742 | 4.41% |
| 262,144 | 16 | 1.8324 | 64 | 1.9155 | 4.54% |

Total context is therefore not a complete lookup key: even for equal request
lengths, request count changes the best measured latency by roughly 4.4% to
6.2% on these fixed-total slices.

### Fixed request count and fixed total context

The largest measured distribution penalty is the 32-request, 65,536-token
slice: one-heavy is 0.6465 ms versus 0.4967 ms for equal lengths, or 30.16%
higher. Other representative slices are:

| Requests | Total context | Equal (ms) | Worst distribution | Worst (ms) | Penalty |
| ---: | ---: | ---: | --- | ---: | ---: |
| 4 | 8,192 | 0.0649 | Zipf-like | 0.0750 | 15.60% |
| 8 | 16,384 | 0.1214 | One-heavy | 0.1475 | 21.47% |
| 16 | 32,768 | 0.2535 | One-heavy | 0.3215 | 26.80% |
| 32 | 65,536 | 0.4967 | One-heavy | 0.6465 | 30.16% |

These are direct paged-cache measurements. A distribution can also change KV
block rounding, which is why the tables retain both logical context and
allocated context instead of attributing the entire difference to one cause.

### Configuration scan

No single configuration wins every workload. The most frequent winner is
`pap_s8_h4_w4_g2_n32` at 36 of 68 shapes. Split 32 configurations dominate
several long-per-request shapes, while split 1 or 2 appears on high-request,
short-per-request shapes.

The current production auto selection has a 3.54% median measured gap to the
best scanned candidate, a 10.94% P90 gap, and a 14.74% maximum gap. Across the
68 shapes, the best vLLM candidate has 6.18% to 23.25% higher latency than the
best PAP grouped candidate, with a 19.23% median difference.

These figures summarize only the configurations that were actually scanned.
They are not a claim that the matrix contains the global optimum.

## Artifacts

Output root:
`/tmp/pap-attention-latency-surface-v026-bounded`

- `timing/result.json`: all 2,108 raw measurements and sample arrays.
- `tables/latency_table.csv`: flat 2,108-row matrix.
- `tables/best_by_shape.csv`: one measured optimum per workload.
- `tables/fixed_total_equal.csv`: request-count comparison at fixed totals.
- `tables/distribution_comparison.csv`: distribution comparison at fixed
  request count and total context.
- `tables/config_summary.csv`: per-configuration measured regret summary.
- `tables/parameter_sensitivity.csv`: controlled one-factor scan summaries.
- `tables/matrix_summary.json`: compact machine-readable findings.
- `tables/artifact_manifest.json`: source and table hashes plus run provenance.

Regenerate the tables without fitting:

```bash
.venv/bin/python benchmarks/pap/tooling/attention_latency_table.py \
  /tmp/pap-attention-latency-surface-v026-bounded/timing/result.json \
  /tmp/pap-attention-latency-surface-v026-bounded/tables
```

The next clean-commit run should use the same workload and candidate sets
before any fitting work begins.
