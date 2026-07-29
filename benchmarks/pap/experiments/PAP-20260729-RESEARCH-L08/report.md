# PAP research loop L08: output-length sensitivity

Date: 2026-07-29

## Question and decision

L08 tests whether output length alone explains the reversal between historical
PAP-favorable O32 results and the current O16 PAP/PD comparison. The derived
dataset preserves every session ID, turn text, role, delay, cache salt, random
seed, and sampling parameter from L07. Only `output_length` and its matching
`min_tokens` field change.

C08 is falsified at its registered threshold. Doubling the output distribution
improves the PAP/PD mean raw request-throughput ratio from 0.758 to 0.820, or
6.15 percentage points. The hypothesis required at least 10 percentage points.
Output length is directionally important, but it does not by itself explain
the historical result reversal.

## Fixed-point result

These are one-repetition diagnostic points, not tuned capacity boundaries.

| Workload | Architecture | C | Req/s | Output tok/s | Mean TTFT | TTFT p95 | Mean ITL | ITL p95 | Standard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| O16 baseline | PAP 7PA1P | 34 | 12.282 | 201.863 | 922.48 ms | 3409.25 ms worst of two | 39.29 ms | 82.03 ms worst of two | fail (1/2) |
| O16 baseline | PD 6P2D | 48 | 16.195 | 266.186 | 1320.38 ms | 7901.00 ms worst of two | 44.88 ms | 82.86 ms worst of two | fail (0/2) |
| O32 treatment | PAP 7PA1P | 34 | 10.664 | 345.518 | 770.43 ms | 3700.60 ms | 36.60 ms | 51.23 ms | pass |
| O32 treatment | PD 6P2D | 48 | 13.006 | 421.430 | 1135.23 ms | 6896.44 ms | 48.42 ms | 64.63 ms | pass |

Relative to O16, request throughput decreases 13.18% for PAP and 19.69% for
PD. PAP therefore degrades more slowly, but still trails PD by 18.0% in raw
request throughput and 16.0% in standard goodput at these fixed points.

## Dataset identity

- O16 SHA-256:
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.
- O32 SHA-256:
  `1d8a6881fc6679aa843ed4ad990f71ad81d06265bde94a2f346cb9ccfa51a68a`.
- O32 sampled output: mean 32.402, median 31, range 16--64 tokens.
- O32 maximum estimated request budget: 11,206 tokens, below the 32,768 model
  limit.
- Programmatic identity check: all 128 sessions and 640 requests have
  identical non-output fields; all 640 output lengths change.

## Provenance

- Runtime commit: `21a4f705e`.
- Tracked worktree: clean.
- Model/hardware: Qwen3-8B FP16 eager on eight NVIDIA L20 GPUs.
- Runtime: corrected same-node UCX 1.22 GET-zcopy path.
- PAP: 7PA1P C34, conversation affinity, static 18/5 MPS split.
- PD: 6P2D C48.
- Service restart between architecture points.

Repository-local raw bundle:

```text
benchmarks/pap/experiments/_staging/capacity/
  20260729_l08_output32_fixeddiag/
```

## Interpretation and limitation

Longer Decode residence moves the relative result toward PAP, consistent with
PAP exposing seven PA-side Attention engines rather than two Decode workers.
The movement is not large enough to establish the registered causal claim,
and the one-repetition treatment cannot support a paper-ready performance
statement.

The next controlled variable is input and append length. Longer contexts
increase both Prefill work and the bytes read by Decode Attention. L09 keeps
the O16 output distribution and delay schedule fixed while doubling only
those input distributions.
