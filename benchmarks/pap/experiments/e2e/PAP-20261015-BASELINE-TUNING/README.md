# E2a: single-knob tuning of the PD 6P2D and DP8 baselines

Request: `paper/handoff/requests/R-002-phase1-e2e-e2a-e2b.md` (E2a).

## Question

Are the paper's Dynamo PD 6P2D and DP8 baselines already tuned, or does one
knob change improve D2 goodput or throughput beyond its CI?

## Protocol

- `dynamo.env` is a byte copy of `PAP-20260907-ALIGNED-C30/results/common.env`
  (the baseline). `s60-t3-native32k-stratified-seed42`, closed-loop C30, one
  service start per run.
- One knob per variant:

| Variant | Architecture | Change |
| --- | --- | --- |
| `6p2d-base` | 6P2D | none |
| `6p2d-pbt4096`, `6p2d-pbt8192` | 6P2D | `DYNAMO_PREFILL_MAX_NUM_BATCHED_TOKENS` 4096, 8192 (base 2048) |
| `6p2d-ls1`, `6p2d-ls4` | 6P2D | `DYNAMO_ROUTER_PREFILL_LOAD_SCALE` 1.0, 4.0 (base 2.0) |
| `dp8-base` | DP8 | none |
| `dp8-abt8192` | DP8 | `DYNAMO_AGG_MAX_NUM_BATCHED_TOKENS` 8192 (base 2048) |

- `run_all.sh` runs all seven variants in one round, three rounds (21 runs),
  so every variant is interleaved with its baseline in time.
- Every 6P2D run is followed by `analyze_dynamo_ttft.py` (TTFT breakdown).

## Criterion (from R-002)

A setting whose D2 goodput or throughput beats its baseline with a CI that
excludes zero replaces the baseline in all later experiments.

## Results (2026-10-15)

21 valid runs, 3 per variant. Three runs failed before serving because the
previous service still held GPU 0 when the next one started
(`dp8-abt8192_r1` at 09:26Z, `dp8-base_r2` at 09:39Z, `dp8-base_r3` at
09:55Z, runner log `ERROR: GPU 0 is occupied`). They are kept, excluded, and
were rerun after waiting for idle GPUs. `run_all.sh` has no such wait; later
experiments' `run.sh` waits for an empty compute-app list before starting.

| Variant | TTFT mean / P99 (ms) | ITL mean (ms) | req/s | D2 attainment | D2 goodput (req/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 6p2d-base | 5707 / 10719 | 49.46 | 2.703 | 96.3% | 2.603 |
| 6p2d-pbt4096 | 5985 / 11404 | 50.37 | 2.630 | 93.5% | 2.460 |
| 6p2d-pbt8192 | 5420 / 10529 | 56.49 | 2.622 | 92.8% | 2.432 |
| 6p2d-ls1 | 5897 / 10856 | 49.56 | 2.669 | 94.6% | 2.526 |
| 6p2d-ls4 | 5788 / 11179 | 49.37 | 2.707 | 94.3% | 2.552 |
| dp8-base | 2228 / 8103 | 77.53 | 2.992 | 48.7% | 1.456 |
| dp8-abt8192 | 2215 / 7509 | 74.63 | 3.018 | 50.6% | 1.527 |

Change versus the same architecture's baseline, 95% run-level bootstrap CI
(n = 3 vs 3):

| Variant | Throughput | D2 goodput |
| --- | --- | --- |
| 6p2d-pbt4096 | −2.70% [−5.55, +0.69] | **−5.48% [−10.53, −0.32]** |
| 6p2d-pbt8192 | −2.98% [−5.65, +0.56] | **−6.54% [−9.88, −2.84]** |
| 6p2d-ls1 | −1.24% [−3.78, +2.44] | −2.95% [−6.44, +1.03] |
| 6p2d-ls4 | +0.16% [−2.50, +3.51] | −1.94% [−7.17, +3.55] |
| dp8-abt8192 | +0.87% [−2.81, +4.69] | +4.82% [−0.62, +10.87] |

## Conclusion

No setting beats its baseline with a CI that excludes zero, so both baselines
stay unchanged for later experiments. Larger 6P2D Prefill batches lower TTFT
(8192: −5%) but raise ITL (+14%) and lose D2 goodput. DP8's D2 attainment is
about 50% at C30 because its mean ITL (≈77 ms) sits at the 75 ms D2 limit;
its D2 goodput is therefore far below its raw throughput.
