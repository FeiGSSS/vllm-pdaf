# A2: contribution of cross-PA Projection aggregation

Request: `paper/handoff/requests/R-004-phase3-ablations-tuning.md` (A2).

## Question

Projection is stateless, so one Projection GPU batches the decode steps of all
PAs. How much do ITL and throughput depend on that aggregation?

## Protocol

- PAP, KV migration on, fixed PAT, `s240-t3-native32k-stratified-seed42`
  (720 requests), closed-loop C30 and C50 (`pap.env`).
- Variants per concurrency: `base` (7PA1P, `PAP_PROJECTION_MAX_NUM_SEQS=256`),
  `seqs64`, `seqs32` (Projection batch cap), `6pa2p` (GPUs 0–5 PA, 6–7
  Projection), `4pa4p` (0–3 PA, 4–7 Projection).
- `run_round.sh <k>` runs all ten points once; two rounds (20 runs).
- One traced 7PA1P C30 run (`run_trace.sh`, variant `trace-c30`):
  `PAP_PROJECTION_PA_TRACE_OUTPUT` plus `capture_aligned_trace.py` freezing an
  aligned PA/Projection window while serving, for Projection step time
  versus batch size. Not used for the throughput comparison.

## Results (2026-10-16)

15 valid runs: five variants × 2 at C30 (round 1–2), plus `seqs16` and
`seqs8` × 2 added after X-004, plus one traced run. C50 was dropped (X-003).

| Variant (C30) | TTFT mean (ms) | ITL mean (ms) | req/s | D2 attainment |
| --- | ---: | ---: | ---: | ---: |
| base (7PA1P, cap 256) | 3817 | 49.38 | 3.638 | 99.6% |
| cap 64 | 3846 | 49.32 | 3.622 | 99.4% |
| cap 32 | 3832 | 49.07 | 3.630 | 99.4% |
| cap 16 | 4063 | 47.57 | 3.574 | 99.1% |
| cap 8 | 8676 | 38.71 | 2.431 | 82.4% |
| 6PA2P | 5482 | 40.29 | 3.247 | 98.8% |
| 4PA4P | 10115 | 34.26 | 2.226 | 48.1% |

Throughput change versus base, run-level 95% CI (n = 2 vs 2, coarse): cap 64
−0.4% [−0.9, +0.1]; cap 32 −0.2% [−1.6, +1.2]; **cap 16 −1.8% [−2.6, −0.9]**;
**cap 8 −33.2% [−33.3, −33.1]**; 6PA2P −10.8%; 4PA4P −38.8%.

Traced run (`trace-c30_r1`, `results/projection_step_vs_batch.csv`): over
1626 Projection steps, batch ranged 8–24 sequences; the 36-layer Projection
time per step was 26.1 ms at batch 9–12, 27.0 at 13–16, 29.9 at 17–20 and
30.5 at 21–24. Doubling the batch adds about 17% step time. The aligned
PA/Projection capture failed (longest aligned window 304 steps, the tool needs
512); `run_trace.sh` then waited for the capture's 30-minute timeout and was
stopped by hand after about 7 minutes of idle GPU.

## Conclusion

- C17 (as reworded in X-004) holds: capping the Projection batch below the
  concurrency lowers throughput beyond its CI, slightly at 16 (−1.8%) and
  strongly at 8 (−33%). Caps of 32 and 64 never bind at C30 (the largest batch
  observed is 24).
- The step-time curve explains why: Projection is strongly sublinear in batch
  size, so aggregating the seven PAs' decode steps into one batch is close to
  free per extra sequence.
- The topology arm is descriptive only: 6PA2P and 4PA4P trade Prefill GPUs for
  Projection GPUs, lowering ITL (40 and 34 ms) but losing 11% and 39%
  throughput.
