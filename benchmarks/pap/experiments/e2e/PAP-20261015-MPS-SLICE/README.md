# A4: cost of carving Attention SMs out of each Prefill GPU

Request: `paper/handoff/requests/R-004-phase3-ablations-tuning.md` (A4).

## Question

How much do TTFT, ITL and throughput change when each PA gives Attention more
or fewer of the GPU's 92 SMs (static MPS partition, 4 SMs per chunk)?

## Protocol

- PAP 7PA1P, KV migration on, fixed PAT, `s60-t3-native32k-stratified-seed42`,
  closed-loop C30 (`pap.env`, the COMMON §2.3 template).
- Variants (`PAP_STATIC_{PREFILL,ATTENTION}_CHUNKS` and the expected SM
  counts): `sm84-8` (21/2), `sm80-12` (20/3, baseline), `sm76-16` (19/4),
  `sm72-20` (18/5).
- `run_round.sh <k>` runs all four once; three rounds (12 runs).

## Criterion (from R-004)

C18: 80/12 raises TTFT by at most 10% relative to 84/8.

## Results (2026-10-16)

12 valid runs. One extra run (`20261015T152143Z_sm76-16_r1`) started while
new, unrelated experiment files were staged in git, so its `launch.env`
records a dirty tracked worktree; it is excluded and was rerun
(`20261015T155048Z_sm76-16_r1`). The runtime code was identical.

| Prefill/Attention SMs | TTFT mean (ms) | ITL mean (ms) | req/s | D2 attainment |
| --- | ---: | ---: | ---: | ---: |
| 84/8 | 3965 | 57.27 | 2.997 | 95.4% |
| 80/12 (baseline) | 4318 | 48.32 | 3.136 | 95.0% |
| 76/16 | 4729 | 44.65 | 3.144 | 95.4% |
| 72/20 | 5083 | 43.01 | 3.032 | 95.0% |

Change, 95% CI (latency: request-level bootstrap by run; throughput and
goodput: run-level, n = 3 vs 3):

| Comparison | TTFT | ITL | Throughput | D2 goodput |
| --- | --- | --- | --- | --- |
| 80/12 vs 84/8 | +8.9% [+1.2, +16.8] | −15.6% [−17.0, −14.2] | +4.7% [+3.7, +5.6] | +4.3% [+0.6, +7.5] |
| 76/16 vs 80/12 | +9.5% [+2.9, +16.6] | −7.6% [−8.9, −6.3] | +0.3% [−1.0, +1.8] | +0.6% [−2.1, +4.3] |
| 72/20 vs 80/12 | +17.7% [+10.8, +25.1] | −11.0% [−12.2, −9.8] | −3.3% [−4.0, −2.7] | −3.4% [−5.7, −0.2] |

## Conclusion

- C18 holds on the point estimate (80/12 raises TTFT by 8.9% over 84/8) but
  the CI upper bound (16.8%) exceeds the 10% criterion.
- Each 4 SMs moved from Prefill to Attention raise mean TTFT by about 9% and
  lower ITL by 8–16%. 80/12 is where throughput and D2 goodput peak: it
  beats 84/8 on both and is indistinguishable from 76/16; 72/20 loses
  throughput.
