# PAP research loop L10: long-input capacity bracket

Date: 2026-07-29

## Question and decision

L10 asks whether the favorable overloaded latency observed in L09 translates
into higher SLO goodput after reducing concurrency. It runs a preselected
one-repetition bracket at PAP C20/C27/C32 and PD C20/C28/C36 on the exact L09
dataset.

C10 is falsified. Both architectures are standard- and relaxed-SLO eligible
only at C20 among the tested points. PAP goodput is 23.1% lower under standard
and 21.8% lower under relaxed, rather than at least 10% higher.

## Result

| Architecture | C | Req/s | Mean TTFT | Mean ITL | Strict good | Standard good | Relaxed good | SLO eligibility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PAP 7PA1P | 20 | 5.956 | 1582.29 ms | 37.56 ms | 92.3% | 95.8% | 98.1% | standard, relaxed |
| PAP 7PA1P | 27 | 6.466 | 2064.17 ms | 48.70 ms | 80.8% | 89.2% | 91.2% | none |
| PAP 7PA1P | 32 | 6.583 | 2419.36 ms | 57.63 ms | 72.0% | 83.6% | 87.0% | none |
| PD 6P2D | 20 | 7.489 | 1302.58 ms | 35.72 ms | 93.4% | 99.1% | 99.8% | standard, relaxed |
| PD 6P2D | 28 | 7.457 | 2083.33 ms | 52.85 ms | 68.6% | 89.8% | 92.7% | none |
| PD 6P2D | 36 | 6.622 | 3085.77 ms | 90.42 ms | 31.9% | 64.8% | 75.8% | none |

Best eligible goodput in the bracket:

| SLO | PAP | PD | PAP vs PD |
| --- | ---: | ---: | ---: |
| Standard | 5.704 req/s at C20 | 7.418 at C20 | -23.1% |
| Relaxed | 5.844 req/s at C20 | 7.477 at C20 | -21.8% |

No tested point passes strict. PAP's overloaded-tail advantage from L09 does
not survive at the shared C20 feasible point: PD has 17.7% lower mean TTFT,
4.9% lower mean ITL, and 25.7% higher raw request throughput.

## Provenance

- Runtime commit: `77548d1aa`.
- Tracked worktree: clean.
- Dataset SHA-256:
  `ae2adf59908bfa7bb6b2ac4cc5d122fdd82d07da11d55361ef87c19f495e6ed5`.
- Model/hardware: Qwen3-8B FP16 eager on eight NVIDIA L20 GPUs.
- Runtime: corrected same-node UCX 1.22 GET-zcopy path.
- Service restart between every concurrency point.
- All six points complete 640/640 requests and pass correctness validation.

Repository-local raw bundle:

```text
benchmarks/pap/experiments/_staging/capacity/
  20260729_l10_longinput_capacity_bracket/
```

## Interpretation and limitation

The bracket is valid for the registered standard/relaxed question: each
architecture has a passing C20 point and at least two higher-pressure failing
points. It is still coarse. The true boundaries may lie at different
intermediate concurrency values and can change the 21.8--23.1% margin.

L11 refines only the C20--28 interval. It tests PAP C21/C23/C25 and PD
C22/C24/C26, reuses C20 only as the existing lower control, and does not
change workload or runtime settings.
