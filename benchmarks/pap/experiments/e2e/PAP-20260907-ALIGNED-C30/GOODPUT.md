# Goodput analysis

This analysis uses the concurrency-30 runs from this record plus the two
concurrency-30 7P1D runs in `PAP-20260907-ALIGNED-C30-7P1D`. Every architecture
has two independent service starts and 360 request records. All runs completed
without request errors.

For each run, a request is good only when it meets every configured SLO.
Goodput follows AIPerf's definition:

```text
goodput = good request count / benchmark duration in seconds
```

The two-run value below is pooled: the numerator is the total number of good
requests and the denominator is the sum of the two AIPerf benchmark durations.
The attainment fraction is good requests divided by 360. AIPerf's
`max-goodput-under-slo` search additionally treats a point as feasible only
when its attainment fraction is at least 95% by default.

## AIPerf template

AIPerf's bundled `goodput_slo.yaml` uses TTFT <= 500 ms, request-level ITL <=
50 ms, E2E latency <= 5 s, and output throughput per user >= 20 tok/s. On these
records, the throughput condition is equivalent to ITL <= 50 ms. The template
was designed for roughly 512-token synthetic inputs, whereas this experiment
uses native contexts between 16K and 32K tokens.

| Architecture | Good requests | Attainment | Goodput |
| --- | ---: | ---: | ---: |
| DP8 | 2 / 360 | 0.56% | 0.016 req/s |
| PD 7P1D | 0 / 360 | 0.00% | 0.000 req/s |
| PD 6P2D | 0 / 360 | 0.00% | 0.000 req/s |
| PD 4P4D | 0 / 360 | 0.00% | 0.000 req/s |
| PD 2P6D | 0 / 360 | 0.00% | 0.000 req/s |
| PAP 7PA1P | 0 / 360 | 0.00% | 0.000 req/s |

This result demonstrates that the bundled example SLO is incompatible with
the native-32K workload. It does not establish DP8 as the preferred topology.

## Native-32K diagnostic SLO points

These points expose the tradeoff without presenting a data-derived threshold
as a production SLO:

- D1: TTFT <= 5 s, ITL <= 50 ms, E2E <= 10 s.
- D2: TTFT <= 10 s, ITL <= 75 ms, E2E <= 20 s.
- D3: TTFT <= 15 s, ITL <= 50 ms, E2E <= 20 s.
- D4: TTFT <= 15 s, ITL <= 75 ms, E2E <= 25 s.

Each cell is `goodput req/s (attainment fraction)`. Bold entries meet AIPerf's
default 95% attainment requirement.

| Architecture | D1 | D2 | D3 | D4 |
| --- | ---: | ---: | ---: | ---: |
| DP8 | 0.516 (17.5%) | 1.465 (49.7%) | 0.598 (20.3%) | 1.465 (49.7%) |
| PD 7P1D | 0.106 (5.8%) | 1.008 (55.3%) | 0.117 (6.4%) | 1.697 (93.1%) |
| PD 6P2D | 0.454 (16.9%) | **2.553 (95.3%)** | 1.280 (47.8%) | **2.680 (100%)** |
| PD 4P4D | 0.136 (6.7%) | 0.559 (27.5%) | **1.955 (96.1%)** | **1.955 (96.1%)** |
| PD 2P6D | 0.025 (3.6%) | 0.052 (7.5%) | 0.070 (10.0%) | 0.070 (10.0%) |
| PAP 7PA1P | 0.906 (31.4%) | **2.775 (96.1%)** | 2.222 (76.9%) | **2.888 (100%)** |

At D2, PAP has the highest goodput and satisfies the 95% attainment
requirement; 6P2D is the only other feasible architecture. At D3, the tighter
50 ms ITL SLO makes 4P4D the only feasible architecture, although PAP has a
higher raw good-request rate. At D4, PAP has the highest feasible goodput.

Goodput and attainment use different denominators. D3 makes the distinction
concrete:

| Architecture | Good requests | Duration | Attainment | Raw goodput | 95% feasible |
| --- | ---: | ---: | ---: | ---: | ---: |
| PAP 7PA1P | 277 / 360 | 124.67 s | 76.9% | 2.222 req/s | No |
| PD 4P4D | 346 / 360 | 176.97 s | 96.1% | 1.955 req/s | Yes |

PAP completes the fixed workload faster, so it produces more good requests per
second despite its lower pass fraction. Under AIPerf's default 95% feasibility
constraint, PAP is rejected and 4P4D is the usable result. Raw goodput and SLO
feasibility must therefore be reported separately.

No architecture is feasible at D1. This means concurrency 30 is already too
high for a 5-second TTFT / 50-millisecond ITL / 10-second E2E service target on
this workload.

## Concurrency sensitivity

At D2, the older concurrency-60 runs have much lower attainment and none meet
the 95% requirement:

| Architecture | C30 goodput | C30 attainment | C60 goodput | C60 attainment |
| --- | ---: | ---: | ---: | ---: |
| DP8 | 1.465 | 49.7% | 0.493 | 15.6% |
| PD 6P2D | 2.553 | 95.3% | 0.403 | 17.5% |
| PD 4P4D | 0.559 | 27.5% | 0.188 | 14.4% |
| PD 2P6D | 0.052 | 7.5% | 0.056 | 8.6% |
| PAP 7PA1P | 2.775 | 96.1% | 0.262 | 10.0% |

Only concurrency 30 and 60 have been measured. These points show that 60
overloads the service under D2, but they do not locate the maximum sustainable
goodput. A proper maximum-goodput result requires a concurrency or request-rate
sweep around the SLO knee using one product-defined TTFT, ITL, and E2E target.
