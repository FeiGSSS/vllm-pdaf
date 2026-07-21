# PAP versus PD: four-GPU AIPerf comparison

## Scope

This controlled experiment validates the AIPerf integration and compares one
PAP topology with all three one-way PD splits on four NVIDIA L20 GPUs. It is a
single-repetition, dirty-worktree result and is not a formal release baseline.

- vLLM/PAP base commit: `3d1fcd38ca5798aae05be01073733976ef815d7e`
- AIPerf: 0.11.0 at `854ff91a4a221f899b806e7660a89b41b80d5689`
- Model: Qwen3-8B, FP16, TP1 per worker
- Load: 12 sessions, five turns, Poisson 12 requests/s, concurrency cap 12
- Shape: 4K-token first user document, about 3K new user tokens per later
  turn, and exactly 256 output tokens per request
- Observed input lengths: 4,114, 7,466, 10,818, 14,170, and 17,522 tokens
- Dataset SHA-256:
  `b2e78fa1ee7b093505a304a2968a6854ed445b7dfd23beae8d02e5c0081ca10a`

Each session has a unique stable `cache_salt`. AIPerf keeps one
`X-Correlation-ID` across that session and inserts the live assistant response
before issuing the next turn. Every topology was restarted before measurement.

## Results

All four runs completed 60/60 requests and 12/12 sessions with no AIPerf
errors. Lower latency is better; higher output throughput is better.

| Architecture | Duration (s) | Output tok/s | TTFT avg / p50 / p90 (ms) | ITL avg (ms) | Request latency avg (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PAP 3PA1P | 72.04 | 211.92 | 2,440 / 2,700 / 3,248 | 41.31 | 12,973 |
| PD 3P1D | 86.72 | 176.99 | 4,644 / 3,668 / 10,022 | 42.17 | 15,396 |
| PD 2P2D | 68.77 | 223.15 | 3,722 / 3,405 / 6,844 | 33.18 | 12,183 |
| PD 1P3D | 90.02 | 170.51 | 7,924 / 7,681 / 17,376 | 29.68 | 15,494 |

Raw aggregate profiles:

- [PAP 3PA1P](../runs/20260716_aiperf_pap_3pa1p_c12_4k_plus3k_v2/aiperf/profile.json)
- [PD 3P1D](../runs/20260716_aiperf_pd_3p1d_oneway_c12_4k_plus3k/aiperf/profile.json)
- [PD 2P2D](../runs/20260716_aiperf_pd_2p2d_oneway_c12_4k_plus3k/aiperf/profile.json)
- [PD 1P3D](../runs/20260716_aiperf_pd_1p3d_oneway_c12_4k_plus3k/aiperf/profile.json)

PAP has the best average and tail TTFT. Against the throughput-leading PD
2P2D point, PAP reduces average TTFT by 34.4% and p90 TTFT by 52.5%, but PD
2P2D provides 5.3% more output throughput and 19.7% lower average ITL. PD 1P3D
has the lowest ITL but becomes Prefill-bound and has the worst TTFT. Therefore
this workload does not support a claim that one topology dominates every
metric: PAP currently favors fast first-token service, while PD 2P2D has the
best aggregate capacity.

## Validity and limitations

The PD proxy reported balanced conversation-affine routing for every topology.
The PAP run passed routing, session-drain, Attention-instance, decode-token
join, and correctness-log audits. On the corresponding near-matched 12-by-5
legacy-client shape, PAP measured 72.11 seconds and 213.02 output tokens/s,
within 0.6% of AIPerf's 72.04 seconds and 211.92 output tokens/s. The API,
prompt construction, per-request arrivals, and round semantics differ, so the
TTFT distributions are not directly interchangeable.

AIPerf cannot currently populate its prompt-cache-read metric because the PAP
gateway exposes Prefill reuse in response headers rather than in
`usage.prompt_tokens_details.cached_tokens`. Cache-hit validity therefore
remains a project E2E gate. DCGM telemetry was unavailable, so these artifacts
contain serving metrics and time slices but no GPU telemetry. Repeat each point
at least three times from a clean commit before promoting the comparison to a
formal milestone.
