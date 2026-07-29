# PAP research loop L09: long-input sensitivity

Date: 2026-07-29

## Question and decision

L09 tests whether long input and appended context, rather than output length
alone, materially improve PAP's position relative to corrected PD. It doubles
the document and append distributions while preserving every sampled O16
output, session ID, delay, and request sampling parameter.

C09 is observed at its registered fixed-point threshold. The PAP/PD mean raw
request-throughput ratio rises from the L07 baseline of 0.758 to 0.870, an
11.17-percentage-point improvement above the required 10 points.

This is not yet a goodput result. PAP C34 and PD C48 are both overloaded under
the doubled-input workload and fail all three SLO tiers. L10 must find each
architecture's new capacity boundary before the paper can claim an advantage.

## Fixed-point result

| Architecture | C | Req/s | Mean TTFT | TTFT p95 | Mean ITL | ITL p95 | Strict good | Standard good | Relaxed good |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PAP 7PA1P | 34 | 5.872 | 2737.54 ms | 8967.89 ms | 59.09 ms | 164.49 ms | 65.9% | 80.2% | 85.9% |
| PD 6P2D | 48 | 6.749 | 4105.73 ms | 8887.14 ms | 109.42 ms | 282.48 ms | 15.2% | 49.2% | 69.1% |

At these overloaded fixed points, PAP remains 13.0% lower in raw request
throughput but has 33.3% lower mean TTFT and 46.0% lower mean ITL. Relative to
the short-input L07 controls, request throughput falls 52.2% for PAP and 58.3%
for PD.

## Dataset identity

- Long-input/O16 SHA-256:
  `ae2adf59908bfa7bb6b2ac4cc5d122fdd82d07da11d55361ef87c19f495e6ed5`.
- Sampled document content: mean 8010.672, median 7898, range
  4504--11264 tokens.
- Sampled appended content: mean 1394.203, median 842, range 16--4250 tokens.
- Sampled output: byte-identical to L07, mean 16.436, median 16, range 8--32
  tokens.
- Delays: byte-identical to L07, 0/1000/1000/300/1000 ms per conversation.
- Maximum estimated request budget: 22,122 tokens, below the 32,768 limit.
- Programmatic identity check: all 640 output and delay fields match; all 640
  input texts change.

## Provenance

- Runtime commit: `5c19606e9`.
- Tracked worktree: clean.
- Model/hardware: Qwen3-8B FP16 eager on eight NVIDIA L20 GPUs.
- Runtime: corrected same-node UCX 1.22 GET-zcopy path.
- PAP: 7PA1P C34, conversation affinity, static 18/5 MPS split.
- PD: 6P2D C48.
- Service restart between architecture points.

Repository-local raw bundle:

```text
benchmarks/pap/experiments/_staging/capacity/
  20260729_l09_longinput_o16_fixeddiag/
```

## Interpretation and limitation

The result supports long context as a plausible PAP operating region. Its
seven PA-side Attention engines degrade more slowly than two PD Decode
workers as KV length doubles. However, fixed overloaded points confound raw
capacity and tail quality, and one repetition is only an observation.

L10 performs a small, preselected concurrency bracket on the exact dataset.
It does not reuse the invalid C34/C48 points as the capacity answer.
