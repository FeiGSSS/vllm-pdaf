# PAP v0.26 S128/C32 performance checkpoint

## Protocol

- Commit: `63a313b365`
- Historical entry point is recoverable from source commit `63a313b365`; the
  retired mixed S128 baseline wrapper is no longer active.
- Model/topology: Qwen3-8B FP16 TP1, 7PA1P, eight L20 GPUs
- Load: 128 conversations, concurrency 32, 455 requests
- Dataset SHA-256:
  `5421e2d4f9868d4b0dc3f36b5a9aa8e256fadfd929dffd789dbb62692591bd9a`
- Expanded input SHA-256:
  `f1da7ff22ef2446ddf9ae5670f28175fadd90fa37af8eba52d1d562fda22cc69`
- AIPerf: 0.11.0
- Prefill budget: 2048 tokens, 256 sequences, async scheduling off
- All runs: 455/455 requests, zero request errors, Graph/routing/drain audits
  passed

## Results

| Run | TTFT mean (ms) | ITL mean (ms) | ITL P99 (ms) | Requests/s | Duration (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v026_s128_c32_noasync_clean_r1` | 4707.717 | 54.945 | 65.039 | 2.3694 | 192.032 |
| `v026_s128_c32_noasync_clean_r2` | 4808.874 | 55.210 | 64.609 | 2.4158 | 188.341 |
| `v026_s128_c32_noasync_clean_r3` | 4797.060 | 55.309 | 66.057 | 2.3953 | 189.955 |
| Median | 4797.060 | 55.210 | 65.039 | 2.3953 | 189.955 |

The source PAP observation is 4705.389-ms mean TTFT, 60.681-ms mean ITL,
73.620-ms P99 ITL, 2.2885 requests/s, and 198.820 seconds.

Relative to the source PAP observation, the v0.26 median changes are:

- mean TTFT: +1.95%;
- mean ITL: -9.02%;
- P99 ITL: -11.66%;
- request throughput: +4.67%;
- benchmark duration: -4.46%.

The exact-equality TTFT point target is not met. The predeclared 5% TTFT
non-inferiority bound passes, as do the ITL, tail-ITL, throughput, and duration
gates. This checkpoint is therefore overall performance non-inferior with a
small mean-TTFT tradeoff and a material decode improvement.

## Scheduler finding

Leaving v0.26 Prefill asynchronous scheduling on produced a three-run median
of 5033.035-ms TTFT, 55.086-ms ITL, and 2.3467 requests/s. One 2K-token run
with Prefill async scheduling disabled produced 4775.912-ms TTFT,
54.567-ms ITL, and about 2.405 requests/s before the clean repetitions above.

PAP treats Prefill completion as a KV-publication and control-plane handoff.
The asynchronous output pipeline delayed that handoff and increased closed-loop
Prefill queueing, so the v0.26 PAP profile explicitly disables it on Prefill
workers. This setting does not disable the Projection whole-step CUDA Graph or
Attention's GPU-side NVSHMEM communication.
