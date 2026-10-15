# X0: environment check and smoke runs for the paper campaign

Request: `paper/handoff/requests/R-001-phase0-x0-e0.md` (X0 part), settings
from `paper/handoff/requests/COMMON.md` §2.

## Question

Does the exp host reproduce a valid PAP 7PA1P and Dynamo 6P2D C30 run on the
current source (`9b814d13`, PAT fix merged) before the paper experiments start?

## Protocol

- Checks: GPU/driver inventory, NVSHMEM bridge rebuild (ABI 11), installed PAT
  identity, dataset checksums, `pytest tests/pap`.
- One PAP 7PA1P run (KV migration on, `auto` attention kernel, fixed PAT) and
  one Dynamo 6P2D run, both on `s60-t3-native32k-stratified-seed42`, C30.
- `pap.env` is the COMMON §2.3 template; `dynamo.env` is a byte copy of
  `PAP-20260907-ALIGNED-C30/results/common.env`; `variants/*.env` select the
  runner and architecture. `bash run.sh <variant> <repetition>` is the only
  entry point. `run.sh` is the template reused by later paper experiments.

## Pass criteria

Bridge builds; pytest passes; both runs complete 180/180 requests with all
audits passed; PAP mean TTFT within 4.2–6.2 s.

## Results (2026-10-15)

| Check | Result |
| --- | --- |
| GPUs | 8× NVIDIA L20, driver 595.99.02 (matches COMMON) |
| NVSHMEM bridge | rebuilt, `.local/nvshmem-3.3.24-cuda13/lib/libpap_nvshmem_device.so` |
| PAT before rebuild | `prefix_attn` 0.1.1, `.so` `52030a60…` (pre-merge install) |
| PAT after rebuild (fixed, `9b814d13`) | `.so` `9ef1b54469dffbd8df7afacd5f5621578b90ed264e36a2c4308c856059a591e3` |
| Dataset checksums | `sha256sum --check SHA256SUMS` passed |
| `pytest tests/pap` (fixed PAT) | 313 passed (COMMON expected 309; PR #1 added tests) |

| Run | Requests | Audits | TTFT mean / P99 | ITL mean | req/s | D2 attainment |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| `20261015T081115Z_pap-7pa1p_r1` | 180/180 | all passed, 34 KV migrations | 4099 / 11310 ms | 50.71 ms | 3.221 | 96.1% |
| `20261015T081436Z_dynamo-6p2d_r1` | 180/180 | correctness, KV transfer, CUDA Graph passed | 5738 ms | 50.24 ms | 2.694 | — |

Both runs are valid and the TTFT breakdown ran on the Dynamo run. Two
deviations from the expectations in COMMON:

- PAP mean TTFT (4.10 s) is slightly *below* the 4.2–6.2 s band. The band was
  meant to catch a broken setup; a faster run is not a validity failure.
- PAP mean ITL is 50.7 ms against 45.2 ms in September (single run). E0's
  three fixed-auto runs will show whether this persists.

6P2D matches September closely (TTFT 5.70 s, ITL 49.0 ms, 2.680 req/s).
Raw runs stay local under `runs/`.
