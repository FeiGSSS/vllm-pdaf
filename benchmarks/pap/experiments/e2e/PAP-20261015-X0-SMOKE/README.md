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

## Results

Pending.
