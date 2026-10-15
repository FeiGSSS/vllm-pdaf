# E0: end-to-end effect of the PAT split-rank fix

Request: `paper/handoff/requests/R-001-phase0-x0-e0.md` (E0 part).

## Question

Did the PAT split-rank collision fixed by PR #1 (`9b814d13`) change PAP
end-to-end performance, and what does PAT gain over Triton end to end?

## Protocol

- Qwen3-8B FP16, PAP 7PA1P, Dynamo routing, KV migration on (snapshot),
  `s60-t3-native32k-stratified-seed42`, closed-loop C30, one service start per
  run (`pap.env`, the COMMON §2.3 template).
- Source `9b814d13`. The unfixed PAT is built from `e39d8058` in
  `.worktrees/pat-unfixed`; the fixed PAT from this checkout. Only one PAT can
  be installed in the shared `.venv`, so the runs are in two blocks:
    - A (unfixed installed): `unfixed-auto r1, triton r1, unfixed-auto r2,
    triton r2, unfixed-auto r3, triton r3`
    - B (fixed installed): `fixed-auto r1, triton r4, fixed-auto r2, triton r5,
    fixed-auto r3, triton r6`
- `variants/*-auto.env` pin the expected PAT `.so` SHA-256; `run.sh` refuses
  to start if another build is installed. Every run records the installed
  hash in `launch.env`.
- Triton does not call PAT; triton-B / triton-A estimates drift between blocks.
- `analyze.py` validates every run (exit code, 180 requests, no AIPerf error or
  output mismatch, all STATUS audits, MPS 80/12, KV migration audit, clean
  tracked worktree) and writes `results/summary.json` and
  `results/requests.csv`.
- Statistics: latency ratios use a request-level bootstrap stratified by run
  (10,000 draws); throughput and D2 goodput use a run-level bootstrap because
  each run contributes one value (n = 3 per arm, so these CIs are coarse).

## Criteria (from R-001)

- Drift (triton-B / triton-A) above 3% → rerun a block A group after block B.
- fixed-auto / unfixed-auto: TTFT, ITL and throughput CIs all contain 0 → no
  measurable effect; otherwise R-001 is blocked.
- fixed-auto / triton: PAT's end-to-end benefit; if none, later requests use
  `triton`.

## Offline precheck: PAT selection in existing archives

Per-PA counts of PAT plan rebuilds and Triton selections from
`attention_fast_path_stats_<i>.json`. In every PA, PAT + Triton equals
`step_context_misses`: a kernel is chosen on each context miss and reused by
the incremental steps that follow (≈1050–1300 per PA).

| Archive | PAT rebuilds per PA | Triton selections per PA |
| --- | --- | --- |
| CLEANUP-REGRESSION current_r1 | 16 38 29 27 32 28 25 | 18 12 16 17 20 14 18 |
| CLEANUP-REGRESSION base_r1 | 16 28 31 36 37 28 26 | 15 15 14 14 17 22 16 |
| CLEANUP-REGRESSION current_r2 | 43 29 23 24 20 25 39 | 12 16 15 19 17 16 17 |
| CLEANUP-REGRESSION base_r2 | 28 24 28 38 26 23 28 | 21 17 14 16 17 17 16 |
| KV-MIGRATION-ARCHITECTURE-C30 r7 | 37 29 26 34 37 31 32 | 12 16 13 14 11 17 13 |
| KV-MIGRATION-ARCHITECTURE-C30 r8 | 29 33 31 42 31 32 36 | 15 14 16 8 14 12 14 |
| ALIGNED-C30 run_pap1 | 33 33 38 36 20 30 20 | 13 13 15 18 20 10 21 |
| ALIGNED-C30 run_pap2 | 30 35 32 25 19 30 37 | 13 15 11 18 22 15 15 |

PAT is chosen for about two thirds of kernel plans in every archived PAP run,
so every September `auto` result could have been affected.

## Results

Pending.
