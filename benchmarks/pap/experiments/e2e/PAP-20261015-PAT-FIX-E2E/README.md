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

## Kernel-level check

| PAT build | `.so` SHA-256 | `test_pap_attention_kernels.py` |
| --- | --- | --- |
| unfixed (`e39d8058`) | `bde45daf26b5096a90a00c538b98da644373207241f31317dd3b6d45634fa7a8` | 21 passed, 3 failed: `test_pat_balancing_writes_each_query_split_exactly_once[2-1024-512]`, `[2-2048-1024]`, `[4-1024-512]` |
| fixed (`9b814d13`), block B | `a136b72b7e322f3750fde5a43e4e620074fdc67775462a037c446f09eab6f5a4` | 24 passed |

PAT builds are not byte-reproducible: the fixed build made for X0 hashed
`9ef1b544…`, and the build that was installed before this experiment
`52030a60…`. A hash therefore identifies one installed build, not its source;
which build the September runs used cannot be recovered from their hashes.

## Results (2026-10-15)

All 12 runs are valid (`analyze.py`: valid=12, invalid=0). Every run completed
180/180 requests with all audits passed. PAT/Triton plan selections per run are
the same in both blocks (about 245 PAT and 84 Triton plans across 7 PAs), so a
difference between blocks is in the kernel, not in kernel selection.

| Group | Runs | TTFT mean / P99 (ms) | ITL mean / P99 (ms) | E2E mean (ms) | req/s | D2 attainment |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| unfixed-auto | 3 | 4324 / 11796 | 46.70 / 59.9 | 8106 | 3.252 | 95.9% |
| fixed-auto | 3 | 4289 / 11970 | 48.04 / 58.6 | 8179 | 3.159 | 95.2% |
| triton (A+B) | 6 | 4184 / 11728 | 49.85 / 60.4 | 8223 | 3.145 | 96.4% |

Ratios minus one, 95% bootstrap CI (latency: request-level stratified by run;
throughput and goodput: run-level, n = 3 vs 3):

| Comparison | TTFT | ITL | E2E | Throughput | D2 goodput |
| --- | --- | --- | --- | --- | --- |
| triton-B / triton-A (drift) | +0.60% [−5.74, +7.24] | −0.01% [−1.54, +1.54] | +0.18% [−4.05, +4.57] | −0.63% [−3.42, +2.25] | −0.43% [−3.51, +2.73] |
| fixed / unfixed | −0.82% [−6.81, +5.58] | **+2.86% [+1.31, +4.45]** | +0.90% [−3.43, +5.47] | **−2.84% [−5.52, −0.09]** | −3.60% [−7.51, +0.69] |
| fixed / triton-B | +2.20% [−4.20, +8.82] | **−3.61% [−5.01, −2.18]** | −0.63% [−4.84, +3.76] | +0.75% [−2.05, +3.64] | −0.59% [−4.11, +3.04] |
| unfixed / triton-A | +3.66% [−2.69, +10.33] | **−6.31% [−7.78, −4.82]** | −1.33% [−5.58, +3.04] | **+3.05% [+0.19, +5.92]** | +2.67% [−1.57, +6.42] |

## Conclusion

- **No drift** between blocks: every triton-B / triton-A CI contains zero and
  the point estimates are within 1%.
- **The fix has a measurable effect (criterion triggered).** Fixed PAT has
  2.9% higher mean ITL (CI excludes zero) and 2.8% lower throughput (CI just
  excludes zero; run-level with n = 3). TTFT, E2E and D2 goodput CIs contain
  zero. The unfixed plan let two CTAs write the same partial-output slot and
  left another slot unwritten, which the gather then read, so affected
  attention outputs were wrong. The fix splits a shared CTA only when all its
  requests have the same next split rank; the likely (unmeasured) cause of the
  slowdown is the resulting loss of split parallelism. The run correctness
  audit checks error-log patterns and fixed output lengths, so it cannot
  detect wrong attention values.
- **PAT versus Triton after the fix:** ITL −3.6% (CI excludes zero);
  throughput, TTFT, E2E and D2 goodput are indistinguishable. Half of the ITL
  advantage measured with the unfixed kernel (−6.3%) came from the bug.
- September `auto` results may include incorrect attention outputs wherever
  the collision occurred, and their ITL and throughput are optimistic by
  roughly 3% relative to the fixed kernel.
