# Paper claims

The research agent owns the claim text and may rewrite, merge or drop claims.
The exp agent owns `Status` and `Evidence`.

Status: `supported` · `partial` · `pending` · `refuted` · `open-gap`
(a measured effect whose mechanism is not established).

## Seed claims from existing evidence (exp agent, 2026-10-15)

Common setup unless noted: Qwen3-8B FP16, TP1, native 32K, 8× NVIDIA L20 on
one host, 60 sessions × 3 turns = 180 requests per run, closed-loop
concurrency 30, two independent service starts per arm, DP/PD on Dynamo 1.4.1
with vLLM 0.26.0. Paths are relative to `benchmarks/pap/experiments/`.

| ID | Claim | Status | Evidence |
| --- | --- | --- | --- |
| C1 | At C30 on native-32K multi-turn, PAP 7PA1P with KV migration has the highest request throughput of DP8, PD 6P2D and PD 4P4D: 3.62 req/s vs 3.26 (DP8), 3.04 (6P2D) and 2.21 (4P4D). | supported | R-003 E1 (fixed PAT, s240, C30): PAP 3.62 req/s vs DP8 3.26, 6P2D 3.04, 4P4D 2.21; `e2e/PAP-20261015-LOAD-SWEEP` |
| C2 | At C30, PAP beats PD 6P2D on TTFT, ITL and throughput at once (TTFT P50 3.67 vs 4.91 s, ITL P50 50.0 vs 55.0 ms, 3.62 vs 3.04 req/s). | supported | R-003 E1 C30: PAP vs 6P2D TTFT P50 3.67 vs 4.91 s, ITL P50 50.0 vs 55.0 ms, 3.62 vs 3.04 req/s; `e2e/PAP-20261015-LOAD-SWEEP` |
| C3 | PAP is on the TTFT/ITL/throughput Pareto frontier with DP8 and PD 4P4D: DP8 keeps the lowest TTFT, 4P4D the lowest ITL, and PAP the highest D2 goodput at every SLO scale from 0.75× to 2×. | supported | R-003 E1: DP8 keeps the lowest TTFT, 4P4D the lowest ITL, PAP the highest D2 goodput at every SLO scale 0.75–2×; `e2e/PAP-20261015-LOAD-SWEEP` |
| C4 | PAP cuts ITL vs DP8 by ~42% (45.2 vs 77.4 ms mean; P99 57 vs 163 ms), consistent with removing Prefill/Decode interference. | partial | C1 report; ITL measured, interference as cause not isolated; pre-fix PAT: ITL/throughput ~3% optimistic, see R-001 |
| C5 | Fixed PD splits strand GPU capacity: 7P1D saturates its Decode GPU (96.5%) with Prefill GPUs at 44–55%; 2P6D the reverse. PAP keeps all 8 GPUs at 87–93%. | supported | `e2e/PAP-20260907-ALIGNED-C30/ARCHITECTURE_COMPARISON.md` (physical-GPU telemetry; cannot split PA vs Projection per card) |
| C6 | PAP and DP8 serve later turns faster than the first turn (not necessarily monotonically), while PD splits get slower on later turns (Sept. C30: PAP 5.95 → 5.18 → 4.96 s; 6P2D 4.66 → 6.16 → 6.27 s; R-003 E1 confirms the pattern on s240). | supported | R-003 E1: PAP C30 4150/3639/3754 ms, 6P2D C30 4498/5108/5411 ms by turn; `e2e/PAP-20261015-LOAD-SWEEP` |
| C7 | Migrating conversation KV on a PA change instead of recomputing the prefix reduces computed Prefill tokens 19.1%, mean TTFT 17.4% and raises throughput 9.6%, at +4.1% ITL. | partial | `e2e/PAP-20260909-DYNAMO-PA-KV-MIGRATION/report.md` (alternating A/B, 2 runs per arm); pre-fix PAT: ITL/throughput ~3% optimistic, see R-001 |
| C8 | Cross-host KV migration costs essentially nothing end to end vs same-host (mean TTFT +7.4 ms, +0.17%). | partial | `e2e/PAP-20260909-CROSS-HOST-KV-MIGRATION-C30/report.md`; pre-fix PAT: ITL/throughput ~3% optimistic, see R-001 |
| C9 | Disaggregated ITL is far less load-sensitive than DP8's: from C10 to C60, ITL P50 rises 35→51 ms for both PAP and PD 6P2D but 27→169 ms for DP8. | supported | R-003 E1 C10→C60: ITL P50 PAP 35→51 ms, 6P2D 35→51 ms, DP8 27→169 ms; `e2e/PAP-20261015-LOAD-SWEEP` |

## Known gaps (likely reviewer questions)

| ID | Gap | Status | Evidence |
| --- | --- | --- | --- |
| G1 | PAP TTFT P99 is 14% above PD 6P2D in the pre-migration comparison; mechanism unexplained. | open-gap | ARCHITECTURE_COMPARISON.md, tail latency |
| G2 | All results use one model (Qwen3-8B), one GPU type (L20), one workload, two runs per arm; generality and variance are untested. | pending | — |
| G3 | Cross-host PD over GPUDirect RDMA can fatal-reset the L20 host when a GPU receives ~28 GB/s inbound peer writes while computing; this limits cross-host RDMA experiments on this platform. | supported (platform issue) | `microbench/PAP-20260910-GDR-P2P-MINIMAL/README.md`, `microbench/PAP-20260915-P2P-PCIE-INGRESS/README.md` |
| G4 | Current code after the 2026-10-15 cleanup matches its base within run-to-run noise (TTFT mean 5686 vs 5798 ms, without KV migration). | supported | `e2e/PAP-20261015-CLEANUP-REGRESSION/README.md` |

## Claims under test (research agent, 2026-10-15)

C1–C9 were measured before the PAT split-rank fix (`9b814d13`). R-001 decides
whether they still stand; the C30 single-point results C1–C3 will be replaced
by the R-003 load sweep. Thresholds below are the falsification criteria the
requests use.

| ID | Claim | Status | Evidence |
| --- | --- | --- | --- |
| C10 | The PAT split-rank bug has no measurable end-to-end effect: fixed vs unfixed PAT TTFT/ITL/throughput ratios all have 95% CIs containing zero. | refuted | R-001: fixed/unfixed ITL +2.9% [+1.3, +4.5], throughput −2.8% [−5.5, −0.1]; `e2e/PAP-20261015-PAT-FIX-E2E` |
| C11 | PAT improves end-to-end PAP performance over Triton on the shared-prefix native-32K workload. | partial | R-001: fixed PAT vs Triton ITL −3.6% [−5.0, −2.2]; TTFT, throughput, D2 goodput not distinguishable |
| C12 | On PCIe L20 with 16–32K contexts, PD's P→D KV handoff is at least 10% of TTFT (below 5% means "no KV movement" is not the main argument). | refuted | R-002 E2e: 6P2D handoff ≈3–4.5% of TTFT at 16–32K (≤7% full-prompt bytes), C30; 3→1 fan-in 3.0–3.3× slower per stream; `microbench/PAP-20261015-PD-KV-TRANSFER` |
| C13 | The PD 6P2D and DP8 baselines are tuned: no single-knob change improves D2 goodput beyond its CI. | supported | R-002 E2a: no 6P2D/DP8 single-knob change beats baseline with CI excluding 0; `e2e/PAP-20261015-BASELINE-TUNING`; R-005: on L07, July's DP batch limit (32768) does not improve DP8 (Standard 7.90 vs 8.91); `e2e/PAP-20261015-DP8-STRENGTH` |
| C14 | On the July L07 short-output workload, current PAP reaches at least 0.95× PD's Standard-SLO max goodput (July: 0.63×). | supported | R-002 E2b: PAP/PD capacity 1.24 Strict, 1.36 Standard, 1.27 Relaxed (July 0.68/0.63/0.76); turn-0 cross-session hit 0; PD 0.885× July; `e2e/PAP-20261015-L07-REPLAY` |
| C15 | On native-32K multi-turn, PAP 7PA1P with KV migration has the highest maximum sustainable D2 goodput among DP8, PD 6P2D and PD 4P4D. | supported | R-003 E1: max D2 goodput PAP 3.63 [3.59, 3.67] vs 6P2D 3.00 [2.96, 3.03], DP8 2.32, 4P4D 2.14; same order under D4 and all SLO scales 0.75–2×; `e2e/PAP-20261015-LOAD-SWEEP` |
| C16 | Input length decides PAP versus PD and output length decides PAP versus DP8: with long inputs PAP beats PD at both output lengths (1.21×, 1.56×); with short outputs PAP beats DP8 (1.56–2.07×), with 256-token outputs DP8 wins (PAP/DP8 0.69 and ≤ 0.50) and PD edges PAP on short inputs (0.94). | partial | R-003 E2d: input half holds (long input: PAP/PD 1.21 at O80, 1.56 at O256); output half reverses vs DP8 (PAP/DP8 2.07→≤0.50 short input, 1.56→0.69 long input) and vs PD with short input (1.36→0.94); `e2e/PAP-20261016-REGIME-2X2` |
| C17 | Cross-PA Projection aggregation contributes to throughput: capping the Projection batch or splitting Projection across more GPUs lowers throughput. | pending | R-004 (A2) |
| C18 | Carving 12 SMs from each Prefill GPU for Attention is cheap: 80/12 raises TTFT by at most 10% relative to 84/8. | pending | R-004 (A4) |
| C19 | At equal load, Dynamo routing with PA→PA KV migration gives lower TTFT than Dynamo routing alone and than sticky session round-robin. | pending | R-004 (A5/A6) |
| C20 | Under load, PD 6P2D becomes ITL-bound first (ITL P95 > 75 ms from C38–C44 on L07) while PAP 7PA1P keeps ITL P95 ≤ 57 ms through C72; on native-32K multi-turn the same ordering holds across the R-003 sweep. | supported | R-002 E2b (L07); R-003 E1: 6P2D ITL P95 69 ms at C30, 74 ms at C40; PAP ≤ 65 ms P95 through C60; `e2e/PAP-20261015-LOAD-SWEEP` |

G1 (TTFT tail) is addressed by R-004 (E5); G2 (generality and variance) is
partly addressed by R-003 (three runs per point, confidence intervals).
