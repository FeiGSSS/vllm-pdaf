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
| C1 | At C30, PAP 7PA1P with KV migration has the highest request throughput of DP8, PD (7P1D/6P2D/4P4D/2P6D) and PAP: 3.194 req/s vs 2.947 (DP8) and 2.680 (best PD, 6P2D). | partial | `e2e/PAP-20260909-KV-MIGRATION-ARCHITECTURE-C30/report.md`; pre-fix PAT: ITL/throughput ~3% optimistic, see R-001 |
| C2 | PAP dominates PD 6P2D and 7P1D on all mean metrics (vs 6P2D: TTFT −20.96%, ITL −7.68%, E2E −15.67%, throughput +19.19%). | partial | same as C1; pre-fix PAT: ITL/throughput ~3% optimistic, see R-001 |
| C3 | PAP is on the TTFT/ITL/throughput Pareto frontier with DP8, PD 4P4D and PD 2P6D; DP8 keeps the lowest TTFT (2.208 s vs 4.504 s), PD 4P4D/2P6D keep lower ITL. | partial | same as C1; pre-fix PAT: ITL/throughput ~3% optimistic, see R-001 |
| C4 | PAP cuts ITL vs DP8 by ~42% (45.2 vs 77.4 ms mean; P99 57 vs 163 ms), consistent with removing Prefill/Decode interference. | partial | C1 report; ITL measured, interference as cause not isolated; pre-fix PAT: ITL/throughput ~3% optimistic, see R-001 |
| C5 | Fixed PD splits strand GPU capacity: 7P1D saturates its Decode GPU (96.5%) with Prefill GPUs at 44–55%; 2P6D the reverse. PAP keeps all 8 GPUs at 87–93%. | supported | `e2e/PAP-20260907-ALIGNED-C30/ARCHITECTURE_COMPARISON.md` (physical-GPU telemetry; cannot split PA vs Projection per card) |
| C6 | PAP and DP8 TTFT improve on later turns while PD splits degrade (PAP 5.95 → 5.18 → 4.96 s; 6P2D 4.66 → 6.16 → 6.27 s). | partial | ARCHITECTURE_COMPARISON.md; trend measured, mechanism (cache reuse vs queueing vs transfer) needs request-ID timelines |
| C7 | Migrating conversation KV on a PA change instead of recomputing the prefix reduces computed Prefill tokens 19.1%, mean TTFT 17.4% and raises throughput 9.6%, at +4.1% ITL. | partial | `e2e/PAP-20260909-DYNAMO-PA-KV-MIGRATION/report.md` (alternating A/B, 2 runs per arm); pre-fix PAT: ITL/throughput ~3% optimistic, see R-001 |
| C8 | Cross-host KV migration costs essentially nothing end to end vs same-host (mean TTFT +7.4 ms, +0.17%). | partial | `e2e/PAP-20260909-CROSS-HOST-KV-MIGRATION-C30/report.md`; pre-fix PAT: ITL/throughput ~3% optimistic, see R-001 |
| C9 | PAP's disaggregated gains hold at C60 and grow at C30 mainly via TTFT; disaggregated ITL changes only 0–8% across load, DP8 ITL is strongly load-sensitive (−46% from C60 to C30). | partial | ARCHITECTURE_COMPARISON.md, concurrency sensitivity (C60 lacks 7P1D and KV migration); pre-fix PAT: ITL/throughput ~3% optimistic, see R-001 |

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
| C12 | On PCIe L20 with 16–32K contexts, PD's P→D KV handoff is at least 10% of TTFT (below 5% means "no KV movement" is not the main argument). | pending | R-002 (E2e) |
| C13 | The PD 6P2D and DP8 baselines are tuned: no single-knob change improves D2 goodput beyond its CI. | pending | R-002 (E2a) |
| C14 | On the July L07 short-output workload, current PAP reaches at least 0.95× PD's Standard-SLO max goodput (July: 0.63×). | pending | R-002 (E2b) |
| C15 | On native-32K multi-turn, PAP 7PA1P with KV migration has the highest maximum sustainable D2 goodput among DP8, PD 6P2D and PD 4P4D. | pending | R-003 (E1) |
| C16 | PAP's advantage over PD grows with input length and shrinks with output length (2×2 regime map). | pending | R-003 (E2d) |
| C17 | Cross-PA Projection aggregation contributes to throughput: capping the Projection batch or splitting Projection across more GPUs lowers throughput. | pending | R-004 (A2) |
| C18 | Carving 12 SMs from each Prefill GPU for Attention is cheap: 80/12 raises TTFT by at most 10% relative to 84/8. | pending | R-004 (A4) |
| C19 | At equal load, Dynamo routing with PA→PA KV migration gives lower TTFT than Dynamo routing alone and than sticky session round-robin. | pending | R-004 (A5/A6) |

G1 (TTFT tail) is addressed by R-004 (E5); G2 (generality and variance) is
partly addressed by R-003 (three runs per point, confidence intervals).
