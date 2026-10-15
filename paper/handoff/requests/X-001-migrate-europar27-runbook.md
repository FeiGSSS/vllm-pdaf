# X-001: Move the Euro-Par 2027 runbook into the handoff protocol

status: open
priority: P0
claims: all
opened: 2026-10-15
from: exp agent (to research agent)

## Context

The runbook `paper/pap-v026/europar27-experiment-runbook.md` on branch
`claude/paper-experiment-plan-6llagm` (`89e60acf`) predates `paper/README.md`.
The human author decided that the protocol in `paper/README.md` applies. The
exp agent has read the whole runbook and checked that every script, tool,
dataset generator, corpus and environment variable it relies on exists on the
exp host. The experimental content is sound; only its packaging conflicts.

## What to change

1. Commit to `feature/pap-v026`, not to a separate branch. Drop the
   `results/europar27` branch and the "do not push to `feature/*`" rule; the
   exp agent commits experiment directories and reports to `feature/pap-v026`.
2. Put the plan in `handoff/requests/`, one request per gated phase, e.g.
   `R-001-phase0-x0-e0`, `R-002-phase1-e2e-e2a-e2b`, `R-003-phase2-e1-e2d`,
   `R-004-phase3-ablations-tuning`. Shared settings (runbook §2) can live in
   one file, e.g. `handoff/requests/COMMON.md`, referenced by each request.
   Gates G0/G1 map to `status: blocked` plus a report the research agent
   answers; phase reports go to `handoff/reports/R-NNN.md`.
3. Map the paper arguments the runbook tests to claims in `handoff/claims.md`
   (rewrite or replace the exp agent's seed claims freely).
4. Commit the documents the runbook cites but which are not in git:
   `paper/experiment-plan.md` (its §3.3 and §8.5 define the E2b criteria and
   the old plan), `novelty-assessment.md`, and the figure input
   `paper/figures/data/requests.csv` format.

## Review notes from the exp agent

- **Run-time estimate.** About 340 runs, roughly 32 h of sequential
  single-host time plus the reply latency at G1. Phases 0–1 (about 7 h) fit
  in one day; please send them first so they can start while the rest is
  being rewritten.
- **E2e microbenchmark safety.** The PCIe fatal reset
  (`microbench/PAP-20260915-P2P-PCIE-INGRESS`) requires inbound peer writes
  into a GPU *while that GPU computes*. The NIXL probes have no concurrent
  compute, so the 1→1 and cross-NUMA points should be safe; the 3→1 fan-in
  is the closest to the fatal regime. The exp agent will run it last and stop
  on the first PCIe error report, as the runbook says.
- **E0 drift check.** Blocks A and B differ by a PAT reinstall and wall-clock
  time; the triton arm covers drift. No change needed.
- **E2b dataset.** The salt-free L07 replay lets different sessions share a
  prefix, unlike July. Agreed this must be stated; consider whether the
  "contrary result removed" verdict is still a fair comparison in that case.
