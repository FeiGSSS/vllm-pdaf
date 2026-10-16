# X-003: Trim the remaining R-003/R-004 runs

status: open
priority: P1
claims: C15–C19, G1
opened: 2026-10-16
from: exp agent (to research agent), agreed with the human author

## Context

Since 2026-10-15 16:00 the exp host has made about 220 service starts
(R-001 14, R-002 ~96, R-005 24, R-004 A4 13, R-003 E1 72 + 16 knee runs in
progress). The runbook estimated far fewer; the excess came from extending
the L07 PAP grid to C96, the added R-005, and reruns. About 14 machine-hours
of planned work remain. The human author asked to trim it.

## Proposal

| Item | Plan | Runs | Hours | Change |
| --- | --- | ---: | ---: | --- |
| R-003 E1 knee runs (in progress) | as requested | 16 | 2 | keep |
| R-003 E2d-lite 2×2 | as requested (coarse scan, then 4 points × 2) | ~72 | 5 | keep: C16 needs all four cells |
| R-004 A5/A6 placement 3-way | as requested | 18 | 2 | keep: direct evidence for C19 |
| R-004 E5 TTFT tail breakdown | as requested | 6 | 0.6 | keep: answers G1 |
| R-004 A2 Projection aggregation | **C30 only** (drop C50): 5 variants × 2 + one traced run | 11 | 1 | halved |
| R-004 O1–O3 config tuning | **drop** | 0 | 0 | they only produce suggestions and do not change paper conclusions |

Total about 10.5 machine-hours instead of about 14.

## What is lost

- A2 at C50: the aggregation effect at higher load. C30 still gives the
  batch-cap and topology comparison; the traced run gives Projection step
  time versus batch size.
- O1–O3: no tuning evidence for PAP's own knobs. The paper keeps the current
  PAP defaults and says they were not tuned beyond A4 (MPS split).

## Default

If there is no reply by the time the E1 knee runs and E2d finish, I proceed
with this trimmed plan and record it in the R-003 and R-004 reports.
