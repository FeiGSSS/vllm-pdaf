# Handoff templates

## Request — `requests/R-NNN-<slug>.md`

```markdown
# R-NNN: <short title>

status: open
priority: P0 | P1 | P2        # P0 blocks writing
claims: C<n>[, C<m>]          # or "new"
opened: YYYY-MM-DD

## Hypothesis
One falsifiable statement.

## Why it matters for the paper
Which section / figure / argument depends on it.

## Expected result and falsification
What outcome supports it, and what outcome refutes it.

## Suggested setup
Model, topology, workload, baselines, metrics. The exp agent may adjust and
must report each deviation.

## Deliverables
Tables, figure data, or yes/no answer needed.
```

## Report — `reports/R-NNN.md`

```markdown
# R-NNN report: <short title>

verdict: supported | refuted | inconclusive | infeasible
source commit: <sha>
experiments: benchmarks/pap/experiments/<e2e|microbench>/PAP-YYYYMMDD-NAME
raw data (local only): <path>/runs/<timestamp>
date: YYYY-MM-DD

## Answer
Two or three sentences.

## Key results
Small table; plot-ready data in `paper/figures/<file>`.

## Deviations from the request

## Validity and limitations

## Unexpected findings / suggested follow-ups
```
