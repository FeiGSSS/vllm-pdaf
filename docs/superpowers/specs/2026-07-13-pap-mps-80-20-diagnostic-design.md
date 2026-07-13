# PAP MPS 80:20 Diagnostic Design

## Goal

Measure whether assigning 80% of the shared PA GPU to Prefill and 20% to the
Attention executor reduces PAP Round-1 TTFT for the frozen C4 five-turn
workload. This is a diagnostic A/B only. The PAP north-star and all subsequent
TPOT optimization remain at Prefill/Attention MPS 70:30.

## Frozen references

Do not rerun or modify the formal PD baselines under
`test/baseline/pap/results/runs/20260713_131649_03d8da336_pd_three_lane_c4_formal`.
Use their pooled request-level medians as read-only references:

- PD-oneway: R1 TTFT 8112.026 ms; R2-R5 TPOT 42.176 ms.
- PD-twoway: R1 TTFT 8128.513 ms; R2-R5 TPOT 42.155 ms.
- PAP 70:30: R1 TTFT 11077.283 ms; R1 TPOT 39.121 ms; R2-R5 TTFT
  249.030 ms; R2-R5 TPOT 51.148 ms.

## Experiment interface

Keep the existing `baseline_70_30` profile as the default in the PAP-only
multi-turn runner. Add one explicit `diagnostic_80_20` profile. The underlying
runner must reject any profile/percentage mismatch, so an experimental setting
cannot silently become the formal default. Record the selected profile and both
percentages in `effective_config.env`.

The three-lane PD/PAP orchestrator remains unchanged and therefore continues to
select `baseline_70_30` implicitly.

## Execution and decision rule

Run one C4 quick PAP cell first with exactly the frozen request shape. Require
20/20 requests, strict correctness audit, routing audit, and zero active
sessions after drain. Compare it with the frozen PAP 70:30 distribution rather
than rerunning PD.

If the quick result shows a change larger than the formal 70:30 run-to-run
spread, run three independently restarted PAP-only repetitions at 80:20 and
aggregate them. Report R1 TTFT, R1 TPOT, R2-R5 TTFT, and R2-R5 TPOT, including
ratios to PAP 70:30 and both frozen PD baselines. Never promote 80:20 to the
default; return to 70:30 before subsequent TPOT work.

## Expected interpretation

More Prefill share may reduce the 16K first-round prefill time. Less Attention
share may increase per-layer synchronization delay and TPOT. The experiment
therefore tests a resource-allocation hypothesis; it does not establish the
root cause unless timing or utilization evidence localizes the saved and added
time to those components.
