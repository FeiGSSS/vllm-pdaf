# End-to-end experiments

An end-to-end experiment starts a complete serving topology, drives it through
AIPerf, and reports client-visible metrics such as TTFT, TBT, end-to-end
latency, throughput, and goodput.

A `PAP-YYYYMMDD-*` record must keep its thin `run.sh`, `experiment.env`, raw
attempts, and promoted results together. It must identify the dataset ID and
SHA-256, topology, effective configuration, source commit, validity audits,
and summary metrics. `_runs/` is legacy/ad hoc scratch space, not the target
for a new formal experiment.
