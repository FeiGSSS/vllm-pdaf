# End-to-end experiments

An end-to-end experiment starts a complete serving topology, drives it through
AIPerf, and reports client-visible metrics such as TTFT, TBT, end-to-end
latency, throughput, and goodput.

Raw runs default to `_runs/`. A promoted `PAP-YYYYMMDD-*` record must identify
the dataset ID and SHA-256, topology, effective configuration, source commit,
validity audits, and summary metrics.
