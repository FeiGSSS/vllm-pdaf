# Native-32K serving diagnosis and validation

This record measures client-visible latency and request-correlated KV lifecycle
events. See [the report](report.md) for definitions, historical evidence,
limitations, and the subsequent end-to-end correctness validation.
Implementation changes and regression tests belong in `vllm/` and `tests/`;
repair motivation belongs in the code-fix commit description.

The Decode commit repair requires no dedicated shell script. Its regression
coverage belongs in the existing test suites. Further service validation uses
the shared `benchmarks/pap/scripts/run_pap_workload.sh` entry point.

## Execution

From this directory:

```bash
bash run.sh    # IPC/lease logging and cached metrics collection
```

The diagnostic experiment uses `experiment.env`: Qwen3-8B FP16, native 32768 context without
YaRN, 7PA1P on Prefill GPUs 0–6 and Projection GPU 7, 60 concurrent sessions,
180 requests. The dataset SHA-256 is checked before launch. Diagnostic instrumentation
adds overhead; this is not a performance A/B against the validation records.

The entry point clears inherited workload settings. Each invocation creates a
unique `runs/<UTC timestamp>-<suffix>/` directory and saves its requested
configuration and launcher source. The shared runner records
effective configuration, source revision/worktree patches, topology and runtime
audits, client output, and service logs. A failed runner remains a failed attempt;
client success alone does not establish internal correctness.

The source checkout may be dirty for diagnosis; its saved patches are part of
the run identity. Re-running the current checkout is a new observation, not an
exact reproduction of a historical revision. Historical `diagnostic-run/` and
the report's legacy `_runs/` references remain unchanged. Analysis scripts still
target those historical records; they do not automatically analyze new runs.
Saved launcher copies are provenance snapshots, not standalone entry points.
