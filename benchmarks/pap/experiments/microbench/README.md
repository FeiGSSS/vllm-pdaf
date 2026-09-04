# Microbenchmarks

A microbenchmark isolates one kernel, communication primitive, or component.
Its unit of work must be stated explicitly, for example one Attention launch,
one Projection stage, or one NVSHMEM round trip. Client-visible serving metrics
do not define a microbenchmark result.

Each formal record owns its `run.sh`, `experiment.env`, raw samples, and
derived results. Shared probe code remains in `../../microbench/`. End-to-end
validation of a microbenchmark candidate belongs in `../e2e/`, linked by
experiment ID. `_runs/` is legacy/ad hoc scratch space, not the target for a
new formal experiment.
