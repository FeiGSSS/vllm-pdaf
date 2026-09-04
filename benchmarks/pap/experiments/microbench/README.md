# Microbenchmarks

A microbenchmark isolates one kernel, communication primitive, or component.
Its unit of work must be stated explicitly, for example one Attention launch,
one Projection stage, or one NVSHMEM round trip. Client-visible serving metrics
do not define a microbenchmark result.

Raw outputs default to `_runs/`. End-to-end validation of a microbenchmark
candidate belongs in `../e2e/`, linked by experiment ID.
