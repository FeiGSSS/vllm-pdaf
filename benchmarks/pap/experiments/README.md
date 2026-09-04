# PAP experiment records

Experiment records are separated by unit of work:

- `e2e/`: complete serving runs measured through the external client;
- `microbench/`: isolated kernels, communication primitives, and component
  probes;
- `legacy/`: read-only evidence created before this taxonomy.

New experiments must not be created directly under this directory. Each
experiment has a dated immutable ID because it records an event. Workload
inputs are not experiment artifacts and must come from `../datasets/`.

Within each class, `_runs/` is ignored scratch storage for raw logs and traces.
A reviewed experiment directory may track its protocol, effective input
hashes, compact results, figures, and conclusion. Raw data remains local unless
it is intentionally promoted.
