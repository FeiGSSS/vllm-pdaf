# PAP experiment records

Experiment records are separated by unit of work:

- `e2e/`: complete serving runs measured through the external client;
- `microbench/`: isolated kernels, communication primitives, and component
  probes.

New experiments must not be created directly under this directory. Each
experiment has a dated immutable ID because it records an event. Workload
inputs are not experiment artifacts and must come from `../datasets/`.

A formal experiment directory owns its `README.md`, requested
`experiment.env`, thin `run.sh`, raw attempts, summaries, and figures. Shared
drivers under `../scripts/` contain reusable execution logic but no frozen
workload settings. Every attempt records the effective post-startup
configuration so that terminal environment overrides cannot silently change a
result.

```text
PAP-YYYYMMDD-NAME/
├── README.md          # question, protocol, validity, and conclusion
├── experiment.env     # frozen requested settings and dataset identity
├── run.sh             # only supported entry point for this experiment
└── results/           # attempts plus promoted summaries and figures
```

Class-level `_runs/` directories contain only historical or ad hoc scratch
runs that were never promoted into a self-contained experiment. New formal
experiments must write beneath their own directory.
