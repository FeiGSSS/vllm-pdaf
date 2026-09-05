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
├── results/           # retained historical summaries and evidence
└── runs/<timestamp>/  # independent new executions, attempts and snapshots
```

New invocations must not rewrite historical result manifests. Resume explicitly
selects a run and verifies its source/configuration identity before reusing any
completed point. The QPS matrix driver implements this convention; older records
may contain only reports or partial evidence and must state those limitations.

Shared dependency setup and process configuration are documented in
[`../scripts/RUNTIMES.md`](../scripts/RUNTIMES.md).

## Entry points versus retained evidence

The current filesystem inventory distinguishes executable protocols from
historical evidence. A historical report is not an instruction to run the
current implementation with old parameters.

| Record | Current entry point / status |
| --- | --- |
| E2E `PAP-20260905-REFACTOR-VALIDATION` | `run.sh` + `experiment.env`; source/dependency snapshots per invocation |
| E2E `PAP-20260903-AGENTIC-CODE-QPS-MATRIX` | Frozen non-Dynamo PAP configuration; requires its recorded source revision, not the current Dynamo-only runner |
| Micro `PAP-20260824-ATTENTION-LATENCY-SURFACE` | `run.sh` + `experiment.env`; kernel workload matrix |
| Micro `PAP-20260905-DYNAMO-CACHE-SALT` | Direct `probe.py` CLI documented in its README; CPU compatibility diagnosis, not serving performance |
| Micro `PAP-20260905-DYNAMO-OWNER-LIFETIME` | Direct `probe.py` CLI documented in its README; CPU lifecycle diagnosis with binary identity and raw observations |
| E2E `PAP-20260824-DYNAMO-ARCH-BASELINES` | Historical architecture comparison |
| E2E `PAP-20260824-QWEN3-131K-YARN-7PA1P` | Historical context-extension validation |
| E2E `PAP-20260824-V026-PORTING` | Historical vLLM integration validation |
| E2E `PAP-20260825-AGENTIC-CODE-1H-MATRIX` | Historical one-hour architecture matrix |
| E2E `PAP-20260827-PREFIX-KERNELS-E2E` | Historical kernel-integration evidence |
| E2E `PAP-20260903-DYNAMO-7PA1P-TRACE` | Historical trace analysis |
| E2E `PAP-20260904-GPU-RESIDENT-TRACE` | Historical dispatcher/first-token analysis |
| Micro `PAP-20260827-PREFIX-KERNELS` | Historical kernel comparison |

Historical records may retain a `run.sh`/`experiment.env` pair; its presence does
not establish compatibility with the current runtime. In particular, the frozen
QPS matrix's conversation-affinity setting is not rewritten to Dynamo, and its
driver rejects that setting before launching any architecture on current code.
Their reports and referenced artifacts are retained as development/comparison
evidence; reproducing the original measurement requires its original source,
dependency and hardware records. If those are absent from an available artifact
bundle, exact reproduction is unproven. Use the report's artifact paths and
Git history as investigation clues, and record missing provenance explicitly.
Do not infer an old runtime identity from the environment currently installed.

Class-level `_runs/` directories contain only historical or ad hoc scratch
runs that were never promoted into a self-contained experiment. New formal
experiments must write beneath their own directory.
