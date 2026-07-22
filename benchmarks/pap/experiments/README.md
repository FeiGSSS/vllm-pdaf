# PAP experiment storage

This directory is the single storage root for new PAP experiment metadata,
conclusions, and repository-owned raw artifacts. Stable architecture
documentation remains in `docs/design/pap/`; dated experimental evidence does
not.

## Experiment layout

A normalized experiment uses this layout:

```text
PAP-YYYYMMDD-STABLE-ID/
├── experiment.json
├── report.md
└── runs/
    └── RUN-ID/
        ├── run.json
        └── raw/
```

`experiment.json`, `run.json`, `report.md`, compact summaries, and the global
indexes are tracked. `raw/` is colocated but ignored by normal Git because a
single trace or serving matrix can be hundreds of MiB.

New uncatalogued runs go to `_staging/`. Promote a run by creating its stable
experiment ID, moving the raw directory below `runs/<run-id>/raw`, recording a
run manifest, and regenerating `INDEX.md`.

## Legacy evidence

`legacy/reports/` contains the preserved human-readable reports that predate
the normalized experiment record. `legacy/runs/` and `legacy/capacity/` hold
the repository-local raw artifacts that have not yet been assigned to one
normalized experiment. They remain local and Git-ignored.

Several multi-gigabyte May/June diagnostic bundles predate this storage rule
and remain under `benchmarks/disagg_benchmarks/results/pap_*` so their ignored
logs and profiles are not moved behind other worktrees. Their locations,
status, and retention rules are recorded in the
[legacy evidence index](legacy/README.md). No new PAP result may be added
there.

The older shared raw root at
`/home/fei/research/PD/test/baseline/pap/results/runs` is still external. It is
27 GiB and may be used by other worktrees, so this migration does not move or
delete it without a separate ownership check.

## Indexes

- `INDEX.md` is the generated status and conclusion index.
- `HISTORY.md` is the detailed historical ledger.
- `history_status.toml` is the compact normalized status overlay for legacy
  rows.

Validate and regenerate with:

```bash
.venv/bin/python benchmarks/pap/validate_registry.py
.venv/bin/python benchmarks/pap/generate_experiment_index.py \
  --output benchmarks/pap/experiments/INDEX.md
```
