# PAP experiment archive

This directory is the single place for PAP-related experiment evidence.

## Layout

- `results/runs/`: PAP benchmark run outputs and service logs.
- `baselines/nixl_disaggregated/results/runs/`: PD/NIXL baseline runs used by
  PAP comparison notes.
- `docs/`: PAP experiment summaries, trace notes, and PD-vs-PAP comparison
  writeups.

## Included PD/NIXL comparison baselines

These runs were moved under `baselines/nixl_disaggregated` because they are
directly referenced by the PAP comparison documents:

- `20260523_055642`: short-output `6P2D`
- `20260523_132823`: short-output `7P1D`
- `20260523_135614`: big-batch `6P2D`
- `20260523_153115`: big-batch `7P1D`
- `20260523_153302`: big-batch `5P3D`
- `20260523_154637`: long-output `5P3D`
- `20260523_154833`: long-output `6P2D`
- `20260527_155219`: 32B TP=2 `2P2D`

Additional 2026-05-25 and 2026-05-27 NIXL/PD sweeps were also moved here
because they were generated during the PAP comparison iterations. Older generic
PD/NIXL runs that are not part of PAP comparison remain in the original
baseline directories.
