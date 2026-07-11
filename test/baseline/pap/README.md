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

## Multi-turn north-star test bed

The fixed profile `qwen3_8b_chat_16k_2turn_o256_c1_v1` compares official
1P1D PD/NIXL with 1PA1P PAP on GPUs 1/2. It uses one two-turn conversation,
a 16K first-turn document, a 120-token second-turn append, and 256 output
tokens per turn. TTFT and TPOT are reported separately.

Run one diagnostic PAP repetition:

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/run_multiturn_north_star.sh quick
```

Run three serial PAP repetitions for an optimization verdict:

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/run_multiturn_north_star.sh formal
```

Bootstrap a new official PD reference candidate only when the frozen PD
reference must be refreshed:

```bash
bash \
  .claude/skills/vllm-pap-benchmark/scripts/bootstrap_pd_multiturn_reference.sh
```

The daily PAP commands never start PD and never update a reference. Raw run
directories remain under `results/runs/`. Tracked references live under
`references/qwen3_8b_chat_16k_2turn_o256_c1_v1/` and can only be updated with
the comparison CLI's explicit `write-reference --allow-reference-write`
operation.

Verdicts have the following meanings:

- `diagnostic`: one quick repetition; no stable optimization claim.
- `improved`: formal round-two TPOT is at least 3% below PAP reference.
- `neutral`: formal round-two TPOT is within 3% of PAP reference.
- `regressed`: formal round-two TPOT is at least 3% above PAP reference.
- `invalid`: request, cache, log, profile, hardware, or lifecycle Gate failed.

Every report also states whether `PAP round-two TPOT < 2 * PD TPOT`. TTFT,
round-one TPOT, and conversation latency are retained as independent regression
signals rather than folded into one score.
