# Legacy PAP experiment archive

This shared 27-GiB directory predates the tracked PAP experiment registry. It
is retained in place because other worktrees may reference its raw runs; it is
not the current source of benchmark defaults or conclusions. New experiments,
metadata, reports, and repository-owned raw artifacts belong under
[`benchmarks/pap/experiments/`](../../../benchmarks/pap/experiments/README.md).
Use the [current development status](../../../docs/design/pap/status.md) for
the active runtime and evidence boundary.

The layout below describes the historical archive. The former
`.claude/skills/vllm-pap-benchmark/` implementation has been removed; command
names are retained only in Git history and dated reports for provenance.

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

## Current AIPerf testbed

All new PAP and PD runtime/performance evidence uses the four-GPU AIPerf
capacity matrix:

```bash
bash benchmarks/pap/aiperf/run_capacity_matrix.sh
```

The fixed testbed serves 32 conversations and 320 randomized multi-turn
requests. Development may select one topology/concurrency point; milestone
claims use three repetitions. The former P17 runner is removed, while its
profile and results remain archived for provenance only.

## Historical two-turn comparison test bed

The frozen profile `qwen3_8b_chat_16k_2turn_o256_c1_v1` compared official
1P1D PD/NIXL with 1PA1P PAP on GPUs 1/2. It uses one two-turn conversation,
a 16K first-turn document, a 120-token second-turn append, and 256 output
tokens per turn. TTFT and TPOT are reported separately. The retired PAP
north-star runner used the same-node `local_fast` CUDA-IPC/P2P ring; the
effective transport remains recorded in each run artifact.

Timing uses `last_output_token_v2`: TTFT ends at the first output-token chunk,
and TPOT spans the first through final output token. The client still consumes
the stream through `[DONE]` and HTTP EOF, but records that tail separately as
`eof_latency_ms` and `post_token_stream_ms`; architecture-specific cleanup is
therefore observable without contaminating TPOT.

The original workflow used `run_multiturn_north_star.sh` and
`bootstrap_pd_multiturn_reference.sh` from the removed benchmark skill. These
references are now immutable historical controls. New regression and capacity
evidence uses the AIPerf matrix under `benchmarks/pap/aiperf/`.

The PD lane deliberately ran the unchanged official streaming proxy. In the
API state frozen by this reference, streaming chat chunks did not carry the
Decode-side `kv_transfer_params`, so both proxy lookups were `MISS`. The lane
therefore used the default one-way NIXL mode instead of paying for unreachable
bidirectional pinning. Its validity gate captured P/D `/metrics` and checked
exact two-round token-source conservation, exact local cache boundaries on
both engines, and a second Prefill-to-Decode transfer. Results record this
frozen behavior as `official_streaming_one_way`. A future upstream semantic
change requires a new reference instead of silently mixing measurements.

Raw run directories remain under `results/runs/`. Tracked references live
under `references/qwen3_8b_chat_16k_2turn_o256_c1_v1/` and are no longer
updated in place. A future comparison with different software semantics must
create a normalized experiment instead of rewriting this reference.

Verdicts have the following meanings:

- `diagnostic`: one quick repetition; no stable optimization claim.
- `improved`: formal round-two TPOT is at least 3% below PAP reference.
- `neutral`: formal round-two TPOT is within 3% of PAP reference.
- `regressed`: formal round-two TPOT is at least 3% above PAP reference.
- `invalid`: request, cache, log, profile, hardware, or lifecycle Gate failed.

Formal aggregation also requires one identical Git commit and implementation
fingerprint across all repetitions. PAP session drain, routing, Attention stats
capture, and fatal-log audit evidence are embedded in each result. PD embeds
its exact one-way NIXL token-source evidence and fatal-log audit. Staged and
unstaged tracked changes both make a formal run fail closed.
The finalizer parses the underlying artifacts again, records their run-relative
paths and SHA-256 digests, and refuses labels that disagree with the evidence.
Reference promotion additionally requires three distinct conversation IDs and
source files, then recomputes every median from the three raw measurements.

Every report also states whether `PAP round-two TPOT < 2 * PD TPOT`. TTFT,
round-one TPOT, and conversation latency are retained as independent regression
signals rather than folded into one score.
