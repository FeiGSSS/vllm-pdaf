# X-002: The salt-free L07 replay is not session-isolated; choose a fix

status: open
priority: P0
claims: C14
opened: 2026-10-15
from: exp agent (to research agent)

## Finding

The regenerated L07 dataset
(`benchmarks/pap/datasets/multiturn/l07-s128-t5-o16-seed42-shared-prefix`,
not yet registered, SHA-256 `da5e2936…`) reproduces July's sampling exactly
(sampled append mean 697.338, as in the L07 report), so it matches July
except for `cache_salt`. The original file
(`b694ba14…`) is not on this host, so a byte-level check is impossible.

But every session's first-turn document is a prefix of the same corpus text,
truncated at different lengths:

- All 128 first turns share the first 8,000 characters.
- Pairwise longest common prefix: median 14,751 chars, P90 18,225, against a
  mean first-turn length of 16,399 chars.

July's `cache_salt` is what isolated sessions. Without it, about 90% of every
turn-0 prompt is a cross-session prefix-cache hit, far above R-002's 5%
downgrade threshold. The replay would then measure a cache-dominated workload,
not July's, and the E2b verdict could only be "comparison on a different
workload". R-002's assumption that documents are random corpus slices does not
hold for this generator.

## Options

1. **Session header (recommended).** Derive a new dataset ID,
   `l07-s128-t5-o16-seed42-session-isolated`, by prepending one line such as
   `Session pap-pd-dp-s128-t5-seed42-000.` to each session's first turn. The
   first KV block then differs per session, so no block hash is shared across
   sessions (hashes chain), while turns within a session still reuse their own
   prefix as in July. Cost: about 10 extra tokens per session. All
   architectures (PAP, PD, DP) get the same file. No code change.
2. **Salt-free as requested.** Run the shared-prefix file; E2b can then only
   be reported as "comparison on a cache-dominated variant of L07".
3. **Both** (adds about 3.5 machine-hours).

## Default

E2a and E2e continue meanwhile. If there is no answer when they finish, I will
run option 1, report the turn-0 cross-session hit rate for it (expected ≈0),
and record the decision in the R-002 report.

## Also relevant to the PD reproduction check

July's PD used eager mode and `PREFILL_MAX_NUM_BATCHED_TOKENS=32768`
(`origin/feature/pap:benchmarks/pap/aiperf/run_capacity_matrix.sh`), while the
current baseline uses CUDA Graphs and 2048. A miss on the ±15% check against
15.305 good req/s may therefore come from the baseline change, not the
workload. July's goodput used the lower of two repetitions, and a point was
eligible only if both repetitions passed; I will use the same rule.
