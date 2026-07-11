# PAP Experiment History Index Design

Date: 2026-07-11

Status: Approved information architecture; ready for implementation planning

## 1. Purpose

Create one Git-tracked entry point for reconstructing the complete PAP
development history. A future reader must be able to start from either a date,
a subsystem, or an observed performance result and progressively reach:

1. the original problem and design motivation;
2. the implementation commits;
3. the controlled comparison or ablation;
4. the raw run, trace, and audit artifacts;
5. the decision that accepted, rejected, rolled back, or superseded the work.

The index is a curated navigation and decision layer. It does not duplicate the
full content of existing design documents or raw logs.

## 2. Deliverable

The main document will be:

`docs/design/pap-experiment-history-index.md`

It will be tracked on `feature/pap`. Existing design documents remain the
authoritative detailed explanations for their modules; the index links to them
and summarizes only the minimum context needed to choose the next disclosure
level.

## 3. Coverage

The first version covers the complete PAP history currently reachable from
`feature/pap`, beginning with the 2026-05-22 NIXL prototype and ending with the
2026-07-11 exact-token and Chat multi-turn cache validation.

Coverage includes all outcomes:

- accepted and currently enabled implementations;
- successful but opt-in research paths;
- negative or regressing ablations;
- correctness failures and invalid measurements;
- dirty-worktree engineering evidence;
- clean formal baselines;
- reverted, removed, and superseded implementations;
- inconclusive experiments whose evidence did not justify a decision.

The index does not claim to recover transient terminal output that was never
written to disk.

## 4. Evidence roots

The index defines symbolic roots once and uses them throughout:

| Symbol | Current path | Role |
| --- | --- | --- |
| `$PAP_REPO` | `/home/fei/research/PD/vllm-pap` | Source, tracked docs, repo-local results |
| `$PAP_RESULTS` | `/home/fei/research/PD/test/baseline/pap/results/runs` | Standard PAP raw runs |
| `$PAP_REPO_RESULTS` | `$PAP_REPO/test/baseline/pap/results/runs` | Older repo-local untracked runs |
| `$PD_RESULTS` | `/home/fei/research/PD/test/baseline/nixl_disaggregated/results/runs` | PD comparison runs |
| `$PAP_PROFILES` | `$PAP_REPO/profile_output` | Profiler and trace-derived artifacts |
| `$PAP_HANDOFF` | `/tmp/pap-handoff-20260707.md` | Temporary historical handoff |

Every artifact link is labelled with one storage class:

- `tracked`: preserved by Git;
- `repo-untracked`: currently present under the repository but not preserved by
  Git;
- `external`: outside the repository and machine-local;
- `temporary`: under `/tmp` and especially vulnerable to deletion;
- `missing`: referenced historically but not found during the 2026-07-11 audit.

## 5. Progressive-disclosure structure

### Level 0: How to use the history

The opening section supports three lookup paths:

- timeline lookup: “what changed between two dates?”;
- module lookup: “why does this subsystem exist and which ablations shaped it?”;
- metric lookup: “which experiment produced this TPOT/TTFT result?”

It also states the current accepted baseline and the distinction between
connection-level x:y support, execution-level many-to-many batching, and
1PA1P multi-turn reuse.

### Level 1: Phase timeline

The timeline groups work into these phases:

1. NIXL and true-split prototype, 2026-05-22 to 2026-05-25;
2. mailbox, topology, and 4PA4P/6PA2P scaling, 2026-05-26;
3. MoE, 30B/32B, wavefront, and TP support, 2026-05-27 to 2026-05-28;
4. upstream synchronization and PD/PAP methodology, 2026-06-30 to 2026-07-01;
5. local-fast, unified KV, lease, and decode commit, 2026-07-02 to 2026-07-07;
6. correctness hardening and same-node TPOT optimization, 2026-07-10;
7. arbitrary x:y and many-to-many cohort/combine/scatter, 2026-07-10 to
   2026-07-11;
8. native APC multi-turn reuse, 2026-07-11.

Each phase row contains the question being answered, the major commits, the
accepted conclusion, and links to its module dossiers.

### Level 2: Module dossiers

The first version contains dossiers for:

1. PAP split architecture and control flow;
2. OFFLOAD_EXEC transport and NIXL mailbox;
3. Projection KV-unaware scheduling;
4. Prefill-owned shared/unified KV;
5. decode commit, lease, and correctness closure;
6. benchmark methodology, tracing, and audit infrastructure;
7. same-node local-fast and TPOT hot-path optimization;
8. arbitrary x:y topology and routing;
9. many-to-many cohort, central dispatcher, combine/scatter, route copy, and
   active-peer membership;
10. exact-token and Chat multi-turn native prefix-cache reuse.

Every dossier uses the same schema:

- problem and motivation;
- hypothesis and intended mechanism;
- design documents;
- implementation and rollback commits;
- experiment groups and workload controls;
- accepted evidence;
- negative, failed, or superseded evidence;
- current status and remaining boundary;
- raw artifact entry points.

### Level 3: Experiment ledger

The ledger is append-only and uses one row per logical comparison rather than
one row per repetition. A logical comparison can link multiple rep directories.

Required fields:

| Field | Meaning |
| --- | --- |
| ID | Stable index identifier, such as `PAP-20260711-M2M-ROUTECOPY` |
| Date/phase | Timeline placement |
| Module | Owning dossier |
| Question | Hypothesis or decision being tested |
| Baseline/treatment | Exact A/B distinction |
| Workload | Model, input/output, QPS, prompts, topology, MPS, and notable flags |
| Code state | Commit, clean/dirty, and patch provenance when known |
| Result | Minimal comparable metrics and correctness status |
| Evidence grade | Formal, controlled, diagnostic, smoke, invalid, or historical |
| Decision | Accept, keep opt-in, reject, roll back, supersede, or inconclusive |
| Artifacts | Design, commit, run directories, audits, and traces |

## 6. Negative-result registry

Negative results are first-class records. Each entry states:

- the attempted mechanism;
- why it was plausible;
- the comparison that rejected it;
- whether the failure was performance, correctness, stability, or methodology;
- the flag or commit that disables/removes it;
- the later approach that replaced it, if any.

Initial entries include mailbox micro-optimizations that regressed TPOT,
Q-first/KV-later variants, wavefront shapes that fragmented batches, adaptive
coalescing windows that did not beat fixed waiting, invalid high-load or OOM
runs, the replaced resident-session multi-turn proposal, and Qwen3
non-thinking template token discontinuity.

## 7. Evidence grades and status vocabulary

Evidence grades are ordered, not interchangeable:

1. `formal-clean`: committed code, clean tracked worktree, repeated controlled
   runs, strict audits;
2. `controlled`: same-code A/B with one intended variable, but possibly dirty;
3. `diagnostic`: useful attribution run that is not a performance baseline;
4. `smoke`: correctness/topology coverage only;
5. `historical`: documented result whose full current metadata is incomplete;
6. `invalid`: failed methodology, OOM, incomplete requests, correctness error,
   or otherwise unusable measurement.

The index never upgrades a dirty, traced, or partial run to a formal baseline.

## 8. Commit and document conventions

- Commit references use the shortest unambiguous hash currently present in the
  repository and include the subject for recognition.
- Tracked documents use relative Markdown links.
- Raw run paths use symbolic roots and code formatting because they are not
  portable Git objects.
- Removed code remains discoverable through its commit and replacement entry;
  the index does not require the old source to remain in the working tree.
- Large tables already present in module documents are summarized and linked,
  not copied wholesale.

## 9. Missing and inconsistent evidence

The index fails visibly rather than silently:

- a missing path is retained and labelled `missing`;
- a path split between standard and repo-local roots is labelled explicitly;
- contradictory metrics are both recorded with workload/provenance differences;
- unknown clean/dirty state is written as `unknown`, not inferred;
- HTTP completion without correctness/session evidence cannot be marked passed;
- traces are labelled diagnostic and are not mixed with normal TPOT baselines.

## 10. Maintenance contract

For every future logical experiment, update at least:

1. the phase timeline if it changes a major decision;
2. the owning module dossier;
3. the experiment ledger;
4. the negative-result registry if the path is rejected or superseded.

The document includes a copyable experiment-entry template. New entries should
be committed with the implementation or result-summary commit when practical.
Raw result directories remain outside Git unless they are small, deliberately
curated summaries.

## 11. Validation

Before committing the first index:

- verify every referenced commit exists in the current repository;
- verify every tracked Markdown link resolves;
- check representative run directories from every phase and classify their
  storage location;
- distinguish missing paths from paths unavailable only outside this machine;
- run `git diff --check`;
- confirm the index and its specification are tracked on `feature/pap`;
- do not run pre-commit, per the current project workflow decision.

## 12. Acceptance criteria

The index is accepted when a reader can answer all of the following without a
full repository search:

- What was the initial PAP architecture, and why did it change?
- Which transport, KV ownership, scheduling, and batching alternatives were
  tried and rejected?
- Which commit implemented any named module?
- Which controlled experiment justified enabling or disabling it?
- Where are the exact raw results and audits?
- Was the evidence clean, dirty, diagnostic, smoke-only, or invalid?
- How did arbitrary x:y support evolve into many-to-many execution?
- How was decode-derived KV reuse proven for exact-token and real Chat history?
- What remains incomplete or externally delegated?
