# PAP Research Workflow

Read this guide before PAP research, paper development, or experiments intended
to support a paper claim. This guide governs research continuity. The PAP NIXL
debugging guide remains mandatory for same-node NIXL runtime, migration, or
benchmark changes.

## Sources of truth

- `docs/design/pap/` defines the current implementation and support boundary.
- `benchmarks/pap/experiments/` owns experiment manifests, reports, and data
  provenance.
- `paper/pap/manuscript.md` is the current paper draft.
- `paper/pap/claims.md` maps paper claims to supporting and conflicting
  evidence.
- `paper/pap/state.md` is the current cross-session research checkpoint.

The paper may synthesize implementation and experimental evidence, but it
never overrides their canonical sources.

## Start gate

Before starting or resuming research:

1. Read the paper workspace README, current state, and claim ledger.
2. Read the current PAP design status and experiment index.
3. Inspect the current commit and worktree before trusting an earlier
   checkpoint.
4. Run `.venv/bin/python tools/pap_research_check.py`.
5. Do not execute a research loop while the state file's execution gate is
   `closed`.

When the gate is open, the active loop must name:

- one paper-level uncertainty;
- a falsifiable hypothesis;
- its falsification condition;
- the expected paper change if supported;
- the minimal experiment or source inspection needed next.

## Research loop

Use this loop:

1. Re-evaluate the paper's largest unresolved gap.
2. Select the highest-value uncertainty that is feasible with current
   resources.
3. Search primary literature and inspect authoritative source code as needed.
4. Run the smallest experiment that can falsify the hypothesis.
5. Diagnose before implementing; distinguish architecture, implementation,
   scheduling, communication, and workload effects.
6. Implement only the solution supported by the diagnosis.
7. Run correctness, causal ablation, and performance evaluation proportional
   to the claim.
8. Update the experiment record, claim ledger, manuscript, and current state.
9. If the paper is incomplete, select the next loop instead of stopping.

Completing one loop is a checkpoint, not completion of the research goal.
Continue across loops unless the user pauses the work, new authority is
required, or an external blocker prevents meaningful progress.

## Evidence rules

- Every performance number in the manuscript must reference a normalized PAP
  experiment ID.
- Every related-work statement must cite a primary source in
  `references.bib`.
- A single run may establish an observation, not a paper-ready claim.
- Paper-ready performance claims require repeated boundary measurements,
  correctness audits, fair per-architecture tuning, and a causal ablation.
- Preserve negative and contradictory results in the claim ledger and
  experiment record. Revise or retire contradicted manuscript text.
- Large raw artifacts may remain outside Git only when their manifest records
  the command, commit, environment, dataset identity, path, size, and digest.
- Figures and tables must identify their source experiment IDs and generation
  command.

Use the experiment evidence grades already defined by the PAP benchmark
methodology. Do not create a second experiment-grading system in the paper.

## Checkpoint gate

Before ending a research turn:

1. Record the current commit, evidence, contradictions, and implementation
   state.
2. Update every affected claim.
3. Update the manuscript only to the maturity supported by the evidence.
4. Update the PAP design status if the implementation boundary changed.
5. Set a non-empty next loop or next action while the paper remains
   incomplete.
6. Run `.venv/bin/python tools/pap_research_check.py`.

Do not mark the overall research goal complete merely because a feature,
experiment, or individual loop is complete. Completion requires an audited
paper-level definition of done from the paper workspace README.

## Versioning

Commit implementation before collecting evidence intended for comparison.
Record that commit and the immutable dataset/config identity in every
experiment. Keep implementation, benchmark evidence, and paper synthesis in
reviewable commits when practical. Use milestone tags only for frozen
baselines, completed mechanisms, evaluation freeze, and paper freeze.

Models and datasets live outside the repository. Record their upstream source,
revision, tokenizer, local identity, and digest in the experiment manifest.
