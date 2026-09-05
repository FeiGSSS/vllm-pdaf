# PAP research diagnosis: stop before candidate generation

The available evidence does not yet support a defensible new mechanism. This is **not** because the question is too broad, and it is not a claim that PAP research is impossible.

The structural tension is real: exact attention readiness, batch-amortized dense work, retained KV capacity, and request latency interact. However, the obvious cures are already occupied:

- CrossPool, Tarragon, and MPK cover independent progress, ready-subset dispatch, and dependency-driven GPU control.
- Feather covers prefix homogeneity versus batch size; Hydragen, FlashInfer, and CoDec cover exact shared-prefix/node computation.
- TokenLake already distributes physical prefix segments, sends queries to their owners, and combines partial attention outputs while decoupling cache placement from compute scheduling.
- LAMPS covers memory-over-time ranking; TOPAS explicitly jointly chooses shared prefix residency and running requests; PRISM couples near-future prefix demand and retention.
- AFD-Ledger already requires component gains to survive a fair full-budget deployment comparison.

These are mechanism collisions, not evidence that every possible interaction is solved. But merely combining them, moving the scheduler onto a GPU, adding a byte-time term, or renaming a batch as a cohort does not establish a new contribution.

The local evidence cannot currently establish the remaining loss. Retained traces prove within-request active KV block aliasing, so historical timing, capacity, and apparent prefix-reuse observations are not valid correct-inference performance premises. The independent Projection probe remains useful: splitting B32 into two B16 invocations raises combined per-layer service from 0.625 to 1.151 ms. That constrains fragmentation costs; it is neither a pipeline makespan nor a serving speedup.

The next useful work is bounded and falsifiable:

1. Complete the permanent KV allocation/ownership fix and numerical validation before new serving interpretation; bug repair itself is not the research contribution.
2. Inspect TokenLake query batching and CoDec's node executor together. State a concrete interaction that their straightforward composition cannot exploit; if composition suffices, retire this direction.
3. On one corrected small workload, join request/step/layer identifiers across readiness, enqueue, execution, attention/transport completion, Projection service, and external delivery. Distinguish execution pins, active-request ownership, and retained cache residency. Determine whether any avoidable interval is before execution, during execution, or after completion.
4. Change only readiness skew or the physical-prefix-sharing graph while keeping work, kernels, topology, device mapping, capacity, clients, and SLO fixed. Compare against independent cohorts and enabled prior prefix/state policies. Invalidate any configuration or correctness mismatch.

Resume ideation only after a reproducible residual remains against the strongest relevant baseline and its mechanism is identified. Do not launch another all-dataset queue simply to search for a favorable ratio.

Detailed diagnosis: [phase1_output.json](phase1/phase1_output.json). Primary mechanism evidence: [mechanism_audit.md](../../literature/mechanism_audit.md) and [targeted_collision.md](../../literature/targeted_collision.md). Local evidence boundaries: [evidence_audit.md](../../evidence_audit.md), [source_audit.md](../../source_audit.md), and [projection_backbone.md](../../projection_backbone.md).
