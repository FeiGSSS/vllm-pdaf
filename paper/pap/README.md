# PAP Paper Workspace

This directory is the persistent workspace for developing PAP into an
ML-systems paper. It contains the evolving manuscript, claim-to-evidence map,
literature synthesis, and current research checkpoint. It does not own runtime
design facts or experiment results.

## Research charter

- **Target:** ML systems work suitable for an AI venue such as ICML.
- **Deadline:** complete a submission-ready paper during August 2026.
- **Current resources:** all GPUs on the current host, with limited GPU memory
  but ample disk capacity.
- **Planned extension:** up to eight 6000D GPUs with 84 GiB available per GPU.
- **Change scope:** PAP runtime, Proxy, scheduling, KV migration, transport,
  benchmark automation, and supporting vLLM integration may change.
- **Versioning rule:** implementation changes, experimental evidence, and
  manuscript claims must remain traceable through Git and experiment IDs.

The research goal is not to force a favorable PAP result. It is to determine
where PAP creates a defensible advantage, explain its causal mechanism, solve
the limiting system problems, and report both its effective region and its
limitations.

## Workspace map

| Artifact | Responsibility |
| --- | --- |
| `manuscript.md` | Current English paper draft; not a chronological log |
| `claims.md` | Claim maturity, evidence, counterevidence, and next test |
| `state.md` | Current loop and cross-session continuation point |
| `related-work.md` | Source-backed comparison matrix and paper positioning |
| `references.bib` | Bibliography for manuscript and related-work statements |
| `figures/` | Paper figures with provenance |
| `tables/` | Paper tables with provenance |

Canonical implementation guidance remains in `docs/design/pap/`. Canonical
experiment records remain in `benchmarks/pap/experiments/`. The paper cites
those sources rather than copying untraceable results.

## Paper-level definition of done

The overall research goal is complete only when all of the following are
audited:

1. The problem, novelty, and paper story are explicit and supported by primary
   literature.
2. The core mechanism is implemented, correct, and separated from incidental
   engineering fixes.
3. Every central claim has repeated measurements, fair baselines, and causal
   ablations.
4. Evaluation covers the required workload region, model and hardware
   generality, scalability, sensitivity, and failure cases.
5. PAP is compared with tuned PD, fused deployment, and the closest applicable
   related systems.
6. Dataset, model, configuration, commit, and artifact provenance are
   reproducible.
7. The English manuscript, figures, tables, citations, limitations, and
   appendix are complete.

Completing an individual diagnosis, implementation, experiment, or research
loop never satisfies this definition by itself.

## Execution control

`state.md` owns the execution gate. While it is `closed`, agents may maintain
or validate this scaffold but must not begin a concrete research loop. Opening
the gate requires an explicitly aligned loop with a hypothesis and
falsification condition.

Run the scaffold check with:

```bash
.venv/bin/python tools/pap_research_check.py
```
