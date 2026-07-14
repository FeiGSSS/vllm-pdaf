# PAP test governance

This directory is governed by the PAP runtime-refactor milestone design. The
registry in `invariants.json` answers two separate questions:

1. Which PAP behavior or failure semantic does a test protect?
2. What should happen to that test while the suite is reorganized?

Test value is determined by the invariant it protects, not by who wrote the
test or by a target test count.

## Frozen inventory

The initial audit is anchored to commit
`dd2073bcf4637827d2adcd326a316b6b67af4fa4` on 2026-07-14, before any runtime
or existing-test changes in this milestone.

| Item | Frozen value |
| --- | ---: |
| Existing pytest cases | 585 |
| Passed | 582 |
| Skipped | 3 |
| Aggregate setup/call/teardown time | 147817.401 ms |

The skipped cases are the two CUDA local-fast stream-signal tests and the NIXL
local-GPU endpoint test. Per-case durations are diagnostic inventory data from
one CPU Gate run; they are not performance thresholds.

`test_invariant_registry.py` is intentionally excluded from the frozen 585-case
inventory because it validates the inventory itself. The current suite
therefore collects the frozen cases plus the registry validator cases.

## Dispositions

The initial audit records a planned disposition for every frozen node ID. Task
1.1 records decisions only; it does not yet delete, rewrite, or move tests.

| Disposition | Meaning | Initial count |
| --- | --- | ---: |
| `keep` | Uniquely protects surviving behavior, an error semantic, or a regression | 308 |
| `merge` | Protects a necessary invariant but duplicates parameter variants | 20 |
| `rewrite` | Protects a necessary outcome through source strings, private mocks, or a retired switch | 85 |
| `delete` | Protects only an explicitly retired path or already-removed implementation | 43 |
| `move` | Is necessary but belongs in the dedicated PAP hierarchy | 129 |

There is no deletion quota. A test marked `delete` may be removed only with the
corresponding retired runtime path and after the invariant is either removed or
mapped to a stronger surviving test.

Source-string assertions are marked `rewrite` when the underlying contract
still matters. They should become behavior tests or structured configuration
tests; shell tests should be limited to syntax and end-to-end assembly.

## Invariant registry

`invariants.json` contains nine namespaces: `protocol`, `topology`,
`lifecycle`, `kv`, `attention`, `transport`, `integration`, `launcher`, and
`benchmark-validator`. Each invariant records:

- a stable ID and behavioral statement;
- its test level and source owner modules;
- all frozen pytest node IDs mapped to it;
- regression commits or experiment IDs;
- whether coverage is required and its current validation status.

The per-test audit additionally records the owner path, direct source owner,
outcome, reference duration, disposition, reason, and destination for a move or
rewrite. `invariants.schema.json` is the versioned machine-readable contract.

Arbitrary xPAyP topology and cross-host NIXL are supported capabilities, not
retired experiments. Their code and interfaces are retained during the
milestone, but they are marked `preserved-unverified`: this milestone does not
claim fresh end-to-end validation for them. Only the P17 1PA1P `local_fast`
path is a runtime release gate.

## Validation

Run the registry checks with the repository virtual environment:

```bash
.venv/bin/python -m pytest tests/pap/test_invariant_registry.py -v
```

Run the complete PAP CPU Gate with:

```bash
.venv/bin/python -m pytest \
  tests/pap \
  tests/benchmarks/test_compare_pap_pd_multiturn.py \
  tests/benchmarks/test_compare_pap_pd_multiturn_load.py \
  tests/benchmarks/test_finalize_pap_pd_multiturn.py \
  tests/benchmarks/test_pap_multiturn_mps_contract.py \
  tests/benchmarks/test_pap_pd_multiturn_client.py \
  tests/benchmarks/test_pap_pd_multiturn_load_client.py \
  tests/benchmarks/test_pd_three_lane_testbed_contract.py
```

The validator fails if the schema is invalid, IDs are duplicated, source owner
paths disappear, invariant mappings disagree, a required invariant loses all
surviving coverage, or the collected frozen test set drifts from the audit.

When intentionally changing the suite, update the registry in the same commit:

1. map new or replacement tests to a behavioral invariant;
2. update node IDs, owners, disposition reasons, and derived counts;
3. retain regression evidence for required invariants;
4. run the registry validator and complete PAP CPU Gate.

P17 C1/C4 runtime validation belongs to the benchmark runner and strict audit,
not to pytest.
