# PAP test strategy

PAP tests protect supported behavior; they do not maintain a permanent record
for every historical pytest node. Tests for retired implementation paths should
be deleted with those paths, and duplicate cases should be merged when the
surviving test covers the same behavior.

## What must remain covered

- P17 1PA1P `local_fast` protocol, lifecycle, unified-KV, Attention, and
  transport behavior;
- fail-closed validation for malformed descriptors, stale generations, and
  invalid lifecycle transitions;
- eager and optional piecewise CUDA Graph role selection, graph boundaries,
  capture-shape selection, and eager fallback;
- stable xPAyP and cross-host NIXL configuration and wire contracts.

Same-host xPAyP has a small controlled E2E smoke for `1PA2P`, `2PA1P`, and
`2PA2P`. It is not a performance or release gate. Cross-host execution remains
supported but `preserved-unverified` during this milestone.

## Current validation lanes

Run tests directly related to each source change. Run the complete PAP CPU
gate before a source milestone, and use a runtime lane only when the change
crosses a runtime boundary:

- P17 1PA1P is the release regression gate for lifecycle, transport, unified
  KV, Attention, or scheduling changes.
- The four-GPU AIPerf lane is for capacity, routing, and performance changes;
  documentation-only or isolated unit changes do not require it.
- Piecewise Graph changes require the focused graph/launcher tests and a
  matched eager/Graph runtime point before a performance claim.

Pre-commit is not required in this environment. A focused Graph check is:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_model_cudagraph.py \
  tests/pap/test_pap_launch_files.py -q
```

The complete PAP CPU gate is:

```bash
.venv/bin/python -m pytest \
  tests/pap \
  tests/benchmarks/pap \
  tests/benchmarks/test_compare_pap_pd_multiturn_load.py \
  tests/benchmarks/test_finalize_pap_pd_multiturn.py \
  tests/benchmarks/test_pap_multiturn_mps_contract.py \
  tests/benchmarks/test_pap_multiturn_common.py \
  tests/benchmarks/test_pap_pd_multiturn_load_client.py \
  tests/benchmarks/test_pd_multiturn_load_reuse_metrics.py \
  tests/benchmarks/test_pd_multiturn_runner_contract.py
```

All current launch contracts target project-owned entry points under
`benchmarks/pap/` and `examples/pap/`; PAP tests do not depend on a repository
skill directory.

Prefer behavior and structured-config tests over source-text assertions. Keep
shell tests for syntax and end-to-end assembly only; keep CUDA-dependent tests
explicitly marked so CPU-only environments skip them clearly.

## Ownership and redundancy audit

Test filenames follow the current owner modules (`attention`, `gateway`,
`integration`, `kv`, `protocol`, `topology`, `transport`, and `lifecycle`), not
retired top-level façade or example-script names. vLLM integration parsing and
adapter behavior belongs in `contract/test_vllm_integration.py`; generic vLLM
tests retain only the scheduling or execution behavior that crosses the seam.
An exact AST-body audit is sufficient for mechanical duplication; broader
similarity is reviewed by behavior because different failure transitions can
have intentionally similar setup.

The post-façade audit removed the two exact duplicate cases it found. The
remaining xPAyP and NIXL unit contracts are retained because those capabilities
are supported, while their E2E lanes remain outside the P17 refactor gate.

The current support and evidence boundary is summarized in the
[PAP development status](../../docs/design/pap/status.md).
