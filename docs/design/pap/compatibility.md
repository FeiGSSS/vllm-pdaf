---
pap_doc_schema: 1
status: current
canonical: null
superseded_by: null
related_experiments:
  - PAP-20260722-AIPERF-CANONICAL-CUTOVER
  - PAP-20260716-TRITON-72-20-BASELINE
  - PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH
  - PAP-20260715-VLLM-INTEGRATION-BOUNDARY
  - PAP-20260715-ARCHITECTURE-MILESTONE
  - PAP-20260715-RUNTIME-BOUNDARY-E2E
  - PAP-20260715-INTEGRATION-E2E
last_validated_commit: 894b81ae9238373ac0950fc7932bed7bfb3dd74c
---

# PAP compatibility retirement record

All P17 measurements below are frozen historical evidence. Current runtime
validation uses the AIPerf matrix described in
[benchmark-methodology.md](benchmark-methodology.md).

The post-P17 cleanup removed every top-level PAP compatibility façade after all
tracked runtime, launcher, test, and tool consumers moved to their owning
modules. Historical documents retain their original paths for traceability.

| Removed path | Current owner |
| --- | --- |
| `attention_executor.py` | `service.py`, `attention/`, `kv/`, `protocol/` |
| `attention_scheduler.py` | `attention/dispatcher.py` |
| `data_plane.py` | `protocol/`, `topology/`, `transport/` |
| `decode_commit_client.py` | `lifecycle/commit.py` |
| `decode_token_client.py` | `lifecycle/decode_token_client.py` |
| `deferred_decode_token.py` | `lifecycle/decode_token.py` |
| `kv_lease.py` | `lifecycle/lease.py` |
| `lease_release_client.py` | `lifecycle/lease_release.py` |
| `local_fast_transport.py` | `transport/local_fast.py` |
| `nixl_mailbox.py` | `transport/mailbox.py`, `transport/nixl.py` |
| `peer_activity.py` | `topology/peer_activity.py` |
| `remote_attention.py` | `protocol/wire.py` |
| `shadow_attention.py` | `attention/client.py`, `kv/handoff.py` |

The stable Attention process entry point is now `python -m vllm.pap.service`.
New top-level forwarding modules are not allowed: runtime code and tests import
the owning package directly. `remote_attention_diagnostics.py` and
`trace_summary.py` moved to `benchmarks/pap/tooling/`; their stable command
entry points remain `tools/pap_remote_attention_diagnostics.py` and
`tools/pap_trace_summary.py`.

## Retirement freeze evidence

The façade retirement was checked with the canonical P17 C4 workload before
the next internal module split. Three controlled repetitions completed 60/60
requests and passed client, cache, Attention stats, correctness, async
decode-token join, routing, commit, lease, static MPS, and session-drain gates.
The raw results remain under
`benchmarks/pap/experiments/legacy/runs/20260715_compat_facade_retirement_p17_c4_controlled/`.

The run used the same profile and implementation fingerprints as the tracked
P17 post-refactor formal baseline. Because the tracked worktree contained the
façade-removal patch, it is controlled evidence rather than a replacement
formal baseline.

| Metric | P17 formal baseline | Façade retirement | Change | Gate |
| --- | ---: | ---: | ---: | --- |
| R1 TTFT median | 10545.48 ms | 10536.77 ms | -0.08% | passed |
| R1 TPOT median | 39.256 ms | 39.273 ms | +0.04% | passed |
| R2-5 TTFT median | 203.78 ms | 204.47 ms | +0.34% | passed |
| R2-5 TPOT median | 50.398 ms | 50.482 ms | +0.17% | passed |

All four metrics remain below the 5% regression threshold. xPAyP and
cross-host NIXL stay preserved but unverified by this freeze.

## Architecture milestone evidence

The controlled retirement evidence was followed by a tracked-clean formal run
on commit `9fb642937d27f8871ce653216f8b70d64176679a`. Three repetitions again
completed 60/60 requests and 48/48 cache transitions with every strict gate
passing. The canonical record is
`PAP-20260715-ARCHITECTURE-MILESTONE`; raw evidence is under
`benchmarks/pap/experiments/PAP-20260715-ARCHITECTURE-MILESTONE/runs/20260715_9fb642937_pap_milestone_formal/raw/`.

Against the preceding post-refactor formal baseline, R1 TTFT changed by
-0.16%, R1 TPOT by -0.10%, steady TTFT by +1.98%, and steady TPOT by +0.06%.
The clean milestone therefore passes the 5% regression gate and supersedes the
controlled result as the release evidence for that stage.

## vLLM integration-boundary evidence

The next tracked-clean freeze moved PAP glue behind the owner-specific
`vllm/pap/integration/` adapters on commit
`9efa92dc60434cf5d5f171374bbacdb17fd3c449`. Its three P17 C4 repetitions
completed 60/60 requests and 48/48 cache transitions with all strict gates
passing. Compared with the architecture milestone, changes were -0.12% R1
TTFT, -0.02% R1 TPOT, -2.83% steady TTFT, and -0.15% steady TPOT. That record
is retained as the integration-boundary evidence and is superseded as the
performance baseline by `PAP-20260716-TRITON-72-20-BASELINE`.

## Current revalidation

At source milestone `894b81ae9`, none of the retired façades has returned.
Piecewise CUDA Graph support is owned by `vllm/pap/model/cudagraph.py` and the
existing `integration/` adapters; it did not add a new top-level forwarding
module. The current module boundary and supported execution modes are recorded
in the [development status](status.md).
