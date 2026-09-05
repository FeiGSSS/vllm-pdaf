# Refactor status and remaining gates

The overall task is **not complete**. No inference experiment is currently
running; the last queue was stopped and GPU cleanup was verified.

## Completed local changes

- Removed the unreferenced legacy custom-op Graph adapter, its environment
  selectors, the two-rank experiment wrapper and an unused world-reset helper.
- Separated NVSHMEM wire encoding from transport, and moved trace allocation
  and export into a recorder. Kept wire version 3 and verified byte equivalence.
- Unified strict boolean/integer/finite-float parsing across the affected PAP
  configuration paths. NVSHMEM initialization uses one immutable timeout;
  malformed topology aliases, cache limits and timeout values fail explicitly.
- Removed launcher-time workload generation and obsolete source-AIPerf records.
  Replay now validates explicit dataset/session/request counts.
- Added source/environment/hardware/model snapshots, immutable new run folders,
  stricter resume checks and automatic freezing of reproducible trace windows.
- Added audit failures for incompatible Dynamo KV events and premature request
  reservation expiry. Invalid results retain their raw evidence and explanations.

Framework implementation changes stay under `vllm/pap/`, plus PAP benchmark
and test files. One shared pre-commit Shell checker was corrected to honor its
input file list and preserve a failure even when a later file passes. Native
vLLM engine/model implementation files are not edited.

## Verified evidence

- Latest host regression: `.venv/bin/python -m pytest tests/pap -q`:
  **265 passed**, no skips (2026-09-05).
- Same-node NIXL setup verification: UCX 1.22.0 library/plugin checks and agent
  creation passed. This is not a measured bandwidth or cross-machine RDMA test.
- Half-length 60-session replay: 180 requests completed, no errors or output
  length mismatches, successful Graph/lifecycle audits and drain.
- Refactored tracing replay: 180 requests completed; the automatic capture
  retained eight raw files and 512 consecutive aligned steps. Hash verification
  and recomputation reproduced all 35 tensor fields exactly.

The E2E checkpoints precede the last invalid-configuration guards; they do not
replace a final all-dataset run after the remaining dependency work.

## Required before completion

| Requirement | Current state |
| --- | --- |
| Three salted long-context fixtures with current Dynamo routing | Blocked by unsupported cache-salt isolation; preflight rejects the combination |
| Unhalved 60-session dataset with valid live load accounting | Invalidated: three live reservations expired before their requests completed |
| Full dataset timed replay and cancellation/drain | Stopped pending the reservation fix; no completed measurement |
| Final current-revision all-dataset E2E validation | Incomplete |

The requested decision is whether to build a project-local, PAP-only corrected
Dynamo routing dependency, leaving official DP/PD environments unchanged.
No such fork, dependency installation or source modification has been performed.
Neither removing cache salts, switching routing to bypass the issue, nor raising
an arbitrary lifetime has been used as a workaround.

See `results.md`, `reservation_lifetime.md`, the cache-salt microbenchmark, and
each run's invalidation note for measured evidence and reproduction clues.
