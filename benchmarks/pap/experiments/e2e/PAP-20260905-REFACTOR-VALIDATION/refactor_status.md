# Refactor closeout

Closeout is complete under the user's confirmed verification scope. The full queue at
`runs/20260905_135804_3149938/` passed all seven cases and exited normally.
The subsequent model-interface cleanup passed 276 tests. Its duplicate full
queue at `runs/20260905_145710_3297423/` was stopped (exit 130) following the user's
objection to repeating all expensive tests. The three context cases already
passed and their stored audits have been checked. The accepted verification
basis is the existing 7/7 full-suite evidence plus the 276 tests and three
post-cleanup E2E cases, with distinct source snapshots. Do not restart the
duplicate queue or require another full replay for this limited cleanup.
Any newly proposed test requires explaining its minimum scope and obtaining
the user's approval first.

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

- Latest regression: `.venv/bin/python -m pytest tests/pap -q` with GPU access:
  **276 passed**, no skips after model-interface cleanup (2026-09-05).
- Same-node NIXL setup verification: UCX 1.22.0 library/plugin checks and agent
  creation passed. This is not a measured bandwidth or cross-machine RDMA test.
- Half-length 60-session replay: 180 requests completed, no errors or output
  length mismatches, successful Graph/lifecycle audits and drain.
- Refactored tracing replay: 180 requests completed; the automatic capture
  retained eight raw files and 512 consecutive aligned steps. Hash verification
  and recomputation reproduced all 35 tensor fields exactly.

The passed E2E queue includes the dependency correction and strict drain guards,
but precedes the final model-interface cleanup. The post-cleanup tests and three
context E2E cases cover that limited change; this is not a claim that the full
matrix ran to completion on the later snapshot.

## Requirement-to-evidence audit

| Requirement | Current state |
| --- | --- |
| Experiment organization and reproduction | Immutable dataset registry; E2E/microbench separation; per-run source/config/package/hardware/model/dependency snapshots; runtime setup and missing-provenance limits documented in `scripts/RUNTIMES.md` and experiment READMEs |
| Obsolete code and experiment cleanup | Commits `c9d8c0da5b` and `9aca8d96ef` removed retired drivers/Graph adapters; retained historical evidence is explicitly classified in `experiments/README.md`, not advertised as current launch instructions |
| PAP modularity and configuration | Gateway/integration/model/KV/transport boundaries retained; NVSHMEM protocol and tracing extracted; strict parsing and topology guards unified; native vLLM implementation unchanged in this closeout |
| Long-context fixtures with current Dynamo routing | All three sharing-enabled fixtures passed; old salted controls remain immutable and are not current default workloads |
| Unhalved 60-session dataset with valid live load accounting | Fresh 180-request replay passed, including four requests over 300 seconds with no premature expiry or release failures |
| Full dataset timed replay and cancellation/drain | Passed: 255 sent, 195 completed, 60 cancelled at 600 seconds; all PAP/native load ownership cleaned up |
| All active datasets and tracing | Full queue passed all seven cases; large dataset covered by its agreed 600-second protocol, not all 16,049 turns |
| Final model-interface cleanup | Tensor-only return and shared boolean parsing; 276 tests passed, then short-context and both long-context E2E cases passed; no further full replay required per user direction |

The user approved a PAP-only corrected Dynamo dependency and default cross-session
sharing on 2026-09-05. The isolated native selector now uses explicit-owner
reservations; the official DP/PD environment is unchanged. No timeout increase or
free/rebook heartbeat is used. The original salted dataset bytes and invalidated
results remain unchanged; new sharing-enabled fixtures have new IDs and hashes.

The short-context E2E checkpoint `runs/20260905_134349_3127559/` passed 14/14
requests, zero routing/lifecycle failures and zero remaining native reservations.
This early subset is supplemented by the full queue and post-cleanup checkpoints above.

See `results.md`, `reservation_lifetime.md`, the cache-salt microbenchmark, and
each run's invalidation note for measured evidence and reproduction clues.

## Model-interface audit resolution

After the seven-case queue finished, `model/projection.py` and
`model/attention_execution.py` were simplified: `execute` returns only its
output tensor; the empty release-message loop, unused direct-send switch/buffer,
unused timeline interface and redundant reset were removed. `_qkv_width`,
`scaling` and `num_hidden_layers` remain because the Graph manager consumes them.

The boolean readers in `model/hooks.py` and the Projection debug setting now use
`config.read_env_bool`, matching integration settings. Invalid values no longer
silently disable model hooks. Ruff, mypy and all 276 PAP tests passed.
The post-cleanup snapshot's 98 PAP runtime/build files were compared byte-for-byte
against the current working tree. Stored client records, audits and native load
snapshots for the three completed context cases were checked again during
closeout. No experiment or test was launched for this final record review.

## Handoff and limits

- No benchmark services or duplicate-run AIPerf workers remain active.
- Current changes remain uncommitted; this closeout does not claim a commit or push.
- Historical invalidated results remain invalidated. No performance improvement
  or bit-for-bit model-output equivalence is claimed from these correctness runs.
- Cross-machine RDMA, explicit salted Dynamo routing and automatic installation
  of every external runtime are outside this validated path; see `RUNTIMES.md`.
- Manual process-supervisor termination has a documented caveat in that file;
  successful timed request cancellation does not establish signal-forwarding
  correctness for externally killed launcher processes.
