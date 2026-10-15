# A5/A6: multi-turn placement, three ways

Request: `paper/handoff/requests/R-004-phase3-ablations-tuning.md` (A5/A6).

## Question

With conversation KV pinned in the PA that served the previous turn, does
"Dynamo load-aware routing + PA→PA KV migration" give lower TTFT than Dynamo
routing alone (recompute on a PA change) and than sticky session
round-robin (always return to the KV holder)?

## Protocol

- PAP 7PA1P, fixed PAT, `s240-t3-native32k-stratified-seed42` (720
  requests), closed-loop C30 and C50 (`pap.env`).
- `migration`: current source, Dynamo routing, KV migration on.
- `dynamo-only`: current source, Dynamo routing, KV migration off.
- `sticky-rr`: commit `c565ecb904` (the last one with session round-robin)
  in `.worktrees/rr-base`, imported through `PYTHONPATH`, with its ABI-10
  NVSHMEM bridge in `.local/nvshmem-3.3.24-cuda13-bridge-v10` (built
  2026-10-15 from that worktree), `PAP_ROUTING_POLICY=session_round_robin`,
  migration off (that runner requires Dynamo routing for migration). It shares
  the fixed PAT installed in `.venv`. **Not a single-variable change**: the
  code version differs; only routing is changed intentionally.
- `run_round.sh <k>` runs all six points once; three rounds (18 runs).

## Results

Pending.
