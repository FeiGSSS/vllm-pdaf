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

## Results (2026-10-16)

18 valid runs, three per point. `sticky-rr-c30_r1` failed in the plugin
preflight because the worktree lacked vLLM's compiled extensions (untracked
build artifacts); they were symlinked from the main tree, and the point was
rerun as `sticky-rr-c30_r4`.

| Variant | C | TTFT mean (ms) | ITL mean (ms) | req/s | D2 attainment |
| --- | ---: | ---: | ---: | ---: | ---: |
| migration | 30 | 3847 | 49.44 | 3.611 | 99.4% |
| dynamo-only | 30 | 5349 | 45.86 | 3.146 | 98.7% |
| sticky-rr | 30 | 4213 | 47.79 | 3.504 | 99.1% |
| migration | 50 | 8077 | 52.74 | 3.751 | 80.0% |
| dynamo-only | 50 | 10766 | 46.35 | 3.187 | 36.8% |
| sticky-rr | 50 | 8418 | 51.90 | 3.383 | 74.4% |

Change relative to `migration` at the same load, 95% CI (latency:
request-level bootstrap by run; throughput and goodput: run-level, n = 3):

| Comparison | TTFT | Throughput | D2 goodput |
| --- | --- | --- | --- |
| dynamo-only, C30 | +39.1% [+35.8, +42.3] | −12.9% [−13.4, −12.3] | −13.5% [−14.4, −12.6] |
| sticky-rr, C30 | +9.5% [+6.6, +12.5] | −3.0% [−3.8, −2.1] | −3.3% [−4.4, −2.1] |
| dynamo-only, C50 | +33.3% [+31.2, +35.4] | −15.0% [−15.8, −14.1] | −60.9% [−62.9, −57.9] |
| sticky-rr, C50 | +4.2% [+1.4, +7.2] | −9.8% [−27.4, −0.4] | −16.1% [−32.0, −6.7] |

## Conclusion

C19 holds: at equal load, Dynamo routing with PA→PA KV migration gives lower
TTFT and higher throughput than both alternatives, with every CI excluding
zero. Without migration, Dynamo's load-aware PA changes force prefix
recomputation, which costs 33–39% TTFT and 13–15% throughput (and 61% of D2
goodput at C50). Sticky session round-robin avoids recomputation but cannot
move a session off a loaded PA: it is 4–10% slower in TTFT and loses 3% (C30)
to 10% (C50) throughput. `sticky-rr` runs older code (`c565ecb904`); only
routing was changed intentionally, so it is not a single-variable comparison.
