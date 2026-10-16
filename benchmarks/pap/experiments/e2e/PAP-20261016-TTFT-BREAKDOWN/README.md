# E5: decomposing PAP's TTFT tail

Request: `paper/handoff/requests/R-004-phase3-ablations-tuning.md` (E5).

## Question

Which stages (routing, KV migration, Prefill queueing and compute, KV
readiness, Projection admission, first Decode step) make up PAP's TTFT, and
what makes the P99 requests slow (G1)?

## Protocol

- PAP 7PA1P, KV migration on, fixed PAT, `s240-t3-native32k-stratified-seed42`
  (720 requests), closed loop at C30 and at the E1 D2 knee C40.
- `profile-c30`, `profile-c40`: `PAP_PREFILL_IPC_PROFILE=1` (Gateway per-request
  phase profile lines) and `PAP_DYNAMO_ROUTING_TRACE_OUTPUT` (placement trace).
- `noprofile-c40`: same as `profile-c40` without profiling, to measure the
  profiling overhead (reported if above 3%).
- `run_round.sh <k>` runs the three variants once; two rounds (6 runs).
- `analyze.py` (to be added with the results) parses the Gateway lines named
  in R-004 and joins them to AIPerf records by request id.

## Results

Pending.
