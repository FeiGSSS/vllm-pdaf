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

## Results (2026-10-16)

6 valid runs; all 4320 AIPerf requests joined to their Gateway profile lines
(request id = AIPerf `x_request_id`). Per-request phases:
`results/requests_phases.csv`; group means: `results/ttft_phases.csv`.
Profiling overhead: mean TTFT at C40 with profiling is 0.2% above without.

Mean phase times (ms) by TTFT group:

| Point | Group | TTFT | Prefill | Tokenize | Route | Migration | Readiness + payloads | Admission | First chunk |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C30 | P45–P55 | 3645 | 3412 | 29 | 13 | 2 | 6 | 0 | 139 |
| C30 | ≥ P95 | 8083 | 6777 | 254 | 119 | 0 | 4 | 0 | 127 |
| C30 | ≥ P99 | 10317 | 8114 | 167 | 44 | 0 | 4 | 0 | 108 |
| C40 | P45–P55 | 5670 | 5397 | 34 | 20 | 1 | 9 | 0 | 146 |
| C40 | ≥ P95 | 10837 | 9522 | 92 | 38 | 0 | 4 | 0 | 126 |
| C40 | ≥ P99 | 13762 | 11497 | 63 | 25 | 0 | 4 | 0 | 97 |

PA balance: requests per PA have a coefficient of variation of 1.2% (C30) and
2.9–3.5% (C40).

## Conclusion

- G1: PAP's TTFT, median and tail alike, is the Prefill phase (79–95% of
  TTFT): time from dispatch to a PA until its Prefill finishes, i.e. queueing
  plus compute on the PA. Routing, migration, KV readiness, Projection
  admission and the first Decode chunk together stay below about 200 ms.
- The tail is not caused by PA imbalance (request counts per PA within ±4%).
  Gateway tokenization is larger for tail requests at C30 (254 vs 29 ms),
  suggesting Gateway CPU contention, but it is a small share.
- The Gateway lines cannot split queueing from compute inside the Prefill
  phase; that needs PA-side scheduler timestamps (not in this request).
