# Refactor validation checkpoints

## coding-half: passed before protocol/tracing extraction

Run: `runs/20260905_111034_2965031/coding-half/`.
Its exact source, dependency, hardware and model snapshots are in the suite's
`provenance/`. Later refactors still require their own end-to-end validation;
this checkpoint is not evidence for code changed after this run.

Configuration: 7PA1P, Dynamo routing, 2K Prefill budget, Qwen3-8B FP16/YaRN 131K,
60 sessions / 180 sequential turns, concurrency 60, Poisson 0.9 req/s, no warmup
or measurement cutoff. Dataset SHA-256:
`258b72c85772c9d372f1b63ee0bf6d710f27cb00234027e2c750c82a5fa9563c`.

| Evidence | Result |
| --- | --- |
| Client records | 180, no errors |
| Output-length mismatches | 0 |
| Total output tokens | 84,155 |
| Measured duration including replay drain | 291.445 seconds |
| Output throughput | 288.751 token/s |
| Mean TTFT | 1,616.764 ms |
| Mean request-level ITL | 43.277 ms |
| Mean end-to-end request latency | 21,692.383 ms |
| Whole-step Graph instances | 8/8 captured, all 7 PA active |
| Routing / lifecycle / correctness audits | Passed |
| Final Attention sessions | All 7 PA drained |
| Dynamo hash mismatch / missing-parent errors | None observed |
| PAT rebuild / Triton selection counters (sum over PA) | 263 / 68 |

Sources: `aiperf/profile.json`, `aiperf/profile.jsonl`,
`pap_whole_step_graph_audit.env`, `routing_audit.json`,
`correctness_audit.env`, `attention_fast_path_stats_*.json`, and `launcher.log`.
This run demonstrates execution of both PAT and Triton, but is not a controlled
performance A/B against an earlier revision. The configured arrival rate and
observed completed-request throughput are different quantities.

## short-context: invalidated

Run: `runs/20260905_105242_2939264/short-context/`.
All 14 requests completed and drained, but salted prefixes caused Dynamo event
index errors that the original audit missed. The suite's `INVALIDATED.md` records
why this result is excluded. The dataset and routing were not altered to bypass
the issue. Salted-Dynamo support remains an unresolved compatibility boundary.

## coding-half-trace: execution passed, raw retention incomplete

Run: `runs/20260905_112723_2989779/coding-half-trace/`, after protocol/tracing
module extraction. All 180 requests completed with zero errors and zero
output-length mismatches; Graph/lifecycle audits and drain passed. A valid
intermediate join contains steps 3314–3825 and the expected `[512,36,7]` PA and
Attention tensors plus `[512,36]` Projection tensor.

Final raw-ring export became empty during drain, so the join cannot be reproduced
from those final raw files. This is a trace-retention failure, not an inference
success claim for the entire diagnostic case. See the suite's
`TRACE_CAPTURE_INCOMPLETE.md`. An automatic raw-window collector is now part of
the diagnostic driver and must be verified by rerunning the case.

## coding-half-trace with automatic capture: passed

Run: `runs/20260905_114511_3015679/coding-half-trace/`. All 180 requests completed
without errors or output-length mismatches; Graph/correctness/lifecycle audits
and drain passed. The automatic collector froze steps 1861–2372 with two
`[512,36,7]` latency tensors and one `[512,36]` Projection tensor. All eight raw
file hashes matched the capture manifest, and replaying the join reproduced
all 35 saved tensor fields exactly. Raw files and `capture.json` are retained
under `trace_capture/`, independently of later rolling exports.

## coding: invalidated by active-reservation expiry

Run: `runs/20260905_114511_3015679/coding/`. All 180 inference requests completed,
but three live requests lost their native Dynamo load reservations before
completion. This result is excluded from valid load-accounting/performance
comparisons. See [reservation_lifetime.md](reservation_lifetime.md).

The queued `coding-full` case was stopped with exit 130 and has no completed
measurement. This suite has no overall success marker.
