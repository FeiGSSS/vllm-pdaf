# Dynamo-managed baselines and PAP comparison

## Decision

The default non-PAP baselines are now managed entirely by Dynamo. Native vLLM
DP8 and the project-local PD proxy are retired. Both baselines use the same
Dynamo frontend and KV-aware router; `DYNAMO_ARCHITECTURE=dp8` versus
`DYNAMO_ARCHITECTURE=6p2d` is the serving-architecture choice.

The active launcher is:

```bash
DYNAMO_ARCHITECTURE=dp8 \
  bash benchmarks/pap/scripts/run_dynamo_workload.sh
DYNAMO_ARCHITECTURE=6p2d \
  bash benchmarks/pap/scripts/run_dynamo_workload.sh
```

## Fixed protocol

- server stack: Dynamo 1.4.1 and upstream vLLM 0.26.0;
- model: Qwen3-8B FP16;
- context limit: 131,072 tokens with official static YaRN factor 4;
- execution: piecewise CUDA Graphs in every worker;
- router: Dynamo KV-aware router behind one frontend;
- GPUs: eight NVIDIA L20 GPUs;
- workload: fixed Agentic Coding subset, 128 complete conversations and 1,630
  turns;
- arrivals: 2 turns/s, Poisson, concurrency cap 64, no authored turn delay;
- dataset SHA-256:
  `70e852e8efcdc84e684f5b4a4323f3cbb11b53322dc052fe0b34b0ffedd316d6`;
- source commit recorded by the three full runs:
  `81d92740aed5035fc86a52ea7ed9205b697bf54e`.
- executed unified-launcher SHA-256:
  `9356692f045951498d746a2267a63d214b3447c9f988e7d271d554d62f86e806`.

DP8 uses eight aggregated Prefill+Decode workers and
`max_num_batched_tokens=32768`. 6P2D uses six Prefill workers and two Decode
workers, both with `max_num_batched_tokens=2048`, plus strict same-node
NIXL/UCX KV transfer. All workers use `max_num_seqs=256`.

PAP uses seven colocated Prefill/Attention instances and one Projection GPU,
static 80/12-SM MPS partitions, conversation-affinity routing without KV
relocation, and its whole-step Attention--Projection CUDA Graph.

All three runs contain the same 1,630 unique `(conversation_id, turn_index)`
keys. Their average measured input lengths differ by at most 0.112% and output
lengths by at most 0.254%, so the workload realization is aligned.

## Results

Lower is better for latency and duration; higher is better for throughput.

| Metric | Dynamo DP8 | Dynamo 6P2D | PAP 7PA1P |
| --- | ---: | ---: | ---: |
| Completed requests | 1,630 / 1,630 | 1,630 / 1,630 | 1,630 / 1,630 |
| Duration (s) | 971.41 | 1,594.67 | 1,460.46 |
| Request throughput (req/s) | 1.678 | 1.022 | 1.116 |
| Output throughput (token/s) | 419.73 | 255.39 | 278.47 |
| Mean TTFT (ms) | 2,044.83 | 27,806.11 | 9,790.54 |
| P50 TTFT (ms) | 1,130.24 | 29,922.71 | 845.45 |
| P99 TTFT (ms) | 12,227.02 | 66,786.90 | 70,929.76 |
| Mean ITL (ms) | 58.16 | 63.72 | 76.08 |
| P50 ITL (ms) | 48.81 | 64.71 | 77.97 |
| P90 ITL (ms) | 98.48 | 69.38 | 93.38 |
| P99 ITL (ms) | 178.09 | 76.70 | 98.46 |
| ITL standard deviation (ms) | 32.15 | 7.56 | 12.72 |
| Mean request latency (ms) | 16,254.56 | 43,638.61 | 28,817.13 |
| P99 request latency (ms) | 62,562.62 | 100,342.44 | 100,517.57 |

Both runs completed with zero request errors or cancellations, passed the
output-correctness audit, and passed the CUDA Graph audit. The 6P2D run also
passed the 5,000 MB/s fail-closed KV-transfer gate:

- 1,630 observed NIXL transfers;
- aggregate transfer throughput: 8,487.78 MB/s;
- weighted transfer time: 278.95 ms;
- weighted data per transfer: 2,367.66 MB.

The AIPerf result digests are:

- DP8 `profile.json`:
  `8e75febe926bd7aa6d718968b9375f3e964df92c41ef2f7b1ea6ef3e81bdbd8e`;
- 6P2D `profile.json`:
  `7d87e4d0499053a4ab533f1c5349b89ff246de22be6773288b06d2117c9f1a58`.
- PAP `profile.json`:
  `4b0b027d51f8a864301327378facd0b36c9539dbf54214a0f4713d6ec1f8e87d`.

## Interpretation

6P2D is Decode-capacity limited on this multi-turn, decode-heavy workload. Its
two Decode GPUs remained saturated while the router accumulated roughly
31--49 waiting requests and the six Prefill GPUs were frequently idle. The
request-level trace decomposes its mean 27.81-second TTFT into approximately:

- 1.96 seconds from routing through Prefill and handoff;
- 25.84 seconds after Decode registration until the first token.

The approximately 279 ms weighted NIXL transfer time is not the cause of the
27.81-second mean TTFT. The dominant cost is waiting for capacity on only two
Decode workers.

The separation still has a measurable benefit: 6P2D removes much of the rare
Prefill interference seen by aggregated DP8. Its P99 ITL is 56.93% lower and
its ITL standard deviation is 76.50% lower, although its median and mean ITL
are worse. This does not compensate for the severe TTFT and throughput loss at
the fixed 6P2D split.

PAP has the best median TTFT but severe PA-local tails. Static round-robin
conversation placement assigned heavy, overlapping long-context sessions to
PA0, PA3, and PA5. PA0 reached 99.6% KV usage, ended at a 7.1% prefix-cache hit
rate, and queued up to eight Prefill requests. PAP therefore lands between DP8
and 6P2D on throughput and mean TTFT, while its mean ITL is the worst of the
three. Its P99 ITL remains 44.7% below DP8 because separated Attention avoids
DP8's rare Prefill-interference spikes.

The full PAP run exposed 12 stale `decode_token_pending_kv` join entries after
all 1,630 requests completed. The cause was a teardown race: an old Attention
step could publish KV readiness after its session had been released and its
join state forgotten. The fix makes active-session epoch validation and KV
readiness registration atomic. A real 7PA1P validation using the first 32
sessions and 405 requests then passed strict correctness, routing, Graph,
gateway/session drain, and decode-token join audits, with every PA reporting
zero pending token, KV, dispatch, mismatch, and dispatch-failure entries. Its
`profile.json` SHA-256 is
`e16d00311e779e29ccb73e6ac2b96d4b3650c687ece9322052e2602e7decc491`.

These are single-run observations that establish the executable default
baseline. Repeated runs are still required before treating small differences
as paper-ready estimates.

## Artifacts

Full-run artifacts remain local because they occupy 412 MiB:

- `runs/dynamo_dp8`;
- `runs/dynamo_6p2d`.
- `runs/pap_7pa1p`.

The pending-KV fix validation occupies 25 MiB at
`../_staging/pap_pending_kv_fix_s32`.

Each run contains `effective_config.env`, AIPerf `profile.json` and
`profile.jsonl`, service logs, package versions, the captured worktree patch,
and correctness/Graph/KV-transfer audits.
