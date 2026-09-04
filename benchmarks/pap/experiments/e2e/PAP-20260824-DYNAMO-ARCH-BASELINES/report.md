# Dynamo-managed baselines and PAP comparison

## Decision

The default non-PAP baselines are now managed entirely by Dynamo. Native vLLM
DP8 and the project-local PD proxy are retired. All baselines use the same
Dynamo frontend and KV-aware router; `DYNAMO_ARCHITECTURE=dp8`, `6p2d`, or
`4p4d` is the serving-architecture choice.

The active launcher is:

```bash
DYNAMO_ARCHITECTURE=dp8 \
  bash benchmarks/pap/scripts/run_dynamo_workload.sh
DYNAMO_ARCHITECTURE=6p2d \
  bash benchmarks/pap/scripts/run_dynamo_workload.sh
DYNAMO_ARCHITECTURE=4p4d \
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
- current source commit for the two new runs:
  `563b804680b3d80c4bf8d9132ba9aaadceffcb51` (with the captured tracked
  worktree patches);
- executed 4P4D launcher SHA-256:
  `a65234f24bba9640079c919696cd22c69636b291b8421901f09ba2d2b5d9891e`;
- executed two-stage PAP launcher SHA-256:
  `cf47e5434bf541351a102b991208b226d9cb718d7d8ccb1425b3354d376805b5`.

DP8 uses eight aggregated Prefill+Decode workers and
`max_num_batched_tokens=32768`. The disaggregated baselines use either six
Prefill plus two Decode workers or four Prefill plus four Decode workers, all
with `max_num_batched_tokens=2048`, plus strict same-node NIXL/UCX KV transfer.
All workers use `max_num_seqs=256`.

PAP uses seven colocated Prefill/Attention instances and one Projection GPU,
static 80/12-SM MPS partitions, conversation-affinity routing without KV
relocation, and its whole-step Attention--Projection CUDA Graph.

All four current comparison runs contain the same 1,630 unique
`(conversation_id, turn_index)` keys. Their measured total prompt tokens differ
by at most 0.341% and completion tokens by at most 0.282%, so the workload
realization is aligned.

## Results

Lower is better for latency and duration; higher is better for throughput.

| Metric | Dynamo DP8 | Dynamo 6P2D | Dynamo 4P4D | PAP 7PA1P |
| --- | ---: | ---: | ---: | ---: |
| Completed requests | 1,630 / 1,630 | 1,630 / 1,630 | 1,630 / 1,630 | 1,630 / 1,630 |
| Duration (s) | 971.41 | 1,594.67 | 1,643.75 | 1,126.80 |
| Request throughput (req/s) | 1.678 | 1.022 | 0.992 | 1.447 |
| Output throughput (token/s) | 419.73 | 255.39 | 248.32 | 361.23 |
| Mean TTFT (ms) | 2,044.83 | 27,806.11 | 31,109.14 | 3,219.95 |
| P50 TTFT (ms) | 1,130.24 | 29,922.71 | 29,375.20 | 780.64 |
| P99 TTFT (ms) | 12,227.02 | 66,786.90 | 90,294.93 | 32,432.37 |
| Mean ITL (ms) | 58.16 | 63.72 | 37.23 | 76.11 |
| P50 ITL (ms) | 48.81 | 64.71 | 34.50 | 79.54 |
| P90 ITL (ms) | 98.48 | 69.38 | 50.01 | 91.93 |
| P99 ITL (ms) | 178.09 | 76.70 | 66.47 | 95.61 |
| ITL standard deviation (ms) | 32.15 | 7.56 | 9.32 | 13.94 |
| Mean request latency (ms) | 16,254.56 | 43,638.61 | 40,448.69 | 22,176.38 |
| P99 request latency (ms) | 62,562.62 | 100,342.44 | 99,753.03 | 90,154.01 |

All four runs completed with zero request errors or cancellations, passed the
output-correctness audit, and passed the CUDA Graph audit. Both disaggregated
runs also passed the 5,000 MB/s fail-closed KV-transfer gate. For 6P2D:

- 1,630 observed NIXL transfers;
- aggregate transfer throughput: 8,487.78 MB/s;
- weighted transfer time: 278.95 ms;
- weighted data per transfer: 2,367.66 MB.

For 4P4D:

- 1,630 observed NIXL transfers;
- aggregate transfer throughput: 9,989.87 MB/s;
- weighted transfer time: 214.62 ms;
- weighted data per transfer: 2,144.02 MB.

The AIPerf result digests are:

- DP8 `profile.json`:
  `8e75febe926bd7aa6d718968b9375f3e964df92c41ef2f7b1ea6ef3e81bdbd8e`;
- 6P2D `profile.json`:
  `7d87e4d0499053a4ab533f1c5349b89ff246de22be6773288b06d2117c9f1a58`.
- 4P4D `profile.json`:
  `bc4424dff9fc25431e252d309ec8ef8d9b6bebfd99fb67c58c1e5d30b2e9facf`.
- PAP round-robin `profile.json`:
  `4b0b027d51f8a864301327378facd0b36c9539dbf54214a0f4713d6ec1f8e87d`.
- PAP initial-context-balanced `profile.json`:
  `30b61d1db17bcd15defe17e1ada371b51718145bcc246f74d987b3f1004c8f3a`.
- PAP two-stage first-turn `profile.json`:
  `a27f79f18ce690c0fb5a2129cd3168875b4a9d779359b7a779295dc05ce2bb3f`.

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

4P4D moves the bottleneck to the opposite side. Its four Decode workers reduce
mean ITL from 63.72 to 37.23 ms, and Decode registration to first token takes
only 367 ms on average. However, its four Prefill workers queue 13--20 requests
each; mean Prefill service time is 30.74 seconds and accounts for 98.8% of mean
TTFT. NIXL transfer takes only 215 ms at 9.99 GB/s, so communication is again
not the cause. On this workload, neither fixed PD split is balanced: 6P2D is
Decode-bound, while 4P4D is Prefill-bound. A 5P3D point is the natural next
split to test if optimizing the Dynamo PD baseline further.

The original PAP run had the best median TTFT but severe PA-local tails. Static
round-robin conversation placement assigned heavy, overlapping long-context
sessions to PA0, PA3, and PA5. PA0 reached 99.6% KV usage, ended at a 7.1%
prefix-cache hit rate, and queued up to eight Prefill requests.

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

## First-turn placement balance

The first routing experiment balanced only a conversation's initial placement.
It counted request-text characters as a tokenizer-free context estimate,
selected the PA with the smallest accumulated initial context, and kept every
later turn sticky on that PA. On this workload, character count correlates
0.99996 with first-turn prompt tokens.

| Metric | Round-robin | Initial balance | Change |
| --- | ---: | ---: | ---: |
| Initial-context max / mean | 1.342 | 1.047 | -22.0% |
| Duration (s) | 1,460.46 | 1,302.30 | -10.83% |
| Request throughput (req/s) | 1.116 | 1.252 | +12.14% |
| Mean TTFT (ms) | 9,790.54 | 5,402.73 | -44.82% |
| P50 TTFT (ms) | 845.45 | 750.25 | -11.26% |
| P95 TTFT (ms) | 56,317.37 | 38,769.38 | -31.16% |
| P99 TTFT (ms) | 70,929.76 | 54,006.87 | -23.86% |
| Mean ITL (ms) | 76.08 | 73.32 | -3.63% |
| P99 ITL (ms) | 98.46 | 92.43 | -6.12% |
| Mean request latency (ms) | 28,817.13 | 23,623.15 | -18.02% |

The optimization is material but incomplete. It eliminated the original early
PA0/PA3/PA5 concentration, yet future turns were unknown at placement time and
later concentrated on PA2. The final conversation counts range from 11 to 23
per PA by design, while accumulated first-turn characters remain within 4.7%
of their mean. The run completed 1,630/1,630 requests and passed strict
correctness, routing, Graph, gateway/session drain, and decode-token join
audits.

## KV-resident-only first-turn routing

A follow-up run selected each new conversation using the Prefill block pool's
non-evictable KV token capacity. This accurately combined KV currently owned
by active Prefill requests and KV leased to Attention, while excluding
evictable historical prefix-cache blocks. It did not include requests still
waiting for their first Prefill scheduling step, because those requests had
not allocated KV blocks yet.

| Metric | Initial balance | KV-resident only | Change |
| --- | ---: | ---: | ---: |
| Duration (s) | 1,302.30 | 1,612.46 | +23.82% |
| Request throughput (req/s) | 1.252 | 1.011 | -19.24% |
| Output throughput (token/s) | 312.02 | 252.03 | -19.23% |
| Mean TTFT (ms) | 5,402.73 | 13,696.04 | +153.50% |
| P50 TTFT (ms) | 750.25 | 934.86 | +24.61% |
| P90 TTFT (ms) | 20,994.85 | 57,886.35 | +175.72% |
| P95 TTFT (ms) | 38,769.38 | 67,416.31 | +73.89% |
| P99 TTFT (ms) | 54,006.87 | 85,327.65 | +57.99% |
| Mean ITL (ms) | 73.32 | 72.14 | -1.61% |
| P99 ITL (ms) | 92.43 | 93.94 | +1.63% |
| Mean request latency (ms) | 23,623.15 | 31,783.63 | +34.54% |

All 1,630 workload keys matched the initial-balance run. Aggregate input
tokens differed by 0.12%, and aggregate output tokens differed by 0.01%.
The router assigned 31 conversations and 365 turns to PA2 but only five
conversations and 82 turns to PA4. Thus, resident KV remained a useful Decode
capacity signal, but the missing Prefill backlog caused severe queue herding.
The small mean-ITL improvement did not compensate for the TTFT and throughput
regressions.

This result motivated adding non-evictable KV, remaining tokens of running
Prefills, prompt tokens of waiting/skipped-waiting Prefills, and a short
Gateway reservation for requests selected but not yet visible to the Prefill
scheduler.

The AIPerf profile completed 1,630/1,630 records. The outer runner was
interrupted before it wrote the post-run topology and correctness audit files,
so this run is performance evidence, not a complete correctness checkpoint.
Its `profile.json` SHA-256 is
`66c6048d833892739418a0115b2e8c0b38a3a387f52a588f24dc73e73758568d`.

## Two-stage first-turn routing

The production first-turn router now separates capacity from compute:

1. It rejects any PA whose projected non-evictable KV, outstanding Prefill
   tokens, requested Decode reservation, and short Gateway reservations would
   exceed the physical block capacity with 4,096 tokens of headroom.
2. Among the capacity-safe PAs, it minimizes outstanding Prefill tokens plus
   the incoming first-turn prompt estimate. Projected KV, accumulated initial
   context, conversation count, and PA index are deterministic tie-breakers.

Only the first turn is placed this way. Later turns remain sticky, and this
stage performs no KV migration. The Gateway queries all seven Prefill workers
concurrently and reserves a choice for two seconds so simultaneous first turns
cannot herd onto a PA before the vLLM scheduler observes them.

| Metric | Initial balance | Two-stage router | Change |
| --- | ---: | ---: | ---: |
| Duration (s) | 1,302.30 | 1,126.80 | -13.48% |
| Request throughput (req/s) | 1.252 | 1.447 | +15.58% |
| Output throughput (token/s) | 312.02 | 361.23 | +15.77% |
| Mean TTFT (ms) | 5,402.73 | 3,219.95 | -40.40% |
| P50 TTFT (ms) | 750.25 | 780.64 | +4.05% |
| P90 TTFT (ms) | 20,994.85 | 8,404.41 | -59.97% |
| P95 TTFT (ms) | 38,769.38 | 20,011.26 | -48.38% |
| P99 TTFT (ms) | 54,006.87 | 32,432.37 | -39.95% |
| Mean ITL (ms) | 73.32 | 76.11 | +3.81% |
| P99 ITL (ms) | 92.43 | 95.61 | +3.43% |
| Mean request latency (ms) | 23,623.15 | 22,176.38 | -6.12% |
| P99 request latency (ms) | 86,551.78 | 90,154.01 | +4.16% |

The run completed 1,630/1,630 requests with zero errors. It passed strict
correctness, routing, eight-process whole-step Graph, seven-process Prefill
Graph, Gateway/session drain, and Decode-token join audits. The request keys
match the initial-balance run exactly; prompt and completion token totals differ
by only 0.111% and 0.169%. The capacity gate delayed three first turns and
reported no load-query failures.

The gain is primarily a Prefill-tail improvement, not a Decode optimization.
Mean and tail TTFT fall substantially, while ITL regresses by roughly 3--4% and
P99 request latency is not improved. Later turns still cannot be relocated, so
their Decode/KV load remains determined by the one-time first-turn choice.
The formal run's `profile.json` SHA-256 is
`a27f79f18ce690c0fb5a2129cd3168875b4a9d779359b7a779295dc05ce2bb3f`.

These are single-run observations that establish the executable default
baseline. Repeated runs are still required before treating small differences
as paper-ready estimates.

## Artifacts

Full-run artifacts remain local; the run directory currently occupies 795 MiB:

- `runs/dynamo_dp8`;
- `runs/dynamo_6p2d`.
- `runs/dynamo_4p4d`.
- `runs/pap_7pa1p`.
- `runs/pap_7pa1p_initial_balance`.
- `runs/pap_7pa1p_kv_load_balance_v2` (AIPerf complete; post-run audits absent).
- `runs/pap_7pa1p_two_stage_first_turn_v3`.

The pending-KV fix validation occupies 25 MiB at
`../_staging/pap_pending_kv_fix_s32`.

Each run contains `effective_config.env`, AIPerf `profile.json` and
`profile.jsonl`, service logs, package versions, the captured worktree patch,
and correctness/Graph/KV-transfer audits.
