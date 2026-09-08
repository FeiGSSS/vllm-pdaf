# Conversation-sticky round-robin PA placement

## Question

This experiment tests whether PAP can avoid PA migration by assigning each new
conversation round-robin and keeping all later turns on its initial PA. It also
compares that policy with the existing per-turn Dynamo selection under the same
single-host workload.

## Policy and failure contract

`session_round_robin` uses the explicit conversation ID already carried by the
Gateway request. A new conversation receives PA 0, 1, ... in rotation. Later
requests with the same conversation ID use the stored PA binding. The policy
fails before Prefill when the ID is absent, a request ID is duplicated, or the
configured conversation-binding capacity is full. It never evicts a binding,
because eviction would silently reintroduce migration.

The binding lasts for the Gateway process lifetime. PAP currently has no
conversation-end signal, so deployments must size
`PAP_SESSION_ROUTER_MAX_CONVERSATIONS` for the expected number of distinct
conversations or restart at an explicit maintenance boundary.

## Unit of work and expected timeline

A request is one conversation turn. A transition is turn 0 to 1 or turn 1 to 2
within one conversation. Each run therefore has 180 requests and 120
transitions.

For Dynamo, every turn enters the KV-aware selector independently. If it picks
a new PA, that PA can reuse only prefixes already present in its own vLLM cache;
PAP has no PA-to-PA KV relocation in this experiment. For session round-robin,
turn 0 establishes a binding and turns 1 and 2 return to that PA. This preserves
the conversation prefix while giving up per-turn load selection.

## Controlled runs

The accepted runs alternate policies to limit time-order bias:

1. `session_r2`
2. `dynamo_r1`
3. `session_r3`
4. `dynamo_r2`

All four use Qwen3-8B FP16 on the same eight L20 GPUs, 7PA1P, static 80/12-SM
Prefill/Attention MPS partitions, CUDA IPC within every PA, NVSHMEM P2P for the
Projection/Attention path, native 32K context, Prefill budget 2048, Projection
budget 256, 60 three-turn sessions, and AIPerf concurrency 30. Every run served
180/180 requests without client errors or output-length mismatches.

The analysis verifies the identical dataset-selection record, Git revision and
dirty-worktree fingerprint, effective configuration after removing run paths
and the intentional routing-policy field, run metadata, and process/device
topology. All service, CUDA-context, MPS, Graph, routing, lifecycle-drain, and
post-run GPU cleanup audits passed. The two policies are the only intentional
experimental variable.

## Aggregate result

The table pools the two runs for each policy, for 360 requests and 240
transitions per policy.

| Measurement | Dynamo | Session round-robin | Change |
| --- | ---: | ---: | ---: |
| PA migrations | 74 / 240 | 0 / 240 | -30.83 pp |
| Cache-read rate | 55.95% | 64.48% | +8.54 pp |
| Computed Prefill tokens | 2,971,060 | 2,395,411 | -19.38% |
| Mean Prefill residence | 5.166 s | 4.081 s | -21.00% |
| Mean TTFT | 5.692 s | 4.677 s | -17.83% |
| P95 TTFT | 9.668 s | 10.078 s | +4.24% |
| P99 TTFT | 11.874 s | 12.860 s | +8.30% |
| Mean ITL | 44.08 ms | 46.00 ms | +4.36% |
| Mean E2E latency | 9.267 s | 8.395 s | -9.41% |
| Request throughput | 2.798 req/s | 3.000 req/s | +7.21% |
| Output throughput | 228.47 tok/s | 244.93 tok/s | +7.21% |

The fixed-session policy improves mean TTFT and throughput, but it does not
dominate Dynamo: its P95/P99 TTFT and ITL are worse. Total request placement is
still balanced. Across two runs the session policy routes
54/54/54/54/48/48/48 requests to PA 0-6, while Dynamo routes
51/57/47/52/50/54/49. The difference is that Dynamo can react to instantaneous
load on every turn, whereas the sticky policy can balance only new
conversations.

## Where the TTFT difference appears

| Turn | Dynamo mean TTFT | Session RR mean TTFT | Change |
| --- | ---: | ---: | ---: |
| 0 | 6.274 s | 6.252 s | -0.35% |
| 1 | 5.532 s | 4.222 s | -23.68% |
| 2 | 5.269 s | 3.557 s | -32.50% |

Turn 0 has no prior conversation binding or reusable prior-turn KV and is
effectively unchanged. The benefit appears on turns 1 and 2, exactly where PA
migration can destroy conversation-local cache reuse.

Within the Dynamo runs, the 74 migrated transitions read 30.10% of prompt
tokens from cache and had 6.349-s mean TTFT. The 166 transitions that stayed on
their PA read 76.12% and had 4.978-s mean TTFT. This split is observational,
because Dynamo selects the target using both cache overlap and load, but it is
consistent with the controlled policy result. Session round-robin kept all 240
transitions and read 75.09% of their prompt tokens from cache.

The two policies processed prompt totals that differ by 33 tokens out of about
6.74 million. Model-generated text is inserted into later conversation turns,
so tokenized prompt length can vary slightly even though the replay template,
requested output lengths, and session order are identical. This 0.00049%
difference does not account for the 575,649-token computed-work reduction.

## Conclusion

Conversation-sticky round-robin is effective when the objective is to prevent
PA migration and reduce average Prefill work. For this C30 workload it removes
all observed migration, saves 19.4% of computed Prefill tokens, and lowers mean
TTFT by 17.8%. It also shows why TTFT changed even without a Prefill-KV network
transfer: selecting another PA makes that PA recompute conversation-prefix KV
before it can emit the first token.

The higher tail TTFT and ITL show the limit of permanent affinity. A busy PA
cannot shed an existing conversation. This makes session round-robin a useful
control and an optional operating mode, but not a complete replacement for
Dynamo. The next experiment should retain Dynamo's load-aware selection and
move reusable KV to the selected PA when the target changes.

## Invalid attempt

`20260909_session_round_robin_r1` stopped before GPU startup because the first
launcher version resolved the repository root one directory too shallow. The
launcher now derives the correct root and every accepted run uses that version.
The invalid attempt contributes no measurements.

## Evidence and replay

`results/archives/` contains the four accepted raw runs without the copied
Dynamo binary or unrelated dirty-worktree patch. It retains effective
configuration, source fingerprints, AIPerf records, service logs, routing and
topology audits, correctness checks, and cleanup evidence.

Reproduce the analysis from the archives:

```bash
.venv/bin/python \
  benchmarks/pap/experiments/e2e/PAP-20260909-SESSION-ROUND-ROBIN/verify.py
```

The verifier checks all saved digests, extracts the four runs, reruns the
analysis and comparability audit, and requires an equivalent summary plus a
byte-identical request table.
