# Why Dynamo migrates PAP sessions more often across hosts

## Question

The native-32K, 60-session, three-turn, concurrency-30 PAP workload migrated
later turns between PA workers much more often after Prefill/Attention moved to
L20T. This experiment determines whether Dynamo saw the same prefix state and
chose a lower-load worker, or whether the previous PA's prefix was missing from
the selector view.

## Unit of work and timeline

A request is one conversation turn. A transition is the boundary from turn 0
to 1 or turn 1 to 2 in one conversation, giving 120 transitions per run. A PA
migration means Dynamo selected a different Prefill/Attention pair for the new
turn.

For each request, the diagnostic records these adjacent boundaries with the
same request and conversation IDs:

1. the next turn arrives about 35 ms after the client receives the previous
   response;
2. before selection, the Gateway queries every PA's indexed device-block
   overlap and current Prefill/Decode load;
3. Dynamo selects and reserves one PA;
4. that PA executes Prefill and returns its actual cache-read token count;
5. AIPerf records TTFT and request completion.

The all-candidate query is enabled only when
`PAP_DYNAMO_ROUTING_TRACE_OUTPUT` is set. It is absent from production and
ordinary benchmark paths.

## Runs

Both accepted runs used Qwen3-8B FP16, native `max_model_len=32768`, Prefill
budget 2048, Projection budget 256, 60 three-turn sessions, concurrency 30, and
dataset SHA256
`5d4357a9a037e9e1a3c25acf402ba06afd1c9e7643678da225b9f5cc63cb257c`.

| Run | Placement | Transport | Requests | Result |
| --- | --- | --- | ---: | --- |
| `samehost_trace_r3` | eight L20 GPUs on L20B | NVSHMEM P2P | 180 | passed |
| `crosshost_trace_r2` | Projection on L20B; seven PA workers on L20T | NVSHMEM IBGDA | 180 | passed |

Both runs passed service, CUDA context, process/device, MPS 80/12-SM, Graph,
request correctness, lifecycle drain, and GPU cleanup checks. The cross-host
run also passed exact participating-source checksum comparison. Its Git
revisions differed because the local cross-host baseline had already been
committed, so revisions are retained as provenance rather than used as a proxy
for source equality.

The trace query is deliberately classified as diagnostic overhead. Its mean
overlap/load query time was 22.77 ms on the same host and 32.94 ms across hosts;
P99 was 317.04 and 810.55 ms. TTFT values from these runs must not replace the
ordinary no-probe performance measurements.

## Result

| Measurement | Same host | Cross host |
| --- | ---: | ---: |
| PA migrations / 120 transitions | 33 | 82 |
| Migration rate | 27.5% | 68.3% |
| Previous full prefix visible to Dynamo | 117 / 120 | 20 / 120 |
| Previous-prefix visibility rate | 97.5% | 16.7% |
| Migrations with previous prefix invisible | 0 | 73 |
| Selected a maximum-overlap PA | 133 / 180 | 55 / 180 |
| Actual Prefill cache-read rate | 58.11% | 39.74% |
| Computed Prefill tokens | 1,412,752 | 2,032,077 |

The 33 same-host migrations all occurred while the previous PA had strictly
more indexed overlap than the selected PA. In those cases Dynamo made an
observable load-versus-locality tradeoff: the selected PA averaged 11,655 fewer
pending Prefill tokens and 452 fewer potential Decode blocks than the previous
PA.

The cross-host behavior is qualitatively different. Of 82 migrations, only
nine showed more overlap on the previous PA, eleven improved overlap, and 62
showed equal overlap. The previous PA's complete prior-turn prompt was absent
from Dynamo's indexed candidate state for 100 of 120 transitions. Once the
session-specific overlap disappeared, the selected PA averaged 12,471 fewer
pending Prefill tokens and 243 fewer potential Decode blocks, so load scoring
frequently moved the turn.

There is a direct counterexample to cache eviction as the explanation. Among
38 cross-host transitions that stayed on the same PA, Dynamo reported the prior
full prefix missing in 27 cases. The selected Prefill then reported an actual
cache hit covering that full prefix in all 27. The KV blocks existed on the PA;
they were absent only from the selector's view at selection time. The analogous
same-host counts were three indexed misses and zero actual full-prefix hits.

The client-to-next-selection gap was effectively identical: about 35 ms for
both visible and missing cases on both placements. Same-host event delivery and
indexing usually completed within this interval; cross-host delivery and
indexing usually did not.

## Conclusion

The measured root cause of the additional cross-host PA migrations is a missing
KV-event visibility contract at the multi-turn boundary. The Gateway allows the
next turn to enter Dynamo selection immediately after the previous response,
without establishing that the remote Prefill's cache events have reached the
selector index. On the same host this asynchronous path usually catches up
within the roughly 35-ms turn gap. Across hosts it does not, so Dynamo sees a
much weaker or tied locality signal and its load term selects another PA.

This result does not distinguish network delivery time from subscriber or
indexer scheduling inside the native Dynamo dependency. That narrower mechanism
would require timestamps inside the publisher and native subscriber. It is not
needed to establish the violated system contract: at selection time the cache
exists, while the selector cannot see it.

The 43.84% increase in computed Prefill tokens and lower cache-read rate explain
the Prefill-residence and TTFT increase observed in the uninstrumented runs.
They are not Prefill KV payload-transfer or NVSHMEM first-token costs.

## Invalid attempts and permanent fixes

- `samehost_trace_r1` exposed a startup cycle: the runner waited for an
  NVSHMEM-ready Attention endpoint before starting the Projection root rank.
  The runner now starts all world participants before enforcing readiness.
- `samehost_trace_r2` verified the trace format but used a temporary health
  semantic while the startup cycle was being isolated. It is superseded by r3.
- `crosshost_trace_r1` stopped before GPU startup because Git revisions differed
  despite matching participating files. The source audit now records both
  revisions and accepts only exact per-file checksum equality.

No failed or superseded run contributes to the tables above. Their raw records
remain under `experiments/_staging/`.

## Evidence and replay

`results/archives/` contains the accepted raw runs. The same-host archive omits
the copied Dynamo binary and unrelated dirty-worktree patch; it retains the
requested/effective configuration, runtime audits, routing trace, AIPerf
records, service logs, correctness checks, and cleanup evidence. The cross-host
archive retains the corresponding local and remote records.

Reproduce the analysis from the archives:

```bash
.venv/bin/python \
  benchmarks/pap/experiments/e2e/PAP-20260909-DYNAMO-PA-MIGRATION-DIAGNOSIS/verify.py
```

`verify.py` checks every saved artifact digest, extracts both archives, reruns
`analyze.py`, and requires byte-identical request tables and equivalent summary
metrics.
