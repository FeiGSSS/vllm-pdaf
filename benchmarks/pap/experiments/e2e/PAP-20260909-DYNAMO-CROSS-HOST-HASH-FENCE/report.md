# Cross-host Dynamo prefix consistency and KV fence

## Question and units

This experiment asks why Dynamo selects different PA workers much more often
after PAP is split across two hosts, and whether a read-after-write fence is
needed after repairing that behavior.

A **request** is one model turn; the workload has 60 conversations and three
turns per conversation, for 180 requests. A **transition** is a later turn with
a completed prior turn in the same conversation, so there are 120 transitions.
A **PA migration** means that Dynamo selected a different PA on a transition;
it does not imply that KV data was copied. **Previous-prefix visibility** means
that Dynamo's overlap index reported enough blocks on the prior PA to cover the
prior turn's block-aligned prompt prefix at the next selection boundary.

## Root cause

vLLM hashes a prefix as a chain whose first parent is `NONE_HASH`. When
`PYTHONHASHSEED` is absent, `init_none_hash` initializes that value from
`os.urandom(32)`. The same-host runner already exported
`PYTHONHASHSEED=123`; the cross-host runner did not. The local Gateway and the
remote Prefill EngineCore processes therefore assigned different block hashes
to identical token blocks. Dynamo received KV events, but their keys could not
match hashes computed for a new request. Replaying a complete event stream
could not repair a key mismatch.

The fix exports `PYTHONHASHSEED=123` for every local and remote role. Before
starting a GPU process, the runner now hashes the same 16-token block on both
hosts and fails unless both 32-byte hashes are identical. The accepted runs
record
`2670ff7a4af1fb824c88637c1057d0fb9278ed6e0adef70aa31eee00f2c1e265`
on both hosts.

This supersedes the earlier inference that contiguous event/index lag was the
main cross-host cause. The controlled seed A/B test directly changes prefix
visibility from 11.7% to 99.2%; the earlier replay test changed neither the
hash seed nor the mismatch.

## Consistency fence

The optional fence tracks each conversation's previous PA. Before a later
selection, it asks that PA's allocator-safe Snapshot service to pin and report
the exact longest resident prefix. The Gateway also queries Dynamo's indexed
overlaps. It pins the prior PA only when the verified source prefix is longer
than every indexed prefix; otherwise Dynamo retains its normal load-aware
choice. The Snapshot lease stays live through Prefill and is then released
asynchronously.

This fence performs no KV copy. It guards the short interval in which the
source allocator contains a prefix that Dynamo has not indexed yet. Snapshot
miss, malformed metadata, release failure, overlapping turns in one
conversation, or a residual lease fails closed.

## Controlled C30 results

All accepted runs use Qwen3-8B FP16, 7 remote Prefill plus 7 remote Attention
roles on L20T, one local Projection, NVSHMEM IBGDA, 60 three-turn sessions,
closed-loop concurrency 30, block size 16, Prefill budget 2048, Projection
budget 256, and routing tracing. Each run completed 180/180 requests with no
client, lifecycle, output-length, Attention, Snapshot, or cleanup failure.

| Run | Seed | Fence | Prefix visible | PA migrations | Cache read | Computed tokens | Mean TTFT | P99 TTFT | Req/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old replay r4 | random | off | 14/120 (11.7%) | 83/120 (69.2%) | 39.62% | 2,036,256 | 8371.52 ms | 16363.93 ms | 2.252 |
| Unseeded fence r7 | random | on | 18/120 (15.0%) | 87/120 (72.5%) | 38.20% | 2,083,947 | 8856.26 ms | 17138.87 ms | 2.219 |
| Hash-only r9 | 123 | off | **119/120 (99.2%)** | 40/120 (33.3%) | 56.13% | 1,479,144 | 5499.45 ms | 11265.09 ms | **2.951** |
| Hash + fence r8 | 123 | on | **119/120 (99.2%)** | **33/120 (27.5%)** | **56.82%** | **1,455,763** | **5409.15 ms** | **10706.20 ms** | 2.869 |

Fixing the hash contract, without the fence, raises visibility by 87.5
percentage points and removes 43 migrations. It reduces computed Prefill work
by 557,112 tokens (27.4%), mean TTFT by 2872.07 ms (34.3%), and raises request
throughput by 31.0% relative to r4. Mean ITL changes from 38.21 to 41.99 ms;
the gain is in Prefill/TTFT and is not an ITL improvement.

In r8, all 120 Snapshot checks exported a real prefix, with zero miss or
failure. Mean Gateway-visible export time was 17.78 ms. Only seven checks found
the source ahead of Dynamo and pinned it. Relative to the hash-only run, the
fence removes seven more migrations and 23,381 computed tokens, while mean
TTFT is 90.30 ms lower and throughput is 2.8% lower. These are independent
single service starts, so the mixed latency/throughput difference is not enough
to claim a performance gain from the fence. Its measured value is closing the
remaining consistency window.

All 40 migrations in the hash-only run occurred when Dynamo already knew that
the previous PA had more overlap. They are deliberate load-balancing overrides,
not stale-prefix decisions. Avoiding them entirely would require session
pinning and give up that load decision, or the Experiment 3 KV migration path
to copy reusable KV after Dynamo chooses a new PA. The hash fix does not absorb
that separate mechanism.

## Failed runs and durable guards

Two startup failures exposed an incomplete source-sync contract. In r5 the
remote `plugin.py` did not pass `model_executor` to `PAPEngineControl`. After
that file was added, r6 showed that the remote `PAPPrefillConnector` still
lacked `migration_catalog()`. The original checksum list covered only selected
files, so both incomplete combinations incorrectly passed.

The runner now checks every tracked file under `vllm/pap`, the shared block
pool and hash implementation, and all cross-host support scripts. It also
checks local and remote child liveness during HTTP readiness waits. Both failed
runs were stopped before AIPerf and cleaned to zero GPU processes; their raw
logs are retained as invalid evidence.

## Reproduction

`results/archives/` contains r4 through r9, including the two invalid startup
runs. `analyze.py` checks the workload, effective configuration, topology,
request identities, routing/Prefill traces, service health, source checksums,
hash audit, and cleanup before regenerating `results/summary.json` and
`results/requests.csv`.

```bash
.venv/bin/python \
  benchmarks/pap/experiments/e2e/PAP-20260909-DYNAMO-CROSS-HOST-HASH-FENCE/verify.py
```
