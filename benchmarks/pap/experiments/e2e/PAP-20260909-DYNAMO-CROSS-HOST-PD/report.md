# Cross-host Dynamo 7P1D diagnosis

## Question and measurement contract

This experiment asks whether moving the seven Dynamo Prefill workers from the
Decode host to a second host changes later-turn Prefill placement, and whether
that explains TTFT. It uses Qwen3-8B FP16, the native-32K agentic-code trace,
60 three-turn sessions, and closed-loop client concurrency 30.

A **request** is one conversation turn. A **transition** is turn 0 to turn 1 or
turn 1 to turn 2 in one conversation. A **Prefill migration** occurs when the
selected Prefill worker ID changes across a transition. This is separate from
the normal PD **KV handoff**, which transfers the current request's Prefill KV
to Decode on every request.

The joined timeline uses one Dynamo request ID at adjacent boundaries:

1. Dynamo selects a Prefill worker.
2. That Prefill worker receives and completes the request.
3. Decode receives the request.
4. Dynamo reports the first token to the frontend.

The NIXL metrics separately measure transfer residence time and bytes. The
subtraction between aggregate timing buckets localizes time but does not prove
one mechanism by itself.

## Audited topology

- L20B (`10.128.7.24`): frontend, etcd, and Decode on physical GPU 7.
- L20T (`10.128.7.25`): Prefill workers on physical GPUs 0 through 6.
- Both hosts: NVIDIA L20, `ai-dynamo==1.4.1`, `vllm==0.26.0`.
- Dynamo request plane: TCP; event plane: ZMQ; router mode: KV.
- NIXL data plane: UCX 1.22, `tcp,cuda_copy`, one 1 Gb/s Ethernet link.
- AIPerf: 60 sessions, three turns each, concurrency 30, 180 requests.

The accepted `full4` run registered all eight workers and observed exactly one
model process on every requested GPU. It completed 180/180 requests with no
errors, cancellations, output-length mismatches, NIXL errors, lease expiry, or
residual processes. All 180 frontend selections, worker timelines, AIPerf
records, and NIXL transfers joined successfully.

## Result

| Metric | Same host, two runs | Cross host, one run | Change |
| --- | ---: | ---: | ---: |
| Prefill migrations | 63 / 240 | 1 / 120 | 26.25% → 0.83% |
| Estimated cache read | 55.56% | 62.40% | +6.84 pp |
| Computed Prefill tokens / request | 8,328.7 | 7,046.3 | -15.40% |
| Prefill service mean | 2.733 s | 2.226 s | -18.56% |
| TTFT mean | 8.888 s | 492.864 s | 55.45× |
| TTFT P95 | 13.654 s | 691.028 s | 50.61× |
| ITL mean | 66.967 ms | 27.398 ms | -59.09% |
| E2E mean | 14.280 s | 495.140 s | 34.67× |
| Request throughput | 1.824 req/s | 0.0546 req/s | -97.00% |

Cross-host Dynamo did **not** cause more Prefill migrations in this PD run.
Only one turn-1 request changed Prefill; all 60 turn-2 requests remained on the
same worker. Later-turn cache read was 69.85% for turn 1 and 78.58% for turn 2.
The one migrated request is insufficient for a causal latency comparison, but
the aggregate result rules out frequent Prefill migration as the source of the
cross-host TTFT increase.

The placement difference is consistent with the measured closed-loop timeline.
On the same host, requests return quickly enough that later turns arrive while
Prefill load is changing; Dynamo's cache and load terms produced a 40.0%
turn-1 migration rate and a 12.5% turn-2 migration rate. Across hosts, the
1 Gb/s handoff serializes progress after Prefill. Prefill workers are mostly
idle when later turns arrive, so the cached worker wins the same Dynamo score.
This is an inference supported by the selection, cache, Prefill service, and
arrival timelines; a fixed open-loop arrival trace would be needed to isolate
the routing feedback from the slower service time.

## Why TTFT grows

Cross-host mean router-to-Prefill time was 41 ms and mean Prefill service time
was 2.226 s. Mean time from Decode receipt to first token was 490.489 s, versus
6.153 s on the same host. Decode reached 30 waiting requests and 100% KV cache
usage during the cross-host run.

NIXL moved 368,001 MB over 180 transfers, or 2,044 MB per request. The run's
wall-clock KV volume rate was 111.71 MB/s, close to the 125 MB/s theoretical
maximum of the audited 1 Gb/s link. Individual transfers overlapped: their
mean residence time was 194.701 s and the residence-weighted rate was
10.50 MB/s. The remaining Decode-side interval includes transfer contention,
waiting/deferred queue time, KV allocation, and first-token execution; the
aggregate subtraction cannot separate those mechanisms further.

This explains the apparent TTFT contradiction. PAP does not require a
Prefill-KV transfer before first token, so its cross-host TTFT regression came
from changed PA placement and cache recomputation. Disaggregated PD always
hands Prefill KV to Decode. On this 1 Gb/s link that mandatory transfer and the
resulting Decode queue dominate TTFT even though Prefill placement and cache
reuse improve. ITL falls because the slow handoff staggers requests reaching
Decode, reducing simultaneous token generation; it is not an end-to-end gain.

## Failure found and fixed during bring-up

The first long run exposed a deployment defect. Remote workers used Dynamo's
default 10-second etcd lease while synchronous cross-host NIXL operations held
their event loops for much longer. All seven Prefill leases expired, the
workers unregistered, and the remaining 120 turns returned HTTP 503. The
failure archive contains the first lease-expiry events, 60 successful requests,
120 HTTP 503 records, and clean shutdown evidence.

The launcher now sets `ETCD_LEASE_TTL=3600` and NIXL
`kv_lease_duration=3600`, records both values in the effective configuration,
and fails on any lease expiry or premature KV-block release. The accepted run
then completed all 180 requests. These values protect this slow bare-metal TCP
experiment; production deployments should put etcd control traffic on a path
whose scheduling is independent of bulk KV traffic.

The cleanup path now stops workers before etcd and allows Dynamo's five-second
grace period before sending KILL. A final smoke completed with no shutdown
traceback, no endpoint-unregistration error, and zero residual GPU processes or
etcd listeners.

## Evidence

- `results/archives/cross_host_full4.tar.gz`: accepted raw run.
- `results/archives/cross_host_full2_failure.tar.gz`: lease-expiry failure.
- `results/summary.json`: recomputed placement and performance comparison.
- `results/requests.csv`: per-request worker, turn, cache, and timing join.
- `verify.py`: archive, topology, correctness, workload, and summary replay.

The same-host controls are the two immutable archives in
`PAP-20260907-ALIGNED-C30-7P1D`. The verifier checks their existing manifests
and reruns the same analysis code over both controls and the accepted
cross-host archive.
