# Cross-host NVSHMEM IBGDA whole-step Graph bring-up

## Outcome

The current PAP whole-step CUDA Graph protocol now runs across L20B and L20T.
Projection and Gateway run on L20B; Prefill and Attention run on L20T. The PA
keeps same-host CUDA IPC for Prefill KV while Projection--Attention QKV/output
traffic uses NVSHMEM IBGDA over `mlx5_0:1`.

The transport was first isolated with `1PA1P`, then validated at `7PA1P` with
the native-32K 60-session, concurrency-30 AIPerf workload.

## Root causes repaired

1. The current world exchanged its NVSHMEM UID through a local file and the
   launcher started every PE locally. Cross-host mode now uses a bounded TCP UID
   exchange with rank validation.
2. `configure_nvshmem.sh` forced `NVSHMEM_REMOTE_TRANSPORT=none` and disabled
   IBGDA. The `ibgda` mode verifies the compiled capability, driver
   `PeerMappingOverride=1`, and HCA before enabling the GPU NIC handler.
3. QKV scratch, Attention output, and control-message sources were ordinary
   PyTorch CUDA allocations. NVSHMEM device/stream puts require symmetric or
   registered sources across InfiniBand. Cross-host mode now owns symmetric
   staging for all three paths.
4. L20T rebuilt the device bridge with the CUDA 13 Python-package NVCC but the
   linker searched only the system CUDA 12.6 libraries. The build now adds the
   selected NVCC package's sibling `lib` directory explicitly.
5. The first service launcher omitted
   `PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS`, so the loaded endpoint plugin did
   not attach Prefill control routes. The launcher now enables the Prefill role
   and audits its routes before starting Gateway.
6. The transport constructor created a CUDA stream and device control buffer
   while its background thread was still initializing NVSHMEM. A fresh process
   could lose the bootstrap peer during that race. Local CUDA resources are now
   created only after `world.wait_ready()` completes.

## Validation

Both hosts use NVIDIA driver 595.84. Their `mlx5_0/1` links were ACTIVE and
`/proc/driver/nvidia/params` contained `PeerMappingOverride=1`.

The transport run at
`experiments/_staging/cross_host_nvshmem_graph/20260908_replay_contract_final/`
validated TCP
UID exchange, transport metadata, two control-plan generations, registered
staging, Graph-resident waits, and 100 round trips of a 16 KiB payload. Both
payload checks passed. NVSHMEM logged `NIC handler will be GPU` and successful
IBGDA initialization on both hosts. Three further independent 20-round-trip
process runs, `20260908_1541_stability_1` through `_3`, also passed.

The service run at
`experiments/_staging/cross_host_nvshmem_1pa1p/20260908_replay_contract_final/`
started from empty GPUs and recorded the effective configuration, matching
local/remote source checksums, and post-start mapping. One 12-token prompt
produced eight output tokens:

| Check | Result |
| --- | ---: |
| Client requests | 1 passed |
| Attention calls | 288 (`8 * 36`) |
| NVSHMEM peer batches | 8 |
| Decode tokens received / matched | 8 / 7 |
| Decode-token mismatches / dispatch failures | 0 / 0 |
| Slot-topology mismatches | 0 |
| Active Attention sessions after request | 0 |
| Dynamo failures / active reservations | 0 / 0 |
| Lifecycle completed / failed | 1 / 0 |

The seventh matched token is expected: the eighth sampled token is terminal and
is not consumed to form a subsequent KV entry. Prefill applied the seven-token
final commit and logged allocator reclamation. Both NVSHMEM PEs logged that
IBGDA was selected for device-side APIs and that the NIC handler ran on GPU.
The launcher cleaned all local and remote GPU processes after the run.

Run the same correctness lane from L20B with:

```bash
bash benchmarks/pap/scripts/run_cross_host_nvshmem_1pa1p.sh
```

The two hosts must contain the same source and Python environment. The launcher
fails before a request if GPUs are already occupied, PAP routes are absent,
the process/device counts disagree, or either IBGDA prerequisite is missing.

## 7PA1P AIPerf result

The accepted runs are
`experiments/_staging/cross_host_nvshmem_7pa1p_c30/20260908_cross_host_7pa1p_c30_r5/`
and `20260908_cross_host_7pa1p_c30_r6/`. They used Qwen3-8B FP16 with native
`max_model_len=32768`, Prefill token budget
2048, Projection token budget 256, and maximum 256 sequences. AIPerf streamed
the immutable
`s60-t3-native32k-stratified-seed42/dataset.jsonl` dataset at 60 sessions and
concurrency 30. The dataset SHA256 was
`5d4357a9a037e9e1a3c25acf402ba06afd1c9e7643678da225b9f5cc63cb257c`.

NVSHMEM 3.3 requires the same number of PEs on every node. The validated world
therefore had 14 PEs: Projection rank 0 and six idle ranks on L20B GPUs 0--6,
and seven Attention ranks on L20T GPUs 0--6. Each remote GPU also hosted its
Prefill through static MPS. The startup audit observed 80 SM for every Prefill,
12 SM for every Attention, and the expected PID/GPU mapping for Projection and
all idle ranks. This consumes 14 physical GPUs even though the service topology
is 7PA1P; comparisons with the same-host eight-GPU lane are allocation-asymmetric.

| AIPerf metric | r5 | r6 | Two-run result |
| --- | ---: | ---: | ---: |
| Completed requests | 180 / 180 | 180 / 180 | 360 / 360 |
| Benchmark duration | 79.746 s | 83.388 s | 163.135 s total |
| Input tokens | 3,372,318 | 3,372,595 | 6,744,913 |
| Output tokens | 14,696 | 14,696 | 29,392 |
| TTFT mean | 8.337 s | 8.750 s | 8.544 s |
| TTFT P99 | 15.374 s | 16.010 s | 15.689 s |
| ITL mean | 38.344 ms | 38.056 ms | 38.200 ms |
| ITL P99 | 45.401 ms | 44.859 ms | 45.435 ms |
| E2E mean | 11.435 s | 11.837 s | 11.636 s |
| E2E P99 | 18.975 s | 19.360 s | 19.045 s |
| Request throughput | 2.257/s | 2.159/s | 2.207/s |
| Output throughput | 184.284 token/s | 176.236 token/s | 180.170 token/s |

Here ITL is AIPerf's per-request mean interval between streamed output tokens;
its P99 is the percentile across 180 request-level ITL values. It is neither a
single GPU step time nor the interval between completed requests.

All seven Attention workers processed traffic in both runs. Their peer-batch
counts were `[1401, 1398, 1617, 1499, 1385, 1421, 1639]` in r5 and
`[1523, 1662, 1598, 1432, 1341, 1404, 1456]` in r6. Both runs recorded zero
output length mismatches, decode-token mismatches, dispatch failures, and
slot-topology mismatches. Gateway lifecycle completed all 180 requests per run
with no failures, all leases and routes drained, and both final cleanup audits
found zero local and remote GPU processes.

For context, the prior same-host 7PA1P C30 lane used the same dataset, AIPerf
phase, model length, output lengths, and scheduling budgets. Its two-run means
were TTFT 5.364 s, ITL 44.850 ms, E2E 8.993 s, aggregate request throughput
2.888/s, and aggregate output throughput 235.760 token/s. Relative to those
two-run values, the cross-host result had 14.83% lower mean ITL, 59.27% higher
mean TTFT, 29.39% higher mean E2E, and 23.58% lower aggregate throughput. Treat
these as observed configuration differences: the cross-host lane uses 14
allocated GPUs and includes the subsequent IBGDA implementation changes.

### TTFT localization

The higher TTFT is not evidence of Prefill KV payload transfer. Prefill and
Attention remained colocated and used CUDA IPC. Joining every AIPerf request ID
to the Gateway's `prefill_ms` boundary localized the two-run difference:

| Request-level mean | Same-host | Cross-host | Cross-host difference |
| --- | ---: | ---: | ---: |
| Gateway Prefill HTTP span | 4.828 s | 8.077 s | +3.249 s |
| TTFT minus Prefill span | 0.537 s | 0.467 s | -0.070 s |
| TTFT | 5.364 s | 8.544 s | +3.179 s |

Thus the measured increase occurs before Prefill returns. The post-Prefill
interval, which contains Attention readiness, Projection admission, the first
Projection--Attention execution step, and external delivery, did not increase.
The subtraction only localizes the interval; it does not split those remaining
operations individually.

The Prefill work was not equivalent across the runs. Server-reported cache-read
tokens fell from 3,897,712 of 6,743,677 prompt tokens (57.80%) on the same host
to 2,671,744 of 6,744,913 (39.61%) across hosts. Consequently, computed Prefill
tokens increased from 2,845,965 to 4,073,169, or 43.12%. The difference grows by
turn: the cross-host Prefill span was about 27% longer on turn 0, 55% on turn 1,
and 120% on turn 2.

This came with different routing locality. Across the two runs, 174 of 240
same-host turn transitions stayed on the previous PA, compared with 72 of 240
cross-host transitions. Dynamo's selected-prefix overlap was also much lower in
the cross-host logs. The current evidence establishes lower cache reuse and
longer Prefill queue/compute residence as the proximate cause of TTFT. It does
not yet distinguish whether the routing difference came from remote KV-event
visibility/timing or the selector's load-versus-locality decision. Therefore
the TTFT comparison must not be interpreted as the isolated cost of NVSHMEM or
cross-host Projection--Attention traffic.

Replay from L20B after synchronizing the source tree to L20T:

```bash
RUN_ID=<unique-id> \
  bash benchmarks/pap/scripts/run_cross_host_nvshmem_7pa1p_aiperf.sh
```

The launcher saves the requested and effective configuration, dataset and
source checksums, runtime context files, AIPerf records, service logs, route and
correctness summaries, and cleanup audit. It invalidates the run before load if
the source, GPUs, MPS partition, process map, IBGDA initialization, or service
health differs from the recorded contract.
