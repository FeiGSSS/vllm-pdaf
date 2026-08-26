# Agentic Coding one-hour architecture matrix

Experiment ID: `PAP-20260825-AGENTIC-CODE-1H-MATRIX`.

## Decision

On this long-context Agentic Coding workload, Dynamo-managed DP8 has the best
end-to-end performance. 4P4D is the strongest fixed PD split. PAP 7PA1P is
slower than both DP8 and 4P4D; it only exceeds the deliberately imbalanced
6P2D split in throughput and TTFT.

The workload is not simply Decode-heavy. It is heavy on both sides and highly
Decode-sensitive: 2P6D minimizes inter-token latency after admission but
severely queues Prefill, while 6P2D queues Decode admission. The throughput
maximum among the measured fixed splits occurs at 4P4D.

## Fixed protocol

- hardware: eight NVIDIA L20 GPUs on one host;
- model: Qwen3-8B FP16, TP1;
- context limit: 131,072 tokens with static YaRN factor 4;
- dataset: 2,092 Agentic Coding conversations with authored turn delays
  removed;
- dataset SHA-256:
  `b6670fac38fef5f43c5d93cd4e8946b1de7a224c80a57399a9e1e04aa6fe425b`;
- arrival process: Poisson, 0.3 request/s, concurrency cap 60;
- measured window: 3,600 seconds, no warmup and no drain grace;
- execution: CUDA Graphs in every architecture;
- batch-token variants: 2,048 and 32,768;
- client: AIPerf with the `mooncake-trace` custom dataset type;
- baseline stack: Dynamo 1.4.1 with upstream vLLM 0.26.0;
- PAP topology: seven colocated Prefill/Attention GPUs and one Projection GPU.

All cells below use `2K / 32K` order.

## Results

| Architecture | Output token/s | Mean TTFT | Mean ITL | Mean request latency |
| --- | ---: | ---: | ---: | ---: |
| DP8 | **205.5 / 204.9** | **126.2s / 138.1s** | 94.5ms / 84.8ms | **215.6s / 218.1s** |
| PAP 7PA1P | 162.0 / 146.3 | 195.5s / 240.2s | 101.2ms / 91.1ms | 292.0s / 327.6s |
| 2P6D | 124.8 / 124.8 | 340.5s / 341.4s | **37.6ms / 37.8ms** | 376.6s / 377.5s |
| 4P4D | 191.8 / 188.6 | 182.7s / 188.6s | 65.8ms / 64.7ms | 245.7s / 250.9s |
| 6P2D | 114.8 / 114.7 | 364.2s / 360.7s | 72.7ms / 73.5ms | 434.5s / 431.1s |

The exact mean, P50, P90, and P99 metrics and each `profile.json` digest are
stored in `results.tsv`.

## Interpretation

### DP8

DP8 reaches approximately 205 output tokens/s in both variants. Reducing the
batch-token limit from 32K to 2K does not change throughput, but moves latency
between phases: mean TTFT improves by 8.6%, while mean ITL regresses by 11.4%.
DP8 benefits from allowing all eight GPUs to change between Prefill and Decode
work over time rather than enforcing a static role split.

### Fixed PD splits

4P4D is the best measured fixed split. Its 2K output throughput is only 6.7%
below DP8-2K and 18.4% above PAP-2K. Its mean ITL is 35.0% below PAP-2K.

2P6D has the best ITL because six Decode workers carry a smaller active batch,
but two Prefill workers cannot admit this workload fast enough. Relative to
4P4D-2K, its output throughput is 34.9% lower and TTFT is 86.4% higher.

6P2D exposes the opposite imbalance. Six Prefill workers feed only two Decode
workers, producing the lowest throughput and highest mean request latency.

Changing the PD Prefill batch-token limit between 2K and 32K has negligible
effect. The fixed P:D allocation, rather than Prefill chunk size, controls the
observed bottleneck.

### PAP

PAP-2K is the stronger PAP point. PAP-32K improves mean ITL by 10.0%, but
reduces output throughput by 9.7%, increases TTFT by 22.9%, and increases mean
request latency by 12.2%.

PAP does not demonstrate an architectural advantage on this workload. Against
DP8 with the same batch-token limit, PAP adds approximately 6.4--6.8 ms of
mean ITL. Against 4P4D, the gap is much larger because 4P4D isolates Prefill
from Decode GPUs.

The current evidence does not attribute PAP's ITL entirely to the per-layer
seven-PA Projection barrier. The matched 2K and 32K comparisons indicate two
components:

1. sharing PA GPUs between Prefill and Attention accounts for a large part of
   the 20--29 ms gap between aggregated DP8 and dedicated-Decode 4P4D;
2. PAP-specific communication, load imbalance, and global synchronization add
   approximately another 6--7 ms relative to DP8.

This is an empirical decomposition, not a kernel-level causal measurement. A
Decode-only Nsight run is still required to separate Projection compute,
QKV dispatch, individual PA Attention kernels, and output-gather wait time.

## KV-transfer and correctness audits

All ten AIPerf profiles report zero request errors. DP8 CUDA Graph and
correctness audits passed. Both PAP runs passed strict correctness, seven
Prefill Graphs, eight-process whole-step Graph, routing, Projection scheduling,
Projection outer-Graph configuration, session/gateway drain, and Decode-token
join audits.

The valid PD KV-transfer aggregate throughputs were:

| Architecture | 2K | 32K |
| --- | ---: | ---: |
| 2P6D | 13,592 MB/s | 9,780 MB/s |
| 4P4D | 10,391 MB/s | 10,555 MB/s |
| 6P2D | 9,876 MB/s | 10,471 MB/s |

Every value exceeds the 5,000 MB/s same-node fail-closed floor. The 4P4D-2K
outer runner encountered a shell parse error after AIPerf export because its
launcher was edited while Bash was still executing it. The profile completed
with zero errors and its CUDA Graph audit passed; a post-hoc scan found no
correctness failure signatures, and 537 transfer windows covering 728
transfers measured 10,391 MB/s. It is retained as valid performance evidence,
with this audit provenance stated explicitly.

## Excluded runs

Two earlier 2P6D attempts are invalid and excluded. The launcher incorrectly
used the six-Decode-worker count as a startup threshold shared by both worker
pools, so the router waited for six Prefill workers although only two existed.
The Prefill router never activated and requests ran directly on Decode workers.
The repaired runs use the minimum of the Prefill and Decode worker counts for
the shared readiness threshold; logs confirm Prefill router activation and
real NIXL transfers.

## Source and raw artifacts

Most runs captured source commit
`563b804680b3d80c4bf8d9132ba9aaadceffcb51` plus their tracked worktree patch.
The repaired 2P6D runs captured milestone commit
`8bc7f0d761589a65dda831d3ac6e5bcbe77c81a5`.

Raw profiles and service logs remain under:

- `benchmarks/pap/experiments/_staging/steady/20260825_4arch_3600s_nowarm_q03_c60`;
- `benchmarks/pap/experiments/_staging/steady/20260825_pap_pd_3600s_nowarm_q03_c60_prefill32k`;
- `benchmarks/pap/experiments/_staging/steady/20260826_pd_3600s_nowarm_q03_c60_prefill2k`;
- `benchmarks/pap/experiments/_staging/steady/20260826_dp8_3600s_nowarm_q03_c60_tokens2k`;
- `benchmarks/pap/experiments/_staging/steady/20260826_2p6d_fixed_3600s_nowarm_q03_c60_prefill2k`;
- `benchmarks/pap/experiments/_staging/steady/20260826_2p6d_fixed_3600s_nowarm_q03_c60_prefill32k`.

These raw directories are machine-local staging artifacts and are not added to
Git. The compact tracked result table contains their profile digests.
