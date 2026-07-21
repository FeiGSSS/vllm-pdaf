# PAP versus PD: four-GPU AIPerf capacity scan

## Scope

This experiment is the first clean controlled run of the fixed AIPerf capacity
testbed. It compares PAP 3PA1P with every one-way PD split on four NVIDIA L20
GPUs. The scan uses one repetition per point, so it is controlled capacity
evidence rather than a formal three-repetition release result.

This historical scan used `total sessions = C`, so each point contained one
cohort. The later fixed-session testbed supersedes that methodology for goodput
comparisons while preserving this result as pilot capacity evidence.

- vLLM/PAP commit: `c86d601e49f35739d4953b6f88bd73b53da69e94`
- Tracked worktree at launch: clean
- AIPerf: 0.11.0 at `854ff91a4a221f899b806e7660a89b41b80d5689`
- Model: Qwen3-8B, FP16, TP1 per worker
- Load mode: pure session concurrency; no request-rate schedule
- Scan: `C=4,8,12,16,24,32`, stopping each topology after its first
  relaxed-SLO failure
- Dataset SHA-256:
  `a9a2283c61946b2198134fe6a17445e55b549bcd01356c2ec521aa5f67636b0a`

Each session has ten turns. The first user text requests 8,192 tokens, later
turns add 512 user tokens, and every turn produces exactly 256 output tokens.
With chat-template overhead and prior assistant responses, observed prompt
lengths range from 8,210 to 15,338 tokens.

The configurations are fixed across concurrency points:

- PAP: 3PA1P, static 72/20-SM Prefill/Attention partition,
  `gpu_memory_utilization=0.76` for PA and Projection workers.
- PD: one-way 1P3D, 2P2D, and 3P1D,
  `gpu_memory_utilization=0.90` for every worker.
- Both: `max_model_len=20000`, `max_num_batched_tokens=8192`, and
  `max_num_seqs=32`.

PAP intentionally remains at 0.76 rather than 0.90 because its workers share
GPU memory with the split Attention/MPS and transport runtime. Projection does
not own prompt KV, so increasing only its reservation would not increase the
number of resident conversations.

## SLOs

A request is good only when both its TTFT and request-level mean ITL meet the
tier. A tier passes when at least 95% of all expected requests are good and the
entire correctness gate passes.

| Tier | TTFT | ITL | Good requests |
| --- | ---: | ---: | ---: |
| Strict | <= 5 s | <= 50 ms | >= 95% |
| Standard | <= 10 s | <= 75 ms | >= 95% |
| Relaxed | <= 20 s | <= 100 ms | >= 95% |

## Results

All 16 executed points passed correctness and conversation-affinity audits.
All 188 sessions completed all ten turns: 1,880/1,880 requests completed with
exactly 256 output tokens and no routing migration.

| Architecture | Topology | C | TTFT p95 ms | ITL p95 ms | Req/s | Strict | Standard | Relaxed |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| PAP | 3PA1P | 4 | 2,407.58 | 34.53 | 0.430 | pass | pass | pass |
| PAP | 3PA1P | 8 | 5,229.58 | 41.24 | 0.704 | fail | pass | pass |
| PAP | 3PA1P | 12 | 6,617.50 | 44.66 | 1.002 | fail | pass | pass |
| PAP | 3PA1P | 16 | 8,248.26 | 51.46 | 1.189 | fail | pass | pass |
| PAP | 3PA1P | 24 | 9,946.99 | 62.48 | 1.418 | fail | pass | pass |
| PAP | 3PA1P | 32 | 29,319.03 | 95.46 | 0.976 | fail | fail | fail |
| PD | 1P3D | 4 | 13,710.67 | 28.83 | 0.353 | fail | fail | pass |
| PD | 1P3D | 8 | 23,277.94 | 31.84 | 0.749 | fail | fail | fail |
| PD | 2P2D | 4 | 7,940.65 | 29.14 | 0.475 | fail | pass | pass |
| PD | 2P2D | 8 | 12,659.22 | 34.94 | 0.788 | fail | fail | pass |
| PD | 2P2D | 12 | 16,641.48 | 40.63 | 1.015 | fail | fail | pass |
| PD | 2P2D | 16 | 22,356.69 | 46.19 | 1.173 | fail | fail | fail |
| PD | 3P1D | 4 | 4,228.64 | 34.64 | 0.426 | pass | pass | pass |
| PD | 3P1D | 8 | 7,176.59 | 46.33 | 0.657 | fail | pass | pass |
| PD | 3P1D | 12 | 14,349.79 | 56.32 | 0.765 | fail | fail | pass |
| PD | 3P1D | 16 | 27,399.73 | 62.94 | 0.627 | fail | fail | fail |

The maximum passing tested concurrency is:

| SLO | PAP 3PA1P | Best PD | Capacity ratio |
| --- | ---: | ---: | ---: |
| Strict | 4 | 4 (3P1D) | 1.0x |
| Standard | 24 | 8 (3P1D) | 3.0x |
| Relaxed | 24 | 12 (2P2D or 3P1D) | 2.0x |

At the shared strict point `C=4`, PAP also has lower TTFT than PD 3P1D
(2.41 versus 4.23 seconds), nearly identical ITL, and nearly identical request
throughput. Under the standard and relaxed SLOs, the fixed 3PA1P layout supports
substantially more active sessions than any fixed PD split in this scan.

PAP scales through `C=24`, reaching 1.418 requests/s. At `C=32`, TTFT rises to
29.3 seconds and throughput falls to 0.976 requests/s, identifying overload
rather than a correctness failure. PD 1P3D is Prefill-limited; PD 3P1D improves
TTFT at low concurrency but its single Decode worker becomes the high-load
bottleneck.

## Evidence and limitations

The complete generated table and compact summaries are under
[`20260720_c86d601e4_aiperf_capacity`](../capacity/20260720_c86d601e4_aiperf_capacity/capacity_results.md).
Every point restarted all services with cold process state. The runner stopped
after the first relaxed failure, which reduced the full 24-point Cartesian
matrix to 16 executed points.

These capacities are the largest passing values among the tested concurrency
levels, not exact mathematical limits. In particular, a missing strict capacity
for PD 1P3D or 2P2D only means neither passed at the minimum tested `C=4`.
PAP `C=24` meets the standard tier at the exact 95% good-request boundary.
Before using the result in a publication or release claim, repeat the boundary
points at least three times; this lean scan should remain the development
testbed.
