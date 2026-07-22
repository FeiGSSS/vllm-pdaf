# PAP versus PD: automatic Projection-memory milestone

## Scope

This four-GPU milestone validates automatic Projection memory sizing in both
eager and piecewise CUDA Graph modes, and refreshes the PAP-versus-PD capacity
comparison after making PD multi-turn P/D pairing deterministic.

- PAP automatic-memory runtime commit: `4aaf9dd77c7b5de8697afa91c89921ca93078bea`
- PD Cartesian-pair benchmark commit: `4484bc983e622fc03dffdec6c2831d9f6ec6396f`
- AIPerf: 0.11.0
- Model and hardware: Qwen3-8B FP16 on four NVIDIA L20 GPUs
- Work per point: 32 conversations, ten turns, 320 requests
- Timing: conversation concurrency with delays
  `0,3,3,1,3,3,1,3,3,1` seconds
- Dataset seed: 42
- Dataset SHA-256:
  `56dfe24c63fbb582f113db6e7f2ec2422bb313dcf23393ea192a062db158ea85`

The randomized workload has 8,192/8,000 initial-input mean/median tokens,
512/500 appended-input mean/median tokens, and 32/30 output mean/median
tokens. Output lengths range from 16 to 64 tokens. The longest estimated
request is 16,224 tokens under `max_model_len=20000`.

Commit `4484bc983` changes only benchmark Proxy routing, result auditing, and
their tests; it does not change the PAP runtime measured at `4aaf9dd77`.
Eager PAP therefore remains directly comparable with the corrected PD
reruns.

## Runtime configuration

| Role | GPU memory | `max_num_seqs` | `max_num_batched_tokens` |
| --- | ---: | ---: | ---: |
| PAP PA Prefill | 0.90 | 64 | 16,384 |
| PAP Attention | colocated outside vLLM budget | - | - |
| PAP Projection | automatic 0.4070 | 64 | 64 |
| PD Prefill/Decode | 0.90 | 64 | 16,384 / 64 |

Projection computes `ceil((checkpoint_bytes / TP) * 1.20 / gpu_bytes)` and
rounds upward to four decimals. For this model and GPU, 16,381,470,720 bytes
of TP1 weights target 19,657,764,864 bytes on a 48,305,799,168-byte GPU,
giving `gpu_memory_utilization=0.4070`. Projection preserves KV metadata and
one null block but allocates no local request-KV tensor.

PAP uses 3PA1P with a static 72/20-SM Prefill/Attention split. PD uses one-way
KV transfer. `max_num_partial_prefills` stays at the vLLM default of 1. Every
point restarts all services.

Piecewise mode captures graph-safe model work while keeping remote Attention,
OFFLOAD_EXEC, and Prefill KV publication outside Graph replay. Every included
vLLM model-engine log contains `Graph capturing finished`, and no included
service log contains a runtime traceback, CUDA error, or OOM.

## Validity and exclusions

Every reported point completed 320/320 requests, passed output validation,
preserved conversation affinity, and reported zero migrations. This is a
single-repetition development milestone, not a release-level statistical
claim.

The original eager 2P2D run exposed nondeterministic P/D pairing: Prefill and
Decode owners were selected at different times, so asynchronous Prefill
completion could skew the pair distribution. `ConversationPairRouter` now
assigns a conversation once over the complete Cartesian P/D pair set. For
2P2D, all accepted runs assign 16/16 conversations per Prefill, 16/16 per
Decode, and 8/8/8/8 across the four pairs, with zero turn migration.

Two transport observations are retained but not used to inflate PAP results:

- The first Graph 2P2D C10 attempt had one Decode lane at roughly 4-6 MB/s
  while the other remained near 300-450 MB/s, including 15-166-second
  transfer windows. Its 0.686 req/s result is excluded as a transport
  diagnostic.
- The targeted Graph C10 repeat recovered both lanes and completed at
  1.660 req/s. One early tail still prevented Standard-SLO compliance, so the
  best Graph PD Standard point is the independent 3P1D C8 result.

For eager 2P2D C16, the first corrected run missed the Relaxed threshold by
two requests. Its single targeted repeat passed exactly 304/320 requests and
is used below. Choosing the better complete PD repeat is conservative with
respect to the PAP advantage.

## Eager results

| Architecture | Topology | C | TTFT p95 ms | ITL p95 ms | Req/s | Strict | Standard | Relaxed |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| PAP | 3PA1P | 12 | 3,205.91 | 40.72 | 2.534 | pass | pass | pass |
| PAP | 3PA1P | 20 | 8,869.97 | 46.38 | 3.460 | fail | pass | pass |
| PAP | 3PA1P | 28 | 11,331.91 | 63.23 | 3.435 | fail | fail | pass |
| PAP | 3PA1P | 32 | 14,591.72 | 66.68 | 4.929 | fail | fail | pass |
| PD | 1P3D | 8 | 9,586.03 | 29.66 | 1.401 | fail | pass | pass |
| PD | 2P2D | 10 | 8,068.92 | 30.28 | 1.871 | fail | pass | pass |
| PD | 2P2D | 16 | 14,083.83 | 33.86 | 2.658 | fail | fail | pass |
| PD | 2P2D | 20 | 27,520.29 | 36.03 | 2.445 | fail | fail | fail |
| PD | 3P1D | 8 | 4,647.59 | 34.64 | 1.747 | pass | pass | pass |
| PD | 3P1D | 14 | 11,604.73 | 42.12 | 2.259 | fail | fail | pass |
| PD | 3P1D | 20 | 24,235.99 | 65.45 | 1.367 | fail | fail | fail |

Only complete and correct points with at least 95% SLO-good requests are
eligible for goodput:

| SLO | PAP capacity | Best PD capacity | PAP goodput | PD goodput | PAP over PD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict | C12 | C8, 3P1D | 2.423 | 1.659 | +46.0% |
| Standard | C20 | C10, 2P2D | 3.330 | 1.795 | +85.5% |
| Relaxed | C32 | C16, 2P2D | 4.929 | 2.525 | +95.2% |

Relative to the preceding tracked eager milestone, PAP best goodput changes
by -1.3%, +1.6%, and -1.4% for Strict, Standard, and Relaxed respectively.
All are within 2%, so automatic Projection sizing introduces no measured
eager regression in this development repetition.

## Piecewise CUDA Graph results

| Architecture | Topology | C | TTFT p95 ms | ITL p95 ms | Req/s | Strict | Standard | Relaxed |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| PAP | 3PA1P | 8 | 1,981.77 | 34.32 | 2.050 | pass | pass | pass |
| PAP | 3PA1P | 12 | 3,796.51 | 42.37 | 2.502 | fail | pass | pass |
| PAP | 3PA1P | 20 | 8,795.66 | 48.58 | 3.362 | fail | pass | pass |
| PAP | 3PA1P | 28 | 12,043.24 | 60.26 | 3.506 | fail | fail | pass |
| PAP | 3PA1P | 32 | 14,582.40 | 74.49 | 4.957 | fail | fail | pass |
| PD | 2P2D | 10 | 12,849.98 | 31.74 | 1.660 | fail | fail | pass |
| PD | 2P2D | 16 | 17,482.48 | 33.22 | 2.370 | fail | fail | pass |
| PD | 2P2D | 20 | 23,696.11 | 36.83 | 1.873 | fail | fail | fail |
| PD | 3P1D | 8 | 4,275.61 | 33.26 | 1.850 | pass | pass | pass |

| SLO | PAP capacity | Best PD capacity | PAP goodput | PD goodput | PAP over PD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict | C8 | C8, 3P1D | 2.031 | 1.792 | +13.3% |
| Standard | C20 | C8, 3P1D | 3.236 | 1.850 | +75.0% |
| Relaxed | C32 | C16, 2P2D | 4.942 | 2.259 | +118.8% |

CUDA Graph is not a uniform per-point speedup:

| PAP point | TTFT eager -> Graph | ITL eager -> Graph | Req/s eager -> Graph |
| --- | ---: | ---: | ---: |
| C12 | 3,205.91 -> 3,796.51 | 40.72 -> 42.37 | 2.534 -> 2.502 |
| C20 | 8,869.97 -> 8,795.66 | 46.38 -> 48.58 | 3.460 -> 3.362 |
| C28 | 11,331.91 -> 12,043.24 | 63.23 -> 60.26 | 3.435 -> 3.506 |
| C32 | 14,591.72 -> 14,582.40 | 66.68 -> 74.49 | 4.929 -> 4.957 |

C12 misses Strict by two requests under Graph, while C8 supplies a valid
Strict point. Higher-concurrency throughput is approximately unchanged or
slightly improved, but tail latency moves in both directions. Piecewise Graph
therefore remains supported and optional rather than becoming the default.

## Conclusion

Automatic Projection sizing is validated in eager and piecewise CUDA Graph
modes. It starts reliably at 0.4070, preserves Projection's metadata-only KV
contract, and shows no eager PAP regression beyond 2% relative to the prior
milestone.

Under the canonical four-GPU AIPerf workload, PAP leads the best valid PD
goodput in every SLO tier. The eager advantages are +46.0%, +85.5%, and
+95.2%; the Graph advantages are +13.3%, +75.0%, and +118.8%. These exact
percentages remain single-repetition development evidence. The architectural
conclusion is stronger than any one number: PAP supports a larger
SLO-compliant concurrency envelope while Projection memory is derived from
its actual weight requirement rather than consuming a PA-style KV budget.
