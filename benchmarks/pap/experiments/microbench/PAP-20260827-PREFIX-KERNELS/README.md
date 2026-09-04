# PAP prefix-aware Attention kernel evaluation

## Decision

Both official implementations were ported to the NVIDIA L20 and measured with
the PAP Qwen3-8B decode shape.  Only PAT passes the applicability gate:

- **ChunkAttention is rejected for PAP.**  Its official kernel exposes MHA,
  not Qwen3-8B's 32-query-head/8-KV-head GQA shape, and is slower than
  FlashInfer Cascade on the realistic long-context workload.
- **PAT is the only shared-KV PAP fast path.**  The current `auto` policy uses
  PAT whenever two requests reference any common physical KV token and uses
  the Triton paged-decode kernel otherwise.

The finalized persistent-PAT path passed matched one-hour 2K and 32K 7PA1P
runs with no service errors or leaked lifecycle state.  Relative to the
historical Triton PAP baseline, the finalized 2K path improves output
throughput by 22.7% and mean TBT by 23.9%; the finalized 32K path improves
output throughput by 20.9% and mean TBT by 24.9%.

In the merged PAP runner, `PAP_ATTENTION_KERNEL_POLICY=auto` is the default.
An exact repeat of the previous step reuses all PAT metadata.  While the batch
and shared-prefix structure stay fixed, decode growth updates only the tail CTA
lengths and private block-table entries; crossing a 16-token page does not
rebuild the radix tree.  A structural change rechecks physical KV reuse and
rebuilds PAT only when selected.  Set the policy to `triton` for a control run.
FlashInfer Cascade is no longer part of the runtime; its measurements below are
retained as historical A/B evidence.

## Implemented paths

### ChunkAttention

The official Microsoft implementation was built at commit
`dde08ce9031ae9bf9e74f08c32cb549ef5f1340d`.  The local build patch:

- targets SM89 instead of SM86;
- uses the active Python 3.12 and PyTorch C++ ABI;
- omits the upstream unit-test/GTest build from this kernel benchmark.

The PAP microbenchmark constructs physically shared prefix chunks and compares
the official ChunkAttention CUDA extension with FlashInfer Cascade using the
same query and KV tensors.  Because the official API has one head count rather
than separate query/KV head counts, this comparison must use an MHA-equivalent
shape.  `production_gqa_supported=false` is recorded in every artifact.

### PAT

The official PAT implementation was built at commit
`b61e589cc8775930931157ff3bb107ba28bafd77`, using CUTLASS commit
`ffa119a1255d78998536107466cc7097ecefa393`.  The port:

- targets SM89;
- launches the single-stream Graph path on PyTorch's current CUDA stream;
- launches PAT's gather kernel on the same stream;
- exposes scheduler metadata needed for fixed-address CUDA Graph buffers;
- makes the supplied `(M, N, warps)` tile set effective.

`vllm/pap/attention/pat.py` builds the PAT radix-tree schedule in PAP's existing
step-prepare phase, copies it into fixed-address GPU metadata buffers, and
replays the kernel inside the existing 36-layer whole-step CUDA Graph.  Each
private decode tail reserves block-table capacity from the request's existing
KV lease.  Ordinary steps update `kv_in_CTAs`; page crossings append the known
private block IDs without rebuilding the tree or recapturing the Graph.

The automatic deployment guard currently requires:

- at least two requests with any common physical KV token;
- block size 16, FP16, 32 query heads, 8 KV heads, head dimension 128;
- `DISABLE_STREAM=1`, required by the captured single-stream implementation.

Unsupported or physically disjoint workloads use Triton.  Batch-size and
minimum-prefix performance heuristics have been removed.

## Kernel results

All constrained measurements use the production PA allocation of 12 visible
SMs, FP16 KV, block size 16, and real context shapes from the 7PA1P trace.

### ChunkAttention

| Shape | FlashInfer Cascade | ChunkAttention | Result |
| --- | ---: | ---: | ---: |
| Batch 6, full L20 | 3.577 ms | 3.803 ms | Chunk 6.3% slower |
| Batch 6, 12 SM | 4.235 ms | 8.651 ms | Chunk 2.04x slower |

Maximum absolute error against the reference is `1.073e-6`, but the production
GQA shape is unsupported.  Artifacts:

- `microbench/chunk_full_b6_mha.json`
- `microbench/chunk_mps12_b6_mha.json`

### PAT versus FlashInfer Cascade

| Batch | FlashInfer Graph | PAT Graph | PAT change | Max abs error |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.455 ms | 0.499 ms | +9.5% | `2.38e-7` |
| 3 | 0.545 ms | 0.636 ms | +16.8% | `1.84e-4` |
| 4 | 1.154 ms | 1.106 ms | -4.2% | `2.99e-4` |
| 5 | 1.539 ms | 1.262 ms | **-18.0%** | `1.98e-4` |
| 6 | 1.649 ms | 1.552 ms | **-5.9%** | `1.99e-4` |
| 7 | 1.550 ms | 1.495 ms | -3.6% | `1.80e-4` |
| 8 | 1.599 ms | 1.592 ms | -0.4% | `1.25e-4` |

Artifacts are `microbench/pat_trace_b{2..8}_mps12.json`.  The batch-5/6-only
selection described by this historical matrix was removed from the current
runtime; current `auto` selects PAT for any physical prefix reuse.

## End-to-end validation (separate record)

The service-level artifacts are stored under
`../../e2e/PAP-20260827-PREFIX-KERNELS-E2E/`. The summary is retained here to
connect the isolated kernel result to its serving validation.

### Isolated PAT increment: 120-second A/B

The control is FlashInfer Cascade.  The candidate is the `auto` policy, which
substitutes PAT only for batch 5 and 6.  Thirty requests have identical
conversation ID, turn index, input length, and output length in both runs.

| Metric | Cascade | Auto | Change |
| --- | ---: | ---: | ---: |
| Mean TTFT | 33.985 s | 33.966 s | -0.06% |
| Mean TBT | 94.301 ms | 88.274 ms | **-6.39%** |
| Mean end-to-end latency | 86.091 s | 82.228 s | -4.49% |
| Per-user output throughput | 10.772 tok/s | 11.586 tok/s | +7.55% |

Inputs:

- Cascade: `../../e2e/PAP-20260827-PREFIX-KERNELS-E2E/results/cascade_120_profile.json`
- Auto: `../../e2e/PAP-20260827-PREFIX-KERNELS-E2E/results/auto_120_profile.json`
- Derived comparison: `../../e2e/PAP-20260827-PREFIX-KERNELS-E2E/short_auto_vs_cascade.json`

### Historical hybrid 2K one-hour test (superseded)

Candidate configuration:

- 7PA1P Qwen3-8B, 131K maximum context;
- agentic-code replay, Poisson 0.3 request/s, concurrency 60;
- no warmup, 3,600-second sending window;
- Prefill `max_num_batched_tokens=2048`;
- historical hybrid policy and `DISABLE_STREAM=1`.

The historical control uses the same dataset and load settings with the
original Triton PAP Attention path.

| Raw fixed-window metric | Triton PAP | Hybrid auto | Change |
| --- | ---: | ---: | ---: |
| Completed requests | 615 | 713 | +15.9% |
| Request throughput | 0.171 req/s | 0.199 req/s | +16.1% |
| Output throughput | 161.96 tok/s | 189.09 tok/s | +16.8% |
| Mean TTFT | 195.51 s | 167.08 s | -14.5% |
| Mean TBT | 101.25 ms | 77.61 ms | **-23.3%** |
| Mean end-to-end latency | 292.02 s | 241.35 s | -17.3% |

Fixed-duration completion changes censor the two raw populations differently.
The strict comparison therefore uses 111 turn-0 requests with identical input
and output token counts:

| Strict paired turn-0 metric | Triton PAP | Hybrid auto | Change |
| --- | ---: | ---: | ---: |
| Mean TTFT | 101.555 s | 99.723 s | -1.80% |
| Mean TBT | 100.946 ms | 76.208 ms | **-24.51%** |
| Mean end-to-end latency | 203.499 s | 175.826 s | -13.60% |
| Per-user output throughput | 10.508 tok/s | 13.642 tok/s | +29.83% |

There are also 206 exact-length request pairs across all turns; their TBT is
24.21% lower.  Later-turn prompt lengths diverge after the two runs generate
different assistant text, so the turn-0 comparison is the cleanest result.

The compact candidate artifacts are in the sibling E2E record's `results/`
directory. Its strict analyzer, captured output, and kernel-specific launch
environment are stored beside those results.

Runtime audits from the candidate run:

- completed 713, cancelled at boundary 57, request errors 0;
- gateway drain passed with zero in-flight/lifecycle records;
- all seven Attention sessions drained;
- all eight PAP whole-step Graph processes captured successfully;
- strict correctness log match count is zero.

This hybrid result is retained as development evidence, but is superseded by
the matched persistent-PAT 2K run below.

### Persistent PAT, matched 2K and 32K one-hour results

The finalized PAT-or-Triton selector, persistent radix-tree metadata, private
page-tail updates, and bounded Attention Graph cache were rerun with the same
fixed-duration workload at both Prefill token limits.  All cells use
`2K / 32K` order.

| Metric | Persistent PAT 2K / 32K |
| --- | ---: |
| Completed requests | 740 / 651 |
| Request throughput | 0.206 / 0.181 req/s |
| Output throughput | 198.64 / 176.88 tok/s |
| Mean TTFT | 152.73 / 196.70 s |
| Mean TBT | 77.03 / **68.49 ms** |
| Mean end-to-end latency | 227.11 / 263.54 s |

Persistent PAT-2K is the new production PAP point.  It is within 3.3% of
DP8-2K output throughput and exceeds 4P4D-2K by 3.6%.  Its mean TBT is 18.5%
lower than DP8-2K but 17.1% higher than 4P4D-2K.  Persistent PAT-32K has the
best PAP TBT, while 2K has the better throughput, TTFT, and end-to-end latency.

| Metric | Historical Triton PAP-32K | Persistent PAT PAP-32K | Change |
| --- | ---: | ---: | ---: |
| Completed requests | 554 | 651 | +17.5% |
| Request throughput | 0.154 req/s | 0.181 req/s | +17.8% |
| Output throughput | 146.33 tok/s | 176.88 tok/s | +20.9% |
| Mean TTFT | 240.20 s | 196.70 s | -18.1% |
| Mean TBT | 91.13 ms | 68.49 ms | **-24.9%** |
| Mean end-to-end latency | 327.58 s | 263.54 s | -19.6% |

The 2K sending boundary completed 740 requests and cancelled 60 in-flight
requests.  The 32K boundary completed 651 and cancelled 59.  Both reported
zero errors and passed strict correctness, routing, Decode-token join,
whole-step Graph, gateway drain, and seven-PA session drain audits.

Across all seven PA instances, the 2K run reused incremental PAT metadata
310,573 times; 1,418 structure changes produced 1,090 PAT rebuilds and 328
Triton selections.  The 32K values were 319,784, 1,195, 819, and 376,
respectively.  Neither run reported token mismatches, dispatch failures,
pending KV, OOM, or correctness failure.  Both exercised the bounded Attention
Graph cache; five PA instances reached the 32-entry limit in the 2K run.

The compact tracked records are in
`../../e2e/PAP-20260827-PREFIX-KERNELS-E2E/results/`.

## Reproduction

The exact upstream commits are in `official_commits.env`; the source changes
are in `../../../patches/pat-sm89-pap.patch`. No third-party binary is vendored.

Build and install the pinned SM89 PAT extension into the project environment:

```bash
benchmarks/pap/scripts/build_pat_attention.sh
```

ChunkAttention is not part of the runtime merge because its official kernel
does not support the production GQA shape and failed the performance gate.

## Sources

- ChunkAttention paper: <https://arxiv.org/abs/2402.15220>
- ChunkAttention code: <https://github.com/microsoft/chunk-attention>
- PAT paper: <https://arxiv.org/abs/2511.22333>
- PAT code: <https://github.com/MachineLearningSystem/26ASPLOS-PAT>
