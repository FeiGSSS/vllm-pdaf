# PAP research loop L06: three-way evidence provenance audit

Date: 2026-07-29

## Question and decision

L06 audited the merged July 28 PAP/PD/fused-DP capacity result before using it
to reorient the paper. Its numerical conclusion is important but does not
qualify as clean paper evidence.

All selected boundaries use the same byte-identical dataset, complete 640/640
requests, pass their runtime correctness audits, and use the corrected
same-node UCX/NIXL settings where applicable. However, every selected run
records `GIT_TRACKED_WORKTREE_DIRTY=1`. The captured 52,776-byte patch changes
benchmark launch and summary code and also changes
`vllm/pap/lifecycle/decode_token_client.py`, which is on the running PAP
lifecycle path.

C06 is therefore superseded without deciding its performance statement. The
July 28 numbers remain directional evidence only. L07 must repeat the eight
selected boundaries on committed code before the manuscript can claim that
PAP loses to corrected PD or retains an advantage over fused DP.

## Audited directional result

The merged report selected:

| SLO | PAP | PD | Fused DP | PAP vs PD | PAP vs DP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict | 6PA2P C32, 8.947 good req/s | 6P2D C31, 14.283 | DP C8, 5.972 | -37.4% | +49.8% |
| Standard | 7PA1P C34, 12.233 | 6P2D C44, 15.665 | DP C18, 10.714 | -21.9% | +14.2% |
| Relaxed | 7PA1P C34, 12.410 | 6P2D C48, 16.005 | DP C28, 14.175 | -22.5% | -12.5% |

Every selected point has at least two underlying repetitions in the merged
result. The table is not promoted because the clean-worktree requirement
fails.

## Identity checks that pass

- Model: Qwen3-8B, FP16, eager.
- Hardware: eight NVIDIA L20 GPUs.
- Dataset SHA-256 for every source matrix:
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.
- Workload: 128 conversations, five turns, 640 requests.
- PD/PAP NIXL runtime: `same_node_ucx122_strict`.
- UCX: 1.22.0.
- `UCX_PROTO_EMULATION_ENABLE=n`.
- `UCX_CUDA_IPC_ENABLE_GET_ZCOPY=y`.
- Model length: 32,768; `max_num_seqs=256`.
- PAP routing: conversation affinity.
- Request and runtime correctness: passed for selected repetitions.

## Provenance failure

All selected rows were collected at base commit `a23210c6c` with a dirty
tracked worktree. Representative source paths include:

```text
20260728_8gpu_three_way_isolated_confirmation/
20260728_8gpu_three_way_isolated_fallback_confirmation/
20260728_8gpu_three_way_isolated_final_edges/
20260728_8gpu_three_way_isolated_final_gaps/
20260728_8gpu_three_way_isolated_last_confirmation/
20260728_8gpu_three_way_isolated_scan/
```

Each representative `tracked_worktree.patch` is 52,776 bytes and touches:

```text
benchmarks/pap/aiperf/README.md
benchmarks/pap/aiperf/run_capacity_matrix.sh
benchmarks/pap/aiperf/run_goodput_scan.sh
benchmarks/pap/aiperf/summarize_capacity_matrix.py
benchmarks/pap/aiperf/summarize_capacity_run.py
benchmarks/pap/scripts/run_pap_workload.sh
vllm/pap/lifecycle/decode_token_client.py
```

Because a PAP runtime file changed, this cannot be waived as documentation-only
dirtiness. The same dirty tree was used by all architectures, but equal
dirtiness is not reproducible provenance and could affect them differently.

## Decision and next loop

Do not cite the July 28 goodput deltas as a paper result. Use them only to
select the minimal confirmation boundaries.

L07 will run two clean repetitions of exactly eight points on the same
dataset:

```text
PAP 6PA2P: C32
PAP 7PA1P: C34
PD 6P2D: C31, C44, C48
fused DP: C8, C18, C28
```

No new concurrency search is allowed in L07. If the clean result reproduces
the directional comparison, the following loop will isolate the workload
dimension responsible for PAP's loss rather than tune against the answer.

