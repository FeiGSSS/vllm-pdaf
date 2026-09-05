# Local research intake and evidence boundary

User asks for an architecture-conference-worthy exact-inference mechanism,
grounded PAP source, primary literature and bounded microbench/trace evidence,
with a complete falsifiable argument for overall serving performance.

Available hardware is one 8x48GB NVIDIA L20 PCIe node. Model is dense Qwen3-8B,
FP16, 36 layers, H=4096, FFN=12288, Q heads=32, KV heads=8, head dimension128.
Current PAP has 7 PA workers (80 Prefill SM / 12 Attention SM each), 1 Projection
GPU, native Dynamo routing, CUDA IPC shared Prefill/Attention KV and NVSHMEM
whole-step CUDA Graph execution. No external API campaign or multi-node runs
are authorized/needed. Research evidence should cost minutes of targeted GPU
probes, not another all-dataset benchmark queue. A later paper campaign needs
an explicitly budgeted and fair baseline matrix.

Critical new finding: current/archived source fails to connect advertised Decode
KV capacity to actual allocation. Retained trace exhibits duplicate physical
blocks even for single-request PA steps. See `source_audit.md` and
`evidence_audit.md`. A repair is isolated at
`/tmp/pap-research-reservation-fix-20260905`; do not assume it is merged or
numerically validated. Historical completion/length audits are not correctness
proofs, and affected PAT speedups, KV reuse/capacity attribution and serving
performance are NOT valid empirical premises for a new research claim.

Fresh independent data: `projection_backbone.json` and `.md`, using 36 distinct
weight sets and existing dense operators. B16→B64 increases per-layer service
0.575→0.655 ms; fixed B32 splitting into two B16 increases combined sequential
Projection service0.625→1.151ms. Not a pipeline/TBT measurement; no attention,
communication, QK norm/RoPE or vocabulary head included. This constrains service
cost but establishes no end-to-end speedup.

Primary-method collision audit in `literature/mechanism_audit.md`:
CrossPool already independent layer cursors/two batches/GPU dispatch;
Tarragon already ready-subset same-layer expert batching;
Lamina already KV-local compute-pool migration and early-Q;
MPK already GPU continuous batching; POD already adaptive SM CTA coexecution;
Hydragen/FlashInfer already hierarchical prefix reuse; NanoFlow already overlap.
Additional targeted audit is checking Feather (prefix-homogeneous batching)
and LAMPS/MARS (memory-over-time scheduling). A broader statement of any of
these is not a novel candidate. Do not force novelty if the evidence does not
support it. Bug fixing itself is engineering, not the requested contribution.

One unresolved design axis to examine, not a selected cure: current PA owns an
entire request KV stream. Prefix replication across request owners trades away
capacity; co-locating all users trades away parallelism. Does a distributed
physical-prefix segment graph with exact partial-attention composition offer
anything beyond elastic context parallelism / distributed prefix caching?
Another unresolved axis is nonadditive release of shared KV under SLOs. Both
need precise prior-art and naive-baseline checks; neither is claimed new.
