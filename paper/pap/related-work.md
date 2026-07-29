# PAP Related-Work Map

This file is a source-backed comparison matrix, not a search diary. Add only
work that changes PAP's positioning, design, evaluation methodology, or
claimed novelty. Prefer peer-reviewed papers and authoritative preprints from
the original authors.

| Work | Venue/year | Boundary | Scheduling unit | KV ownership | Workload and baselines | Relevance to PAP |
| --- | --- | --- | --- | --- | --- | --- |
| Adrenaline [liang2025adrenaline] | arXiv 2025 | Partially offloads Decode Attention from a PD Decode instance to colocated Prefill GPUs; QKV/O-projection/FFN remain on Decode. | Per-request offload admission bounded by sequence-length, batch, memory, and TPOT profiles. | Offloaded KV occupies Prefill-side HBM; non-offloaded requests retain Decode-local Attention. | Llama-2 7B/13B on 8xA100 NVLink; ShareGPT and OpenThoughts; vLLM PD baseline; §3.2--3.4 and §4. | Closest implemented system. PAP must not claim attention offload, QKV aggregation, MPS colocation, or token-load-aware admission as novel. PAP's candidate distinction is full PA-owned Attention/KV with stateless Projection aggregation and multi-turn post-Prefill migration. |
| Analytical Provisioning [song2026analytical] | arXiv 2026, submitted to NeurIPS | Models an \(rA\)-to-\(1F\) operator-disaggregated Decode pipeline. | Fixed Attention microbatches with continuous replenishment; chooses the A/F provisioning ratio. | Stateful Attention workers retain KV; FFN is stateless. | Trace-calibrated simulation; arbitrary prompt/decode distributions; §3--5. | It already formalizes slowest-worker barrier load and A/F ratio selection. A PAP paper cannot sell the barrier model or ratio search alone; real-system evidence, multi-round state movement, and a distinct control mechanism are required. |
| BF-IO [chen2026universal] | arXiv 2026 | Balances barrier-synchronized DP Decode workers with non-migratable state. | Online assignment from a central waiting pool using current load or short lookahead. | Sticky per-worker KV; migration and preemption are excluded. | Discrete-event simulation from LongBench/BurstGPT and a proprietary trace; §2--7. | It directly precedes KV-token-load and barrier-aware placement. PAP's current peak-gain rule resembles the \(H=0\) regime. A defensible extension would need to exploit efficient KV migration or multi-round post-Prefill boundaries and validate on a real serving system. |
| AFD challenges [liu2026challenges] | arXiv 2026 | Studies Attention--FFN disaggregation for MoE serving. | Joint topology and batch allocation under a fixed stage budget. | Attention-side state; FFN/expert-side weights. | Analytical/measurement-driven comparison with large-scale EP; §2--3. | Establishes that AFD has communication dead zones and discrete imbalance penalties. PAP evaluation must map its valid workload/hardware region instead of implying universal superiority. |
| How Far Can Disaggregation Go? [wu2026howfar] | arXiv 2026 | Explores PD, aggregated, and operator-level AFD for MoE. | Design-space search over topology and three/four-way microbatch overlap. | Attention-side KV separated from FFN/expert weights. | Kernel measurements plus network simulation on 128xB200; workload and SLO sweep; §3--4. | Treats cross-layer microbatch overlap as fundamental. PAP currently forbids layer-interleaved microbatches, so its single-batch design must be justified as a different operating point and evaluated against the lost-overlap cost. |

## Current positioning consequence

The initial literature pass rejects two easy novelty claims:

1. balancing total resident KV across barrier-coupled Attention workers is
   already explicit prior art; and
2. selecting the Attention/Projection-or-FFN ratio from the slowest-worker
   barrier is already modeled.

The open paper question is narrower: whether a real vLLM implementation can
use low-overhead, topology-local KV migration at post-Prefill boundaries to
escape sticky placement in multi-round workloads, while retaining the capacity
and batching benefits of full Attention/KV ownership on PA nodes. This is a
candidate position, not an approved claim.

For each entry:

1. add the primary source to `references.bib`;
2. distinguish the paper's explicit claims from our inference;
3. record the exact section, figure, or experiment that supports the entry;
4. update or remove the entry when later evidence changes the interpretation.
