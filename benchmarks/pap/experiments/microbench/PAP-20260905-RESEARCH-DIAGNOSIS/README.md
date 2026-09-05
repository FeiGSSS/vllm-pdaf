# PAP architecture research diagnosis

Question: which exact-inference mechanism can improve PAP beyond the current
PAT + Dynamo + NVSHMEM Graph implementation, with a defensible distinction
from prior work and a falsifiable end-to-end benefit?

Source baseline: `306b75a894`. This record contains source analysis, literature
retrieval, existing-trace reanalysis and bounded component probes. It is not a
new full serving benchmark or a claim of a measured production speedup.
Runtime code and the immutable workload datasets remain unchanged.

Hardware scope: one 8-GPU NVIDIA L20 PCIe node, Qwen3-8B FP16, 131K context.
Separate historical results from this source revision; every measurement must
identify its source, workload, unit, configuration and timing boundaries.

Artifacts:

- `source_audit.md`: current execution and ownership evidence.
- `evidence_audit.md`: rechecked retained performance evidence.
- `literature/`: primary-paper retrieval and novelty risks.
- `ideaspark_run/pap-inference-architecture/`: structured idea audit workflow.

No paper-acceptance or novelty guarantee is made. Candidate claims remain
conditional until their negative controls and end-to-end tests pass.

Current outcome: see [research_report.md](research_report.md). Historical KV
aliasing blocks the old correct-inference performance attribution. The selected
implementation uses Attention-initiated Decode allocation plus generation-fenced
Prefill revocation and passed a bounded structural E2E checkpoint. Independent
end-to-end numerical equivalence and saturation backpressure remain open. The
idea-spark workflow stopped at diagnosis rather than inventing a new mechanism.
Primary-paper full-text caches are retained locally and excluded from commits.
