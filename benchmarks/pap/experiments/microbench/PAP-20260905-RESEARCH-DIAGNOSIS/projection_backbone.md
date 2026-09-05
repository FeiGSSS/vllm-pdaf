# Fresh Projection component measurement

Command: `CUDA_VISIBLE_DEVICES=7 .venv/bin/python benchmarks/pap/experiments/microbench/PAP-20260905-RESEARCH-DIAGNOSIS/probe_projection.py --output benchmarks/pap/experiments/microbench/PAP-20260905-RESEARCH-DIAGNOSIS/projection_backbone.json`.

One idle L20 (92 SM), FP16, Qwen3-8B shapes, 36 distinct layer weight sets,
CUDA Graph replay, seven samples of five replays after warmup. Capture replay
matches eager outputs within `rtol=atol=1e-3`. Raw samples and source hash are
in `projection_backbone.json`.

The region is O projection, residual RMS norm, gate/up, SiLU, down projection,
residual RMS norm and next QKV. It **excludes** Q/K normalization, RoPE, final
vocabulary projection, attention, NVSHMEM and CPU scheduling. These are dense
operator service times, not whole-step TBT or a complete compiled serving graph.
The dummy inter-layer dependency is not a proposed attention approximation.

| Total requests | One batch, ms/layer | Two sequential halves, combined ms/layer |
| --- | ---: | ---: |
| 8 | 0.571 | 1.138 |
| 16 | 0.575 | 1.143 |
| 32 | 0.625 | 1.151 |
| 64 | 0.655 | 1.250 |
| 128 | 0.705 | 1.311 |

Observed: B=16 to B=64 increases the dense backbone's per-layer service time
by only 13.8%, whereas splitting B=32 into two B=16 invocations increases its
combined GPU service demand by 84.1%. Thus splitting a fixed resident batch
does not halve Projection cost. The two-half column is not a prediction of
the pipelined makespan: real overlap also depends on Attention and dependencies.

Each layer has 192,937,984 dense weight elements (385,875,968 FP16 bytes),
13,891,534,848 bytes across 36 layers. Repeated invocations increase logical
weight demand; the JSON's logical GB/s is a workload-derived metric, not a
hardware-counter measurement of HBM traffic. These weights exceed L20 cache
capacity by far, but only hardware counters can establish exact traffic.

This probe is independent of the discovered PAP Decode KV ownership defect.
It does not validate old serving throughput, PAT's physical-prefix statistics,
or any proposed end-to-end optimization.

Initial attempt failed before timing because the probe used an obsolete SiLU
Python wrapper name. The saved script calls the same registered native operator
as current `SiluAndMul.forward_cuda`; no runtime fallback or override was used.
