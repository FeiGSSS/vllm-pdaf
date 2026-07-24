# PAP batch-scaling component probe

## Question

When the PA decode batch grows, which per-layer component explains the
sublinear increase in end-to-end ITL: Projection, local-fast transfer, or
Attention?

## Setup

- Runtime commit: `1ce452dc7336efd1444f59695a8fa3e905040da8`
- Model shape: Qwen3-8B FP16
- Hardware: two NVIDIA L20 GPUs
- Sequence length: 8,192 tokens for every request
- Batch sizes: 1, 2, 4, 8, 16, 32
- Sampling: 10 warmups, five samples, 30 calls per sample
- PA split: Attention measured with the production 20-SM MPS partition

Projection uses Qwen3-shaped eager PyTorch operations for QKV,
normalization/RoPE, output projection, and MLP. It is a scaling proxy, not an
exact replay of every vLLM fused kernel. Local-fast measures the actual
bidirectional CUDA P2P payload copies and intentionally excludes
batch-independent control-plane cost. Attention measures the production KV
append operation plus the current PAP paged-decode kernel.

## Results

Median latency is milliseconds per layer.

| Batch | Projection | Local P2P | Attention (20 SM) | Sum | Sum x 36 layers |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5526 | 0.0186 | 0.0832 | 0.6543 | 23.56 |
| 2 | 0.5798 | 0.0188 | 0.1064 | 0.7051 | 25.38 |
| 4 | 0.5807 | 0.0231 | 0.2304 | 0.8342 | 30.03 |
| 8 | 0.5831 | 0.0365 | 0.4453 | 1.0648 | 38.33 |
| 16 | 0.5873 | 0.0394 | 0.8125 | 1.4393 | 51.82 |
| 32 | 0.6306 | 0.0501 | 1.5587 | 2.2394 | 80.62 |

From batch 4 to 8, the reconstructed 36-layer time rises from 30.03 to
38.33 ms/token, or 27.7%. The 8.30 ms/token increase is attributable to:

- Attention: 93.2%
- Local P2P: 5.8%
- Projection: 1.0%

KV append stays near 0.0038 ms/layer. The paged Attention kernel accounts for
essentially all Attention growth. Its logical KV-read bandwidth is 594 GB/s
at batch 4 and 607 GB/s at batch 8, so doubling the batch approximately
doubles KV traffic while bandwidth remains flat.

The same Attention kernel on the full GPU takes 0.1918 ms/layer at batch 4
and 0.3767 ms/layer at batch 8. More SMs improve absolute latency modestly,
but do not change the batch-scaling behavior.

The full-GPU probe was repeated after commit `cb6fe3500`. Projection and
Attention differed by less than 0.3% at every batch size. Local P2P differed
by less than 0.6% except at B4, where the very small measurement moved from
0.0231 to 0.0206 ms. The single-Projection-batch refactor therefore did not
change the same-shape component kernels.

## Conclusion

The observed roughly 30% ITL increase is explained by the current paged
Attention path, not Projection or local-fast. To limit the batch 4 to 8
increase to 10% while other costs stay unchanged, batch-8 Attention must fall
from 0.445 ms/layer to about 0.298 ms/layer: a 33% reduction. That requires
roughly 900 GB/s effective KV-read bandwidth or one-third less KV traffic.

The next targeted experiment should compare Attention kernels/backends at
batch 4 and 8. Local-fast changes alone cannot close this gap: removing its
entire measured batch-8 cost would save only about 1.31 ms/token.

## Local artifacts

Raw sample files are colocated under `raw/` and remain Git-ignored according
to the benchmark retention policy:

- `full_gpu.json`
- `attention_20sm.json`
- `post_single_batch_full_gpu.json`

The reusable probe is
`benchmarks/pap/tooling/batch_scaling_probe.py`.
