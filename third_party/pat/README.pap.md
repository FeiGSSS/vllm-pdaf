# Vendored PAT kernel

This directory is a source snapshot of
`MachineLearningSystem/26ASPLOS-PAT` at commit
`b61e589cc8775930931157ff3bb107ba28bafd77`.  The upstream MIT license and
NOTICE are preserved in this directory.

PAP carries the following L20/CUDA Graph adaptations:

- compile for SM89;
- launch the single-stream path and gather kernel on PyTorch's current stream;
- expose scheduler tensors required by graph-stable metadata buffers;
- honor the caller-provided PAT tile buckets.

The source is built into the project's Python environment by
`benchmarks/pap/scripts/build_pat_attention.sh`.  Compiled objects and shared
libraries are not vendored.
