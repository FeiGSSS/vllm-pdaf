# Model-history screening

Read `/home/fei/.codex/skills/model-pr-history-knowledge/vllm/qwen3-core/README.en.md`
before selecting a model-specific probe. The local index covers Qwen3 dense,
MoE and draft-model paths; these must not be conflated.

- PR #39419 concerns draft-model vocabulary/TP communication. This PAP point
  uses dense target decoding with TP=1 and no speculative decoding, so that
  optimization is not evidence of an available Projection latency reduction.
- PR #43167 consolidates KV scale loading. It does not justify claiming a
  speedup for the current unquantized FP16 path.
- PR #40671 and #24727 mostly concern MoE/VL model integration, not Qwen3-8B's
  dense MLP service demand. Treating their history as dense-kernel evidence
  would be a category error.

The index is a discovery aid, not proof about the installed revision. Source
inspection at `306b75a894` establishes that Qwen3 uses `Qwen2MLP` and that CUDA
unquantized linear dispatch calls `torch.nn.functional.linear`. The component
probe therefore uses these dense GEMMs and existing vLLM normalization/SiLU
operators. It does not substitute a quantized or speculative model.

No model-family kernel or weight-loader changes were made for this diagnosis.
