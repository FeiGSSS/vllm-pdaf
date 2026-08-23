# Qwen3 model-history notes for the v0.26 port

Date: 2026-08-24

## Sources

- `vllm/qwen3-core/README.en.md` from the model PR history knowledge base.
- Official `v0.26.0` Qwen3 and generic Attention sources.
- PAP donor `feature/pap` at `5c78ea8c4d`; runtime performance authority
  `a1d8ec918`.

The history cards cover Qwen3 support, model loading, and speculative decode.
They do not identify a dense Qwen3-8B TP1 optimization that justifies
replacing the v0.26 model with the old PAP fork.

## Decisions influenced

- Preserve the upstream Qwen3/Qwen2 implementation and its validation lanes.
- Do not port the old `qwen3.py` fork, legacy V1 runner, Qwen3 MoE guards, or
  speculative-decode changes.
- Intercept normalized/rotated Q/K/V through a generic Attention execution
  interface so future dense models can register without copying model files.
- Keep an optional packed-QKV optimization separate from correctness and
  validate it with a causal ablation before enabling it.
- Rerun Qwen3 model loading/generation smoke after the generic interface and
  again after any packed-QKV specialization.
