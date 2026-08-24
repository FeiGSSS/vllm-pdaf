# Qwen3 long-context model-history notes

- Source read:
  `/home/fei/.codex/skills/model-pr-history-knowledge/vllm/qwen3-core/README.en.md`.
- The Qwen3 history covers model registration, weight loading, multimodal
  integration, and speculative decoding, but contains no model-forward patch
  required for YaRN context extension.
- Decision: use the current vLLM configuration path (`hf_overrides` with
  `rope_parameters`) rather than patching `vllm/model_executor/models/qwen3.py`.
- Local configuration validation resolved Qwen3-8B to `max_model_len=131072`
  with YaRN factor 4 and original context 32768.
- Validation lanes implied by the history and this change: model startup,
  long Prefill, KV publication/import, whole-step CUDA Graph decode, output
  completion, and a short-context regression smoke before making the setting
  default.
