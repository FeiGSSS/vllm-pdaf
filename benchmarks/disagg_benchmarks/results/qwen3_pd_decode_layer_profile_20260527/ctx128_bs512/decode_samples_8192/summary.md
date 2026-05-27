# Qwen3 Dense Decode Layer Profile

Times are CUDA-event GPU times from the original dense vLLM decode path.
Projection core = qkv_proj + qk_norm_rope + o_proj + mlp.
Projection with LN additionally includes input_layernorm and post_attention_layernorm.

| model | TP | batch | context | attention mean ms/layer | projection core mean ms/layer | projection+LN mean ms/layer | qkv ms | o_proj ms | mlp ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-32B-PD-D | 2 | 512 | 128 | 0.1033 | 0.9639 | 0.9972 | 0.0999 | 0.1410 | 0.6758 |
