# Qwen3 Dense Decode Layer Profile

Times are CUDA-event GPU times from the original dense vLLM decode path.
Projection core = qkv_proj + qk_norm_rope + o_proj + mlp.
Projection with LN additionally includes input_layernorm and post_attention_layernorm.

| model | TP | batch | context | attention mean ms/layer | projection core mean ms/layer | projection+LN mean ms/layer | qkv ms | o_proj ms | mlp ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-32B-PD-D | 2 | 64 | 2048 | 0.1172 | 0.8412 | 0.8741 | 0.0926 | 0.1058 | 0.5957 |
