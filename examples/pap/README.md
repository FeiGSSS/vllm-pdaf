# PAP NIXL Qwen3-8B Experiment

This directory contains the first runnable PAP experiment for vLLM.

PAP separates decode responsibilities into explicit roles:

- **Prefill** is a normal vLLM prefill server using `NixlConnector` with `kv_producer`.
- **Attention** is an internal executor colocated with Prefill. It records the prefill KV handle, owns PAP attention control-plane state, and receives shape-only shadow events from Projection at the real Qwen3 `q/k/v -> self.attn` boundary. It is not an OpenAI API server.
- **Projection** is a vLLM decode server using `NixlConnector` with `kv_consumer`. It owns decode projection work, `lm_head`, and sampling for the runnable first experiment.

The current implementation is a runnable PAP data-plane prototype. It keeps vLLM's standard NIXL KV transfer path for Prefill -> Projection KV movement, then adds a layer-level remote Attention output path. Projection still runs local `self.attn(q, k, v)` first so vLLM updates its local KV cache and has a fallback output; when `pap_remote_attention=true`, it gathers the updated paged KV cache, calls the internal Attention executor, and feeds the returned attention output into Qwen3 `o_proj`. This proves the PAP split and control-plane shape, but it is not yet the performance-clean version because local attention compute has not been removed.

Attention records the NIXL KV handle returned by Prefill, including `remote_block_ids`, receives Qwen3 q/k/v layer-boundary events, and serves `/v1/pap/attention/compute` for remote attention output. The prototype supports one decode request per remote attention call and uses HTTP/base64 tensor payloads for debuggability; the next implementation step is replacing this with local IPC/NIXL-style tensor transport and splitting KV update from local attention compute.

Run:

```bash
PAP_MAX_TOKENS=8 bash examples/pap/launch_pap_qwen3_8b_nixl.sh
```

Run Prefill and Attention under CUDA MPS on the same physical GPU:

```bash
PAP_ENABLE_MPS=1 \
PAP_PREFILL_GPU=0 \
PAP_PROJECTION_GPU=1 \
PAP_PREFILL_MPS_PERCENT=70 \
PAP_ATTENTION_MPS_PERCENT=30 \
PAP_MAX_TOKENS=8 \
bash examples/pap/launch_pap_qwen3_8b_nixl.sh
```

The script starts a private `nvidia-cuda-mps-control` daemon when `PAP_ENABLE_MPS=1`, sets `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` separately for Prefill and Attention, and passes the MPS pipe only to those two colocated processes. Projection remains a normal vLLM process on its own GPU. MPS controls active SM thread share; it does not provide a hard HBM-bandwidth reservation, so memory capacity is still controlled by vLLM's `--gpu-memory-utilization` and the natural PAP data locality.

Proxy view of one request:

```text
client -> PAP proxy :9000
proxy  -> Prefill :8100      one-token prefill, NIXL producer
proxy  -> Attention :8300    register request_id -> prefill KV handle
proxy  -> Projection :8200   original request + kv_transfer_params
Projection -> Prefill        NIXL KV pull data-plane
Projection -> Attention      q/k/v boundary events + remote attention compute
Attention  -> Projection     attention output used before Qwen3 o_proj
```

Defaults:

- Model: `/data/ssd1/llm-models/Qwen3-8B`
- Prefill port: `8100`
- Projection port: `8200`
- Attention internal executor port: `8300`
- Proxy port: `9000`
- Prefill GPU: `0`
- Projection GPU: `1`

The proxy exposes `/v1/completions` and `/v1/chat/completions`. It sends a one-token non-streaming request to Prefill, registers the returned `kv_transfer_params` with Attention, and forwards the original request plus those KV parameters to Projection.


## Current Verification Status

The latest consistency harness run used Qwen3-8B with `PAP_MAX_MODEL_LEN=1024`,
`PAP_MAX_NUM_SEQS=2`, `PAP_MAX_TOKENS=8`, and NIXL for both native PD and PAP.
It produced these deterministic one-request outputs:

| Architecture | Output |
| --- | --- |
| Fused vLLM | ` What is the purpose of the PAP` |
| Native vLLM PD | ` What is the purpose of PAP?` |
| PAP | ` What is the purpose of PAP?` |

This means the runnable PAP prototype currently matches native vLLM PD for the
same NIXL prefill/decode split. Both disaggregated paths differ from the fused
single-instance baseline on this prompt, while token counts and finish reasons
match across all three paths.

Run the comparison harness with:

```bash
PAP_ENABLE_MPS=1 \
PAP_MAX_TOKENS=8 \
PAP_MAX_MODEL_LEN=1024 \
PAP_MAX_NUM_SEQS=2 \
bash examples/pap/run_arch_consistency_qwen3_8b_nixl.sh
```

The harness writes per-architecture logs and `compare.json` under
`examples/pap/logs/consistency/<run-id>/`. Runtime artifacts under
`examples/pap/logs/` are intentionally ignored by git.
