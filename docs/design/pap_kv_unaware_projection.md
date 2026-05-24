# PAP KV-Unaware Projection Design

## Goal

Projection should keep using vLLM's production OpenAI server, model loading,
sampling, output processing, batching, and model runner plumbing, but it must not
receive prompt KV data from Prefill/Profile. In PAP mode, Prefill/Profile owns
prompt computation, Attention owns historical KV and attention execution, and
Projection owns stateless per-token projection-side compute.

## Current State

The current prototype still routes Prefill KV descriptors through the Projection
request payload:

- The proxy forwards Prefill's `kv_transfer_params` to Projection.
- Projection runs with the NIXL connector configured as a KV consumer.
- The scheduler has no PAP-specific request phase. It only knows local prefix
  cache hits and KVConnector external hits.
- `pap_attention_kv_installed=True` currently prevents the NIXL connector from
  pulling KV into Projection, but it also returns zero external matched tokens.
  That means Projection does not skip prompt computation through scheduler state.
- PAP attention only takes over once the batch is decode-only, one token per
  request, and Attention KV is marked ready.

This keeps correctness experiments moving, but it is not the intended PAP data
plane. The Projection node remains KV-aware through scheduler admission and
through the remote KV descriptor schema, even when no bytes are pulled.

## Target Contract

Projection receives metadata only:

- `pap_projection_kv_unaware=True`
- `pap_remote_prefix_len=<int>`
- `pap_attention_kv_installed=True`
- `pap_prefill_kv_handle=<str>`
- `pap_attention_tcp_endpoint=<tcp endpoint>`
- `pap_offload_exec_zmq_endpoint=<host:port>`

Projection payload must not include these Prefill KV transport fields:

- `remote_block_ids`
- `remote_engine_id`
- `remote_request_id`
- `remote_host`
- `remote_port`

The `kv_transfer_params` request field is still used for now because it is the
existing vLLM OpenAI metadata plumbing. In Projection KV-unaware mode, it carries
PAP control metadata, not a KV-transfer command.

## Scheduler Semantics

Add a PAP Projection scheduler path for waiting requests with
`pap_projection_kv_unaware=True`.

For a prompt of length `N` where Prefill/Profile has already installed `N` prompt
tokens into Attention:

1. Projection scheduler treats `N - 1` prompt tokens as remotely computed.
2. It allocates local scheduler/block-table metadata for the remote prefix using
   `num_external_computed_tokens=N - 1`, but does not call the KVConnector to
   receive blocks.
3. It schedules exactly one token for the first Projection step: token `N - 1`.
4. Model runner sees `num_computed_tokens=N - 1`, input position `N - 1`,
   query length `1`, and seq len `N`.
5. Qwen3 PAP attention routes Q/K/V/O for that token to Attention through
   OFFLOAD_EXEC.
6. Subsequent decode steps stay on the normal vLLM running-request path and use
   PAP attention for every one-token step.

This is still not "zero local KV allocation": Projection will retain vLLM's
block-table and slot machinery for scheduling the current token and generated
tokens. It is "zero Prefill/Profile-to-Projection KV data": no prompt KV payload,
remote KV blocks, NIXL recv, or Prefill KV descriptor is required by Projection.

## Phase Plan

### Phase 1: Metadata-Only Projection Admission

Files:

- `examples/pap/pd_payloads.py`
- `examples/pap/pap_proxy_server.py`
- `examples/pap/multi_pap_proxy_server.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/worker/gpu/model_runner.py`
- `tests/pap/test_pd_payloads.py`
- `tests/pap/test_pap_proxy_server.py`
- `tests/pap/test_multi_pap_proxy_server.py`
- `tests/v1/core/test_scheduler.py`

Tasks:

1. Add tests proving Projection payload strips remote KV transport fields and
   carries only PAP metadata plus `pap_remote_prefix_len`.
2. Add scheduler tests proving a PAP KV-unaware request schedules only the last
   prompt token on the first step, without entering `WAITING_FOR_REMOTE_KVS`.
3. Implement a scheduler helper that detects and validates
   `pap_projection_kv_unaware`.
4. Use `num_external_computed_tokens=prefix_len - 1` for metadata allocation and
   skip `connector.get_num_new_matched_tokens` / `connector.update_state_after_alloc`
   for that request.
5. Teach model runner to read `pap_remote_prefix_len` in addition to the legacy
   `remote_num_tokens`.
6. Keep existing non-PAP NIXL behavior unchanged.

Verification:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pd_payloads.py \
  tests/pap/test_pap_proxy_server.py \
  tests/pap/test_multi_pap_proxy_server.py \
  tests/v1/core/test_scheduler.py -q
```

Then run the 0.6B PAP experiment with `1pa1p` and confirm:

- Projection logs show no remote KV descriptor fields in its payload.
- Projection does not enter `WAITING_FOR_REMOTE_KVS`.
- First Projection scheduled step is one token at prompt position `N - 1`.
- Output is readable and no PAP errors occur.

### Phase 2: Runtime Cleanup

Remove the old `pap_attention_kv_installed` NIXL short-circuit dependency from
the Projection path. Prefill/Profile may still use NIXL to publish KV metadata
for other modes, but PAP Projection should not depend on that connector for
admission.

### Phase 3: Projection KV Ownership Tightening

Reduce Projection-side KV allocation further once phase 1 is stable:

- Avoid caching remote-prefix metadata as local prefix-cache entries.
- Audit block-table retention and freeing for long-running decode.
- Decide whether generated-token KV on Projection can also become metadata-only
  because Attention is the true KV owner.

### Phase 4: IPC Prefill/Profile to Attention

Replace the remaining Prefill/Profile-to-Attention transfer path with the target
IPC mechanism. Projection should be unaffected because it already consumes only
metadata.
