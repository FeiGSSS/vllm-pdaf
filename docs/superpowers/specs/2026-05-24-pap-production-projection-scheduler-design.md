# PAP Production Projection Scheduler Design

## Purpose

PAP should reuse vLLM production capabilities wherever they are already the
right abstraction: OpenAI-compatible request handling, tokenizer integration,
model weight loading, batching, sampling, output processing, and the GPU model
runner. The Projection role should not become a separate hand-written model
server.

The required change is narrower and deeper: the Projection scheduler must
understand a PAP request state where prompt prefix progress is owned remotely by
the colocated PA/Attention side. Projection must schedule decode work from that
remote prefix without receiving prompt KV blocks or consulting a KVConnector for
prompt KV.

## Current Data Plane

The current committed state already removed Prefill-to-Projection prompt
KV transport:

- The proxy sends Projection only PAP metadata such as
  `pap_projection_kv_unaware`, `pap_remote_prefix_len`,
  `pap_attention_kv_installed`, `pap_prefill_kv_handle`,
  `pap_attention_tcp_endpoint`, and `pap_offload_exec_zmq_endpoint`.
- Projection starts as a regular vLLM server without NIXL
  `--kv-transfer-config`.
- Projection scheduler treats `pap_remote_prefix_len - 1` tokens as remote
  prefix progress and schedules the last prompt token as the first local
  Projection step.
- Projection no longer receives Prefill KV transport fields such as
  `remote_block_ids`, `remote_engine_id`, `remote_request_id`, `remote_host`, or
  `remote_port`.

There is still one prompt-KV data path in the system, but it is not
Projection-related:

- Prefill imports prompt KV into the colocated Attention executor from
  `Qwen3Attention._maybe_import_pap_prefill_kv_to_attention`.
- That path calls `import_prefill_kv_from_paged_cache` in
  `vllm/pap/shadow_attention.py`.
- The current implementation gathers paged KV into contiguous tensors,
  serializes the tensors, and sends them over the Attention TCP control endpoint.

That remaining path should become PAP `OFFLOAD_KV` over CUDA IPC or an
equivalent same-GPU shared-memory mechanism. Projection should remain out of
that data path.

## Target Projection Architecture

Projection remains a normal vLLM production engine with a PAP scheduling mode.

Reused vLLM components:

- OpenAI API server and request parsing.
- Tokenizer and prompt token accounting.
- Model weight loading.
- GPU model runner, forward context, CUDA graph plumbing, sampling, logprobs,
  stop handling, and output streaming.
- Normal running-request decode loop after the first PAP decode step.

PAP-specific Projection behavior:

- Admission accepts PAP metadata in `kv_transfer_params`; in PAP Projection mode
  this field is control metadata, not a KV-transfer command.
- Scheduler validates `pap_projection_kv_unaware=True` and a positive
  `pap_remote_prefix_len`.
- Scheduler treats `pap_remote_prefix_len - 1` as remote computed prefix
  progress.
- Scheduler allocates only the local metadata/block slots needed for vLLM to
  execute the current token and future generated tokens.
- Scheduler never asks a KVConnector to match, receive, or update remote prompt
  KV for this request.
- Model runner forwards PAP Attention endpoints and readiness metadata so the
  attention layer routes decode Q/K/V/O through PAP Attention.

This is KV-unaware in the important data-plane sense: Projection does not
receive prompt KV bytes or prompt KV handles. It may still keep vLLM-internal
slot/block-table state because the production model runner and attention
metadata builders require sequence progress and slot mappings.

## Scheduler State Model

For a request with prompt length `N` and `pap_remote_prefix_len=N`:

1. Prefill computes the prompt and installs prompt KV into Attention.
2. Proxy sends Projection a metadata-only request.
3. Projection scheduler sets effective computed progress to `N - 1`.
4. Projection schedules exactly one token, the last prompt token at position
   `N - 1`.
5. The Qwen3 attention layer sees a decode-only one-token step and calls PAP
   Attention instead of local attention.
6. Attention computes over the prompt KV it already owns and the current
   token's Q/K/V from Projection.
7. Projection receives attention output, continues through O projection, MLP,
   logits, sampling, and output processing.
8. Subsequent generated-token steps follow the normal vLLM running-request path
   while PAP Attention remains the attention backend.

This lets vLLM's request lifecycle stay intact while changing only the meaning
of "computed prefix" for PAP Projection requests.

## OFFLOAD_KV IPC Boundary

Prefill to Attention should become an explicit local `OFFLOAD_KV`
channel.

Control metadata:

- request id
- layer name
- sequence length
- block ids or slot mapping needed by Attention
- dtype, shape, layout, and device metadata
- CUDA IPC handle metadata or shared-cache descriptor
- readiness and acknowledgement state

Tensor data:

- Prompt KV must not be serialized into HTTP/TCP payloads.
- Prompt KV must not pass through Projection.
- Initial implementation may copy from an opened CUDA IPC tensor into
  Attention's existing registry storage to preserve current semantics.
- Later implementation can optimize to zero-copy shared ownership once lifetime
  and mutation rules are proven.

The first IPC implementation should reuse existing vLLM CUDA IPC patterns where
possible, especially `torch.multiprocessing.reductions.reduce_tensor` /
`rebuild_cuda_tensor` as used by vLLM weight transfer code. That keeps the first
version in Python and avoids adding a custom CUDA extension before correctness is
settled.

## Approaches Considered

### Recommended: vLLM Production Projection plus PAP Scheduler State

Keep Projection as a vLLM server and add a first-class PAP scheduler state for
remote prefix progress. Implement Prefill-to-Attention KV installation
as a separate `OFFLOAD_KV` IPC path.

Pros:

- Preserves vLLM's production API, model loading, batching, sampling, and output
  behavior.
- Keeps Projection free of prompt KV data.
- Limits the risky change to scheduler semantics and PAP model-runner metadata.
- Matches current successful 1PA1P and 4PA2P metadata-only experiments.

Cons:

- Projection still carries some vLLM slot/block metadata for the remote prefix.
- Scheduler code needs careful tests because `num_computed_tokens` is a shared
  abstraction used by prefix caching, chunked prefill, speculative decode, and
  KVConnector integration.

### Alternative: Separate Stateless Projection Service

Build a custom Projection process that loads model weights and implements only
projection-side decode compute.

Pros:

- Could make the Projection process conceptually cleaner and fully KV-free.
- Could eventually remove vLLM block-table assumptions from Projection.

Cons:

- Reimplements large production surfaces: model loading, API handling, batching,
  sampling, output streaming, and operational behavior.
- Higher correctness risk and slower iteration.
- Not needed to remove prompt KV data from Projection.

### Alternative: Keep Projection as a KVConnector Consumer

Continue using vLLM's KVConnector abstractions and make the connector return
remote prefix hits without transferring data.

Pros:

- Minimal scheduler changes.
- Stays close to existing P/D disaggregation hooks.

Cons:

- Keeps Projection semantically KV-aware.
- Makes PAP metadata look like ordinary KV-transfer state.
- Makes it too easy for future changes to reintroduce prompt KV descriptors or
  connector receives on Projection.

## Implementation Phases

### Phase 1: Solidify Projection Scheduler Role

- Keep the already working metadata-only Projection path.
- Rename remaining internal PAP attention naming away from legacy "true split"
  where it affects new code and tests.
- Add focused tests for scheduler behavior:
  - `pap_projection_kv_unaware` requires `pap_remote_prefix_len`.
  - Projection does not call KVConnector match/update.
  - First scheduled step starts from `pap_remote_prefix_len - 1`.
  - Non-PAP KVConnector behavior is unchanged.
- Add tests that Projection payloads contain no Prefill KV transport fields.

### Phase 2: Add OFFLOAD_KV IPC Descriptor Path

- Add a PAP IPC descriptor dataclass for Prefill-to-Attention KV install.
- Add a control command to Attention executor for IPC descriptors.
- Export gathered Prefill KV tensors with CUDA IPC handles instead of
  serializing tensor bytes.
- Import the IPC tensors in Attention and copy into the existing Attention
  registry storage.
- Keep the old TCP tensor bundle path as a debug fallback only, gated by an
  explicit environment setting.

### Phase 3: Make IPC the Default PAP KV Install Path

- Update launcher defaults to use CUDA IPC for PAP `OFFLOAD_KV`.
- Add E2E validation for 1PA1P and X:Y topologies with Projection still
  metadata-only.
- Add log assertions that Prefill-to-Attention import uses IPC
  descriptors and not tensor bundles.

### Phase 4: Tighten Projection Local KV Assumptions

- Audit whether generated-token KV on Projection is still needed once Attention
  owns all attention history.
- Avoid caching remote-prefix metadata as reusable local prefix-cache entries.
- Decide whether Projection can use a smaller synthetic block-table state for
  PAP requests without disturbing vLLM production paths.

## Test Plan

Unit tests:

- PAP payload tests for metadata-only Projection fields.
- Scheduler tests for remote-prefix progress and KVConnector bypass.
- Attention executor tests for IPC descriptor validation and import semantics.
- Data-plane tests for descriptor serialization and invalid shape/dtype errors.

E2E tests:

- 1PA1P with Qwen3-0.6B, high input and high output.
- 4PA2P or 4PA4P with multiple sequential requests to verify routing coverage.
- Log checks:
  - Projection starts with no `kv_transfer_config`.
  - Projection payload has only PAP metadata.
  - Prefill-to-Attention prompt KV install uses IPC descriptors.
  - No TCP tensor-bundle prompt KV import in default PAP mode.
  - Projection and Attention OFFLOAD_EXEC trace counts match
    `completion_tokens * num_layers`.

## Open Constraints

- The first IPC implementation should prefer correctness and debuggability over
  zero-copy lifetime complexity.
- CUDA IPC handle lifetime must be explicit: Prefill keeps exported
  buffers alive until Attention acknowledges import.
- A conservative stream synchronization point is acceptable for the first IPC
  version. CUDA event IPC can be added after correctness is stable.
- The design intentionally does not make Projection a separate custom model
  server.
