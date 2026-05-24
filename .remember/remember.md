# Handoff

## State
Branch `feature/pap-true-split`. Latest committed code state before this
handoff update is `a5a0781d1` (`Record PAP experiment results`), after
`fd7b00453` (`Implement PAP attention ready flow`) and `3e4686f85`
(`Remove legacy PAP launcher wrapper`).

The model-agnostic PAP launcher is now `examples/pap/launch_pap_nixl.sh`.
It accepts `--model MODEL_PATH_OR_NAME`, also honors `PAP_MODEL_PATH`, and
defaults the local experiment path to `/data/ssd1/llm-models/Qwen3-0.6B`.
OFFLOAD_EXEC shape parameters are inferred from `config.json`, so both
Qwen3-0.6B and Qwen3-8B can run without hard-coded head/qkv sizes.

PAP attention no longer uses the legacy "true split" naming in code paths.
Projection enters PAP attention only when the batch is decode-only, has one
token per request, uses OpenAI request ids, and every active request has
PA-side attention KV marked ready.

Prefill is the explicit owner of importing prefill KV into the Attention
executor. Projection no longer re-imports Prefill KV, which avoids overwriting
the Attention executor's PA-side cache with Projection-side cache contents.
Prefill imports after local attention writes the current layer's paged KV cache;
this fixed the stale-import issue.

Current runtime still uses the restored NCCL/P2P OFFLOAD_EXEC transport in
`vllm/pap/data_plane.py`. IPC for Profile/Prefill KV to Attention Executor is
the next architectural cleanup, not implemented yet.

New in the working tree: phase-1 KV-unaware Projection work has started.
`docs/design/pap_kv_unaware_projection.md` defines the target contract.
Projection payloads now use `pap_projection_kv_unaware=True` and
`pap_remote_prefix_len` metadata and strip Prefill KV transport fields such as
`remote_block_ids`, `remote_engine_id`, `remote_request_id`, `remote_host`, and
`remote_port`. Scheduler has a PAP Projection branch that treats
`pap_remote_prefix_len - 1` tokens as remote-prefix metadata and bypasses
KVConnector match/update for that request. This is not yet committed and still
needs final review/commit.

## Verified
- Unit/PAP tests: `.venv/bin/python -m pytest tests/pap/test_pap_launch_files.py tests/pap -q`
  passed with `124 passed`.
- E2E 1PA1P using `/data/ssd1/llm-models/Qwen3-0.6B` returned
  `"<think>\nOkay,"` instead of the previous repeated `!`.
- E2E high input/output 1PA1P run used `prompt_tokens=1185`,
  `completion_tokens=256`, completed in `10206 ms`, and produced normal model
  text. Attention/Projection OFFLOAD_EXEC traces were both `7168`, matching
  `256 output tokens * 28 layers`.
- E2E 4PA4P run sent 8 sequential requests and covered PA/P pairs
  `8190/8290`, `8191/8291`, `8192/8292`, and `8193/8293` twice. All requests
  returned HTTP `200`; every Attention and Projection instance logged `448`
  OFFLOAD_EXEC traces, matching `2 requests * 8 output tokens * 28 layers`.
- The 4PA4P responses were valid model text, not garbled output.
- Attention executor logs showed exactly 28 `PAP prefill KV imported` entries
  for Qwen3-0.6B.
- Projection and Attention OFFLOAD_EXEC traces were present; Projection showed
  external prefix cache hit rate `100.0%`.
- E2E PAP/MPS/EngineCore processes were cleaned after verification.
- Current phase-1 unit coverage:
  `.venv/bin/python -m pytest tests/pap/test_pd_payloads.py tests/pap/test_pap_proxy_server.py tests/pap/test_multi_pap_proxy_server.py tests/pap/test_pap_true_split_contract.py tests/v1/core/test_scheduler.py::test_pap_projection_remote_prefix_len_parser -q`
  passed with `43 passed`.
- Phase-1 1PA1P E2E with Qwen3-0.6B returned HTTP `200` for a request with
  `prompt_tokens=628`, `completion_tokens=96`, latency `7004 ms`. Projection
  prompt throughput was `0.0 tokens/s`; Attention and Projection each logged
  `2688` OFFLOAD_EXEC traces, matching `96 * 28`.

## Next
1. For future E2E, use:
   `bash examples/pap/launch_pap_nixl.sh --model /data/ssd1/llm-models/Qwen3-0.6B`
   or override with the 8B path.
2. Run a short E2E after the new proxy `projection_kv_keys` log line is
   committed/restarted to explicitly capture that Projection payload contains
   only PAP metadata keys.
3. Start designing the IPC path for Prefill/Profile KV to Attention Executor.
   The current NIXL/NCCL transport is working but is not the desired final
   local PA-to-Attention KV path.

## Rules
- Use `.venv/bin/python`/`uv`; do not use system `python3` or bare `pip`.
- Poll service startup and logs frequently; do not wait silently on long E2E
  runs.
- Full restart PAP experiments instead of restarting individual services.
- Clean E2E processes by PID and verify with `pgrep` before starting another
  run.
