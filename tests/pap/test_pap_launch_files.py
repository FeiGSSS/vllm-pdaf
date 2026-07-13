from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pap_benchmark_runner_captures_attention_fast_path_stats() -> None:
    runner = (
        ROOT
        / ".claude"
        / "skills"
        / "vllm-pap-benchmark"
        / "scripts"
        / "run_pap_same_pd_workload.sh"
    )
    text = runner.read_text()

    assert "PAP_DECODE_SLOT_PLAN_CACHE_LIMIT" in text
    assert "/v1/pap/attention/stats" in text
    assert "attention_fast_path_stats.json" in text
    assert "capture_attention_fast_path_stats" in text
    assert (
        'PAP_ATTENTION_DISPATCH_MODE="${PAP_ATTENTION_DISPATCH_MODE:-legacy}"' in text
    )
    assert (
        'PAP_ATTENTION_COMBINE_WAIT_US="${PAP_ATTENTION_COMBINE_WAIT_US:-'
        '${DEFAULT_ATTENTION_COMBINE_WAIT_US}}"' in text
    )
    assert "DEFAULT_ATTENTION_COMBINE_WAIT_US=200" in text
    assert "DEFAULT_ATTENTION_COMBINE_WAIT_US=1000" in text
    assert 'PAP_BATCHED_ROUTE_COPY="${PAP_BATCHED_ROUTE_COPY:-1}"' in text
    assert "export PAP_BATCHED_ROUTE_COPY" in text
    assert "printf 'PAP_BATCHED_ROUTE_COPY=%q\\n'" in text
    assert '"batched_route_copy"' in text
    assert '"attention_dispatch_mode"' in text
    assert '"attention_combine_wait_us"' in text
    assert "DEFAULT_ATTENTION_ACTIVE_PEER_TRACKING=0" in text
    assert "DEFAULT_ATTENTION_ACTIVE_PEER_TRACKING=1" in text
    assert (
        'PAP_ATTENTION_ACTIVE_PEER_TRACKING="'
        "${PAP_ATTENTION_ACTIVE_PEER_TRACKING:-"
        '${DEFAULT_ATTENTION_ACTIVE_PEER_TRACKING}}"' in text
    )
    assert "export PAP_ATTENTION_ACTIVE_PEER_TRACKING" in text
    assert "printf 'PAP_ATTENTION_ACTIVE_PEER_TRACKING=%q\\n'" in text
    assert '"attention_active_peer_tracking"' in text


def test_pap_runner_captures_projection_deferred_trace_after_drain() -> None:
    runner = (
        ROOT
        / ".claude"
        / "skills"
        / "vllm-pap-benchmark"
        / "scripts"
        / "run_pap_same_pd_workload.sh"
    )
    text = runner.read_text(encoding="utf-8")

    assert 'PAP_DEFERRED_CUDA_TRACE="${PAP_DEFERRED_CUDA_TRACE:-0}"' in text
    assert "PAP_DEFERRED_TRACE_ROLE=projection" in text
    assert "PAP_DEFERRED_TRACE_OUTPUT=" in text
    assert "capture_projection_deferred_traces" in text
    assert "validate_deferred_trace.py" in text
    assert "projection_deferred_trace.json" in text
    assert '"${output_path}.flush"' in text
    assert text.rindex("wait_attention_sessions_drained") < text.rindex(
        "capture_projection_deferred_traces"
    )


def test_pap_benchmark_runner_supports_arbitrary_xy_topology() -> None:
    runner = (
        ROOT
        / ".claude"
        / "skills"
        / "vllm-pap-benchmark"
        / "scripts"
        / "run_pap_same_pd_workload.sh"
    )
    text = runner.read_text()

    assert "^([0-9]+)pa([0-9]+)p$" in text
    assert 'PA_COUNT="${PAP_PA_COUNT:-${BASH_REMATCH[1]}}"' in text
    assert 'PROJECTION_COUNT="${PAP_PROJECTION_COUNT:-${BASH_REMATCH[2]}}"' in text
    assert "This runner is intentionally fixed to 1pa1p" not in text
    assert "for (( idx=0; idx<PA_COUNT; idx++ ))" in text
    assert "for (( idx=0; idx<PROJECTION_COUNT; idx++ ))" in text
    assert "topology_manifest.json" in text
    assert "routing_audit.json" in text
    assert "audit_xy_routes" in text
    assert 'PAP_ROUTING_POLICY="${PAP_ROUTING_POLICY:-round_robin}"' in text


def test_pap_benchmark_runner_supports_two_turn_prefix_cache_audit() -> None:
    runner = (
        ROOT
        / ".claude"
        / "skills"
        / "vllm-pap-benchmark"
        / "scripts"
        / "run_pap_same_pd_workload.sh"
    )
    text = runner.read_text()

    assert 'PAP_BENCH_CLIENT_MODE="${PAP_BENCH_CLIENT_MODE:-canonical}"' in text
    assert "multiturn_prefix_cache" in text
    assert "pap_multiturn_prefix_cache.py" in text
    assert "--enable-prompt-tokens-details" in text
    assert "multiturn_prefix_cache.json" in text
    assert (
        "canonical | multiturn_prefix_cache | multiturn_chat_prefix_cache"
        in text
    )
    assert "pap_multiturn_chat_prefix_cache.py" in text
    assert "multiturn_chat_prefix_cache.json" in text
    assert 'PAP_PREFIX_CACHE_AUDIT="${PAP_PREFIX_CACHE_AUDIT:-0}"' in text
    assert 'PAP_PREFIX_CACHE_AUDIT="${PAP_PREFIX_CACHE_AUDIT}"' in text


def test_pap_runner_supports_multiturn_north_star() -> None:
    runner = (
        ROOT
        / ".claude"
        / "skills"
        / "vllm-pap-benchmark"
        / "scripts"
        / "run_pap_same_pd_workload.sh"
    )
    text = runner.read_text()

    assert "multiturn_north_star" in text
    assert "benchmarks/multi_turn/pap_pd_multiturn_client.py" in text
    assert 'PAP_VLLM_DTYPE="${PAP_VLLM_DTYPE:-auto}"' in text
    assert text.count('--dtype "${PAP_VLLM_DTYPE}"') == 3
    assert '[[ "${TOPOLOGY}" == "1pa1p" ]]' in text
    assert '[[ "${INPUT_LEN}" == "16000" ]]' in text
    assert '[[ "${OUTPUT_LEN}" == "256" ]]' in text
    assert '[[ "${MAX_MODEL_LEN}" == "20000" ]]' in text
    assert '[[ "${MAX_NUM_BATCHED_TOKENS}" == "4096" ]]' in text
    assert '[[ "${MAX_NUM_SEQS}" == "2" ]]' in text
    assert '[[ "${PAP_PREFIX_CACHE_AUDIT}" == "0" ]]' in text
    assert '[[ "${PAP_ENABLE_PROMPT_TOKENS_DETAILS}" == "1" ]]' in text
    assert '--architecture "pap"' in text
    assert '--topology "${TOPOLOGY}"' in text
    assert '--hardware-signature "${PAP_NORTH_STAR_HARDWARE_SIGNATURE}"' in text
    assert '--git-commit "${GIT_COMMIT}"' in text
    assert '--git-tracked-worktree-dirty "${GIT_TRACKED_WORKTREE_DIRTY}"' in text
    assert '--offload-exec-transport "${PAP_OFFLOAD_EXEC_TRANSPORT}"' in text
    assert '--direct-mailbox-output "${PAP_DIRECT_MAILBOX_OUTPUT}"' in text
    assert '--unified-md-fast-key "${PAP_UNIFIED_MD_FAST_KEY}"' in text
    assert 'PAP_UNIFIED_MD_FAST_KEY="${PAP_UNIFIED_MD_FAST_KEY:-1}"' in text
    assert "PAP_UNIFIED_MD_FAST_KEY=%q" in text
    assert '--result "${RUN_ROOT}/result.json"' in text
    assert "validate_north_star_result" in text
    assert "finalize_pap_pd_multiturn.py" in text
    assert "git diff --cached --binary" in text
    assert "CUDA out of memory" in text
    assert "EngineDeadError" in text
    assert text.index("validate_north_star_result") < text.rindex(
        "wait_attention_sessions_drained"
    )


def test_multiturn_north_star_orchestrator_is_fixed_and_serial() -> None:
    script = (
        ROOT
        / ".claude"
        / "skills"
        / "vllm-pap-benchmark"
        / "scripts"
        / "run_multiturn_north_star.sh"
    )
    assert script.is_file()
    text = script.read_text()

    assert 'MODE="${1:-quick}"' in text
    assert "REPETITIONS=1" in text
    assert "REPETITIONS=3" in text
    assert "for (( rep=1; rep<=REPETITIONS; rep++ ))" in text
    assert "run_pap_same_pd_workload.sh" in text
    assert "PAP_BENCH_CLIENT_MODE=multiturn_north_star" in text
    assert "PAP_TOPOLOGY=1pa1p" in text
    assert "PAP_PREFILL_GPUS=1" in text
    assert "PAP_PROJECTION_GPUS=2" in text
    assert "INPUT_LEN=16000" in text
    assert "OUTPUT_LEN=256" in text
    assert "MAX_MODEL_LEN=20000" in text
    assert "MAX_NUM_BATCHED_TOKENS=4096" in text
    assert "MAX_NUM_SEQS=2" in text
    assert "PAP_VLLM_DTYPE=float16" in text
    assert "PAP_OFFLOAD_EXEC_TRANSPORT=local_fast" in text
    assert "PAP_DIRECT_MAILBOX_OUTPUT=1" in text
    assert "PAP_OFFLOAD_KV_TRANSPORT=cuda_ipc" in text
    assert "PAP_LOCAL_FAST_STREAM_ORDERED=1" in text
    assert "PAP_LOCAL_FAST_SLOT_COUNT=2" in text
    assert "PAP_PREFIX_CACHE_AUDIT=0" in text
    assert "PAP_ENABLE_PROMPT_TOKENS_DETAILS=1" in text
    assert "PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS=256" in text
    assert "PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE" in text
    assert "nvidia-smi" in text
    assert "compare_pap_pd_multiturn.py" in text
    assert " aggregate " in text
    assert " compare " in text
    assert "REFERENCES_READY" in text
    assert "validate-references" in text
    assert "write-reference" not in text
    assert "pkill" not in text
    assert "python3" not in text
    assert "pip install" not in text


def test_pd_multiturn_reference_bootstrap_uses_unchanged_official_proxy() -> None:
    script = (
        ROOT
        / ".claude"
        / "skills"
        / "vllm-pap-benchmark"
        / "scripts"
        / "bootstrap_pd_multiturn_reference.sh"
    )
    assert script.is_file()
    text = script.read_text()
    auditor = ROOT / "benchmarks" / "multi_turn" / "pd_multiturn_reuse_metrics.py"
    audit_text = auditor.read_text()

    assert "REPETITIONS=3" in text
    assert "examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py" in text
    assert "pap_pd_multiturn_client.py" in text
    assert '--architecture "pd"' in text
    assert '--topology "1p1d"' in text
    assert "--unified-md-fast-key 0" in text
    assert "CUDA_VISIBLE_DEVICES=1" in text
    assert "CUDA_VISIBLE_DEVICES=2" in text
    assert '"kv_role":"kv_producer"' in text
    assert '"kv_role":"kv_consumer"' in text
    assert '"bidirectional_kv_xfer":true' not in text
    assert '"bidirectional_kv_xfer":false' in text
    assert "git diff --cached --quiet" in text
    assert "git diff --cached --binary" in text
    assert "finalize_pap_pd_multiturn.py" in text
    assert "--passed-gate pd_reuse_metrics" in text
    assert "prefill_metrics.prom" in text
    assert "decode_metrics.prom" in text
    assert "official_streaming_one_way_metrics_passed" in audit_text
    assert "pd_multiturn_reuse_metrics.py" in text
    assert "local_cache_hit" in audit_text
    assert "external_kv_transfer" in audit_text
    assert "cache MISS" in audit_text
    assert "/tmp/pap_pd_multiturn_reference_candidate.json" in text
    assert "write-reference" not in text
    assert "pkill" not in text
    assert "python3" not in text
    assert "pip install" not in text


def test_pd_benchmark_runner_is_self_contained_and_scoped() -> None:
    skill_dir = ROOT / ".claude" / "skills" / "vllm-pap-benchmark"
    runner = skill_dir / "scripts" / "run_pd_same_workload.sh"
    proxy = skill_dir / "scripts" / "nixl_pd_proxy.py"
    text = runner.read_text()

    assert proxy.is_file()
    assert 'PROXY_SCRIPT="${SKILL_DIR}/scripts/nixl_pd_proxy.py"' in text
    assert "launch_service.sh" not in text
    assert "run_benchmark.sh" not in text
    assert "pkill" not in text
    assert "nvidia-smi" not in text
    assert "PD_PREFILL_GPU:-1" in text
    assert "PD_DECODE_GPU:-2" in text
    assert 'QPS="${QPS:-16}"' in text
    assert 'VLLM_DTYPE="${PD_VLLM_DTYPE:-float16}"' in text


def test_pap_readme_documents_projection_as_decode_owner() -> None:
    readme = ROOT / "examples" / "pap" / "README.md"
    text = readme.read_text()

    assert "Projection" in text
    assert "Q/K/V" in text
    assert "Attention" in text
    assert "internal executor" in text


def test_pap_6pa2p_launch_uses_multi_proxy_and_expected_counts() -> None:
    script = ROOT / "examples" / "pap" / "launch_pap_nixl.sh"
    text = script.read_text()

    assert 'MODEL_PATH="${PAP_MODEL_PATH:-/data/ssd1/llm-models/Qwen3-0.6B}"' in text
    assert "-m | --model" in text
    assert "export PAP_MODEL_PATH" in text
    assert 'TOPOLOGY="${PAP_TOPOLOGY:-6pa2p}"' in text
    assert 'PA_COUNT="${PAP_PA_COUNT:-${BASH_REMATCH[1]}}"' in text
    assert 'PROJECTION_COUNT="${PAP_PROJECTION_COUNT:-${BASH_REMATCH[2]}}"' in text
    assert "TOTAL_GPU_COUNT=$(((PA_COUNT + PROJECTION_COUNT) * PAP_TP_SIZE))" in text
    assert "multi_pap_proxy_server.py" in text
    assert "pap_attention_executor.py" in text
    assert 'ATTENTION_TCP_PORT_BASE="${PAP_ATTENTION_TCP_PORT_BASE:-9300}"' in text
    assert 'ATTENTION_ZMQ_PORT_BASE="${PAP_ATTENTION_ZMQ_PORT_BASE:-10300}"' in text
    assert 'PROJECTION_ZMQ_PORT_BASE="${PAP_PROJECTION_ZMQ_PORT_BASE:-11300}"' in text
    assert '--tcp-port "$attention_tcp_port"' in text
    assert '--offload-exec-zmq-port "$attention_zmq_port"' in text
    assert "attention_zmq_port" in text
    assert (
        'PAP_OFFLOAD_EXEC_TRANSPORT="${PAP_OFFLOAD_EXEC_TRANSPORT:-nixl_mailbox}"'
        in text
    )
    assert (
        "PAP_OFFLOAD_EXEC_TRANSPORT=$PAP_OFFLOAD_EXEC_TRANSPORT is not supported"
        in text
    )
    assert 'PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT:-cuda_ipc}"' in text
    assert (
        'PAP_REMOTE_ATTENTION_PARALLELISM="${PAP_REMOTE_ATTENTION_PARALLELISM:-16}"'
        in text
    )
    assert 'PAP_OFFLOAD_EXEC_TRANSPORT="$PAP_OFFLOAD_EXEC_TRANSPORT"' in text
    assert 'PAP_OFFLOAD_KV_TRANSPORT="$PAP_OFFLOAD_KV_TRANSPORT"' in text
    assert "PAP_OFFLOAD_EXEC_ZMQ_PORT" in text
    assert "PAP_OFFLOAD_EXEC_HOST=127.0.0.1" in text
    assert "Projection vLLM metadata-only" in text
    assert 'kv_role":"kv_consumer"' not in text
    assert "projection_kv_transfer_config" not in text
    projection = text.split("for (( idx=0; idx<PROJECTION_COUNT; idx++ )); do", 1)[
        1
    ].split("done", 1)[0]
    assert "--kv-transfer-config" not in projection
    assert "PAP_MPS_PIPE_BASE_DIR" in text
    assert "build_pap_groups_spec" in text
    assert "build_projections_spec" in text
    assert "exec env CUDA_VISIBLE_DEVICES=0" in text


def test_pap_launch_supports_tp_sized_logical_instances() -> None:
    script = ROOT / "examples" / "pap" / "launch_pap_nixl.sh"
    text = script.read_text()

    assert 'PAP_TP_SIZE="${PAP_TP_SIZE:-1}"' in text
    assert "TOTAL_GPU_COUNT=$(((PA_COUNT + PROJECTION_COUNT) * PAP_TP_SIZE))" in text
    assert "PREFILL_GPU_COUNT=$((PA_COUNT * PAP_TP_SIZE))" in text
    assert "PROJECTION_GPU_COUNT=$((PROJECTION_COUNT * PAP_TP_SIZE))" in text
    assert '--tensor-parallel-size "$PAP_TP_SIZE"' in text
    assert (
        'PAP_DISABLE_CUSTOM_ALL_REDUCE="${PAP_DISABLE_CUSTOM_ALL_REDUCE:-auto}"' in text
    )
    assert 'vllm_tp_args+=("--disable-custom-all-reduce")' in text
    assert '"${vllm_tp_args[@]}"' in text
    assert "PAP_ENABLE_MPS=1 is not supported with PAP_TP_SIZE > 1" in text
    assert "build_rank_ports" in text


def test_pap_6pa2p_launch_supports_benchmark_service_mode() -> None:
    script = ROOT / "examples" / "pap" / "launch_pap_nixl.sh"
    text = script.read_text()

    assert "PAP_SERVICE_ONLY" in text
    assert "PAP_STATUS_FILE" in text
    assert "PAP_SKIP_SMOKE_REQUEST" in text
    assert 'echo "$PROXY_PORT" >"$STATUS_FILE"' in text
    assert 'if [[ "${SERVICE_ONLY}" == "1" ]]' in text


def test_pap_6pa2p_launch_uses_performance_transport_by_default() -> None:
    script = ROOT / "examples" / "pap" / "launch_pap_nixl.sh"
    text = script.read_text()

    assert (
        'PAP_OFFLOAD_EXEC_TRANSPORT="${PAP_OFFLOAD_EXEC_TRANSPORT:-nixl_mailbox}"'
        in text
    )
    assert '"kv_role":"kv_consumer"' not in text


def test_pap_6pa2p_launch_isolates_vllm_init_ports() -> None:
    script = ROOT / "examples" / "pap" / "launch_pap_nixl.sh"
    text = script.read_text()

    assert 'VLLM_PORT_BASE="${PAP_VLLM_PORT_BASE:-50000}"' in text
    assert 'VLLM_PORT="$((VLLM_PORT_BASE + idx * 20))"' in text
    assert 'VLLM_PORT="$((VLLM_PORT_BASE + PA_COUNT * 20 + idx * 20))"' in text


def test_pap_6pa2p_launch_forwards_routing_policy_to_proxy() -> None:
    script = ROOT / "examples" / "pap" / "launch_pap_nixl.sh"
    text = script.read_text()

    assert 'PAP_ROUTING_POLICY="${PAP_ROUTING_POLICY:-round_robin}"' in text
    assert '--routing-policy "$PAP_ROUTING_POLICY"' in text


def test_pap_launch_exports_unified_kv_control_endpoints() -> None:
    script = ROOT / "examples" / "pap" / "launch_pap_nixl.sh"
    text = script.read_text()

    assert 'PAP_UNIFIED_KV="${PAP_UNIFIED_KV:-1}"' in text
    assert "PAP_DECODE_COMMIT_ENDPOINT" in text
    assert "PAP_LEASE_RELEASE_ENDPOINT" in text
    assert 'PAP_DECODE_COMMIT_ENDPOINT="$decode_commit_endpoint"' in text
    assert 'PAP_LEASE_RELEASE_ENDPOINT="$lease_release_endpoint"' in text
    assert "PREFILL_PORT_BASE + idx" in text
    assert 'PAP_KV_LEASE_TTL_SECONDS="${PAP_KV_LEASE_TTL_SECONDS:-300}"' in text


def test_pap_launch_can_expose_prefill_cached_token_usage() -> None:
    script = ROOT / "examples" / "pap" / "launch_pap_nixl.sh"
    text = script.read_text()

    assert (
        'PAP_ENABLE_PROMPT_TOKENS_DETAILS="'
        '${PAP_ENABLE_PROMPT_TOKENS_DETAILS:-0}"'
    ) in text
    assert 'vllm_prefill_observability_args+=("--enable-prompt-tokens-details")' in text
    assert '"${vllm_prefill_observability_args[@]}"' in text


def test_pap_launch_assigns_unique_projection_mailbox_actor_ids() -> None:
    script = ROOT / "examples" / "pap" / "launch_pap_nixl.sh"
    text = script.read_text()

    assert 'PAP_NIXL_MAILBOX_ACTOR_ID="projection-${idx}"' in text


def test_pap_launch_does_not_expose_pap_runner_microbatch() -> None:
    script = ROOT / "examples" / "pap" / "launch_pap_nixl.sh"
    text = script.read_text()

    assert "PAP_RUNNER_MICROBATCH" not in text
    assert "projection_microbatch_args" not in text
    assert "--ubatch-size" not in text
    assert "--dbo-decode-token-threshold" not in text


def test_pap_baseline_config_exposes_pap_enabled() -> None:
    config = Path("/home/fei/research/PD/test/baseline/pap/config.sh")
    text = config.read_text()

    assert 'PAP_ENABLED="${PAP_ENABLED:-true}"' in text


def test_pap_128_testbed_enables_native_kv_append() -> None:
    script = ROOT / "benchmarks" / "disagg_benchmarks" / "run_pap_128_testbed.sh"
    text = script.read_text()

    assert (
        'PAP_ATTENTION_LOCAL_PAGED_NATIVE_CACHE_APPEND="${PAP_ATTENTION_LOCAL_PAGED_NATIVE_CACHE_APPEND:-1}"'
        in text
    )


def test_pap_128_testbed_does_not_enable_pap_runner_microbatch() -> None:
    script = ROOT / "benchmarks" / "disagg_benchmarks" / "run_pap_128_testbed.sh"
    text = script.read_text()

    assert "PAP_RUNNER_MICROBATCH" not in text
    assert 'PAP_NIXL_MAILBOX_SLOT_COUNT="${PAP_NIXL_MAILBOX_SLOT_COUNT:-3}"' in text
    assert (
        'PAP_NIXL_MAILBOX_RECV_SLOT_COUNT="${PAP_NIXL_MAILBOX_RECV_SLOT_COUNT:-3}"'
        in text
    )
