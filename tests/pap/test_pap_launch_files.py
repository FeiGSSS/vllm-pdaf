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
