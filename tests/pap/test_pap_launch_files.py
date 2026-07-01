from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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
