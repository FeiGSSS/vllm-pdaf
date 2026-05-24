from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pap_launch_uses_qwen3_8b_and_nixl_roles() -> None:
    script = ROOT / "examples" / "pap" / "launch_pap_qwen3_8b_nixl.sh"
    text = script.read_text()

    assert "/data/ssd1/llm-models/Qwen3-8B" in text
    assert '"kv_connector":"NixlConnector"' in text
    assert '"kv_role":"kv_producer"' in text
    assert '"kv_role":"kv_consumer"' in text
    assert "pap_attention_executor.py" in text
    assert "pap_remote_attention" in text
    assert "import nixl" in text
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" in text


def test_pap_readme_documents_projection_as_decode_owner() -> None:
    readme = ROOT / "examples" / "pap" / "README.md"
    text = readme.read_text()

    assert "Projection" in text
    assert "lm_head" in text
    assert "sampling" in text
    assert "Attention" in text
    assert "internal executor" in text


def test_pap_design_doc_marks_prefill_to_attention_nixl_as_missing() -> None:
    doc = ROOT / "docs" / "design" / "pap_prefill_attention_nixl.md"
    text = doc.read_text()

    assert "Prefill to Attention: same-GPU CUDA IPC" in text
    assert "/v1/pap/attention/import-prefill-kv" in text
    assert "Projection should not receive Prefill KV" in text
    assert "Projection-to-Attention data plane must not be TCP/HTTP" in text
    assert "OFFLOAD_KV" in text
    assert "OFFLOAD_EXEC" in text
    assert "PAP-prototype-http-tcp-data-plane" in text


def test_pap_launch_mps_wrapper_preserves_env_assignments() -> None:
    script = ROOT / "examples" / "pap" / "launch_pap_qwen3_8b_nixl.sh"
    text = script.read_text()

    assert "PAP_ENABLE_MPS" in text
    assert "nvidia-cuda-mps-control -d" in text
    assert "with_prefill_attention_mps_env" in text
    assert 'env "$@"' in text


def test_pap_consistency_runner_compares_three_architectures() -> None:
    script = ROOT / "examples" / "pap" / "run_arch_consistency_qwen3_8b_nixl.sh"
    text = script.read_text()

    assert "launch_fused_qwen3_8b.sh" in text
    assert "launch_native_pd_qwen3_8b_nixl.sh" in text
    assert "launch_pap_qwen3_8b_nixl.sh" in text
    assert "compare_outputs.py" in text
    assert "PAP_MPS_PIPE_DIR" in text


def test_pap_baseline_launchers_use_deterministic_generation_config() -> None:
    for name in ["launch_fused_qwen3_8b.sh", "launch_native_pd_qwen3_8b_nixl.sh"]:
        text = (ROOT / "examples" / "pap" / name).read_text()
        assert "--generation-config vllm" in text
        assert '--output "$RESULT_PATH"' in text


def test_pap_6pa2p_launch_uses_multi_proxy_and_expected_counts() -> None:
    script = ROOT / "examples" / "pap" / "launch_pap_nixl.sh"
    text = script.read_text()

    assert 'MODEL_PATH="${PAP_MODEL_PATH:-/data/ssd1/llm-models/Qwen3-0.6B}"' in text
    assert "-m | --model" in text
    assert "export PAP_MODEL_PATH" in text
    assert 'TOPOLOGY="${PAP_TOPOLOGY:-6pa2p}"' in text
    assert 'PA_COUNT="${PAP_PA_COUNT:-${BASH_REMATCH[1]}}"' in text
    assert 'PROJECTION_COUNT="${PAP_PROJECTION_COUNT:-${BASH_REMATCH[2]}}"' in text
    assert "TOTAL_GPU_COUNT=$((PA_COUNT + PROJECTION_COUNT))" in text
    assert "multi_pap_proxy_server.py" in text
    assert "pap_attention_executor.py" in text
    assert 'ATTENTION_TCP_PORT_BASE="${PAP_ATTENTION_TCP_PORT_BASE:-9300}"' in text
    assert 'ATTENTION_ZMQ_PORT_BASE="${PAP_ATTENTION_ZMQ_PORT_BASE:-10300}"' in text
    assert 'PROJECTION_ZMQ_PORT_BASE="${PAP_PROJECTION_ZMQ_PORT_BASE:-11300}"' in text
    assert '--tcp-port "$attention_tcp_port"' in text
    assert '--offload-exec-zmq-port "$attention_zmq_port"' in text
    assert "attention_zmq_port" in text
    assert 'PAP_OFFLOAD_EXEC_TRANSPORT="${PAP_OFFLOAD_EXEC_TRANSPORT:-nccl}"' in text
    assert 'PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT:-cuda_ipc}"' in text
    assert 'PAP_REMOTE_ATTENTION_PARALLELISM="${PAP_REMOTE_ATTENTION_PARALLELISM:-16}"' in text
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

    assert 'PAP_OFFLOAD_EXEC_TRANSPORT="${PAP_OFFLOAD_EXEC_TRANSPORT:-nccl}"' in text
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


def test_pap_baseline_config_exposes_pap_enabled() -> None:
    config = Path("/home/fei/research/PD/test/baseline/pap/config.sh")
    text = config.read_text()

    assert 'PAP_ENABLED="${PAP_ENABLED:-true}"' in text
