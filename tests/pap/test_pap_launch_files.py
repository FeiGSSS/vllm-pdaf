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
