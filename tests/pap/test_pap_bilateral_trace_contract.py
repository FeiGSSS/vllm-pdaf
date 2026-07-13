from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QWEN3 = ROOT / "vllm/model_executor/models/qwen3.py"
LOCAL_FAST = ROOT / "vllm/pap/local_fast_transport.py"
MODEL_RUNNER = ROOT / "vllm/v1/worker/gpu/model_runner.py"
FLASH_ATTN = ROOT / "vllm/v1/attention/backends/flash_attn.py"


def test_projection_trace_contains_all_critical_chain_spans() -> None:
    qwen = QWEN3.read_text(encoding="utf-8")
    local_fast = LOCAL_FAST.read_text(encoding="utf-8")
    model_runner = MODEL_RUNNER.read_text(encoding="utf-8")

    assert '"qkv_norm_rope_gpu_ms"' in qwen
    assert '"projection_qk_repack_gpu_ms"' in qwen
    assert '"qkv_p2p_copy_gpu_ms"' in local_fast
    assert '"output_doorbell_wait_wall_ms"' in local_fast
    assert '"output_ready_wait_gpu_ms"' in local_fast
    assert '"token_boundary_input_ids_d2h_wall_ms"' in model_runner


def test_pd_trace_uses_same_qkv_span_and_main_paged_fa_boundary() -> None:
    qwen = QWEN3.read_text(encoding="utf-8")
    flash = FLASH_ATTN.read_text(encoding="utf-8")

    assert '"qkv_norm_rope_gpu_ms"' in qwen
    assert '"pd_paged_fa_gpu_ms"' in flash
    assert "deferred_trace_role" in qwen
    assert "max_query_len" in flash
