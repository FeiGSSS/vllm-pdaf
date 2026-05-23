from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_qwen3_true_split_checks_mode_before_local_attention() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()

    mode_check = text.index("if self._pap_true_split_enabled")
    local_attention = text.index("attn_output = self.attn(q, k, v)")
    assert mode_check < local_attention
    assert "_compute_pap_true_split_attention" in text


def test_model_runner_passes_pap_mode_to_forward_context() -> None:
    text = (ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py").read_text()

    assert "pap_mode" in text
    assert '"pap_mode", "debug_remote_attention"' in text
    assert "pap_mode_by_req_id" in text
    assert 'kv_transfer_params.get("pap_mode")' in text
    assert "self._pap_mode_for_batch(input_batch)" in text


def test_qwen3_true_split_method_updates_kv_and_calls_remote_attention() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()

    start = text.index("    def _compute_pap_true_split_attention")
    end = text.index("    def _maybe_compute_pap_remote_attention")
    method = text[start:end]

    assert "self.attn(" not in method
    assert "compute_stateful_remote_attention_output" in method
    assert "compute_remote_attention_output" not in method
    assert "reshape_and_cache_flash" not in method
    assert "NotImplementedError" not in method


def test_qwen3_true_split_gate_rejects_profile_and_warmup_forwards() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()

    start = text.index("    def _pap_true_split_enabled")
    end = text.index("    def _compute_pap_true_split_attention")
    method = text[start:end]

    assert 'pap_mode = additional_kwargs.get("pap_mode")' in method
    assert 'pap_mode not in {"true_split", "true_split_performance"}' in method
    assert "forward_context.attn_metadata" in method
    assert 'metadata.get(self.attn.layer_name)' in method
    assert 'getattr(attn_metadata, "max_query_len", 0)' in method
    assert "pap_num_scheduled_tokens" in method
    assert "_select_pap_request_id" in method
    assert "return not any" in method

def test_qwen3_true_split_sends_scheduler_descriptor_to_attention() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_true_split_attention")
    end = text.index("    def _maybe_compute_pap_remote_attention")
    method = text[start:end]

    assert "slot_mapping" in method
    assert "seq_lens" in method
    assert "pap_positions" in method
    assert "block_id = slot // block_size" in method
    assert "positions_cpu.reshape(-1)[req_index]" in method
    assert '"block_id": block_id' in method
    assert '"slot": slot' in method
    assert '"seq_len": seq_len' in method
    assert "ThreadPoolExecutor" in text
    assert "PAP_REMOTE_ATTENTION_PARALLELISM" in method

def test_model_runner_passes_pap_block_size_to_forward_context() -> None:
    text = (
        ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"
    ).read_text()

    assert '"pap_block_size"' in text
    assert '"pap_positions": input_batch.positions' in text
    assert "self.vllm_config.cache_config.block_size" in text


def test_qwen3_true_split_reads_block_size_from_forward_context() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_true_split_attention")
    end = text.index("    def _maybe_compute_pap_remote_attention")
    method = text[start:end]

    assert 'additional_kwargs.get("pap_block_size")' in method
    assert "PAP true_split missing cache block_size" in method

def test_qwen3_true_split_imports_prompt_kv_before_append() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_true_split_attention")
    end = text.index("    def _maybe_compute_pap_remote_attention")
    method = text[start:end]

    assert "import_prefill_kv_from_paged_cache" in method
    assert 'additional_kwargs.get("pap_prefill_prefix_len_by_request")' in method
    assert 'additional_kwargs.get("pap_attention_kv_installed_by_request")' in method
    assert "request_id in attention_kv_installed_by_request" in method
    assert "self._pap_imported_prefill_kv" in method
    assert "get_kv_cache_layout" in method
    assert "kv_cache" in method


def test_model_runner_passes_pap_prefill_prefix_len_to_forward_context() -> None:
    text = (
        ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"
    ).read_text()

    assert "pap_prefill_prefix_len_by_req_id" in text
    assert '"pap_prefill_prefix_len_by_request"' in text
    assert 'kv_transfer_params.get("remote_num_tokens")' in text


def test_qwen3_true_split_performance_rejects_prototype_transport() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_true_split_attention")
    end = text.index("    def _maybe_compute_pap_remote_attention")
    method = text[start:end]

    assert 'additional_kwargs.get("pap_mode") == "true_split_performance"' in method
    assert "performance_mode_requires_gpu_data_plane" in method
    assert "PAPTensorTransport.PROTOTYPE_HTTP" in method
    assert "offload_exec_transport_from_env()" in method
    assert "import_prefill_kv_from_paged_cache" in method


def test_qwen3_prefill_imports_kv_to_attention_before_projection_decode() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    end = text.index("    def _maybe_report_pap_attention_boundary")
    method = text[start:end]

    assert "for req_index in range(num_reqs)" in method
    assert "num_scheduled_tokens[req_index] <= 1" in method
    assert 'getattr(attn_metadata, "seq_lens", None)' in method
    assert 'additional_kwargs.get("pap_prefill_kv_handle_by_request")' in method
    assert '"pap_attention_tcp_endpoint_by_request"' in method
    assert "import_prefill_kv_from_paged_cache" in method
    assert "block_table[req_index : req_index + 1]" in method
    assert "tcp_endpoint=tcp_endpoint" in method
    assert "self._pap_imported_prefill_kv.add(import_key)" in method


def test_qwen3_true_split_skips_shadow_boundary_reports() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _maybe_report_pap_attention_boundary")
    end = text.index("class Qwen3DecoderLayer")
    method = text[start:end]

    assert '"true_split"' in method
    assert '"true_split_performance"' in method
    assert "maybe_report_qkv_boundary" in method


def test_model_runner_passes_pap_prefill_kv_handle_to_forward_context() -> None:
    text = (
        ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"
    ).read_text()

    assert "pap_prefill_kv_handle_by_req_id" in text
    assert '"pap_prefill_kv_handle_by_request"' in text
    assert 'kv_transfer_params.get("pap_prefill_kv_handle")' in text
    assert "pap_attention_kv_installed_by_req_id" in text
    assert '"pap_attention_kv_installed_by_request"' in text
    assert 'kv_transfer_params.get("pap_attention_kv_installed")' in text


def test_model_runner_passes_pap_offload_exec_endpoint_to_forward_context() -> None:
    text = (
        ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"
    ).read_text()

    assert "pap_offload_exec_zmq_endpoint_by_req_id" in text
    assert '"pap_offload_exec_zmq_endpoint_by_request"' in text
    assert 'kv_transfer_params.get("pap_offload_exec_zmq_endpoint")' in text

    qwen3 = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    assert "pap_offload_exec_zmq_endpoint_by_request" in qwen3
    assert "PAP OFFLOAD_EXEC ZMQ endpoint selected" in qwen3
    assert "_pap_offload_exec_transport" in qwen3
    assert "PAPOffloadExecDescriptor" in qwen3
    assert "transport.send_qkv" in qwen3
    assert "transport.recv_output" in qwen3
    assert "offload_exec_calls" in qwen3
    assert "executor.submit(" in qwen3
    assert "trigger_offload_exec_call" in qwen3


def test_nixl_pap_performance_mode_skips_projection_kv_recv() -> None:
    text = (
        ROOT
        / "vllm"
        / "distributed"
        / "kv_transfer"
        / "kv_connector"
        / "v1"
        / "nixl"
        / "scheduler.py"
    ).read_text()

    assert "self.pap_true_split_performance" in text
    assert '"pap_mode", ""' in text
    assert '== "true_split_performance"' in text
    assert "reqs_to_finish_recv" in text
    assert "meta.reqs_to_finish_recv.update" in text
