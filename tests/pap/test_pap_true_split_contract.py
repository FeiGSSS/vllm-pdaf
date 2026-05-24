from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_qwen3_pap_enabled_checks_pap_enabled_flag() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()

    start = text.index("    def _should_use_pap_attention")
    end = text.index("    def _pap_attention_kv_ready_for_requests")
    method = text[start:end]

    assert 'additional_kwargs.get("pap_enabled")' in method
    assert "_select_pap_request_id" in method
    assert "attn_metadata" in method


def test_model_runner_passes_pap_enabled_to_forward_context() -> None:
    text = (ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py").read_text()

    assert '"pap_enabled"' in text
    assert "_pap_enabled_for_batch" in text
    assert "PAP enabled via per-request TCP endpoint" in text


def test_scheduler_sends_only_local_pap_projection_blocks_to_model_runner() -> None:
    text = (ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py").read_text()

    start = text.index("                request = request_queue.pop_request()")
    end = text.index("                num_scheduled_tokens[request_id] = num_new_tokens")
    block = text[start:end]

    assert "pap_remote_prefix_len is not None" in block
    assert "req_to_new_blocks[request_id] = new_blocks" in block
    assert "req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(" in block


def test_scheduler_disables_external_block_allocation_for_pap_projection() -> None:
    text = (ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py").read_text()

    start = text.index("                new_blocks = self.kv_cache_manager.allocate_slots(")
    end = text.index("                if new_blocks is None:")
    block = text[start:end]

    assert "allocate_external_computed_blocks=pap_remote_prefix_len is None" in block


def test_scheduler_offsets_running_pap_projection_local_progress() -> None:
    text = (ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py").read_text()

    start = text.index("        while req_index < len(self.running)")
    end = text.index("        # Record the LoRAs in scheduled_running_reqs")
    block = text[start:end]

    assert "_get_pap_projection_local_computed_token_offset(request)" in block
    assert "local_computed_token_offset=pap_local_computed_token_offset" in block


def test_engine_core_allows_pap_metadata_without_kv_connector() -> None:
    text = (ROOT / "vllm" / "v1" / "engine" / "core.py").read_text()

    assert "_is_pap_metadata_only_request" in text
    assert "pap_projection_kv_unaware" in text
    assert "Got kv_transfer_params, but no KVConnector found" in text


def test_qwen3_pap_path_uses_nccl_offload_exec() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()

    start = text.index("    def _compute_pap_attention")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert "self.attn(" not in method
    assert "transport.send_qkv" in method
    assert "transport.recv_output" in method
    assert "PAPOffloadExecDescriptor" in method
    assert "trigger_offload_exec_attention" in method


def test_qwen3_pap_gate_checks_decode_only() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()

    start = text.index("    def _should_use_pap_attention")
    end = text.index("    def _pap_attention_kv_ready_for_requests")
    method = text[start:end]

    assert 'additional_kwargs.get("pap_enabled")' in method
    assert "max_query_len" in method
    assert "num_scheduled_tokens" in method
    assert "pap_num_scheduled_tokens" in method


def test_qwen3_pap_sends_position_only_offload_exec_descriptor() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert "seq_lens" in method
    assert "pap_positions" in method
    assert "positions_cpu.reshape(-1)[req_index]" in method
    assert '"seq_len": seq_len' in method
    assert "slot_mapping" not in method
    assert "block_id = slot // block_size" not in method
    assert '"block_id": block_id' not in method
    assert '"slot": slot' not in method
    assert "ThreadPoolExecutor" in text
    assert "PAP_REMOTE_ATTENTION_PARALLELISM" in method


def test_model_runner_passes_pap_block_size_to_forward_context() -> None:
    text = (
        ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"
    ).read_text()

    assert '"pap_block_size"' in text
    assert '"pap_positions": input_batch.positions' in text
    assert "self.vllm_config.cache_config.block_size" in text


def test_qwen3_pap_attention_does_not_require_projection_block_size() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert 'additional_kwargs.get("pap_block_size")' not in method
    assert "PAP attention missing cache block_size" not in method


def test_qwen3_pap_imports_prefill_kv_for_offload() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    end = text.index("    @staticmethod")
    method = text[start:end]

    assert 'additional_kwargs.get("pap_attention_kv_installed_by_request")' in method
    assert 'additional_kwargs.get("pap_prefill_kv_handle_by_request")' in method
    assert "attention_kv_installed_by_request" in method
    assert "PAP_OFFLOAD_KV_TRANSPORT" in method
    assert "PAPTensorTransport.CUDA_IPC" in method


def test_model_runner_passes_pap_prefill_prefix_len_to_forward_context() -> None:
    text = (
        ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"
    ).read_text()

    assert "pap_prefill_prefix_len_by_req_id" in text
    assert '"pap_prefill_prefix_len_by_request"' in text
    assert 'kv_transfer_params.get("pap_remote_prefix_len")' in text
    assert 'kv_transfer_params.get("remote_num_tokens")' in text


def test_qwen3_pap_uses_tcp_control_endpoint() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert "pap_attention_tcp_endpoint_by_request" in method
    assert "pap_attention_tcp_endpoint" in method
    assert "tcp_endpoint_by_request" in method
    assert "select_attention_endpoint_for_request" in method
    assert "trigger_offload_exec_attention" in method


def test_qwen3_prefill_detects_pap_prefill_kv_handle() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    end = text.index("    @staticmethod")
    method = text[start:end]

    assert "pap_enabled" in method
    assert "pap_prefill_kv_handle_by_request" in method
    assert "prefill_kv_handle" in method
    assert "pap_import_prefill_kv_to_attention_by_request" in method
    assert "transport=offload_kv_transport" in method


def test_qwen3_prefill_import_marks_attention_kv_ready() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    end = text.index("    @staticmethod")
    method = text[start:end]

    assert "self._pap_imported_prefill_kv.add(import_key)" in method
    assert 'additional_kwargs.get("pap_attention_kv_installed_by_request")' in method
    assert "if import_key in self._pap_imported_prefill_kv:" in method
    assert "installed.add(request_id)" in method
    assert (
        'additional_kwargs["pap_attention_kv_installed_by_request"] = installed'
        in method
    )


def test_qwen3_pap_skips_requests_without_kv_handle() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert "prefill_kv_handle" in method
    assert "attention_kv_installed_by_request" in method


def test_model_runner_passes_pap_prefill_kv_handle_to_forward_context() -> None:
    text = (
        ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"
    ).read_text()

    assert "pap_prefill_kv_handle_by_req_id" in text
    assert '"pap_prefill_kv_handle_by_request"' in text
    assert 'kv_transfer_params.get("pap_prefill_kv_handle")' in text
    assert "pap_import_prefill_kv_to_attention_by_req_id" in text
    assert '"pap_import_prefill_kv_to_attention_by_request"' in text
    assert 'kv_transfer_params.get("pap_import_prefill_kv_to_attention")' in text
    assert "pap_attention_kv_installed_by_req_id" in text
    assert '"pap_attention_kv_installed_by_request"' in text
    assert 'kv_transfer_params.get("pap_attention_kv_installed")' in text


def test_qwen3_pap_attention_gate_requires_attention_kv_ready() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _should_use_pap_attention")
    end = text.index("    def _pap_attention_kv_ready_for_requests")
    gate = text[start:end]
    ready_start = text.index("    def _pap_attention_kv_ready_for_requests")
    ready_end = text.index("    def _compute_pap_attention")
    ready = text[ready_start:ready_end]

    assert "request_ids[:num_reqs]" in gate
    assert "self._pap_attention_kv_ready_for_requests" in gate
    assert 'additional_kwargs.get("pap_attention_kv_installed_by_request")' in ready
    assert "all(str(request_id) in installed for request_id in request_ids)" in ready
    assert "pap_prefill_kv_handle_by_request" not in ready


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


def test_nixl_pap_skips_projection_kv_recv_when_kv_installed() -> None:
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

    assert "pap_attention_kv_installed" in text
    assert "_reqs_to_finish_recv" in text
