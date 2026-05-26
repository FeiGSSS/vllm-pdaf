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


def test_pap_projection_process_skips_startup_kv_tensor_allocation() -> None:
    model_runner = (
        ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"
    ).read_text()
    attn_utils = (
        ROOT / "vllm" / "v1" / "worker" / "gpu" / "attn_utils.py"
    ).read_text()
    launcher = (ROOT / "examples" / "pap" / "launch_pap_nixl.sh").read_text()

    assert "PAP_PROJECTION_KV_UNAWARE=1" in launcher
    assert "_pap_projection_kv_unaware_process" in model_runner
    assert "init_kv_cache_metadata_only" in model_runner
    assert "init_kv_cache(" in model_runner
    assert "PAP Projection KV-unaware process skips local-attention warmup" in (
        (ROOT / "vllm" / "v1" / "worker" / "gpu_worker.py").read_text()
    )
    assert "def init_kv_cache_metadata_only" in attn_utils
    assert "_allocate_kv_cache(" not in attn_utils[
        attn_utils.index("def init_kv_cache_metadata_only") :
    ].split("def build_slot_mappings_by_layer", 1)[0]


def test_scheduler_sends_only_local_pap_projection_blocks_to_model_runner() -> None:
    text = (ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py").read_text()

    start = text.index("                request = request_queue.pop_request()")
    end = text.index(
        "                num_scheduled_tokens[request_id] = num_new_tokens"
    )
    block = text[start:end]

    assert "pap_projection_state is not None" in block
    assert "req_to_new_blocks[request_id] = new_blocks" in block
    assert (
        "req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(" in block
    )


def test_scheduler_disables_external_block_allocation_for_pap_projection() -> None:
    text = (ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py").read_text()

    start = text.index(
        "                new_blocks = self.kv_cache_manager.allocate_slots("
    )
    end = text.index("                if new_blocks is None:")
    block = text[start:end]

    assert "pap_projection_state.allocate_external_computed_blocks" in block


def test_scheduler_offsets_running_pap_projection_local_progress() -> None:
    text = (ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py").read_text()

    start = text.index("        while req_index < len(self.running)")
    end = text.index("        # Record the LoRAs in scheduled_running_reqs")
    block = text[start:end]

    assert "_get_pap_projection_schedule_state(request)" in block
    assert "pap_projection_state.local_computed_token_offset" in block
    assert "local_computed_token_offset=pap_local_computed_token_offset" in block


def test_scheduler_disables_local_slot_allocation_for_pap_projection() -> None:
    text = (ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py").read_text()

    running_start = text.index("        while req_index < len(self.running)")
    running_end = text.index("        # Record the LoRAs in scheduled_running_reqs")
    running_block = text[running_start:running_end]

    waiting_start = text.index(
        "                new_blocks = self.kv_cache_manager.allocate_slots("
    )
    waiting_end = text.index("                if new_blocks is None:")
    waiting_block = text[waiting_start:waiting_end]

    assert "allocate_pap_local_slots" in running_block
    assert "pap_projection_state.allocate_local_slots" in waiting_block


def test_scheduler_uses_explicit_pap_projection_schedule_state() -> None:
    text = (ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py").read_text()

    assert "class PAPProjectionScheduleState" in text
    assert "_get_pap_projection_schedule_state" in text
    assert "remote_computed_tokens=remote_computed_tokens" in text
    assert "allocate_external_computed_blocks: bool = False" in text
    assert "allocate_local_slots: bool = False" in text


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
    assert "transport.send_qkv_batch" in method
    assert "transport.recv_output_batch" in method
    assert "recv_output_batch_message" in method
    assert "output_message.release()" in method
    assert "PAPOffloadExecBatchDescriptor" in method
    assert "PAPOffloadExecDescriptor" in method
    assert "trigger_offload_exec_attention_batch" in method
    assert "_pap_offload_exec_local_address()" in method
    assert "local_offload_exec_zmq_endpoint" in method


def test_nixl_mailbox_zero_copy_recv_is_default_enabled() -> None:
    text = (ROOT / "vllm" / "pap" / "nixl_mailbox.py").read_text()

    assert '"PAP_NIXL_MAILBOX_ZERO_COPY_RECV", True' in text


def test_qwen3_pap_forward_returns_before_local_attention() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()

    start = text.index("    def forward(\n")
    end = text.index("    def _should_use_pap_attention")
    method = text[start:end]

    pap_branch = method.index("if self._should_use_pap_attention():")
    pap_compute = method.index("attn_output, pap_release_messages = self._compute_pap_attention", pap_branch)
    pap_return = method.index("return output", pap_compute)
    local_attention = method.index("attn_output = self.attn(q, k, v)")
    prefill_import = method.index("self._maybe_import_pap_prefill_kv_to_attention()")

    assert pap_branch < pap_compute < pap_return < local_attention
    assert local_attention < prefill_import


def test_qwen3_pap_q_first_projection_compute_is_opt_in() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    forward_start = text.index("    def forward(\n")
    forward_end = text.index("    def _should_use_pap_attention")
    forward_method = text[forward_start:forward_end]

    assert "PAP_Q_FIRST_PROJECTION" in text
    assert "_compute_pap_attention_q_first_projection" in text
    assert "_pap_qkv_projection_split_supported" in text
    assert "_pap_q_first_projection_transport_supported" in text
    assert forward_method.index("if self._should_use_pap_attention():") < forward_method.index(
        "_compute_pap_attention_q_first_projection"
    )
    assert forward_method.index("_compute_pap_attention_q_first_projection") < forward_method.index(
        "qkv, _ = self.qkv_proj(hidden_states)"
    )


def test_qwen3_pap_q_first_projection_sends_query_before_kv_projection() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention_q_first_projection")
    end = text.index("    def _send_pap_query_batch", start)
    method = text[start:end]

    assert "_pap_qkv_projection_slice" in method
    assert "_send_pap_query_batch" in method
    assert "query_already_sent=True" in method
    assert method.index("_pap_q_first_projection_transport_supported") < method.index(
        "_pap_qkv_projection_slice"
    )
    assert method.index("_send_pap_query_batch") < method.index("kv = _pap_qkv_projection_slice")
    assert method.index("kv = _pap_qkv_projection_slice") < method.index(
        "query_already_sent=True"
    )


def test_qwen3_pap_q_first_projection_validates_all_groups_before_query_send() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _send_pap_query_batch")
    end = text.index("    def _compute_pap_attention", start)
    method = text[start:end]

    validation = "if any(\n            not attention_endpoint"
    assert validation in method
    assert method.index(validation) < method.index("send_query_batch(")


def test_qwen3_pap_q_first_kv_later_is_opt_in() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert "PAP_Q_FIRST_KV_LATER" in method
    assert "q_first_kv_later_enabled" in method
    assert "send_query_batch = getattr(" in method
    assert "send_kv_batch = getattr(" in method
    assert "send_query_batch(" in method
    assert "send_kv_batch(" in method
    assert "supports_query_first_kv_later" in method
    q_first_gate = method[
        method.index("q_first_kv_later_enabled") : method.index("elif segmented_qkv_enabled")
    ]
    assert "q_first_kv_later_enabled" in q_first_gate
    assert "send_query_batch(" in q_first_gate
    assert "send_kv_batch(" in q_first_gate


def test_qwen3_pap_mailbox_segmented_qkv_is_opt_in() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert "PAP_SEGMENTED_QKV" in method
    assert "segmented_qkv_enabled" in method
    assert "send_qkv_batch_segments = getattr(" in method
    assert "send_qkv_batch_segments(" in method
    assert "payload_shape=(len(group_items), qkv_width)" in method
    assert method.index("send_qkv_batch_segments = getattr(") < method.index(
        "if segmented_qkv_enabled and callable(send_qkv_batch_segments):"
    )
    segment_gate = method[method.index("if segmented_qkv_enabled and callable(send_qkv_batch_segments):") : method.index("transport.send_qkv_batch(")]
    assert "segmented_qkv_enabled" in segment_gate


def test_qwen3_pap_single_group_avoids_qkv_batch_cat() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert "if len(group_items) == 1:" in method
    assert "qkv_segments = group_items[0][2]" in method
    assert "qkv_segments = tuple(" in method
    assert "transport.send_qkv_batch(" in method
    assert "qkv_batch," in method


def test_qwen3_pap_direct_mailbox_output_is_opt_in() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert "PAP_DIRECT_MAILBOX_OUTPUT" in method
    assert "direct_mailbox_output_enabled" in method
    assert "can_use_direct_output = (" in method
    assert "direct_mailbox_output_enabled" in method[
        method.index("can_use_direct_output = (") : method.index("if can_use_direct_output:")
    ]


def test_qwen3_pap_defers_direct_mailbox_output_release_until_after_o_proj() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()

    forward_start = text.index("    def forward(\n")
    forward_end = text.index("    def _should_use_pap_attention")
    forward_method = text[forward_start:forward_end]

    assert "attn_output, pap_release_messages = self._compute_pap_attention" in forward_method
    assert "try:" in forward_method
    assert "output, _ = self.o_proj(attn_output)" in forward_method
    assert "finally:" in forward_method
    assert "for message in pap_release_messages:" in forward_method
    assert "message.release()" in forward_method
    assert forward_method.index("output, _ = self.o_proj(attn_output)") < forward_method.index(
        "message.release()"
    )

    compute_start = text.index("    def _compute_pap_attention")
    compute_end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    compute_method = text[compute_start:compute_end]

    assert "pap_release_messages: list[Any] = []" in compute_method
    assert "direct_output = output_batch.view" in compute_method
    assert "pap_release_messages.append(output_message)" in compute_method
    assert "return direct_output, pap_release_messages" in compute_method


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


def test_qwen3_pap_projection_trace_uses_batch_descriptor() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert "offload_exec_batches[0][3].layer_name" in method
    assert "sum(len(batch[3].items) for batch in offload_exec_batches)" in method
    assert "offload_exec_batches[0][2].layer_name" not in method
    assert "sum(len(batch[2].items) for batch in offload_exec_batches)" not in method


def test_qwen3_pap_attention_does_not_require_projection_block_size() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert 'additional_kwargs.get("pap_block_size")' not in method
    assert "PAP attention missing cache block_size" not in method


def test_qwen3_pap_uses_empty_output_when_all_requests_offloaded() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert "all_requests_offloaded = len(remote_attention_calls) == num_reqs" in method
    assert "torch.empty_like(query)" in method
    assert "torch.zeros_like(query)" in method
    assert method.index("remote_attention_calls: list") < method.index(
        "all_requests_offloaded = len(remote_attention_calls) == num_reqs"
    )


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


def test_qwen3_prefill_uses_paged_kv_import_for_cuda_ipc() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    end = text.index("    @staticmethod")
    method = text[start:end]

    assert "import_prefill_paged_kv" in method
    assert "import_prefill_kv_from_paged_cache" not in method
    assert "block_ids=_pap_block_ids_from_block_table(" in method
    assert "kv_cache=kv_cache" in method
