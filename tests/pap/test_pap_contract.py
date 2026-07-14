from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_qwen3_pap_enabled_checks_pap_enabled_flag() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()

    start = text.index("    def _should_use_pap_attention")
    end = text.index("    def _pap_attention_kv_ready_for_requests")
    method = text[start:end]

    assert 'additional_kwargs.get("pap_enabled")' in method
    assert "pap_request_ids_are_routable(request_ids, num_reqs)" in method
    assert "attn_metadata" in method


def test_qwen3_moe_attention_rejects_pap_attention_path() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3_moe.py").read_text()

    class_start = text.index("class Qwen3MoeAttention")
    class_end = text.index("\n\nclass Qwen3MoeDecoderLayer", class_start)
    cls = text[class_start:class_end]

    assert "Qwen3Attention" in text
    assert "class Qwen3MoeAttention(Qwen3Attention):" in cls
    assert "super().__init__(" in cls
    assert "max_position=max_position_embeddings" in cls
    assert "def _should_use_pap_attention(" in cls
    assert "PAP is disabled for qwen3_moe" in cls
    assert "raise RuntimeError" in cls


def test_model_runner_passes_pap_enabled_to_forward_context() -> None:
    text = (ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py").read_text()

    assert '"pap_enabled"' in text
    assert "_pap_enabled_for_batch" in text
    assert "PAP enabled via per-request mailbox endpoint" in text
    assert "_pap_offload_exec_route_groups_for_batch" in text
    assert '"pap_offload_exec_route_groups"' in text
    assert '"pap_finished_request_ids"' in text
    assert "scheduler_output.finished_req_ids" in text


def test_gpu_model_runner_passes_pap_request_context() -> None:
    text = (ROOT / "vllm" / "v1" / "worker" / "gpu_model_runner.py").read_text()

    assert "_add_pap_attention_endpoint" in text
    assert "new_req_data.kv_transfer_params" in text
    assert "_pap_forward_context_kwargs" in text
    assert "additional_kwargs=pap_additional_kwargs" in text
    assert "_pap_offload_exec_route_groups_for_request_ids" in text
    assert '"pap_offload_exec_route_groups"' in text
    assert '"pap_attention_tcp_endpoint_by_request"' in text
    assert '"pap_finished_request_ids"' in text
    assert "scheduler_output.finished_req_ids" in text
    assert "PAPProjectionPeerActivity" in text
    assert "sync_pap_projection_peer_activity" in text
    state_refresh = text.index("self.input_batch.refresh_metadata()")
    activity_sync = text.index(
        "sync_pap_projection_peer_activity(",
        state_refresh,
    )
    assert activity_sync > state_refresh


def test_v2_gpu_model_runner_syncs_membership_before_empty_return() -> None:
    text = (ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py").read_text()

    state_updated = text.index("self.block_tables.apply_staged_writes()")
    activity_sync = text.index(
        "sync_pap_projection_peer_activity(",
        state_updated,
    )
    empty_return = text.index(
        "if scheduler_output.total_num_scheduled_tokens == 0:",
        state_updated,
    )
    assert state_updated < activity_sync < empty_return


def test_v2_model_runner_has_no_pap_runner_microbatch_contexts() -> None:
    text = (ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py").read_text()

    assert "PAP_RUNNER_MICROBATCH" not in text
    assert "pap_runner_microbatch" not in text
    assert "_pap_forward_context_kwargs_for_ubatch" not in text
    assert '"ubatch_additional_kwargs"' not in text
    assert "num_ubatches=pap_runner_microbatch_count" not in text
    assert "_pap_forward_context_kwargs" in text


def test_new_gpu_model_runner_has_no_pap_runner_microbatch_contexts() -> None:
    text = (ROOT / "vllm" / "v1" / "worker" / "gpu_model_runner.py").read_text()

    assert "PAP_RUNNER_MICROBATCH" not in text
    assert "pap_runner_microbatch" not in text
    assert "_pap_forward_context_kwargs_for_ubatch" not in text
    assert '"ubatch_additional_kwargs"' not in text
    assert "num_ubatches=pap_runner_microbatch_count" not in text


def test_ubatch_wrapper_has_no_pap_specific_extensions() -> None:
    text = (ROOT / "vllm" / "v1" / "worker" / "gpu_ubatch_wrapper.py").read_text()

    assert '"ubatch_additional_kwargs"' not in text
    assert "additional_kwargs=ubatch_additional_kwargs" not in text
    assert "num_ubatches: int | None = None" not in text
    assert "self.num_ubatches" not in text
    assert "ubatch_dp_metadata = [None]" not in text
    assert "assert dp_metadata is not None" in text


def test_pap_projection_process_skips_startup_kv_tensor_allocation() -> None:
    model_runner = (
        ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"
    ).read_text()
    attn_utils = (ROOT / "vllm" / "v1" / "worker" / "gpu" / "attn_utils.py").read_text()
    launcher = (ROOT / "examples" / "pap" / "launch_pap_nixl.sh").read_text()

    assert "PAP_PROJECTION_KV_UNAWARE=1" in launcher
    assert "_pap_projection_kv_unaware_process" in model_runner
    assert "init_kv_cache_metadata_only" in model_runner
    assert "init_kv_cache(" in model_runner
    assert "PAP Projection KV-unaware process skips local-attention warmup" in (
        (ROOT / "vllm" / "v1" / "worker" / "gpu_worker.py").read_text()
    )
    assert "def init_kv_cache_metadata_only" in attn_utils
    assert (
        "_allocate_kv_cache("
        not in attn_utils[attn_utils.index("def init_kv_cache_metadata_only") :].split(
            "def build_slot_mappings_by_layer", 1
        )[0]
    )


def test_scheduler_sends_only_local_pap_projection_blocks_to_model_runner() -> None:
    text = (ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py").read_text()

    start = text.index("                request = request_queue.pop_request()")
    end = text.index(
        "                num_scheduled_tokens[request_id] = num_new_tokens"
    )
    block = text[start:end]

    assert "pap_projection_state is not None" in block
    assert "req_to_new_blocks[request_id] = new_blocks" in block
    assert "req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(" in block


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


def test_nixl_pull_scheduler_has_no_pap_projection_bypass() -> None:
    text = (
        ROOT
        / "vllm"
        / "distributed"
        / "kv_transfer"
        / "kv_connector"
        / "v1"
        / "nixl"
        / "pull_scheduler.py"
    ).read_text()

    assert "pap_attention_kv_installed" not in text


def test_qwen3_pap_path_uses_nixl_mailbox_offload_exec_without_tcp_trigger() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()

    start = text.index("    def _compute_pap_attention(\n")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert "self.attn(" not in method
    assert "_pap_offload_exec_step_groups(" in method
    assert "select_attention_endpoint_for_request" not in method
    assert "trigger_offload_exec_attention_batch" not in method
    assert "requires_tcp_trigger" not in method
    assert "transport.send_qkv_batch" in method
    assert "transport.recv_output_batch" in method
    assert "recv_output_batch_message" in method
    assert "output_message.release()" in method
    assert "PAPOffloadExecBatchDescriptor" in method
    assert "PAPOffloadExecDescriptor" not in method


def test_qwen3_pap_mailbox_transport_is_per_attention_endpoint() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention(\n")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert "def _pap_offload_exec_transport_for_attention_endpoint" in text
    assert "def _pap_nixl_mailbox_offload_exec_transport" in text
    assert "_pap_offload_exec_transport_for_attention_endpoint" in method
    assert "_pap_bind_offload_exec_mailbox_peer" in method


def test_nixl_mailbox_zero_copy_recv_is_default_enabled() -> None:
    text = (ROOT / "vllm" / "pap" / "nixl_mailbox.py").read_text()

    assert '"PAP_NIXL_MAILBOX_ZERO_COPY_RECV", True' in text


def test_qwen3_dense_pap_uses_runner_dbo_path_only() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    forward_start = text.index("    def forward(\n")
    forward_end = text.index("    def _should_use_pap_attention")
    forward_method = text[forward_start:forward_end]

    assert "PAP_OFFLOAD_EXEC_MICROBATCH" not in text
    assert "_pap_attention_microbatch_pipeline" not in text
    assert "_run_pap_attention_microbatch_pipeline" not in text
    assert "_pap_offload_exec_microbatch_count" not in text
    pap_compute = forward_method.index(
        "attn_output, pap_release_messages = self._compute_pap_attention"
    )
    assert pap_compute < forward_method.index("return output", pap_compute)


def test_qwen3_pap_attention_does_not_yield_through_dbo() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention", start)
    method = text[start:end]

    assert "dbo_enabled" not in method
    assert "dbo_yield" not in method


def test_qwen3_pap_microbatch_recv_supports_direct_mailbox_output() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _recv_pap_attention_batch")
    end = text.index("    def _compute_pap_attention", start)
    method = text[start:end]

    assert "PAP_DIRECT_MAILBOX_OUTPUT" in method
    assert "can_use_direct_output = (" in method
    assert "direct_output = output_batch.view_as(query)" in method
    assert "pap_release_messages.append(output_message)" in method
    assert "return direct_output, pap_release_messages" in method


def test_qwen3_decoder_layer_has_no_attention_boundary_microbatch_overlap() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    forward_start = text.index(
        "    def forward(", text.index("class Qwen3DecoderLayer")
    )
    forward_end = text.index("\n\nALL_DECODER_LAYER_TYPES", forward_start)
    forward_method = text[forward_start:forward_end]

    assert "_pap_microbatch_forward_after_input_norm" not in text
    assert "PAP_OFFLOAD_EXEC_MICROBATCH_OVERLAP_MLP" not in text
    assert "hidden_states = self.self_attn(" in forward_method


def test_qwen3_moe_has_no_attention_boundary_microbatch_overlap() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3_moe.py").read_text()
    forward_start = text.index(
        "    def forward(", text.index("class Qwen3MoeDecoderLayer")
    )
    forward_end = text.index("\n\n@support_torch_compile", forward_start)
    forward_method = text[forward_start:forward_end]

    assert "PAP_OFFLOAD_EXEC_MICROBATCH" not in text
    assert "_run_pap_attention_microbatch_pipeline" not in text
    assert "_pap_microbatch_forward_after_input_norm" not in text
    assert "hidden_states = self.self_attn(" in forward_method


def test_qwen3_moe_model_has_no_pap_layer_wavefront() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3_moe.py").read_text()
    model_start = text.index("class Qwen3MoeModel")
    model_end = text.index("\n\nclass Qwen3MoeForCausalLM", model_start)
    model_cls = text[model_start:model_end]

    assert "PAP_OFFLOAD_EXEC_LAYER_WAVEFRONT" not in text
    assert "_pap_layer_wavefront_forward" not in text
    assert "_pap_should_use_layer_wavefront" not in text
    assert "_pap_start_attention_after_input_norm" not in text
    assert "_pap_finish_attention_after_input_norm" not in text
    assert "_start_pap_attention_wavefront_batch" not in text
    assert "_finish_pap_attention_wavefront_batch" not in text
    assert "override_forward_context" not in model_cls
    assert "UBatchWrapper" not in model_cls


def test_qwen3_moe_has_no_pap_layer_wavefront_microbatch_count() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3_moe.py").read_text()
    model_start = text.index("class Qwen3MoeModel")
    model_end = text.index("\n\nclass Qwen3MoeForCausalLM", model_start)
    model_cls = text[model_start:model_end]

    assert "_pap_moe_layer_wavefront_microbatch_count" not in text
    assert "PAP_RUNNER_MICROBATCH_COUNT" not in model_cls
    assert "_pap_offload_exec_microbatch_count(num_reqs)" not in model_cls


def test_qwen3_pap_direct_mailbox_output_is_opt_in() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert "PAP_DIRECT_MAILBOX_OUTPUT" in method
    assert "direct_mailbox_output_enabled" in method
    assert "can_use_direct_output = (" in method
    assert (
        "direct_mailbox_output_enabled"
        in method[
            method.index("can_use_direct_output = (") : method.index(
                "if can_use_direct_output:"
            )
        ]
    )


def test_qwen3_pap_defers_direct_mailbox_output_release_until_after_o_proj() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()

    forward_start = text.index("    def forward(\n")
    forward_end = text.index("    def _should_use_pap_attention")
    forward_method = text[forward_start:forward_end]

    assert (
        "attn_output, pap_release_messages = self._compute_pap_attention"
        in forward_method
    )
    assert "try:" in forward_method
    assert "output, _ = self.o_proj(attn_output)" in forward_method
    assert "finally:" in forward_method
    assert "for message in pap_release_messages:" in forward_method
    assert "message.release()" in forward_method
    assert "projection_timeline: dict[str, Any] | None" in forward_method
    assert "PAP OFFLOAD_EXEC projection timeline" in forward_method
    direct_pap_branch = forward_method[
        forward_method.index("attn_output, pap_release_messages =") :
    ]
    assert direct_pap_branch.index(
        "output, _ = self.o_proj(attn_output)"
    ) < direct_pap_branch.index("message.release()")

    compute_start = text.index("    def _compute_pap_attention")
    compute_end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    compute_method = text[compute_start:compute_end]

    assert "pap_release_messages: list[Any] = []" in compute_method
    assert "projection_timeline: dict[str, Any] | None = None" in compute_method
    assert "record_projection_trace()" in compute_method
    assert "ubatch_id" not in compute_method
    assert "direct_output = output_batch.view" in compute_method
    assert "pap_release_messages.append(output_message)" in compute_method
    assert "return direct_output, pap_release_messages" in compute_method


def test_pap_projection_runtime_emits_operation_level_ttft_trace() -> None:
    v1_runner_text = (
        ROOT / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
    ).read_text()
    v2_runner_text = (
        ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"
    ).read_text()
    core_text = (ROOT / "vllm" / "v1" / "engine" / "core.py").read_text()

    assert "PAP OFFLOAD_EXEC projection runner forward detail" in v1_runner_text
    assert "PAP OFFLOAD_EXEC projection runner forward detail" in v2_runner_text
    for field in (
        "input_prep_ms",
        "metadata_ms",
        "preprocess_ms",
        "model_forward_ms",
        "hidden_slice_ms",
        "logits_ms",
        "postprocess_tail_ms",
    ):
        assert field in v1_runner_text
        assert field in v2_runner_text

    assert "PAP OFFLOAD_EXEC projection first output" in core_text
    assert "batch_queue.appendleft" in core_text
    for field in (
        "generated_tokens",
        "sched_ms",
        "exec_and_sample_ms",
        "scheduler_update_ms",
        "step_to_first_output_ms",
    ):
        assert field in core_text


def test_qwen3_pap_gate_checks_decode_only() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()

    start = text.index("    def _should_use_pap_attention")
    end = text.index("    def _pap_attention_kv_ready_for_requests")
    method = text[start:end]

    assert 'additional_kwargs.get("pap_enabled")' in method
    assert "max_query_len" in method
    assert "num_scheduled_tokens" in method
    assert "pap_num_scheduled_tokens" in method


def test_qwen3_pap_uses_route_plan_steps_for_offload_exec_descriptor() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    helper_start = text.index("def _pap_offload_exec_step_groups(\n")
    helper_end = text.index("\ndef _pap_offload_exec_base_zmq_port")
    helper = text[helper_start:helper_end]
    start = text.index("    def _compute_pap_attention(\n")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert 'route_group.get("steps"' in helper
    assert "session_request_ids" in helper
    assert "_pap_offload_exec_session_request_id(" in helper
    assert '"r": tuple(session_request_ids)' in helper
    assert "if request_id != str(request_ids[req_index])" in helper
    assert "metadata_template=" in helper
    assert '"a": (float(scaling),) * len(group_steps)' in helper
    assert "_pap_offload_exec_step_groups(" in method
    assert "items=()" in method
    assert "PAPOffloadExecDescriptor(" not in method
    assert "batch_descriptor.item_count" in method
    assert "len(batch_descriptor.items)" not in method
    assert "positions_cpu" not in method
    assert "seq_lens_cpu" not in method
    assert "seq_len != max_seq_len" not in method
    assert '"request_id": request_id' not in method
    assert '"layer_name": self.attn.layer_name' not in method
    assert '"scale": float(self.scaling)' not in method
    assert '"seq_len": seq_len' not in method
    assert "slot_mapping" not in method
    assert "block_id = slot // block_size" not in method
    assert '"block_id": block_id' not in method
    assert '"slot": slot' not in method
    assert "ThreadPoolExecutor" in text
    assert "PAP_REMOTE_ATTENTION_PARALLELISM" not in method


def test_model_runner_passes_pap_block_size_to_forward_context() -> None:
    text = (ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py").read_text()

    assert '"pap_block_size"' in text
    assert '"pap_positions": input_batch.positions' in text
    assert "self.vllm_config.cache_config.block_size" in text


def test_qwen3_pap_projection_trace_uses_batch_descriptor() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _compute_pap_attention")
    end = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    method = text[start:end]

    assert "offload_exec_batches[0][2].layer_name" in method
    assert "sum(batch[2].item_count for batch in offload_exec_batches)" in method
    assert "offload_exec_batches[0][3].layer_name" not in method
    assert "sum(len(batch[3].items) for batch in offload_exec_batches)" not in method


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

    all_offloaded = "sum(len(group.req_indices) for group in step_groups) == num_reqs"
    assert all_offloaded in method
    assert "torch.empty_like(query)" in method
    assert "torch.zeros_like(query)" in method
    assert method.index("step_groups = _pap_offload_exec_step_groups") < method.index(
        all_offloaded
    )


def test_qwen3_pap_imports_prefill_kv_for_offload() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    end = text.index("\n\nclass Qwen3DecoderLayer")
    method = text[start:end]

    assert 'additional_kwargs.get("pap_prefill_kv_handle_by_request")' in method
    assert "PAP_OFFLOAD_KV_TRANSPORT" in method
    assert "PAPTensorTransport.CUDA_IPC" in method
    assert "_pap_prune_imported_prefill_kv(" in method
    assert 'additional_kwargs.get("pap_finished_request_ids")' in method


def test_qwen3_prefill_uses_paged_kv_import_for_cuda_ipc() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    end = text.index("\n\nclass Qwen3DecoderLayer")
    method = text[start:end]

    assert "_publish_pap_prefill_kv_manifests(" in method
    assert "import_prefill_paged_kv(" not in method
    assert "import_prefill_kv_from_paged_cache" not in method
    assert "PAP paged Prefill KV export requires cuda_ipc" in method
    assert "kv_cache=kv_cache" in method


def test_qwen3_prefill_import_does_not_mark_request_ready_for_all_layers() -> None:
    text = (ROOT / "vllm" / "model_executor" / "models" / "qwen3.py").read_text()
    start = text.index("    def _maybe_import_pap_prefill_kv_to_attention")
    end = text.index("\n\nclass Qwen3DecoderLayer")
    method = text[start:end]

    assert 'additional_kwargs["pap_attention_kv_installed_by_request"]' not in method
    assert "installed.add(request_id)" not in method
