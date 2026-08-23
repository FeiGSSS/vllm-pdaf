# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from vllm.pap.runtime_cuda_context_audit import write_runtime_cuda_context_audit


def test_runtime_cuda_context_audit_records_live_partition(monkeypatch, tmp_path):
    output = tmp_path / "context.json"
    monkeypatch.setenv("PAP_RUNTIME_CUDA_CONTEXT_AUDIT_PATH", str(output))
    monkeypatch.setenv("CUDA_MPS_SM_PARTITION", "GPU-fake/prefill")
    monkeypatch.setattr("torch.accelerator.current_device_index", lambda: 0)
    monkeypatch.setattr(
        "vllm.pap.runtime_cuda_context_audit.current_platform.get_device_name",
        lambda _index: "fake-gpu",
    )
    monkeypatch.setattr(
        "vllm.pap.runtime_cuda_context_audit.current_platform.get_device_uuid",
        lambda _index: "GPU-fake",
    )
    monkeypatch.setattr(
        "vllm.pap.runtime_cuda_context_audit.current_platform.num_compute_units",
        lambda _index: 64,
    )

    payload = write_runtime_cuda_context_audit(role="prefill")

    assert payload is not None
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert payload["multiprocessor_count"] == 64
    assert payload["cuda_mps_sm_partition"] == "GPU-fake/prefill"
