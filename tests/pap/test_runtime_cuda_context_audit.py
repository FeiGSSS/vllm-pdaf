# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

from vllm.pap.runtime_cuda_context_audit import write_runtime_cuda_context_audit


def test_runtime_cuda_context_audit_records_live_partition(monkeypatch, tmp_path):
    output = tmp_path / "context.json"
    properties = SimpleNamespace(
        name="fake-gpu",
        uuid="GPU-fake",
        multi_processor_count=64,
    )
    monkeypatch.setenv("PAP_RUNTIME_CUDA_CONTEXT_AUDIT_PATH", str(output))
    monkeypatch.setenv("CUDA_MPS_SM_PARTITION", "GPU-fake/prefill")
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("torch.cuda.get_device_properties", lambda _index: properties)

    payload = write_runtime_cuda_context_audit(role="prefill")

    assert payload is not None
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert payload["multiprocessor_count"] == 64
    assert payload["cuda_mps_sm_partition"] == "GPU-fake/prefill"
