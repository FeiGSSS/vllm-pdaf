# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest

from vllm.pap.kv_connector import PAPPrefillConnector
from vllm.pap.runtime_cuda_context_audit import write_runtime_cuda_context_audit


@pytest.mark.parametrize("through_connector", [False, True])
def test_runtime_cuda_context_audit_records_live_partition(
    monkeypatch, tmp_path, through_connector
):
    output = tmp_path / "context.json"
    monkeypatch.setenv("PAP_RUNTIME_CUDA_CONTEXT_AUDIT_PATH", str(output))
    monkeypatch.setenv("CUDA_MPS_SM_PARTITION", "GPU-fake/prefill")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr("torch.accelerator.current_device_index", lambda: 0)
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda _index: SimpleNamespace(
            name="fake-gpu", uuid="fake", multi_processor_count=64
        ),
    )

    if through_connector:
        connector = object.__new__(PAPPrefillConnector)
        connector.register_kv_caches({})
        payload = json.loads(output.read_text(encoding="utf-8"))
    else:
        payload = write_runtime_cuda_context_audit(role="prefill")

    assert payload is not None
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert payload["multiprocessor_count"] == 64
    assert payload["device_uuid"] == "GPU-fake"
    assert payload["role"] == "prefill"
    assert payload["cuda_mps_sm_partition"] == "GPU-fake/prefill"
