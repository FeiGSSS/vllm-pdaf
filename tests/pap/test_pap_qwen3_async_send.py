from __future__ import annotations

import torch


def test_direct_qkv_batch_for_indices_reuses_contiguous_projection_layout() -> None:
    from vllm.model_executor.models.qwen3 import _pap_direct_qkv_batch_for_indices

    qkv = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    direct = _pap_direct_qkv_batch_for_indices(qkv, (1, 2, 3))

    assert direct is not None
    assert direct.data_ptr() == qkv[1].data_ptr()
    torch.testing.assert_close(direct, qkv[1:4])


def test_direct_qkv_batch_for_indices_rejects_non_contiguous_rows() -> None:
    from vllm.model_executor.models.qwen3 import _pap_direct_qkv_batch_for_indices

    qkv = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    assert _pap_direct_qkv_batch_for_indices(qkv, (1, 3)) is None
