from __future__ import annotations

import torch


def test_direct_qkv_batch_for_group_reuses_contiguous_projection_layout() -> None:
    from vllm.model_executor.models.qwen3 import _pap_direct_qkv_batch_for_group

    qkv = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    group_items = [
        (0, None, ()),
        (1, None, ()),
        (2, None, ()),
    ]

    direct = _pap_direct_qkv_batch_for_group(qkv, group_items)

    assert direct is not None
    assert direct.data_ptr() == qkv.data_ptr()
    assert direct.shape == (3, 4)
    torch.testing.assert_close(direct, qkv)


def test_direct_qkv_batch_for_group_rejects_non_contiguous_request_rows() -> None:
    from vllm.model_executor.models.qwen3 import _pap_direct_qkv_batch_for_group

    qkv = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    group_items = [
        (0, None, ()),
        (2, None, ()),
    ]

    assert _pap_direct_qkv_batch_for_group(qkv, group_items) is None
