from __future__ import annotations

import pytest
import torch


def test_direct_qkv_batch_for_indices_reuses_contiguous_projection_layout() -> None:
    from vllm.pap.model.projection_io import _pap_direct_qkv_batch_for_indices

    qkv = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    direct = _pap_direct_qkv_batch_for_indices(qkv, (1, 2, 3))

    assert direct is not None
    assert direct.data_ptr() == qkv[1].data_ptr()
    torch.testing.assert_close(direct, qkv[1:4])


def test_direct_qkv_batch_for_indices_rejects_non_contiguous_rows() -> None:
    from vllm.pap.model.projection_io import _pap_direct_qkv_batch_for_indices

    qkv = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    assert _pap_direct_qkv_batch_for_indices(qkv, (1, 3)) is None


def test_qkv_batch_for_indices_gathers_non_contiguous_rows_once() -> None:
    from vllm.pap.model.projection_io import _pap_qkv_batch_for_indices

    qkv = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    indices = torch.tensor([3, 1], dtype=torch.long)

    gathered, direct = _pap_qkv_batch_for_indices(
        qkv,
        (3, 1),
        index_tensor=indices,
    )

    assert gathered is not None
    assert direct is False
    assert gathered.is_contiguous()
    torch.testing.assert_close(gathered, qkv[[3, 1]])


def test_route_index_tensor_is_cached_for_all_layers() -> None:
    from vllm.pap.model.projection_io import _pap_route_index_tensor

    additional_kwargs = {}

    first = _pap_route_index_tensor(
        additional_kwargs,
        (3, 1),
        device=torch.device("cpu"),
    )
    second = _pap_route_index_tensor(
        additional_kwargs,
        (3, 1),
        device=torch.device("cpu"),
    )

    assert second is first
    torch.testing.assert_close(first, torch.tensor([3, 1]))


def test_batched_route_copy_can_be_disabled_for_ab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.pap.config import PAPConfigError, PAPRuntimeConfig

    monkeypatch.setenv("PAP_BATCHED_ROUTE_COPY", "0")
    with pytest.raises(PAPConfigError, match="was removed"):
        PAPRuntimeConfig.from_env()


def test_scatter_attention_output_group_uses_non_contiguous_indices() -> None:
    from vllm.pap.model.projection_io import (
        _pap_scatter_attention_output_group,
    )

    output = torch.zeros((4, 2, 2), dtype=torch.float32)
    remote_output = torch.tensor(
        [
            [30.0, 31.0, 32.0, 33.0],
            [10.0, 11.0, 12.0, 13.0],
        ]
    )

    _pap_scatter_attention_output_group(
        output,
        remote_output,
        req_indices=(3, 1),
        index_tensor=torch.tensor([3, 1]),
    )

    torch.testing.assert_close(output[3].reshape(-1), remote_output[0])
    torch.testing.assert_close(output[1].reshape(-1), remote_output[1])
    torch.testing.assert_close(output[0], torch.zeros((2, 2)))
    torch.testing.assert_close(output[2], torch.zeros((2, 2)))


def test_scatter_attention_output_group_reuses_contiguous_slice() -> None:
    from vllm.pap.model.projection_io import (
        _pap_scatter_attention_output_group,
    )

    output = torch.zeros((4, 2, 2), dtype=torch.float32)
    remote_output = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    _pap_scatter_attention_output_group(
        output,
        remote_output,
        req_indices=(1, 2),
        index_tensor=None,
    )

    torch.testing.assert_close(output[1:3].reshape(2, 4), remote_output)
