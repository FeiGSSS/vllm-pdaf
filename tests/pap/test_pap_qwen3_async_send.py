from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import torch

from vllm.pap.data_plane import (
    PAPOffloadExecBatchDescriptor,
    PAPOffloadExecDescriptor,
)


class _CapturingExecutor:
    def __init__(self) -> None:
        self.tasks: list[tuple[Callable[[], Any], Future[Any]]] = []

    def submit(self, fn: Callable[[], Any]) -> Future[Any]:
        future: Future[Any] = Future()
        self.tasks.append((fn, future))
        return future

    def run_next(self) -> None:
        fn, future = self.tasks.pop(0)
        try:
            future.set_result(fn())
        except Exception as exc:  # pragma: no cover - mirrors executor behavior.
            future.set_exception(exc)


class _FakeTransport:
    def __init__(self) -> None:
        self.sent: list[tuple[Any, torch.Tensor, str]] = []

    def send_qkv_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        qkv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        self.sent.append((descriptor, qkv.clone(), remote_address))


def test_async_qkv_send_defers_pack_and_transport_send_until_worker_runs() -> None:
    from vllm.model_executor.models.qwen3 import _pap_submit_async_qkv_batch_send

    descriptor_a = PAPOffloadExecDescriptor("req-a", "layer0", 7, 0.125)
    descriptor_b = PAPOffloadExecDescriptor("req-b", "layer0", 8, 0.125)
    batch_descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(descriptor_a, descriptor_b),
    )
    group_items = [
        (
            0,
            descriptor_a,
            (
                torch.tensor([[1.0, 2.0]]),
                torch.tensor([[3.0]]),
                torch.tensor([[4.0]]),
            ),
        ),
        (
            1,
            descriptor_b,
            (
                torch.tensor([[5.0, 6.0]]),
                torch.tensor([[7.0]]),
                torch.tensor([[8.0]]),
            ),
        ),
    ]
    transport = _FakeTransport()
    executor = _CapturingExecutor()

    handle = _pap_submit_async_qkv_batch_send(
        batch_descriptor=batch_descriptor,
        group_items=group_items,
        transport=transport,
        remote_address="attention-rank0",
        executor=executor,
    )

    assert transport.sent == []
    assert len(executor.tasks) == 1

    executor.run_next()
    stats = handle.wait()

    assert stats.send_done_ns > 0
    assert stats.send_ms >= 0.0
    assert len(transport.sent) == 1
    sent_descriptor, sent_qkv, remote_address = transport.sent[0]
    assert sent_descriptor == batch_descriptor
    assert remote_address == "attention-rank0"
    torch.testing.assert_close(
        sent_qkv,
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        ),
    )


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
