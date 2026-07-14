from __future__ import annotations

import torch

from vllm.pap.transport.mailbox import (
    InProcessPAPMailboxBackend,
    PAPMailboxActor,
    PAPMailboxMessage,
)


def test_mailbox_actor_delivers_output_message_to_peer_task_pool() -> None:
    backend = InProcessPAPMailboxBackend()
    projection = PAPMailboxActor("projection", backend=backend)
    attention = PAPMailboxActor("attention", backend=backend)
    projection.bind_peer(attention)
    attention.bind_peer(projection)

    message = PAPMailboxMessage(
        msg_id="msg-1",
        kind="attention_task",
        metadata={
            "layer_name": "layer0",
            "request_id": "req-a",
            "step": 1,
        },
        tensor=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
    )

    projection.enqueue_output(message)
    delivered = attention.pop_task("msg-1", timeout=1.0)

    assert delivered.msg_id == "msg-1"
    assert delivered.metadata["layer_name"] == "layer0"
    torch.testing.assert_close(delivered.tensor, message.tensor)
    assert projection.output_pool_size == 0


def test_mailbox_actor_preserves_output_until_delivery_ack() -> None:
    backend = InProcessPAPMailboxBackend(auto_ack=False)
    projection = PAPMailboxActor("projection", backend=backend)
    attention = PAPMailboxActor("attention", backend=backend)
    projection.bind_peer(attention)
    attention.bind_peer(projection)

    message = PAPMailboxMessage(
        msg_id="msg-2",
        kind="attention_task",
        metadata={"layer_name": "layer0"},
        tensor=torch.tensor([[4.0]], dtype=torch.float32),
    )

    projection.enqueue_output(message)

    assert attention.pop_task("msg-2", timeout=1.0).msg_id == "msg-2"
    assert projection.output_pool_size == 1

    backend.ack("projection", "msg-2")

    assert projection.output_pool_size == 0
