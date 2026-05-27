from __future__ import annotations

import gc
from collections import OrderedDict
from threading import Condition, RLock

import torch
import pytest

import vllm.pap.nixl_mailbox as nixl_mailbox
from vllm.pap.nixl_mailbox import (
    InProcessPAPMailboxBackend,
    PAPMailboxActor,
    PAPMailboxMessage,
    PAPNixlMailboxEndpoint,
    _decode_nixl_mailbox_agent_metadata,
    _nixl_mailbox_agent_config,
    _nixl_mailbox_env_float,
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


def test_nixl_mailbox_local_metadata_includes_slot_layout() -> None:
    class FakeWrapper:
        def get_agent_metadata(self):
            return b"raw-agent"

    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint._wrapper = FakeWrapper()
    endpoint._send_buffer = torch.empty(64, dtype=torch.uint8)
    endpoint.device_id = 3
    endpoint.buffer_bytes = 64
    endpoint._slot_count = 4
    endpoint._slot_bytes = 16
    endpoint._slot_protocol_enabled = True

    metadata = _decode_nixl_mailbox_agent_metadata(endpoint.local_agent_metadata)

    assert metadata.agent_metadata == b"raw-agent"
    assert metadata.send_buffer_addr == endpoint._send_buffer.data_ptr()
    assert metadata.device_id == 3
    assert metadata.slot_count == 4
    assert metadata.slot_bytes == 16


def test_nixl_mailbox_bind_peer_accepts_slot_metadata() -> None:
    class FakeWrapper:
        def get_agent_metadata(self):
            return b"raw-agent"

        def add_remote_agent(self, agent_metadata):
            self.agent_metadata = agent_metadata
            return "remote-agent"

    peer = object.__new__(PAPNixlMailboxEndpoint)
    peer._wrapper = FakeWrapper()
    peer._send_buffer = torch.empty(128, dtype=torch.uint8)
    peer.device_id = 2
    peer.buffer_bytes = 128
    peer._slot_count = 2
    peer._slot_bytes = 64
    peer._slot_protocol_enabled = True

    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint._wrapper = FakeWrapper()

    endpoint.bind_peer(peer.local_agent_metadata)

    assert endpoint._peer_agent_name == "remote-agent"
    assert endpoint._wrapper.agent_metadata == b"raw-agent"
    assert endpoint._peer_send_buffer_addr == peer._send_buffer.data_ptr()
    assert endpoint._peer_device_id == 2
    assert endpoint._peer_slot_count == 2
    assert endpoint._peer_slot_bytes == 64


def test_nixl_mailbox_publish_uses_slot_address_when_enabled() -> None:
    class FakeWrapper:
        def send_notif(self, agent_name, notif_msg):
            self.agent_name = agent_name
            self.notif_msg = notif_msg

    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint.actor_id = "projection"
    endpoint.device = torch.device("cpu")
    endpoint.device_id = 0
    endpoint.buffer_bytes = 64
    endpoint._slot_count = 4
    endpoint._slot_bytes = 16
    endpoint._next_send_slot = 0
    endpoint._lock = RLock()
    endpoint._send_buffer = torch.empty(64, dtype=torch.uint8)
    endpoint._peer_agent_name = "attention"
    endpoint._wrapper = FakeWrapper()
    endpoint._trace_enabled = False
    endpoint._slot_protocol_enabled = True

    message = PAPMailboxMessage(
        msg_id="msg-slot",
        kind="attention_task_batch",
        metadata={"layer_name": "layer0"},
        tensor=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )

    endpoint._publish_message(message)

    import json

    payload = json.loads(endpoint._wrapper.notif_msg.decode("utf-8"))
    assert payload["slot_id"] == 0
    assert "addr" not in payload
    assert payload["nbytes"] == 8
    assert endpoint._next_send_slot == 1


def test_nixl_mailbox_publish_uses_free_send_slot_when_previous_slot_leased() -> None:
    class FakeWrapper:
        def send_notif(self, agent_name, notif_msg):
            self.agent_name = agent_name
            self.notif_msg = notif_msg

    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint.actor_id = "projection"
    endpoint.device = torch.device("cpu")
    endpoint.device_id = 0
    endpoint.buffer_bytes = 64
    endpoint._slot_count = 2
    endpoint._slot_bytes = 32
    endpoint._next_send_slot = 0
    endpoint._send_slot_leases = {0: "msg-old"}
    endpoint._send_slot_by_msg = {"msg-old": 0}
    endpoint._send_slot_wait_seconds = 0.0
    endpoint._lock = RLock()
    endpoint._cv = Condition(endpoint._lock)
    endpoint._send_buffer = torch.empty(64, dtype=torch.uint8)
    endpoint._peer_agent_name = "attention"
    endpoint._wrapper = FakeWrapper()
    endpoint._trace_enabled = False
    endpoint._slot_protocol_enabled = True

    message = PAPMailboxMessage(
        msg_id="msg-new",
        kind="attention_task_batch",
        metadata={"layer_name": "layer0"},
        tensor=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )

    endpoint._publish_message(message)

    import json

    payload = json.loads(endpoint._wrapper.notif_msg.decode("utf-8"))
    assert payload["slot_id"] == 1
    assert endpoint._send_slot_leases == {0: "msg-old", 1: "msg-new"}
    assert endpoint._send_slot_by_msg == {"msg-old": 0, "msg-new": 1}


def test_nixl_mailbox_publish_single_slot_sync_avoids_send_slot_lease() -> None:
    class FakeWrapper:
        def send_notif(self, agent_name, notif_msg):
            self.agent_name = agent_name
            self.notif_msg = notif_msg

    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint.actor_id = "projection"
    endpoint.device = torch.device("cpu")
    endpoint.device_id = 0
    endpoint.buffer_bytes = 64
    endpoint._slot_count = 1
    endpoint._slot_bytes = 64
    endpoint._next_send_slot = 0
    endpoint._send_slot_leases = {}
    endpoint._send_slot_by_msg = {}
    endpoint._send_slot_wait_seconds = 0.0
    endpoint._lock = RLock()
    endpoint._cv = Condition(endpoint._lock)
    endpoint._send_buffer = torch.empty(64, dtype=torch.uint8)
    endpoint._peer_agent_name = "attention"
    endpoint._wrapper = FakeWrapper()
    endpoint._trace_enabled = False
    endpoint._slot_protocol_enabled = True
    endpoint._async_send_slots_enabled = False

    message = PAPMailboxMessage(
        msg_id="msg-sync",
        kind="attention_task_batch",
        metadata={"layer_name": "layer0"},
        tensor=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )

    endpoint._publish_message(message)

    import json

    payload = json.loads(endpoint._wrapper.notif_msg.decode("utf-8"))
    assert payload["slot_id"] == 0
    assert endpoint._send_slot_leases == {}
    assert endpoint._send_slot_by_msg == {}


def test_nixl_mailbox_publish_copies_payload_segments_without_packed_tensor() -> None:
    class FakeWrapper:
        def send_notif(self, agent_name, notif_msg):
            self.agent_name = agent_name
            self.notif_msg = notif_msg

    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint.actor_id = "projection"
    endpoint.device = torch.device("cpu")
    endpoint.device_id = 0
    endpoint.buffer_bytes = 128
    endpoint._slot_count = 1
    endpoint._slot_bytes = 128
    endpoint._next_send_slot = 0
    endpoint._send_slot_leases = {}
    endpoint._send_slot_by_msg = {}
    endpoint._lock = RLock()
    endpoint._cv = Condition(endpoint._lock)
    endpoint._send_buffer = torch.empty(128, dtype=torch.uint8)
    endpoint._peer_agent_name = "attention"
    endpoint._wrapper = FakeWrapper()
    endpoint._trace_enabled = False
    endpoint._slot_protocol_enabled = True
    endpoint._async_send_slots_enabled = False
    endpoint._piggyback_acks_enabled = False
    endpoint._msgpack_notifications_enabled = True

    query = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    key = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    value = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
    message = PAPMailboxMessage(
        msg_id="msg-segments",
        kind="attention_task_batch",
        metadata={"layer_name": "layer0"},
        tensor=query,
        payload_segments=(query, key, value),
        payload_shape=(1, 6),
    )

    endpoint._publish_message(message)

    expected = torch.cat([query, key, value], dim=-1).reshape(-1).view(torch.uint8)
    torch.testing.assert_close(endpoint._send_buffer[: expected.numel()], expected)
    payload = nixl_mailbox._decode_nixl_mailbox_notification(
        endpoint._wrapper.notif_msg
    )
    assert payload["shape"] == [1, 6]
    assert payload["nbytes"] == int(expected.numel())


def test_nixl_mailbox_publish_uses_msgpack_notification_when_enabled() -> None:
    class FakeWrapper:
        def send_notif(self, agent_name, notif_msg):
            self.agent_name = agent_name
            self.notif_msg = notif_msg

    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint.actor_id = "projection"
    endpoint.device = torch.device("cpu")
    endpoint.device_id = 0
    endpoint.buffer_bytes = 64
    endpoint._slot_count = 1
    endpoint._slot_bytes = 64
    endpoint._next_send_slot = 0
    endpoint._send_slot_leases = {}
    endpoint._send_slot_by_msg = {}
    endpoint._lock = RLock()
    endpoint._cv = Condition(endpoint._lock)
    endpoint._send_buffer = torch.empty(64, dtype=torch.uint8)
    endpoint._peer_agent_name = "attention"
    endpoint._wrapper = FakeWrapper()
    endpoint._trace_enabled = False
    endpoint._slot_protocol_enabled = True
    endpoint._async_send_slots_enabled = False
    endpoint._piggyback_acks_enabled = False
    endpoint._msgpack_notifications_enabled = True

    message = PAPMailboxMessage(
        msg_id="msg-compact-publish",
        kind="attention_task_batch",
        metadata={"layer_name": "layer0"},
        tensor=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )

    endpoint._publish_message(message)

    assert not endpoint._wrapper.notif_msg.startswith(b"{")
    payload = nixl_mailbox._decode_nixl_mailbox_notification(
        endpoint._wrapper.notif_msg
    )
    assert payload["msg_id"] == "msg-compact-publish"
    assert payload["slot_id"] == 0



def test_nixl_mailbox_ack_releases_send_slot_and_output() -> None:
    import json

    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint._lock = RLock()
    endpoint._cv = Condition(endpoint._lock)
    message = PAPMailboxMessage(
        msg_id="msg-ack",
        kind="attention_task_batch",
        metadata={"layer_name": "layer0"},
        tensor=torch.tensor([[1.0]], dtype=torch.float32),
    )
    endpoint._output_pool = OrderedDict({message.msg_id: message})
    endpoint._send_enqueued_at = {message.msg_id: 1.0}
    endpoint._acked = set()
    endpoint._send_slot_leases = {0: message.msg_id}
    endpoint._send_slot_by_msg = {message.msg_id: 0}

    endpoint._handle_notification(
        json.dumps({"type": "ack", "msg_id": message.msg_id}).encode("utf-8")
    )

    assert endpoint._output_pool == OrderedDict()
    assert endpoint._send_enqueued_at == {}
    assert endpoint._send_slot_leases == {}
    assert endpoint._send_slot_by_msg == {}


def test_nixl_mailbox_piggybacks_pending_acks_on_next_message() -> None:
    import json

    class FakeWrapper:
        def send_notif(self, agent_name, notif_msg):
            self.agent_name = agent_name
            self.notif_msg = notif_msg

    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint.actor_id = "attention"
    endpoint.device = torch.device("cpu")
    endpoint.device_id = 0
    endpoint.buffer_bytes = 64
    endpoint._slot_count = 1
    endpoint._slot_bytes = 64
    endpoint._next_send_slot = 0
    endpoint._send_slot_leases = {}
    endpoint._send_slot_by_msg = {}
    endpoint._send_slot_wait_seconds = 0.0
    endpoint._pending_acks = OrderedDict({"msg-qkv": None})
    endpoint._lock = RLock()
    endpoint._cv = Condition(endpoint._lock)
    endpoint._send_buffer = torch.empty(64, dtype=torch.uint8)
    endpoint._peer_agent_name = "projection"
    endpoint._wrapper = FakeWrapper()
    endpoint._trace_enabled = False
    endpoint._slot_protocol_enabled = True
    endpoint._piggyback_acks_enabled = True

    message = PAPMailboxMessage(
        msg_id="msg-output",
        kind="attention_result_batch",
        metadata={"layer_name": "layer0"},
        tensor=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )

    endpoint._publish_message(message)

    payload = json.loads(endpoint._wrapper.notif_msg.decode("utf-8"))
    assert payload["acks"] == ["msg-qkv"]
    assert endpoint._pending_acks == OrderedDict()


def test_nixl_mailbox_defers_message_ack_when_piggyback_enabled() -> None:
    import json

    class FakeWrapper:
        def __init__(self):
            self.sent_notifs = []

        def send_notif(self, agent_name, notif_msg):
            self.sent_notifs.append((agent_name, notif_msg))

    message = PAPMailboxMessage(
        msg_id="msg-qkv",
        kind="attention_task_batch",
        metadata={"layer_name": "layer0"},
        tensor=torch.tensor([[1.0]], dtype=torch.float32),
    )
    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint._lock = RLock()
    endpoint._cv = Condition(endpoint._lock)
    endpoint._incoming = OrderedDict()
    endpoint._peer_agent_name = "projection"
    endpoint._wrapper = FakeWrapper()
    endpoint._slot_protocol_enabled = True
    endpoint._piggyback_acks_enabled = True
    endpoint._pending_acks = OrderedDict()
    endpoint._read_remote_message = lambda data: message

    endpoint._handle_notification(
        json.dumps(
            {
                "type": "message",
                "msg_id": message.msg_id,
                "kind": message.kind,
            }
        ).encode("utf-8")
    )

    assert endpoint._incoming[message.msg_id] is message
    assert endpoint._pending_acks == OrderedDict({message.msg_id: None})
    assert endpoint._wrapper.sent_notifs == []


def test_nixl_mailbox_processes_piggyback_acks_before_message() -> None:
    import json

    class FakeWrapper:
        def __init__(self):
            self.sent_notifs = []

        def send_notif(self, agent_name, notif_msg):
            self.sent_notifs.append((agent_name, notif_msg))

    incoming = PAPMailboxMessage(
        msg_id="msg-qkv-next",
        kind="attention_task_batch",
        metadata={"layer_name": "layer1"},
        tensor=torch.tensor([[2.0]], dtype=torch.float32),
    )
    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint._lock = RLock()
    endpoint._cv = Condition(endpoint._lock)
    endpoint._incoming = OrderedDict()
    endpoint._output_pool = OrderedDict(
        {
            "msg-output-prev": PAPMailboxMessage(
                msg_id="msg-output-prev",
                kind="attention_result_batch",
                metadata={"layer_name": "layer0"},
                tensor=torch.tensor([[1.0]], dtype=torch.float32),
            )
        }
    )
    endpoint._send_enqueued_at = {"msg-output-prev": 1.0}
    endpoint._acked = set()
    endpoint._send_slot_leases = {0: "msg-output-prev"}
    endpoint._send_slot_by_msg = {"msg-output-prev": 0}
    endpoint._peer_agent_name = "projection"
    endpoint._wrapper = FakeWrapper()
    endpoint._piggyback_acks_enabled = True
    endpoint._pending_acks = OrderedDict()
    endpoint._read_remote_message = lambda data: incoming

    endpoint._handle_notification(
        json.dumps(
            {
                "type": "message",
                "acks": ["msg-output-prev"],
                "msg_id": incoming.msg_id,
                "kind": incoming.kind,
            }
        ).encode("utf-8")
    )

    assert endpoint._output_pool == OrderedDict()
    assert endpoint._send_enqueued_at == {}
    assert endpoint._send_slot_leases == {}
    assert endpoint._send_slot_by_msg == {}
    assert endpoint._incoming[incoming.msg_id] is incoming


def test_nixl_mailbox_msgpack_notification_round_trips_without_json_prefix() -> None:
    payload = {
        "type": "message",
        "msg_id": "msg-compact",
        "kind": "attention_task_batch",
        "metadata": {"layer_name": "layer0"},
        "shape": [2, 3],
        "dtype": "float16",
        "nbytes": 12,
        "slot_id": 0,
        "acks": ["msg-prev"],
    }

    encoded = nixl_mailbox._encode_nixl_mailbox_notification(payload, use_msgpack=True)

    assert not encoded.startswith(b"{")
    assert nixl_mailbox._decode_nixl_mailbox_notification(encoded) == payload


def test_nixl_mailbox_msgpack_ack_releases_send_slot_and_output() -> None:
    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint._lock = RLock()
    endpoint._cv = Condition(endpoint._lock)
    message = PAPMailboxMessage(
        msg_id="msg-compact-ack",
        kind="attention_task_batch",
        metadata={"layer_name": "layer0"},
        tensor=torch.tensor([[1.0]], dtype=torch.float32),
    )
    endpoint._output_pool = OrderedDict({message.msg_id: message})
    endpoint._send_enqueued_at = {message.msg_id: 1.0}
    endpoint._acked = set()
    endpoint._send_slot_leases = {0: message.msg_id}
    endpoint._send_slot_by_msg = {message.msg_id: 0}

    endpoint._handle_notification(
        nixl_mailbox._encode_nixl_mailbox_notification(
            {"type": "ack", "msg_id": message.msg_id}, use_msgpack=True
        )
    )

    assert endpoint._output_pool == OrderedDict()
    assert endpoint._send_slot_leases == {}



def test_nixl_mailbox_sender_loop_does_not_wait_ack_for_slot_protocol() -> None:
    class FakeQueue:
        def __init__(self, message):
            self.message = message
            self.get_calls = 0
            self.task_done_calls = 0

        def get(self, timeout):
            self.get_calls += 1
            if self.get_calls > 1:
                raise KeyboardInterrupt
            return self.message

        def task_done(self):
            self.task_done_calls += 1

    message = PAPMailboxMessage(
        msg_id="msg-send-loop",
        kind="attention_task_batch",
        metadata={"layer_name": "layer0"},
        tensor=torch.tensor([[1.0]], dtype=torch.float32),
    )
    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint._closed = type("Closed", (), {"is_set": lambda self: False})()
    endpoint._send_queue = FakeQueue(message)
    endpoint._trace_enabled = False
    endpoint._lock = RLock()
    endpoint._send_enqueued_at = {}
    endpoint._slot_protocol_enabled = True
    endpoint._async_send_slots_enabled = True
    endpoint._publish_message = lambda msg: {"nbytes": 4}

    def wait_ack(msg_id):
        raise AssertionError("slot protocol sender loop should not wait for ACK")

    endpoint._wait_ack = wait_ack

    try:
        endpoint._sender_loop()
    except KeyboardInterrupt:
        pass

    assert endpoint._send_queue.task_done_calls == 1



def test_nixl_mailbox_sender_loop_does_not_wait_ack_when_piggybacking() -> None:
    class FakeQueue:
        def __init__(self, message):
            self.message = message
            self.get_calls = 0
            self.task_done_calls = 0

        def get(self, timeout):
            self.get_calls += 1
            if self.get_calls > 1:
                raise KeyboardInterrupt
            return self.message

        def task_done(self):
            self.task_done_calls += 1

    message = PAPMailboxMessage(
        msg_id="msg-piggyback-loop",
        kind="attention_task_batch",
        metadata={"layer_name": "layer0"},
        tensor=torch.tensor([[1.0]], dtype=torch.float32),
    )
    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint._closed = type("Closed", (), {"is_set": lambda self: False})()
    endpoint._send_queue = FakeQueue(message)
    endpoint._trace_enabled = False
    endpoint._lock = RLock()
    endpoint._send_enqueued_at = {}
    endpoint._slot_protocol_enabled = True
    endpoint._async_send_slots_enabled = False
    endpoint._piggyback_acks_enabled = True
    endpoint._publish_message = lambda msg: {"nbytes": 4}

    def wait_ack(msg_id):
        raise AssertionError("piggybacked ACKs cannot wait for immediate ACK")

    endpoint._wait_ack = wait_ack

    try:
        endpoint._sender_loop()
    except KeyboardInterrupt:
        pass

    assert endpoint._send_queue.task_done_calls == 1


def test_nixl_mailbox_read_resolves_slot_address_from_peer_metadata() -> None:
    class FakeWrapper:
        def __init__(self):
            self.remote_blocks_data = None

        def get_xfer_descs(self, blocks_data, memory_type):
            if blocks_data[0][0] >= 1000:
                self.remote_blocks_data = blocks_data
            return list(blocks_data)

        def prep_xfer_dlist(self, agent_name, descs):
            return (agent_name, tuple(descs))

        def make_prepped_xfer(self, *args, **kwargs):
            return "xfer"

        def transfer(self, handle):
            return "PROC"

        def check_xfer_state(self, handle):
            return "DONE"

        def release_xfer_handle(self, handle):
            pass

        def release_dlist_handle(self, handle):
            pass

    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint.actor_id = "attention"
    endpoint.device = torch.device("cpu")
    endpoint.device_id = 0
    endpoint.memory_type = "DRAM"
    endpoint.buffer_bytes = 64
    endpoint._peer_agent_name = "projection"
    endpoint._peer_send_buffer_addr = 1000
    endpoint._peer_device_id = 7
    endpoint._peer_slot_count = 4
    endpoint._peer_slot_bytes = 16
    endpoint._wrapper = FakeWrapper()
    endpoint._lock = RLock()
    endpoint._recv_buffer = torch.tensor([1.0, 2.0], dtype=torch.float32).view(torch.uint8)
    endpoint._xfer_poll_sleep_seconds = 0.0
    endpoint._trace_enabled = False
    endpoint._zero_copy_recv_enabled = True

    endpoint._read_remote_message(
        {
            "msg_id": "msg-slot-read",
            "kind": "attention_task_batch",
            "metadata": {"layer_name": "layer0"},
            "shape": [1, 2],
            "dtype": "float32",
            "nbytes": 8,
            "slot_id": 2,
        }
    )

    assert endpoint._wrapper.remote_blocks_data == [(1032, 8, 7)]


def test_nixl_mailbox_read_uses_slot_specific_recv_buffer() -> None:
    class FakeWrapper:
        def __init__(self):
            self.local_blocks_data = None

        def get_xfer_descs(self, blocks_data, memory_type):
            if self.local_blocks_data is None:
                self.local_blocks_data = blocks_data
            return list(blocks_data)

        def prep_xfer_dlist(self, agent_name, descs):
            return (agent_name, tuple(descs))

        def make_prepped_xfer(self, *args, **kwargs):
            return "xfer"

        def transfer(self, handle):
            return "PROC"

        def check_xfer_state(self, handle):
            return "DONE"

        def release_xfer_handle(self, handle):
            pass

        def release_dlist_handle(self, handle):
            pass

    first_slot = torch.tensor([1.0, 2.0], dtype=torch.float32).view(torch.uint8)
    second_slot = torch.tensor([3.0, 4.0], dtype=torch.float32).view(torch.uint8)
    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint.actor_id = "attention"
    endpoint.device = torch.device("cpu")
    endpoint.device_id = 0
    endpoint.memory_type = "DRAM"
    endpoint.buffer_bytes = 16
    endpoint._peer_agent_name = "projection"
    endpoint._peer_send_buffer_addr = 1000
    endpoint._peer_device_id = 7
    endpoint._peer_slot_count = 2
    endpoint._peer_slot_bytes = 8
    endpoint._wrapper = FakeWrapper()
    endpoint._lock = RLock()
    endpoint._recv_buffer = torch.cat([first_slot, second_slot])
    endpoint._recv_slot_count = 2
    endpoint._recv_slot_bytes = 8
    endpoint._xfer_poll_sleep_seconds = 0.0
    endpoint._trace_enabled = False
    endpoint._zero_copy_recv_enabled = True

    message = endpoint._read_remote_message(
        {
            "msg_id": "msg-slot-recv",
            "kind": "attention_task_batch",
            "metadata": {"layer_name": "layer0"},
            "shape": [1, 2],
            "dtype": "float32",
            "nbytes": 8,
            "slot_id": 1,
        }
    )

    expected_addr = endpoint._recv_buffer.data_ptr() + 8
    assert endpoint._wrapper.local_blocks_data == [(expected_addr, 8, 0)]
    assert message.tensor.data_ptr() == expected_addr
    torch.testing.assert_close(
        message.tensor,
        torch.tensor([[3.0, 4.0]], dtype=torch.float32),
    )


def test_nixl_mailbox_zero_copy_uses_free_recv_slot_and_releases_it() -> None:
    class FakeWrapper:
        def __init__(self):
            self.local_blocks_data = None

        def get_xfer_descs(self, blocks_data, memory_type):
            if self.local_blocks_data is None:
                self.local_blocks_data = blocks_data
            return list(blocks_data)

        def prep_xfer_dlist(self, agent_name, descs):
            return (agent_name, tuple(descs))

        def make_prepped_xfer(self, *args, **kwargs):
            return "xfer"

        def transfer(self, handle):
            return "PROC"

        def check_xfer_state(self, handle):
            return "DONE"

        def release_xfer_handle(self, handle):
            pass

        def release_dlist_handle(self, handle):
            pass

    first_slot = torch.tensor([1.0, 2.0], dtype=torch.float32).view(torch.uint8)
    second_slot = torch.tensor([3.0, 4.0], dtype=torch.float32).view(torch.uint8)
    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint.actor_id = "attention"
    endpoint.device = torch.device("cpu")
    endpoint.device_id = 0
    endpoint.memory_type = "DRAM"
    endpoint.buffer_bytes = 16
    endpoint._peer_agent_name = "projection"
    endpoint._peer_send_buffer_addr = 1000
    endpoint._peer_device_id = 7
    endpoint._peer_slot_count = 1
    endpoint._peer_slot_bytes = 16
    endpoint._wrapper = FakeWrapper()
    endpoint._lock = RLock()
    endpoint._cv = Condition(endpoint._lock)
    endpoint._recv_buffer = torch.cat([first_slot, second_slot])
    endpoint._recv_slot_count = 2
    endpoint._recv_slot_bytes = 8
    endpoint._recv_slot_leases = {0: "msg-old"}
    endpoint._recv_slot_wait_seconds = 0.0
    endpoint._xfer_poll_sleep_seconds = 0.0
    endpoint._trace_enabled = False
    endpoint._zero_copy_recv_enabled = True

    message = endpoint._read_remote_message(
        {
            "msg_id": "msg-slot-free",
            "kind": "attention_task_batch",
            "metadata": {"layer_name": "layer0"},
            "shape": [1, 2],
            "dtype": "float32",
            "nbytes": 8,
            "slot_id": 0,
        }
    )

    expected_addr = endpoint._recv_buffer.data_ptr() + 8
    assert endpoint._wrapper.local_blocks_data == [(expected_addr, 8, 0)]
    assert message.tensor.data_ptr() == expected_addr
    assert endpoint._recv_slot_leases == {
        0: "msg-old",
        1: "msg-slot-free",
    }

    message.release()
    message.release()

    assert endpoint._recv_slot_leases == {0: "msg-old"}


def test_mailbox_message_del_releases_unreleased_receive_slot() -> None:
    released = []

    message = PAPMailboxMessage(
        msg_id="msg-leaked",
        kind="attention_result_batch",
        metadata={},
        tensor=torch.tensor([1.0]),
        release_callback=lambda: released.append(True),
    )

    del message
    gc.collect()

    assert released == [True]


def test_nixl_mailbox_failed_zero_copy_read_releases_recv_slot() -> None:
    class FakeWrapper:
        def get_xfer_descs(self, blocks_data, memory_type):
            return list(blocks_data)

        def prep_xfer_dlist(self, agent_name, descs):
            return (agent_name, tuple(descs))

        def make_prepped_xfer(self, *args, **kwargs):
            return "xfer"

        def transfer(self, handle):
            return "PROC"

        def check_xfer_state(self, handle):
            return "ERR"

        def release_xfer_handle(self, handle):
            pass

        def release_dlist_handle(self, handle):
            pass

    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint.actor_id = "attention"
    endpoint.device = torch.device("cpu")
    endpoint.device_id = 0
    endpoint.memory_type = "DRAM"
    endpoint.buffer_bytes = 8
    endpoint._peer_agent_name = "projection"
    endpoint._peer_send_buffer_addr = 1000
    endpoint._peer_device_id = 7
    endpoint._peer_slot_count = 1
    endpoint._peer_slot_bytes = 8
    endpoint._wrapper = FakeWrapper()
    endpoint._lock = RLock()
    endpoint._cv = Condition(endpoint._lock)
    endpoint._recv_buffer = torch.empty(8, dtype=torch.uint8)
    endpoint._recv_slot_count = 1
    endpoint._recv_slot_bytes = 8
    endpoint._recv_slot_leases = {}
    endpoint._recv_slot_wait_seconds = 0.0
    endpoint._xfer_poll_sleep_seconds = 0.0
    endpoint._trace_enabled = False
    endpoint._zero_copy_recv_enabled = True

    with pytest.raises(RuntimeError, match="PAP NIXL transfer failed"):
        endpoint._read_remote_message(
            {
                "msg_id": "msg-fail",
                "kind": "attention_task_batch",
                "metadata": {"layer_name": "layer0"},
                "shape": [1, 2],
                "dtype": "float32",
                "nbytes": 8,
                "slot_id": 0,
            }
        )

    assert endpoint._recv_slot_leases == {}



def test_nixl_mailbox_read_reuses_cached_xfer_dlist_handles() -> None:
    class FakeWrapper:
        def __init__(self):
            self.get_xfer_descs_calls = 0
            self.prep_xfer_dlist_calls = 0
            self.released_dlists = []
            self.released_xfers = []

        def get_xfer_descs(self, blocks_data, memory_type):
            self.get_xfer_descs_calls += 1
            return [tuple(blocks_data[0])]

        def prep_xfer_dlist(self, agent_name, descs):
            self.prep_xfer_dlist_calls += 1
            return (agent_name, tuple(descs))

        def make_prepped_xfer(self, *args, **kwargs):
            return f"xfer-{len(self.released_xfers)}"

        def transfer(self, handle):
            return "PROC"

        def check_xfer_state(self, handle):
            return "DONE"

        def release_xfer_handle(self, handle):
            self.released_xfers.append(handle)

        def release_dlist_handle(self, handle):
            self.released_dlists.append(handle)

    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint.actor_id = "attention"
    endpoint.device = torch.device("cpu")
    endpoint.device_id = 0
    endpoint.memory_type = "DRAM"
    endpoint.buffer_bytes = 64
    endpoint._peer_agent_name = "projection"
    endpoint._peer_send_buffer_addr = 1000
    endpoint._peer_device_id = 7
    endpoint._peer_slot_count = 1
    endpoint._peer_slot_bytes = 64
    endpoint._wrapper = FakeWrapper()
    endpoint._lock = RLock()
    endpoint._recv_buffer = torch.tensor([1.0, 2.0], dtype=torch.float32).view(torch.uint8)
    endpoint._xfer_poll_sleep_seconds = 0.0
    endpoint._trace_enabled = False
    endpoint._zero_copy_recv_enabled = True
    endpoint._cache_xfer_dlists_enabled = True
    endpoint._xfer_dlist_cache = {}
    payload = {
        "msg_id": "msg-cache",
        "kind": "attention_task_batch",
        "metadata": {"layer_name": "layer0"},
        "shape": [1, 2],
        "dtype": "float32",
        "nbytes": 8,
        "slot_id": 0,
    }

    first = endpoint._read_remote_message(payload)
    first.release()
    endpoint._read_remote_message({**payload, "msg_id": "msg-cache-2"})

    assert endpoint._wrapper.get_xfer_descs_calls == 2
    assert endpoint._wrapper.prep_xfer_dlist_calls == 2
    assert endpoint._wrapper.released_xfers == ["xfer-0", "xfer-1"]
    assert endpoint._wrapper.released_dlists == []


def test_nixl_mailbox_close_releases_cached_xfer_dlist_handles() -> None:
    class FakeWrapper:
        def __init__(self):
            self.released = []

        def release_dlist_handle(self, handle):
            self.released.append(handle)

    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint._closed = type("Closed", (), {"set": lambda self: None})()
    endpoint._sender_thread = None
    endpoint._receiver_thread = None
    endpoint._wrapper = FakeWrapper()
    endpoint._xfer_dlist_cache = {
        (1, 2, 3, 4, 5): ("local-h", "remote-h"),
    }

    endpoint.close()

    assert endpoint._wrapper.released == ["local-h", "remote-h"]
    assert endpoint._xfer_dlist_cache == {}


def test_nixl_mailbox_agent_config_preserves_tuned_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConfig:
        def __init__(self, *, num_threads, capture_telemetry):
            self.num_threads = num_threads
            self.capture_telemetry = capture_telemetry

    monkeypatch.delenv("PAP_NIXL_MAILBOX_NUM_THREADS", raising=False)
    monkeypatch.delenv("PAP_NIXL_MAILBOX_CAPTURE_TELEMETRY", raising=False)

    config = _nixl_mailbox_agent_config(FakeConfig)

    assert config.num_threads == 4
    assert config.capture_telemetry is True


def test_nixl_mailbox_agent_config_allows_explicit_thread_count_and_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConfig:
        def __init__(self, *, num_threads, capture_telemetry):
            self.num_threads = num_threads
            self.capture_telemetry = capture_telemetry

    monkeypatch.setenv("PAP_NIXL_MAILBOX_NUM_THREADS", "2")
    monkeypatch.setenv("PAP_NIXL_MAILBOX_CAPTURE_TELEMETRY", "0")

    config = _nixl_mailbox_agent_config(FakeConfig)

    assert config.num_threads == 2
    assert config.capture_telemetry is False


def test_nixl_mailbox_poll_interval_defaults_to_low_latency() -> None:
    import inspect

    source = inspect.getsource(PAPNixlMailboxEndpoint.__init__)

    assert "\"PAP_NIXL_MAILBOX_POLL_SECONDS\", 0.00001" in source


def test_nixl_mailbox_msgpack_notifications_default_enabled() -> None:
    import inspect

    source = inspect.getsource(PAPNixlMailboxEndpoint.__init__)

    assert "PAP_NIXL_MAILBOX_MSGPACK_NOTIF" in source
    assert "\"PAP_NIXL_MAILBOX_MSGPACK_NOTIF\", True" in source



def test_nixl_mailbox_xfer_poll_interval_defaults_to_busy_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PAP_NIXL_MAILBOX_XFER_POLL_SECONDS", raising=False)

    assert _nixl_mailbox_env_float("PAP_NIXL_MAILBOX_XFER_POLL_SECONDS", 0.0) == 0.0


def test_nixl_mailbox_poll_interval_allows_busy_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PAP_NIXL_MAILBOX_POLL_SECONDS", "0")

    assert _nixl_mailbox_env_float("PAP_NIXL_MAILBOX_POLL_SECONDS", 0.00005) == 0.0


def test_nixl_mailbox_poll_interval_rejects_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PAP_NIXL_MAILBOX_POLL_SECONDS", "-0.1")

    with pytest.raises(ValueError, match="must be non-negative"):
        _nixl_mailbox_env_float("PAP_NIXL_MAILBOX_POLL_SECONDS", 0.00005)


def test_nixl_mailbox_poll_notifications_reports_work() -> None:
    class FakeWrapper:
        def __init__(self, notifications):
            self._notifications = notifications

        def get_new_notifs(self):
            return self._notifications

    handled: list[bytes] = []
    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint._wrapper = FakeWrapper({"peer": [b"one", b"two"]})
    endpoint._handle_notification = handled.append

    assert endpoint._poll_notifications() is True
    assert handled == [b"one", b"two"]

    endpoint._wrapper = FakeWrapper({})

    assert endpoint._poll_notifications() is False


def test_nixl_mailbox_recv_polls_inline_before_wait() -> None:
    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint._lock = RLock()
    endpoint._cv = Condition(endpoint._lock)
    endpoint._incoming = OrderedDict()
    endpoint._poll_sleep_seconds = 0.0
    endpoint._inline_poll_enabled = True
    message = PAPMailboxMessage(
        msg_id="msg-inline",
        kind="attention_result_batch",
        metadata={"layer_name": "layer0"},
        tensor=torch.tensor([[1.0]], dtype=torch.float32),
    )
    poll_calls = 0

    def poll_notifications() -> bool:
        nonlocal poll_calls
        poll_calls += 1
        with endpoint._cv:
            endpoint._incoming[message.msg_id] = message
            endpoint._cv.notify_all()
        return True

    endpoint._poll_notifications = poll_notifications

    assert endpoint.recv(message.msg_id, timeout=0.1) is message
    assert poll_calls == 1


def test_nixl_mailbox_wait_ack_polls_inline_before_wait() -> None:
    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint._lock = RLock()
    endpoint._cv = Condition(endpoint._lock)
    endpoint._output_pool = OrderedDict(
        {
            "msg-inline-ack": PAPMailboxMessage(
                msg_id="msg-inline-ack",
                kind="attention_task_batch",
                metadata={"layer_name": "layer0"},
                tensor=torch.tensor([[1.0]], dtype=torch.float32),
            )
        }
    )
    endpoint._send_enqueued_at = {"msg-inline-ack": 1.0}
    endpoint._acked = set()
    endpoint._poll_sleep_seconds = 0.0
    endpoint._inline_poll_enabled = True
    poll_calls = 0

    def poll_notifications() -> bool:
        nonlocal poll_calls
        poll_calls += 1
        with endpoint._cv:
            endpoint._acked.add("msg-inline-ack")
            endpoint._cv.notify_all()
        return True

    endpoint._poll_notifications = poll_notifications

    endpoint._wait_ack("msg-inline-ack")

    assert poll_calls == 1
    assert endpoint._output_pool == OrderedDict()
    assert endpoint._send_enqueued_at == {}


def test_nixl_mailbox_zero_copy_recv_returns_registered_buffer_view() -> None:
    class FakeWrapper:
        def get_xfer_descs(self, blocks_data, memory_type):
            return list(blocks_data)

        def prep_xfer_dlist(self, agent_name, descs):
            return (agent_name, tuple(descs))

        def make_prepped_xfer(self, *args, **kwargs):
            return "xfer"

        def transfer(self, handle):
            return "PROC"

        def check_xfer_state(self, handle):
            return "DONE"

        def release_xfer_handle(self, handle):
            pass

        def release_dlist_handle(self, handle):
            pass

    source = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    raw = source.reshape(-1).view(torch.uint8)
    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint.actor_id = "attention"
    endpoint.device = torch.device("cpu")
    endpoint.device_id = 0
    endpoint.memory_type = "DRAM"
    endpoint.buffer_bytes = int(raw.numel())
    endpoint._peer_agent_name = "projection"
    endpoint._wrapper = FakeWrapper()
    endpoint._recv_buffer = raw.clone()
    endpoint._xfer_poll_sleep_seconds = 0.0
    endpoint._trace_enabled = False
    endpoint._zero_copy_recv_enabled = True

    message = endpoint._read_remote_message(
        {
            "msg_id": "msg-zero-copy",
            "kind": "attention_task_batch",
            "metadata": {"layer_name": "layer0"},
            "shape": [1, 2],
            "dtype": "float32",
            "nbytes": int(raw.numel()),
            "addr": 1234,
            "device_id": 0,
        }
    )

    assert message.tensor.data_ptr() == endpoint._recv_buffer.data_ptr()
    torch.testing.assert_close(message.tensor, source)


def test_nixl_mailbox_inline_publish_sends_without_sender_queue() -> None:
    class FailingQueue:
        def put(self, message):
            raise AssertionError("inline publish should not use sender queue")

    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint._lock = RLock()
    endpoint._output_pool = OrderedDict()
    endpoint._send_enqueued_at = {}
    endpoint._trace_enabled = False
    endpoint._inline_publish_enabled = True
    endpoint._send_queue = FailingQueue()
    published: list[PAPMailboxMessage] = []
    endpoint._publish_message = lambda message: published.append(message) or {}

    message = PAPMailboxMessage(
        msg_id="msg-inline-publish",
        kind="attention_task_batch",
        metadata={"layer_name": "layer0"},
        tensor=torch.tensor([[1.0]], dtype=torch.float32),
    )

    endpoint.send(message)

    assert published == [message]
    assert list(endpoint._output_pool) == [message.msg_id]


def test_nixl_mailbox_inline_publish_waits_before_reusing_send_buffer() -> None:
    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint._lock = RLock()
    old_message = PAPMailboxMessage(
        msg_id="msg-old",
        kind="attention_task_batch",
        metadata={"layer_name": "layer0"},
        tensor=torch.tensor([[1.0]], dtype=torch.float32),
    )
    new_message = PAPMailboxMessage(
        msg_id="msg-new",
        kind="attention_task_batch",
        metadata={"layer_name": "layer1"},
        tensor=torch.tensor([[2.0]], dtype=torch.float32),
    )
    endpoint._output_pool = OrderedDict({old_message.msg_id: old_message})
    endpoint._send_enqueued_at = {}
    endpoint._trace_enabled = False
    endpoint._inline_publish_enabled = True
    events: list[tuple[str, str]] = []

    def wait_ack(msg_id: str) -> None:
        events.append(("wait", msg_id))
        endpoint._output_pool.pop(msg_id, None)

    def publish(message: PAPMailboxMessage) -> dict[str, float | int]:
        events.append(("publish", message.msg_id))
        return {}

    endpoint._wait_ack = wait_ack
    endpoint._publish_message = publish

    endpoint.send(new_message)

    assert events == [("wait", old_message.msg_id), ("publish", new_message.msg_id)]
    assert list(endpoint._output_pool) == [new_message.msg_id]


def test_nixl_mailbox_endpoint_round_trips_message_on_local_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for NIXL mailbox endpoint smoke")
    try:
        from vllm.distributed.nixl_utils import NixlWrapper
    except ImportError:
        pytest.skip("NIXL is not available")
    if NixlWrapper is None:
        pytest.skip("NIXL is not available")

    with torch.inference_mode():
        projection = PAPNixlMailboxEndpoint(
            actor_id="projection-test",
            device=torch.device("cuda:0"),
            buffer_bytes=4096,
        )
        attention = PAPNixlMailboxEndpoint(
            actor_id="attention-test",
            device=torch.device("cuda:0"),
            buffer_bytes=4096,
        )
    projection.bind_peer(attention.local_agent_metadata)
    attention.bind_peer(projection.local_agent_metadata)
    projection.start()
    attention.start()
    try:
        tensor = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
        projection.send(
            PAPMailboxMessage(
                msg_id="nixl-msg-1",
                kind="attention_task",
                metadata={"layer_name": "layer0"},
                tensor=tensor,
            )
        )

        received = attention.recv("nixl-msg-1", timeout=5.0)

        assert received.msg_id == "nixl-msg-1"
        assert received.metadata["layer_name"] == "layer0"
        torch.testing.assert_close(received.tensor, tensor)
        assert projection.output_pool_size == 0
    finally:
        projection.close()
        attention.close()
