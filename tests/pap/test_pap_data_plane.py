import os

import pytest
import torch

from vllm.pap.data_plane import (
    PAPOffloadExecDescriptor,
    PAPOffloadKVDescriptor,
    PAPP2PNCCLOffloadExecTransport,
    PAPTensorTransport,
    build_p2p_nccl_offload_exec_transport,
    offload_exec_transport_from_env,
    performance_mode_requires_gpu_data_plane,
)


class FakeP2PEngine:
    def __init__(self) -> None:
        self.sent = []
        self.received = {}

    def send_tensor(self, tensor_id, tensor, remote_address):
        self.sent.append((tensor_id, tensor, remote_address))
        return True

    def recv_tensor(self, tensor_id, remote_address):
        return self.received[(tensor_id, remote_address)]


class FakeP2PEngineWithConfig(FakeP2PEngine):
    def __init__(self, *, local_rank, config, hostname, port_offset):
        super().__init__()
        self.local_rank = local_rank
        self.config = config
        self.hostname = hostname
        self.port_offset = port_offset


class FakeRetryP2PEngine(FakeP2PEngine):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor
        self.calls = 0

    def recv_tensor(self, tensor_id, remote_address):
        self.calls += 1
        if self.calls == 1:
            return None
        return self.tensor


def test_offload_exec_descriptor_uses_stable_tensor_ids() -> None:
    descriptor = PAPOffloadExecDescriptor(
        request_id="cmpl-1",
        layer_name="model.layers.0.self_attn.attn",
        step=7,
        scale=0.125,
    )

    assert descriptor.qkv_tensor_id == "cmpl-1#model.layers.0.self_attn.attn#7#qkv"
    assert (
        descriptor.output_tensor_id
        == "cmpl-1#model.layers.0.self_attn.attn#7#attn_out"
    )


def test_p2p_nccl_offload_exec_transport_delegates_to_engine() -> None:
    engine = FakeP2PEngine()
    transport = PAPP2PNCCLOffloadExecTransport(engine)
    descriptor = PAPOffloadExecDescriptor(
        request_id="cmpl-1",
        layer_name="layer0",
        step=3,
        scale=1.0,
    )
    qkv = torch.ones(1, 8)
    out = torch.zeros(1, 4)
    engine.received[(descriptor.output_tensor_id, "127.0.0.1:9000")] = out

    transport.send_qkv(descriptor, qkv, remote_address="127.0.0.1:9000")
    received = transport.recv_output(descriptor, remote_address="127.0.0.1:9000")

    assert engine.sent == [
        (descriptor.qkv_tensor_id, qkv, "127.0.0.1:9000"),
    ]
    assert received is out


def test_build_p2p_nccl_offload_exec_transport_creates_engine_config() -> None:
    transport = build_p2p_nccl_offload_exec_transport(
        local_rank=1,
        kv_port=10300,
        hostname="127.0.0.1",
        port_offset=2,
        kv_buffer_size=1234,
        extra_config={"send_type": "PUT", "nccl_num_channels": "4"},
        engine_cls=FakeP2PEngineWithConfig,
    )

    engine = transport.engine
    assert engine.local_rank == 1
    assert engine.hostname == "127.0.0.1"
    assert engine.port_offset == 2
    assert engine.config.kv_connector == "P2pNcclConnector"
    assert engine.config.kv_role == "kv_both"
    assert engine.config.kv_port == 10300
    assert engine.config.kv_buffer_size == 1234
    assert engine.config.get_from_extra_config("send_type", "") == "PUT"
    assert engine.config.get_from_extra_config("nccl_num_channels", "") == "4"


def test_build_p2p_nccl_offload_exec_transport_defaults_to_get() -> None:
    transport = build_p2p_nccl_offload_exec_transport(
        local_rank=0,
        kv_port=10300,
        hostname="127.0.0.1",
        engine_cls=FakeP2PEngineWithConfig,
    )

    assert transport.engine.config.get_from_extra_config("send_type", "") == "GET"


def test_build_p2p_nccl_offload_exec_transport_defaults_to_nccl_p2p_fallback(
    monkeypatch,
) -> None:
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.delenv("PAP_OFFLOAD_EXEC_NCCL_P2P_DISABLE", raising=False)

    build_p2p_nccl_offload_exec_transport(
        local_rank=0,
        kv_port=10300,
        hostname="127.0.0.1",
        engine_cls=FakeP2PEngineWithConfig,
    )

    assert os.environ["NCCL_P2P_DISABLE"] == "1"


def test_build_p2p_nccl_offload_exec_transport_allows_nccl_p2p_override(
    monkeypatch,
) -> None:
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NCCL_P2P_DISABLE", "0")

    build_p2p_nccl_offload_exec_transport(
        local_rank=0,
        kv_port=10300,
        hostname="127.0.0.1",
        engine_cls=FakeP2PEngineWithConfig,
    )

    assert os.environ["NCCL_P2P_DISABLE"] == "0"


def test_p2p_nccl_offload_exec_transport_retries_empty_get(monkeypatch) -> None:
    out = torch.zeros(1, 4)
    engine = FakeRetryP2PEngine(out)
    transport = PAPP2PNCCLOffloadExecTransport(engine)
    descriptor = PAPOffloadExecDescriptor(
        request_id="cmpl-1",
        layer_name="layer0",
        step=3,
        scale=1.0,
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_RECV_TIMEOUT", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_RECV_POLL_SECONDS", "0")

    received = transport.recv_output(descriptor, remote_address="127.0.0.1:9000")

    assert received is out
    assert engine.calls == 2


def test_offload_kv_descriptor_rejects_prototype_transport() -> None:
    with pytest.raises(ValueError, match="OFFLOAD_KV"):
        PAPOffloadKVDescriptor(
            request_id="cmpl-1",
            layer_name="layer0",
            seq_len=32,
            block_ids=(1, 2),
            transport=PAPTensorTransport.PROTOTYPE_HTTP,
        )


def test_performance_mode_rejects_http_tcp_tensor_transports() -> None:
    with pytest.raises(RuntimeError, match="Prefill-to-Attention"):
        performance_mode_requires_gpu_data_plane(
            pap_mode="true_split_performance",
            prefill_attention_transport=PAPTensorTransport.PROTOTYPE_HTTP,
            projection_attention_transport=PAPTensorTransport.NCCL_P2P,
        )
    with pytest.raises(RuntimeError, match="Projection-to-Attention"):
        performance_mode_requires_gpu_data_plane(
            pap_mode="true_split_performance",
            prefill_attention_transport=PAPTensorTransport.CUDA_IPC,
            projection_attention_transport=PAPTensorTransport.PROTOTYPE_TCP,
        )


def test_performance_mode_accepts_cuda_ipc_and_nccl() -> None:
    performance_mode_requires_gpu_data_plane(
        pap_mode="true_split_performance",
        prefill_attention_transport=PAPTensorTransport.CUDA_IPC,
        projection_attention_transport=PAPTensorTransport.NCCL_P2P,
    )


def test_offload_exec_transport_from_env(monkeypatch) -> None:
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_TRANSPORT", "nccl")
    assert offload_exec_transport_from_env() is PAPTensorTransport.NCCL_P2P
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_TRANSPORT", "tcp")
    assert offload_exec_transport_from_env() is PAPTensorTransport.PROTOTYPE_TCP
