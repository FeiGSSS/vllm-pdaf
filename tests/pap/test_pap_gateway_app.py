# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP gateway topology and static routing tests."""

import asyncio
import sys

import pytest

import vllm.pap.gateway.handoff as gateway_handoff
from vllm.pap.gateway.admission import PAPProjectionAdmission
from vllm.pap.gateway.app import parse_args
from vllm.pap.gateway.observability import _prefill_usage_headers
from vllm.pap.gateway.request_pipeline import _pop_conversation_id
from vllm.pap.gateway.routing import PAPConversationRouter, select_instances
from vllm.pap.gateway.topology import (
    PAPGroup,
    ProjectionInstance,
    build_projection_payload_for_group,
    parse_pap_groups,
    parse_projection_instances,
)


def _groups(count: int) -> list[PAPGroup]:
    return [
        PAPGroup(
            "127.0.0.1",
            8100 + index,
            "127.0.0.1",
            8300 + index,
        )
        for index in range(count)
    ]


def _projections(count: int) -> list[ProjectionInstance]:
    return [ProjectionInstance("127.0.0.1", 8200 + index) for index in range(count)]


def test_conversation_id_body_takes_priority_over_session_header() -> None:
    payload = {"conversation_id": "body-session", "model": "qwen"}

    assert _pop_conversation_id(payload, "header-session") == "body-session"
    assert payload == {"model": "qwen"}


def test_conversation_id_falls_back_to_aiperf_session_header() -> None:
    payload = {"model": "qwen"}

    assert _pop_conversation_id(payload, "aiperf-session") == "aiperf-session"


def test_gateway_defaults_to_conversation_affinity(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pap-gateway",
            "--pap-groups",
            "127.0.0.1:8100:127.0.0.1:8300",
            "--projections",
            "127.0.0.1:8200",
        ],
    )

    assert parse_args().routing_policy == "conversation_affinity"


def test_parse_pap_groups_from_compact_spec() -> None:
    groups = parse_pap_groups(
        "127.0.0.1:8100:127.0.0.1:8300,127.0.0.1:8101:127.0.0.1:8301"
    )

    assert groups == _groups(2)


def test_parse_pap_groups_accepts_attention_tcp_port() -> None:
    groups = parse_pap_groups("127.0.0.1:8100:127.0.0.1:8300:9300")

    assert groups[0].attention_tcp_port == 9300
    assert groups[0].attention_tcp_endpoint == "tcp://127.0.0.1:9300"


def test_parse_pap_groups_accepts_ranked_attention_ports() -> None:
    groups = parse_pap_groups("127.0.0.1:8100:127.0.0.1:8300|8301:9300")

    assert groups[0].attention_port == (8300, 8301)
    assert groups[0].attention_base_url == (
        "http://127.0.0.1:8300,http://127.0.0.1:8301"
    )
    assert groups[0].attention_tcp_endpoint == "tcp://127.0.0.1:9300"


def test_prefill_usage_headers_expose_local_cache_tokens() -> None:
    headers = _prefill_usage_headers(
        {
            "usage": {
                "prompt_tokens": 1024,
                "prompt_tokens_details": {"cached_tokens": 768},
            }
        }
    )

    assert headers == {
        "X-PAP-Prefill-Prompt-Tokens": "1024",
        "X-PAP-Prefill-Cached-Tokens": "768",
        "X-PAP-Prefill-Computed-Tokens": "256",
    }


def test_parse_projection_instances_from_compact_spec() -> None:
    assert parse_projection_instances("127.0.0.1:8200,127.0.0.1:8201") == (
        _projections(2)
    )


@pytest.mark.parametrize(
    ("policy", "expected_groups", "expected_projections"),
    [
        (
            "round_robin",
            [8100, 8101, 8100, 8101],
            [8200, 8201, 8200, 8201],
        ),
        (
            "crossbar_round_robin",
            [8100, 8101, 8100, 8101],
            [8200, 8201, 8201, 8200],
        ),
        (
            "projection_sticky",
            [8100, 8101, 8100, 8101],
            [8200, 8201, 8200, 8201],
        ),
    ],
)
def test_static_routing_policies(
    policy: str,
    expected_groups: list[int],
    expected_projections: list[int],
) -> None:
    selected = [
        select_instances(
            index,
            _groups(2),
            _projections(2),
            routing_policy=policy,
        )
        for index in range(4)
    ]

    assert [group.prefill_port for group, _ in selected] == expected_groups
    assert [projection.port for _, projection in selected] == expected_projections


def test_projection_affinity_groups_pas_by_projection() -> None:
    groups = _groups(5)
    projections = _projections(2)
    selected = [
        select_instances(
            index,
            groups,
            projections,
            routing_policy="projection_affinity",
        )
        for index in range(5)
    ]

    assert [projection.port for _, projection in selected] == [
        8200,
        8200,
        8200,
        8201,
        8201,
    ]


def test_conversation_affinity_round_robins_new_conversations() -> None:
    groups = _groups(3)
    projections = _projections(1)
    router = PAPConversationRouter(groups)

    first_round = [
        select_instances(
            index,
            groups,
            projections,
            routing_policy="conversation_affinity",
            conversation_id=f"conv-{index}",
            conversation_router=router,
        )[0].prefill_port
        for index in range(6)
    ]
    second_round = [
        select_instances(
            6 + index,
            groups,
            projections,
            routing_policy="conversation_affinity",
            conversation_id=f"conv-{index}",
            conversation_router=router,
        )[0].prefill_port
        for index in reversed(range(6))
    ]

    assert first_round == [8100, 8101, 8102, 8100, 8101, 8102]
    assert second_round == [8102, 8101, 8100, 8102, 8101, 8100]
    assert router.snapshot() == {
        "conversations": 6,
        "pa_assignments": {"0": 2, "1": 2, "2": 2},
        "pa_requests": {"0": 4, "1": 4, "2": 4},
    }


def test_build_projection_payload_for_group_keeps_kv_uninstalled() -> None:
    group = PAPGroup("127.0.0.1", 8103, "127.0.0.1", 8303, 9303)
    kv_params = {
        "remote_engine_id": "prefill-3",
        "remote_host": "127.0.0.1",
        "remote_port": 5562,
    }

    payload = build_projection_payload_for_group(
        {"model": "qwen", "prompt": "hello"},
        kv_params,
        group,
        prompt_token_ids=[1],
    )

    result = payload["kv_transfer_params"]
    assert result["pap_projection_kv_unaware"] is True
    assert result["pap_attention_endpoint"] == "http://127.0.0.1:8303"
    assert result["pap_attention_tcp_endpoint"] == "tcp://127.0.0.1:9303"
    assert "pap_attention_kv_installed" not in result
    assert "remote_engine_id" not in result
    assert "remote_host" not in result


def test_build_projection_payload_for_group_attaches_prefill_kv_handle() -> None:
    group = PAPGroup("127.0.0.1", 8103, "127.0.0.1", 8303)

    payload = build_projection_payload_for_group(
        {"model": "qwen", "prompt": "hello"},
        {"remote_engine_id": "prefill-3"},
        group,
        prompt_token_ids=[1],
        pap_prefill_kv_handle="req-9",
    )

    assert payload["kv_transfer_params"]["pap_prefill_kv_handle"] == "req-9"


def test_projection_admission_switches_owner_only_between_waves() -> None:
    async def run() -> None:
        group = _groups(1)[0]
        projection_0, projection_1 = _projections(2)
        admission = PAPProjectionAdmission([group])

        await admission.acquire(group, projection_0)
        projection_1_admitted = asyncio.Event()

        async def acquire_projection_1() -> None:
            await admission.acquire(group, projection_1)
            projection_1_admitted.set()

        task = asyncio.create_task(acquire_projection_1())
        await asyncio.sleep(0)
        assert not projection_1_admitted.is_set()
        await admission.release(group, projection_0)
        await asyncio.wait_for(projection_1_admitted.wait(), timeout=1)
        await admission.release(group, projection_1)
        await task

    asyncio.run(run())


@pytest.mark.parametrize("split_done", [False, True])
def test_stream_cleanup_precedes_done_event(monkeypatch, split_done: bool) -> None:
    events: list[object] = []

    async def fake_stream(*args, **kwargs):
        del args, kwargs
        if split_done:
            yield b'data: {"token":1}\n\ndata: [DO'
            yield b"NE]\n\n"
        else:
            yield b'data: {"token":1}\n\n'
            yield b"data: [DONE]\n\n"

    async def fake_cleanup(*args, **kwargs) -> None:
        del args, kwargs
        events.append("cleanup")

    class FakeAdmission:
        async def release(self, group, projection) -> None:
            del group, projection
            events.append("release")

    async def run() -> None:
        stream = gateway_handoff._stream_projection_with_cleanup(
            None,
            "/v1/completions",
            {},
            "request-0",
            [],
            FakeAdmission(),
            _groups(1)[0],
            _projections(1)[0],
        )
        async for chunk in stream:
            events.append(chunk)

    monkeypatch.setattr(gateway_handoff, "_stream_projection", fake_stream)
    monkeypatch.setattr(gateway_handoff, "_cleanup_attention_sessions", fake_cleanup)
    asyncio.run(run())

    cleanup_index = events.index("cleanup")
    release_index = events.index("release")
    before_cleanup = b"".join(
        event for event in events[:cleanup_index] if isinstance(event, bytes)
    )
    after_release = b"".join(
        event for event in events[release_index + 1 :] if isinstance(event, bytes)
    )
    assert before_cleanup == b'data: {"token":1}\n\n'
    assert after_release == b"data: [DONE]\n\n"
