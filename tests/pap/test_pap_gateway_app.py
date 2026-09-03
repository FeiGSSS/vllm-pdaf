# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP gateway topology and static routing tests."""

import asyncio
import sys

import pytest

import vllm.pap.gateway.handoff as gateway_handoff
import vllm.pap.gateway.lifecycle as gateway_lifecycle
from vllm.pap.gateway.admission import PAPProjectionAdmission
from vllm.pap.gateway.app import parse_args
from vllm.pap.gateway.lifecycle import PAPLifecycleManager
from vllm.pap.gateway.load_tracker import PAPLoadTracker
from vllm.pap.gateway.observability import (
    _extract_prefill_cache_usage,
    _merge_prefill_cache_usage,
    _prefill_usage_headers,
)
from vllm.pap.gateway.request_pipeline import (
    _cancel_on_client_disconnect,
    _pop_conversation_id,
)
from vllm.pap.gateway.routing import (
    PAPConversationRouter,
    estimate_initial_context_load,
    estimate_initial_context_tokens,
    select_instances,
)
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


def test_client_disconnect_cancels_downstream_request_task() -> None:
    async def run() -> None:
        class DisconnectedRequest:
            async def receive(self) -> dict[str, str]:
                return {"type": "http.disconnect"}

        async def downstream_request() -> None:
            await asyncio.Event().wait()

        request_task = asyncio.create_task(downstream_request())
        await _cancel_on_client_disconnect(DisconnectedRequest(), request_task)
        with pytest.raises(asyncio.CancelledError):
            await request_task

    asyncio.run(run())


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


def test_load_tracker_accounts_request_lifecycle_without_runtime_rpc(
    monkeypatch,
) -> None:
    async def run() -> None:
        groups = _groups(2)
        calls: list[int] = []

        async def sample(client):
            calls.append(client)
            return {
                "total_kv_tokens": 100_000,
                "kv_block_size": 16,
            }

        monkeypatch.setattr(
            "vllm.pap.gateway.load_tracker.get_prefill_kv_load",
            sample,
        )
        tracker = PAPLoadTracker({groups[0]: 1, groups[1]: 2})
        await tracker.start()
        tracker.begin_request(
            "request-0",
            groups[0],
            prefill_tokens=1000,
            decode_capacity_tokens=100,
        )
        assert tracker.snapshot()[groups[0]]["projected_kv_tokens"] == 1100
        tracker.mark_prefill_completed("request-0", 1200)
        assert tracker.snapshot()[groups[0]]["projected_kv_tokens"] == 1300
        tracker.finish_request("request-0")
        assert tracker.snapshot()[groups[0]]["projected_kv_tokens"] == 0
        assert calls == [1, 2]

    asyncio.run(run())


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


def test_projection_usage_reports_prefill_cache_tokens() -> None:
    prefill_usage = _extract_prefill_cache_usage(
        {
            "usage": {
                "prompt_tokens": 1024,
                "prompt_tokens_details": {"cached_tokens": 768},
            }
        }
    )

    response = _merge_prefill_cache_usage(
        {
            "usage": {
                "prompt_tokens": 1024,
                "completion_tokens": 32,
                "total_tokens": 1056,
            }
        },
        prefill_usage,
    )

    assert response["usage"] == {
        "prompt_tokens": 1024,
        "completion_tokens": 32,
        "total_tokens": 1056,
        "prompt_tokens_details": {"cached_tokens": 768},
    }


def test_stream_usage_reports_prefill_cache_tokens_across_chunks() -> None:
    async def chunks():
        yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\ndata: {"choices":[],'
        yield b'"usage":{"prompt_tokens":1024,"completion_tokens":2,'
        yield b'"total_tokens":1026}}\n\ndata: [DONE]\n\n'

    async def run() -> bytes:
        prefill_usage = _extract_prefill_cache_usage(
            {
                "usage": {
                    "prompt_tokens": 1024,
                    "prompt_tokens_details": {"cached_tokens": 768},
                }
            }
        )
        return b"".join(
            [
                chunk
                async for chunk in gateway_handoff._stream_with_prefill_cache_usage(
                    chunks(),
                    prefill_usage,
                )
            ]
        )

    output = asyncio.run(run())

    assert b'"prompt_tokens_details":{"cached_tokens":768}' in output
    assert output.endswith(b"data: [DONE]\n\n")


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


def test_conversation_affinity_balances_new_conversation_context() -> None:
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
            initial_context_load=(100, 10, 10, 10, 10, 10)[index],
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

    assert first_round == [8100, 8101, 8102, 8101, 8102, 8101]
    assert second_round == [8101, 8102, 8101, 8102, 8101, 8100]
    assert router.snapshot() == {
        "conversations": 6,
        "pa_assignments": {"0": 1, "1": 3, "2": 2},
        "pa_requests": {"0": 2, "1": 6, "2": 4},
        "pa_initial_context_characters": {"0": 100, "1": 30, "2": 20},
        "pa_reserved_prefill_tokens": {"0": 0, "1": 0, "2": 0},
        "pa_reserved_kv_tokens": {"0": 0, "1": 0, "2": 0},
    }


def test_conversation_affinity_keeps_each_pa_on_one_projection() -> None:
    groups = _groups(6)
    projections = _projections(2)
    router = PAPConversationRouter(groups)

    selected = [
        select_instances(
            index,
            groups,
            projections,
            routing_policy="conversation_affinity",
            conversation_id=f"conv-{index}",
            conversation_router=router,
        )
        for index in range(6)
    ]

    assert [group.prefill_port for group, _ in selected] == [
        8100,
        8101,
        8102,
        8103,
        8104,
        8105,
    ]
    assert [projection.port for _, projection in selected] == [
        8200,
        8200,
        8200,
        8201,
        8201,
        8201,
    ]


def _pa_load(
    *,
    prefill: int,
    projected_kv: int,
    total_kv: int = 100_000,
) -> dict[str, int]:
    return {
        "outstanding_prefill_tokens": prefill,
        "projected_kv_tokens": projected_kv,
        "non_evictable_kv_tokens": projected_kv,
        "total_kv_tokens": total_kv,
        "kv_block_size": 16,
    }


def test_conversation_affinity_filters_capacity_then_prefill_load() -> None:
    groups = _groups(3)
    router = PAPConversationRouter(groups)

    selected, _ = select_instances(
        0,
        groups,
        _projections(1),
        routing_policy="conversation_affinity",
        conversation_id="conv-live-load",
        conversation_router=router,
        initial_context_load=100,
        initial_context_tokens=100,
        decode_capacity_tokens=100,
        request_id="request-0",
        current_pa_loads={
            groups[0]: _pa_load(prefill=0, projected_kv=96_000),
            groups[1]: _pa_load(prefill=500, projected_kv=20_000),
            groups[2]: _pa_load(prefill=100, projected_kv=30_000),
        },
    )

    assert selected == groups[2]
    assert router.has_assignment("conv-live-load")


def test_concurrent_first_turns_see_router_reservations() -> None:
    groups = _groups(2)
    router = PAPConversationRouter(groups)
    loads = {group: _pa_load(prefill=0, projected_kv=0) for group in groups}

    first = router.select_group(
        "conv-0",
        request_number=0,
        initial_context_tokens=1000,
        request_id="request-0",
        current_pa_loads=loads,
    )
    second = router.select_group(
        "conv-1",
        request_number=1,
        initial_context_tokens=1000,
        request_id="request-1",
        current_pa_loads=loads,
    )

    assert first == groups[0]
    assert second == groups[1]


def test_first_turn_routes_to_least_loaded_pa_when_all_are_over_capacity() -> None:
    groups = _groups(2)
    router = PAPConversationRouter(groups)
    loads = {
        group: _pa_load(
            prefill=0,
            projected_kv=98_000,
            total_kv=100_000,
        )
        for group in groups
    }

    selected = router.select_group(
        "conv-full",
        request_number=0,
        initial_context_tokens=1000,
        decode_capacity_tokens=1000,
        request_id="request-full",
        current_pa_loads=loads,
    )

    assert selected == groups[0]
    assert router.has_assignment("conv-full")


def test_initial_context_load_counts_text_without_tokenization() -> None:
    assert (
        estimate_initial_context_load(
            {
                "messages": [
                    {"role": "system", "content": "abcd"},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "123456"}],
                    },
                ]
            }
        )
        == 10
    )
    assert estimate_initial_context_load({"prompt": [1, 2, 3]}) == 12
    assert estimate_initial_context_tokens({"prompt": [1, 2, 3]}) == 3


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
        group = _groups(1)[0]
        projection = _projections(1)[0]
        manager = PAPLifecycleManager()
        lifecycle = manager.create(
            request_id="request-0",
            attention_clients=[],
            prefill_client=None,
            projection_client=None,
            admission=FakeAdmission(),
            group=group,
            projection=projection,
            on_finished=lambda: events.append("finished"),
        )
        lifecycle.mark_attention_registered()
        lifecycle.mark_projection_admitted()
        stream = gateway_handoff._stream_projection_with_cleanup(
            None,
            "/v1/completions",
            {},
            "request-0",
            lifecycle,
        )
        async for chunk in stream:
            events.append(chunk)

    monkeypatch.setattr(gateway_handoff, "_stream_projection", fake_stream)
    monkeypatch.setattr(
        gateway_lifecycle,
        "release_attention_sessions",
        fake_cleanup,
    )
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


def test_lifecycle_cleanup_survives_caller_cancellation(monkeypatch) -> None:
    async def run() -> None:
        events: list[str] = []
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()

        async def delayed_cleanup(*args, **kwargs) -> None:
            del args, kwargs
            events.append("cleanup-started")
            cleanup_started.set()
            await allow_cleanup.wait()
            events.append("cleanup-finished")

        class FakeAdmission:
            async def release(self, group, projection) -> None:
                del group, projection
                events.append("admission-released")

        monkeypatch.setattr(
            gateway_lifecycle,
            "release_attention_sessions",
            delayed_cleanup,
        )
        manager = PAPLifecycleManager()
        lifecycle = manager.create(
            request_id="request-cancelled",
            attention_clients=[],
            prefill_client=None,
            projection_client=None,
            admission=FakeAdmission(),
            group=_groups(1)[0],
            projection=_projections(1)[0],
            on_finished=lambda: events.append("finished"),
        )
        lifecycle.mark_attention_registered()
        lifecycle.mark_projection_admitted()

        caller = asyncio.create_task(lifecycle.terminate("client_cancelled"))
        await cleanup_started.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert manager.stats()["active"] == 1

        allow_cleanup.set()
        while manager.stats()["active"]:
            await asyncio.sleep(0)
        await lifecycle.terminate("duplicate_termination")

        assert events == [
            "cleanup-started",
            "cleanup-finished",
            "admission-released",
            "finished",
        ]
        assert manager.stats()["completed"] == 1
        assert manager.stats()["failed"] == 0

    asyncio.run(run())
