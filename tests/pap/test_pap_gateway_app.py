# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP gateway topology and Dynamo routing tests."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import vllm.pap.gateway.handoff as gateway_handoff
import vllm.pap.gateway.lifecycle as gateway_lifecycle
from vllm.pap.gateway.admission import PAPProjectionAdmission
from vllm.pap.gateway.app import parse_args
from vllm.pap.gateway.dynamo_routing import PAPDynamoRouter
from vllm.pap.gateway.lifecycle import PAPLifecycleManager
from vllm.pap.gateway.load_tracker import PAPLoadTracker
from vllm.pap.gateway.observability import (
    _extract_prefill_cache_usage,
    _merge_prefill_cache_usage,
    _prefill_usage_headers,
)
from vllm.pap.gateway.request_pipeline import (
    _cancel_on_client_disconnect,
    _handle_openai_request,
    _pop_conversation_id,
)
from vllm.pap.gateway.topology import (
    PAPGroup,
    ProjectionInstance,
    build_projection_payload_for_group,
    parse_pap_groups,
    parse_projection_instances,
    select_projection_for_group,
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


def test_gateway_defaults_to_dynamo(monkeypatch) -> None:
    monkeypatch.delenv("PAP_ROUTING_POLICY", raising=False)
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

    args = parse_args()
    assert args.routing_policy == "dynamo"
    assert args.dynamo_prefill_load_scale == 2.0


@pytest.mark.parametrize("source", ["env", "cli"])
def test_gateway_empty_hf_overrides_disable_model_overrides(monkeypatch, source):
    monkeypatch.delenv("PAP_HF_OVERRIDES", raising=False)
    argv = [
        "pap-gateway",
        "--pap-groups",
        "127.0.0.1:8100:127.0.0.1:8300",
        "--projections",
        "127.0.0.1:8200",
    ]
    if source == "env":
        monkeypatch.setenv("PAP_HF_OVERRIDES", "")
    else:
        argv.extend(["--hf-overrides", ""])
    monkeypatch.setattr(sys, "argv", argv)

    assert parse_args().hf_overrides == {}


@pytest.mark.parametrize("source", ["env", "cli"])
@pytest.mark.parametrize(
    "policy",
    [
        "round_robin",
        "crossbar_round_robin",
        "projection_affinity",
        "projection_sticky",
        "conversation_affinity",
    ],
)
def test_gateway_rejects_retired_routing_without_fallback(monkeypatch, source, policy):
    monkeypatch.delenv("PAP_ROUTING_POLICY", raising=False)
    argv = [
        "pap-gateway",
        "--pap-groups",
        "127.0.0.1:8100:127.0.0.1:8300",
        "--projections",
        "127.0.0.1:8200",
    ]
    if source == "env":
        monkeypatch.setenv("PAP_ROUTING_POLICY", policy)
    else:
        argv.extend(["--routing-policy", policy])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "pa_count,projection_count,owners",
    [(7, 1, [0] * 7), (6, 2, [0, 0, 0, 1, 1, 1]), (7, 2, [0, 0, 0, 0, 1, 1, 1])],
)
def test_dynamo_selected_pa_has_fixed_projection_owner(
    pa_count, projection_count, owners
):
    groups, projections = _groups(pa_count), _projections(projection_count)
    assert [
        projections.index(select_projection_for_group(group, groups, projections))
        for group in groups
    ] == owners


def test_dynamo_rejects_cache_salt_before_prefill_can_publish_incompatible_events():
    request = SimpleNamespace(
        json=AsyncMock(return_value={"cache_salt": "tenant-a"}),
        app=SimpleNamespace(
            state=SimpleNamespace(args=SimpleNamespace(routing_policy="dynamo"))
        ),
    )
    response = asyncio.run(_handle_openai_request("/v1/chat/completions", request))
    assert response.status_code == 400
    assert b"cache_salt" in response.body


def test_dynamo_router_reserves_prefill_and_frees_decode() -> None:
    class FakeSelectionService:
        def __init__(self) -> None:
            self.completed: list[str] = []
            self.freed: list[str] = []

        async def select_and_reserve(self, request):
            assert request["token_ids"] == [1, 2, 3]
            assert request["expected_output_tokens"] == 64
            return {
                "worker_id": 1,
                "effective_prefill_tokens": 3,
                "overlap": {},
            }

        async def prefill_complete(self, request_id: str) -> None:
            self.completed.append(request_id)

        async def free_reservation(self, request_id: str) -> None:
            self.freed.append(request_id)

        def loads(self, *, model_name: str):
            return [{"model_name": model_name}]

        def shutdown(self) -> None:
            return None

        def stop_scheduling(self) -> None:
            return None

    async def run() -> None:
        groups = _groups(2)
        service = FakeSelectionService()
        router = PAPDynamoRouter(groups, service, model_name="pap")

        selected = await router.select_group(
            [1, 2, 3],
            request_id="request-0",
            expected_output_tokens=64,
        )
        await router.mark_prefill_completed("request-0")
        router.finish_request("request-0")
        await router.shutdown()

        assert selected == groups[1]
        assert service.completed == ["request-0"]
        assert service.freed == ["request-0"]
        assert router.stats()["active_reservations"] == 0

    asyncio.run(run())


@pytest.mark.parametrize("cancel_before_booking", [False, True])
def test_dynamo_cancelled_selection_releases_committed_booking(cancel_before_booking):
    """Cancellation may race a native booking; cleanup must wait for its outcome."""

    async def run():
        entered, allow_booking = asyncio.Event(), asyncio.Event()
        freed = []

        async def select(request):
            entered.set()
            await allow_booking.wait()
            return {"worker_id": 0}

        async def free(request_id):
            freed.append(request_id)

        service = SimpleNamespace(
            select_and_reserve=select,
            free_reservation=free,
            stop_scheduling=lambda: None,
            shutdown=lambda: None,
        )
        router = PAPDynamoRouter(_groups(1), service, model_name="pap")
        task = asyncio.create_task(
            router.select_group([1], request_id="r", expected_output_tokens=1)
        )
        await entered.wait()
        if not cancel_before_booking:
            allow_booking.set()
            # Complete the native task before its shielded caller resumes.
            await asyncio.sleep(0)
        task.cancel()
        allow_booking.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await router.shutdown()
        assert freed == ["r"]
        assert router.stats()["active_reservations"] == 0

    asyncio.run(run())


def test_dynamo_release_failure_retains_ownership_and_blocks_new_routing():
    async def run():
        service = SimpleNamespace(
            select_and_reserve=AsyncMock(return_value={"worker_id": 0}),
            free_reservation=AsyncMock(side_effect=RuntimeError("release failed")),
            loads=lambda **kwargs: [],
            stop_scheduling=lambda: None,
            shutdown=lambda: None,
        )
        router = PAPDynamoRouter(_groups(1), service, model_name="pap")
        await router.select_group([1], request_id="r", expected_output_tokens=1)
        router.finish_request("r")
        await asyncio.sleep(0)
        assert router.stats()["active_reservations"] == 1
        assert not router.stats()["healthy"]
        with pytest.raises(RuntimeError, match="failed load accounting"):
            await router.select_group([1], request_id="s", expected_output_tokens=1)
        await router.shutdown()

    asyncio.run(run())


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


def test_load_tracker_counts_shared_prefix_blocks_once(monkeypatch) -> None:
    async def run() -> None:
        group = _groups(1)[0]

        async def sample(client):
            del client
            return {"total_kv_tokens": 100_000, "kv_block_size": 16}

        monkeypatch.setattr(
            "vllm.pap.gateway.load_tracker.get_prefill_kv_load",
            sample,
        )
        tracker = PAPLoadTracker({group: 1})
        await tracker.start()
        tracker.begin_request(
            "request-0",
            group,
            prefill_tokens=32,
            decode_capacity_tokens=0,
            prompt_hashes=(b"a", b"b"),
        )
        tracker.mark_prefill_completed(
            "request-0",
            32,
            prompt_hashes=(b"a", b"b"),
        )
        tracker.begin_request(
            "request-1",
            group,
            prefill_tokens=16,
            decode_capacity_tokens=0,
            prompt_hashes=(b"a", b"b", b"c"),
            cached_tokens=32,
        )
        tracker.mark_prefill_completed(
            "request-1",
            48,
            prompt_hashes=(b"a", b"b", b"c"),
            cached_tokens=32,
        )

        assert tracker.snapshot()[group]["non_evictable_kv_tokens"] == 48
        tracker.finish_request("request-0")
        assert tracker.snapshot()[group]["non_evictable_kv_tokens"] == 48

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
