# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP gateway topology and routing tests."""

import asyncio

import vllm.pap.gateway.app as gateway_app
from vllm.pap.gateway.app import (
    PAPConversationRouter,
    PAPGroup,
    PAPProjectionAdmission,
    ProjectionInstance,
    _pop_conversation_id,
    _prefill_usage_headers,
    build_projection_payload_for_group,
    parse_pap_groups,
    parse_projection_instances,
    select_instances,
)


def test_conversation_id_body_takes_priority_over_session_header() -> None:
    payload = {"conversation_id": "body-session", "model": "qwen"}

    assert _pop_conversation_id(payload, "header-session") == "body-session"
    assert payload == {"model": "qwen"}


def test_conversation_id_falls_back_to_aiperf_session_header() -> None:
    payload = {"model": "qwen"}

    assert _pop_conversation_id(payload, "aiperf-session") == "aiperf-session"


def test_parse_pap_groups_from_compact_spec() -> None:
    groups = parse_pap_groups(
        "127.0.0.1:8100:5559:127.0.0.1:8300,127.0.0.1:8101:5560:127.0.0.1:8301"
    )

    assert groups == [
        PAPGroup(
            prefill_host="127.0.0.1",
            prefill_port=8100,
            prefill_nixl_port=5559,
            attention_host="127.0.0.1",
            attention_port=8300,
        ),
        PAPGroup(
            prefill_host="127.0.0.1",
            prefill_port=8101,
            prefill_nixl_port=5560,
            attention_host="127.0.0.1",
            attention_port=8301,
        ),
    ]


def test_parse_pap_groups_accepts_attention_tcp_port() -> None:
    groups = parse_pap_groups("127.0.0.1:8100:5559:127.0.0.1:8300:9300:10300")

    assert groups == [
        PAPGroup(
            prefill_host="127.0.0.1",
            prefill_port=8100,
            prefill_nixl_port=5559,
            attention_host="127.0.0.1",
            attention_port=8300,
            attention_tcp_port=9300,
            attention_zmq_port=10300,
        )
    ]
    assert groups[0].attention_tcp_endpoint == "tcp://127.0.0.1:9300"
    assert groups[0].attention_zmq_endpoint == "127.0.0.1:10300"


def test_parse_pap_groups_accepts_ranked_attention_ports() -> None:
    groups = parse_pap_groups(
        "127.0.0.1:8100:5559:127.0.0.1:8300|8301:9300|9301:10300|10301"
    )

    assert groups == [
        PAPGroup(
            prefill_host="127.0.0.1",
            prefill_port=8100,
            prefill_nixl_port=5559,
            attention_host="127.0.0.1",
            attention_port=(8300, 8301),
            attention_tcp_port=(9300, 9301),
            attention_zmq_port=(10300, 10301),
        )
    ]
    assert groups[0].attention_base_url == (
        "http://127.0.0.1:8300,http://127.0.0.1:8301"
    )
    assert groups[0].attention_tcp_endpoint == (
        "tcp://127.0.0.1:9300,tcp://127.0.0.1:9301"
    )
    assert groups[0].attention_zmq_endpoint == ("127.0.0.1:10300,127.0.0.1:10301")


def test_prefill_usage_headers_expose_local_cache_tokens() -> None:
    headers = _prefill_usage_headers(
        {
            "usage": {
                "prompt_tokens": 193,
                "prompt_tokens_details": {"cached_tokens": 176},
            }
        }
    )

    assert headers == {
        "X-PAP-Prefill-Prompt-Tokens": "193",
        "X-PAP-Prefill-Cached-Tokens": "176",
        "X-PAP-Prefill-Computed-Tokens": "17",
    }


def test_prefill_usage_headers_ignore_missing_details() -> None:
    assert _prefill_usage_headers({"usage": {"prompt_tokens": 193}}) == {
        "X-PAP-Prefill-Prompt-Tokens": "193"
    }
    assert _prefill_usage_headers({}) == {}


def test_parse_projection_instances_from_compact_spec() -> None:
    projections = parse_projection_instances("127.0.0.1:8200,127.0.0.1:8201")

    assert projections == [
        ProjectionInstance(host="127.0.0.1", port=8200),
        ProjectionInstance(host="127.0.0.1", port=8201),
    ]


def test_select_instances_round_robins_pa_and_projection_independently() -> None:
    groups = [
        PAPGroup("127.0.0.1", 8100 + idx, 5559 + idx, "127.0.0.1", 8300 + idx)
        for idx in range(6)
    ]
    projections = [
        ProjectionInstance("127.0.0.1", 8200),
        ProjectionInstance("127.0.0.1", 8201),
    ]

    selected = [select_instances(i, groups, projections) for i in range(8)]

    assert [pair[0].prefill_port for pair in selected] == [
        8100,
        8101,
        8102,
        8103,
        8104,
        8105,
        8100,
        8101,
    ]
    assert [pair[1].port for pair in selected] == [
        8200,
        8201,
        8200,
        8201,
        8200,
        8201,
        8200,
        8201,
    ]


def test_select_instances_crossbar_round_robin_covers_all_2x2_pairs() -> None:
    groups = [
        PAPGroup("127.0.0.1", 8100 + idx, 5559 + idx, "127.0.0.1", 8300 + idx)
        for idx in range(2)
    ]
    projections = [
        ProjectionInstance("127.0.0.1", 8200 + idx) for idx in range(2)
    ]

    selected = [
        select_instances(
            request_number,
            groups,
            projections,
            routing_policy="crossbar_round_robin",
        )
        for request_number in range(4)
    ]

    assert [
        (group.prefill_port, projection.port)
        for group, projection in selected
    ] == [
        (8100, 8200),
        (8101, 8201),
        (8100, 8201),
        (8101, 8200),
    ]


def test_crossbar_round_robin_balances_non_divisible_topology() -> None:
    groups = [
        PAPGroup("127.0.0.1", 8100 + idx, 5559 + idx, "127.0.0.1", 8300 + idx)
        for idx in range(3)
    ]
    projections = [
        ProjectionInstance("127.0.0.1", 8200 + idx) for idx in range(2)
    ]

    selected = [
        select_instances(
            request_number,
            groups,
            projections,
            routing_policy="crossbar_round_robin",
        )
        for request_number in range(12)
    ]
    pair_counts: dict[tuple[int, int], int] = {}
    for group, projection in selected:
        pair = (group.prefill_port, projection.port)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    assert len(pair_counts) == 6
    assert set(pair_counts.values()) == {2}


def test_select_instances_can_group_pa_by_projection_affinity() -> None:
    groups = [
        PAPGroup("127.0.0.1", 8100 + idx, 5559 + idx, "127.0.0.1", 8300 + idx)
        for idx in range(6)
    ]
    projections = [
        ProjectionInstance("127.0.0.1", 8200),
        ProjectionInstance("127.0.0.1", 8201),
    ]

    selected = [
        select_instances(
            i,
            groups,
            projections,
            routing_policy="projection_affinity",
        )
        for i in range(8)
    ]

    assert [pair[0].prefill_port for pair in selected] == [
        8100,
        8101,
        8102,
        8103,
        8104,
        8105,
        8100,
        8101,
    ]
    assert [pair[1].port for pair in selected] == [
        8200,
        8200,
        8200,
        8201,
        8201,
        8201,
        8200,
        8200,
    ]


def test_projection_affinity_is_stable_for_non_divisible_xy_topology() -> None:
    groups = [
        PAPGroup("127.0.0.1", 8100 + idx, 5559 + idx, "127.0.0.1", 8300 + idx)
        for idx in range(5)
    ]
    projections = [
        ProjectionInstance("127.0.0.1", 8200 + idx) for idx in range(3)
    ]

    selected = [
        select_instances(
            request_number,
            groups,
            projections,
            routing_policy="projection_affinity",
        )
        for request_number in range(15)
    ]
    projection_by_group: dict[int, set[int]] = {}
    for group, projection in selected:
        projection_by_group.setdefault(group.prefill_port, set()).add(projection.port)

    assert all(len(ports) == 1 for ports in projection_by_group.values())
    assert set().union(*projection_by_group.values()) == {8200, 8201, 8202}


def test_select_instances_can_stick_pa_to_projection() -> None:
    groups = [
        PAPGroup("127.0.0.1", 8100 + idx, 5559 + idx, "127.0.0.1", 8300 + idx)
        for idx in range(6)
    ]
    projections = [
        ProjectionInstance("127.0.0.1", 8200),
        ProjectionInstance("127.0.0.1", 8201),
    ]

    selected = [
        select_instances(
            i,
            groups,
            projections,
            routing_policy="projection_sticky",
        )
        for i in range(8)
    ]

    assert [pair[0].prefill_port for pair in selected] == [
        8100,
        8101,
        8100,
        8101,
        8100,
        8101,
        8100,
        8101,
    ]
    assert [pair[1].port for pair in selected] == [
        8200,
        8201,
        8200,
        8201,
        8200,
        8201,
        8200,
        8201,
    ]


def test_conversation_affinity_round_robins_new_conversations() -> None:
    groups = [
        PAPGroup("127.0.0.1", 8100 + idx, 5559 + idx, "127.0.0.1", 8300 + idx)
        for idx in range(3)
    ]
    projections = [ProjectionInstance("127.0.0.1", 8200)]
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
    group = PAPGroup("127.0.0.1", 8103, 5562, "127.0.0.1", 8303, 9303, 10303)
    kv_params = {
        "remote_engine_id": "prefill-3",
        "remote_host": "127.0.0.1",
        "remote_port": 5562,
    }

    payload = build_projection_payload_for_group(
        {"model": "qwen", "prompt": "hello"},
        kv_params,
        group,
    )

    assert payload["kv_transfer_params"]["pap_projection_kv_unaware"] is True
    assert payload["kv_transfer_params"]["pap_attention_endpoint"] == (
        "http://127.0.0.1:8303"
    )
    assert payload["kv_transfer_params"]["pap_attention_tcp_endpoint"] == (
        "tcp://127.0.0.1:9303"
    )
    assert payload["kv_transfer_params"]["pap_offload_exec_zmq_endpoint"] == (
        "127.0.0.1:10303"
    )
    assert "pap_attention_kv_installed" not in payload["kv_transfer_params"]
    assert "pap_attention_endpoint" not in kv_params
    assert "remote_engine_id" not in payload["kv_transfer_params"]
    assert "remote_host" not in payload["kv_transfer_params"]


def test_build_projection_payload_for_group_attaches_prefill_kv_handle() -> None:
    group = PAPGroup("127.0.0.1", 8103, 5562, "127.0.0.1", 8303)

    payload = build_projection_payload_for_group(
        {"model": "qwen", "prompt": "hello"},
        {"remote_engine_id": "prefill-3"},
        group,
        pap_prefill_kv_handle="req-9",
    )

    assert payload["kv_transfer_params"]["pap_prefill_kv_handle"] == "req-9"


def test_projection_admission_switches_pa_owner_only_between_waves() -> None:
    async def run() -> None:
        group = PAPGroup("127.0.0.1", 8100, 5559, "127.0.0.1", 8300)
        projection_0 = ProjectionInstance("127.0.0.1", 8200)
        projection_1 = ProjectionInstance("127.0.0.1", 8201)
        admission = PAPProjectionAdmission([group])

        await admission.acquire(group, projection_0)
        projection_1_admitted = asyncio.Event()
        late_projection_0_admitted = asyncio.Event()

        async def acquire(
            projection: ProjectionInstance,
            admitted: asyncio.Event,
        ) -> None:
            await admission.acquire(group, projection)
            admitted.set()

        projection_1_task = asyncio.create_task(
            acquire(projection_1, projection_1_admitted)
        )
        await asyncio.sleep(0)
        late_projection_0_task = asyncio.create_task(
            acquire(projection_0, late_projection_0_admitted)
        )
        await asyncio.sleep(0)
        assert not projection_1_admitted.is_set()
        assert not late_projection_0_admitted.is_set()

        await admission.release(group, projection_0)
        await asyncio.wait_for(projection_1_admitted.wait(), timeout=1)
        assert not late_projection_0_admitted.is_set()
        await admission.release(group, projection_1)
        await asyncio.wait_for(late_projection_0_admitted.wait(), timeout=1)
        await admission.release(group, projection_0)
        await asyncio.gather(projection_1_task, late_projection_0_task)

        assert await admission.snapshot() == [
            {
                "pa_index": 0,
                "projection_port": None,
                "active_requests": 0,
                "waiting_requests": 0,
            }
        ]

    asyncio.run(run())


def test_projection_admission_batches_same_source_until_handoff_waits() -> None:
    async def run() -> None:
        group = PAPGroup("127.0.0.1", 8100, 5559, "127.0.0.1", 8300)
        projection_0 = ProjectionInstance("127.0.0.1", 8200)
        admission = PAPProjectionAdmission([group])

        await admission.acquire(group, projection_0)
        await admission.acquire(group, projection_0)
        assert await admission.snapshot() == [
            {
                "pa_index": 0,
                "projection_port": 8200,
                "active_requests": 2,
                "waiting_requests": 0,
            }
        ]
        await admission.release(group, projection_0)
        await admission.release(group, projection_0)

    asyncio.run(run())


def test_stream_cleanup_precedes_done_event(monkeypatch) -> None:
    events: list[object] = []

    async def fake_stream(*args, **kwargs):
        del args, kwargs
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
        group = PAPGroup("127.0.0.1", 8100, 5559, "127.0.0.1", 8300)
        projection = ProjectionInstance("127.0.0.1", 8200)
        stream = gateway_app._stream_projection_with_cleanup(
            None,
            "/v1/completions",
            {},
            "request-0",
            [],
            FakeAdmission(),
            group,
            projection,
        )
        async for chunk in stream:
            events.append(chunk)

    monkeypatch.setattr(gateway_app, "_stream_projection", fake_stream)
    monkeypatch.setattr(
        gateway_app,
        "_cleanup_attention_sessions",
        fake_cleanup,
    )
    asyncio.run(run())

    assert events == [
        b'data: {"',
        b'token":1}\n\n',
        "cleanup",
        "release",
        b"data: [DONE]\n\n",
    ]


def test_stream_cleanup_detects_done_split_across_chunks(monkeypatch) -> None:
    events: list[object] = []

    async def fake_stream(*args, **kwargs):
        del args, kwargs
        yield b'data: {"token":1}\n\ndata: [DO'
        yield b'NE]\n\n'

    async def fake_cleanup(*args, **kwargs) -> None:
        del args, kwargs
        events.append("cleanup")

    class FakeAdmission:
        async def release(self, group, projection) -> None:
            del group, projection
            events.append("release")

    async def run() -> None:
        group = PAPGroup("127.0.0.1", 8100, 5559, "127.0.0.1", 8300)
        projection = ProjectionInstance("127.0.0.1", 8200)
        stream = gateway_app._stream_projection_with_cleanup(
            None,
            "/v1/completions",
            {},
            "request-0",
            [],
            FakeAdmission(),
            group,
            projection,
        )
        async for chunk in stream:
            events.append(chunk)

    monkeypatch.setattr(gateway_app, "_stream_projection", fake_stream)
    monkeypatch.setattr(
        gateway_app,
        "_cleanup_attention_sessions",
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
