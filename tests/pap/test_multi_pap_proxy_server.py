# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from examples.pap.multi_pap_proxy_server import (
    PAPGroup,
    ProjectionInstance,
    build_projection_payload_for_group,
    parse_pap_groups,
    parse_projection_instances,
    select_instances,
)

ROOT = Path(__file__).resolve().parents[2]


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


def test_multi_proxy_logs_ranked_attention_port_as_string() -> None:
    text = (ROOT / "examples" / "pap" / "multi_pap_proxy_server.py").read_text()

    assert "attention=%s:%s projection=%s:%d" in text
    assert "attention=%s:%d projection=%s:%d" not in text


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


def test_multi_proxy_registers_attention_before_prefill_for_local_import() -> None:
    text = (ROOT / "examples/pap/multi_pap_proxy_server.py").read_text()

    register = text.index("attention_sessions = await register_attention_handles")
    prefill = text.index("prefill_resp = await _post_json")
    assert register < prefill
    assert "prefill_prefix_len_from_kv_params(kv_params)" in text
    assert "attach_pap_prefill_attention_params" in text
    assert "pap_offload_exec_zmq_endpoint=group.attention_zmq_endpoint" in text
    assert "register_attention_handles" in text


def test_multi_proxy_marks_attention_kv_installed_only_after_prefill() -> None:
    text = (ROOT / "examples/pap/multi_pap_proxy_server.py").read_text()

    prefill = text.index("prefill_resp = await _post_json")
    prefix_len = text.index("prefix_len = prefill_prefix_len_from_kv_params")
    installed = text.index("pap_attention_kv_installed=prefix_len is not None")
    assert prefill < prefix_len < installed


def test_multi_proxy_decode_barrier_runs_before_projection_request() -> None:
    text = (ROOT / "examples/pap/multi_pap_proxy_server.py").read_text()

    installed = text.index("pap_attention_kv_installed=prefix_len is not None")
    barrier = text.index("decode_barrier.wait")
    projection = text.index("projection_resp = await _post_json")
    assert installed < barrier < projection


def test_multi_proxy_does_not_store_conversation_placement() -> None:
    text = (ROOT / "examples/pap/multi_pap_proxy_server.py").read_text()

    assert "PAPConversationPlacement" not in text
    assert "conversation_placements" not in text
    assert "select_conversation_instances" not in text


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
