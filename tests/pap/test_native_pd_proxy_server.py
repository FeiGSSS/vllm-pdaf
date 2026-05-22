# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from examples.pap.native_pd_proxy_server import PDServiceClient, _headers


def test_pd_service_client_tracks_role_and_endpoint() -> None:
    client = PDServiceClient(
        client=object(),
        host="127.0.0.1",
        port=8110,
        base_url="http://127.0.0.1:8110",
        role="prefill",
    )

    assert client.role == "prefill"
    assert client.base_url == "http://127.0.0.1:8110"


def test_native_pd_headers_forward_request_id(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "token")

    headers = _headers("req-7")

    assert headers == {
        "Authorization": "Bearer token",
        "X-Request-Id": "req-7",
    }
