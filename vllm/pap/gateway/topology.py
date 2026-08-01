# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP gateway topology values and service endpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import httpx

from vllm.pap.gateway.clients import PAPServiceClient
from vllm.pap.gateway.payloads import build_projection_kv_unaware_payload

PortSpec = int | tuple[int, ...]


def _parse_port_spec(value: str) -> PortSpec:
    if "|" not in value:
        return int(value)
    ports = tuple(int(part) for part in value.split("|") if part)
    if not ports:
        raise argparse.ArgumentTypeError(f"invalid empty ranked port spec {value!r}")
    return ports


def _format_ranked_endpoints(
    host: str,
    ports: PortSpec,
    *,
    scheme: str,
) -> str:
    if isinstance(ports, int):
        return f"{scheme}{host}:{ports}"
    return ",".join(f"{scheme}{host}:{port}" for port in ports)


@dataclass(frozen=True)
class PAPGroup:
    prefill_host: str
    prefill_port: int
    prefill_nixl_port: int
    attention_host: str
    attention_port: PortSpec
    attention_tcp_port: PortSpec | None = None
    attention_zmq_port: PortSpec | None = None

    @property
    def prefill_base_url(self) -> str:
        return f"http://{self.prefill_host}:{self.prefill_port}"

    @property
    def attention_base_url(self) -> str:
        return _format_ranked_endpoints(
            self.attention_host,
            self.attention_port,
            scheme="http://",
        )

    @property
    def attention_tcp_endpoint(self) -> str | None:
        if self.attention_tcp_port is None:
            return None
        return _format_ranked_endpoints(
            self.attention_host,
            self.attention_tcp_port,
            scheme="tcp://",
        )

    @property
    def attention_zmq_endpoint(self) -> str | None:
        if self.attention_zmq_port is None:
            return None
        return _format_ranked_endpoints(
            self.attention_host,
            self.attention_zmq_port,
            scheme="",
        )


@dataclass(frozen=True)
class ProjectionInstance:
    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def _parse_host_port(value: str, *, expected_parts: int, kind: str) -> list[str]:
    parts = value.split(":")
    if len(parts) != expected_parts or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError(
            f"invalid {kind} spec {value!r}; expected {expected_parts} "
            "colon-separated fields"
        )
    return parts


def parse_pap_groups(spec: str) -> list[PAPGroup]:
    groups: list[PAPGroup] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) not in {5, 6, 7} or any(part == "" for part in parts):
            raise argparse.ArgumentTypeError(
                f"invalid PAP group spec {item!r}; expected 5, 6, or 7 "
                "colon-separated fields"
            )
        groups.append(
            PAPGroup(
                prefill_host=parts[0],
                prefill_port=int(parts[1]),
                prefill_nixl_port=int(parts[2]),
                attention_host=parts[3],
                attention_port=_parse_port_spec(parts[4]),
                attention_tcp_port=None
                if len(parts) == 5
                else _parse_port_spec(parts[5]),
                attention_zmq_port=None
                if len(parts) < 7
                else _parse_port_spec(parts[6]),
            )
        )
    if not groups:
        raise argparse.ArgumentTypeError("at least one PAP group is required")
    return groups


def parse_projection_instances(spec: str) -> list[ProjectionInstance]:
    projections: list[ProjectionInstance] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = _parse_host_port(item, expected_parts=2, kind="Projection")
        projections.append(ProjectionInstance(host=parts[0], port=int(parts[1])))
    if not projections:
        raise argparse.ArgumentTypeError("at least one Projection instance is required")
    return projections


def build_projection_payload_for_group(
    req_data: dict[str, Any],
    kv_transfer_params: dict[str, Any],
    group: PAPGroup,
    *,
    pap_prefill_kv_handle: str | None = None,
    pap_attention_kv_installed: bool = False,
) -> dict[str, Any]:
    return build_projection_kv_unaware_payload(
        req_data,
        kv_transfer_params,
        pap_attention_endpoint=group.attention_base_url,
        pap_attention_tcp_endpoint=group.attention_tcp_endpoint,
        pap_offload_exec_zmq_endpoint=group.attention_zmq_endpoint,
        pap_prefill_kv_handle=pap_prefill_kv_handle,
        pap_attention_kv_installed=pap_attention_kv_installed,
    )


def _make_client(host: str, port: int, role: str) -> PAPServiceClient:
    base_url = f"http://{host}:{port}"
    return PAPServiceClient(
        client=httpx.AsyncClient(
            timeout=None,
            base_url=base_url,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        ),
        host=host,
        port=port,
        base_url=base_url,
        role=role,
    )
