# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP OFFLOAD_EXEC wire metadata."""

from __future__ import annotations

import hashlib
from typing import Any

from vllm.pap.protocol.descriptors import (
    PAPOffloadExecBatchDescriptor,
    PAPOffloadExecDescriptor,
)


def _offload_exec_batch_descriptor_to_metadata(
    descriptor: PAPOffloadExecBatchDescriptor,
) -> dict[str, Any]:
    if descriptor.metadata_template is not None:
        if "a" in descriptor.metadata_template:
            scales = [float(scale) for scale in descriptor.metadata_template["a"]]
        else:
            scales = [float(item.scale) for item in descriptor.items]
        request_ids = list(descriptor.metadata_template["r"])
        steps = [int(step) for step in descriptor.metadata_template["s"]]
        if not (len(request_ids) == len(steps) == len(scales)):
            raise ValueError("compact PAP OFFLOAD_EXEC batch metadata length mismatch")
        return {
            "v": 2,
            "l": descriptor.layer_name,
            "r": request_ids,
            "s": steps,
            "a": scales,
        }
    return {
        "v": 2,
        "l": descriptor.layer_name,
        "r": [item.request_id for item in descriptor.items],
        "s": [int(item.step) for item in descriptor.items],
        "a": [float(item.scale) for item in descriptor.items],
    }


def _offload_exec_batch_plan_payload(
    descriptor: PAPOffloadExecBatchDescriptor,
) -> dict[str, Any]:
    metadata = _offload_exec_batch_descriptor_to_metadata(descriptor)
    payload: dict[str, Any] = {
        "b": descriptor.batch_id_suffix
        or ",".join(
            f"{request_id}@{step}"
            for request_id, step in zip(metadata["r"], metadata["s"])
        ),
        "r": list(metadata["r"]),
        "s": [int(step) for step in metadata["s"]],
        "a": [float(scale) for scale in metadata["a"]],
    }
    return payload


def _offload_exec_batch_plan_id(plan_payload: dict[str, Any]) -> str:
    key = (
        str(plan_payload["b"]),
        tuple(str(request_id) for request_id in plan_payload["r"]),
        tuple(int(step) for step in plan_payload["s"]),
        tuple(float(scale) for scale in plan_payload["a"]),
    )
    return hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:16]


def _offload_exec_batch_descriptor_to_plan_metadata(
    descriptor: PAPOffloadExecBatchDescriptor,
    *,
    sent_plans: set[str],
) -> dict[str, Any]:
    plan_payload = _offload_exec_batch_plan_payload(descriptor)
    plan_id = _offload_exec_batch_plan_id(plan_payload)
    if plan_id in sent_plans:
        return {
            "v": 5,
            "l": descriptor.layer_name,
            "p": plan_id,
        }
    sent_plans.add(plan_id)
    return {
        "v": 4,
        "l": descriptor.layer_name,
        "p": plan_id,
        **plan_payload,
    }


def _offload_exec_batch_descriptor_from_plan_payload(
    layer_name: str,
    plan_payload: dict[str, Any],
    *,
    template_only: bool = False,
) -> PAPOffloadExecBatchDescriptor:
    if "t" in plan_payload:
        raise ValueError(
            "PAP OFFLOAD_EXEC decode-token metadata was removed; "
            "use asynchronous decode-token delivery"
        )
    request_ids = list(plan_payload["r"])
    steps = [int(step) for step in plan_payload["s"]]
    scales = [float(scale) for scale in plan_payload["a"]]
    if not (len(request_ids) == len(steps) == len(scales)):
        raise ValueError("compact PAP OFFLOAD_EXEC batch metadata length mismatch")
    if template_only:
        return PAPOffloadExecBatchDescriptor(
            layer_name=layer_name,
            items=(),
            batch_id_suffix=str(plan_payload["b"]),
            metadata_template={
                "r": tuple(str(request_id) for request_id in request_ids),
                "s": tuple(steps),
                "a": tuple(scales),
            },
        )
    return PAPOffloadExecBatchDescriptor(
        layer_name=layer_name,
        items=tuple(
            PAPOffloadExecDescriptor(
                request_id=str(request_id),
                layer_name=layer_name,
                step=int(step),
                scale=float(scale),
            )
            for request_id, step, scale in zip(
                request_ids,
                steps,
                scales,
            )
        ),
        batch_id_suffix=str(plan_payload["b"]),
    )


def _offload_exec_batch_descriptor_from_metadata(
    metadata: dict[str, Any],
    *,
    plan_cache: dict[str, dict[str, Any]] | None = None,
    template_only: bool = False,
) -> PAPOffloadExecBatchDescriptor:
    if metadata.get("v") == 4:
        if "t" in metadata:
            raise ValueError(
                "PAP OFFLOAD_EXEC decode-token metadata was removed; "
                "use asynchronous decode-token delivery"
            )
        layer_name = str(metadata["l"])
        plan_id = str(metadata["p"])
        plan_payload: dict[str, Any] = {
            "b": str(metadata["b"]),
            "r": list(metadata["r"]),
            "s": list(metadata["s"]),
            "a": list(metadata["a"]),
        }
        if plan_cache is not None:
            plan_cache[plan_id] = plan_payload
        return _offload_exec_batch_descriptor_from_plan_payload(
            layer_name,
            plan_payload,
            template_only=template_only,
        )
    if metadata.get("v") == 5:
        if plan_cache is None:
            raise ValueError("PAP OFFLOAD_EXEC batch plan cache is required")
        layer_name = str(metadata["l"])
        plan_id = str(metadata["p"])
        try:
            plan_payload = plan_cache[plan_id]
        except KeyError as exc:
            raise ValueError(
                f"unknown PAP OFFLOAD_EXEC batch plan id: {plan_id}"
            ) from exc
        return _offload_exec_batch_descriptor_from_plan_payload(
            layer_name,
            plan_payload,
            template_only=template_only,
        )
    if metadata.get("v") == 3 or "t" in metadata:
        raise ValueError(
            "PAP OFFLOAD_EXEC decode-token metadata was removed; "
            "use asynchronous decode-token delivery"
        )
    if metadata.get("v") == 2:
        layer_name = str(metadata["l"])
        request_ids = list(metadata["r"])
        steps = list(metadata["s"])
        scales = list(metadata["a"])
        if not (len(request_ids) == len(steps) == len(scales)):
            raise ValueError("compact PAP OFFLOAD_EXEC batch metadata length mismatch")
        return PAPOffloadExecBatchDescriptor(
            layer_name=layer_name,
            items=tuple(
                PAPOffloadExecDescriptor(
                    request_id=str(request_id),
                    layer_name=layer_name,
                    step=int(step),
                    scale=float(scale),
                )
                for request_id, step, scale in zip(request_ids, steps, scales)
            ),
        )

    layer_name = str(metadata["layer_name"])
    if any("decode_token_ids" in item for item in metadata["items"]):
        raise ValueError(
            "PAP OFFLOAD_EXEC decode-token metadata was removed; "
            "use asynchronous decode-token delivery"
        )
    return PAPOffloadExecBatchDescriptor(
        layer_name=layer_name,
        items=tuple(
            PAPOffloadExecDescriptor(
                request_id=str(item["request_id"]),
                layer_name=layer_name,
                step=int(item["step"]),
                scale=float(item["scale"]),
            )
            for item in metadata["items"]
        ),
    )
