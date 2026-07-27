"""Fail-closed validation for bilateral deferred trace artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_PROJECTION_SCOPE = "projection_process_critical_chain"
_PD_SCOPE = "pd_decode_process_critical_chain"
_SCOPES = {
    _PROJECTION_SCOPE: {
        "role": "projection",
        "layer_spans": (
            "qkv_norm_rope_gpu_ms",
            "projection_qk_repack_gpu_ms",
        ),
        "peer_spans": ("output_ready_wait_gpu_ms",),
    },
    _PD_SCOPE: {
        "role": "pd_decode",
        "layer_spans": (
            "qkv_norm_rope_gpu_ms",
            "pd_paged_fa_gpu_ms",
        ),
        "peer_spans": (),
    },
}


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _span_count(spans: Mapping[str, Any], name: str) -> int:
    span = spans.get(name)
    if not isinstance(span, Mapping):
        raise ValueError(f"trace is missing required span {name}")
    count = _nonnegative_int(span.get("count"), name=f"{name}.count")
    if count == 0:
        raise ValueError(f"trace span {name} has no records")
    return count


def validate_trace(
    payload: Mapping[str, Any],
    *,
    scope: str,
    num_layers: int = 36,
    reference_peer_batches: int | None = None,
) -> dict[str, int]:
    """Validate one Projection or PD Decode trace artifact.

    Args:
        payload: Parsed trace JSON object.
        scope: Exact trace scope expected from the target process.
        num_layers: Number of transformer layers in the model.
        reference_peer_batches: Optional PAP Attention-side batch count.

    Returns:
        Validated layer-call and decode-forward counts.

    Raises:
        ValueError: If collection or span-count evidence is incomplete.
    """

    contract = _SCOPES.get(scope)
    if contract is None:
        raise ValueError(f"unsupported deferred trace scope: {scope}")
    if payload.get("enabled") is not True:
        raise ValueError("deferred trace is not enabled")
    if payload.get("scope") != scope:
        raise ValueError(
            f"deferred trace scope mismatch: {payload.get('scope')} != {scope}"
        )
    if payload.get("role") != contract["role"]:
        raise ValueError(
            "deferred trace role mismatch: "
            f"{payload.get('role')} != {contract['role']}"
        )
    collector_count = _nonnegative_int(
        payload.get("collector_count"),
        name="collector_count",
    )
    if collector_count == 0:
        raise ValueError("deferred trace has no collectors")
    for field in ("pending_records", "dropped_records", "error_records"):
        value = _nonnegative_int(payload.get(field), name=field)
        if value != 0:
            raise ValueError(f"deferred trace has nonzero {field}: {value}")

    spans = payload.get("spans")
    if not isinstance(spans, Mapping):
        raise ValueError("deferred trace has no span mapping")
    layer_counts = {
        str(name): _span_count(spans, str(name))
        for name in contract["layer_spans"]
    }
    unique_layer_counts = set(layer_counts.values())
    if len(unique_layer_counts) != 1:
        raise ValueError(f"deferred trace layer-span count mismatch: {layer_counts}")
    layer_calls = unique_layer_counts.pop()
    layers = int(num_layers)
    if layers <= 0 or layer_calls % layers != 0:
        raise ValueError(
            f"layer-call count {layer_calls} is not divisible by {layers}"
        )
    decode_forwards = layer_calls // layers

    if scope == _PROJECTION_SCOPE:
        batched_fanout = spans.get("qkv_batched_fanout_gpu_ms")
        if isinstance(batched_fanout, Mapping):
            batched_count = _span_count(
                spans,
                "qkv_batched_fanout_gpu_ms",
            )
            if batched_count != layer_calls:
                raise ValueError(
                    "deferred trace batched fan-out count mismatch: "
                    f"{batched_count} != {layer_calls}"
                )
        else:
            legacy_copy_count = _span_count(spans, "qkv_p2p_copy_gpu_ms")
            output_wait_count = _span_count(
                spans,
                "output_ready_wait_gpu_ms",
            )
            if legacy_copy_count != output_wait_count:
                raise ValueError(
                    "deferred trace legacy peer-span count mismatch: "
                    f"{legacy_copy_count} != {output_wait_count}"
                )

    peer_counts = {
        str(name): _span_count(spans, str(name))
        for name in contract["peer_spans"]
    }
    unique_peer_counts = set(peer_counts.values())
    if len(unique_peer_counts) > 1:
        raise ValueError(f"deferred trace peer-span count mismatch: {peer_counts}")
    peer_batches = unique_peer_counts.pop() if unique_peer_counts else None

    if reference_peer_batches is not None:
        reference = _nonnegative_int(
            reference_peer_batches,
            name="reference_peer_batches",
        )
        if peer_batches != reference:
            raise ValueError(
                "Attention peer-batch mismatch: "
                f"{peer_batches} != {reference}"
            )
    return {
        "decode_forwards": decode_forwards,
        "layer_calls": layer_calls,
    }


def _load_mapping(path: Path, *, name: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _attention_peer_batches(payload: Mapping[str, Any]) -> int:
    instances = payload.get("instances")
    if instances is None:
        values: Sequence[object] = (payload,)
    elif isinstance(instances, Sequence) and not isinstance(
        instances,
        (str, bytes),
    ):
        values = instances
    else:
        raise ValueError("Attention stats instances must be a sequence")
    total = 0
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"Attention stats instance {index} is invalid")
        stats = value.get("stats", value)
        if not isinstance(stats, Mapping):
            raise ValueError(f"Attention stats instance {index} has no stats")
        total += _nonnegative_int(
            stats.get("offload_exec_peer_batches"),
            name=f"Attention stats instance {index} peer batches",
        )
    return total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--scope", choices=tuple(_SCOPES), required=True)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--attention-stats", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reference = None
    if args.attention_stats is not None:
        reference = _attention_peer_batches(
            _load_mapping(args.attention_stats, name="Attention stats")
        )
    summary = validate_trace(
        _load_mapping(args.trace, name="deferred trace"),
        scope=args.scope,
        num_layers=args.num_layers,
        reference_peer_batches=reference,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
