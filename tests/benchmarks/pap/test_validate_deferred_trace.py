from __future__ import annotations

import pytest

from benchmarks.pap.tooling.validate_deferred_trace import validate_trace


def _span(count: int) -> dict[str, float | int]:
    return {
        "count": count,
        "mean_ms": 0.1,
        "p50_ms": 0.1,
        "p90_ms": 0.2,
        "p99_ms": 0.3,
        "max_ms": 0.4,
    }


def _projection_trace(
    *,
    layer_count: int,
    peer_count: int | None = None,
    batched_fanout: bool = False,
    model_forward_count: int | None = None,
) -> dict[str, object]:
    spans: dict[str, object] = {
        name: _span(layer_count)
        for name in (
            "qkv_norm_rope_gpu_ms",
            "projection_qk_repack_gpu_ms",
        )
    }
    resolved_peer_count = (
        peer_count if peer_count is not None else layer_count
    )
    spans["output_ready_wait_gpu_ms"] = _span(resolved_peer_count)
    if batched_fanout:
        spans["qkv_batched_fanout_gpu_ms"] = _span(layer_count)
    else:
        spans["qkv_p2p_copy_gpu_ms"] = _span(resolved_peer_count)
    if model_forward_count is not None:
        spans["projection_model_forward_gpu_ms"] = _span(model_forward_count)
    return {
        "enabled": True,
        "scope": "projection_process_critical_chain",
        "role": "projection",
        "collector_count": 1,
        "pending_records": 0,
        "dropped_records": 0,
        "error_records": 0,
        "spans": spans,
    }


def _pd_trace(*, layer_count: int) -> dict[str, object]:
    return {
        "enabled": True,
        "scope": "pd_decode_process_critical_chain",
        "role": "pd_decode",
        "collector_count": 1,
        "pending_records": 0,
        "dropped_records": 0,
        "error_records": 0,
        "spans": {
            "qkv_norm_rope_gpu_ms": _span(layer_count),
            "pd_paged_fa_gpu_ms": _span(layer_count),
        },
    }


def test_projection_trace_requires_matching_layer_and_forward_counts() -> None:
    payload = _projection_trace(layer_count=72)

    counts = validate_trace(
        payload,
        scope="projection_process_critical_chain",
        num_layers=36,
        reference_peer_batches=72,
    )

    assert counts == {"decode_forwards": 2, "layer_calls": 72}


def test_projection_trace_rejects_attention_count_mismatch() -> None:
    payload = _projection_trace(layer_count=72)

    with pytest.raises(ValueError, match="Attention peer-batch mismatch"):
        validate_trace(
            payload,
            scope="projection_process_critical_chain",
            num_layers=36,
            reference_peer_batches=71,
        )


def test_projection_trace_accepts_multi_pa_peer_batches() -> None:
    payload = _projection_trace(layer_count=72, peer_count=216)

    counts = validate_trace(
        payload,
        scope="projection_process_critical_chain",
        num_layers=36,
        reference_peer_batches=216,
    )

    assert counts == {"decode_forwards": 2, "layer_calls": 72}


def test_projection_trace_accepts_batched_fanout() -> None:
    payload = _projection_trace(
        layer_count=72,
        peer_count=216,
        batched_fanout=True,
        model_forward_count=2,
    )

    counts = validate_trace(
        payload,
        scope="projection_process_critical_chain",
        num_layers=36,
        reference_peer_batches=216,
    )

    assert counts == {"decode_forwards": 2, "layer_calls": 72}


def test_projection_trace_rejects_model_forward_count_mismatch() -> None:
    payload = _projection_trace(
        layer_count=72,
        peer_count=216,
        batched_fanout=True,
        model_forward_count=3,
    )

    with pytest.raises(ValueError, match="model-forward count mismatch"):
        validate_trace(
            payload,
            scope="projection_process_critical_chain",
            num_layers=36,
            reference_peer_batches=216,
        )


def test_projection_trace_rejects_batched_fanout_count_mismatch() -> None:
    payload = _projection_trace(
        layer_count=72,
        peer_count=216,
        batched_fanout=True,
    )
    spans = payload["spans"]
    assert isinstance(spans, dict)
    spans["qkv_batched_fanout_gpu_ms"] = _span(71)

    with pytest.raises(ValueError, match="batched fan-out count mismatch"):
        validate_trace(
            payload,
            scope="projection_process_critical_chain",
            num_layers=36,
            reference_peer_batches=216,
        )


def test_pd_trace_rejects_qkv_fa_count_mismatch() -> None:
    payload = _pd_trace(layer_count=72)
    spans = payload["spans"]
    assert isinstance(spans, dict)
    spans["pd_paged_fa_gpu_ms"] = _span(71)

    with pytest.raises(ValueError, match="count mismatch"):
        validate_trace(
            payload,
            scope="pd_decode_process_critical_chain",
            num_layers=36,
        )


@pytest.mark.parametrize(
    "field",
    ["pending_records", "dropped_records", "error_records"],
)
def test_trace_rejects_incomplete_collection(field: str) -> None:
    payload = _pd_trace(layer_count=72)
    payload[field] = 1

    with pytest.raises(ValueError, match=field):
        validate_trace(
            payload,
            scope="pd_decode_process_critical_chain",
            num_layers=36,
        )
