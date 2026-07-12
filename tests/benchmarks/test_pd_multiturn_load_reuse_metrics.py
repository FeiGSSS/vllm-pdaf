from __future__ import annotations

import json
import sys
from copy import deepcopy

import pytest

from benchmarks.multi_turn.pd_multiturn_load_reuse_metrics import (
    STATUS,
    main,
    parse_nixl_metrics,
    validate_pd_multiturn_load_reuse,
)


MIB = 2**20
PROMPTS = (
    (100, 140, 180, 220, 260),
    (104, 144, 184, 224, 264),
)


def _result() -> dict[str, object]:
    requests = []
    transitions = []
    for conversation, prompts in enumerate(PROMPTS):
        for round_index, prompt_tokens in enumerate(prompts, start=1):
            requests.append(
                {
                    "conversation_index": conversation,
                    "round": round_index,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 16,
                    "ttft_ms": 10.0 + round_index,
                    "tpot_ms": 2.0,
                    "latency_ms": 42.0,
                    "eof_latency_ms": 43.0,
                    "prefill": {
                        "prompt_tokens": None,
                        "cached_tokens": None,
                        "computed_tokens": None,
                    },
                }
            )
        for from_round in range(1, 5):
            previous_prompt = prompts[from_round - 1]
            previous_boundary = previous_prompt // 16 * 16
            transitions.append(
                {
                    "conversation_index": conversation,
                    "from_round": from_round,
                    "to_round": from_round + 1,
                    "previous_prompt_tokens": previous_prompt,
                    "expected_cached_tokens": previous_boundary + 16,
                    "decode_derived_hit_tokens": 16,
                    "actual_cached_tokens": None,
                }
            )
    return {
        "architecture": "pd",
        "profile": {
            "rounds": 5,
            "active_conversations": 2,
            "block_size": 16,
            "arrival": {"mode": "staggered", "interval_ms": 25},
        },
        "implementation": {"offload_exec_transport": "nixl-push"},
        "requests": requests,
        "cache_validation": {
            "status": "requires_pd_metrics",
            "transitions": transitions,
        },
        "validity": {"status": "passed", "load_gate": "passed"},
    }


def _metrics(
    *,
    compute: int,
    local: int,
    external: int,
    transfers: int,
    transfer_seconds: float,
    transferred_bytes: int,
    descriptors: int,
    bytes_count: int | None = None,
    descriptor_count: int | None = None,
    failed_transfers: int = 0,
    failed_notifications: int = 0,
    expired_requests: int = 0,
) -> str:
    lines = [
        "vllm:prompt_tokens_by_source_total{"
        f'engine="0",source="local_compute"}} {compute}.0',
        "vllm:prompt_tokens_by_source_total{"
        f'engine="0",source="local_cache_hit"}} {local}.0',
        "vllm:prompt_tokens_by_source_total{"
        f'engine="0",source="external_kv_transfer"}} {external}.0',
        f'vllm:nixl_xfer_time_seconds_count{{engine="0"}} {transfers}.0',
        "vllm:nixl_xfer_time_seconds_sum{engine=\"0\"} "
        f"{transfer_seconds}",
        "vllm:nixl_bytes_transferred_count{engine=\"0\"} "
        f"{transfers if bytes_count is None else bytes_count}.0",
        "vllm:nixl_bytes_transferred_sum{engine=\"0\"} "
        f"{transferred_bytes}.0",
        "vllm:nixl_num_descriptors_count{engine=\"0\"} "
        f"{transfers if descriptor_count is None else descriptor_count}.0",
        f'vllm:nixl_num_descriptors_sum{{engine="0"}} {descriptors}.0',
        "vllm:nixl_num_failed_transfers_total{engine=\"0\"} "
        f"{failed_transfers}.0",
        "vllm:nixl_num_failed_notifications_total{engine=\"0\"} "
        f"{failed_notifications}.0",
        "vllm:nixl_num_kv_expired_reqs_total{engine=\"0\"} "
        f"{expired_requests}.0",
    ]
    return "\n".join(lines)


def _prefill_metrics(**overrides: object) -> str:
    values: dict[str, object] = {
        "compute": 572,
        "local": 1248,
        "external": 0,
        "transfers": 10,
        "transfer_seconds": 2.0,
        "transferred_bytes": 20 * MIB,
        "descriptors": 10,
    }
    values.update(overrides)
    return _metrics(**values)  # type: ignore[arg-type]


def _decode_metrics(**overrides: object) -> str:
    values: dict[str, object] = {
        "compute": 0,
        "local": 1376,
        "external": 444,
        "transfers": 0,
        "transfer_seconds": 0.0,
        "transferred_bytes": 0,
        "descriptors": 0,
    }
    values.update(overrides)
    return _metrics(**values)  # type: ignore[arg-type]


def _effective_config(*, emulation: str = "n", cross_layers: str = "True") -> str:
    return (
        f"UCX_PROTO_EMULATION_ENABLE={emulation}\n"
        f"ENABLE_CROSS_LAYERS_BLOCKS={cross_layers}\n"
    )


def _validate(
    result: dict[str, object] | None = None,
    *,
    prefill_metrics: str | None = None,
    decode_metrics: str | None = None,
    effective_config: str | None = None,
    service_logs: tuple[str, ...] = (),
) -> dict[str, object]:
    return validate_pd_multiturn_load_reuse(
        result or _result(),
        prefill_metrics=prefill_metrics or _prefill_metrics(),
        decode_metrics=decode_metrics or _decode_metrics(),
        effective_config=(
            _effective_config()
            if effective_config is None
            else effective_config
        ),
        service_logs=service_logs,
    )


def test_validate_pd_load_reuse_checks_five_turn_conservation_and_nixl() -> None:
    evidence = _validate(service_logs=("healthy", "also healthy"))

    assert evidence["status"] == STATUS
    assert evidence["request_count"] == 10
    assert evidence["transition_count"] == 8
    assert evidence["total_prompt_tokens"] == 1820
    assert evidence["expected_prefill_local_cache_hit_tokens"] == 1248
    assert evidence["expected_decode_local_cache_hit_tokens"] == 1376
    assert evidence["decode_derived_hit_tokens"] == 128
    assert evidence["prefill_prompt_tokens_by_source"] == {
        "local_compute": 572,
        "local_cache_hit": 1248,
        "external_kv_transfer": 0,
    }
    assert evidence["decode_prompt_tokens_by_source"] == {
        "local_compute": 0,
        "local_cache_hit": 1376,
        "external_kv_transfer": 444,
    }
    nixl = evidence["nixl"]
    assert nixl["total"]["transfer_count"] == 10
    assert nixl["total"]["descriptors"] == 10
    assert nixl["total"]["transferred_mib"] == 20.0
    assert nixl["total"]["aggregate_throughput_mib_s"] == 10.0
    assert evidence["ucx"]["software_emulation_disabled"] is True
    assert evidence["service_logs_checked"] == 2


def test_parse_nixl_metrics_rejects_histogram_count_disagreement() -> None:
    with pytest.raises(ValueError, match="histogram counts disagree"):
        parse_nixl_metrics(_prefill_metrics(bytes_count=9))


def test_validate_rejects_nonadjacent_or_incomplete_transition_evidence() -> None:
    result = _result()
    transitions = result["cache_validation"]["transitions"]
    transitions[0]["to_round"] = 3

    with pytest.raises(ValueError, match="not an adjacent"):
        _validate(result)


def test_validate_rejects_transition_previous_prompt_mismatch() -> None:
    result = _result()
    result["cache_validation"]["transitions"][0]["previous_prompt_tokens"] = 99

    with pytest.raises(ValueError, match="previous prompt mismatch"):
        _validate(result)


def test_validate_rejects_nonnull_pd_actual_cached_tokens() -> None:
    result = _result()
    result["cache_validation"]["transitions"][0]["actual_cached_tokens"] = 112

    with pytest.raises(ValueError, match="must be null"):
        _validate(result)


def test_validate_rejects_prefill_cache_hit_mismatch() -> None:
    metrics = _prefill_metrics(compute=573, local=1247)

    with pytest.raises(ValueError, match="Prefill local cache hit mismatch"):
        _validate(prefill_metrics=metrics)


def test_validate_rejects_decode_local_compute() -> None:
    metrics = _decode_metrics(compute=1, external=443)

    with pytest.raises(ValueError, match="unexpected local compute"):
        _validate(decode_metrics=metrics)


def test_validate_rejects_push_transfer_count_mismatch() -> None:
    metrics = _prefill_metrics(
        transfers=9,
        transfer_seconds=1.8,
        transferred_bytes=18 * MIB,
        descriptors=9,
    )

    with pytest.raises(ValueError, match="push-mode NIXL transfer count"):
        _validate(prefill_metrics=metrics)


def test_validate_rejects_cross_layer_descriptor_mismatch() -> None:
    metrics = _prefill_metrics(descriptors=20)

    with pytest.raises(ValueError, match="cross-layer NIXL descriptor"):
        _validate(prefill_metrics=metrics)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"failed_transfers": 1}, "failed NIXL transfers"),
        ({"failed_notifications": 1}, "failed NIXL notifications"),
        ({"expired_requests": 1}, "expired NIXL requests"),
    ],
)
def test_validate_rejects_nixl_failure_counters(
    overrides: dict[str, int],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _validate(prefill_metrics=_prefill_metrics(**overrides))


@pytest.mark.parametrize("value", ["", "y", "true", "1"])
def test_validate_requires_explicitly_disabled_ucx_emulation(value: str) -> None:
    config = _effective_config(emulation=value)

    with pytest.raises(ValueError, match="UCX|invalid"):
        _validate(effective_config=config)


def test_validate_rejects_nixl_errors_in_optional_service_logs() -> None:
    with pytest.raises(ValueError, match="service logs"):
        _validate(service_logs=("NIXL transfer failure request=x",))


def test_validate_does_not_mutate_result() -> None:
    result = _result()
    original = deepcopy(result)

    _validate(result)

    assert result == original


def test_main_writes_audit_evidence_back_into_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_path = tmp_path / "result.json"
    prefill_path = tmp_path / "prefill.prom"
    decode_path = tmp_path / "decode.prom"
    config_path = tmp_path / "effective_config.env"
    result_path.write_text(json.dumps(_result()), encoding="utf-8")
    prefill_path.write_text(_prefill_metrics(), encoding="utf-8")
    decode_path.write_text(_decode_metrics(), encoding="utf-8")
    config_path.write_text(_effective_config(), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pd_multiturn_load_reuse_metrics.py",
            "--result",
            str(result_path),
            "--prefill-metrics",
            str(prefill_path),
            "--decode-metrics",
            str(decode_path),
            "--effective-config",
            str(config_path),
        ],
    )

    main()

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["pd_reuse_validation"]["status"] == STATUS
    assert result["cache_validation"]["status"] == STATUS
    assert result["validity"] == {
        "status": "passed",
        "load_gate": "passed",
        "cache_gate": STATUS,
    }
