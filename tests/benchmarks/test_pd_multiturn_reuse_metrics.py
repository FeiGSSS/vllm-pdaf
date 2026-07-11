from __future__ import annotations

from copy import deepcopy

import pytest

from benchmarks.multi_turn.pd_multiturn_reuse_metrics import (
    parse_prompt_token_sources,
    validate_official_streaming_one_way,
)


def _metrics(*, compute: int, local: int, external: int) -> str:
    values = {
        "local_compute": compute,
        "local_cache_hit": local,
        "external_kv_transfer": external,
    }
    return "\n".join(
        "vllm:prompt_tokens_by_source_total{"
        f'model_name="qwen",engine="0",source="{source}"'
        f"}} {value}.0"
        for source, value in values.items()
    )


def _result() -> dict[str, object]:
    return {
        "architecture": "pd",
        "profile": {"block_size": 16},
        "rounds": [
            {"round": 1, "prompt_tokens": 16018},
            {"round": 2, "prompt_tokens": 16418},
        ],
        "cache_validation": {
            "status": "requires_official_log",
            "first_prompt_block_boundary": 16016,
            "expected_cached_tokens": 16272,
            "decode_derived_hit_tokens": 256,
        },
        "validity": {
            "status": "passed",
            "cache_gate": "requires_official_log",
        },
    }


def test_parse_prompt_token_sources_requires_one_series_per_source() -> None:
    duplicate = _metrics(compute=1, local=2, external=3) + "\n" + (
        'vllm:prompt_tokens_by_source_total{model_name="other",'
        'engine="0",source="local_compute"} 4.0'
    )

    with pytest.raises(ValueError, match="exactly one"):
        parse_prompt_token_sources(duplicate)


def test_validate_official_streaming_one_way_checks_exact_reuse() -> None:
    evidence = validate_official_streaming_one_way(
        _result(),
        proxy_log="cache MISS\ncache MISS\n",
        prefill_metrics=_metrics(compute=16420, local=16016, external=0),
        decode_metrics=_metrics(compute=4, local=16272, external=16160),
    )

    assert evidence["status"] == "official_streaming_one_way_metrics_passed"
    assert evidence["prefill_prompt_tokens_by_source"]["local_cache_hit"] == 16016
    assert evidence["decode_prompt_tokens_by_source"]["local_cache_hit"] == 16272


def test_validate_rejects_decode_hit_without_decode_derived_tokens() -> None:
    with pytest.raises(ValueError, match="Decode local cache"):
        validate_official_streaming_one_way(
            _result(),
            proxy_log="cache MISS\ncache MISS\n",
            prefill_metrics=_metrics(compute=16420, local=16016, external=0),
            decode_metrics=_metrics(compute=260, local=16016, external=16160),
        )


def test_validate_rejects_missing_second_prefill_to_decode_transfer() -> None:
    with pytest.raises(ValueError, match="second-turn Prefill-to-Decode"):
        validate_official_streaming_one_way(
            _result(),
            proxy_log="cache MISS\ncache MISS\n",
            prefill_metrics=_metrics(compute=16420, local=16016, external=0),
            decode_metrics=_metrics(compute=148, local=16272, external=16016),
        )


def test_validate_rejects_prompt_source_conservation_failure() -> None:
    with pytest.raises(ValueError, match="conservation"):
        validate_official_streaming_one_way(
            _result(),
            proxy_log="cache MISS\ncache MISS\n",
            prefill_metrics=_metrics(compute=16419, local=16016, external=0),
            decode_metrics=_metrics(compute=4, local=16272, external=16160),
        )


def test_validate_does_not_mutate_result() -> None:
    result = _result()
    original = deepcopy(result)

    validate_official_streaming_one_way(
        result,
        proxy_log="cache MISS\ncache MISS\n",
        prefill_metrics=_metrics(compute=16420, local=16016, external=0),
        decode_metrics=_metrics(compute=4, local=16272, external=16160),
    )

    assert result == original
