from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.multi_turn.compare_pap_pd_multiturn_load import (
    aggregate_repetitions,
    compare_aggregates,
    compare_three_aggregates,
    main,
    render_markdown,
    render_three_lane_markdown,
    validate_repetition,
)


def _fingerprint(value: dict[str, object]) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def make_repetition(
    *,
    architecture: str = "pap",
    repetition: int = 1,
    metric_scale: float = 1.0,
    dirty: bool = False,
    hardware: str = "NVIDIA-L20x2-same-node",
    pd_mode: str = "oneway",
) -> dict[str, object]:
    profile: dict[str, object] = {
        "profile_id": "qwen3_8b_chat_16k_5round_o256_c2_q2_v1",
        "rounds": 5,
        "active_conversations": 2,
        "request_rate_per_round": 2.0,
        "output_tokens_per_round": 256,
    }
    requests: list[dict[str, object]] = []
    for conversation_index in range(2):
        for round_index in range(1, 6):
            ttft_ms = metric_scale * (
                round_index * 100.0 + conversation_index * 10.0 + repetition
            )
            tpot_ms = metric_scale * (
                round_index * 10.0 + conversation_index + repetition
            )
            latency_ms = ttft_ms + tpot_ms * 255
            requests.append(
                {
                    "conversation_index": conversation_index,
                    "round": round_index,
                    "prompt_tokens": (
                        16000 + round_index * 400 + conversation_index * 10
                    ),
                    "completion_tokens": 256,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "latency_ms": latency_ms,
                    "eof_latency_ms": latency_ms + 2.0,
                    "prompt_token_digest": _digest(
                        f"prompt:{conversation_index}:{round_index}"
                    ),
                    "output_token_digest": _digest(
                        f"output:{conversation_index}:{round_index}"
                    ),
                    "assistant_text_digest": _digest(
                        f"text:{conversation_index}:{round_index}"
                    ),
                }
            )
    return {
        "schema_version": 1,
        "metric_definition": "last_output_token_v2",
        "architecture": architecture,
        "profile": profile,
        "profile_fingerprint": _fingerprint(profile),
        "hardware_signature": hardware,
        "git_commit": "a" * 40,
        "git_tracked_worktree_dirty": dirty,
        "repetition_id": f"{architecture}-{pd_mode}-rep-{repetition}",
        "implementation": {
            "offload_exec_transport": (
                f"nixl-{pd_mode}" if architecture == "pd" else "local_fast"
            )
        },
        "validity": {"status": "passed"},
        "external_validation": {
            "status": "passed",
            "gates": {"correctness_logs": "passed"},
        },
        "cache_validation": {"status": "passed", "transitions": []},
        "rounds": [
            {"round": round_index, "request_count": 2}
            for round_index in range(1, 6)
        ],
        "requests": requests,
    }


def _request(
    result: dict[str, object],
    conversation_index: int,
    round_index: int,
) -> dict[str, object]:
    requests = result["requests"]
    assert isinstance(requests, list)
    return next(
        request
        for request in requests
        if request["conversation_index"] == conversation_index
        and request["round"] == round_index
    )


def test_validate_repetition_accepts_complete_request_level_result() -> None:
    validate_repetition(make_repetition())


def test_validate_repetition_accepts_finalized_pd_cache_status() -> None:
    result = make_repetition(architecture="pd")
    result["cache_validation"]["status"] = (
        "pd_multiturn_load_reuse_metrics_passed"
    )

    validate_repetition(result)


def test_validate_repetition_checks_nested_round_request_copy() -> None:
    result = make_repetition()
    requests = result["requests"]
    assert isinstance(requests, list)
    rounds = result["rounds"]
    assert isinstance(rounds, list)
    for round_result in rounds:
        round_result["requests"] = [
            deepcopy(request)
            for request in requests
            if request["round"] == round_result["round"]
        ]
    validate_repetition(result)

    round_request = rounds[2]["requests"][0]
    round_request["prompt_tokens"] += 1
    with pytest.raises(ValueError, match="nested request summary mismatch"):
        validate_repetition(result)


@pytest.mark.parametrize("gate", ["validity", "external_validation"])
def test_validate_repetition_rejects_failed_validity_gate(gate: str) -> None:
    result = make_repetition()
    result[gate] = {"status": "failed"}

    with pytest.raises(ValueError, match="did not pass"):
        validate_repetition(result)


def test_validate_repetition_rejects_non_256_completion() -> None:
    result = make_repetition()
    _request(result, 0, 3)["completion_tokens"] = 255

    with pytest.raises(ValueError, match="must be 256"):
        validate_repetition(result)


def test_validate_repetition_rejects_incomplete_conversation() -> None:
    result = make_repetition()
    requests = result["requests"]
    assert isinstance(requests, list)
    requests.pop()

    with pytest.raises(ValueError, match="round set mismatch"):
        validate_repetition(result)


def test_validate_repetition_rejects_tpot_accounting_error() -> None:
    result = make_repetition()
    _request(result, 0, 2)["tpot_ms"] = 1.0

    with pytest.raises(ValueError, match="TPOT accounting"):
        validate_repetition(result)


def test_quick_aggregate_pools_request_metrics_by_round_and_steady() -> None:
    aggregate = aggregate_repetitions([make_repetition()])

    assert aggregate["mode"] == "quick"
    assert aggregate["repetition_count"] == 1
    assert aggregate["metrics"]["round_1"]["ttft_ms"] == {
        "count": 2,
        "median": 106.0,
        "p90": 111.0,
        "max": 111.0,
    }
    assert aggregate["metrics"]["steady_rounds_2_5"]["ttft_ms"] == {
        "count": 8,
        "median": 356.0,
        "p90": 511.0,
        "max": 511.0,
    }


def test_formal_aggregate_pools_all_three_repetitions() -> None:
    aggregate = aggregate_repetitions(
        [make_repetition(repetition=index) for index in range(1, 4)]
    )

    assert aggregate["mode"] == "formal"
    assert aggregate["repetition_count"] == 3
    assert aggregate["metrics"]["round_1"]["ttft_ms"]["count"] == 6
    assert (
        aggregate["metrics"]["steady_rounds_2_5"]["tpot_ms"]["count"]
        == 24
    )
    assert len(aggregate["per_repetition_metrics"]) == 3


def test_formal_aggregate_rejects_dirty_repetition() -> None:
    results = [make_repetition(repetition=index) for index in range(1, 4)]
    results[1]["git_tracked_worktree_dirty"] = True

    with pytest.raises(ValueError, match="dirty tracked worktree"):
        aggregate_repetitions(results)


def test_aggregate_rejects_mixed_profile_fingerprint() -> None:
    results = [make_repetition(repetition=index) for index in range(1, 4)]
    results[1]["profile_fingerprint"] = "0" * 64

    with pytest.raises(ValueError, match="profile fingerprint"):
        aggregate_repetitions(results)


def test_aggregate_rejects_mixed_hardware() -> None:
    results = [make_repetition(repetition=index) for index in range(1, 4)]
    results[1]["hardware_signature"] = "different-hardware"

    with pytest.raises(ValueError, match="mixed hardware"):
        aggregate_repetitions(results)


def test_aggregate_fails_closed_on_prompt_token_shape_divergence() -> None:
    results = [make_repetition(repetition=index) for index in range(1, 4)]
    _request(results[1], 1, 4)["prompt_tokens"] = 9999

    with pytest.raises(ValueError, match="prompt token shape mismatch"):
        aggregate_repetitions(results)


def test_aggregate_reports_digest_variance_as_warning() -> None:
    results = [make_repetition(repetition=index) for index in range(1, 4)]
    _request(results[1], 0, 2)["output_token_digest"] = _digest("different")

    aggregate = aggregate_repetitions(results)

    assert any(
        "output_token_digest differs across repetitions" in warning
        for warning in aggregate["warnings"]
    )


def test_compare_calculates_pd_pap_ratios_for_rounds_and_steady() -> None:
    pd_aggregate = aggregate_repetitions(
        [make_repetition(architecture="pd")]
    )
    pap_aggregate = aggregate_repetitions(
        [make_repetition(architecture="pap", metric_scale=2.0)]
    )

    comparison = compare_aggregates(pd_aggregate, pap_aggregate)

    assert comparison["shape_parity"]["status"] == "passed"
    assert (
        comparison["metrics"]["round_3"]["tpot_ms"]["pap_over_pd"][
            "median"
        ]
        == 2.0
    )
    assert (
        comparison["metrics"]["steady_rounds_2_5"]["latency_ms"][
            "pap_over_pd"
        ]["p90"]
        == pytest.approx(2.0)
    )


def test_compare_fails_closed_on_cross_architecture_prompt_shape() -> None:
    pd_result = make_repetition(architecture="pd")
    pap_result = make_repetition(architecture="pap")
    _request(pap_result, 1, 5)["prompt_tokens"] += 1
    pd_aggregate = aggregate_repetitions([pd_result])
    pap_aggregate = aggregate_repetitions([pap_result])

    with pytest.raises(ValueError, match="prompt token shape mismatch"):
        compare_aggregates(pd_aggregate, pap_aggregate)


def test_compare_reports_digest_difference_without_invalidating() -> None:
    pd_result = make_repetition(architecture="pd")
    pap_result = make_repetition(architecture="pap")
    _request(pap_result, 1, 5)["prompt_token_digest"] = _digest("different")

    comparison = compare_aggregates(
        aggregate_repetitions([pd_result]),
        aggregate_repetitions([pap_result]),
    )

    assert comparison["status"] == "valid"
    assert comparison["digest_checks"]["prompt_token_digest"]["status"] == (
        "warning"
    )
    assert any(
        "prompt_token_digest differs between PD and PAP" in warning
        for warning in comparison["warnings"]
    )


def test_compare_rejects_profile_hardware_or_mode_mismatch() -> None:
    pd_aggregate = aggregate_repetitions(
        [make_repetition(architecture="pd")]
    )
    pap_result = make_repetition(architecture="pap")
    pap_result["hardware_signature"] = "different-hardware"
    pap_aggregate = aggregate_repetitions([pap_result])

    with pytest.raises(ValueError, match="hardware signature mismatch"):
        compare_aggregates(pd_aggregate, pap_aggregate)

    formal_pap = aggregate_repetitions(
        [
            make_repetition(architecture="pap", repetition=index)
            for index in range(1, 4)
        ]
    )
    with pytest.raises(ValueError, match="repetition mode mismatch"):
        compare_aggregates(pd_aggregate, formal_pap)


def test_compare_revalidates_aggregate_request_statistics() -> None:
    pd_aggregate = aggregate_repetitions(
        [make_repetition(architecture="pd")]
    )
    pap_aggregate = aggregate_repetitions(
        [make_repetition(architecture="pap")]
    )
    pd_aggregate["metrics"]["round_1"]["ttft_ms"]["median"] = 1.0

    with pytest.raises(ValueError, match="invalid PD aggregate.*median mismatch"):
        compare_aggregates(pd_aggregate, pap_aggregate)


def test_markdown_contains_all_summary_statistics_and_warnings() -> None:
    pd_result = make_repetition(architecture="pd")
    pap_result = make_repetition(architecture="pap")
    _request(pap_result, 0, 1)["assistant_text_digest"] = _digest("different")
    comparison = compare_aggregates(
        aggregate_repetitions([pd_result]),
        aggregate_repetitions([pap_result]),
    )

    markdown = render_markdown(comparison)

    assert "R2-R5 steady" in markdown
    assert "| R1 | TTFT | median |" in markdown
    assert "| R5 | Latency | max |" in markdown
    assert "assistant_text_digest differs between PD and PAP" in markdown


def test_compare_three_builds_absolute_matrix_and_requested_ratios() -> None:
    oneway = aggregate_repetitions(
        [make_repetition(architecture="pd", pd_mode="oneway")]
    )
    twoway = aggregate_repetitions(
        [
            make_repetition(
                architecture="pd",
                pd_mode="twoway",
                metric_scale=0.8,
            )
        ]
    )
    pap = aggregate_repetitions(
        [make_repetition(architecture="pap", metric_scale=1.2)]
    )

    matrix = compare_three_aggregates(oneway, twoway, pap)

    assert matrix["status"] == "valid"
    assert matrix["ratios"]["pd_twoway_over_pd_oneway"]["round_1"][
        "ttft_ms"
    ]["median"] == pytest.approx(0.8)
    assert matrix["ratios"]["pap_over_pd_oneway"]["round_1"]["tpot_ms"][
        "median"
    ] == pytest.approx(1.2)
    assert matrix["ratios"]["pap_over_pd_twoway"][
        "steady_rounds_2_5"
    ]["latency_ms"]["p90"] == pytest.approx(1.5)
    assert matrix["metrics"]["round_1"]["ttft_ms"]["pd_oneway"][
        "median"
    ] == 106.0

    markdown = render_three_lane_markdown(matrix)
    assert "# PD-oneway / PD-twoway / PAP" in markdown
    assert "PAP/PD-twoway" in markdown


def test_compare_three_rejects_lane_identity_and_prompt_digest_mismatch() -> None:
    oneway = aggregate_repetitions(
        [make_repetition(architecture="pd", pd_mode="oneway")]
    )
    wrong_twoway = aggregate_repetitions(
        [make_repetition(architecture="pd", pd_mode="oneway", repetition=2)]
    )
    pap_result = make_repetition(architecture="pap")
    pap = aggregate_repetitions([pap_result])

    with pytest.raises(ValueError, match="PD-twoway.*transport"):
        compare_three_aggregates(oneway, wrong_twoway, pap)

    twoway = aggregate_repetitions(
        [make_repetition(architecture="pd", pd_mode="twoway")]
    )
    _request(pap_result, 1, 3)["prompt_token_digest"] = _digest("different")
    pap = aggregate_repetitions([pap_result])
    with pytest.raises(ValueError, match="prompt token digest mismatch"):
        compare_three_aggregates(oneway, twoway, pap)


def test_compare_three_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    aggregate_paths = {}
    for name, result in {
        "pd_oneway": make_repetition(architecture="pd", pd_mode="oneway"),
        "pd_twoway": make_repetition(architecture="pd", pd_mode="twoway"),
        "pap": make_repetition(architecture="pap"),
    }.items():
        aggregate = aggregate_repetitions([result])
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(aggregate), encoding="utf-8")
        aggregate_paths[name] = path

    output_json = tmp_path / "comparison.json"
    output_markdown = tmp_path / "report.md"
    main(
        [
            "compare-three",
            "--pd-oneway",
            str(aggregate_paths["pd_oneway"]),
            "--pd-twoway",
            str(aggregate_paths["pd_twoway"]),
            "--pap",
            str(aggregate_paths["pap"]),
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ]
    )

    assert json.loads(output_json.read_text())["status"] == "valid"
    assert output_markdown.read_text().startswith(
        "# PD-oneway / PD-twoway / PAP"
    )


def test_cli_writes_aggregate_comparison_json_and_markdown(
    tmp_path: Path,
) -> None:
    pd_result_path = tmp_path / "pd_result.json"
    pap_result_path = tmp_path / "pap_result.json"
    pd_aggregate_path = tmp_path / "pd_aggregate.json"
    pap_aggregate_path = tmp_path / "pap_aggregate.json"
    comparison_path = tmp_path / "comparison.json"
    markdown_path = tmp_path / "comparison.md"
    pd_result_path.write_text(
        json.dumps(make_repetition(architecture="pd")),
        encoding="utf-8",
    )
    pap_result_path.write_text(
        json.dumps(make_repetition(architecture="pap")),
        encoding="utf-8",
    )

    main(
        [
            "aggregate",
            str(pd_result_path),
            "--output",
            str(pd_aggregate_path),
        ]
    )
    main(
        [
            "aggregate",
            "--result",
            str(pap_result_path),
            "--output",
            str(pap_aggregate_path),
        ]
    )
    main(
        [
            "compare",
            "--pd",
            str(pd_aggregate_path),
            "--pap",
            str(pap_aggregate_path),
            "--output-json",
            str(comparison_path),
            "--output-markdown",
            str(markdown_path),
        ]
    )

    assert json.loads(pd_aggregate_path.read_text())["mode"] == "quick"
    assert json.loads(comparison_path.read_text())["status"] == "valid"
    assert markdown_path.read_text().startswith(
        "# PAP/PD Multi-turn Load Comparison"
    )


def test_aggregate_requires_one_or_three_distinct_repetitions() -> None:
    result = make_repetition()
    with pytest.raises(ValueError, match="one quick or three formal"):
        aggregate_repetitions([result, deepcopy(result)])

    with pytest.raises(ValueError, match="distinct repetitions"):
        aggregate_repetitions([result, deepcopy(result), deepcopy(result)])
