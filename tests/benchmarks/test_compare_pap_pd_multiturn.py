from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.multi_turn.compare_pap_pd_multiturn import (
    aggregate_repetitions,
    classify_tpot,
    compare_candidate,
    make_reference,
    render_markdown,
    validate_repetition,
    write_reference_atomic,
)


def _digest(character: str) -> str:
    return character * 64


def make_repetition(
    *,
    architecture: str = "pap",
    round_1_ttft: float = 6500.0,
    round_1_tpot: float = 55.0,
    round_2_ttft: float = 360.0,
    round_2_tpot: float = 54.0,
    fingerprint: str = "profile-fingerprint",
    hardware: str = "NVIDIA-L20x2",
) -> dict[str, object]:
    cache_status = "passed" if architecture == "pap" else "official_log_passed"
    topology = (
        {
            "name": "1pa1p",
            "pa_count": 1,
            "projection_count": 1,
            "pd_prefill_count": 0,
            "pd_decode_count": 0,
        }
        if architecture == "pap"
        else {
            "name": "1p1d",
            "pa_count": 0,
            "projection_count": 0,
            "pd_prefill_count": 1,
            "pd_decode_count": 1,
        }
    )
    rounds = [
        {
            "round": 1,
            "request_id": "request-1",
            "prompt_tokens": 16022,
            "completion_tokens": 256,
            "ttft_ms": round_1_ttft,
            "tpot_ms": round_1_tpot,
            "latency_ms": round_1_ttft + round_1_tpot * 255,
            "finish_reason": "length",
            "saw_done": True,
            "prompt_token_digest": _digest("a"),
            "output_token_digest": _digest("b"),
            "assistant_text_digest": _digest("c"),
            "prefill": {},
        },
        {
            "round": 2,
            "request_id": "request-2",
            "prompt_tokens": 16420,
            "completion_tokens": 256,
            "ttft_ms": round_2_ttft,
            "tpot_ms": round_2_tpot,
            "latency_ms": round_2_ttft + round_2_tpot * 255,
            "finish_reason": "length",
            "saw_done": True,
            "prompt_token_digest": _digest("d"),
            "output_token_digest": _digest("e"),
            "assistant_text_digest": _digest("f"),
            "prefill": {},
        },
    ]
    return {
        "schema_version": 1,
        "profile": {
            "profile_id": "qwen3_8b_chat_16k_2turn_o256_c1_v1",
            "output_tokens_per_round": 256,
        },
        "profile_fingerprint": fingerprint,
        "architecture": architecture,
        "topology": topology,
        "hardware_signature": hardware,
        "rounds": rounds,
        "conversation_latency_ms": sum(
            float(round_result["latency_ms"]) for round_result in rounds
        ),
        "cache_validation": {
            "status": cache_status,
            "decode_derived_hit_tokens": 240,
        },
        "validity": {"status": "passed", "cache_gate": cache_status},
    }


def test_validate_repetition_accepts_complete_result() -> None:
    validate_repetition(make_repetition())


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("rounds", 0, "completion_tokens"), 255, "completion"),
        (("rounds", 0, "finish_reason"), "stop", "finish"),
        (("rounds", 0, "saw_done"), False, "DONE"),
        (("validity", "status"), "failed", "validity"),
        (("cache_validation", "status"), "failed", "cache"),
    ],
)
def test_validate_repetition_rejects_failed_gates(
    path: tuple[object, ...],
    value: object,
    match: str,
) -> None:
    result = make_repetition()
    target: object = result
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=match):
        validate_repetition(result)


def test_aggregate_repetitions_uses_cross_run_median() -> None:
    results = [
        make_repetition(round_2_tpot=57.0, round_2_ttft=370.0),
        make_repetition(round_2_tpot=53.0, round_2_ttft=350.0),
        make_repetition(round_2_tpot=55.0, round_2_ttft=360.0),
    ]

    aggregate = aggregate_repetitions(results)

    assert aggregate["mode"] == "formal"
    assert aggregate["repetition_count"] == 3
    assert aggregate["metrics"]["round_2"]["tpot_ms"] == 55.0
    assert aggregate["metrics"]["round_2"]["ttft_ms"] == 360.0
    assert aggregate["raw_metrics"]["round_2"]["tpot_ms"] == [57.0, 53.0, 55.0]


def test_aggregate_one_repetition_is_diagnostic() -> None:
    aggregate = aggregate_repetitions([make_repetition()])

    assert aggregate["mode"] == "quick"
    assert aggregate["repetition_count"] == 1


def test_aggregate_rejects_mixed_profile_fingerprints() -> None:
    with pytest.raises(ValueError, match="fingerprint"):
        aggregate_repetitions(
            [
                make_repetition(fingerprint="one"),
                make_repetition(fingerprint="two"),
                make_repetition(fingerprint="one"),
            ]
        )


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (96.999, "improved"),
        (97.0, "improved"),
        (97.001, "neutral"),
        (102.999, "neutral"),
        (103.0, "regressed"),
        (103.001, "regressed"),
    ],
)
def test_classify_tpot_uses_three_percent_boundary(
    candidate: float,
    expected: str,
) -> None:
    assert classify_tpot(candidate, 100.0) == expected


def _formal_reference(
    architecture: str,
    *,
    round_2_tpot: float,
    round_2_ttft: float,
) -> dict[str, object]:
    aggregate = aggregate_repetitions(
        [
            make_repetition(
                architecture=architecture,
                round_2_tpot=round_2_tpot + offset,
                round_2_ttft=round_2_ttft + offset,
            )
            for offset in (-1.0, 0.0, 1.0)
        ]
    )
    return make_reference(aggregate, architecture=architecture)


def test_compare_candidate_reports_both_references_and_target() -> None:
    pd_reference = _formal_reference("pd", round_2_tpot=25.0, round_2_ttft=340.0)
    pap_reference = _formal_reference(
        "pap", round_2_tpot=55.0, round_2_ttft=360.0
    )
    candidate = aggregate_repetitions(
        [
            make_repetition(round_2_tpot=value, round_2_ttft=355.0)
            for value in (52.0, 52.5, 53.0)
        ]
    )

    comparison = compare_candidate(candidate, pd_reference, pap_reference)

    assert comparison["classification"] == "improved"
    assert comparison["north_star_target_met"] is False
    assert comparison["metrics"]["round_2"]["tpot_ms"] == {
        "pd_reference": 25.0,
        "pap_reference": 55.0,
        "candidate": 52.5,
        "candidate_over_pd": 2.1,
        "candidate_over_pap_reference": pytest.approx(52.5 / 55.0),
    }


def test_quick_candidate_is_always_diagnostic() -> None:
    comparison = compare_candidate(
        aggregate_repetitions([make_repetition(round_2_tpot=1.0)]),
        _formal_reference("pd", round_2_tpot=25.0, round_2_ttft=340.0),
        _formal_reference("pap", round_2_tpot=55.0, round_2_ttft=360.0),
    )

    assert comparison["classification"] == "diagnostic"


def test_compare_rejects_hardware_mismatch() -> None:
    candidate = aggregate_repetitions(
        [make_repetition(hardware="different") for _ in range(3)]
    )

    with pytest.raises(ValueError, match="hardware"):
        compare_candidate(
            candidate,
            _formal_reference("pd", round_2_tpot=25.0, round_2_ttft=340.0),
            _formal_reference("pap", round_2_tpot=55.0, round_2_ttft=360.0),
        )


def test_render_markdown_contains_primary_metrics_and_verdict() -> None:
    comparison = compare_candidate(
        aggregate_repetitions([make_repetition()]),
        _formal_reference("pd", round_2_tpot=25.0, round_2_ttft=340.0),
        _formal_reference("pap", round_2_tpot=55.0, round_2_ttft=360.0),
    )

    report = render_markdown(comparison)

    assert "TTFT" in report
    assert "TPOT" in report
    assert "PAP/PD" in report
    assert "diagnostic" in report


def test_make_reference_requires_formal_valid_aggregate() -> None:
    with pytest.raises(ValueError, match="formal"):
        make_reference(
            aggregate_repetitions([make_repetition()]),
            architecture="pap",
        )


def test_write_reference_atomic_replaces_target(tmp_path: Path) -> None:
    reference = _formal_reference("pap", round_2_tpot=55.0, round_2_ttft=360.0)
    output = tmp_path / "pap_reference.json"
    output.write_text('{"old": true}\n')

    write_reference_atomic(output, reference)

    assert json.loads(output.read_text()) == reference
    assert list(tmp_path.glob("*.tmp")) == []


def test_make_reference_does_not_mutate_aggregate() -> None:
    aggregate = aggregate_repetitions([make_repetition() for _ in range(3)])
    original = deepcopy(aggregate)

    make_reference(aggregate, architecture="pap")

    assert aggregate == original
