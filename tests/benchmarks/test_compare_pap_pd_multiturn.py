from __future__ import annotations

import hashlib
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
    validate_reference_pair,
    validate_repetition,
    write_reference_atomic,
)
from benchmarks.multi_turn.pap_pd_multiturn_client import profile_fingerprint


def _digest(character: str) -> str:
    return character * 64


def make_repetition(
    *,
    architecture: str = "pap",
    round_1_ttft: float = 6500.0,
    round_1_tpot: float = 55.0,
    round_2_ttft: float = 360.0,
    round_2_tpot: float = 54.0,
    fingerprint: str | None = None,
    hardware: str = "NVIDIA-L20x2",
    round_1_output_digest: str | None = None,
    round_2_output_digest: str | None = None,
    git_commit: str = "a" * 40,
    git_tracked_worktree_dirty: bool = False,
    offload_exec_transport: str | None = None,
    direct_mailbox_output: bool | None = None,
    repetition_id: str = "default",
) -> dict[str, object]:
    cache_status = (
        "passed"
        if architecture == "pap"
        else "official_streaming_one_way_metrics_passed"
    )
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
            "eof_latency_ms": round_1_ttft + round_1_tpot * 255 + 2.0,
            "post_token_stream_ms": 2.0,
            "finish_reason": "length",
            "saw_done": True,
            "prompt_token_digest": _digest("a"),
            "output_token_digest": round_1_output_digest or _digest("b"),
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
            "eof_latency_ms": round_2_ttft + round_2_tpot * 255 + 2.0,
            "post_token_stream_ms": 2.0,
            "finish_reason": "length",
            "saw_done": True,
            "prompt_token_digest": _digest("d"),
            "output_token_digest": round_2_output_digest or _digest("e"),
            "assistant_text_digest": _digest("f"),
            "prefill": {},
        },
    ]
    implementation = {
        "offload_exec_transport": offload_exec_transport
        or ("local_fast" if architecture == "pap" else "nixl"),
        "direct_mailbox_output": (
            direct_mailbox_output
            if direct_mailbox_output is not None
            else architecture == "pap"
        ),
    }
    required_gates = (
        {
            "session_drain": "passed",
            "routing": "passed",
            "correctness_logs": "passed",
            "attention_stats_capture": "passed",
        }
        if architecture == "pap"
        else {
            "pd_reuse_metrics": "passed",
            "correctness_logs": "passed",
        }
    )
    required_artifacts = (
        {
            "session_drain",
            "routing",
            "correctness_logs",
            "attention_stats",
            "run_metadata",
            "tracked_worktree_patch",
            "tracked_index_patch",
        }
        if architecture == "pap"
        else {
            "proxy_log",
            "prefill_metrics",
            "decode_metrics",
            "effective_config",
            "correctness_logs",
            "tracked_worktree_patch",
            "tracked_index_patch",
        }
    )
    profile = {
        "profile_id": "qwen3_8b_chat_16k_2turn_o256_c1_v1",
        "output_tokens_per_round": 256,
    }
    return {
        "schema_version": 2,
        "metric_definition": "last_output_token_v2",
        "profile": profile,
        "profile_fingerprint": fingerprint or profile_fingerprint(profile),
        "architecture": architecture,
        "topology": topology,
        "hardware_signature": hardware,
        "git_commit": git_commit,
        "git_tracked_worktree_dirty": git_tracked_worktree_dirty,
        "implementation": implementation,
        "implementation_fingerprint": profile_fingerprint(implementation),
        "conversation_id_digest": hashlib.sha256(
            (
                f"{architecture}:{round_1_ttft}:{round_2_ttft}:"
                f"{round_2_tpot}:{repetition_id}"
            ).encode()
        ).hexdigest(),
        "rounds": rounds,
        "conversation_latency_ms": (
            float(rounds[0]["eof_latency_ms"])
            + float(rounds[1]["latency_ms"])
        ),
        "conversation_eof_latency_ms": sum(
            float(round_result["eof_latency_ms"]) for round_result in rounds
        ),
        "cache_validation": {
            "status": cache_status,
            "decode_derived_hit_tokens": 240,
        },
        "validity": {"status": "passed", "cache_gate": cache_status},
        "external_validation": {
            "status": "passed",
            "gates": required_gates,
            "artifacts": {
                name: {
                    "path": f"{name}.artifact",
                    "sha256": "0" * 64,
                }
                for name in required_artifacts
            },
        },
    }


def test_validate_repetition_accepts_complete_result() -> None:
    validate_repetition(make_repetition())


def test_validate_repetition_accepts_official_streaming_pd_metrics() -> None:
    validate_repetition(make_repetition(architecture="pd"))


def test_validate_repetition_rejects_missing_external_artifact() -> None:
    result = make_repetition()
    result["external_validation"]["artifacts"].pop("routing")

    with pytest.raises(ValueError, match="missing artifacts"):
        validate_repetition(result)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("rounds", 0, "completion_tokens"), 255, "completion"),
        (("rounds", 0, "finish_reason"), "stop", "finish"),
        (("rounds", 0, "saw_done"), False, "DONE"),
        (("rounds", 0, "tpot_ms"), 1.0, "TPOT accounting"),
        (("validity", "status"), "failed", "validity"),
        (("cache_validation", "status"), "failed", "cache"),
        (("conversation_latency_ms",), 1.0, "conversation"),
        (("external_validation", "status"), "failed", "external"),
        (
            ("external_validation", "gates", "session_drain"),
            "failed",
            "session_drain",
        ),
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
    assert aggregate["correctness_signatures"]["round_2"] == {
        "prompt_token_digest": _digest("d"),
        "output_token_digest": _digest("e"),
        "assistant_text_digest": _digest("f"),
    }


def test_aggregate_rejects_output_digest_variance() -> None:
    with pytest.raises(ValueError, match="output token digest"):
        aggregate_repetitions(
            [
                make_repetition(
                    round_2_output_digest=_digest("e"), repetition_id="1"
                ),
                make_repetition(
                    round_2_output_digest=_digest("1"), repetition_id="2"
                ),
                make_repetition(
                    round_2_output_digest=_digest("e"), repetition_id="3"
                ),
            ]
        )


def test_aggregate_rejects_mixed_implementation() -> None:
    with pytest.raises(ValueError, match="implementation"):
        aggregate_repetitions(
            [
                make_repetition(repetition_id="1"),
                make_repetition(
                    offload_exec_transport="nixl_mailbox",
                    repetition_id="2",
                ),
                make_repetition(repetition_id="3"),
            ]
        )


def test_formal_aggregate_rejects_dirty_tracked_worktree() -> None:
    with pytest.raises(ValueError, match="dirty"):
        aggregate_repetitions(
            [make_repetition(git_tracked_worktree_dirty=True) for _ in range(3)]
        )


def test_reference_rejects_forged_dirty_formal_aggregate() -> None:
    aggregate = aggregate_repetitions(
        [make_repetition(repetition_id=str(index)) for index in range(3)]
    )
    aggregate["git_tracked_worktree_dirty"] = True

    with pytest.raises(ValueError, match="dirty"):
        make_reference(aggregate, architecture="pap")


def test_reference_rejects_forged_raw_metric_median() -> None:
    aggregate = aggregate_repetitions(
        [make_repetition(repetition_id=str(index)) for index in range(3)]
    )
    aggregate["source_results"] = [
        f"runs/formal/rep{index}/result.json" for index in range(1, 4)
    ]
    aggregate["raw_metrics"]["round_2"]["tpot_ms"] = [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match="median"):
        make_reference(aggregate, architecture="pap")


def test_reference_rejects_cross_column_tpot_forgery() -> None:
    aggregate = aggregate_repetitions(
        [
            make_repetition(
                round_2_tpot=54.0,
                repetition_id=str(index),
            )
            for index in range(3)
        ]
    )
    aggregate["source_results"] = [
        f"runs/formal/rep{index}/result.json" for index in range(1, 4)
    ]
    aggregate["raw_metrics"]["round_2"]["tpot_ms"] = [1.0, 1.0, 1.0]
    aggregate["metrics"]["round_2"]["tpot_ms"] = 1.0

    with pytest.raises(ValueError, match="TPOT accounting"):
        make_reference(aggregate, architecture="pap")


def test_aggregate_one_repetition_is_diagnostic() -> None:
    aggregate = aggregate_repetitions([make_repetition()])

    assert aggregate["mode"] == "quick"
    assert aggregate["repetition_count"] == 1


def test_aggregate_rejects_duplicate_repetition_identity() -> None:
    result = make_repetition()

    with pytest.raises(ValueError, match="distinct repetitions"):
        aggregate_repetitions([result, deepcopy(result), deepcopy(result)])


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
    aggregate["source_results"] = [
        f"runs/formal/rep{index}/result.json" for index in range(1, 4)
    ]
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
    assert comparison["correctness"]["round_2"] == {
        "prompt_digest_match": True,
        "candidate_output_matches_pd": True,
        "candidate_output_matches_pap_reference": True,
        "pd_output_matches_pap_reference": True,
        "candidate_text_matches_pd": True,
        "candidate_text_matches_pap_reference": True,
        "pd_text_matches_pap_reference": True,
    }


def test_compare_reports_cross_architecture_output_difference() -> None:
    pd_reference = _formal_reference("pd", round_2_tpot=25.0, round_2_ttft=340.0)
    pap_reference = _formal_reference(
        "pap", round_2_tpot=55.0, round_2_ttft=360.0
    )
    pd_reference["correctness_signatures"]["round_2"][
        "output_token_digest"
    ] = _digest("0")

    comparison = compare_candidate(
        aggregate_repetitions([make_repetition()]),
        pd_reference,
        pap_reference,
    )

    assert comparison["correctness"]["round_2"] == {
        "prompt_digest_match": True,
        "candidate_output_matches_pd": False,
        "candidate_output_matches_pap_reference": True,
        "pd_output_matches_pap_reference": False,
        "candidate_text_matches_pd": True,
        "candidate_text_matches_pap_reference": True,
        "pd_text_matches_pap_reference": True,
    }
    assert any("PD output digest" in warning for warning in comparison["warnings"])


@pytest.mark.parametrize(
    "digest_name",
    ["output_token_digest", "assistant_text_digest"],
)
def test_compare_rejects_candidate_correctness_drift(digest_name: str) -> None:
    candidate = aggregate_repetitions([make_repetition()])
    candidate["correctness_signatures"]["round_2"][digest_name] = _digest("1")

    with pytest.raises(ValueError, match="PAP reference"):
        compare_candidate(
            candidate,
            _formal_reference("pd", round_2_tpot=25.0, round_2_ttft=340.0),
            _formal_reference("pap", round_2_tpot=55.0, round_2_ttft=360.0),
        )


def test_quick_candidate_is_always_diagnostic() -> None:
    comparison = compare_candidate(
        aggregate_repetitions([make_repetition(round_2_tpot=1.0)]),
        _formal_reference("pd", round_2_tpot=25.0, round_2_ttft=340.0),
        _formal_reference("pap", round_2_tpot=55.0, round_2_ttft=360.0),
    )

    assert comparison["classification"] == "diagnostic"


def test_quick_dirty_candidate_is_explicitly_warned() -> None:
    comparison = compare_candidate(
        aggregate_repetitions(
            [make_repetition(git_tracked_worktree_dirty=True)]
        ),
        _formal_reference("pd", round_2_tpot=25.0, round_2_ttft=340.0),
        _formal_reference("pap", round_2_tpot=55.0, round_2_ttft=360.0),
    )

    assert any("dirty tracked" in warning for warning in comparison["warnings"])


def test_compare_rejects_hardware_mismatch() -> None:
    candidate = aggregate_repetitions(
        [
            make_repetition(hardware="different", repetition_id=str(index))
            for index in range(3)
        ]
    )

    with pytest.raises(ValueError, match="hardware"):
        compare_candidate(
            candidate,
            _formal_reference("pd", round_2_tpot=25.0, round_2_ttft=340.0),
            _formal_reference("pap", round_2_tpot=55.0, round_2_ttft=360.0),
        )


def test_compare_rejects_legacy_aggregate_schema() -> None:
    candidate = aggregate_repetitions([make_repetition()])
    candidate["schema_version"] = 1

    with pytest.raises(ValueError, match="schema"):
        compare_candidate(
            candidate,
            _formal_reference("pd", round_2_tpot=25.0, round_2_ttft=340.0),
            _formal_reference("pap", round_2_tpot=55.0, round_2_ttft=360.0),
        )


def test_compare_rejects_quick_payload_used_as_reference() -> None:
    quick_reference = aggregate_repetitions([make_repetition(architecture="pd")])

    with pytest.raises(ValueError, match="reference"):
        compare_candidate(
            aggregate_repetitions([make_repetition()]),
            quick_reference,
            _formal_reference("pap", round_2_tpot=55.0, round_2_ttft=360.0),
        )


def test_validate_reference_pair_rejects_mismatched_profile() -> None:
    pd_reference = _formal_reference(
        "pd",
        round_2_tpot=25.0,
        round_2_ttft=340.0,
    )
    pap_reference = _formal_reference(
        "pap",
        round_2_tpot=55.0,
        round_2_ttft=360.0,
    )
    pap_reference["profile"]["variant"] = "different"
    pap_reference["profile_fingerprint"] = profile_fingerprint(
        pap_reference["profile"]
    )

    with pytest.raises(ValueError, match="profile fingerprint mismatch"):
        validate_reference_pair(pd_reference, pap_reference)


def test_validate_reference_pair_rejects_mismatched_hardware() -> None:
    pd_reference = _formal_reference(
        "pd",
        round_2_tpot=25.0,
        round_2_ttft=340.0,
    )
    pap_reference = _formal_reference(
        "pap",
        round_2_tpot=55.0,
        round_2_ttft=360.0,
    )
    pap_reference["hardware_signature"] = "different-hardware"

    with pytest.raises(ValueError, match="hardware signature mismatch"):
        validate_reference_pair(pd_reference, pap_reference)


def test_validate_reference_pair_rejects_missing_hardware_on_both_sides() -> None:
    pd_reference = _formal_reference(
        "pd",
        round_2_tpot=25.0,
        round_2_ttft=340.0,
    )
    pap_reference = _formal_reference(
        "pap",
        round_2_tpot=55.0,
        round_2_ttft=360.0,
    )
    del pd_reference["hardware_signature"]
    del pap_reference["hardware_signature"]

    with pytest.raises(ValueError, match="hardware signature"):
        validate_reference_pair(pd_reference, pap_reference)


def test_validate_reference_pair_rejects_wrong_profile_on_both_sides() -> None:
    pd_reference = _formal_reference(
        "pd",
        round_2_tpot=25.0,
        round_2_ttft=340.0,
    )
    pap_reference = _formal_reference(
        "pap",
        round_2_tpot=55.0,
        round_2_ttft=360.0,
    )
    for reference in (pd_reference, pap_reference):
        reference["profile"]["profile_id"] = "wrong-profile"
        reference["profile_fingerprint"] = profile_fingerprint(
            reference["profile"]
        )

    with pytest.raises(ValueError, match="profile ID"):
        validate_reference_pair(pd_reference, pap_reference)


def test_validate_reference_pair_rejects_non_string_hardware() -> None:
    pd_reference = _formal_reference(
        "pd",
        round_2_tpot=25.0,
        round_2_ttft=340.0,
    )
    pap_reference = _formal_reference(
        "pap",
        round_2_tpot=55.0,
        round_2_ttft=360.0,
    )
    pd_reference["hardware_signature"] = ["invalid"]
    pap_reference["hardware_signature"] = ["invalid"]

    with pytest.raises(ValueError, match="hardware signature"):
        validate_reference_pair(pd_reference, pap_reference)


def test_validate_reference_pair_rejects_prompt_digest_mismatch() -> None:
    pd_reference = _formal_reference(
        "pd",
        round_2_tpot=25.0,
        round_2_ttft=340.0,
    )
    pap_reference = _formal_reference(
        "pap",
        round_2_tpot=55.0,
        round_2_ttft=360.0,
    )
    pap_reference["correctness_signatures"]["round_2"][
        "prompt_token_digest"
    ] = _digest("9")

    with pytest.raises(ValueError, match="prompt token digest mismatch"):
        validate_reference_pair(pd_reference, pap_reference)


def test_compare_reports_conversation_latency_and_implementation() -> None:
    comparison = compare_candidate(
        aggregate_repetitions([make_repetition()]),
        _formal_reference("pd", round_2_tpot=25.0, round_2_ttft=340.0),
        _formal_reference("pap", round_2_tpot=55.0, round_2_ttft=360.0),
    )

    assert "conversation_latency_ms" in comparison["metrics"]
    assert comparison["implementations"]["candidate"] == {
        "git_commit": "a" * 40,
        "git_tracked_worktree_dirty": False,
        "implementation": {
            "offload_exec_transport": "local_fast",
            "direct_mailbox_output": True,
        },
    }


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
    assert "Post-token stream tail" in report


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
    aggregate = aggregate_repetitions(
        [make_repetition(repetition_id=str(index)) for index in range(3)]
    )
    aggregate["source_results"] = [
        f"runs/formal/rep{index}/result.json" for index in range(1, 4)
    ]
    original = deepcopy(aggregate)

    make_reference(aggregate, architecture="pap")

    assert aggregate == original
