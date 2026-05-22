# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from examples.pap.compare_outputs import (
    compare_architecture_outputs,
    normalize_completion_response,
)


def _response(text: str, completion_tokens: int = 8) -> dict:
    return {
        "id": "unstable-id",
        "created": 1,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": completion_tokens,
            "total_tokens": 8 + completion_tokens,
        },
    }


def test_normalize_completion_response_ignores_unstable_metadata() -> None:
    normalized = normalize_completion_response(_response("same"))

    assert normalized == {
        "text": "same",
        "finish_reason": "length",
        "prompt_tokens": 8,
        "completion_tokens": 8,
        "total_tokens": 16,
    }


def test_compare_architecture_outputs_reports_exact_match() -> None:
    report = compare_architecture_outputs(
        {
            "fused": _response("same"),
            "native_pd": _response("same"),
            "pap": _response("same"),
        }
    )

    assert report["match"] is True
    assert report["mismatches"] == []


def test_compare_architecture_outputs_reports_field_mismatch() -> None:
    report = compare_architecture_outputs(
        {
            "fused": _response("same"),
            "native_pd": _response("different"),
            "pap": _response("same", completion_tokens=7),
        }
    )

    assert report["match"] is False
    assert {
        "architecture": "native_pd",
        "field": "text",
        "expected": "same",
        "actual": "different",
    } in report["mismatches"]
    assert {
        "architecture": "pap",
        "field": "completion_tokens",
        "expected": 8,
        "actual": 7,
    } in report["mismatches"]
