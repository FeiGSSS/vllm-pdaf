from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from benchmarks.multi_turn.pap_multiturn_common import (
    block_aligned_prefix_metrics,
    calculate_tpot_ms,
    consume_sse_lines,
    parse_prefill_headers,
    profile_fingerprint,
)


def test_tpot_excludes_first_token() -> None:
    assert calculate_tpot_ms(110.0, 10.0, 5) == 25.0


@pytest.mark.parametrize(
    ("latency_ms", "ttft_ms", "completion_tokens"),
    [
        (0.0, 0.0, 2),
        (10.0, 11.0, 2),
        (10.0, 1.0, 0),
        (float("inf"), 1.0, 2),
    ],
)
def test_tpot_rejects_invalid_measurements(
    latency_ms: float,
    ttft_ms: float,
    completion_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        calculate_tpot_ms(latency_ms, ttft_ms, completion_tokens)


def test_lcp_excludes_unmaterialized_final_sample() -> None:
    metrics = block_aligned_prefix_metrics(
        list(range(32)),
        list(range(100, 133)),
        [*range(32), *range(100, 132), 999],
        block_size=16,
    )

    assert metrics == {
        "materialized_history_tokens": 64,
        "committed_lcp_tokens": 64,
        "expected_cached_tokens": 64,
        "first_prompt_block_boundary": 32,
        "decode_derived_hit_tokens": 32,
    }


def test_lcp_uses_retokenized_second_prompt() -> None:
    metrics = block_aligned_prefix_metrics(
        list(range(32)),
        list(range(100, 133)),
        [*range(32), 777],
        block_size=16,
    )

    assert metrics["committed_lcp_tokens"] == 32
    assert metrics["expected_cached_tokens"] == 32
    assert metrics["decode_derived_hit_tokens"] == 0


def test_parse_prefill_headers_is_case_insensitive() -> None:
    assert parse_prefill_headers(
        {
            "x-pap-prefill-prompt-tokens": "16420",
            "X-PAP-Prefill-Cached-Tokens": "16272",
            "x-pap-prefill-computed-tokens": "148",
            "X-PAP-Prefill-Ms": "360",
        }
    ) == {
        "prompt_tokens": 16420,
        "cached_tokens": 16272,
        "computed_tokens": 148,
        "prefill_ms": 360,
    }


def test_parse_prefill_headers_rejects_inconsistent_accounting() -> None:
    with pytest.raises(ValueError, match="cached.*computed"):
        parse_prefill_headers(
            {
                "X-PAP-Prefill-Prompt-Tokens": "100",
                "X-PAP-Prefill-Cached-Tokens": "80",
                "X-PAP-Prefill-Computed-Tokens": "19",
            }
        )


def test_profile_fingerprint_is_canonical() -> None:
    left = {"model": "local", "workload": {"rounds": 5, "output": 32}}
    right = {"workload": {"output": 32, "rounds": 5}, "model": "local"}

    assert profile_fingerprint(left) == profile_fingerprint(right)
    right["workload"]["output"] = 128
    assert profile_fingerprint(left) != profile_fingerprint(right)


class RecordingLines:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.consumed: list[str] = []

    def __iter__(self) -> Iterator[str]:
        for line in self.lines:
            self.consumed.append(line)
            yield line


def _data(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}"


def test_consume_sse_lines_reads_token_ids_usage_and_eof() -> None:
    sentinel = ": after-done-sentinel"
    lines = RecordingLines(
        [
            _data(
                {
                    "prompt_token_ids": [1, 2, 3],
                    "choices": [{"delta": {"content": ""}}],
                }
            ),
            _data(
                {
                    "choices": [
                        {
                            "delta": {"content": "A"},
                            "token_ids": [7],
                            "finish_reason": None,
                        }
                    ]
                }
            ),
            _data(
                {
                    "choices": [
                        {
                            "delta": {"content": "B"},
                            "token_ids": [8],
                            "finish_reason": "length",
                        }
                    ]
                }
            ),
            _data(
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                }
            ),
            "data: [DONE]",
            sentinel,
        ]
    )
    times = iter([0.010, 0.020, 0.030])

    result = consume_sse_lines(lines, started_at=0.0, clock=lambda: next(times))

    assert lines.consumed[-1] == sentinel
    assert result.pop("post_token_stream_ms") == pytest.approx(10.0)
    assert result == {
        "prompt_token_ids": [1, 2, 3],
        "output_token_ids": [7, 8],
        "assistant_text": "AB",
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "finish_reason": "length",
        "saw_done": True,
        "ttft_ms": 10.0,
        "latency_ms": 20.0,
        "eof_latency_ms": 30.0,
        "tpot_ms": 10.0,
    }


def test_consume_sse_lines_reads_completion_text() -> None:
    lines = [
        _data(
            {
                "choices": [
                    {
                        "text": "A",
                        "prompt_token_ids": [1],
                        "token_ids": [7],
                        "finish_reason": "length",
                    }
                ],
            }
        ),
        _data(
            {
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        ),
        "data: [DONE]",
    ]
    times = iter([0.01, 0.02])

    result = consume_sse_lines(
        lines,
        started_at=0.0,
        clock=lambda: next(times),
    )

    assert result["assistant_text"] == "A"
    assert result["output_token_ids"] == [7]


@pytest.mark.parametrize(
    ("tail", "message"),
    [([], "DONE"), (["data: [DONE]"], "final usage")],
)
def test_consume_sse_lines_requires_done_and_usage(
    tail: list[str],
    message: str,
) -> None:
    lines = [
        _data(
            {
                "prompt_token_ids": [1],
                "choices": [
                    {
                        "delta": {"content": "A"},
                        "token_ids": [7],
                        "finish_reason": "length",
                    }
                ],
            }
        ),
        *tail,
    ]

    with pytest.raises(ValueError, match=message):
        consume_sse_lines(lines, started_at=0.0, clock=lambda: 0.01)
