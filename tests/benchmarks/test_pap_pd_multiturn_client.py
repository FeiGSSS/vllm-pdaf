from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from benchmarks.multi_turn.pap_pd_multiturn_client import (
    NorthStarConfig,
    block_aligned_prefix_metrics,
    build_chat_payload,
    build_workload,
    calculate_tpot_ms,
    consume_sse_lines,
    execute_two_turn,
    parse_prefill_headers,
    profile_fingerprint,
    validate_pap_cache_reuse,
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
    left = {"model": "local", "workload": {"rounds": 2, "output": 256}}
    right = {"workload": {"output": 256, "rounds": 2}, "model": "local"}

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
                    "choices": [
                        {
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
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
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
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


def test_consume_sse_lines_requires_done_and_usage() -> None:
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
        )
    ]

    with pytest.raises(ValueError, match="DONE"):
        consume_sse_lines(lines, started_at=0.0, clock=lambda: 0.01)


class FakeTokenizer:
    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        assert text == "local corpus"
        assert add_special_tokens is False
        return list(range(17000))

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return f"decoded:{token_ids[0]}-{token_ids[-1]}"


def test_build_workload_uses_fixed_nonoverlapping_corpus_slices() -> None:
    first_messages, append_text = build_workload(
        FakeTokenizer(),
        "local corpus",
        document_tokens=16000,
        append_tokens=120,
    )

    assert first_messages == [
        {
            "role": "user",
            "content": (
                "Read the following document and summarize its main argument.\n\n"
                "decoded:0-15999"
            ),
        }
    ]
    assert append_text == "decoded:16000-16119"


def test_build_chat_payload_freezes_north_star_sampling() -> None:
    messages = [{"role": "user", "content": "hello"}]

    payload = build_chat_payload(
        model="/local/model",
        messages=messages,
        conversation_id="conversation-1",
        cache_salt="salt-1",
        output_tokens=256,
    )

    assert payload == {
        "model": "/local/model",
        "messages": messages,
        "conversation_id": "conversation-1",
        "cache_salt": "salt-1",
        "max_tokens": 256,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_validate_pap_cache_reuse_requires_decode_derived_block() -> None:
    metrics = {
        "materialized_history_tokens": 64,
        "committed_lcp_tokens": 64,
        "expected_cached_tokens": 64,
        "first_prompt_block_boundary": 32,
        "decode_derived_hit_tokens": 32,
    }

    assert validate_pap_cache_reuse(
        first_prefill={
            "prompt_tokens": 32,
            "cached_tokens": 0,
            "computed_tokens": 32,
            "prefill_ms": 10,
        },
        second_prefill={
            "prompt_tokens": 65,
            "cached_tokens": 64,
            "computed_tokens": 1,
            "prefill_ms": 1,
        },
        prefix_metrics=metrics,
        block_size=16,
    ) == {"status": "passed", **metrics, "actual_cached_tokens": 64}


def test_validate_pap_cache_reuse_rejects_prompt_only_hit() -> None:
    metrics = {
        "materialized_history_tokens": 64,
        "committed_lcp_tokens": 32,
        "expected_cached_tokens": 32,
        "first_prompt_block_boundary": 32,
        "decode_derived_hit_tokens": 0,
    }

    with pytest.raises(ValueError, match="Decode-derived"):
        validate_pap_cache_reuse(
            first_prefill={
                "prompt_tokens": 32,
                "cached_tokens": 0,
                "computed_tokens": 32,
                "prefill_ms": 10,
            },
            second_prefill={
                "prompt_tokens": 65,
                "cached_tokens": 32,
                "computed_tokens": 33,
                "prefill_ms": 1,
            },
            prefix_metrics=metrics,
            block_size=16,
        )


class FakeResponse:
    def __init__(self, lines: list[str], headers: dict[str, str]) -> None:
        self._lines = lines
        self.headers = headers

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    @staticmethod
    def raise_for_status() -> None:
        return None

    def iter_lines(self) -> Iterator[str]:
        yield from self._lines


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = iter(responses)
        self.payloads: list[dict[str, object]] = []

    def stream(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
    ) -> FakeResponse:
        assert method == "POST"
        assert url.endswith("/v1/chat/completions")
        assert headers["X-Request-Id"].startswith("conversation-1-turn-")
        self.payloads.append(json)
        return next(self._responses)


def _stream_lines(
    *,
    prompt_ids: list[int],
    output_ids: list[int],
    text: str,
) -> list[str]:
    return [
        _data(
            {
                "prompt_token_ids": prompt_ids,
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
        ),
        _data(
            {
                "choices": [
                    {
                        "delta": {"content": text},
                        "token_ids": output_ids,
                        "finish_reason": "length",
                    }
                ]
            }
        ),
        _data(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(output_ids),
                    "total_tokens": len(prompt_ids) + len(output_ids),
                },
            }
        ),
        "data: [DONE]",
        ": eof",
    ]


def test_execute_two_turn_uses_one_conversation_and_validates_pap_cache() -> None:
    first_prompt = [1, 2, 3, 4]
    first_output = [10, 11, 12]
    second_prompt = [*first_prompt, 10, 11, 99]
    client = FakeClient(
        [
            FakeResponse(
                _stream_lines(
                    prompt_ids=first_prompt,
                    output_ids=first_output,
                    text="answer-one",
                ),
                {
                    "X-PAP-Prefill-Prompt-Tokens": "4",
                    "X-PAP-Prefill-Cached-Tokens": "0",
                    "X-PAP-Prefill-Computed-Tokens": "4",
                    "X-PAP-Prefill-Ms": "10",
                },
            ),
            FakeResponse(
                _stream_lines(
                    prompt_ids=second_prompt,
                    output_ids=[20, 21, 22],
                    text="answer-two",
                ),
                {
                    "X-PAP-Prefill-Prompt-Tokens": "7",
                    "X-PAP-Prefill-Cached-Tokens": "6",
                    "X-PAP-Prefill-Computed-Tokens": "1",
                    "X-PAP-Prefill-Ms": "1",
                },
            ),
        ]
    )
    times = iter([0.0, 0.01, 0.03, 0.10, 0.11, 0.14])
    config = NorthStarConfig(
        base_url="http://127.0.0.1:9000",
        model="/local/model",
        corpus_path="/local/corpus",
        result_path="/tmp/result.json",
        architecture="pap",
        topology="1pa1p",
        conversation_id="conversation-1",
        cache_salt="cache-salt-1",
        hardware_signature="NVIDIA L20x2",
        git_commit="a" * 40,
        git_tracked_worktree_dirty=False,
        offload_exec_transport="local_fast",
        direct_mailbox_output=True,
        unified_md_fast_key=True,
        document_tokens=4,
        append_tokens=2,
        output_tokens=3,
        block_size=2,
    )

    result = execute_two_turn(
        config,
        tokenizer=FakeTokenizer(),
        corpus="local corpus",
        client=client,
        clock=lambda: next(times),
    )

    assert result["validity"] == {"status": "passed", "cache_gate": "passed"}
    assert result["cache_validation"]["decode_derived_hit_tokens"] == 2
    assert result["cache_validation"]["actual_cached_tokens"] == 6
    assert result["git_commit"] == "a" * 40
    assert result["git_tracked_worktree_dirty"] is False
    assert result["implementation"] == {
        "offload_exec_transport": "local_fast",
        "direct_mailbox_output": True,
        "unified_md_fast_key": True,
    }
    assert len(result["implementation_fingerprint"]) == 64
    assert len(result["rounds"]) == 2
    assert "prompt_token_ids" not in result["rounds"][0]
    assert client.payloads[0]["conversation_id"] == "conversation-1"
    assert client.payloads[1]["conversation_id"] == "conversation-1"
    assert client.payloads[0]["cache_salt"] == "cache-salt-1"
    assert client.payloads[1]["cache_salt"] == "cache-salt-1"
    assert client.payloads[1]["messages"][-2:] == [
        {"role": "assistant", "content": "answer-one"},
        {
            "role": "user",
            "content": (
                "Using this additional passage, refine the summary and identify "
                "one changed conclusion.\n\ndecoded:4-5"
            ),
        },
    ]
