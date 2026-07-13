from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

from benchmarks.multi_turn.pap_pd_multiturn_load_client import (
    ConversationIdentity,
    LoadConfig,
    build_completion_payload,
    build_conversation_identities,
    build_load_workload,
    build_profile_id,
    execute_load,
    interval_concurrency,
    validate_config,
    validate_pap_transition,
)


def _config(**overrides: object) -> LoadConfig:
    values: dict[str, object] = {
        "base_url": "http://127.0.0.1:9000",
        "model": "/local/model",
        "corpus_path": "/local/corpus",
        "result_path": "/tmp/result.json",
        "architecture": "pap",
        "topology": "1pa1p",
        "conversation_id_prefix": "run-conversation",
        "cache_salt_prefix": "run-cache-salt",
        "hardware_signature": "NVIDIA L20x2",
        "git_commit": "a" * 40,
        "git_tracked_worktree_dirty": False,
        "offload_exec_transport": "local_fast",
        "direct_mailbox_output": True,
        "unified_md_fast_key": True,
    }
    values.update(overrides)
    return LoadConfig(**values)  # type: ignore[arg-type]


def test_profile_id_versions_c1_and_c4_lanes() -> None:
    c4 = _config()
    c1 = replace(c4, active_conversations=1)

    assert build_profile_id(c1) == "qwen3_8b_token_16k_5turn_o256_c1_v2"
    assert build_profile_id(c4) == "qwen3_8b_token_16k_5turn_o256_c4_v2"


def test_config_requires_engine_capacity_for_active_conversations() -> None:
    with pytest.raises(ValueError, match="max_num_seqs"):
        validate_config(_config(active_conversations=4, max_num_seqs=2))


class SliceTokenizer:
    def __init__(self, token_count: int = 32) -> None:
        self.token_count = token_count
        self.encoded_texts: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.encoded_texts.append(text)
        if text == "local corpus":
            return list(range(self.token_count))
        if text.startswith("Read the following document"):
            return [100, 101]
        if text.startswith("\n\nUser:"):
            return [103, 104]
        if text == "\n\nAssistant:":
            return [102]
        raise AssertionError(f"unexpected text: {text!r}")


def test_workload_uses_exact_nonoverlapping_token_slices() -> None:
    tokenizer = SliceTokenizer()

    initial_prompt, append_suffixes = build_load_workload(
        tokenizer,
        "local corpus",
        document_tokens=4,
        append_tokens=3,
        rounds=4,
    )

    assert initial_prompt == [100, 101, 0, 1, 2, 3, 102]
    assert append_suffixes == (
        (103, 104, 4, 5, 6, 102),
        (103, 104, 7, 8, 9, 102),
        (103, 104, 10, 11, 12, 102),
    )


def test_completion_payload_uses_exact_tokens_and_conversation_identity() -> None:
    identity = ConversationIdentity(0, "conversation", "cache-salt")

    pap = build_completion_payload(
        config=_config(),
        identity=identity,
        prompt_token_ids=[1, 2, 3],
    )
    pd = build_completion_payload(
        config=_config(architecture="pd", topology="1p1d"),
        identity=identity,
        prompt_token_ids=[1, 2, 3],
    )

    assert pap["prompt"] == [1, 2, 3]
    assert pap["add_special_tokens"] is False
    assert pap["conversation_id"] == "conversation"
    assert pd["conversation_id"] == "conversation"


def test_conversation_ids_and_cache_salts_are_unique() -> None:
    identities = build_conversation_identities(_config())

    assert len({item.conversation_id for item in identities}) == 4
    assert len({item.cache_salt for item in identities}) == 4
    assert identities[3].conversation_id.endswith("-conv-3")
    assert identities[3].cache_salt.endswith("-conv-3")


def test_interval_concurrency_reports_peak_and_time_weighted_average() -> None:
    summary = interval_concurrency([(0.0, 2.0), (1.0, 3.0)])

    assert summary == {
        "interval_count": 2,
        "peak": 2,
        "time_weighted_average": pytest.approx(4.0 / 3.0),
        "span_ms": 3000.0,
    }


def test_pap_transition_rejects_inexact_cached_tokens() -> None:
    with pytest.raises(ValueError, match="retokenized LCP"):
        validate_pap_transition(
            previous_prompt_ids=[1, 2, 3, 4],
            previous_output_ids=[10, 11, 12],
            current_prompt_ids=[1, 2, 3, 4, 10, 11, 99],
            current_prefill={
                "prompt_tokens": 7,
                "cached_tokens": 4,
                "computed_tokens": 3,
                "prefill_ms": 1,
            },
            block_size=2,
        )


class FakeLoadExecutor:
    def __init__(self, *, architecture: str) -> None:
        self.architecture = architecture
        self.previous_prompts: dict[int, list[int]] = {}
        self.previous_outputs: dict[int, list[int]] = {}
        self.calls: list[tuple[int, int, tuple[int, ...]]] = []

    async def __call__(
        self,
        client: Any,
        config: LoadConfig,
        identity: ConversationIdentity,
        prompt_token_ids: Sequence[int],
        round_index: int,
    ) -> tuple[dict[str, object], dict[str, int | None]]:
        assert client == f"client-{identity.index}"
        prompt_ids = [int(token_id) for token_id in prompt_token_ids]
        self.calls.append((round_index, identity.index, tuple(prompt_ids)))
        previous_prompt = self.previous_prompts.get(identity.index)
        previous_output = self.previous_outputs.get(identity.index)
        output_ids = [
            1000 + identity.index * 100 + round_index * 10 + offset
            for offset in range(config.output_tokens)
        ]
        self.previous_prompts[identity.index] = prompt_ids
        self.previous_outputs[identity.index] = output_ids
        start = round_index * 10.0 + identity.index / config.request_rate
        observation: dict[str, object] = {
            "request_id": (
                f"{identity.conversation_id}-turn-{round_index}"
            ),
            "prompt_token_ids": prompt_ids,
            "output_token_ids": output_ids,
            "assistant_text": f"answer-{identity.index}-{round_index}",
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(output_ids),
            "finish_reason": "length",
            "saw_done": True,
            "ttft_ms": 1000.0,
            "tpot_ms": 1000.0,
            "latency_ms": 3000.0,
            "eof_latency_ms": 3100.0,
            "post_token_stream_ms": 100.0,
            "request_started_at": start,
            "first_token_at": start + 1.0,
            "last_token_at": start + 3.0,
            "eof_at": start + 3.1,
        }
        if self.architecture == "pd":
            prefill = {
                "prompt_tokens": None,
                "cached_tokens": None,
                "computed_tokens": None,
                "prefill_ms": None,
            }
        elif previous_prompt is None or previous_output is None:
            prefill = {
                "prompt_tokens": len(prompt_ids),
                "cached_tokens": 0,
                "computed_tokens": len(prompt_ids),
                "prefill_ms": 10,
            }
        else:
            materialized = [*previous_prompt, *previous_output[:-1]]
            lcp = 0
            for left, right in zip(materialized, prompt_ids):
                if left != right:
                    break
                lcp += 1
            cached = lcp // config.block_size * config.block_size
            prefill = {
                "prompt_tokens": len(prompt_ids),
                "cached_tokens": cached,
                "computed_tokens": len(prompt_ids) - cached,
                "prefill_ms": 1,
            }
        return observation, prefill


def _execute_fake_load(architecture: str) -> tuple[dict[str, object], FakeLoadExecutor]:
    config = _config(
        architecture=architecture,
        topology="1pa1p" if architecture == "pap" else "1p1d",
        document_tokens=4,
        append_tokens=2,
        output_tokens=3,
        rounds=3,
        active_conversations=2,
        request_rate=2.0,
        block_size=2,
        max_model_len=100,
        max_num_seqs=2,
    )
    executor = FakeLoadExecutor(architecture=architecture)
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    result = asyncio.run(
        execute_load(
            config,
            tokenizer=SliceTokenizer(),
            corpus="local corpus",
            clients=("client-0", "client-1"),
            turn_executor=executor,
            sleeper=fake_sleep,
        )
    )
    assert sleep_calls == [0.5, 0.5, 0.5]
    return result, executor


def test_execute_load_enforces_barriers_and_exact_token_history() -> None:
    result, executor = _execute_fake_load("pap")

    assert [(round_index, index) for round_index, index, _ in executor.calls] == [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    ]
    round_one_prompt = executor.calls[0][2]
    round_two_prompt = executor.calls[2][2]
    round_three_prompt = executor.calls[4][2]
    assert round_one_prompt == (100, 101, 0, 1, 2, 3, 102)
    assert round_two_prompt[: len(round_one_prompt)] == round_one_prompt
    assert round_two_prompt[len(round_one_prompt) :][:3] == (1010, 1011, 1012)
    assert round_two_prompt[-5:] == (103, 104, 4, 5, 102)
    assert round_three_prompt[: len(round_two_prompt)] == round_two_prompt
    assert round_three_prompt[len(round_two_prompt) :][:3] == (1020, 1021, 1022)
    assert round_three_prompt[-5:] == (103, 104, 6, 7, 102)

    assert result["validity"] == {"status": "passed", "cache_gate": "passed"}
    assert result["overall"]["completed_requests"] == 6
    assert result["overall"]["failed_requests"] == 0
    assert len(result["rounds"]) == 3
    assert len(result["requests"]) == 6
    assert result["requests"] == [
        request
        for round_result in result["rounds"]
        for request in round_result["requests"]
    ]
    assert result["rounds"][0]["request_start_skew_ms"] == 500.0
    assert result["rounds"][0]["effective_concurrency"]["http"]["peak"] == 2
    assert result["rounds"][0]["effective_concurrency"]["decode"]["peak"] == 2
    assert result["rounds"][1]["metrics"]["ttft_ms"] == {
        "count": 2,
        "median": 1000.0,
        "p90": 1000.0,
        "max": 1000.0,
    }
    assert len(result["workload_shape_digest"]) == 64
    assert len(result["conversation_identity_digests"]) == 2
    assert len(result["cache_validation"]["transitions"]) == 4

    transition_requests = [
        request
        for round_result in result["rounds"][1:]
        for request in round_result["requests"]
    ]
    assert all(
        request["cache_validation"]["status"] == "passed"
        for request in transition_requests
    )
    assert all(
        request["cache_validation"]["decode_derived_hit_tokens"] >= 2
        for request in transition_requests
    )


def test_pd_load_requires_external_cache_validation() -> None:
    result, _executor = _execute_fake_load("pd")

    assert result["validity"] == {
        "status": "passed",
        "cache_gate": "requires_external_validation",
    }
    cache_validation = result["cache_validation"]
    assert cache_validation["status"] == "requires_external_validation"
    assert cache_validation["request_count"] == 6
    assert cache_validation["transition_count"] == 4
    assert len(cache_validation["transitions"]) == 4
    assert cache_validation["transitions"][0] == {
        "conversation_index": 0,
        "from_round": 1,
            "to_round": 2,
            "previous_prompt_tokens": 7,
            "materialized_history_tokens": 9,
            "expected_cached_tokens": 8,
        "decode_derived_hit_tokens": 2,
        "actual_cached_tokens": None,
    }
    assert all(
        request["cache_validation"]["status"]
        == "requires_external_validation"
        for round_result in result["rounds"]
        for request in round_result["requests"]
    )
