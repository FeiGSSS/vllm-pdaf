from benchmarks.pap.tooling.chat_prefix_cache_diagnostic import (
    _render_chat_prompt_token_ids,
    build_chat_payload,
    build_second_turn_messages,
    chat_prefix_metrics,
)


def test_build_second_turn_messages_preserves_assistant_text() -> None:
    first = [{"role": "user", "content": "first"}]

    result = build_second_turn_messages(first, "answer", "follow-up")

    assert result == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow-up"},
    ]
    assert first == [{"role": "user", "content": "first"}]


def test_chat_prefix_metrics_use_true_retokenized_lcp() -> None:
    metrics = chat_prefix_metrics(
        list(range(32)),
        list(range(100, 133)),
        [*range(32), *range(100, 127), 999],
        block_size=16,
    )

    assert metrics == {
        "committed_lcp_tokens": 59,
        "expected_prefix_hit_tokens": 48,
        "decode_derived_hit_tokens": 16,
    }


def test_build_chat_payload_is_deterministic_and_enables_thinking() -> None:
    messages = [{"role": "user", "content": "hello"}]

    payload = build_chat_payload(
        model="local-model",
        messages=messages,
        max_tokens=48,
        cache_salt="warm-salt",
    )

    assert payload == {
        "model": "local-model",
        "messages": messages,
        "max_tokens": 48,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": False,
        "return_token_ids": True,
        "cache_salt": "warm-salt",
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_render_chat_prompt_accepts_tokenizer_batch_encoding() -> None:
    class Tokenizer:
        @staticmethod
        def apply_chat_template(*args, **kwargs):
            assert kwargs["enable_thinking"] is True
            return {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
            }

    assert _render_chat_prompt_token_ids(
        Tokenizer(),
        [{"role": "user", "content": "hello"}],
    ) == [1, 2, 3]
