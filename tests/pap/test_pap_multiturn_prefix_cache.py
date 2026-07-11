from examples.pap.pap_multiturn_prefix_cache import (
    build_second_prompt,
    expected_prefix_hit_tokens,
)


def test_expected_prefix_hit_includes_decode_blocks() -> None:
    first_prompt = list(range(32))
    first_output = list(range(100, 117))
    second_prompt = [*first_prompt, *first_output, 200, 201]

    expected = expected_prefix_hit_tokens(
        first_prompt,
        first_output,
        second_prompt,
        block_size=16,
    )

    assert expected == 48
    assert expected - len(first_prompt) == 16


def test_expected_prefix_hit_excludes_last_sampled_token() -> None:
    first_prompt = list(range(16))
    first_output = list(range(100, 116))
    second_prompt = [*first_prompt, *first_output]

    assert (
        expected_prefix_hit_tokens(
            first_prompt,
            first_output,
            second_prompt,
            block_size=16,
        )
        == 16
    )


def test_expected_prefix_hit_stops_at_token_mismatch() -> None:
    first_prompt = list(range(32))
    first_output = list(range(100, 117))
    second_prompt = [*first_prompt, 999, *first_output]

    assert (
        expected_prefix_hit_tokens(
            first_prompt,
            first_output,
            second_prompt,
            block_size=16,
        )
        == 32
    )


def test_build_second_prompt_keeps_exact_output_token_ids() -> None:
    assert build_second_prompt([1, 2], [3, 4], [5, 6]) == [1, 2, 3, 4, 5, 6]
