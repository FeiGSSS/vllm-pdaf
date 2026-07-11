# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.serving import (
    _make_prompt_tokens_details,
)
from vllm.outputs import RequestOutput

pytestmark = pytest.mark.cpu_test


def test_request_output_exposes_exclusive_cached_token_counts() -> None:
    output = RequestOutput(
        request_id="request-1",
        prompt="prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=True,
        num_cached_tokens=96,
        num_local_cached_tokens=64,
        num_external_cached_tokens=32,
    )

    assert output.num_cached_tokens == 96
    assert output.num_local_cached_tokens == 64
    assert output.num_external_cached_tokens == 32


def test_prompt_token_details_expose_exclusive_cached_token_counts() -> None:
    details = _make_prompt_tokens_details(True, 96, None, 64, 32)

    assert details is not None
    assert details.model_dump(exclude_none=True) == {
        "cached_tokens": 96,
        "local_cached_tokens": 64,
        "external_cached_tokens": 32,
    }


def test_prompt_token_details_reject_inconsistent_cached_token_counts() -> None:
    with pytest.raises(ValueError):
        _make_prompt_tokens_details(True, 96, None, 64, 48)


@pytest.mark.parametrize(
    ("num_cached_tokens", "num_local_cached_tokens", "num_external_cached_tokens"),
    [
        (-1, -1, 0),
        (96, -1, 97),
        (96, 97, -1),
    ],
)
def test_prompt_token_details_reject_negative_cached_token_counts(
    num_cached_tokens: int,
    num_local_cached_tokens: int,
    num_external_cached_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        _make_prompt_tokens_details(
            True,
            num_cached_tokens,
            None,
            num_local_cached_tokens,
            num_external_cached_tokens,
        )
