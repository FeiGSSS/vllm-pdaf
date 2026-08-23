# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Projection admission adapter for OpenAI Chat requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vllm.entrypoints.chat_utils import ConversationMessage
from vllm.inputs import EngineInput, tokens_input
from vllm.pap.integration.request import PAPRequestMetadata

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )


def prepare_pap_projection_chat_input(
    request: ChatCompletionRequest,
) -> tuple[list[ConversationMessage], list[EngineInput]] | None:
    """Consume Prefill token IDs for a PAP Projection Chat request."""
    params = request.kv_transfer_params
    metadata = PAPRequestMetadata.from_mapping(params)
    if not metadata.projection_kv_unaware:
        return None
    if params is None:
        raise ValueError("PAP Projection request is missing metadata")

    updated_params = dict(params)
    prompt_token_ids = updated_params.pop("pap_prompt_token_ids", None)
    prompt_text = updated_params.pop("pap_prompt_text", None)
    if (
        not isinstance(prompt_token_ids, list)
        or not prompt_token_ids
        or any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in prompt_token_ids
        )
    ):
        raise ValueError("PAP Projection requires valid Prefill prompt token IDs")
    if any(token_id < 0 for token_id in prompt_token_ids):
        raise ValueError("PAP Projection prompt token IDs must be non-negative")
    if metadata.remote_prefix_len != len(prompt_token_ids):
        raise ValueError(
            "PAP Projection prompt token length mismatch "
            f"prompt_token_ids={len(prompt_token_ids)} "
            f"pap_remote_prefix_len={metadata.remote_prefix_len}"
        )
    if prompt_text is not None and not isinstance(prompt_text, str):
        raise ValueError("PAP Projection prompt text must be a string")

    # Do not forward the large token vector through SamplingParams into the
    # EngineCore. The EngineInput is its sole owner after admission.
    request.kv_transfer_params = updated_params
    engine_input = tokens_input(
        prompt_token_ids,
        prompt=prompt_text,
        cache_salt=request.cache_salt,
    )
    conversation = cast(list[ConversationMessage], list(request.messages))
    return conversation, [engine_input]
