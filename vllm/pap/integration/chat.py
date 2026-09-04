# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP tokenized-input adapters for internal OpenAI requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vllm.entrypoints.chat_utils import ConversationMessage
from vllm.inputs import EngineInput, tokens_input
from vllm.logger import init_logger
from vllm.pap.integration.request import PAPRequestMetadata

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

logger = init_logger(__name__)


def _consume_pap_tokenized_input(
    request: ChatCompletionRequest | CompletionRequest,
) -> EngineInput | None:
    params = request.kv_transfer_params
    metadata = PAPRequestMetadata.from_mapping(params)
    if not metadata.projection_kv_unaware and not metadata.tokenized_input:
        return None
    if params is None:
        raise ValueError("PAP tokenized request is missing metadata")

    updated_params = dict(params)
    prompt_token_ids = updated_params.pop("pap_prompt_token_ids", None)
    prompt_text = updated_params.pop("pap_prompt_text", None)
    updated_params.pop("pap_tokenized_input", None)
    if (
        not isinstance(prompt_token_ids, list)
        or not prompt_token_ids
        or any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in prompt_token_ids
        )
    ):
        raise ValueError("PAP request requires valid prompt token IDs")
    if any(token_id < 0 for token_id in prompt_token_ids):
        raise ValueError("PAP prompt token IDs must be non-negative")
    if metadata.projection_kv_unaware and metadata.remote_prefix_len != len(
        prompt_token_ids
    ):
        raise ValueError(
            "PAP Projection prompt token length mismatch "
            f"prompt_token_ids={len(prompt_token_ids)} "
            f"pap_remote_prefix_len={metadata.remote_prefix_len}"
        )
    if prompt_text is not None and not isinstance(prompt_text, str):
        raise ValueError("PAP prompt text must be a string")

    request.kv_transfer_params = updated_params
    if metadata.tokenized_input:
        logger.info_once("PAP Prefill is reusing Gateway-rendered prompt token IDs")
    return tokens_input(
        prompt_token_ids,
        prompt=prompt_text,
        cache_salt=request.cache_salt,
    )


def prepare_pap_tokenized_chat_input(
    request: ChatCompletionRequest,
) -> tuple[list[ConversationMessage], list[EngineInput]] | None:
    """Consume Gateway-rendered token IDs without rendering Chat again."""
    engine_input = _consume_pap_tokenized_input(request)
    if engine_input is None:
        return None
    conversation = cast(list[ConversationMessage], list(request.messages))
    return conversation, [engine_input]


def prepare_pap_tokenized_completion_input(
    request: CompletionRequest,
) -> list[EngineInput] | None:
    """Consume Gateway-tokenized IDs without tokenizing Completion again."""
    engine_input = _consume_pap_tokenized_input(request)
    return None if engine_input is None else [engine_input]


def prepare_pap_projection_chat_input(
    request: ChatCompletionRequest,
) -> tuple[list[ConversationMessage], list[EngineInput]] | None:
    """Backward-compatible name for Projection token reuse."""
    return prepare_pap_tokenized_chat_input(request)


__all__ = [
    "prepare_pap_projection_chat_input",
    "prepare_pap_tokenized_chat_input",
    "prepare_pap_tokenized_completion_input",
]
