# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt rendering and token hashing for PAP Dynamo routing."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.inputs import SingletonPrompt
from vllm.renderers import BaseRenderer, merge_kwargs, renderer_from_config
from vllm.renderers.inputs.preprocess import parse_model_prompt, prompt_to_seq
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    ExternalBlockHash,
    hash_block_tokens,
    init_none_hash,
    maybe_convert_block_hash,
)

logger = logging.getLogger("pap_gateway")

PromptHashes: TypeAlias = tuple[ExternalBlockHash, ...]
PromptRoutingInput: TypeAlias = tuple[list[int], PromptHashes, str | None]


class PAPPromptTokenizer:
    """Render routing tokens once and forward the same tokens to Prefill."""

    def __init__(
        self,
        *,
        model: str,
        block_size: int,
        max_model_len: int | None = None,
        hf_overrides: Mapping[str, Any] | None = None,
        generation_config: str = "vllm",
    ) -> None:
        if block_size <= 0:
            raise ValueError("prefix block size must be positive")
        if not model:
            raise ValueError("PAP Dynamo routing requires a model for tokenization")
        self._block_size = block_size
        self._model_config: ModelConfig | None = None
        self._renderer: BaseRenderer | None = None
        if model:
            model_config_kwargs: dict[str, Any] = {
                "model": model,
                "hf_overrides": dict(hf_overrides or {}),
                "generation_config": generation_config,
            }
            if max_model_len is not None:
                model_config_kwargs["max_model_len"] = max_model_len
            model_config = ModelConfig(**model_config_kwargs)
            vllm_config = VllmConfig(
                model_config=model_config,
                device_config=DeviceConfig(device="cpu"),
            )
            self._model_config = model_config
            self._renderer = renderer_from_config(vllm_config)
        self._hash_function = get_hash_fn_by_name("sha256")
        init_none_hash(self._hash_function)

    @property
    def enabled(self) -> bool:
        return self._renderer is not None

    async def prompt_hashes(
        self,
        api_path: str,
        request: Mapping[str, Any],
    ) -> PromptRoutingInput | None:
        if not self.enabled:
            return None
        return await asyncio.to_thread(self._prompt_hashes_sync, api_path, request)

    def _prompt_hashes_sync(
        self,
        api_path: str,
        request: Mapping[str, Any],
    ) -> PromptRoutingInput | None:
        assert self._model_config is not None
        assert self._renderer is not None
        if api_path == "/v1/chat/completions":
            chat_request = ChatCompletionRequest.model_validate(dict(request))
            tool_dicts = (
                None
                if chat_request.tools is None
                else [tool.model_dump() for tool in chat_request.tools]
            )
            defaults = merge_kwargs(
                None,
                {
                    "tools": tool_dicts,
                    "tokenize": (
                        is_mistral_tokenizer(self._renderer.tokenizer)
                        or self._model_config.enable_prompt_embeds
                    ),
                },
            )
            mm_config = self._model_config.multimodal_config
            chat_params = chat_request.build_chat_params(None, "auto").with_defaults(
                defaults,
                default_media_io_kwargs=(
                    mm_config.media_io_kwargs if mm_config else None
                ),
                default_mm_processor_kwargs=chat_request.mm_processor_kwargs,
            )
            _, engine_inputs = self._renderer.render_chat(
                [chat_request.messages],
                chat_params,
                chat_request.build_tok_params(self._model_config),
                prompt_extras=self._prompt_extras(chat_request),
                skip_mm_cache=True,
            )
        elif api_path == "/v1/completions":
            completion_request = CompletionRequest.model_validate(dict(request))
            prompts: list[SingletonPrompt | bytes] = []
            if completion_request.prompt_embeds is not None:
                return None
            if completion_request.prompt is not None:
                prompts.extend(prompt_to_seq(completion_request.prompt))
            engine_inputs = self._renderer.render_cmpl(
                [parse_model_prompt(self._model_config, prompt) for prompt in prompts],
                completion_request.build_tok_params(self._model_config),
                prompt_extras=self._prompt_extras(completion_request),
                skip_mm_cache=True,
            )
        else:
            return None
        if len(engine_inputs) != 1:
            return None
        engine_input = engine_inputs[0]
        if engine_input.get("multi_modal_data") or engine_input.get("mm_placeholders"):
            return None
        token_ids = engine_input.get("prompt_token_ids")
        if not isinstance(token_ids, list) or any(
            not isinstance(token_id, int) for token_id in token_ids
        ):
            return None
        prompt = engine_input.get("prompt")
        return (
            token_ids,
            self.hash_token_ids(
                token_ids,
                cache_salt=request.get("cache_salt"),
            ),
            prompt if isinstance(prompt, str) else None,
        )

    @staticmethod
    def _prompt_extras(request: Any) -> dict[str, Any] | None:
        extras = {
            name: value
            for name in ("mm_processor_kwargs", "cache_salt")
            if (value := getattr(request, name, None)) is not None
        }
        return extras or None

    def hash_token_ids(
        self,
        token_ids: Sequence[int],
        *,
        cache_salt: object = None,
    ) -> PromptHashes:
        parent: BlockHash | None = None
        hashes = []
        for start in range(0, len(token_ids) - self._block_size + 1, self._block_size):
            extra_keys = (str(cache_salt),) if start == 0 and cache_salt else None
            parent = hash_block_tokens(
                self._hash_function,
                parent,
                token_ids[start : start + self._block_size],
                extra_keys,
            )
            hashes.append(maybe_convert_block_hash(parent))
        return tuple(hashes)


__all__ = ["PAPPromptTokenizer", "PromptHashes", "PromptRoutingInput"]
