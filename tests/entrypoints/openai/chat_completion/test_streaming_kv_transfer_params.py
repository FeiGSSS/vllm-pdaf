# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.outputs import CompletionOutput, RequestOutput


def test_streaming_response_returns_terminal_kv_transfer_params():
    serving_chat = object.__new__(OpenAIServingChat)
    serving_chat.parser_cls = None
    serving_chat.response_role = "assistant"
    serving_chat.enable_force_include_usage = False
    serving_chat.enable_log_outputs = False
    serving_chat.enable_prompt_tokens_details = False
    serving_chat.request_logger = None
    serving_chat.system_fingerprint = None

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        stream=True,
    )
    kv_transfer_params = {
        "remote_engine_id": "decode-engine",
        "remote_block_ids": [[1, 2]],
    }

    async def result_generator():
        yield RequestOutput(
            request_id="test-request",
            prompt="test",
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="done",
                    token_ids=[4],
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason="stop",
                )
            ],
            finished=True,
            kv_transfer_params=kv_transfer_params,
        )

    async def collect_chunks():
        chunks = []
        async for chunk_str in serving_chat.chat_completion_stream_generator(
            request=request,
            result_generator=result_generator(),
            request_id="test-request",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=RequestResponseMetadata(
                request_id="test-request",
                model_name="test-model",
            ),
        ):
            if chunk_str.startswith("data: {"):
                chunks.append(json.loads(chunk_str[6:]))
        return chunks

    terminal_chunks = [
        chunk
        for chunk in asyncio.run(collect_chunks())
        if any(choice.get("finish_reason") for choice in chunk["choices"])
    ]
    assert len(terminal_chunks) == 1
    assert terminal_chunks[0]["kv_transfer_params"] == kv_transfer_params
