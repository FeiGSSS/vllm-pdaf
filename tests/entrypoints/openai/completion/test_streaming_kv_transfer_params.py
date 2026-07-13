# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json

from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.outputs import CompletionOutput, RequestOutput


def test_streaming_response_returns_terminal_kv_transfer_params():
    serving = object.__new__(OpenAIServingCompletion)
    serving.enable_force_include_usage = False
    serving.enable_prompt_tokens_details = False
    serving.system_fingerprint = None

    request = CompletionRequest(
        model="test-model",
        prompt=[1, 2, 3],
        max_tokens=1,
        stream=True,
        stream_options={"include_usage": True},
    )
    kv_transfer_params = {
        "remote_engine_id": "decode-engine",
        "remote_block_ids": [[1, 2]],
    }

    async def result_generator():
        yield 0, RequestOutput(
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
        async for chunk_str in serving.completion_stream_generator(
            request=request,
            engine_inputs=[{"prompt_token_ids": [1, 2, 3]}],
            result_generator=result_generator(),
            request_id="test-request",
            created_time=1,
            model_name="test-model",
            num_prompts=1,
            tokenizer=None,
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
