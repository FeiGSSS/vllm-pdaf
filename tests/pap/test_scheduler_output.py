# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.request import Request


def test_new_request_data_preserves_kv_transfer_params() -> None:
    sampling_params = SamplingParams(
        max_tokens=1,
        extra_args={
            "kv_transfer_params": {"pap_attention_endpoint": "http://pa0:8300"}
        },
    )
    request = Request(
        request_id="cmpl-a-0-deadbeef",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params,
        pooling_params=None,
    )

    data = NewRequestData.from_request(
        request,
        block_ids=([4],),
        prefill_token_ids=[1, 2, 3],
    )
    request.kv_transfer_params["pap_attention_endpoint"] = "http://mutated:8300"

    assert data.kv_transfer_params == {"pap_attention_endpoint": "http://pa0:8300"}
