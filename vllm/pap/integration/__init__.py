# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP adapters for vLLM request and execution integration."""

from vllm.pap.integration.chat import (
    prepare_pap_projection_chat_input,
    prepare_pap_tokenized_chat_input,
    prepare_pap_tokenized_completion_input,
)
from vllm.pap.integration.engine import PAPEngineAdapter
from vllm.pap.integration.kv_cache import PAPKVCacheAdapter
from vllm.pap.integration.projection import (
    build_projection_forward_context,
    select_projection_request_ids,
)
from vllm.pap.integration.request import (
    PAPProjectionRequestStore,
    PAPRequestMetadata,
)
from vllm.pap.integration.runner import PAPModelRunnerAdapter
from vllm.pap.integration.scheduler import (
    PAPAcceptedDecodeTokenPublisher,
    PAPProjectionScheduleState,
    PAPSchedulerAdapter,
)
from vllm.pap.integration.settings import PAPRuntimeSettings

__all__ = [
    "PAPAcceptedDecodeTokenPublisher",
    "PAPEngineAdapter",
    "PAPKVCacheAdapter",
    "PAPModelRunnerAdapter",
    "PAPProjectionScheduleState",
    "PAPProjectionRequestStore",
    "PAPRequestMetadata",
    "PAPRuntimeSettings",
    "PAPSchedulerAdapter",
    "build_projection_forward_context",
    "prepare_pap_projection_chat_input",
    "prepare_pap_tokenized_chat_input",
    "prepare_pap_tokenized_completion_input",
    "select_projection_request_ids",
]
