# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection scratch allocation at the vLLM KV-cache boundary."""

from __future__ import annotations

from dataclasses import replace
from typing import Any


class PAPKVCacheAdapter:
    """Keep Projection scratch allocation out of model-runner mechanics."""

    @staticmethod
    def projection_scratch_config(config: Any, *, enabled: bool) -> Any:
        """Shrink Projection KV allocation to one structural scratch block."""
        if not enabled or int(config.num_blocks) <= 1:
            return config
        num_blocks = int(config.num_blocks)
        scratch_tensors = []
        for tensor in config.kv_cache_tensors:
            if tensor.block_stride > 0:
                scratch_size = int(tensor.block_stride)
            else:
                if int(tensor.size) % num_blocks:
                    raise ValueError("Projection KV tensor is not block-aligned")
                scratch_size = int(tensor.size) // num_blocks
            scratch_tensors.append(replace(tensor, size=scratch_size))
        return replace(config, num_blocks=1, kv_cache_tensors=scratch_tensors)
