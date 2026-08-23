# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the PAP NVSHMEM control and Graph paths."""

from __future__ import annotations

import regex as re
import torch

_LAYER_INDEX_PATTERN = re.compile(r"^(.*\.layers\.)(\d+)(\..*)$")


def layer_index_and_template(
    layer_name: str,
) -> tuple[int, tuple[str, str]] | None:
    """Split a model layer name into its numeric index and stable template."""
    match = _LAYER_INDEX_PATTERN.match(str(layer_name))
    if match is None:
        return None
    return int(match.group(2)), (match.group(1), match.group(3))


def layer_name_from_template(template: tuple[str, str], layer_index: int) -> str:
    """Render one model layer name from a stable template."""
    return f"{template[0]}{int(layer_index)}{template[1]}"


def dtype_name(dtype: torch.dtype) -> str:
    """Return the stable wire name for a supported tensor dtype."""
    return str(dtype).replace("torch.", "")


def dtype_from_name(name: str) -> torch.dtype:
    """Resolve a wire dtype name to a torch dtype."""
    dtype = getattr(torch, str(name), None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"unsupported PAP tensor dtype: {name}")
    return dtype
