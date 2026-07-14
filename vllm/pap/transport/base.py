# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-neutral PAP transport contracts."""

from vllm.pap.protocol import PAPOffloadExecTransport, PAPTensorTransport

__all__ = ["PAPOffloadExecTransport", "PAPTensorTransport"]
