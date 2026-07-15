# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch the canonical PAP gateway from the examples directory."""

from vllm.pap.gateway.app import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
