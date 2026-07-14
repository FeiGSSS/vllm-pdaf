# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility launcher for the packaged PAP Attention runtime."""

from vllm.pap.attention_executor import main


if __name__ == "__main__":
    main()
