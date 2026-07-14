# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility entry point for the PAP Attention service."""

import sys

from vllm.pap import service as _service

sys.modules[__name__] = _service

if __name__ == "__main__":
    _service.main()
