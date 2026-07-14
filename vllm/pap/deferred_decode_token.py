# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility alias for :mod:`vllm.pap.lifecycle.decode_token`."""

import sys

from vllm.pap.lifecycle import decode_token as _decode_token

sys.modules[__name__] = _decode_token
