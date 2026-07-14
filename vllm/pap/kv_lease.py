# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility alias for :mod:`vllm.pap.lifecycle.lease`."""

import sys

from vllm.pap.lifecycle import lease as _lease

sys.modules[__name__] = _lease
