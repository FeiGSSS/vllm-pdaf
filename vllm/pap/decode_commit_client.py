# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility alias for :mod:`vllm.pap.lifecycle.commit`."""

import sys

from vllm.pap.lifecycle import commit as _commit

sys.modules[__name__] = _commit
