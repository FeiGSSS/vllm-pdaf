# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI wrapper for PAP remote-attention diagnostics."""

from benchmarks.pap.tooling.remote_attention_diagnostics import main


if __name__ == "__main__":
    raise SystemExit(main())
