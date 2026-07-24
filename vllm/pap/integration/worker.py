# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP ownership boundary for vLLM GPU worker setup."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from vllm.pap.integration.settings import PAPRuntimeSettings
from vllm.pap.runtime_cuda_context_audit import write_runtime_cuda_context_audit


@dataclass(frozen=True, slots=True)
class PAPWorkerAdapter:
    """Stable PAP settings and hooks used by one vLLM GPU worker."""

    settings: PAPRuntimeSettings

    @classmethod
    def from_environ(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> PAPWorkerAdapter:
        """Create one worker adapter from process environment settings."""
        return cls(PAPRuntimeSettings.from_environ(environ))

    @property
    def projection_kv_unaware(self) -> bool:
        return self.settings.projection_kv_unaware

    @property
    def cudagraph_compatible(self) -> bool:
        return self.settings.cudagraph_compatible

    @property
    def skip_model_warmup(self) -> bool:
        """Return whether Projection can skip all model warmup work."""
        return self.projection_kv_unaware and not self.cudagraph_compatible

    @property
    def skip_local_attention_kernel_warmup(self) -> bool:
        """Return whether local-attention kernel warmup is inapplicable."""
        return self.projection_kv_unaware

    @property
    def critical_trace(self) -> bool:
        return self.settings.critical_trace

    def write_cuda_context_audit(self) -> None:
        """Record the worker CUDA context after distributed initialization."""
        write_runtime_cuda_context_audit(role=self.settings.cuda_context_role)
