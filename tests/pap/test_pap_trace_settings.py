# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.pap.config import PAPConfigError, PAPStepTraceConfig


@pytest.mark.parametrize(
    "overrides",
    [
        {"PAP_PROJECTION_PA_TRACE_RING_STEPS": "0"},
        {"PAP_PROJECTION_PA_TRACE_SAMPLES": "2049"},
        {"PAP_PROJECTION_PA_TRACE_FLUSH_SECONDS": "nan"},
        {"PAP_PROJECTION_PA_TRACE_FLUSH_SECONDS": "inf"},
        {"PAP_PROJECTION_PA_TRACE_FLUSH_SECONDS": "0"},
    ],
)
def test_step_trace_rejects_invalid_settings_before_allocating_buffers(overrides):
    with pytest.raises(PAPConfigError):
        PAPStepTraceConfig.from_env(
            {"PAP_PROJECTION_PA_TRACE_OUTPUT": "/tmp/trace.pt", **overrides}
        )


def test_step_trace_settings_are_a_snapshot():
    env = {"PAP_PROJECTION_PA_TRACE_OUTPUT": "/tmp/trace.pt"}
    config = PAPStepTraceConfig.from_env(env)
    env["PAP_PROJECTION_PA_TRACE_SAMPLES"] = "1"
    assert config.sample_steps == 512
    assert config.ring_steps == 2048
    assert config.export_interval_seconds == 5.0
