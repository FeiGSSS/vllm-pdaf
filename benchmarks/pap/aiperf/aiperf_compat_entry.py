# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run AIPerf 0.11 with local tokenizer paths supported in workers."""

from pathlib import Path

from aiperf.common.enums import GenericMetricUnit
from aiperf.common.models import TelemetryRecord
from aiperf.common.tokenizer import Tokenizer
from aiperf.gpu_telemetry.constants import (
    DCGM_TO_FIELD_MAPPING,
    GPU_TELEMETRY_METRICS_CONFIG,
)
from aiperf.gpu_telemetry.dcgm_collector import (
    SCALING_FACTORS,
    DCGMTelemetryCollector,
)
from prometheus_client.parser import text_string_to_metric_families

_resolve_local_snapshot = Tokenizer._resolve_local_snapshot.__func__


def _resolve_snapshot(cls: type[Tokenizer], name: str, revision: str) -> str:
    local_path = Path(name).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())
    return _resolve_local_snapshot(cls, name, revision)


Tokenizer._resolve_local_snapshot = classmethod(_resolve_snapshot)

_parse_dcgm_metrics = DCGMTelemetryCollector._parse_metrics_to_records


def _parse_metrics_with_hostname(
    self: DCGMTelemetryCollector,
    metrics_data: str,
) -> list[TelemetryRecord]:
    records = _parse_dcgm_metrics(self, metrics_data)
    if all(record.hostname is not None for record in records):
        return records

    hostnames: dict[int, str] = {}
    for family in text_string_to_metric_families(metrics_data):
        for sample in family.samples:
            gpu = sample.labels.get("gpu")
            hostname = sample.labels.get("Hostname") or sample.labels.get("hostname")
            if gpu is not None and hostname:
                hostnames[int(gpu)] = hostname
    for record in records:
        if record.hostname is None:
            record.hostname = hostnames.get(record.gpu_index)
    return records


DCGMTelemetryCollector._parse_metrics_to_records = _parse_metrics_with_hostname

_dcgm_activity_fields = {
    "DCGM_FI_PROF_GR_ENGINE_ACTIVE": "gr_engine_active",
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE": "tensor_active",
    "DCGM_FI_PROF_DRAM_ACTIVE": "dram_active",
}
DCGM_TO_FIELD_MAPPING.update(_dcgm_activity_fields)
SCALING_FACTORS.update({field: 100 for field in _dcgm_activity_fields.values()})
GPU_TELEMETRY_METRICS_CONFIG.extend(
    [
        (
            "Graphics/Compute Engine Active",
            "gr_engine_active",
            GenericMetricUnit.PERCENT,
        ),
        ("Tensor Pipe Active", "tensor_active", GenericMetricUnit.PERCENT),
        ("DRAM Active", "dram_active", GenericMetricUnit.PERCENT),
    ]
)


if __name__ == "__main__":
    from aiperf.cli import app

    app()
