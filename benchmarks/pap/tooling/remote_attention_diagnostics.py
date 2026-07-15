# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP remote-attention benchmark and trace diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.pap.tooling.trace_summary import summarize_pap_trace_logs

_RESULT_NAME_RE = re.compile(
    r"(?P<topology>[A-Za-z0-9]+)_i(?P<input_len>\d+)_o(?P<output_len>\d+)"
    r"_q(?P<qps>[^_.]+)(?:_c(?P<max_concurrency>\d+))?"
    r"(?:_w(?P<num_warmups>\d+))?"
)


@dataclass(frozen=True)
class LowerBoundConfig:
    batch_size: int
    q_size: int
    kv_size: int
    output_size: int
    dtype_bytes: int
    p2p_bandwidth_gbps: float
    attention_compute_ms: float
    num_layers: int


@dataclass(frozen=True)
class LowerBoundEstimate:
    bytes_per_layer: int
    transfer_ms_per_layer: float
    lower_bound_ms_per_layer: float
    lower_bound_ms_per_token: float


@dataclass(frozen=True)
class DiagnosticRow:
    path: str
    topology: str
    input_len: str
    output_len: str
    qps: str
    max_concurrency: str
    num_warmups: str
    completed: int
    failed: int
    request_throughput: float
    output_throughput: float
    median_ttft_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float
    projection_remote_total_median_ms: float
    projection_recv_median_ms: float
    attention_compute_median_ms: float
    attention_total_median_ms: float
    lower_bound_ms_per_layer: float
    lower_bound_ms_per_token: float
    e2e_ms_per_layer: float
    fast_path_status: dict[str, str]


def estimate_remote_attention_lower_bound(
    config: LowerBoundConfig,
) -> LowerBoundEstimate:
    elements_per_request = (
        config.q_size + config.kv_size + config.kv_size + config.output_size
    )
    bytes_per_layer = config.batch_size * elements_per_request * config.dtype_bytes
    bytes_per_ms = config.p2p_bandwidth_gbps * 1_000_000.0
    transfer_ms = bytes_per_layer / bytes_per_ms
    per_layer = transfer_ms + config.attention_compute_ms
    return LowerBoundEstimate(
        bytes_per_layer=bytes_per_layer,
        transfer_ms_per_layer=transfer_ms,
        lower_bound_ms_per_layer=per_layer,
        lower_bound_ms_per_token=per_layer * config.num_layers,
    )


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _result_json_for_run(run_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in run_dir.glob("*.json")
        if path.name != "run_metadata.json" and not path.name.startswith(".")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one result JSON in {run_dir}, found {len(candidates)}"
        )
    return candidates[0]


def _stat_median(summary: dict[str, Any], section: str, field: str) -> float:
    value = summary.get(section, {}).get(field)
    if value is None:
        return 0.0
    return float(getattr(value, "median", 0.0))


def _fmt_stat(value: float) -> str:
    return f"{value:.3f}"


def _fast_path_status(trace_summary: dict[str, Any]) -> dict[str, str]:
    paged_flash = _stat_median(trace_summary, "attention_trace", "paged_flash_ms")
    fallback = _stat_median(trace_summary, "attention_trace", "fallback_ms")
    calls = _stat_median(trace_summary, "attention_trace", "calls")
    return {
        "paged_flash": "active" if paged_flash > 0.0 else "inactive",
        "fallback": "active" if fallback > 0.0 else "inactive",
        "attention_batch_calls_median": _fmt_stat(calls),
    }


def summarize_run_directory(
    run_dir: str | Path,
    *,
    lower_bound_config: LowerBoundConfig | None = None,
) -> DiagnosticRow:
    run_path = Path(run_dir)
    result_path = _result_json_for_run(run_path)
    result = _load_json(result_path)
    match = _RESULT_NAME_RE.search(result_path.stem)
    groups = match.groupdict() if match else {}
    service_logs = run_path / "service_logs"
    trace_summary = summarize_pap_trace_logs(service_logs) if service_logs.exists() else {}
    attention_compute = _stat_median(trace_summary, "attention_trace", "compute_ms")
    config = lower_bound_config or LowerBoundConfig(
        batch_size=64,
        q_size=4096,
        kv_size=1024,
        output_size=4096,
        dtype_bytes=2,
        p2p_bandwidth_gbps=21.0,
        attention_compute_ms=attention_compute or 0.12,
        num_layers=36,
    )
    lower_bound = estimate_remote_attention_lower_bound(config)
    median_tpot = float(result.get("median_tpot_ms") or 0.0)
    return DiagnosticRow(
        path=str(run_path),
        topology=str(groups.get("topology") or ""),
        input_len=str(groups.get("input_len") or ""),
        output_len=str(groups.get("output_len") or ""),
        qps=str(groups.get("qps") or ""),
        max_concurrency=str(groups.get("max_concurrency") or ""),
        num_warmups=str(groups.get("num_warmups") or ""),
        completed=int(result.get("completed") or 0),
        failed=int(result.get("failed") or 0),
        request_throughput=float(result.get("request_throughput") or 0.0),
        output_throughput=float(result.get("output_throughput") or 0.0),
        median_ttft_ms=float(result.get("median_ttft_ms") or 0.0),
        median_tpot_ms=median_tpot,
        p99_tpot_ms=float(result.get("p99_tpot_ms") or 0.0),
        projection_remote_total_median_ms=_stat_median(
            trace_summary, "projection_timeline", "remote_total_ms"
        ),
        projection_recv_median_ms=_stat_median(
            trace_summary, "projection_timeline", "recv_ms"
        ),
        attention_compute_median_ms=attention_compute,
        attention_total_median_ms=_stat_median(
            trace_summary, "attention_trace", "total_ms"
        ),
        lower_bound_ms_per_layer=lower_bound.lower_bound_ms_per_layer,
        lower_bound_ms_per_token=lower_bound.lower_bound_ms_per_token,
        e2e_ms_per_layer=median_tpot / config.num_layers if config.num_layers else 0.0,
        fast_path_status=_fast_path_status(trace_summary),
    )


def _row_values(row: DiagnosticRow) -> list[str]:
    fast_paths = ",".join(
        f"{key}={value}" for key, value in sorted(row.fast_path_status.items())
    )
    return [
        row.topology,
        row.input_len,
        row.output_len,
        row.qps,
        row.num_warmups,
        row.max_concurrency,
        f"{row.completed}/{row.failed}",
        f"{row.output_throughput:.3f}",
        f"{row.median_tpot_ms:.3f}",
        f"{row.projection_remote_total_median_ms:.3f}",
        f"{row.projection_recv_median_ms:.3f}",
        f"{row.attention_compute_median_ms:.3f}",
        f"{row.lower_bound_ms_per_layer:.3f}",
        f"{row.e2e_ms_per_layer:.3f}",
        fast_paths,
        row.path,
    ]


def rows_to_markdown(rows: list[DiagnosticRow]) -> str:
    headers = [
        "topology",
        "input",
        "output",
        "qps",
        "warmup",
        "max conc",
        "done/failed",
        "out tok/s",
        "median TPOT",
        "proj remote",
        "proj recv",
        "attn compute",
        "lb/layer",
        "e2e/layer",
        "fast paths",
        "path",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_row_values(row)) + " |")
    return "\n".join(lines)


def _candidate_run_dirs(paths: list[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".json":
            run_dirs.append(path.parent)
        elif path.is_dir() and any(child.suffix == ".json" for child in path.iterdir()):
            run_dirs.append(path)
        elif path.is_dir():
            run_dirs.extend(
                child.parent
                for child in path.rglob("*.json")
                if child.name != "run_metadata.json"
            )
    return sorted(set(run_dirs))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    rows = [summarize_run_directory(path) for path in _candidate_run_dirs(args.paths)]
    print(rows_to_markdown(rows))
    return 0
