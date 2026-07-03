# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize PD/PAP vLLM benchmark result JSON files.

This helper is intentionally read-only. It accepts result JSON files or run
directories and prints a compact table with the fields used by the PAP-vs-PD
comparison notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_RESULT_NAME_RE = re.compile(
    r"(?P<prefix>.+?)_i(?P<input_len>\d+)_o(?P<output_len>\d+)"
    r"_q(?P<qps>[^_.]+)(?:_c(?P<max_concurrency>\d+))?"
    r"(?:_w(?P<num_warmups>\d+))?$"
)
_SKIP_NAMES = {"run_metadata.json"}


@dataclass(frozen=True)
class BenchRow:
    path: str
    mode: str
    topology: str
    model: str
    tp_size: str
    dtype: str
    input_len: str
    output_len: str
    qps: str
    max_concurrency: str
    num_prompts: str
    num_warmups: str
    completed: str
    failed: str
    request_throughput: str
    output_throughput: str
    total_token_throughput: str
    mean_ttft_ms: str
    median_ttft_ms: str
    p99_ttft_ms: str
    mean_tpot_ms: str
    median_tpot_ms: str
    p99_tpot_ms: str
    max_concurrent_requests: str
    server_batch_evidence: str

    @classmethod
    def headers(cls) -> list[str]:
        return list(cls.__dataclass_fields__.keys())

    def values(self) -> list[str]:
        return [str(getattr(self, field)) for field in self.headers()]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _metadata_for_result(path: Path) -> dict[str, Any]:
    metadata_path = path.parent / "run_metadata.json"
    return _load_json(metadata_path) or {}


def _first_value(*sources: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return value
    return None


def _candidate_json_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".json":
            files.append(path)
            continue
        if path.is_dir():
            files.extend(path.rglob("*.json"))
    return sorted(
        {
            file
            for file in files
            if file.name not in _SKIP_NAMES and not file.name.startswith(".")
        }
    )


def _row_from_result(path: Path) -> BenchRow | None:
    data = _load_json(path)
    if not data or "completed" not in data:
        return None

    metadata = _metadata_for_result(path)
    match = _RESULT_NAME_RE.search(path.stem)
    groups = match.groupdict() if match else {}

    return BenchRow(
        path=str(path),
        mode=_fmt(metadata.get("mode")),
        topology=_fmt(metadata.get("topology") or groups.get("prefix")),
        model=_fmt(_first_value(metadata, data, keys=("model", "model_name", "model_id"))),
        tp_size=_fmt(_first_value(metadata, data, keys=("tp_size", "tensor_parallel_size"))),
        dtype=_fmt(_first_value(metadata, data, keys=("dtype", "kv_cache_dtype"))),
        input_len=_fmt(
            _first_value(metadata, data, keys=("input_len", "random_input_len"))
            or groups.get("input_len")
        ),
        output_len=_fmt(
            _first_value(metadata, data, keys=("output_len", "random_output_len"))
            or groups.get("output_len")
        ),
        qps=_fmt(
            _first_value(metadata, data, keys=("qps", "request_rate"))
            or groups.get("qps")
        ),
        max_concurrency=_fmt(
            _first_value(metadata, data, keys=("max_concurrency",))
            or groups.get("max_concurrency")
        ),
        num_prompts=_fmt(_first_value(metadata, data, keys=("num_prompts",))),
        num_warmups=_fmt(
            _first_value(metadata, data, keys=("num_warmups",))
            or groups.get("num_warmups")
        ),
        completed=_fmt(data.get("completed")),
        failed=_fmt(data.get("failed")),
        request_throughput=_fmt(data.get("request_throughput")),
        output_throughput=_fmt(data.get("output_throughput")),
        total_token_throughput=_fmt(data.get("total_token_throughput")),
        mean_ttft_ms=_fmt(data.get("mean_ttft_ms")),
        median_ttft_ms=_fmt(data.get("median_ttft_ms")),
        p99_ttft_ms=_fmt(data.get("p99_ttft_ms")),
        mean_tpot_ms=_fmt(data.get("mean_tpot_ms")),
        median_tpot_ms=_fmt(data.get("median_tpot_ms")),
        p99_tpot_ms=_fmt(data.get("p99_tpot_ms")),
        max_concurrent_requests=_fmt(data.get("max_concurrent_requests")),
        server_batch_evidence=_fmt(
            _first_value(
                metadata,
                data,
                keys=(
                    "server_batch_evidence",
                    "batch_evidence",
                    "effective_server_side_batch_evidence",
                ),
            )
        ),
    )


def _print_csv(rows: list[BenchRow]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(BenchRow.headers())
    for row in rows:
        writer.writerow(row.values())


def _print_markdown(rows: list[BenchRow]) -> None:
    headers = [
        "mode",
        "topology",
        "model",
        "TP",
        "dtype",
        "input",
        "output",
        "qps",
        "max conc",
        "observed max conc",
        "num prompts",
        "warmup",
        "completed",
        "failed",
        "req/s",
        "out tok/s",
        "total tok/s",
        "mean TTFT",
        "median TTFT",
        "p99 TTFT",
        "mean TPOT",
        "median TPOT",
        "p99 TPOT",
        "batch evidence",
        "artifact",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [
            row.mode,
            row.topology,
            row.model,
            row.tp_size,
            row.dtype,
            row.input_len,
            row.output_len,
            row.qps,
            row.max_concurrency,
            row.max_concurrent_requests,
            row.num_prompts,
            row.num_warmups,
            row.completed,
            row.failed,
            row.request_throughput,
            row.output_throughput,
            row.total_token_throughput,
            row.mean_ttft_ms,
            row.median_ttft_ms,
            row.p99_ttft_ms,
            row.mean_tpot_ms,
            row.median_tpot_ms,
            row.p99_tpot_ms,
            row.server_batch_evidence,
            row.path,
        ]
        print("| " + " | ".join(values) + " |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize PD/PAP vLLM benchmark result JSON files."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--format",
        choices=("csv", "markdown"),
        default="markdown",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        row
        for path in _candidate_json_files(args.paths)
        if (row := _row_from_result(path)) is not None
    ]
    if args.format == "csv":
        _print_csv(rows)
    else:
        _print_markdown(rows)


if __name__ == "__main__":
    main()
