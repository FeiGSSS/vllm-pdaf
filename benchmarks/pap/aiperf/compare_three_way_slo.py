#!/usr/bin/env python3
"""Summarize DP/PD/PAP capacity matrices for SLO-oriented comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SLO_TIERS = ("strict", "standard", "relaxed")


def _fmt_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return "PASS" if value else "FAIL"


def _format_row(row: dict[str, Any], architecture: str) -> list[str]:
    entries = [
        architecture.upper(),
        str(row["topology"]),
        str(row["concurrency"]),
        _fmt_bool(bool(row.get("correctness"))),
    ]
    for tier in SLO_TIERS:
        pass_key = _fmt_bool(bool(row.get(tier)) if row.get(tier) is not None else None)
        throughput = _fmt_float(row.get(f"{tier}_goodput_requests_per_second"))
        good = "-"
        good_frac = row.get(f"{tier}_good_fraction")
        if good_frac is not None:
            good = f"{good_frac * 100:.1f}%"
        entries.extend([pass_key, throughput, good])
    return entries


def _fmt_float(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def _fmt_pct(value: float | None, precision: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}%"


def load_matrix(matrix_root: Path) -> tuple[dict, list[dict]]:
    results_path = matrix_root / "capacity_results.json"
    envelope_path = matrix_root / "capacity_envelope.json"
    if not results_path.exists():
        raise FileNotFoundError(f"missing capacity_results.json: {results_path}")
    if not envelope_path.exists():
        raise FileNotFoundError(f"missing capacity_envelope.json: {envelope_path}")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    return envelope, results["rows"]


def best_point(rows: list[dict], architecture: str, tier: str) -> dict | None:
    candidates = [
        row
        for row in rows
        if row["architecture"] == architecture
        and row["correctness"]
        and row[tier]
        and row[f"{tier}_goodput_requests_per_second"] is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row[f"{tier}_goodput_requests_per_second"],
            row["concurrency"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_root", type=Path)
    args = parser.parse_args()

    envelope, rows = load_matrix(args.matrix_root)
    output_lines = [
        "# Three-way DP/PD/PAP SLO comparison",
        "",
    ]

    output_lines.append("## Per-SLO best goodput (correct runs)")
    for tier in SLO_TIERS:
        output_lines.append(f"- {tier}:")
        for architecture in ("pap", "pd", "dp"):
            row = best_point(rows, architecture, tier)
            if row is None:
                topology = "-"
                c = "-"
                goodput = "-"
                ttft = "-"
                ttft_p99 = "-"
                itl = "-"
                itl_p99 = "-"
                errors = "-"
                good = "-"
            else:
                topology = row["topology"]
                c = row["concurrency"]
                goodput = _fmt_float(row[f"{tier}_goodput_requests_per_second"])
                ttft = _fmt_float(row["ttft_p95_ms"])
                ttft_p99 = _fmt_float(row.get("ttft_p99_ms"))
                itl = _fmt_float(row["itl_p95_ms"])
                itl_p99 = _fmt_float(row.get("itl_p99_ms"))
                errors = _fmt_float(row.get("request_error_count"), precision=0)
                good = (
                    f"{row[f'{tier}_good_fraction'] * 100:.1f}%"
                    if row[f"{tier}_good_fraction"] is not None
                    else "-"
                )
            output_lines.append(
                f"  - {architecture.upper()}: {topology} C={c}, "
                f"req/s={goodput}, TTFTp95={ttft}ms/TTFTp99={ttft_p99}ms, "
                f"ITLp95={itl}ms/ITLp99={itl_p99}ms, errors={errors}, good={good}"
            )
    output_lines.append("")

    output_lines.append("## Capacity envelope (strict/standard/relaxed)")
    for tier in SLO_TIERS:
        value = envelope["capacity_by_slo"][tier]
        output_lines.append(
            f"- {tier}: best_c: PAP={value['best_pap']['concurrency']}, "
            f"PD={value['best_pd']['concurrency']}, "
            f"DP={value['best_dp']['concurrency']}"
        )
    output_lines.append("")

    output_lines.append("## Compliant-goodput deltas")
    for tier in SLO_TIERS:
        value = envelope["compliant_goodput_by_slo"][tier]
        output_lines.append(
            f"- {tier}: "
            f"PAP over PD={_fmt_pct(value['pap_over_pd_percent'])}, "
            f"PAP over DP={_fmt_pct(value['pap_over_dp_percent'])}"
        )
    output_lines.append("")

    output_lines.append("## Full concurrency sweep")
    output_lines.append(
        "| Arch | Topology | C | Correct | "
        "Strict(pass) | Strict goodput(req/s) | Strict good(%) | "
        "Standard(pass) | Standard goodput(req/s) | Standard good(%) | "
        "Relaxed(pass) | Relaxed goodput(req/s) | Relaxed good(%) |"
    )
    output_lines.append(
        "| --- | --- | ---: | --- | --- | ---: | ---: | "
        "--- | ---: | ---: | --- | ---: | ---: |"
    )

    for row in sorted(
        rows,
        key=lambda row: (
            row["architecture"],
            row["topology"],
            row["concurrency"],
        ),
    ):
        output_lines.append(
            "| " + " | ".join(_format_row(row, row["architecture"])) + " |"
        )
    output_lines.append("")

    output_lines.append("Artifacts generated:")
    output_lines.append(f"- {args.matrix_root / 'capacity_results.md'}")
    output_lines.append(f"- {args.matrix_root / 'capacity_envelope.json'}")
    output_lines.append(f"- {args.matrix_root / 'capacity_results.json'}")

    report_path = args.matrix_root / "three_way_slo_summary.md"
    report_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
