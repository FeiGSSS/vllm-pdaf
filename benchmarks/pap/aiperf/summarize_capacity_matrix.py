"""Aggregate compact PAP/PD AIPerf capacity summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SLO_TIER_NAMES = ("strict", "standard", "relaxed")


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_summaries(matrix_root: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((matrix_root / "runs").glob("*/capacity_summary.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"capacity summary is not an object: {path}")
        value["summary_path"] = str(path.relative_to(matrix_root))
        summaries.append(value)
    return summaries


def build_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        key = (
            summary["architecture"],
            summary["topology"],
            int(summary["concurrency"]),
        )
        grouped[key].append(summary)

    rows = []
    for (architecture, topology, concurrency), repetitions in grouped.items():
        ttft_p95 = [
            item["metrics"]["ttft_ms"]["p95"]
            for item in repetitions
            if item["metrics"]["ttft_ms"]["p95"] is not None
        ]
        itl_p95 = [
            item["metrics"]["itl_ms"]["p95"]
            for item in repetitions
            if item["metrics"]["itl_ms"]["p95"] is not None
        ]
        throughput = [
            item["metrics"]["request_throughput_per_second"]
            for item in repetitions
            if item["metrics"]["request_throughput_per_second"] is not None
        ]
        row = {
            "architecture": architecture,
            "topology": topology,
            "concurrency": concurrency,
            "repetitions": len(repetitions),
            "correctness": all(
                item["correctness"]["passed"] for item in repetitions
            ),
            "ttft_p95_ms": max(ttft_p95) if ttft_p95 else None,
            "itl_p95_ms": max(itl_p95) if itl_p95 else None,
            "request_throughput_per_second": min(throughput)
            if throughput
            else None,
        }
        for tier in SLO_TIER_NAMES:
            row[tier] = all(item["slo"][tier]["passed"] for item in repetitions)
            fractions = [
                item["slo"][tier]["good_request_fraction"]
                for item in repetitions
            ]
            row[f"{tier}_good_fraction"] = min(fractions)
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            0 if row["architecture"] == "pap" else 1,
            row["topology"],
            row["concurrency"],
        ),
    )


def build_envelope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    configs = sorted({(row["architecture"], row["topology"]) for row in rows})
    capacity_by_slo = {}
    for tier in SLO_TIER_NAMES:
        capacities = {}
        for architecture, topology in configs:
            passing = [
                row["concurrency"]
                for row in rows
                if row["architecture"] == architecture
                and row["topology"] == topology
                and row[tier]
            ]
            capacities[f"{architecture}:{topology}"] = max(passing, default=None)
        pd_capacities = {
            key: value
            for key, value in capacities.items()
            if key.startswith("pd:") and value is not None
        }
        pd_best_key = (
            max(pd_capacities, key=pd_capacities.get) if pd_capacities else None
        )
        pap_capacity = capacities.get("pap:3pa1p")
        pd_capacity = pd_capacities.get(pd_best_key) if pd_best_key else None
        capacity_by_slo[tier] = {
            "configurations": capacities,
            "pap_3pa1p": pap_capacity,
            "best_pd": {
                "topology": pd_best_key.partition(":")[2]
                if pd_best_key
                else None,
                "concurrency": pd_capacity,
            },
            "pap_minus_best_pd": (
                pap_capacity - pd_capacity
                if pap_capacity is not None and pd_capacity is not None
                else None
            ),
        }
    return {"schema_version": 1, "capacity_by_slo": capacity_by_slo}


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "architecture",
        "topology",
        "concurrency",
        "repetitions",
        "correctness",
        "ttft_p95_ms",
        "itl_p95_ms",
        "request_throughput_per_second",
        "strict",
        "standard",
        "relaxed",
        "strict_good_fraction",
        "standard_good_fraction",
        "relaxed_good_fraction",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    rows: list[dict[str, Any]],
    envelope: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# PAP/PD four-GPU AIPerf capacity matrix",
        "",
        "| Architecture | Topology | C | Correct | TTFT p95 ms | "
        "ITL p95 ms | Req/s | Strict | Standard | Relaxed |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {architecture} | {topology} | {concurrency} | {correctness} | "
            "{ttft} | {itl} | {throughput} | {strict} | {standard} | "
            "{relaxed} |".format(
                architecture=row["architecture"].upper(),
                topology=row["topology"],
                concurrency=row["concurrency"],
                correctness="pass" if row["correctness"] else "fail",
                ttft=_fmt(row["ttft_p95_ms"]),
                itl=_fmt(row["itl_p95_ms"]),
                throughput=_fmt(row["request_throughput_per_second"], 3),
                strict="pass" if row["strict"] else "fail",
                standard="pass" if row["standard"] else "fail",
                relaxed="pass" if row["relaxed"] else "fail",
            )
        )
    lines.extend(
        [
            "",
            "## Tested capacity envelope",
            "",
            "| SLO | PAP 3PA1P | Best PD topology | Best PD | PAP - PD |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    for tier, value in envelope["capacity_by_slo"].items():
        lines.append(
            f"| {tier} | {_fmt(value['pap_3pa1p'], 0)} | "
            f"{_fmt(value['best_pd']['topology'], 0)} | "
            f"{_fmt(value['best_pd']['concurrency'], 0)} | "
            f"{_fmt(value['pap_minus_best_pd'], 0)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = load_summaries(args.matrix_root)
    if not summaries:
        raise SystemExit(f"no capacity summaries under {args.matrix_root}")
    rows = build_rows(summaries)
    envelope = build_envelope(rows)
    write_tsv(rows, args.matrix_root / "capacity_results.tsv")
    write_markdown(rows, envelope, args.matrix_root / "capacity_results.md")
    (args.matrix_root / "capacity_envelope.json").write_text(
        json.dumps(envelope, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.matrix_root / "capacity_results.md")


if __name__ == "__main__":
    main()
