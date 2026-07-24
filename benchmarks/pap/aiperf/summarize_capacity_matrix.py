"""Aggregate compact PAP/PD AIPerf capacity summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SLO_TIER_NAMES = ("strict", "standard", "relaxed")
RUN_STATUS_LABELS = {
    "completed": "completed",
    "completed_with_errors": "completed with request errors",
    "completed_launcher_failed": "completed; launcher failed",
    "early_stopped_slo_impossible": "early-stopped: SLO impossible",
    "incomplete_slo_impossible": "incomplete: SLO impossible",
    "request_timeout": "incomplete: request timeout",
    "service_failed": "service failed",
    "incomplete": "incomplete",
}


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _summary_run_state(summary: dict[str, Any]) -> str:
    state = summary.get("run_status", {}).get("state")
    if isinstance(state, str) and state:
        return state
    completed = summary.get("correctness", {}).get("completed_requests")
    expected = summary.get("workload", {}).get("expected_requests")
    if completed == expected and expected is not None:
        return "completed"
    return "incomplete"


def _run_status_label(row: dict[str, Any]) -> str:
    states = row["run_statuses"]
    labels = [RUN_STATUS_LABELS.get(state, state) for state in states]
    return labels[0] if len(labels) == 1 else "mixed: " + ", ".join(labels)


def _completion_label(row: dict[str, Any]) -> str:
    completed = row["completed_requests"]
    expected = row["expected_requests"]
    if completed is None or expected is None:
        return "-"
    return f"{completed}/{expected}"


def _validation_label(row: dict[str, Any]) -> str:
    if row["correctness"]:
        return "pass"
    if any(state != "completed" for state in row["run_statuses"]):
        return "ineligible"
    return "fail"


def _slo_label(row: dict[str, Any], tier: str) -> str:
    if not row["correctness"]:
        return "ineligible"
    return "pass" if row[tier] else "fail"


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
        run_statuses = sorted({_summary_run_state(item) for item in repetitions})
        completed_requests = [
            item.get("correctness", {}).get("completed_requests")
            for item in repetitions
        ]
        completed_requests = [
            value for value in completed_requests if isinstance(value, int)
        ]
        expected_requests = [
            item.get("workload", {}).get("expected_requests") for item in repetitions
        ]
        expected_requests = [
            value for value in expected_requests if isinstance(value, int)
        ]
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
            "run_status": run_statuses[0] if len(run_statuses) == 1 else "mixed",
            "run_statuses": run_statuses,
            "completed_requests": min(completed_requests)
            if completed_requests
            else None,
            "expected_requests": max(expected_requests) if expected_requests else None,
            "correctness": all(item["correctness"]["passed"] for item in repetitions),
            "ttft_p95_ms": max(ttft_p95) if ttft_p95 else None,
            "itl_p95_ms": max(itl_p95) if itl_p95 else None,
            "request_throughput_per_second": min(throughput) if throughput else None,
        }
        for tier in SLO_TIER_NAMES:
            row[tier] = all(item["slo"][tier]["passed"] for item in repetitions)
            fractions = [
                item["slo"][tier]["good_request_fraction"] for item in repetitions
            ]
            row[f"{tier}_good_fraction"] = min(fractions)
            goodputs = [
                item["slo"][tier].get("goodput_requests_per_second")
                for item in repetitions
            ]
            goodputs = [value for value in goodputs if value is not None]
            row[f"{tier}_goodput_requests_per_second"] = (
                min(goodputs) if goodputs else None
            )
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            {"pap": 0, "pd": 1, "dp": 2}.get(row["architecture"], 3),
            row["topology"],
            row["concurrency"],
        ),
    )


def _best_compliant_goodput(
    rows: list[dict[str, Any]],
    tier: str,
    architecture: str,
) -> dict[str, Any]:
    goodput_key = f"{tier}_goodput_requests_per_second"
    eligible = [
        row
        for row in rows
        if row["architecture"] == architecture
        and row["correctness"]
        and row[tier]
        and row.get(goodput_key) is not None
    ]
    if not eligible:
        return {
            "topology": None,
            "concurrency": None,
            "requests_per_second": None,
        }
    best = max(
        eligible,
        key=lambda row: (row[goodput_key], row["concurrency"]),
    )
    return {
        "topology": best["topology"],
        "concurrency": best["concurrency"],
        "requests_per_second": best[goodput_key],
    }


def build_envelope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    configs = sorted({(row["architecture"], row["topology"]) for row in rows})
    capacity_by_slo = {}
    compliant_goodput_by_slo = {}
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
        best_by_architecture = {}
        for architecture in ("pap", "pd", "dp"):
            candidates = {
                key: value
                for key, value in capacities.items()
                if key.startswith(f"{architecture}:") and value is not None
            }
            best_key = max(candidates, key=candidates.get) if candidates else None
            best_by_architecture[architecture] = {
                "topology": best_key.partition(":")[2] if best_key else None,
                "concurrency": candidates.get(best_key) if best_key else None,
            }
        pap_capacity = best_by_architecture["pap"]["concurrency"]
        pd_capacity = best_by_architecture["pd"]["concurrency"]
        dp_capacity = best_by_architecture["dp"]["concurrency"]
        capacity_by_slo[tier] = {
            "configurations": capacities,
            "best_pap": best_by_architecture["pap"],
            "best_pd": best_by_architecture["pd"],
            "best_dp": best_by_architecture["dp"],
            "pap_minus_best_pd": (
                pap_capacity - pd_capacity
                if pap_capacity is not None and pd_capacity is not None
                else None
            ),
            "pap_minus_dp": (
                pap_capacity - dp_capacity
                if pap_capacity is not None and dp_capacity is not None
                else None
            ),
        }

        pap_goodput = _best_compliant_goodput(rows, tier, "pap")
        pd_goodput = _best_compliant_goodput(rows, tier, "pd")
        dp_goodput = _best_compliant_goodput(rows, tier, "dp")
        pap_value = pap_goodput["requests_per_second"]
        pd_value = pd_goodput["requests_per_second"]
        dp_value = dp_goodput["requests_per_second"]
        compliant_goodput_by_slo[tier] = {
            "best_pap": pap_goodput,
            "best_pd": pd_goodput,
            "best_dp": dp_goodput,
            "pap_over_pd_percent": (
                (pap_value / pd_value - 1) * 100
                if pap_value is not None and pd_value is not None and pd_value > 0
                else None
            ),
            "pap_over_dp_percent": (
                (pap_value / dp_value - 1) * 100
                if pap_value is not None and dp_value is not None and dp_value > 0
                else None
            ),
        }
    return {
        "schema_version": 2,
        "capacity_by_slo": capacity_by_slo,
        "compliant_goodput_by_slo": compliant_goodput_by_slo,
    }


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "architecture",
        "topology",
        "concurrency",
        "repetitions",
        "run_status",
        "completed_requests",
        "expected_requests",
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
        "strict_goodput_requests_per_second",
        "standard_goodput_requests_per_second",
        "relaxed_goodput_requests_per_second",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    rows: list[dict[str, Any]],
    envelope: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# PAP/PD/DP AIPerf capacity matrix",
        "",
        "| Architecture | Topology | C | Run status | Completed (worst) | "
        "Validation | TTFT p95 ms | ITL p95 ms | Req/s | Strict | "
        "Standard | Relaxed |",
        "| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | "
        "--- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {architecture} | {topology} | {concurrency} | {run_status} | "
            "{completed} | {validation} | {ttft} | {itl} | {throughput} | "
            "{strict} | {standard} | {relaxed} |".format(
                architecture=row["architecture"].upper(),
                topology=row["topology"],
                concurrency=row["concurrency"],
                run_status=_run_status_label(row),
                completed=_completion_label(row),
                validation=_validation_label(row),
                ttft=_fmt(row["ttft_p95_ms"]),
                itl=_fmt(row["itl_p95_ms"]),
                throughput=_fmt(row["request_throughput_per_second"], 3),
                strict=_slo_label(row, "strict"),
                standard=_slo_label(row, "standard"),
                relaxed=_slo_label(row, "relaxed"),
            )
        )
    lines.extend(
        [
            "",
            "## Tested capacity envelope",
            "",
            "| SLO | Best PAP | PAP C | Best PD | PD C | DP C | "
            "PAP - PD | PAP - DP |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for tier, value in envelope["capacity_by_slo"].items():
        lines.append(
            f"| {tier} | {_fmt(value['best_pap']['topology'], 0)} | "
            f"{_fmt(value['best_pap']['concurrency'], 0)} | "
            f"{_fmt(value['best_pd']['topology'], 0)} | "
            f"{_fmt(value['best_pd']['concurrency'], 0)} | "
            f"{_fmt(value['best_dp']['concurrency'], 0)} | "
            f"{_fmt(value['pap_minus_best_pd'], 0)} | "
            f"{_fmt(value['pap_minus_dp'], 0)} |"
        )
    lines.extend(
        [
            "",
            "## Best compliant request goodput",
            "",
            "Only complete, correct configurations with at least 95% of "
            "requests meeting the tier are eligible.",
            "",
            "| SLO | Best PAP | PAP req/s | PAP C | Best PD | PD req/s | "
            "PD C | DP req/s | DP C | PAP over PD | PAP over DP |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | "
            "---: | ---: | ---: |",
        ]
    )
    for tier, value in envelope["compliant_goodput_by_slo"].items():
        pap = value["best_pap"]
        pd = value["best_pd"]
        dp = value["best_dp"]
        pd_difference = value["pap_over_pd_percent"]
        dp_difference = value["pap_over_dp_percent"]
        pd_label = f"{pd_difference:+.1f}%" if pd_difference is not None else "-"
        dp_label = f"{dp_difference:+.1f}%" if dp_difference is not None else "-"
        lines.append(
            f"| {tier} | {_fmt(pap['topology'], 0)} | "
            f"{_fmt(pap['requests_per_second'], 3)} | "
            f"{_fmt(pap['concurrency'], 0)} | {_fmt(pd['topology'], 0)} | "
            f"{_fmt(pd['requests_per_second'], 3)} | "
            f"{_fmt(pd['concurrency'], 0)} | "
            f"{_fmt(dp['requests_per_second'], 3)} | "
            f"{_fmt(dp['concurrency'], 0)} | {pd_label} | {dp_label} |"
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
