# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common contracts for the PAP/PD multi-turn long-context benchmark."""

import argparse
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence


KV_CAPACITY_RE = re.compile(
    r"GPU KV cache size:\s*([0-9][0-9,]*)\s+tokens"
)


@dataclass(frozen=True)
class TokenAccounting:
    prompt_tokens: int
    local_reused_tokens: int
    remote_loaded_tokens: int
    recomputed_tokens: int


@dataclass(frozen=True)
class CapacityAdmission:
    reported_capacity_tokens_by_service: Mapping[str, int]
    required_services: tuple[str, ...]
    usable_kv_token_capacity: int
    active_conversations: int
    max_rendered_context_tokens_per_conversation: int
    required_live_tokens: int
    safety_fraction: float
    budget_tokens: int
    decision: str
    reason: str
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reported_capacity_tokens_by_service",
            MappingProxyType(dict(self.reported_capacity_tokens_by_service)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "reported_capacity_tokens_by_service": dict(
                self.reported_capacity_tokens_by_service
            ),
            "required_services": list(self.required_services),
            "usable_kv_token_capacity": self.usable_kv_token_capacity,
            "active_conversations": self.active_conversations,
            "max_rendered_context_tokens_per_conversation": (
                self.max_rendered_context_tokens_per_conversation
            ),
            "required_live_tokens": self.required_live_tokens,
            "safety_fraction": self.safety_fraction,
            "budget_tokens": self.budget_tokens,
            "decision": self.decision,
            "reason": self.reason,
        }


def parse_kv_capacities(log_paths: Mapping[str, Path]) -> dict[str, int]:
    if not log_paths:
        raise ValueError("at least one service log is required")

    capacities: dict[str, int] = {}
    for service, log_path in log_paths.items():
        try:
            log_text = Path(log_path).read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"cannot decode KV capacity log for service {service!r} "
                f"as UTF-8: {log_path}"
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"cannot read KV capacity log for service {service!r}: "
                f"{log_path}"
            ) from exc

        matches = {
            int(match.group(1).replace(",", ""))
            for match in KV_CAPACITY_RE.finditer(log_text)
        }
        if not matches:
            raise ValueError(
                f"missing GPU KV cache capacity for service {service!r} "
                f"in {log_path}"
            )
        if len(matches) != 1:
            values = ", ".join(str(value) for value in sorted(matches))
            raise ValueError(
                f"conflicting GPU KV cache capacities for service "
                f"{service!r}: {values}"
            )

        capacity = next(iter(matches))
        if capacity <= 0:
            raise ValueError(
                f"non-positive GPU KV cache capacity for service "
                f"{service!r}: {capacity}"
            )
        capacities[service] = capacity

    return capacities


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def decide_capacity_admission(
    capacities: Mapping[str, int],
    required_services: Sequence[str],
    active_conversations: int,
    max_rendered_context_tokens: int,
    safety_fraction: float = 0.70,
) -> CapacityAdmission:
    required = tuple(required_services)
    if not required:
        raise ValueError("at least one required service is required")
    if any(not service for service in required):
        raise ValueError("required service names must be non-empty")

    reported = dict(capacities)
    for service, capacity in reported.items():
        if not service:
            raise ValueError("capacity service names must be non-empty")
        _require_positive_int(capacity, f"capacity for service {service!r}")

    missing = [service for service in required if service not in reported]
    if missing:
        raise ValueError(
            "missing capacities for required services: " + ", ".join(missing)
        )

    _require_positive_int(active_conversations, "active_conversations")
    _require_positive_int(
        max_rendered_context_tokens, "max_rendered_context_tokens"
    )
    if isinstance(safety_fraction, bool):
        raise ValueError("safety_fraction must be greater than 0 and at most 1")
    try:
        fraction_as_float = float(safety_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "safety_fraction must be greater than 0 and at most 1"
        ) from exc
    if not math.isfinite(fraction_as_float) or not (
        0 < fraction_as_float <= 1
    ):
        raise ValueError("safety_fraction must be greater than 0 and at most 1")

    fraction_decimal = Decimal(str(safety_fraction))
    usable_capacity = min(reported[service] for service in required)
    required_live_tokens = (
        active_conversations * max_rendered_context_tokens
    )
    budget_tokens = math.floor(
        Decimal(usable_capacity) * fraction_decimal
    )
    admitted = required_live_tokens <= budget_tokens
    decision = "admitted" if admitted else "admission-limited"
    comparison = "within" if admitted else "exceed"
    reason = (
        f"Required live tokens ({required_live_tokens}) {comparison} the "
        f"safety budget ({budget_tokens}) derived from required-service "
        f"capacity {usable_capacity} at fraction {fraction_as_float}."
    )

    return CapacityAdmission(
        reported_capacity_tokens_by_service=reported,
        required_services=required,
        usable_kv_token_capacity=usable_capacity,
        active_conversations=active_conversations,
        max_rendered_context_tokens_per_conversation=(
            max_rendered_context_tokens
        ),
        required_live_tokens=required_live_tokens,
        safety_fraction=fraction_as_float,
        budget_tokens=budget_tokens,
        decision=decision,
        reason=reason,
    )


def _format_accounting(accounting: TokenAccounting) -> str:
    return (
        f"prompt_tokens={accounting.prompt_tokens}, "
        f"local_reused_tokens={accounting.local_reused_tokens}, "
        f"remote_loaded_tokens={accounting.remote_loaded_tokens}, "
        f"recomputed_tokens={accounting.recomputed_tokens}"
    )


def validate_token_accounting(accounting: TokenAccounting) -> None:
    values = (
        accounting.prompt_tokens,
        accounting.local_reused_tokens,
        accounting.remote_loaded_tokens,
        accounting.recomputed_tokens,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in values
    ):
        raise ValueError(
            "token accounting counts must be non-negative integers: "
            + _format_accounting(accounting)
        )

    accounted_prompt_tokens = (
        accounting.local_reused_tokens
        + accounting.remote_loaded_tokens
        + accounting.recomputed_tokens
    )
    if accounted_prompt_tokens != accounting.prompt_tokens:
        raise ValueError(
            "exclusive prompt token accounting mismatch: "
            + _format_accounting(accounting)
        )


def calculate_tpot_s(
    turn_latency_s: float, ttft_s: float, output_tokens: int
) -> float:
    if not math.isfinite(turn_latency_s) or turn_latency_s < 0:
        raise ValueError("turn_latency_s must be finite and non-negative")
    if not math.isfinite(ttft_s) or ttft_s < 0:
        raise ValueError("ttft_s must be finite and non-negative")
    if ttft_s > turn_latency_s:
        raise ValueError("ttft_s cannot exceed turn_latency_s")
    _require_positive_int(output_tokens, "output_tokens")

    return (turn_latency_s - ttft_s) / max(output_tokens - 1, 1)


def stable_cell_id(
    matrix: str,
    lane: str,
    context_tokens: int,
    decode_tokens: int,
    rounds: int,
    active_conversations: int,
    repetition: int,
) -> str:
    if not matrix or not lane:
        raise ValueError("matrix and lane must be non-empty")
    _require_positive_int(context_tokens, "context_tokens")
    if (
        isinstance(decode_tokens, bool)
        or not isinstance(decode_tokens, int)
        or decode_tokens < 0
    ):
        raise ValueError(
            f"decode_tokens must be a non-negative integer, got "
            f"{decode_tokens!r}"
        )
    _require_positive_int(rounds, "rounds")
    _require_positive_int(active_conversations, "active_conversations")
    _require_positive_int(repetition, "repetition")

    return (
        f"MT-{matrix}-{lane}-ctx{context_tokens}-d{decode_tokens}-r{rounds}"
        f"-c{active_conversations}-rep{repetition}"
    )


def _parse_service_logs(specifications: Sequence[str]) -> dict[str, Path]:
    log_paths: dict[str, Path] = {}
    for specification in specifications:
        service, separator, raw_path = specification.partition("=")
        if not separator or not service or not raw_path:
            raise ValueError(
                "--service-log values must use the form NAME=PATH"
            )
        if service in log_paths:
            raise ValueError(f"duplicate --service-log service: {service}")
        log_paths[service] = Path(raw_path)
    return log_paths


def _atomic_write_json(output_path: Path, payload: Mapping[str, object]) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capacity_parser = subparsers.add_parser(
        "capacity", help="parse service capacities and decide admission"
    )
    capacity_parser.add_argument(
        "--service-log", action="append", required=True, metavar="NAME=PATH"
    )
    capacity_parser.add_argument(
        "--required-service", action="append", required=True, metavar="NAME"
    )
    capacity_parser.add_argument(
        "--active-conversations", required=True, type=int
    )
    capacity_parser.add_argument(
        "--max-rendered-context-tokens",
        "--max-rendered-context-tokens-per-conversation",
        dest="max_rendered_context_tokens",
        required=True,
        type=int,
    )
    capacity_parser.add_argument(
        "--safety-fraction", default=0.70, type=float
    )
    capacity_parser.add_argument("--output", required=True, type=Path)
    return parser


def _run_capacity_command(args: argparse.Namespace) -> int:
    try:
        log_paths = _parse_service_logs(args.service_log)
        capacities = parse_kv_capacities(log_paths)
        admission = decide_capacity_admission(
            capacities=capacities,
            required_services=args.required_service,
            active_conversations=args.active_conversations,
            max_rendered_context_tokens=args.max_rendered_context_tokens,
            safety_fraction=args.safety_fraction,
        )
        _atomic_write_json(args.output, admission.to_dict())
    except (OSError, ValueError) as exc:
        print(f"capacity: error: {exc}", file=sys.stderr)
        return 2

    return 0 if admission.decision == "admitted" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "capacity":
        return _run_capacity_command(args)
    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
