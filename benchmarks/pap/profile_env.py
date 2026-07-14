"""Translate a PAP benchmark profile into runner environment values."""

from __future__ import annotations

import argparse
import shlex
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_profile(path: Path) -> dict[str, Any]:
    """Load a TOML benchmark profile."""
    with path.open("rb") as file_obj:
        return tomllib.load(file_obj)


def _section(profile: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = profile.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"profile section {name!r} must be a table")
    return value


def _bool(value: object, name: str) -> str:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return "1" if value else "0"


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_number(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return str(value)


def _devices(value: object, name: str, expected_count: int) -> str:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"{name} must contain {expected_count} devices")
    devices = [_integer(item, name) for item in value]
    if len(set(devices)) != len(devices):
        raise ValueError(f"{name} contains duplicate devices")
    return ",".join(str(device) for device in devices)


def runner_environment(profile: Mapping[str, Any]) -> dict[str, str]:
    """Return the canonical shell inputs for a PAP profile."""
    model = _section(profile, "model")
    workload = _section(profile, "workload")
    corpus = _section(workload, "corpus")
    topology = _section(profile, "topology")
    placement = _section(profile, "placement")
    transport = _section(profile, "transport")
    mps = _section(profile, "mps")
    runtime = _section(profile, "runtime")
    audit = _section(profile, "audit")

    rounds = _integer(workload.get("rounds"), "workload.rounds", minimum=1)
    conversations = _integer(
        workload.get("active_conversations"),
        "workload.active_conversations",
        minimum=1,
    )
    requests = _integer(
        workload.get("requests_per_repetition"),
        "workload.requests_per_repetition",
        minimum=1,
    )
    if requests != rounds * conversations:
        raise ValueError(
            "requests_per_repetition must equal rounds * active_conversations"
        )

    pa_count = _integer(topology.get("pa_count"), "topology.pa_count", minimum=1)
    projection_count = _integer(
        topology.get("projection_count"),
        "topology.projection_count",
        minimum=1,
    )
    topology_name = _text(topology.get("name"), "topology.name")
    if topology_name != f"{pa_count}pa{projection_count}p":
        raise ValueError("topology.name does not match its PA/Projection counts")

    if mps.get("mode") != "static":
        raise ValueError("the canonical PAP runner requires static MPS")
    if transport.get("same_host") is not True:
        raise ValueError("the canonical PAP runner requires same-host transport")

    prefill_devices = _devices(
        placement.get("prefill_devices"),
        "placement.prefill_devices",
        pa_count,
    )
    attention_devices = _devices(
        placement.get("attention_devices"),
        "placement.attention_devices",
        pa_count,
    )
    if attention_devices != prefill_devices:
        raise ValueError("P17 Attention must be colocated with Prefill")

    return {
        "P17_PROFILE_ID": _text(profile.get("profile_id"), "profile_id"),
        "P17_REPETITIONS": str(
            _integer(profile.get("repetitions"), "repetitions", minimum=1)
        ),
        "P17_MODEL_RELATIVE_PATH": _text(
            model.get("relative_path"), "model.relative_path"
        ),
        "P17_CORPUS_RELATIVE_PATH": _text(
            corpus.get("relative_path"), "workload.corpus.relative_path"
        ),
        "PAP_VLLM_DTYPE": _text(model.get("dtype"), "model.dtype"),
        "PAP_TP_SIZE": str(
            _integer(
                model.get("tensor_parallel_size"),
                "model.tensor_parallel_size",
                minimum=1,
            )
        ),
        "MAX_MODEL_LEN": str(
            _integer(model.get("max_model_len"), "model.max_model_len", minimum=1)
        ),
        "MAX_NUM_BATCHED_TOKENS": str(
            _integer(
                model.get("max_num_batched_tokens"),
                "model.max_num_batched_tokens",
                minimum=1,
            )
        ),
        "MAX_NUM_SEQS": str(
            _integer(model.get("max_num_seqs"), "model.max_num_seqs", minimum=1)
        ),
        "PAP_MULTITURN_BLOCK_SIZE": str(
            _integer(model.get("block_size"), "model.block_size", minimum=1)
        ),
        "INPUT_LEN": str(
            _integer(
                workload.get("document_tokens"),
                "workload.document_tokens",
                minimum=1,
            )
        ),
        "OUTPUT_LEN": str(
            _integer(
                workload.get("output_tokens_per_round"),
                "workload.output_tokens_per_round",
                minimum=1,
            )
        ),
        "PAP_MULTITURN_APPEND_TOKENS": str(
            _integer(
                workload.get("append_tokens_per_later_round"),
                "workload.append_tokens_per_later_round",
            )
        ),
        "PAP_MULTITURN_LOAD_ROUNDS": str(rounds),
        "PAP_MULTITURN_LOAD_CONVERSATIONS": str(conversations),
        "PAP_MULTITURN_LOAD_REQUEST_RATE": _positive_number(
            workload.get("request_rate_per_round"),
            "workload.request_rate_per_round",
        ),
        "PAP_TOPOLOGY": topology_name,
        "PAP_PREFILL_GPUS": prefill_devices,
        "PAP_PROJECTION_GPUS": _devices(
            placement.get("projection_devices"),
            "placement.projection_devices",
            projection_count,
        ),
        "PAP_OFFLOAD_EXEC_TRANSPORT": _text(
            transport.get("offload_exec"), "transport.offload_exec"
        ),
        "PAP_OFFLOAD_KV_TRANSPORT": _text(
            transport.get("offload_kv"), "transport.offload_kv"
        ),
        "PAP_ROUTING_POLICY": _text(
            topology.get("routing_policy"), "topology.routing_policy"
        ),
        "PAP_PREFILL_MPS_PERCENT": str(
            _integer(
                mps.get("prefill_requested_percent"),
                "mps.prefill_requested_percent",
                minimum=1,
            )
        ),
        "PAP_ATTENTION_MPS_PERCENT": str(
            _integer(
                mps.get("attention_requested_percent"),
                "mps.attention_requested_percent",
                minimum=1,
            )
        ),
        "PAP_STATIC_PREFILL_CHUNKS": str(
            _integer(mps.get("prefill_chunks"), "mps.prefill_chunks", minimum=1)
        ),
        "PAP_STATIC_ATTENTION_CHUNKS": str(
            _integer(
                mps.get("attention_chunks"), "mps.attention_chunks", minimum=1
            )
        ),
        "PAP_STATIC_PREFILL_EXPECTED_SMS": str(
            _integer(
                mps.get("prefill_visible_sms"),
                "mps.prefill_visible_sms",
                minimum=1,
            )
        ),
        "PAP_STATIC_ATTENTION_EXPECTED_SMS": str(
            _integer(
                mps.get("attention_visible_sms"),
                "mps.attention_visible_sms",
                minimum=1,
            )
        ),
        "PAP_DIRECT_MAILBOX_OUTPUT": _bool(
            runtime.get("direct_mailbox_output"), "runtime.direct_mailbox_output"
        ),
        "PAP_LOCAL_FAST_STREAM_ORDERED": _bool(
            runtime.get("local_fast_stream_ordered"),
            "runtime.local_fast_stream_ordered",
        ),
        "PAP_LOCAL_FAST_SLOT_COUNT": str(
            _integer(
                runtime.get("local_fast_slot_count"),
                "runtime.local_fast_slot_count",
                minimum=1,
            )
        ),
        "PAP_DECODE_SLOT_PLAN_CACHE_LIMIT": str(
            _integer(
                runtime.get("decode_slot_plan_cache_limit"),
                "runtime.decode_slot_plan_cache_limit",
                minimum=1,
            )
        ),
        "PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND": _bool(
            runtime.get("offload_exec_direct_qkv_send"),
            "runtime.offload_exec_direct_qkv_send",
        ),
        "PAP_LOCAL_FAST_BATCH_PLAN": _bool(
            runtime.get("local_fast_batch_plan"), "runtime.local_fast_batch_plan"
        ),
        "PAP_ENABLE_PROMPT_TOKENS_DETAILS": _bool(
            runtime.get("prompt_token_details"), "runtime.prompt_token_details"
        ),
        "PAP_PREFIX_CACHE_AUDIT": _bool(
            runtime.get("prefix_cache_audit"), "runtime.prefix_cache_audit"
        ),
        "PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS": str(
            _integer(
                runtime.get("unified_kv_decode_capacity_tokens"),
                "runtime.unified_kv_decode_capacity_tokens",
                minimum=1,
            )
        ),
        "PAP_BENCH_STRICT_CORRECTNESS_AUDIT": _bool(
            audit.get("strict_correctness"), "audit.strict_correctness"
        ),
        "PAP_BENCH_SESSION_DRAIN_TIMEOUT": str(
            _integer(
                audit.get("session_drain_timeout_seconds"),
                "audit.session_drain_timeout_seconds",
                minimum=1,
            )
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print shell assignments for a PAP benchmark profile"
    )
    parser.add_argument("profile", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    values = runner_environment(load_profile(args.profile))
    for name, value in values.items():
        print(f"{name}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
