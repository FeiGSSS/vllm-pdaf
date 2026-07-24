# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for summarizing PAP OFFLOAD_EXEC trace logs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TraceStat:
    count: int
    mean: float
    median: float
    p90: float
    p99: float
    max: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "p90": self.p90,
            "p99": self.p99,
            "max": self.max,
        }


_PROJECTION_TRACE_RE = re.compile(
    r"projection trace .*?(?:batches=(\d+) )?calls=(\d+) "
    r"send_ms=([0-9.]+) trigger_ms=([0-9.]+) "
    r"(?:yield_ms=([0-9.]+) )?recv_ms=([0-9.]+) total_ms=([0-9.]+)"
)
_PROJECTION_TIMELINE_RE = re.compile(
    r"projection timeline .*?batches=(\d+) calls=(\d+) "
    r"pre_attn_compute_ms=([0-9.]+) send_ms=([0-9.]+) "
    r"trigger_ms=([0-9.]+) yield_ms=([0-9.]+) recv_ms=([0-9.]+) "
    r"o_proj_ms=([0-9.]+) remote_total_ms=([0-9.]+) "
    r"self_attn_total_ms=([0-9.]+)"
)
_PROJECTION_LAYER_TIMELINE_RE = re.compile(
    r"projection layer timeline .*?input_norm_ms=([0-9.]+) "
    r"self_attn_ms=([0-9.]+) post_attention_layernorm_ms=([0-9.]+) "
    r"mlp_ms=([0-9.]+) layer_total_ms=([0-9.]+)"
)
_PROJECTION_CRITICAL_PATH_RE = re.compile(
    r"projection critical path .*?calls=(\d+) "
    r"input_norm_ms=([0-9.]+) qkv_ms=([0-9.]+) send_ms=([0-9.]+) "
    r"recv_ms=([0-9.]+) o_proj_ms=([0-9.]+) post_norm_ms=([0-9.]+) "
    r"mlp_ms=([0-9.]+) layer_total_ms=([0-9.]+) gaps_ms=([0-9.]+)"
)
_PROJECTION_MODEL_FORWARD_RE = re.compile(
    r"projection model forward .*?num_tokens=(\d+) "
    r"model_forward_ms=([0-9.]+)"
)
_PROJECTION_LOGITS_RE = re.compile(
    r"projection logits .*?num_tokens=(\d+) logits_ms=([0-9.]+)"
)
_PROJECTION_RUNNER_FORWARD_RE = re.compile(
    r"projection runner forward .*?num_tokens=(\d+) "
    r"forward_and_postprocess_ms=([0-9.]+)"
)
_PROJECTION_RUNNER_FORWARD_DETAIL_RE = re.compile(
    r"projection runner forward detail .*?num_tokens=(\d+) "
    r"input_prep_ms=([0-9.]+) metadata_ms=([0-9.]+) "
    r"preprocess_ms=([0-9.]+) model_forward_ms=([0-9.]+) "
    r"hidden_slice_ms=([0-9.]+) logits_ms=([0-9.]+) "
    r"postprocess_tail_ms=([0-9.]+) total_ms=([0-9.]+)"
)
_PROJECTION_WORKER_EXEC_RE = re.compile(
    r"projection worker execute_model .*?num_tokens=(\d+) exec_ms=([0-9.]+)"
)
_PROJECTION_WORKER_SAMPLE_RE = re.compile(
    r"projection worker sample_tokens .*?sample_ms=([0-9.]+)"
)
_PROJECTION_ENGINE_STEP_RE = re.compile(
    r"projection engine step .*?num_gen=(\d+) .*?"
    r"sched_ms=([0-9.]+) exec_and_sample_ms=([0-9.]+) "
    r"postprocess_ms=([0-9.]+) step_ms=([0-9.]+)"
)
_PROJECTION_FIRST_OUTPUT_RE = re.compile(
    r"projection first output .*?generated_tokens=(\d+) "
    r"sched_ms=([0-9.]+) exec_and_sample_ms=([0-9.]+) "
    r"scheduler_update_ms=([0-9.]+) step_to_first_output_ms=([0-9.]+)"
)
_ATTENTION_TRACE_RE = re.compile(
    r"attention mailbox batch trace .* calls=(\d+) recv_qkv_ms=([0-9.]+) "
    r"compute_ms=([0-9.]+) send_output_ms=([0-9.]+) total_ms=([0-9.]+)"
)
_ATTENTION_RECV_DETAIL_RE = re.compile(
    r"recv_wait_ms=([0-9.]+) recv_read_ms=([0-9.]+) "
    r"recv_materialize_ms=([0-9.]+) recv_transfer_ms=([0-9.]+) "
    r"(?:recv_wait_other_ms=([0-9.]+) )?"
    r"recv_unaccounted_ms=([0-9.]+)"
)
_ATTENTION_COMPUTE_DETAIL_RE = re.compile(
    r"append_kv_ms=([0-9.]+) pack_ms=([0-9.]+) "
    r"sdpa_ms=([0-9.]+) reshape_ms=([0-9.]+)"
    r"(?: paged_metadata_ms=([0-9.]+))?"
    r"(?: paged_flash_ms=([0-9.]+))?"
    r"(?: fallback_ms=[0-9.]+)?"
    r"(?: shape_lookup_ms=([0-9.]+))?"
    r"(?: qkv_split_ms=([0-9.]+))?"
    r"(?: query_move_ms=([0-9.]+))?"
    r"(?: query_cat_ms=([0-9.]+))?"
    r"(?: append_lock_wait_ms=([0-9.]+))?"
    r"(?: append_prepare_ms=([0-9.]+))?"
    r"(?: append_record_ms=([0-9.]+))?"
    r"(?: append_tensor_ms=([0-9.]+))?"
    r"(?: append_copy_ms=([0-9.]+))?"
    r"(?: append_state_ms=([0-9.]+))?"
    r"(?: metadata_build_ms=([0-9.]+))?"
    r"(?: paged_flash_kernel_ms=([0-9.]+))?"
    r"(?: attention_output_reshape_ms=([0-9.]+))?"
    r"(?: compute_unaccounted_ms=([0-9.]+))?"
)
_PROJECTION_CORRELATION_RE = re.compile(
    r"batch_keys=(\S+) "
    r"(?:route_rows=(\S+) route_kv_tokens=(\S+) )?"
    r"send_done_ns=(\d+) yield_start_ns=(\d+) "
    r"yield_end_ns=(\d+) recv_done_ns=(\d+)"
)
_ATTENTION_CORRELATION_OLD_RE = re.compile(
    r"batch_key=(\S+) "
    r"recv_done_ns=(\d+) "
    r"compute_done_ns=(\d+) "
    r"send_done_ns=(\d+)"
)
_ATTENTION_CORRELATION_NEW_RE = re.compile(
    r"batch_key=(\S+) "
    r"recv_done_ns=(\d+) compute_done_ns=(\d+) send_done_ns=(\d+) "
    r"recv_start_ns=(\d+) pre_compute_start_ns=(\d+) "
    r"pre_compute_done_ns=(\d+) paged_flash_done_ns=(\d+) reshape_done_ns=(\d+) "
    r"send_start_ns=(\d+)"
)
_MAILBOX_SEND_RE = re.compile(
    r"PAP NIXL mailbox send trace actor=(\S+) .* kind=(\S+) nbytes=(\d+) "
    r"queue_ms=([0-9.]+) publish_ms=([0-9.]+) pack_ms=([0-9.]+) "
    r"(?:slot_wait_ms=([0-9.]+) )?copy_ms=([0-9.]+) "
    r"(?:payload_ms=([0-9.]+) )?(?:piggyback_ms=([0-9.]+) )?"
    r"notify_ms=([0-9.]+) "
    r"(?:write_ms=([0-9.]+) )?"
    r"(?:write_prepare_ms=([0-9.]+) )?"
    r"(?:write_transfer_ms=([0-9.]+) )?"
    r"(?:write_polls=(\d+) )?ack_wait_ms=([0-9.]+) "
    r"total_ms=([0-9.]+)"
)
_MAILBOX_INLINE_SEND_RE = re.compile(
    r"PAP NIXL mailbox inline send trace actor=(\S+) .* kind=(\S+) nbytes=(\d+) "
    r"publish_ms=([0-9.]+) pack_ms=([0-9.]+) "
    r"slot_wait_ms=([0-9.]+) copy_ms=([0-9.]+) "
    r"payload_ms=([0-9.]+) piggyback_ms=([0-9.]+) "
    r"notify_ms=([0-9.]+) write_ms=([0-9.]+) "
    r"write_prepare_ms=([0-9.]+) write_transfer_ms=([0-9.]+) "
    r"write_polls=(\d+) total_ms=([0-9.]+)"
)
_MAILBOX_READ_RE = re.compile(
    r"PAP NIXL mailbox read trace actor=(\S+) .* kind=(\S+) nbytes=(\d+) "
    r"prepare_ms=([0-9.]+) "
    r"(?:slot_wait_ms=([0-9.]+) )?"
    r"(?:handle_prepare_ms=([0-9.]+) )?"
    r"transfer_ms=([0-9.]+) transfer_polls=(\d+) "
    r"materialize_ms=([0-9.]+) total_ms=([0-9.]+)"
)
_MAILBOX_WAIT_RE = re.compile(
    r"PAP NIXL mailbox recv wait trace actor=(\S+) .* kind=(\S+) "
    r"requested_msg_id=.* wait_ms=([0-9.]+)"
)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    index = min(
        len(sorted_values) - 1,
        int(round((len(sorted_values) - 1) * percentile / 100.0)),
    )
    return sorted_values[index]


def _stat(values: Iterable[float]) -> TraceStat:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return TraceStat(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return TraceStat(
        count=len(sorted_values),
        mean=statistics.mean(sorted_values),
        median=statistics.median(sorted_values),
        p90=_percentile(sorted_values, 90),
        p99=_percentile(sorted_values, 99),
        max=max(sorted_values),
    )


def _add_grouped_value(
    grouped: dict[str, dict[str, list[float]]],
    group: str,
    field: str,
    value: float,
) -> None:
    grouped.setdefault(group, {}).setdefault(field, []).append(float(value))


def _mailbox_actor_group(actor: str) -> str:
    if actor.startswith("projection-"):
        return "projection"
    return actor


def _mailbox_kind_group(actor: str, kind: str) -> str:
    return f"{_mailbox_actor_group(actor)}:{kind}"


def summarize_pap_trace_logs(
    log_dir: str | Path,
    *,
    max_total_ms: float | None = 10.0,
) -> dict[str, object]:
    """Summarize PAP trace timings from a benchmark service_logs directory."""

    path = Path(log_dir)
    projection: dict[str, list[float]] = {
        "batches": [],
        "calls": [],
        "send_ms": [],
        "trigger_ms": [],
        "yield_ms": [],
        "recv_ms": [],
        "gap_ms": [],
        "total_ms": [],
    }
    projection_timeline: dict[str, list[float]] = {
        "batches": [],
        "calls": [],
        "pre_attn_compute_ms": [],
        "send_ms": [],
        "trigger_ms": [],
        "yield_ms": [],
        "recv_ms": [],
        "o_proj_ms": [],
        "remote_total_ms": [],
        "self_attn_total_ms": [],
    }
    projection_layer_timeline: dict[str, list[float]] = {
        "input_norm_ms": [],
        "self_attn_ms": [],
        "post_attention_layernorm_ms": [],
        "mlp_ms": [],
        "layer_total_ms": [],
    }
    projection_critical_path: dict[str, list[float]] = {
        "calls": [],
        "input_norm_ms": [],
        "qkv_ms": [],
        "send_ms": [],
        "recv_ms": [],
        "o_proj_ms": [],
        "post_norm_ms": [],
        "mlp_ms": [],
        "layer_total_ms": [],
        "gaps_ms": [],
    }
    projection_model_forward: dict[str, list[float]] = {
        "num_tokens": [],
        "model_forward_ms": [],
    }
    projection_logits: dict[str, list[float]] = {
        "num_tokens": [],
        "logits_ms": [],
    }
    projection_runner_forward: dict[str, list[float]] = {
        "num_tokens": [],
        "forward_and_postprocess_ms": [],
    }
    projection_runner_forward_detail: dict[str, list[float]] = {
        "num_tokens": [],
        "input_prep_ms": [],
        "metadata_ms": [],
        "preprocess_ms": [],
        "model_forward_ms": [],
        "hidden_slice_ms": [],
        "logits_ms": [],
        "postprocess_tail_ms": [],
        "total_ms": [],
    }
    projection_worker_exec: dict[str, list[float]] = {
        "num_tokens": [],
        "exec_ms": [],
    }
    projection_worker_sample: dict[str, list[float]] = {
        "sample_ms": [],
    }
    projection_engine_step: dict[str, list[float]] = {
        "num_gen": [],
        "sched_ms": [],
        "exec_and_sample_ms": [],
        "postprocess_ms": [],
        "step_ms": [],
    }
    projection_first_output: dict[str, list[float]] = {
        "generated_tokens": [],
        "sched_ms": [],
        "exec_and_sample_ms": [],
        "scheduler_update_ms": [],
        "step_to_first_output_ms": [],
    }
    attention: dict[str, list[float]] = {
        "calls": [],
        "recv_qkv_ms": [],
        "compute_ms": [],
        "send_output_ms": [],
        "total_ms": [],
        "recv_wait_ms": [],
        "recv_read_ms": [],
        "recv_materialize_ms": [],
        "recv_transfer_ms": [],
        "recv_wait_other_ms": [],
        "recv_unaccounted_ms": [],
        "append_kv_ms": [],
        "pack_ms": [],
        "sdpa_ms": [],
        "reshape_ms": [],
        "paged_metadata_ms": [],
        "paged_flash_ms": [],
        "shape_lookup_ms": [],
        "qkv_split_ms": [],
        "query_move_ms": [],
        "query_cat_ms": [],
        "append_lock_wait_ms": [],
        "append_prepare_ms": [],
        "append_record_ms": [],
        "append_tensor_ms": [],
        "append_copy_ms": [],
        "append_state_ms": [],
        "metadata_build_ms": [],
        "paged_flash_kernel_ms": [],
        "attention_output_reshape_ms": [],
        "compute_unaccounted_ms": [],
    }
    mailbox_send: dict[str, dict[str, list[float]]] = {}
    mailbox_read: dict[str, dict[str, list[float]]] = {}
    mailbox_send_by_kind: dict[str, dict[str, list[float]]] = {}
    mailbox_read_by_kind: dict[str, dict[str, list[float]]] = {}
    mailbox_wait_by_kind: dict[str, dict[str, list[float]]] = {}
    projection_correlation_entries: list[
        tuple[list[str], list[int], list[int], int, int, int]
    ] = []
    attention_timestamps_by_key: dict[str, dict[str, int]] = {}

    for log_path in sorted(path.glob("*.log")):
        for line in log_path.read_text(errors="ignore").splitlines():
            if match := _PROJECTION_TIMELINE_RE.search(line):
                (
                    batches,
                    calls,
                    pre_attn_compute_ms,
                    send_ms,
                    trigger_ms,
                    yield_ms,
                    recv_ms,
                    o_proj_ms,
                    remote_total_ms,
                    self_attn_total_ms,
                ) = match.groups()
                self_attn_total = float(self_attn_total_ms)
                if max_total_ms is None or self_attn_total <= max_total_ms:
                    for field, value in (
                        ("batches", batches),
                        ("calls", calls),
                        ("pre_attn_compute_ms", pre_attn_compute_ms),
                        ("send_ms", send_ms),
                        ("trigger_ms", trigger_ms),
                        ("yield_ms", yield_ms),
                        ("recv_ms", recv_ms),
                        ("o_proj_ms", o_proj_ms),
                        ("remote_total_ms", remote_total_ms),
                        ("self_attn_total_ms", self_attn_total_ms),
                    ):
                        projection_timeline[field].append(float(value))
                continue
            if match := _PROJECTION_LAYER_TIMELINE_RE.search(line):
                (
                    input_norm_ms,
                    self_attn_ms,
                    post_attention_layernorm_ms,
                    mlp_ms,
                    layer_total_ms,
                ) = match.groups()
                layer_total = float(layer_total_ms)
                if max_total_ms is None or layer_total <= max_total_ms:
                    for field, value in (
                        ("input_norm_ms", input_norm_ms),
                        ("self_attn_ms", self_attn_ms),
                        ("post_attention_layernorm_ms", post_attention_layernorm_ms),
                        ("mlp_ms", mlp_ms),
                        ("layer_total_ms", layer_total_ms),
                    ):
                        projection_layer_timeline[field].append(float(value))
                continue
            if match := _PROJECTION_CRITICAL_PATH_RE.search(line):
                (
                    calls,
                    input_norm_ms,
                    qkv_ms,
                    send_ms,
                    recv_ms,
                    o_proj_ms,
                    post_norm_ms,
                    mlp_ms,
                    layer_total_ms,
                    gaps_ms,
                ) = match.groups()
                layer_total = float(layer_total_ms)
                if max_total_ms is None or layer_total <= max_total_ms:
                    for field, value in (
                        ("calls", calls),
                        ("input_norm_ms", input_norm_ms),
                        ("qkv_ms", qkv_ms),
                        ("send_ms", send_ms),
                        ("recv_ms", recv_ms),
                        ("o_proj_ms", o_proj_ms),
                        ("post_norm_ms", post_norm_ms),
                        ("mlp_ms", mlp_ms),
                        ("layer_total_ms", layer_total_ms),
                        ("gaps_ms", gaps_ms),
                    ):
                        projection_critical_path[field].append(float(value))
                continue
            if match := _PROJECTION_MODEL_FORWARD_RE.search(line):
                num_tokens, model_forward_ms = match.groups()
                model_forward = float(model_forward_ms)
                if max_total_ms is None or model_forward <= max_total_ms * 100:
                    projection_model_forward["num_tokens"].append(float(num_tokens))
                    projection_model_forward["model_forward_ms"].append(model_forward)
                continue
            if match := _PROJECTION_LOGITS_RE.search(line):
                num_tokens, logits_ms = match.groups()
                logits = float(logits_ms)
                if max_total_ms is None or logits <= max_total_ms * 100:
                    projection_logits["num_tokens"].append(float(num_tokens))
                    projection_logits["logits_ms"].append(logits)
                continue
            if match := _PROJECTION_RUNNER_FORWARD_RE.search(line):
                num_tokens, forward_ms = match.groups()
                forward = float(forward_ms)
                if max_total_ms is None or forward <= max_total_ms * 100:
                    projection_runner_forward["num_tokens"].append(float(num_tokens))
                    projection_runner_forward["forward_and_postprocess_ms"].append(forward)
                continue
            if match := _PROJECTION_RUNNER_FORWARD_DETAIL_RE.search(line):
                (
                    num_tokens,
                    input_prep_ms,
                    metadata_ms,
                    preprocess_ms,
                    model_forward_ms,
                    hidden_slice_ms,
                    logits_ms,
                    postprocess_tail_ms,
                    total_ms,
                ) = match.groups()
                total = float(total_ms)
                if max_total_ms is None or total <= max_total_ms * 100:
                    for field, value in (
                        ("num_tokens", num_tokens),
                        ("input_prep_ms", input_prep_ms),
                        ("metadata_ms", metadata_ms),
                        ("preprocess_ms", preprocess_ms),
                        ("model_forward_ms", model_forward_ms),
                        ("hidden_slice_ms", hidden_slice_ms),
                        ("logits_ms", logits_ms),
                        ("postprocess_tail_ms", postprocess_tail_ms),
                        ("total_ms", total_ms),
                    ):
                        projection_runner_forward_detail[field].append(float(value))
                continue
            if match := _PROJECTION_WORKER_EXEC_RE.search(line):
                num_tokens, exec_ms = match.groups()
                exec_val = float(exec_ms)
                if max_total_ms is None or exec_val <= max_total_ms * 100:
                    projection_worker_exec["num_tokens"].append(float(num_tokens))
                    projection_worker_exec["exec_ms"].append(exec_val)
                continue
            if match := _PROJECTION_WORKER_SAMPLE_RE.search(line):
                sample_ms, = match.groups()
                sample_val = float(sample_ms)
                if max_total_ms is None or sample_val <= max_total_ms * 100:
                    projection_worker_sample["sample_ms"].append(sample_val)
                continue
            if match := _PROJECTION_ENGINE_STEP_RE.search(line):
                num_gen, sched_ms, exec_and_sample_ms, postprocess_ms, step_ms = (
                    match.groups()
                )
                step_val = float(step_ms)
                if max_total_ms is None or step_val <= max_total_ms * 100:
                    projection_engine_step["num_gen"].append(float(num_gen))
                    projection_engine_step["sched_ms"].append(float(sched_ms))
                    projection_engine_step["exec_and_sample_ms"].append(
                        float(exec_and_sample_ms)
                    )
                    projection_engine_step["postprocess_ms"].append(
                        float(postprocess_ms)
                    )
                    projection_engine_step["step_ms"].append(step_val)
                continue
            if match := _PROJECTION_FIRST_OUTPUT_RE.search(line):
                (
                    generated_tokens,
                    sched_ms,
                    exec_and_sample_ms,
                    scheduler_update_ms,
                    step_to_first_output_ms,
                ) = match.groups()
                first_output = float(step_to_first_output_ms)
                if max_total_ms is None or first_output <= max_total_ms * 100:
                    for field, value in (
                        ("generated_tokens", generated_tokens),
                        ("sched_ms", sched_ms),
                        ("exec_and_sample_ms", exec_and_sample_ms),
                        ("scheduler_update_ms", scheduler_update_ms),
                        ("step_to_first_output_ms", step_to_first_output_ms),
                    ):
                        projection_first_output[field].append(float(value))
                continue
            if match := _PROJECTION_TRACE_RE.search(line):
                (
                    batches,
                    calls,
                    send_ms,
                    trigger_ms,
                    yield_ms,
                    recv_ms,
                    total_ms,
                ) = match.groups()
                batches_value = int(batches) if batches is not None else 0
                calls_value = int(calls)
                send_ms = float(send_ms)
                trigger_ms = float(trigger_ms)
                yield_ms = float(yield_ms or 0.0)
                recv_ms = float(recv_ms)
                total_ms = float(total_ms)
                if max_total_ms is None or total_ms <= max_total_ms:
                    projection["batches"].append(batches_value)
                    projection["calls"].append(calls_value)
                    projection["send_ms"].append(send_ms)
                    projection["trigger_ms"].append(trigger_ms)
                    projection["yield_ms"].append(yield_ms)
                    projection["recv_ms"].append(recv_ms)
                    projection["gap_ms"].append(
                        max(0.0, total_ms - send_ms - trigger_ms - yield_ms - recv_ms)
                    )
                    projection["total_ms"].append(total_ms)
                    if correlation := _PROJECTION_CORRELATION_RE.search(line):
                        (
                            batch_keys,
                            route_rows,
                            route_kv_tokens,
                            send_done_ns,
                            _yield_start_ns,
                            yield_end_ns,
                            recv_done_ns,
                        ) = correlation.groups()
                        projection_correlation_entries.append(
                            (
                                batch_keys.split("|"),
                                (
                                    [int(value) for value in route_rows.split("|")]
                                    if route_rows
                                    else []
                                ),
                                (
                                    [
                                        int(value)
                                        for value in route_kv_tokens.split("|")
                                    ]
                                    if route_kv_tokens
                                    else []
                                ),
                                int(send_done_ns),
                                int(yield_end_ns),
                                int(recv_done_ns),
                            )
                        )
                continue
            if match := _ATTENTION_TRACE_RE.search(line):
                (
                    calls,
                    recv_ms,
                    compute_ms,
                    send_ms,
                    total_ms,
                ) = match.groups()
                calls = int(calls)
                recv_ms, compute_ms, send_ms, total_ms = map(
                    float, (recv_ms, compute_ms, send_ms, total_ms)
                )
                if max_total_ms is None or total_ms <= max_total_ms:
                    attention["calls"].append(calls)
                    attention["recv_qkv_ms"].append(recv_ms)
                    attention["compute_ms"].append(compute_ms)
                    attention["send_output_ms"].append(send_ms)
                    attention["total_ms"].append(total_ms)
                    if recv_detail := _ATTENTION_RECV_DETAIL_RE.search(line):
                        (
                            recv_wait_ms,
                            recv_read_ms,
                            recv_materialize_ms,
                            recv_transfer_ms,
                            recv_wait_other_ms,
                            recv_unaccounted_ms,
                        ) = recv_detail.groups()
                    else:
                        recv_wait_ms = 0.0
                        recv_read_ms = 0.0
                        recv_materialize_ms = 0.0
                        recv_transfer_ms = 0.0
                        recv_wait_other_ms = 0.0
                        recv_unaccounted_ms = 0.0
                    attention["recv_wait_ms"].append(float(recv_wait_ms))
                    attention["recv_read_ms"].append(float(recv_read_ms))
                    attention["recv_materialize_ms"].append(
                        float(recv_materialize_ms)
                    )
                    attention["recv_transfer_ms"].append(float(recv_transfer_ms))
                    attention["recv_wait_other_ms"].append(
                        float(recv_wait_other_ms or 0.0)
                    )
                    attention["recv_unaccounted_ms"].append(
                        float(recv_unaccounted_ms)
                    )
                    if detail := _ATTENTION_COMPUTE_DETAIL_RE.search(line):
                        (
                            append_kv_ms,
                            pack_ms,
                            sdpa_ms,
                            reshape_ms,
                            paged_metadata_ms,
                            paged_flash_ms,
                            shape_lookup_ms,
                            qkv_split_ms,
                            query_move_ms,
                            query_cat_ms,
                            append_lock_wait_ms,
                            append_prepare_ms,
                            append_record_ms,
                            append_tensor_ms,
                            append_copy_ms,
                            append_state_ms,
                            metadata_build_ms,
                            paged_flash_kernel_ms,
                            attention_output_reshape_ms,
                            compute_unaccounted_ms,
                        ) = detail.groups()
                    else:
                        append_kv_ms = pack_ms = sdpa_ms = reshape_ms = 0.0
                        paged_metadata_ms = 0.0
                        paged_flash_ms = 0.0
                        shape_lookup_ms = 0.0
                        qkv_split_ms = 0.0
                        query_move_ms = 0.0
                        query_cat_ms = 0.0
                        append_lock_wait_ms = 0.0
                        append_prepare_ms = 0.0
                        append_record_ms = 0.0
                        append_tensor_ms = 0.0
                        append_copy_ms = 0.0
                        append_state_ms = 0.0
                        metadata_build_ms = 0.0
                        paged_flash_kernel_ms = 0.0
                        attention_output_reshape_ms = 0.0
                        compute_unaccounted_ms = 0.0
                    attention["append_kv_ms"].append(float(append_kv_ms))
                    attention["pack_ms"].append(float(pack_ms))
                    attention["sdpa_ms"].append(float(sdpa_ms))
                    attention["reshape_ms"].append(float(reshape_ms))
                    attention["paged_metadata_ms"].append(
                        float(paged_metadata_ms or 0.0)
                    )
                    attention["paged_flash_ms"].append(float(paged_flash_ms or 0.0))
                    attention["shape_lookup_ms"].append(float(shape_lookup_ms or 0.0))
                    attention["qkv_split_ms"].append(float(qkv_split_ms or 0.0))
                    attention["query_move_ms"].append(float(query_move_ms or 0.0))
                    attention["query_cat_ms"].append(float(query_cat_ms or 0.0))
                    attention["append_lock_wait_ms"].append(
                        float(append_lock_wait_ms or 0.0)
                    )
                    attention["append_prepare_ms"].append(
                        float(append_prepare_ms or 0.0)
                    )
                    attention["append_record_ms"].append(float(append_record_ms or 0.0))
                    attention["append_tensor_ms"].append(float(append_tensor_ms or 0.0))
                    attention["append_copy_ms"].append(float(append_copy_ms or 0.0))
                    attention["append_state_ms"].append(float(append_state_ms or 0.0))
                    attention["metadata_build_ms"].append(
                        float(metadata_build_ms or 0.0)
                    )
                    attention["paged_flash_kernel_ms"].append(
                        float(paged_flash_kernel_ms or 0.0)
                    )
                    attention["attention_output_reshape_ms"].append(
                        float(attention_output_reshape_ms or 0.0)
                    )
                    attention["compute_unaccounted_ms"].append(
                        float(compute_unaccounted_ms or 0.0)
                    )
                    if correlation_new := _ATTENTION_CORRELATION_NEW_RE.search(line):
                        (
                            batch_key,
                            recv_done_ns,
                            compute_done_ns,
                            send_done_ns,
                            recv_start_ns,
                            pre_compute_start_ns,
                            pre_compute_done_ns,
                            paged_flash_done_ns,
                            reshape_done_ns,
                            send_start_ns,
                        ) = correlation_new.groups()
                        pcd_raw = int(pre_compute_done_ns)
                        attention_timestamps_by_key[batch_key] = {
                            "recv_start_ns": int(recv_start_ns),
                            "recv_done_ns": int(recv_done_ns),
                            "pre_compute_start_ns": int(pre_compute_start_ns),
                            "pre_compute_done_ns": (
                                pcd_raw if pcd_raw > 0 else int(compute_done_ns)
                            ),
                            "compute_done_ns": int(compute_done_ns),
                            "paged_flash_done_ns": int(paged_flash_done_ns),
                            "reshape_done_ns": int(reshape_done_ns),
                            "post_compute_done_ns": (
                                int(reshape_done_ns)
                                if int(reshape_done_ns) > 0
                                else int(compute_done_ns)
                            ),
                            "send_start_ns": int(send_start_ns),
                            "send_done_ns": int(send_done_ns),
                            "rows": calls,
                            "pre_compute_done_ns_raw": pcd_raw,
                        }
                    elif correlation_old := _ATTENTION_CORRELATION_OLD_RE.search(line):
                        (
                            batch_key,
                            recv_done_ns,
                            compute_done_ns,
                            send_done_ns,
                        ) = correlation_old.groups()
                        recv_done = int(recv_done_ns)
                        compute_done = int(compute_done_ns)
                        send_done = int(send_done_ns)
                        attention_timestamps_by_key[batch_key] = {
                            "recv_start_ns": recv_done,
                            "recv_done_ns": recv_done,
                            "pre_compute_start_ns": recv_done,
                            "pre_compute_done_ns": compute_done,
                            "compute_done_ns": compute_done,
                            "post_compute_done_ns": compute_done,
                            "send_start_ns": compute_done,
                            "send_done_ns": send_done,
                            "rows": calls,
                            "pre_compute_done_ns_raw": 0,
                        }
                continue
            if match := _MAILBOX_SEND_RE.search(line):
                (
                    actor,
                    kind,
                    nbytes,
                    queue_ms,
                    publish_ms,
                    pack_ms,
                    slot_wait_ms,
                    copy_ms,
                    payload_ms,
                    piggyback_ms,
                    notify_ms,
                    write_ms,
                    write_prepare_ms,
                    write_transfer_ms,
                    write_polls,
                    ack_wait_ms,
                    total_ms,
                ) = match.groups()
                total = float(total_ms)
                if max_total_ms is not None and total > max_total_ms:
                    continue
                actor_group = _mailbox_actor_group(actor)
                kind_group = _mailbox_kind_group(actor, kind)
                for field, value in (
                    ("nbytes", nbytes),
                    ("queue_ms", queue_ms),
                    ("publish_ms", publish_ms),
                    ("pack_ms", pack_ms),
                    ("slot_wait_ms", slot_wait_ms or 0.0),
                    ("copy_ms", copy_ms),
                    ("payload_ms", payload_ms or 0.0),
                    ("piggyback_ms", piggyback_ms or 0.0),
                    ("notify_ms", notify_ms),
                    ("write_ms", write_ms or 0.0),
                    ("write_prepare_ms", write_prepare_ms or 0.0),
                    ("write_transfer_ms", write_transfer_ms or 0.0),
                    ("write_polls", write_polls or 0.0),
                    ("ack_wait_ms", ack_wait_ms),
                    ("total_ms", total_ms),
                ):
                    _add_grouped_value(mailbox_send, actor_group, field, float(value))
                    _add_grouped_value(
                        mailbox_send_by_kind, kind_group, field, float(value)
                    )
                continue
            if match := _MAILBOX_INLINE_SEND_RE.search(line):
                (
                    actor,
                    kind,
                    nbytes,
                    publish_ms,
                    pack_ms,
                    slot_wait_ms,
                    copy_ms,
                    payload_ms,
                    piggyback_ms,
                    notify_ms,
                    write_ms,
                    write_prepare_ms,
                    write_transfer_ms,
                    write_polls,
                    total_ms,
                ) = match.groups()
                total = float(total_ms)
                if max_total_ms is not None and total > max_total_ms:
                    continue
                actor_group = _mailbox_actor_group(actor)
                kind_group = _mailbox_kind_group(actor, kind)
                for field, value in (
                    ("nbytes", nbytes),
                    ("queue_ms", 0.0),
                    ("publish_ms", publish_ms),
                    ("pack_ms", pack_ms),
                    ("slot_wait_ms", slot_wait_ms),
                    ("copy_ms", copy_ms),
                    ("payload_ms", payload_ms),
                    ("piggyback_ms", piggyback_ms),
                    ("notify_ms", notify_ms),
                    ("write_ms", write_ms),
                    ("write_prepare_ms", write_prepare_ms),
                    ("write_transfer_ms", write_transfer_ms),
                    ("write_polls", write_polls),
                    ("ack_wait_ms", 0.0),
                    ("total_ms", total_ms),
                ):
                    _add_grouped_value(mailbox_send, actor_group, field, float(value))
                    _add_grouped_value(
                        mailbox_send_by_kind, kind_group, field, float(value)
                    )
                continue
            if match := _MAILBOX_READ_RE.search(line):
                (
                    actor,
                    kind,
                    nbytes,
                    prepare_ms,
                    slot_wait_ms,
                    handle_prepare_ms,
                    transfer_ms,
                    transfer_polls,
                    materialize_ms,
                    total_ms,
                ) = match.groups()
                total = float(total_ms)
                if max_total_ms is not None and total > max_total_ms:
                    continue
                actor_group = _mailbox_actor_group(actor)
                kind_group = _mailbox_kind_group(actor, kind)
                for field, value in (
                    ("nbytes", nbytes),
                    ("prepare_ms", prepare_ms),
                    ("slot_wait_ms", slot_wait_ms or 0.0),
                    ("handle_prepare_ms", handle_prepare_ms or 0.0),
                    ("transfer_ms", transfer_ms),
                    ("transfer_polls", transfer_polls),
                    ("materialize_ms", materialize_ms),
                    ("total_ms", total_ms),
                ):
                    _add_grouped_value(mailbox_read, actor_group, field, float(value))
                    _add_grouped_value(
                        mailbox_read_by_kind, kind_group, field, float(value)
                    )
                continue
            if match := _MAILBOX_WAIT_RE.search(line):
                actor, kind, wait_ms = match.groups()
                _add_grouped_value(
                    mailbox_wait_by_kind,
                    _mailbox_kind_group(actor, kind),
                    "wait_ms",
                    float(wait_ms),
                )

    projection_attention_correlation: dict[str, list[float]] = {
        "matched_batches": [],
        "matched_batches_fine": [],
        "projection_send_done_to_attention_recv_start_ms": [],
        "attention_recv_ms": [],
        "attention_pre_compute_ms": [],
        "attention_compute_ms": [],
        "attention_post_compute_ms": [],
        "attention_send_ms": [],
        "attention_send_done_to_projection_recv_done_ms": [],
        "attention_path_after_projection_send_ms": [],
        "projection_resume_after_attention_ready_ms": [],
        "attention_ready_after_projection_resume_ms": [],
        "projection_resume_to_recv_done_ms": [],
        "pa_recv_start_skew_ms": [],
        "pa_compute_completion_skew_ms": [],
        "pa_completion_skew_ms": [],
        "pa_mean_idle_until_slowest_ms": [],
        "route_rows_range": [],
        "route_rows_max_over_mean": [],
        "route_kv_tokens_range": [],
        "route_kv_tokens_max_over_mean": [],
        "slowest_pa_rows": [],
        "slowest_pa_kv_tokens": [],
    }
    for (
        batch_keys,
        route_rows,
        route_kv_tokens,
        send_done_ns,
        yield_end_ns,
        recv_done_ns,
    ) in projection_correlation_entries:
        attention_times = [
            attention_timestamps_by_key[key]
            for key in batch_keys
            if key in attention_timestamps_by_key
        ]
        if len(attention_times) != len(batch_keys):
            continue
        recv_start_ns = min(item["recv_start_ns"] for item in attention_times)
        recv_done_ns_attention = max(item["recv_done_ns"] for item in attention_times)
        pre_compute_start_ns = min(
            item["pre_compute_start_ns"] for item in attention_times
        )
        pre_compute_done_ns = max(
            item["pre_compute_done_ns"] for item in attention_times
        )
        compute_done_ns = max(item["compute_done_ns"] for item in attention_times)
        post_compute_done_ns = max(
            item["post_compute_done_ns"] for item in attention_times
        )
        send_start_ns = min(item["send_start_ns"] for item in attention_times)
        max_attention_done_ns = max(item["send_done_ns"] for item in attention_times)
        if len(attention_times) > 1:
            completion_times = [
                item["send_done_ns"] for item in attention_times
            ]
            slowest_index = max(
                range(len(completion_times)),
                key=completion_times.__getitem__,
            )
            projection_attention_correlation["pa_recv_start_skew_ms"].append(
                (
                    max(item["recv_start_ns"] for item in attention_times)
                    - min(item["recv_start_ns"] for item in attention_times)
                )
                / 1_000_000.0
            )
            projection_attention_correlation[
                "pa_compute_completion_skew_ms"
            ].append(
                (
                    max(item["compute_done_ns"] for item in attention_times)
                    - min(item["compute_done_ns"] for item in attention_times)
                )
                / 1_000_000.0
            )
            projection_attention_correlation["pa_completion_skew_ms"].append(
                (max(completion_times) - min(completion_times)) / 1_000_000.0
            )
            projection_attention_correlation[
                "pa_mean_idle_until_slowest_ms"
            ].append(
                statistics.mean(
                    max_attention_done_ns - value
                    for value in completion_times
                )
                / 1_000_000.0
            )
            if len(route_rows) == len(attention_times):
                projection_attention_correlation["route_rows_range"].append(
                    float(max(route_rows) - min(route_rows))
                )
                projection_attention_correlation[
                    "route_rows_max_over_mean"
                ].append(max(route_rows) / statistics.mean(route_rows))
                projection_attention_correlation["slowest_pa_rows"].append(
                    float(route_rows[slowest_index])
                )
            if len(route_kv_tokens) == len(attention_times):
                projection_attention_correlation[
                    "route_kv_tokens_range"
                ].append(float(max(route_kv_tokens) - min(route_kv_tokens)))
                projection_attention_correlation[
                    "route_kv_tokens_max_over_mean"
                ].append(
                    max(route_kv_tokens) / statistics.mean(route_kv_tokens)
                )
                projection_attention_correlation[
                    "slowest_pa_kv_tokens"
                ].append(float(route_kv_tokens[slowest_index]))
        projection_attention_correlation["matched_batches"].append(
            float(len(attention_times))
        )
        has_fine = all(
            item.get("pre_compute_done_ns_raw", 0) > 0 for item in attention_times
        )
        if has_fine:
            projection_attention_correlation["matched_batches_fine"].append(
                float(len(attention_times))
            )
            projection_attention_correlation[
                "projection_send_done_to_attention_recv_start_ms"
            ].append((recv_start_ns - send_done_ns) / 1_000_000.0)
            projection_attention_correlation["attention_recv_ms"].append(
                (recv_done_ns_attention - recv_start_ns) / 1_000_000.0
            )
            projection_attention_correlation["attention_pre_compute_ms"].append(
                (pre_compute_done_ns - recv_done_ns_attention) / 1_000_000.0
            )
            projection_attention_correlation["attention_compute_ms"].append(
                (compute_done_ns - pre_compute_done_ns) / 1_000_000.0
            )
            projection_attention_correlation["attention_post_compute_ms"].append(
                (post_compute_done_ns - compute_done_ns) / 1_000_000.0
            )
            projection_attention_correlation["attention_send_ms"].append(
                (max_attention_done_ns - send_start_ns) / 1_000_000.0
            )
            projection_attention_correlation[
                "attention_send_done_to_projection_recv_done_ms"
            ].append((recv_done_ns - max_attention_done_ns) / 1_000_000.0)
        projection_attention_correlation[
            "attention_path_after_projection_send_ms"
        ].append((max_attention_done_ns - send_done_ns) / 1_000_000.0)
        projection_attention_correlation[
            "projection_resume_after_attention_ready_ms"
        ].append(max(0.0, (yield_end_ns - max_attention_done_ns) / 1_000_000.0))
        projection_attention_correlation[
            "attention_ready_after_projection_resume_ms"
        ].append(max(0.0, (max_attention_done_ns - yield_end_ns) / 1_000_000.0))
        projection_attention_correlation["projection_resume_to_recv_done_ms"].append(
            (recv_done_ns - yield_end_ns) / 1_000_000.0
        )

    return {
        "projection_trace": {
            field: _stat(values) for field, values in projection.items()
        },
        "projection_timeline": {
            field: _stat(values) for field, values in projection_timeline.items()
        },
        "projection_layer_timeline": {
            field: _stat(values) for field, values in projection_layer_timeline.items()
        },
        "projection_critical_path": {
            field: _stat(values) for field, values in projection_critical_path.items()
        },
        "projection_model_forward": {
            field: _stat(values) for field, values in projection_model_forward.items()
        },
        "projection_logits": {
            field: _stat(values) for field, values in projection_logits.items()
        },
        "projection_runner_forward": {
            field: _stat(values) for field, values in projection_runner_forward.items()
        },
        "projection_runner_forward_detail": {
            field: _stat(values)
            for field, values in projection_runner_forward_detail.items()
        },
        "projection_worker_exec": {
            field: _stat(values) for field, values in projection_worker_exec.items()
        },
        "projection_worker_sample": {
            field: _stat(values) for field, values in projection_worker_sample.items()
        },
        "projection_engine_step": {
            field: _stat(values) for field, values in projection_engine_step.items()
        },
        "projection_first_output": {
            field: _stat(values) for field, values in projection_first_output.items()
        },
        "attention_trace": {
            field: _stat(values) for field, values in attention.items()
        },
        "projection_attention_correlation": {
            field: _stat(values)
            for field, values in projection_attention_correlation.items()
        },
        "mailbox_send": {
            actor: {field: _stat(values) for field, values in fields.items()}
            for actor, fields in mailbox_send.items()
        },
        "mailbox_read": {
            actor: {field: _stat(values) for field, values in fields.items()}
            for actor, fields in mailbox_read.items()
        },
        "mailbox_send_by_kind": {
            group: {field: _stat(values) for field, values in fields.items()}
            for group, fields in mailbox_send_by_kind.items()
        },
        "mailbox_read_by_kind": {
            group: {field: _stat(values) for field, values in fields.items()}
            for group, fields in mailbox_read_by_kind.items()
        },
        "mailbox_wait_by_kind": {
            group: {field: _stat(values) for field, values in fields.items()}
            for group, fields in mailbox_wait_by_kind.items()
        },
    }


def summary_to_jsonable(summary: dict[str, object]) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, TraceStat):
            return value.to_dict()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    return convert(summary)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="include trace rows above the default 10ms warmup/outlier cutoff",
    )
    args = parser.parse_args(argv)
    summary = summarize_pap_trace_logs(
        args.log_dir,
        max_total_ms=None if args.include_outliers else 10.0,
    )
    print(json.dumps(summary_to_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
