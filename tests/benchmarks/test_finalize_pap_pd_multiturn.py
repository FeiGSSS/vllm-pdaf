from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.multi_turn.finalize_pap_pd_multiturn import finalize_result

_DEFERRED_TRACE_SPANS = (
    "qkv_ready_wait_gpu_ms",
    "kv_append_gpu_ms",
    "paged_fa_gpu_ms",
    "output_p2p_copy_gpu_ms",
)


def _client_result(architecture: str = "pap") -> dict[str, object]:
    result: dict[str, object] = {
        "architecture": architecture,
        "git_commit": "a" * 40,
        "git_tracked_worktree_dirty": False,
        "implementation": {
            "offload_exec_transport": "local_fast",
            "direct_mailbox_output": True,
            "unified_md_fast_key": True,
        },
        "validity": {"status": "passed", "cache_gate": "passed"},
        "cache_validation": {"status": "passed"},
    }
    if architecture == "pd":
        result["profile"] = {"block_size": 16}
        result["rounds"] = [
            {"round": 1, "prompt_tokens": 16018},
            {"round": 2, "prompt_tokens": 16418},
        ]
        result["cache_validation"] = {
            "status": "official_streaming_one_way_metrics_passed",
            "first_prompt_block_boundary": 16016,
            "expected_cached_tokens": 16272,
            "decode_derived_hit_tokens": 256,
        }
        result["pd_reuse_validation"] = {
            "status": "official_streaming_one_way_metrics_passed",
            "mode": "official_streaming_one_way",
            "proxy_cache_misses": 2,
            "proxy_cache_hits": 0,
            "total_prompt_tokens": 32436,
            "prefill_prompt_tokens_by_source": {
                "local_compute": 16420,
                "local_cache_hit": 16016,
                "external_kv_transfer": 0,
            },
            "decode_prompt_tokens_by_source": {
                "local_compute": 0,
                "local_cache_hit": 16272,
                "external_kv_transfer": 16164,
            },
        }
    return result


def _pap_artifacts(tmp_path: Path) -> dict[str, Path]:
    artifacts = {
        "session_drain": tmp_path / "session_drain.env",
        "routing": tmp_path / "routing_audit.json",
        "correctness_logs": tmp_path / "correctness_audit.env",
        "attention_stats": tmp_path / "attention_stats.json",
        "run_metadata": tmp_path / "run_metadata.json",
        "effective_config": tmp_path / "effective_config.env",
        "tracked_worktree_patch": tmp_path / "tracked_worktree.patch",
        "tracked_index_patch": tmp_path / "tracked_index.patch",
    }
    artifacts["session_drain"].write_text(
        "STATUS=passed\nACTIVE_SESSIONS=0\n",
        encoding="utf-8",
    )
    artifacts["routing"].write_text(
        json.dumps({"status": "passed", "errors": []}),
        encoding="utf-8",
    )
    artifacts["correctness_logs"].write_text(
        "STATUS=passed\nMATCH_COUNT=0\nSTRICT=1\n",
        encoding="utf-8",
    )
    artifacts["attention_stats"].write_text(
        json.dumps(
            {
                "offload_exec_compute_calls": 72,
                "unified_md_hits": 70,
                "unified_md_fast_key_lookups": 72,
                "unified_md_fast_key_hits": 70,
                "unified_md_full_key_scans": 2,
            }
        ),
        encoding="utf-8",
    )
    artifacts["run_metadata"].write_text(
        json.dumps(
            {
                "git_commit": "a" * 40,
                "git_tracked_worktree_dirty": False,
                "offload_exec_transport": "local_fast",
                "direct_mailbox_output": True,
                "unified_md_fast_key": True,
            }
        ),
        encoding="utf-8",
    )
    artifacts["tracked_worktree_patch"].write_bytes(b"")
    artifacts["tracked_index_patch"].write_bytes(b"")
    artifacts["effective_config"].write_text(
        "PAP_UNIFIED_MD_FAST_KEY=1\n",
        encoding="utf-8",
    )
    return artifacts


def _enable_deferred_trace(artifacts: dict[str, Path]) -> None:
    attention = json.loads(artifacts["attention_stats"].read_text())
    instances = attention.get("instances")
    payloads = [attention] if instances is None else instances
    for payload in payloads:
        stats = payload.get("stats", payload)
        compute_calls = stats["offload_exec_compute_calls"]
        stats["offload_exec_peer_batches"] = compute_calls
        stats["fast_path_hits"] = compute_calls - 3
        stats["fallbacks"] = 1
        stats["deferred_cuda_trace"] = {
            "enabled": True,
            "scope": "attention_process_critical_chain",
            "collector_count": 1,
            "pending_records": 0,
            "dropped_records": 0,
            "error_records": 0,
            "spans": {
                name: {
                    "count": (
                        compute_calls - 2
                        if name == "kv_append_gpu_ms"
                        else compute_calls
                    )
                }
                for name in _DEFERRED_TRACE_SPANS
            },
        }
    artifacts["attention_stats"].write_text(json.dumps(attention))
    with artifacts["effective_config"].open("a", encoding="utf-8") as file_obj:
        file_obj.write(
            "PAP_DEFERRED_CUDA_TRACE=1\n"
            "PAP_OFFLOAD_EXEC_TRACE=0\n"
        )


def test_finalize_result_records_required_gates_and_artifact_hash(
    tmp_path: Path,
) -> None:
    artifacts = _pap_artifacts(tmp_path)

    result = finalize_result(
        _client_result(),
        architecture="pap",
        passed_gates=(
            "session_drain",
            "routing",
            "correctness_logs",
            "attention_stats_capture",
        ),
        artifacts=artifacts,
    )

    assert result["external_validation"]["status"] == "passed"
    assert result["external_validation"]["gates"]["routing"] == "passed"
    assert result["external_validation"]["artifacts"]["session_drain"] == {
        "path": "session_drain.env",
        "sha256": hashlib.sha256(
            artifacts["session_drain"].read_bytes()
        ).hexdigest(),
    }


def test_finalize_result_fails_closed_on_missing_gate() -> None:
    with pytest.raises(ValueError, match="missing required"):
        finalize_result(
            _client_result(),
            architecture="pap",
            passed_gates=("session_drain",),
            artifacts={},
        )


def test_finalize_result_rejects_metadata_fast_key_mismatch(
    tmp_path: Path,
) -> None:
    artifacts = _pap_artifacts(tmp_path)
    metadata = json.loads(artifacts["run_metadata"].read_text())
    metadata["unified_md_fast_key"] = False
    artifacts["run_metadata"].write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="run metadata mismatch"):
        finalize_result(
            _client_result(),
            architecture="pap",
            passed_gates=(
                "session_drain",
                "routing",
                "correctness_logs",
                "attention_stats_capture",
            ),
            artifacts=artifacts,
        )


def test_finalize_result_rejects_fast_key_without_runtime_evidence(
    tmp_path: Path,
) -> None:
    artifacts = _pap_artifacts(tmp_path)
    stats = json.loads(artifacts["attention_stats"].read_text())
    stats["unified_md_hits"] = 0
    stats["unified_md_fast_key_lookups"] = 0
    stats["unified_md_fast_key_hits"] = 0
    artifacts["attention_stats"].write_text(json.dumps(stats))

    with pytest.raises(ValueError, match="metadata fast key"):
        finalize_result(
            _client_result(),
            architecture="pap",
            passed_gates=(
                "session_drain",
                "routing",
                "correctness_logs",
                "attention_stats_capture",
            ),
            artifacts=artifacts,
        )


def test_finalize_result_rejects_metadata_fast_key_config_mismatch(
    tmp_path: Path,
) -> None:
    artifacts = _pap_artifacts(tmp_path)
    artifacts["effective_config"].write_text(
        "PAP_UNIFIED_MD_FAST_KEY=0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="effective config"):
        finalize_result(
            _client_result(),
            architecture="pap",
            passed_gates=(
                "session_drain",
                "routing",
                "correctness_logs",
                "attention_stats_capture",
            ),
            artifacts=artifacts,
        )


def test_finalize_result_accepts_disabled_metadata_fast_key(
    tmp_path: Path,
) -> None:
    artifacts = _pap_artifacts(tmp_path)
    result = _client_result()
    result["implementation"]["unified_md_fast_key"] = False
    metadata = json.loads(artifacts["run_metadata"].read_text())
    metadata["unified_md_fast_key"] = False
    artifacts["run_metadata"].write_text(json.dumps(metadata))
    stats = json.loads(artifacts["attention_stats"].read_text())
    stats["unified_md_fast_key_lookups"] = 0
    stats["unified_md_fast_key_hits"] = 0
    artifacts["attention_stats"].write_text(json.dumps(stats))
    artifacts["effective_config"].write_text(
        "PAP_UNIFIED_MD_FAST_KEY=0\n",
        encoding="utf-8",
    )

    finalized = finalize_result(
        result,
        architecture="pap",
        passed_gates=(
            "session_drain",
            "routing",
            "correctness_logs",
            "attention_stats_capture",
        ),
        artifacts=artifacts,
    )

    assert finalized["external_validation"]["status"] == "passed"


def test_finalize_result_aggregates_multi_instance_attention_stats(
    tmp_path: Path,
) -> None:
    artifacts = _pap_artifacts(tmp_path)
    stats = json.loads(artifacts["attention_stats"].read_text())
    artifacts["attention_stats"].write_text(
        json.dumps(
            {
                "instances": [
                    {"attention_index": 0, "stats": stats},
                    {"attention_index": 1, "stats": stats},
                ]
            }
        )
    )
    _enable_deferred_trace(artifacts)

    finalized = finalize_result(
        _client_result(),
        architecture="pap",
        passed_gates=(
            "session_drain",
            "routing",
            "correctness_logs",
            "attention_stats_capture",
        ),
        artifacts=artifacts,
    )

    assert finalized["external_validation"]["status"] == "passed"


def test_finalize_result_accepts_complete_deferred_cuda_trace(
    tmp_path: Path,
) -> None:
    artifacts = _pap_artifacts(tmp_path)
    _enable_deferred_trace(artifacts)

    finalized = finalize_result(
        _client_result(),
        architecture="pap",
        passed_gates=(
            "session_drain",
            "routing",
            "correctness_logs",
            "attention_stats_capture",
        ),
        artifacts=artifacts,
    )

    assert finalized["external_validation"]["status"] == "passed"


@pytest.mark.parametrize(
    "field",
    ("pending_records", "dropped_records", "error_records"),
)
def test_finalize_result_rejects_incomplete_deferred_cuda_trace(
    tmp_path: Path,
    field: str,
) -> None:
    artifacts = _pap_artifacts(tmp_path)
    _enable_deferred_trace(artifacts)
    stats = json.loads(artifacts["attention_stats"].read_text())
    stats["deferred_cuda_trace"][field] = 1
    artifacts["attention_stats"].write_text(json.dumps(stats))

    with pytest.raises(ValueError, match=field):
        finalize_result(
            _client_result(),
            architecture="pap",
            passed_gates=(
                "session_drain",
                "routing",
                "correctness_logs",
                "attention_stats_capture",
            ),
            artifacts=artifacts,
        )


def test_finalize_result_rejects_missing_deferred_cuda_span(
    tmp_path: Path,
) -> None:
    artifacts = _pap_artifacts(tmp_path)
    _enable_deferred_trace(artifacts)
    stats = json.loads(artifacts["attention_stats"].read_text())
    del stats["deferred_cuda_trace"]["spans"]["paged_fa_gpu_ms"]
    artifacts["attention_stats"].write_text(json.dumps(stats))

    with pytest.raises(ValueError, match="missing paged_fa_gpu_ms"):
        finalize_result(
            _client_result(),
            architecture="pap",
            passed_gates=(
                "session_drain",
                "routing",
                "correctness_logs",
                "attention_stats_capture",
            ),
            artifacts=artifacts,
        )


def test_finalize_result_rejects_deferred_cuda_span_count_mismatch(
    tmp_path: Path,
) -> None:
    artifacts = _pap_artifacts(tmp_path)
    _enable_deferred_trace(artifacts)
    stats = json.loads(artifacts["attention_stats"].read_text())
    spans = stats["deferred_cuda_trace"]["spans"]
    spans["output_p2p_copy_gpu_ms"]["count"] -= 1
    artifacts["attention_stats"].write_text(json.dumps(stats))

    with pytest.raises(ValueError, match="count mismatch"):
        finalize_result(
            _client_result(),
            architecture="pap",
            passed_gates=(
                "session_drain",
                "routing",
                "correctness_logs",
                "attention_stats_capture",
            ),
            artifacts=artifacts,
        )


def test_finalize_result_requires_matching_architecture() -> None:
    with pytest.raises(ValueError, match="architecture"):
        finalize_result(
            _client_result("pd"),
            architecture="pap",
            passed_gates=(
                "session_drain",
                "routing",
                "correctness_logs",
                "attention_stats_capture",
            ),
            artifacts={},
        )


def test_finalize_result_rejects_failed_correctness_artifact(
    tmp_path: Path,
) -> None:
    artifacts = _pap_artifacts(tmp_path)
    artifacts["correctness_logs"].write_text(
        "STATUS=failed\nMATCH_COUNT=1\nSTRICT=0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="correctness"):
        finalize_result(
            _client_result(),
            architecture="pap",
            passed_gates=(
                "session_drain",
                "routing",
                "correctness_logs",
                "attention_stats_capture",
            ),
            artifacts=artifacts,
        )


def test_finalize_pd_requires_reuse_evidence_and_clean_config(
    tmp_path: Path,
) -> None:
    artifacts = {
        "proxy_log": tmp_path / "proxy.log",
        "prefill_metrics": tmp_path / "prefill.prom",
        "decode_metrics": tmp_path / "decode.prom",
        "effective_config": tmp_path / "effective_config.env",
        "correctness_logs": tmp_path / "correctness.env",
        "tracked_worktree_patch": tmp_path / "tracked_worktree.patch",
        "tracked_index_patch": tmp_path / "tracked_index.patch",
    }
    artifacts["proxy_log"].write_text(
        "cache MISS\ncache MISS\n",
        encoding="utf-8",
    )
    metric_template = (
        'vllm:prompt_tokens_by_source_total{{model_name="qwen",engine="0",'
        'source="{source}"}} {value}.0'
    )
    prefill_sources = {
        "local_compute": 16420,
        "local_cache_hit": 16016,
        "external_kv_transfer": 0,
    }
    decode_sources = {
        "local_compute": 0,
        "local_cache_hit": 16272,
        "external_kv_transfer": 16164,
    }
    artifacts["prefill_metrics"].write_text(
        "\n".join(
            metric_template.format(source=source, value=value)
            for source, value in prefill_sources.items()
        ),
        encoding="utf-8",
    )
    artifacts["decode_metrics"].write_text(
        "\n".join(
            metric_template.format(source=source, value=value)
            for source, value in decode_sources.items()
        ),
        encoding="utf-8",
    )
    artifacts["effective_config"].write_text(
        f"GIT_COMMIT={'a' * 40}\n",
        encoding="utf-8",
    )
    artifacts["correctness_logs"].write_text(
        "STATUS=passed\nMATCH_COUNT=0\n",
        encoding="utf-8",
    )
    artifacts["tracked_worktree_patch"].write_bytes(b"")
    artifacts["tracked_index_patch"].write_bytes(b"")

    result = finalize_result(
        _client_result("pd"),
        architecture="pd",
        passed_gates=("pd_reuse_metrics", "correctness_logs"),
        artifacts=artifacts,
    )

    assert result["external_validation"]["status"] == "passed"
