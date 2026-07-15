from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

import pytest

from benchmarks.pap.import_legacy_run import (
    _async_decode_token_setting,
    _validate_output_path,
    import_legacy_run,
)


ROOT = Path(__file__).parents[3]
SCHEMA_PATH = (
    ROOT / "benchmarks" / "pap" / "schemas" / "run_manifest.schema.json"
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_legacy_run(tmp_path: Path) -> Path:
    source = tmp_path / "legacy" / "formal"
    rep = source / "rep1"
    logs = rep / "service_logs"
    logs.mkdir(parents=True)
    digest = "a" * 64
    _write_json(
        source / "aggregate.json",
        {
            "profile": {
                "profile_id": "legacy-profile",
                "profile_version": 1,
                "model": "/unknown/models/Qwen3-8B",
                "corpus_path": "/unknown/corpus.txt",
                "corpus_sha256": digest,
                "dtype": "float16",
                "tensor_parallel_size": 1,
                "document_tokens": 16,
                "output_tokens_per_round": 2,
                "rounds": 1,
                "active_conversations": 1,
            },
            "profile_fingerprint": digest,
            "architecture": "pap",
            "implementation": {"offload_exec_transport": "local_fast"},
            "implementation_fingerprint": "b" * 64,
            "hardware_signature": "test-gpu",
            "mode": "historical",
            "repetition_count": 1,
            "repetition_ids": ["rep-one"],
            "validity": {"status": "passed"},
            "client_validation": {"status": "passed"},
            "cache_validation": {"status": "passed"},
            "external_validation": {
                "gates": {
                    "attention_stats_capture": "passed",
                    "correctness_logs": "passed",
                    "decode_token_join": "passed",
                    "routing": "passed",
                    "session_drain": "passed",
                }
            },
            "request_shape": {"fingerprint": "c" * 64},
            "aggregation_method": "single legacy repetition",
            "metrics": {
                "round_1": {
                    "ttft_ms": {"median": 1.0},
                    "tpot_ms": {"median": 2.0},
                },
                "steady_rounds_2_5": {
                    "ttft_ms": {"median": 3.0},
                    "tpot_ms": {"median": 4.0},
                },
            },
            "warnings": [],
        },
    )
    _write_json(
        rep / "result.json",
        {
            "topology": {
                "name": "1pa1p",
                "pa_count": 1,
                "projection_count": 1,
                "pd_prefill_count": 0,
                "pd_decode_count": 0,
            },
            "overall": {"completed_requests": 1, "failed_requests": 0},
        },
    )
    _write_json(
        rep / "run_metadata.json",
        {
            "git_commit": "d" * 40,
            "git_tracked_worktree_dirty": False,
            "started_at": "2026-07-14T00:00:00+08:00",
            "routing_policy": "round_robin",
            "offload_exec_transport": "local_fast",
            "offload_kv_transport": "cuda_ipc",
        },
    )
    _write_json(
        rep / "topology_manifest.json",
        {
            "pa_groups": [{"gpu": "1"}],
            "projections": [{"gpu": "2"}],
        },
    )
    _write_json(rep / "attention_fast_path_stats.json", {"calls": 1})
    _write_json(rep / "routing_audit.json", {"status": "passed"})
    (rep / "effective_config.env").write_text(
        "PAP_BENCH_MPS_PROFILE=baseline_static_64_28\n"
        "PAP_ASYNC_DECODE_TOKEN=1\n"
        "PAP_UNIFIED_KV=1\n"
        "PAP_BATCHED_ROUTE_COPY=1\n"
        "PAP_ATTENTION_DISPATCH_MODE=legacy\n",
        encoding="utf-8",
    )
    (rep / "mps_static_audit_pa_0.env").write_text(
        "PAP_MPS_MODE=static\n"
        "PREFILL_VISIBLE_SMS=64\n"
        "ATTENTION_VISIBLE_SMS=28\n",
        encoding="utf-8",
    )
    for name in (
        "correctness_audit.env",
        "decode_token_join_audit.env",
        "session_drain.env",
    ):
        (rep / name).write_text("STATUS=passed\n", encoding="utf-8")
    (rep / "tracked_worktree.patch").write_bytes(b"")
    (rep / "tracked_index.patch").write_bytes(b"")
    (logs / "projection_0.log").write_text(
        "server version 0.0.test\n",
        encoding="utf-8",
    )
    return source


def _snapshot(source: Path) -> dict[str, bytes]:
    return {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }


def test_legacy_import_is_read_only_and_uses_root_references(
    tmp_path: Path,
) -> None:
    source = _make_legacy_run(tmp_path)
    before = _snapshot(source)

    manifest = import_legacy_run(
        source,
        experiment_id="PAP-20260714-LEGACY-TEST",
        run_id="legacy_test_run",
        profile_id="p17_1pa1p",
        evidence="historical",
        artifact_root_id="raw-root",
        roots={"raw-root": tmp_path},
    )

    assert _snapshot(source) == before
    assert manifest["workload"]["model"]["path"] == "missing"
    assert manifest["workload"]["corpus"]["path"] == "missing"
    assert all(
        artifact["path"]["root_id"] == "raw-root"
        and not Path(artifact["path"]["relative_path"]).is_absolute()
        for artifact in manifest["artifacts"]
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = list(
        Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        ).iter_errors(manifest)
    )
    assert not errors


def test_legacy_import_rejects_output_inside_raw_tree(tmp_path: Path) -> None:
    source = _make_legacy_run(tmp_path)

    with pytest.raises(ValueError, match="outside the raw source tree"):
        _validate_output_path(source, source / "tracked_manifest.json")


def test_legacy_import_reads_old_and_unconditional_decode_token_evidence() -> None:
    assert _async_decode_token_setting(
        {},
        {"DECODE_TOKEN_DELIVERY": "async"},
    ) is True
    assert _async_decode_token_setting(
        {"PAP_ASYNC_DECODE_TOKEN": "0"},
        {},
    ) is False
    assert _async_decode_token_setting({}, {}) == "missing"


def test_p17_import_reads_converged_metadata_and_mps_audit(tmp_path: Path) -> None:
    source = _make_legacy_run(tmp_path)
    rep = source / "rep1"
    metadata_path = rep / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "batched_route_copy": True,
            "attention_dispatch_mode": "legacy",
        }
    )
    _write_json(metadata_path, metadata)
    (rep / "effective_config.env").write_text("", encoding="utf-8")
    (rep / "mps_static_audit_pa_0.env").write_text(
        "MPS_MODE=static\n"
        "PREFILL_VISIBLE_SMS=64\n"
        "ATTENTION_VISIBLE_SMS=28\n",
        encoding="utf-8",
    )

    manifest = import_legacy_run(
        source,
        experiment_id="PAP-20260715-P17-IMPORT-TEST",
        run_id="p17_import_test_run",
        profile_id="p17_1pa1p",
        evidence="formal-clean",
        artifact_root_id="raw-root",
        roots={"raw-root": tmp_path},
    )

    assert manifest["mps"] == {
        "mode": "static",
        "profile_id": "baseline_static_64_28",
        "prefill_visible_sms": 64,
        "attention_visible_sms": 28,
    }
    assert manifest["runtime"]["settings"]["unified_kv"] is True
    assert manifest["runtime"]["settings"]["batched_route_copy"] is True
    assert (
        manifest["runtime"]["settings"]["attention_dispatch_mode"]
        == "legacy"
    )
