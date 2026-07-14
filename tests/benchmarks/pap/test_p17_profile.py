from __future__ import annotations

import re
import tomllib
from pathlib import Path

from benchmarks.pap.profile_env import runner_environment


ROOT = Path(__file__).parents[3]
PROFILE_PATH = ROOT / "benchmarks" / "pap" / "profiles" / "p17_1pa1p.toml"


def _load_profile() -> dict[str, object]:
    with PROFILE_PATH.open("rb") as handle:
        return tomllib.load(handle)


def test_p17_profile_freezes_the_accepted_release_path() -> None:
    profile = _load_profile()

    assert profile["profile_id"] == "p17_1pa1p"
    assert profile["status"] == "canonical"
    assert profile["release_gate"] is True
    assert profile["repetitions"] == 3
    assert profile["model"]["dtype"] == "float16"
    assert profile["model"]["tensor_parallel_size"] == 1
    assert profile["workload"]["document_tokens"] == 16000
    assert profile["workload"]["rounds"] == 5
    assert profile["workload"]["active_conversations"] == 4
    assert profile["workload"]["output_tokens_per_round"] == 256
    assert profile["topology"]["name"] == "1pa1p"
    assert profile["transport"]["offload_exec"] == "local_fast"
    assert profile["mps"]["mode"] == "static"
    assert profile["mps"]["prefill_visible_sms"] == 64
    assert profile["mps"]["attention_visible_sms"] == 28


def test_p17_profile_has_no_absolute_artifact_paths() -> None:
    profile = _load_profile()

    for reference in (profile["model"], profile["workload"]["corpus"]):
        assert reference["root_id"]
        assert not Path(reference["relative_path"]).is_absolute()
        assert ".." not in Path(reference["relative_path"]).parts


def test_p17_profile_records_fingerprints_and_unverified_capabilities() -> None:
    profile = _load_profile()

    assert all(
        re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in profile["baseline_fingerprints"].values()
    )
    assert profile["compatibility"] == {
        "xpayp": "preserved-unverified",
        "cross_host_nixl": "preserved-unverified",
    }


def test_p17_profile_is_the_runner_configuration_source() -> None:
    environment = runner_environment(_load_profile())

    assert environment == {
        "P17_PROFILE_ID": "p17_1pa1p",
        "P17_REPETITIONS": "3",
        "P17_MODEL_RELATIVE_PATH": "Qwen3-8B",
        "P17_CORPUS_RELATIVE_PATH": "sonnet_4x.txt",
        "PAP_VLLM_DTYPE": "float16",
        "PAP_TP_SIZE": "1",
        "MAX_MODEL_LEN": "20000",
        "MAX_NUM_BATCHED_TOKENS": "4096",
        "MAX_NUM_SEQS": "4",
        "PAP_MULTITURN_BLOCK_SIZE": "16",
        "INPUT_LEN": "16000",
        "OUTPUT_LEN": "256",
        "PAP_MULTITURN_APPEND_TOKENS": "120",
        "PAP_MULTITURN_LOAD_ROUNDS": "5",
        "PAP_MULTITURN_LOAD_CONVERSATIONS": "4",
        "PAP_MULTITURN_LOAD_REQUEST_RATE": "2.0",
        "PAP_TOPOLOGY": "1pa1p",
        "PAP_PREFILL_GPUS": "1",
        "PAP_PROJECTION_GPUS": "2",
        "PAP_OFFLOAD_EXEC_TRANSPORT": "local_fast",
        "PAP_OFFLOAD_KV_TRANSPORT": "cuda_ipc",
        "PAP_ROUTING_POLICY": "round_robin",
        "PAP_PREFILL_MPS_PERCENT": "70",
        "PAP_ATTENTION_MPS_PERCENT": "30",
        "PAP_STATIC_PREFILL_CHUNKS": "16",
        "PAP_STATIC_ATTENTION_CHUNKS": "7",
        "PAP_STATIC_PREFILL_EXPECTED_SMS": "64",
        "PAP_STATIC_ATTENTION_EXPECTED_SMS": "28",
        "PAP_DIRECT_MAILBOX_OUTPUT": "1",
        "PAP_LOCAL_FAST_STREAM_ORDERED": "1",
        "PAP_LOCAL_FAST_SLOT_COUNT": "2",
        "PAP_DECODE_SLOT_PLAN_CACHE_LIMIT": "256",
        "PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND": "1",
        "PAP_LOCAL_FAST_BATCH_PLAN": "1",
        "PAP_ENABLE_PROMPT_TOKENS_DETAILS": "1",
        "PAP_PREFIX_CACHE_AUDIT": "0",
        "PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS": "256",
        "PAP_BENCH_STRICT_CORRECTNESS_AUDIT": "1",
        "PAP_BENCH_SESSION_DRAIN_TIMEOUT": "60",
    }
