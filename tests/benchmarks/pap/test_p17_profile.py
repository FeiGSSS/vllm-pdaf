from __future__ import annotations

import re
import tomllib
from pathlib import Path


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
